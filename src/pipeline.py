# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "agenttune==1.1.6",
#   "huggingface_hub>=1.0",
# ]
# ///
"""VALXR-ToolCall training pipeline (runs on Hugging Face Jobs, single A10G/L4/A100).

Stages (each pushes its artifacts to the Hub before the next starts):
  base  : greedy generations of the untouched Tiny Aya on the test split
  sft   : LoRA supervised fine-tune on train (TRL SFTTrainer, completion-only loss)
  sfteval
  heal  : AgentTune self-heal loop — run the SFT policy on train prompts, turn every
          failure (and hard safety negatives) into TrainingExample rows via the AgentTune
          spine, then offline DPO with create_agentic_trainer("dpo", ...)
  dpoeval

    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 4h src/pipeline.py \
        --hub_user <user> --stages base,sft,sfteval,heal,dpoeval
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.modules.setdefault("vllm", None)  # TRL imports vllm eagerly; we do not use it

import torch
from huggingface_hub import HfApi, snapshot_download

WORK = Path("/tmp/valxr")
WORK.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ------------------------------------------------------------------ io
def fetch_code_and_data(dataset_repo: str) -> Path:
    d = Path(snapshot_download(dataset_repo, repo_type="dataset", local_dir=WORK / "ds"))
    sys.path.insert(0, str(d / "src"))
    return d


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def push_file(api, repo, local, remote, repo_type="dataset"):
    for i in range(3):
        try:
            api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo, repo_type=repo_type)
            return
        except Exception as e:  # noqa: BLE001
            log("upload retry", i, e)
            time.sleep(5)


# ------------------------------------------------------------------ model helpers
def chat_inputs(tok, messages):
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:  # template without system role -> fold into user turn
        sys_msg = messages[0]["content"]
        merged = [{"role": "user", "content": sys_msg + "\n\nAnalyst: " + messages[1]["content"]}]
        return tok.apply_chat_template(merged, add_generation_prompt=True, tokenize=False)


def to_train_messages(tok, messages):
    """Return messages in a form the chat template accepts (system folded if needed)."""
    try:
        tok.apply_chat_template(messages, tokenize=False)
        return messages
    except Exception:
        return [{"role": "user", "content": messages[0]["content"] + "\n\nAnalyst: " + messages[1]["content"]}]


@torch.no_grad()
def generate_all(model, tok, rows, bs=16, max_new_tokens=192):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    outs = []
    # sort by length for efficient batching
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["user"]))
    res = [None] * len(rows)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        prompts = [chat_inputs(tok, rows[i]["messages"]) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        t0 = time.time()
        o = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.pad_token_id)
        dt = (time.time() - t0) / len(idx)
        for j, i in enumerate(idx):
            text = tok.decode(o[j, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            res[i] = {"id": rows[i]["id"], "output": text, "sec_per_sample": round(dt, 4)}
        log(f"gen {min(s + bs, len(order))}/{len(order)}")
    return res


def write_card(folder, title, body, base="CohereLabs/tiny-aya-global"):
    """Replace PEFT's auto card (it records local paths as base_model, which the Hub rejects)."""
    card = "\n".join([
        "---", "license: cc-by-nc-4.0", f"base_model: {base}", "library_name: transformers",
        "datasets: [mayankkkkkk/valxr-toolcall]", "language: [en, hi, es]",
        "tags: [tool-calling, agents, ai-safety, agenttune, auditkit, lexsi-hackathon, valxr]", "---",
        f"# {title}", "", body, ""])
    Path(folder, "README.md").write_text(card, encoding="utf-8")


def load_model(path_or_id, adapter=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_id)
    model = AutoModelForCausalLM.from_pretrained(path_or_id, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    return model, tok


def score_rows(rows, gens):
    from valxr_tools import score_one
    by = {g["id"]: g for g in gens}
    agg: dict[str, list] = {}
    for r in rows:
        s = score_one(r["gold"], by[r["id"]]["output"])
        for k, v in s.items():
            if v is not None:
                agg.setdefault(k, []).append(v)
    return {k: round(sum(v) / len(v), 4) for k, v in agg.items()}


def eval_stage(name, model, tok, test, api, a):
    gens = generate_all(model, tok, test, bs=a.gen_bs)
    p = WORK / "results" / f"gen_{name}.jsonl"
    write_jsonl(p, gens)
    m = score_rows(test, gens)
    log(name, json.dumps(m))
    (WORK / "results" / f"metrics_{name}.json").write_text(json.dumps(m, indent=2))
    push_file(api, a.dataset_repo, p, f"results/gen_{name}.jsonl")
    push_file(api, a.dataset_repo, WORK / "results" / f"metrics_{name}.json", f"results/metrics_{name}.json")
    return m


# ------------------------------------------------------------------ stages
def stage_sft(train, a, api):
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(a.base)
    rows = []
    for r in train:
        msgs = to_train_messages(tok, r["messages"])
        rows.append({"prompt": msgs, "completion": [{"role": "assistant", "content": r["target"]}]})
    ds = Dataset.from_list(rows).shuffle(seed=0)
    cfg = SFTConfig(
        output_dir=str(WORK / "sft"), num_train_epochs=a.sft_epochs, per_device_train_batch_size=a.bs,
        gradient_accumulation_steps=max(1, 16 // a.bs), learning_rate=2e-4, lr_scheduler_type="cosine",
        warmup_steps=10, logging_steps=10, save_strategy="no", bf16=True, report_to=[],
        gradient_checkpointing=True, max_length=a.max_len, completion_only_loss=True,
    )
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    tr = SFTTrainer(model=a.base, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft)
    tr.train()
    out = WORK / "sft_adapter"
    tr.save_model(str(out))
    hist = [h for h in tr.state.log_history if "loss" in h]
    (out / "train_log.json").write_text(json.dumps(tr.state.log_history, indent=1))
    log("SFT done; first/last loss", hist[0]["loss"] if hist else None, hist[-1]["loss"] if hist else None)
    del tr
    torch.cuda.empty_cache()
    write_card(out, "VALXR-ToolCall SFT LoRA (Tiny Aya Global)", "Stage-1 LoRA SFT adapter. See the dataset repo for the full write-up.")
    repo = f"{a.hub_user}/{a.prefix}-sft-lora"
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message="SFT LoRA adapter")
    return out


def merged_sft_path(a):
    p = WORK / "sft_merged"
    if p.exists():
        return p
    adapter = WORK / "sft_adapter"
    if not adapter.exists():
        adapter = Path(snapshot_download(f"{a.hub_user}/{a.prefix}-sft-lora", local_dir=WORK / "sft_adapter"))
    model, tok = load_model(a.base, adapter=str(adapter))
    model.save_pretrained(str(p))
    tok.save_pretrained(str(p))
    del model
    torch.cuda.empty_cache()
    return p


def _corrupt_args(args, rng):
    """Copy-fidelity negative: the kinds of id-mangling a small model makes on unseen ids."""
    bad = json.loads(json.dumps(args))
    keys = [k for k, v in bad.items() if isinstance(v, str) and k != "reasoning"]
    mode = rng.choice(["colon", "colon", "prefix", "extra"]) if keys else "extra"
    if mode == "extra":
        extra = {"window": "today", "target": "", "max_hops": 3, "relationship_types": ["can_access"]}
        free = [k for k in extra if k not in bad]
        if not free:
            return None
        k = rng.choice(free); bad[k] = extra[k]
        return bad
    k = rng.choice(keys); v = bad[k]
    if mode == "prefix" and "-" in v:
        bad[k] = v.split("-", 1)[1]
    elif ":" in v:
        bad[k] = v.replace(":", rng.choice(["", "_", ".", "--"]), 1)
    else:
        return None
    return bad


def stage_heal(train, a, api):
    """AgentTune self-heal: on-policy failures on a disjoint dev probe set + copy-fidelity and
    hard safety negatives -> DPO."""
    from datasets import Dataset
    from agenttune.agentic import build_dataset
    from agenttune.core.backend_factory import create_agentic_trainer
    from agenttune.decide.closed_loop.contracts import TrainingExample
    from valxr_tools import DESTRUCTIVE, format_target, parse, score_one

    sft_path = merged_sft_path(a)
    model, tok = load_model(str(sft_path))
    rng = random.Random(1)
    pool = read_jsonl(WORK / "ds" / "data" / "dev.jsonl") if a.probe_split == "dev" else train
    probe = rng.sample(pool, min(a.heal_probe, len(pool)))
    gens = generate_all(model, tok, probe, bs=a.gen_bs)
    del model
    torch.cuda.empty_cache()

    examples, n_fail = [], 0
    for r, g in zip(probe, gens):
        s = score_one(r["gold"], g["output"])
        msgs = to_train_messages(tok, r["messages"])
        chosen = [{"role": "assistant", "content": r["target"]}]
        if s["full_match"] != 1.0:  # on-policy failure -> corrective pair
            n_fail += 1
            examples.append(TrainingExample(
                trajectory_id=r["id"], original_failure_type="policy_failure:" + r["category"],
                root_cause=f"expected {r['gold']['kind']} got {parse(g['output'])['kind']}",
                prompt=msgs, chosen=chosen, rejected=[{"role": "assistant", "content": g["output"]}]))
    # copy-fidelity negatives on the probe pool: right tool, mangled ids (dropped colon, stripped prefix, extra args)
    n_copy = 0
    if a.copy_negs:
        for r in probe:
            if r["gold"]["kind"] != "call" or rng.random() > a.copy_negs:
                continue
            bad = _corrupt_args(r["gold"]["arguments"], rng)
            if not bad:
                continue
            n_copy += 1
            examples.append(TrainingExample(
                trajectory_id=r["id"] + "-copy", original_failure_type="arg_corruption:" + r["category"],
                root_cause="argument not copied verbatim from the request",
                prompt=to_train_messages(tok, r["messages"]),
                chosen=[{"role": "assistant", "content": r["target"]}],
                rejected=[{"role": "assistant", "content": format_target("call", {"name": r["gold"]["name"], "arguments": bad})}]))
        # clarify rows: the realistic unsafe move is acting on whatever id the analyst did mention
        import re
        for r in probe:
            if r["category"] != "clarify":
                continue
            ids = re.findall(r"\b(?:user|sa|role|agent|cred|pod|node|vm|s3|db|secret|kms|arn):[\w./:@-]+", r["user"])
            if not ids:
                continue
            bad = rng.choice([{"name": "approve_containment", "arguments": {"action_id": ids[0]}},
                              {"name": "propose_containment", "arguments": {"subject": ids[0], "isolate": ids[-1:],
                                                                            "reasoning": "assumed from request"}}])
            examples.append(TrainingExample(
                trajectory_id=r["id"] + "-clar", original_failure_type="unsafe_action:clarify",
                root_cause="acted on an entity that is not a containment action id",
                prompt=to_train_messages(tok, r["messages"]),
                chosen=[{"role": "assistant", "content": r["target"]}],
                rejected=[{"role": "assistant", "content": format_target("call", bad)}]))
    log(f"heal: {n_copy} copy-fidelity negatives")
    # hard safety negatives: what an over-eager agent would do
    aid = lambda: f"act-{rng.randint(1000, 9999)}"  # noqa: E731
    for r in train:
        if r["category"] not in ("clarify", "refuse_policy", "injection") or rng.random() > a.neg_frac:
            continue
        if r["category"] == "injection":
            bad = {"name": "approve_containment", "arguments": {"action_id": aid()}}
        elif r["category"] == "clarify":
            bad = {"name": rng.choice(sorted(DESTRUCTIVE)), "arguments": {"action_id": aid()} if rng.random() < .5 else
                   {"subject": "user:unknown", "isolate": ["pod:unknown"], "reasoning": "assumed target"}}
        else:
            bad = {"name": "get_audit_log", "arguments": {}}  # complying-ish instead of refusing
        examples.append(TrainingExample(
            trajectory_id=r["id"] + "-neg", original_failure_type="unsafe_action:" + r["category"],
            root_cause="took an action the safety policy forbids",
            prompt=to_train_messages(tok, r["messages"]),
            chosen=[{"role": "assistant", "content": r["target"]}],
            rejected=[{"role": "assistant", "content": format_target("call", bad)}]))
    rows = build_dataset(examples)
    log(f"heal: {n_fail} on-policy failures / {len(probe)} probed; {len(rows)} preference rows total")
    write_jsonl(WORK / "results" / "dpo_pairs.jsonl", rows)
    push_file(api, a.dataset_repo, WORK / "results" / "dpo_pairs.jsonl", "data/dpo_pairs.jsonl")

    trainer = create_agentic_trainer(
        "dpo", backend="trl", model=str(sft_path), train_dataset=Dataset.from_list(rows),
        beta=0.1, learning_rate=a.dpo_lr, num_train_epochs=a.dpo_epochs, per_device_train_batch_size=max(1, a.bs // 2),
        gradient_accumulation_steps=8, max_length=a.max_len, bf16=True, gradient_checkpointing=True,
        logging_steps=5, save_strategy="no", report_to=[], output_dir=str(WORK / "dpo_adapter"),
        peft_config={"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "task_type": "CAUSAL_LM",
                     "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    )
    stats = trainer.train()
    log("DPO stats", json.dumps(stats, default=str)[:2000])
    out = WORK / "dpo_adapter"
    try:
        (out / "train_log.json").write_text(json.dumps(trainer.trainer.state.log_history, indent=1))
    except Exception as e:  # noqa: BLE001
        log("no log history", e)
    write_card(out, "VALXR-ToolCall AgentTune self-heal DPO LoRA",
               "Stage-2 DPO LoRA trained with AgentTune `create_agentic_trainer('dpo')` on top of the merged SFT model "
               f"({a.hub_user}/{a.prefix}-sft-lora merged into CohereLabs/tiny-aya-global).")
    repo = f"{a.hub_user}/{a.prefix}-dpo-lora"
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message="AgentTune self-heal DPO LoRA (on top of SFT merge)")


def final_merge_and_push(a, api):
    sft_path = merged_sft_path(a)
    adapter = WORK / "dpo_adapter"
    if not (adapter / "adapter_config.json").exists():
        adapter = Path(snapshot_download(f"{a.hub_user}/{a.prefix}-dpo-lora", local_dir=WORK / "dpo_adapter"))
    model, tok = load_model(str(sft_path), adapter=str(adapter))
    out = WORK / "final"
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    write_card(out, "VALXR-ToolCall: Tiny Aya Global for safe multilingual tool calling",
               "Merged model = Tiny Aya Global + SFT LoRA + AgentTune self-heal DPO LoRA. See the dataset repo for the write-up and metrics.")
    repo = f"{a.hub_user}/{a.prefix}"
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo, commit_message="Merged Tiny Aya Global + SFT + AgentTune DPO")
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub_user", required=True)
    ap.add_argument("--prefix", default="valxr-toolcall-tiny-aya")
    ap.add_argument("--dataset_repo", default=None)
    ap.add_argument("--base", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--stages", default="base,sft,sfteval,heal,dpoeval")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--gen_bs", type=int, default=24)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--sft_epochs", type=float, default=2)
    ap.add_argument("--dpo_epochs", type=float, default=2)
    ap.add_argument("--heal_probe", type=int, default=800)
    ap.add_argument("--neg_frac", type=float, default=0.5)
    ap.add_argument("--probe_split", default="dev", choices=["dev", "train"])
    ap.add_argument("--copy_negs", type=float, default=0.6, help="fraction of probe call rows that get an arg-corruption negative (0=off)")
    ap.add_argument("--dpo_lr", type=float, default=1e-5)
    ap.add_argument("--limit_test", type=int, default=0)
    a = ap.parse_args()
    a.dataset_repo = a.dataset_repo or f"{a.hub_user}/valxr-toolcall"
    stages = a.stages.split(",")

    log("GPU:", torch.cuda.get_device_name(0), "stages:", stages)
    subprocess.run(["nvidia-smi"], check=False)
    api = HfApi()
    d = fetch_code_and_data(a.dataset_repo)
    train = read_jsonl(d / "data" / "train.jsonl")
    test = read_jsonl(d / "data" / "test.jsonl")
    if a.limit_test:
        test = test[: a.limit_test]
    log(f"train={len(train)} test={len(test)}")

    if "base" in stages:
        m, t = load_model(a.base)
        eval_stage("base", m, t, test, api, a)
        del m
        torch.cuda.empty_cache()
    if "sft" in stages:
        stage_sft(train, a, api)
    if "sfteval" in stages:
        m, t = load_model(str(merged_sft_path(a)))
        eval_stage("sft", m, t, test, api, a)
        del m
        torch.cuda.empty_cache()
    if "heal" in stages:
        stage_heal(train, a, api)
    if "dpoeval" in stages:
        m, t = final_merge_and_push(a, api)
        eval_stage("sft_dpo", m, t, test, api, a)
    log("ALL DONE")


if __name__ == "__main__":
    main()
