# HANDOFF: Lexsi hackathon submission (read this first if you are a new Claude session)

**Deadline: Nov 15 2026, 11:59 PM AoE = Nov 16, 5:29 PM IST.** Submission form: https://hackathon.lexsi.ai/submission/
(the form renders without an accessibility tree, so fill it from screenshots in Chrome). You can submit more than once; the latest submission counts.
Registered email: the user registered with arora7827@gmail.com (they said so in chat). Get the user's OK before submitting the form.

## What we're building
- **Track 3, Agentic Fine-tuning**, problem statement #1 (Reliable tool calling), plus #9 (reward hacking / safety negatives).
- **VALXR-ToolCall:** fine-tune `CohereLabs/tiny-aya-global` (3.35B, gated; the user already has access) to call the 10 real
  VALXR API tools (`src/valxr_tools.py`), and to clarify, refuse or resist prompt injection, in en, hinglish, hi and es.
- Lexsi repos used: **AgentTune** (`TrainingExample` → `build_dataset` → `create_agentic_trainer("dpo")`, self-heal loop)
  and **AuditKIT** (`ak.scorer`, `ak.evaluate(model="precomputed")`, `ak.compare(...).significance()`).

## Files (all in `lexsi-hackathon/`)
| file | purpose |
|---|---|
| `src/valxr_tools.py` | tool schemas, system prompt + policy, output parser, `score_one()` metrics |
| `src/gen_data.py` | deterministic data generator (seed 7) → `data/train.jsonl` (2000), `data/test.jsonl` (362) |
| `src/pipeline.py` | HF Jobs script (uv inline deps): stages `base,sft,sfteval,heal,dpoeval`; pushes everything to the Hub after each stage |
| `src/eval_auditkit.py` | local AuditKIT scoring + significance → `results/report.md`, `results/report.json` (tested on fake data, works) |
| `README.md` | write-up / HF article draft; `{{PLACEHOLDERS}}` still need the real numbers |
| `.env` | `hugging_face_token=...` (gitignored, NEVER print or commit it) |

## Hugging Face
- User: `mayankkkkkk`. Local CLI is logged in (if not, run `python -m huggingface_hub.cli.hf auth login --token <value from .env>` without echoing the token).
- `hf` is not on PATH; use `python -m huggingface_hub.cli.hf ...`
- Credits: $20 (expire Dec 2). a100-large costs $2.50/h. **The user approved roughly $6 of GPU spend (smoke test + one full run).** Ask before spending more.
- Dataset repo (public): https://huggingface.co/datasets/mayankkkkkk/valxr-toolcall. It contains `data/` and `src/`, and the job downloads code and data from it.
  **If you change `src/*.py`, re-upload them to this repo before launching a job.**
- Model repos the full run creates: `mayankkkkkk/valxr-toolcall-tiny-aya-sft-lora`, `...-dpo-lora`, and the merged model `mayankkkkkk/valxr-toolcall-tiny-aya`.
  The smoke run uses the prefix `valxr-smoke-*`; delete those repos at the end (ask the user first).

## Job status
- **FULL RUN launched: `mayankkkkkk/6af9c64a23985fb9cd8370e3`** (a100-large, 4h timeout, started ~14:52 UTC Nov 15). Local log copy: `results/full.log`.
  Stream it with `PYTHONIOENCODING=utf-8 python -m huggingface_hub.cli.hf jobs logs -f <id>`. Without the utf-8 setting, Windows crashes on the progress bars.
- Smoke test (done): base 0.125 → SFT (0.05 epoch) 0.458 full_match on 24 rows. All stages worked except the upload of the DPO adapter's model card, which is now fixed with `write_card()`.
- Smoke job: `mayankkkkkk/6af9c47e29dd05ff0368d0b9` (a100-large, 50 min timeout, tiny settings). Check it with
  `python -m huggingface_hub.cli.hf jobs inspect <id>` and `... jobs logs <id>`.
- Full-run command, to use once the smoke test passes:
```bash
cd lexsi-hackathon
export HF_TOKEN="$(sed -nE 's/^hugging_face_token=//p' .env | tr -d '\r\n" ')"
python -m huggingface_hub.cli.hf jobs uv run --flavor a100-large --secrets HF_TOKEN --timeout 4h --name valxr-full -d \
  src/pipeline.py --hub_user mayankkkkkk --stages base,sft,sfteval,heal,dpoeval
```

## Remaining TODO (in order)
1. [ ] Smoke test passes (fix any errors in `pipeline.py`, re-upload `src/`, re-run).
2. [ ] Launch the full run (~1.5–2h).
3. [ ] Download `results/gen_{base,sft,sft_dpo}.jsonl` and `results/metrics_*.json` from the dataset repo into `lexsi-hackathon/results/`, then run
       `cd src && PYTHONIOENCODING=utf-8 python eval_auditkit.py --results ../results --data ../data`.
4. [ ] Fill the README `{{...}}` placeholders and paste `results/report.md` into §6. Add the AuditKIT significance results (p-values, CIs) from `report.json`.
5. [ ] Write the full dataset card (`data/README.md`) and model cards (licence cc-by-nc-4.0, base_model tag, eval table). Upload them.
6. [ ] Publish the write-up as an HF article (huggingface.co/blog/community, via the user's browser session) or link the GitHub README.
       The GitHub repo is `MayankAroraXD/valxr-ai` (branch `feature/agent-squad`). Commit `lexsi-hackathon/` and ask before pushing.
7. [ ] Create public HF **collections**: one for models, one for the dataset (`HfApi().create_collection(...)` + `add_collection_item`).
8. [ ] Fill the submission form: registered email, abstract (<500 words; the README abstract works), model collection URL,
       dataset collection URL, write-up URL, GitHub URL, repos used (AgentTune, AuditKIT), Track 3, problem statements #1 and #9.
       **Confirm with the user before clicking submit.**

## Known gotchas
- Tiny Aya's chat template injects a ~400-token Cohere preamble. Prompts run to ~1,630 tokens in total, hence `max_len=2048`.
- The vocabulary has 262k entries, so the logits are memory-heavy. bs=4 is fine on an A100 80GB; on 24GB cards use bs=1.
- AgentTune pins torch==2.11, transformers>=5.15, trl==1.7.1, so it requires the transformers v5 API (`dtype=`, not `torch_dtype=`).
- Windows + bash: Python can't see `/tmp`; use the scratchpad directory or the project path. Set `PYTHONIOENCODING=utf-8` when printing Hindi.
