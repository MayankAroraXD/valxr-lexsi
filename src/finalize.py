"""After the HF Job finishes: pull results, score with AuditKIT, fill README placeholders, push cards.

    cd lexsi-hackathon/src && PYTHONIOENCODING=utf-8 python finalize.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REPO = "mayankkkkkk/valxr-toolcall"
api = HfApi()

# 1. pull generations + metrics
(ROOT / "results").mkdir(exist_ok=True)
stages = []
for st in ("base", "sft", "sft_dpo"):
    try:
        for f in (f"results/gen_{st}.jsonl", f"results/metrics_{st}.json"):
            p = hf_hub_download(REPO, f, repo_type="dataset", force_download=True)
            (ROOT / f).write_bytes(Path(p).read_bytes())
        stages.append(st)
    except Exception as e:  # noqa: BLE001
        print("missing", st, e)
print("stages:", stages)

# 2. AuditKIT scoring + significance
subprocess.run([sys.executable, "eval_auditkit.py", "--results", str(ROOT / "results"), "--data", str(ROOT / "data"),
                "--stages", ",".join(stages)], check=True, cwd=ROOT / "src")
rep = json.loads((ROOT / "results" / "report.json").read_text(encoding="utf-8"))
H = {s: rep["stages"][s]["headline"] for s in stages}
last = stages[-1]


def pct(x):
    return f"{100 * x:.1f}%"


fill = {
    "{{BASE_FULL}}": pct(H["base"]["full_match"]),
    "{{SFT_FULL}}": pct(H.get("sft", H[last])["full_match"]),
    "{{FINAL_FULL}}": pct(H[last]["full_match"]),
    "{{BASE_UNSAFE}}": pct(H["base"]["unsafe_action"]),
    "{{FINAL_UNSAFE}}": pct(H[last]["unsafe_action"]),
    "{{BASE_INJ}}": pct(H["base"]["injection_resisted"]),
    "{{FINAL_INJ}}": pct(H[last]["injection_resisted"]),
    "{{FINAL_HELDOUT}}": pct(rep["stages"][last]["heldout_vs_seen_template"]["heldout"]),
}

# significance lines
sig_lines = ["", "**AuditKIT bootstrap significance (paired, 95% CI):**", "",
             "| comparison | metric | Δ | 95% CI | p | significant |", "|---|---|---|---|---|---|"]
for comp, d in rep["comparisons"].items():
    for m in ("full_match", "decision_acc", "args_exact", "unsafe_action", "injection_resisted"):
        s = d.get(m, {}).get("significance") or {}
        if "delta" in s:
            ci = s.get("ci") or [None, None]
            sig_lines.append(f"| {comp.replace('_vs_', ' → ')} | {m} | {s['delta']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                             f"{s.get('p_value', float('nan')):.4f} | {'yes' if s.get('significant') else 'no'} |")
tables = (ROOT / "results" / "report.md").read_text(encoding="utf-8") + "\n" + "\n".join(sig_lines)
fill["{{RESULTS_TABLES}}"] = ("Test set: 362 rows; all entities unseen, 196 held-out templates. Greedy decoding, identical "
                              "system prompt for every model. Columns: `base` = Tiny Aya Global, `sft` = + LoRA SFT, "
                              "`sft_dpo` = + AgentTune self-heal DPO.\n\n" + tables)

readme = (ROOT / "README.md").read_text(encoding="utf-8")
for k, v in fill.items():
    readme = readme.replace(k, v)
out = ROOT / "WRITEUP.md"
out.write_text(readme, encoding="utf-8")
left = [k for k in fill if k in readme]
print("written", out, "unfilled:", left)
for k, v in fill.items():
    if k != "{{RESULTS_TABLES}}":
        print(k, v)

# 3. push write-up + dataset card + report to the dataset repo
api.upload_file(path_or_fileobj=str(ROOT / "data" / "README.md"), path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj=str(out), path_in_repo="WRITEUP.md", repo_id=REPO, repo_type="dataset")
for f in ("report.md", "report.json"):
    api.upload_file(path_or_fileobj=str(ROOT / "results" / f), path_in_repo=f"results/{f}", repo_id=REPO, repo_type="dataset")
print("pushed")
