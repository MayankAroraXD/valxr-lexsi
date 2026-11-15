"""Score every stage's generations with AuditKIT and compare stages with significance tests.

    python src/eval_auditkit.py --results results --data data
Writes results/report.json and results/report.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import auditkit as ak

from valxr_tools import score_one

METRICS = ["full_match", "decision_acc", "tool_acc", "args_exact", "format_valid", "strict_format",
           "unsafe_action", "over_refusal", "injection_resisted"]
LOWER_BETTER = {"unsafe_action", "over_refusal"}


def make_scorer(metric):
    direction = ak.Direction.MINIMIZE if metric in LOWER_BETTER else ak.Direction.MAXIMIZE

    @ak.scorer(direction=direction, name=metric)
    def _s(sample, output):
        v = score_one(sample.metadata["gold"], output)[metric]
        return float(v)
    return _s


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def subset_for(metric, rows):
    if metric in ("tool_acc", "args_exact", "over_refusal"):
        return [r for r in rows if r["gold"]["kind"] == "call"]
    if metric == "unsafe_action":
        return [r for r in rows if r["gold"]["kind"] != "call"]
    if metric == "injection_resisted":
        return [r for r in rows if r["category"] == "injection"]
    return rows


def run_stage(name, rows, gens):
    by = {g["id"]: g["output"] for g in gens}
    runs, headline = {}, {}
    for m in METRICS:
        sub = subset_for(m, rows)
        samples = [ak.Sample(input=r["user"], id=r["id"], actual_output=by[r["id"]], tags=[r["lang"], r["category"]],
                             metadata={"gold": r["gold"]}) for r in sub]
        res = ak.evaluate(samples, model="precomputed", scorers=[make_scorer(m)], cache=False,
                          experiment_name=f"valxr-{name}-{m}")
        runs[m] = res
        headline[m] = round(float(res.headline[m]), 4)
    return runs, headline


def breakdown(rows, gens, key):
    by = {g["id"]: g["output"] for g in gens}
    agg = defaultdict(list)
    for r in rows:
        agg[r[key]].append(score_one(r["gold"], by[r["id"]])["full_match"])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(agg.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--data", default="data")
    ap.add_argument("--stages", default="base,sft,sft_dpo")
    a = ap.parse_args()
    rows = read_jsonl(Path(a.data) / "test.jsonl")
    R = Path(a.results)
    report = {"n_test": len(rows), "stages": {}, "comparisons": {}}
    stage_runs = {}
    for st in a.stages.split(","):
        gp = R / f"gen_{st}.jsonl"
        if not gp.exists():
            print("skip", st)
            continue
        gens = read_jsonl(gp)
        runs, head = run_stage(st, rows, gens)
        stage_runs[st] = runs
        lat = [g.get("sec_per_sample", 0) for g in gens]
        report["stages"][st] = {
            "headline": head,
            "by_lang": breakdown(rows, gens, "lang"),
            "by_category": breakdown(rows, gens, "category"),
            "heldout_vs_seen_template": {
                "heldout": breakdown([r for r in rows if r["heldout_template"]], gens, "split").get("test"),
                "seen": breakdown([r for r in rows if not r["heldout_template"]], gens, "split").get("test")},
            "sec_per_sample": round(sum(lat) / max(1, len(lat)), 4),
        }
        print(st, json.dumps(head))
    names = list(stage_runs)
    pairs = [(names[0], c) for c in names[1:]] + [(names[i - 1], names[i]) for i in range(2, len(names))]
    for b, c in pairs:
        comp = {}
        for m in METRICS:
            try:
                rc = ak.compare(stage_runs[b][m], stage_runs[c][m])
                comp[m] = {"deltas": [vars(x) if hasattr(x, "__dict__") else str(x) for x in rc.metric_deltas()],
                           "significance": rc.significance(m), "samples": rc.sample_summary(m)}
            except Exception as e:  # noqa: BLE001
                comp[m] = {"error": str(e)[:300]}
        report["comparisons"][f"{b}_vs_{c}"] = comp
    (R / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # markdown table
    lines = ["| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for m in METRICS:
        arrow = "↓" if m in LOWER_BETTER else "↑"
        lines.append(f"| {m} {arrow} | " + " | ".join(f"{report['stages'][s]['headline'][m]:.3f}" for s in names) + " |")
    lines += ["", "| language (full_match) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for lang in report["stages"][names[0]]["by_lang"]:
        lines.append(f"| {lang} | " + " | ".join(f"{report['stages'][s]['by_lang'][lang]:.3f}" for s in names) + " |")
    lines += ["", "| category (full_match) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for cat in report["stages"][names[0]]["by_category"]:
        lines.append(f"| {cat} | " + " | ".join(f"{report['stages'][s]['by_category'][cat]:.3f}" for s in names) + " |")
    lines += ["", "| template split (full_match) | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in ("seen", "heldout"):
        lines.append(f"| {k} | " + " | ".join(f"{report['stages'][s]['heldout_vs_seen_template'][k]:.3f}" for s in names) + " |")
    lines += ["", "| sec/sample (A100 80GB, bs=24, greedy) | " + " | ".join(f"{report['stages'][s]['sec_per_sample']:.3f}" for s in names) + " |"]
    (R / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
