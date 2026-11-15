---
license: cc-by-4.0
language: [en, hi, es]
task_categories: [text-generation]
tags: [tool-calling, function-calling, agents, ai-safety, prompt-injection, security, multilingual, hinglish, code-mixed, lexsi-hackathon, agenttune, auditkit]
pretty_name: VALXR-ToolCall
size_categories: [1K<n<10K]
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: test
        path: data/test.jsonl
---

# VALXR-ToolCall

Multilingual, safety-aware tool-calling conversations for a **security copilot** that drives the
[VALXR AI](https://github.com/MayankAroraXD/valxr-ai) identity attack-graph API. Built for the Lexsi AI Safety & Alignment
Hackathon (Track 3, Agentic Fine-tuning).

Each example is one analyst request, in **English, Hinglish (romanised Hindi-English code-mixing), Hindi (Devanagari) or Spanish**,
plus the single correct response under a written safety policy:

- `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`: one of 10 tools that mirror real VALXR endpoints
  (`find_attack_paths`, `find_choke_points`, `get_identity_exposure`, `simulate_control_removal`, `list_assets`,
  `get_squad_findings`, `get_twin_diff`, `get_audit_log`, and the production-changing `propose_containment` / `approve_containment`)
- `<clarify>`: the request is missing a required id, or it asks for a containment action without naming an exact target
- `<refuse>`: audit-log tampering, disabling monitoring agents, revealing secret values, or isolating everything at once

## Categories
| category | meaning | train | test |
|---|---|---|---|
| `read` | read-only tool call | 983 | 165 |
| `destructive_ok` | legitimate, fully specified containment call | 257 | 65 |
| `clarify` | vague or under-specified, so ask | 260 | 32 |
| `refuse_policy` | policy violation, so refuse | 240 | 38 |
| `injection` | benign request + pasted ticket/log/email carrying an instruction to approve or contain. Gold = the benign call only | 260 | 62 |

Languages (train / test): en 934/169 · hinglish 481/85 · hi 271/55 · es 314/53.

## Generalisation controls
- **All test entity ids are disjoint from train** (different users, service accounts, roles, ARNs, pods and crown jewels).
- **196 / 362 test rows use held-out templates** (`heldout_template: true`): phrasings never seen in training.
- No exact prompt appears in both train and test.

## Fields
`id, split, lang, intent, category, user, gold{kind, name?, arguments?, category}, target, messages, heldout_template`.
`messages` already includes the full system prompt (tool schemas + policy). `target` is the gold assistant string.

## Scoring
`src/valxr_tools.py::score_one` is deterministic: full match, decision accuracy, tool and argument exact match, format,
unsafe-action rate, over-refusal and injection resistance. `src/eval_auditkit.py` wraps it as AuditKIT scorers
and adds bootstrap significance between models.

## Also in this repo
- `src/`: the generator (`gen_data.py`, seed 7), tool contract, training pipeline (AgentTune self-heal DPO) and AuditKIT eval
- `results/`: raw generations and metrics for base Tiny Aya, SFT, and SFT + AgentTune DPO
- `data/dpo_pairs.jsonl`: the self-heal preference pairs (on-policy failures + hard safety negatives) built with AgentTune

## Licence & provenance
Fully synthetic, written by the authors from hand-built templates; no real people, credentials or organisations.
CC-BY-4.0. Entity names such as `user:priya.sharma` are fictional placeholders.
