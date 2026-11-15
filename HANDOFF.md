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
- **[Nov 15 ~22:45 IST] DPO v2 DONE** (`mayankkkkkk/6af9e22323985fb9cd83893f`, COMPLETED). Probe: 94 failures / 714 dev prompts, 628 preference rows, loss 0.587. Test: exact match stays **40.1% → 88.7% (SFT) → 88.7% (DPO)**. AuditKIT SFT vs DPO: Δ = 0, p = 1.0. DPO changed 7/362 outputs; the only metric flips are `test-00180` fixed (`8462` → `act-8462`) and `test-00341` regressed (out-of-schema `relationship_types`). Unsafe stays 2.9% (test-00293, test-00298). Over-refusal 44.2% → 0%. Injection resistance 100% (metric does not catch pasted text copied into optional read-tool args).
- Write-up is honest about the tie. Merged model repo was **not** reverted to SFT-only: the weights tie on headline metrics, and the card says so. v1 generations stay in `results/v1_dpo/`.
- Collections: models https://huggingface.co/collections/mayankkkkkk/valxr-toolcall-models-6af9ecab71680167f7080c2e · dataset https://huggingface.co/collections/mayankkkkkk/valxr-toolcall-dataset-6af9ecabc0f1dc548baed08f
- **[Nov 15 ~22:30 IST] FULL RUN DONE.** base 40.1% → SFT 88.7% → SFT+DPO(v1) 88.7% full_match. DPO v1 changed only 1/362 outputs:
  the SFT model had 0 failures on the train probe, so DPO only saw already-learned safety negatives. v1 results archived
  in the dataset repo under `results/v1_dpo/`.
- Main SFT error = dropped colons in unseen ids (`s3:ml-training-data` → `s3ml-training-data`), `act-1234` → `1234`, plus 2 unsafe
  `approve_containment` calls on vague clarify prompts.
- GitHub (public, already origin): https://github.com/MayankAroraXD/valxr-lexsi. Commit and push the updates there.
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
1. [x] Smoke test passed.
2. [x] Full run done. DPO v2 done. Headline: 40.1% → 88.7% → 88.7%.
3. [x] Generations downloaded. `eval_auditkit.py` re-run on v2. SFT vs DPO p = 1.0.
4. [x] README / WRITEUP filled, including the null DPO result and the two unsafe calls.
5. [x] Dataset card and model cards uploaded. Licence cc-by-nc-4.0 on the models, cc-by-4.0 on the data.
6. [x] Write-up is the GitHub README (HF articles returned 403). Dataset repo also has `WRITEUP.md`.
7. [x] Collections created (URLs in Job status and README §9).
8. [ ] Fill the submission form and **confirm with the user before clicking submit**.
       Form: https://hackathon.lexsi.ai/submission/
       - Email: arora7827@gmail.com
       - Track 3. Problem statements #1 and #9.
       - Repos used: AgentTune, AuditKIT.
       - GitHub: https://github.com/MayankAroraXD/valxr-lexsi
       - Write-up: that README (after the push below).
       - Model collection: https://huggingface.co/collections/mayankkkkkk/valxr-toolcall-models-6af9ecab71680167f7080c2e
       - Dataset collection: https://huggingface.co/collections/mayankkkkkk/valxr-toolcall-dataset-6af9ecabc0f1dc548baed08f
       - Abstract: the README abstract. Do not claim DPO improved accuracy or that unsafe actions fell.

## Known gotchas
- Tiny Aya's chat template injects a ~400-token Cohere preamble. Prompts run to ~1,630 tokens in total, hence `max_len=2048`.
- The vocabulary has 262k entries, so the logits are memory-heavy. bs=4 is fine on an A100 80GB; on 24GB cards use bs=1.
- AgentTune pins torch==2.11, transformers>=5.15, trl==1.7.1, so it requires the transformers v5 API (`dtype=`, not `torch_dtype=`).
- Windows + bash: Python can't see `/tmp`; use the scratchpad directory or the project path. Set `PYTHONIOENCODING=utf-8` when printing Hindi.
