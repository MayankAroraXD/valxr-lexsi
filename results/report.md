| metric | base | sft | sft_dpo |
|---|---|---|---|
| full_match ↑ | 0.401 | 0.887 | 0.887 |
| decision_acc ↑ | 0.575 | 0.986 | 0.986 |
| tool_acc ↑ | 0.531 | 1.000 | 1.000 |
| args_exact ↑ | 0.336 | 0.877 | 0.877 |
| format_valid ↑ | 0.992 | 1.000 | 1.000 |
| strict_format ↑ | 0.983 | 1.000 | 1.000 |
| unsafe_action ↓ | 0.014 | 0.029 | 0.029 |
| over_refusal ↓ | 0.442 | 0.000 | 0.000 |
| injection_resisted ↑ | 1.000 | 1.000 | 1.000 |

| language (full_match) | base | sft | sft_dpo |
|---|---|---|---|
| en | 0.479 | 0.852 | 0.858 |
| es | 0.472 | 0.943 | 0.943 |
| hi | 0.364 | 0.946 | 0.946 |
| hinglish | 0.224 | 0.882 | 0.871 |

| category (full_match) | base | sft | sft_dpo |
|---|---|---|---|
| clarify | 0.375 | 0.844 | 0.844 |
| destructive_ok | 0.200 | 0.892 | 0.908 |
| injection | 0.548 | 0.935 | 0.919 |
| read | 0.309 | 0.849 | 0.849 |
| refuse_policy | 0.921 | 1.000 | 1.000 |

| template split (full_match) | base | sft | sft_dpo |
|---|---|---|---|
| seen | 0.373 | 0.916 | 0.916 |
| heldout | 0.423 | 0.862 | 0.862 |

| sec/sample (A100 80GB, bs=24, greedy) | 0.252 | 0.408 | 0.398 |