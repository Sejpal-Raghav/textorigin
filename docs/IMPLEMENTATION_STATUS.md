# TextOrigin implementation status

Verification snapshot: 9 October 2026. Application features and training infrastructure are implemented; the full release classifier is still being prepared/trained on the user-authorized local CPU run. The project is not yet accepted as fully complete.

The original SPEC.md, BACKLOG.md and IMPLEMENTATION-HANDOFF.md references were not found in this checkout. docs/PROJECT_COMPLETION_PROMPT.md was restored from the conversation after its original write was interrupted and is the working completion scope.

## Implemented and verified

| Requirement | Status and evidence |
|---|---|
| Dependency environment | Complete suite collects; compatible Arrow 21 pin; verified Windows/Python 3.12 lock; pip check passes |
| Deterministic CLI rewrite | Six style/strength combinations, text/file/stdin, JSON and overwrite protection |
| Protected content and structure | Whole multiline/unterminated/tilde fences, inline code, single/double/smart quotes, indented code, numbers/URLs/numeric citations and indentation/newlines; no placeholder collisions |
| Full semantic coverage | Bounded corresponding MiniLM token windows; minimum acceptance score; tail-change regression and actual-model compatibility check |
| Final-output metrics | final_metrics corresponds to accepted text, including fallback; UI uses accepted metrics |
| Full-document classification | Disjoint token windows, bounded configurable batches, token-weighted aggregation, unique coverage metadata; controlled-model regression plus actual 4,575-token/19-window check |
| Shared compare | Original and accepted output through shared services; same model version; signed score deltas; CLI/API/job/web contracts |
| Jobs and cancellation | Bounded transient queue/count/TTL/memory admission, opaque IDs, cancellation races, worker cleanup, failure redaction, capacity release, feature-window progress/cancellation |
| Web workflows | Classify/rewrite/compare, progress/cancel including before submission returns, source/options invalidation, masked/clear BYOK, accepted metrics, copy/download, focus and labels |
| CLI credential behavior | Masked interactive getpass, noninteractive missing-key failure, shared polishing, Ctrl+C exit 130 and documented exit codes |
| Training preparation | Bounded streaming sample, provenance, malformed/duplicate/unchanged-derivative audit, disjoint lineage checks, smoke/full separation, identity-validated resume, optimizer checkpoints every 250 steps plus epoch saves |
| Training evaluation | Implemented single-window/document metrics, macro/per-class F1/precision/recall, confusion matrices, Brier/ECE, majority baseline, human false positives, length-bucket error analysis, JSON/Markdown reports |
| Packaging | 0.3.0 wheel built; isolated runtime dependencies installed; imports confirmed from installed package outside checkout; deterministic CLI and offline actual-model HTTP/CLI checks pass |
| CI and docs | Deterministic Python/build/browser/audit workflow; updated README, API/job policy and training report |

## Verification results

- 81 Python tests passed. One upstream Starlette TestClient/httpx deprecation warning remains.
- Next.js 16.4.0 production build passed.
- Seven Chromium workflow tests passed. They use actual browser interaction with explicitly simulated API responses; they are not model-quality evidence.
- npm audit reports zero vulnerabilities after compatible framework/transitive updates.
- pip check passed in both the application and isolated wheel environments.
- All 50 Python source/training files parse successfully.
- Actual provisioned offline inference passed against the separate smoke artifact: HTTP long classification, rewrite, compare and jobs; CLI rewrite/classify/compare.
- A real 90-row dataset was audited and a two-step CPU smoke training run passed. Peak process RSS was about 2.50 GiB; total measured run time about 21.16 seconds. No held-out smoke test quality was presented.

## Remaining mandatory gates

| Gate | Current state |
|---|---|
| Full audited dataset | Running: 5,000 human and 5,000 AI originals loaded; 5,000 synthetic paraphrases being generated with per-batch checkpoints |
| Full three-epoch release model | Queued behind generation/audit in the local full-run sequence |
| Held-out release quality and error analysis | Pending full training; automatic evaluation reports are implemented |
| Actual release-artifact runtime/offline validation | Pending full model; automatic validation follows training |
| Release readiness in a real demo | Pending release artifact; smoke artifact must not be presented as release quality |

Live DeepSeek verification is unverified and requires a runtime key and authorized usage. Remote push/publication was not attempted; historical authentication failure was not rechecked. Neither is represented as successful.

## Active local CPU run

The user explicitly selected this local CPU. Do not start a duplicate run.

- Entry point: python -u -m scripts.complete_training
- Durable state and PIDs: data/processed/training-run.json
- UTF-8 log: data/processed/full-training.log
- Generation checkpoint: data/raw/ai_paraphrased.checkpoint.csv
- Final reports/artifact: models/roberta-classifier/
- Separate engineering artifact: models/smoke-classifier/

The state file's complete flag becomes true only after dataset, audit, full training and actual-artifact validation finish. The validator also performs a provisioned offline follow-up. See docs/TRAINING_EVALUATION.md for limitations and safe resume instructions. A running process is not evidence of completed training.
