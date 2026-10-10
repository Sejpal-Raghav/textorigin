# TextOrigin project completion prompt

Restored from the draft in the conversation after the original file creation was interrupted. Execute these requirements; do not stop at a plan or scaffold.

## Objective and starting point

Complete the existing local-first English text classifier and conservative humanizer: Python CLI, FastAPI, Next.js, local RoBERTa, similarity/feature models, optional DeepSeek. Preserve working behavior and compatibility. Read applicable AGENTS.md and installed Next.js documentation. Search for SPEC.md, BACKLOG.md and IMPLEMENTATION-HANDOFF.md; record if absent. Maintain an evidence checklist and implementation ledger.

Verified 9 October 2026: branch feat/humanizer-mvp at 5059db2; deterministic rewrite, chunked guarded web rewriting, BYOK, CLI compare and shared-service polish exist. No dataset or classifier artifact exists locally. Classification truncates to one window. Jobs/cancellation and masked CLI key entry are missing. Lightweight tests: 35 pass; full collection blocked by missing NumPy/pandas. Frontend build passes. Ledger is stale.

## Invariants

Classification stays local. DeepSeek only polishes with explicit consent. Request keys override owner keys; never fall back on authentication failure. No persisted user input, result, or credentials. Log neither text nor secrets. Model scores are not calibrated confidence or proof of authorship; keep observations separate. Preserve protected content and structure. Deterministic local CLI rewrite remains usable without a classifier or network and is not described as semantic validation. No arbitrary product word cap or silent truncation. Do not replace production scores with mocks. Preserve existing user changes. Do not publish, push, deploy publicly, or spend on cloud/API services without authorization.

## Environment

Verify pins against actual package availability and platform compatibility. Synchronize pyproject.toml and requirement files. Establish reproducible Python and frontend installs. Keep model imports/downloads lazy and caches provisionable for offline use. Run complete test collection. Test clean wheel installation outside the source checkout and configurable model paths.

## Rewrite correctness

Protect whole multiline fenced code, inline code, quotes, URLs, citations, numeric units and Markdown indentation/line breaks. Fix line-by-line protection and placeholder collisions. Oversized protected spans must pass through unchanged with a warning rather than exceed processing bounds. Semantic acceptance must cover every edited segment, including tails beyond tokenizer limits; enforce minimum per segment, not only average. Validate optional polish against the original. Return the last accepted text on failure. Metrics must describe the final accepted output, with intermediate metrics labeled separately. Test CRLF, Unicode, fenced code, formatting, long spans, tail changes and fallback.

## Classification and compare

Implement token windows covering all content, special-token limits, deterministic aggregation and bounded batches. Document stride/overlap and ensure repeated context is not overweighted. Return normalized three-class scores, unique analyzed/total token count, windows, completion/coverage and model version. Preserve short inputs. Full analysis must not be reported truncated. Shared compare service must classify original and final accepted output under the same model/configuration, with signed percentage-point differences. Expose in API, jobs, CLI and web. Test coverage and arithmetic with controlled models, then real trained artifacts.

## Jobs

Implement bounded in-memory queue and worker lifecycle for classify/humanize/compare, retaining synchronous compatibility. Typed POST /jobs, GET /jobs/{id}, POST /jobs/{id}/cancel or consistent equivalents. States queued/running/cancel_requested/cancelled/completed/failed. Report actual stages and counts; do not invent progress while loading. Check cancellation between batches/chunks/provider calls; network/kernel work may finish before cancellation. Enforce admission, concurrency, result TTL/count limits, opaque IDs, credential cleanup, permit release, race safety and graceful shutdown. Keep status/health responsive. Document single-process and restart constraints. Test lifecycle, saturation, expiry, failures and races.

## Frontend

Separate classify/rewrite/compare actions; allow rewrite without classifier readiness. Progress, cancellation, errors, retry/reset, source-bound results and stale response protection. Show final-output metrics, coverage and compare scores. Keys remain only in component memory with a clear action. Retain copy/download. Keyboard labels, visible focus, accessible status and responsive layout. Production build plus actual browser workflow tests, including unavailable model and cancellation.

## CLI

Retain text/file/stdin, JSON/text and overwrite protection. Masked getpass for missing interactive DeepSeek keys; never hang noninteractive input. Environment keys or secure prompt only, no secret command arguments. Shared services, Ctrl+C, documented exit codes, stdout JSON/stderr errors and progress. Actionable model errors. Preserve classify.py compatibility.

## Data and training

Verify upstream schema, pinned revisions, provenance and license constraints. Deterministic three-class sample (currently 5000 per class), resumable paraphrases, clean audit output as training input. Detect invalid labels, blanks, duplicates/conflicting lineage; no original/derivative or normalized duplicate across split boundaries. Persist disjoint groups and split hashes; tune only on validation.

Run separate small measured smoke training first, then intended full three-epoch run on authorized available local compute. Never install smoke artifacts as the release model. Correct CPU/CUDA precision, checkpoint resume, memory limits, optimizer/scheduler/seed/split identity. Save model/tokenizer, labels, dependency/hardware/configuration and dataset/code provenance manifest.

Evaluate accuracy, per-class precision/recall/F1, macro/weighted F1, confusion matrix, Brier/ECE, baselines, human false positives and error analysis. Evaluate serving document policy too. Do not fabricate quality thresholds or claim scores are calibrated; validation-only calibration if implemented. Explain synthetic paraphrase and domain limitations. Actual full artifact and real inference are mandatory; if data access/compute is blocked, finish independent work and record exact reproducible resume instructions.

## Operations, tests and deliverables

Cheap liveness; readiness distinguishes file presence, validated load and warm state, corrupt artifacts and labels. No unrelated import failures without classifier. Correct installed-wheel model paths, CORS, timeout, safe concurrency and consistent typed errors. Trusted-local scope without database or hosting expansion.

Run all Python tests, frontend build, browser flows, actual-model CLI/HTTP long-document tests, optional authorized live DeepSeek, clean wheel install and provisioned offline tests. Keep heavyweight/provider tests opt-in and run explicitly when prerequisites exist. No skipped failing tests or fabricated green reports. Add deterministic CI without paid calls/downloads.

Update README, implementation status, API/job docs, evaluation report, env examples, packaging instructions and local artifact hashes. Do not commit data, weights or secrets. Prepare local deliverables; remote push only with authorization.

## Mandatory completion gates

- [ ] Reproducible dependency installation and complete Python suite
- [ ] Frontend build and browser workflows
- [ ] Protected rewrite, full semantic coverage and accurate final metrics
- [ ] Full-document classification and shared compare through all interfaces
- [ ] Bounded jobs, truthful progress and server cancellation
- [ ] Secure interactive/noninteractive CLI behavior
- [ ] Audited lineage-safe data and full trained/evaluated classifier
- [ ] Actual model readiness and long-document CLI/HTTP inference
- [ ] Provisioned offline runtime and clean wheel installation
- [ ] Accurate documentation/status and no secrets or fake production results

Report mandatory gates individually with evidence. Mark live-provider/remote publication prerequisites separately. Do not say complete while required training or real-model gates remain blocked. Give exact startup and resume commands with artifact paths.
