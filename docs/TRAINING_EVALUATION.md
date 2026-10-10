# Training and evaluation evidence

## Verified smoke run (9 October 2026)

This run validates engineering paths only. It is not the release classifier and establishes no useful classifier accuracy.

- Dataset: pinned artem9k/ai-text-detection-pile revision 9bc306f3fee2dab8affc53de97d012722e96f181.
- Base model: FacebookAI/roberta-base revision e2da8e2f811d1448a5b465c236feacd80ffbac7b.
- Paraphraser: humarin/chatgpt_paraphraser_on_T5_base revision f86b18a7feb0b8149b7612a620b61e2e26dbf82e.
- Sample: 30 human, 30 AI-written, 30 AI-paraphrased records, 90 audited rows.
- Audit: no blank/invalid or cross-family exact duplicate exclusions in this small sample.
- Clean dataset SHA-256: 92d675ffee5b3e7232ec0c769f073c107008fb44143ef3440e79612f3fe8bd7f.
- CPU-only PyTorch 2.13.0; batch size 1; accumulation 1; gradient checkpointing; two update steps; no held-out test evaluation.
- Peak process RSS: 2,689,245,184 bytes (about 2.50 GiB).
- Measured total training/evaluation/save time: about 21.16 seconds, with cached base weights.
- Separate artifact: models/smoke-classifier. Manifest purpose is smoke_only_not_release.
- Provisioned offline real-model validation passed: 4,575 content tokens across 19 classifier windows, HTTP rewrite/compare/jobs and CLI rewrite/classify/compare.
- Smoke weights SHA-256: ae774f5d9b3bae964eeb1e7f41fab4ea258f16f38eb5e6fc1659255fc4ebda46.

Small-sample paraphrase generation took about 3 minutes 36 seconds. Scaling that measurement to 5,000 paraphrases suggests roughly ten hours before classifier training, with substantial variability. The full runner uses four CPU threads and batch size 1 for memory control, so the initial measurement is only an estimate.

## Authorized full CPU run

The user explicitly selected local CPU execution. The full sequence is started by:

    .\.venv\Scripts\python.exe -u -m scripts.complete_training

Its durable status is data/processed/training-run.json and its UTF-8 child log is data/processed/full-training.log. The JSON records the parent/child PIDs, current stage, completed stages and complete flag. It is authoritative for current progress; this document is a verification snapshot.

The intended sequence builds 5,000 samples per class, audits the output, trains three epochs with lineage-disjoint train/validation/test partitions, and invokes actual-artifact runtime validation. The final dataset may have fewer rows after justified audit exclusions. Models and checkpoints are local ignored artifacts.

Dataset sampling is deterministic bounded-buffer streaming (seed 42, buffer 10,000, first valid unique sample per class), not a claim of perfectly uniform sampling over the entire corpus. Content hashes provide stable streaming identities. Original AI samples and derivatives share source IDs.

The current paraphraser uses a 512-token source context and 128-token generation maximum. This is an offline synthetic-label limitation and is recorded here explicitly; it is independent of full-document runtime classification and rewrite processing. Synthetic derivatives can differ in length/style from original AI and create shortcuts. Report domain and length-related errors rather than claiming robustness from one corpus.

No full-run quality results are available until models/roberta-classifier/evaluation.json exists. Do not use smoke scores as release results.

## Final artifacts and interpretation

The full trainer saves evaluation.json, textorigin-manifest.json, training-identity.json, resource-measurement.json, model/tokenizer files and resumable Trainer checkpoints. Optimizer/scheduler/RNG state is saved every 250 update steps as well as at epoch boundaries; validation-based model selection stays at epoch boundaries.

The report contains per-class precision/recall/F1, macro and weighted F1, accuracy, confusion matrices, multiclass Brier score, ten-bin ECE, majority baseline, human false-positive rate, split hashes, code/dependency provenance and memory/device measurements. It evaluates both single-window and document aggregation policies. Model selection uses validation; held-out evaluation follows the full run. No calibrated-confidence claim or fabricated quality target is made.

Resume an interrupted training phase with:

    .\.venv\Scripts\python.exe -u -m scripts.complete_training --resume-training

The checkpoint must belong to the same output directory and identity. If a stale training-run.lock remains after an abnormal machine shutdown, first verify both recorded processes have exited, then remove only that specific lock. Do not delete data/checkpoints to force a clean restart.

Optional empirical heuristic calibration remains separate from classifier scores. Optional live DeepSeek verification requires a runtime key and authorized usage and is not implied by offline/model tests.
