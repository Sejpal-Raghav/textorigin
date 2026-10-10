# TextOrigin

TextOrigin is a local-first English text-origin classifier and conservative rewriting tool. A fine-tuned RoBERTa model produces human-written, AI-written and AI-paraphrased model scores. Separate linguistic observations describe the text; they do not explain or alter classifier scores.

The application includes a Python CLI, FastAPI backend and Next.js interface. The web UI supports classify, rewrite and compare workflows with actual job progress and server cancellation. Classification covers the complete document in bounded token windows. Rewriting preserves protected content and structure and, through the service path, validates every semantic window before accepting edits. DeepSeek is an optional polishing stage; classification stays local.

The full release classifier is being trained on the explicitly authorized local CPU sequence. Check data/processed/training-run.json for its current state. The separate models/smoke-classifier artifact is for engineering validation and must not be presented as release-quality classification. See docs/IMPLEMENTATION_STATUS.md and docs/TRAINING_EVALUATION.md for evidence and open gates.

## Architecture

    Offline: pinned public data -> lineage -> resumable paraphrases -> audit -> grouped split -> RoBERTa -> evaluation/manifest
    Runtime: Next.js / CLI -> FastAPI / services -> local model adapters / optional DeepSeek polish
    Jobs: bounded single-process memory queue -> progress / cooperative cancellation -> expiring result

- src/api: HTTP contracts, lifecycle and compatibility facades
- src/services: classification, humanization, comparison, capacity and jobs
- src/inference: local classifier, GPT-2 features and MiniLM similarity
- src/integrations: optional DeepSeek adapter
- src/data, src/model, scripts: offline data/training/evaluation
- frontend: Next.js interface and Playwright workflow tests
- tests: deterministic Python and API contract tests

No database is used. Runtime text and results are not persisted. Job payloads/results expire from memory and disappear on restart.

## Setup

Python 3.10–3.12 is supported by the pinned stack; verification used Python 3.12 on Windows. Install serving dependencies:

    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

For training and tests:

    .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

For the exact verified Windows/Python 3.12 dependency set:

    .\.venv\Scripts\python.exe -m pip install -r requirements-windows-py312.txt

Training pins Arrow 21 because newer Arrow wheels require NumPy 2 at import time while this project uses NumPy 1.26/Pandas 2.1. GPU installations require an appropriate compatible PyTorch CUDA wheel; the verified local environment is CPU-only. Explicit fp16/bf16 training is rejected on unsupported devices.

Use .env.example as a configuration reference. Python reads process environment variables and does not implicitly load .env. Model imports and CLI help do not download model resources. Provision pinned feature/similarity models into the Hugging Face cache before offline use.

Classifier artifacts default to models/roberta-classifier in a source checkout. An installed wheel defaults to ~/.textorigin/models/roberta-classifier. TEXTORIGIN_MODEL_PATH overrides either default; use an absolute path when operating from several working directories. Relative overrides resolve from the package root.

## Run the application

Start the API from the checkout:

    .\.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

Use a single process for jobs. Start the frontend separately:

    cd frontend
    npm ci
    npm run dev -- --hostname 127.0.0.1

Open http://localhost:3000. Set NEXT_PUBLIC_TEXTORIGIN_API_URL at frontend build time to change the API location.

GET /health is process liveness. GET /ready checks artifact files/configuration and reports whether weights are loaded; GET /ready?validate=true also validates weight/tokenizer loading. The frontend describes artifact availability accurately and leaves rewrite accessible without a classifier, provided feature/similarity models are provisioned.

POST /classify, /humanize and /compare remain synchronous compatibility interfaces. The web app uses POST /jobs, GET /jobs/{id} and POST /jobs/{id}/cancel. See docs/API_AND_JOBS.md and the live OpenAPI /docs for contracts.

## Rewrite and compare policy

Styles: natural, formal, casual. Strengths: light, medium.

Fenced and inline code, quoted content, URLs, numeric citations and numbers/units are protected. The editor preserves indentation and newline/list structure. Very long protected regions pass through unchanged with a warning. Per-window similarity failures and provider failures return the last accepted text. final_metrics describes the accepted output; intermediate metrics are labeled separately.

The semantic policy uses the minimum score across bounded corresponding token windows, covering both texts completely. It is intentionally conservative and may reject valid large rearrangements. It is a safeguard, not a proof of semantic equivalence.

Compare uses shared services to classify the original and final accepted rewrite with the same model. Score deltas are signed percentage-point changes, not proof of writing quality or authorship.

## Optional DeepSeek

Set DEEPSEEK_API_KEY in the process environment for an owner-configured key, or provide a request-scoped web key. Interactive CLI entry is masked when needed. Noninteractive CLI calls never wait for a key prompt.

A request key takes precedence and is never silently replaced after an authentication failure. Web keys remain in page memory with a clear control. Text is sent externally only when polishing is selected. Provider calls/retries have a finite deadline and safe fallback behavior.

## CLI and packaging

Build and install the current wheel:

    .\.venv\Scripts\python.exe -m build --wheel
    python -m pip install dist/textorigin_humanizer-0.3.0-py3-none-any.whl

Examples:

    humanizer rewrite --text "It is worth noting that we utilize this." --format json
    humanizer rewrite --file sample.txt --style formal --strength medium --output revised.txt
    humanizer rewrite --file sample.txt --provider deepseek --format json
    humanizer classify --file sample.txt --format json
    humanizer compare --file sample.txt --style natural --strength light --format json

Text/file/stdin input is supported. Existing output files require --force. Deterministic local rewrite works without trained models or network access; it does not run the semantic model. Provider rewriting and compare use the validated service. Classify/compare require an actual trained artifact.

JSON goes to stdout, errors to stderr. Exit codes: 0 success, 1 application failure, 2 usage/input failure, 3 unavailable classifier/local model, 130 interrupted operation. The original python classify.py compatibility interface remains.

## Full local training and progress

The authorized full sequence is launched once:

    .\.venv\Scripts\python.exe -u -m scripts.complete_training

It generates 5,000 examples per class, audits data, trains three epochs with memory-controlled CPU settings, then validates the real artifact. Do not launch a duplicate while data/processed/training-run.lock is live. Progress and child PIDs are in data/processed/training-run.json; the UTF-8 log is data/processed/full-training.log.

Manual stages:

    .\.venv\Scripts\python.exe -m src.data.pipeline --sample-size 5000
    .\.venv\Scripts\python.exe -m scripts.audit_dataset
    .\.venv\Scripts\python.exe -m scripts.train --data data/processed/combined.clean.csv --audit-report data/processed/audit.json --output-dir models/roberta-classifier --epochs 3

A separate measured smoke run must precede a full run on a new machine. Use --smoke with a separate output directory. Existing run output requires explicit --resume-from-checkpoint latest. The full runner supports --resume-training after interruption. Resume validates dataset/split/configuration identity.

Original AI and derived samples share source_id and remain in one partition. Cross-family normalized exact duplicates must be audited before splitting. Streaming sampling is bounded and deterministic, with its buffer-sampling approximation recorded. Provenance, split hashes, dependency/code state, training configuration, device/memory and evaluation are persisted alongside the artifact.

Training selects with validation and reports held-out per-class precision/recall/F1, macro/weighted F1, accuracy, confusion matrices, Brier/ECE, majority baseline and human false positives. It evaluates the serving document aggregation policy separately. Do not call scores calibrated confidence or generalize one synthetic dataset to proof of authorship.

New full training runs use every nonoverlapping document window, balance class/document loss weights, and select checkpoints by full-document validation macro F1. `--window-policy first_window` retains prefix training for comparison; smoke runs stay prefix-only. These changes require a fresh model output directory. The completed GPU baseline has substantial full-document errors; see [the improvement plan](docs/MODEL_IMPROVEMENT_PLAN.md) for diagnostics, retraining commands, and the evaluation procedure. Improvements must be measured on a new candidate.

Feature extraction/calibration remain optional analysis; they are not prerequisites for classifier training and never change classifier scores.

## Windows GPU training

From the repository root in PowerShell, use Python 3.11 and an NVIDIA GPU with a compatible driver. Setup requires internet access and at least 15 GiB of free disk space; 25 GiB or more is recommended.

```powershell
git pull origin master
.\run_gpu.cmd --run-dir runs/gpu
```

The launcher creates or reuses `.venv-gpu`, installs CUDA PyTorch and training dependencies, checks CUDA with an actual GPU operation, then generates the dataset, audits it, trains the classifier and validates the saved model. CPU fallback is disabled. Defaults are 5,000 samples, three epochs, FP16, paraphrase batch two, training batch 16, evaluation batch 32, one gradient accumulation step and gradient checkpointing off. CUDA training uses fused AdamW, cached inputs and padding to multiples of eight. See [GPU throughput settings](docs/GPU_TRAINING.md#throughput-on-an-8-gb-gpu) for memory overrides and benchmarking. For available options, run `py -3.11 run_gpu.py --help`.

Each stage writes a log under the selected run directory, such as `setup-cuda.log` or `training.log`. `status.json` contains progress and process IDs. Successful output is saved under `<run-dir>/model`; wait for `ALL STAGES PASSED` before sharing the complete model folder.

### Recovering from the Windows status-file error

If an older runner stopped with `[WinError 5] Access is denied` while replacing `status.tmp` with `status.json`, update the checkout and start in a fresh run directory:

```powershell
git pull origin master
.\run_gpu.cmd --run-dir runs/gpu-fixed
```

Keep any sample-size, epoch, precision or batch-size options from your previous command. Use another fresh directory name if `runs/gpu-fixed` already contains a run with different settings or source code. The runner records a source hash, so updating `run_gpu.py` requires a new run directory; the previous artifacts remain in their original folder.

Status writes now use unique temporary files and retry permission errors. If access remains denied, the runner warns that status may be stale and continues; check the console and stage logs for progress. Critical configuration and checkpoint metadata write failures still stop the run.

For an interruption with unchanged source and settings, rerun the same command to reuse verified completed stages and resume training from its latest checkpoint. If training stopped before its first checkpoint, preserve or rename the run's `model` directory before retrying. If the runner reports `Run is locked`, verify that the PID in `pipeline.lock` and all child processes have stopped before removing that lock and restarting.

## Verification

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m pip check
    cd frontend
    npm run build
    npx playwright install chromium
    npm test
    npm audit --audit-level=high

Default Python tests use fake services and do not download models or call paid providers. Browser workflow tests run actual Chromium with explicitly simulated API responses. CI performs deterministic Python/package/frontend checks.

Actual-artifact verification is opt-in:

    .\.venv\Scripts\python.exe -m scripts.verify_runtime --model-path models/roberta-classifier
    .\.venv\Scripts\python.exe -m scripts.verify_runtime --model-path models/roberta-classifier --offline

Provision the caches before --offline. This validator checks real HTTP and CLI classify/rewrite/compare/jobs and long-document coverage. Smoke artifacts require explicit --allow-smoke and never satisfy release-model acceptance. Live DeepSeek verification is separate and requires a runtime key and authorized usage.

## Deployment boundary

Defaults are for a trusted local workstation, without authentication or rate limiting. Public deployment requires an authorized edge enforcing TLS, authentication and request limits. Each server process owns its model copies; use one process for transient jobs. Keep inference concurrency at 1 unless capacity has been measured.

Queue/count/TTL/memory settings are operational controls, not silent product truncation. Cancellation is cooperative: current model kernels/network calls may finish first. Jobs expire, results are source-bound in the UI, and input changes cannot attach a late result to new text.
