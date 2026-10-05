# TextOrigin

TextOrigin is a local-first text-origin classifier and rewriting tool. A fine-tuned RoBERTa model produces scores for human-written, AI-written, and AI-paraphrased text. Separate linguistic observations describe the input but do not cause or explain the classifier score.

The humanizer applies conservative deterministic edits, verifies each bounded chunk for semantic similarity, and can optionally ask DeepSeek for a minor polished candidate. Both stages must meet a server-controlled similarity threshold or the last accepted text is returned. DeepSeek is only an editing stage; classification stays local.

## Architecture

```text
Offline: Hugging Face data -> source lineage -> paraphrases -> grouped split -> RoBERTa -> model manifest
Runtime: Next.js / CLI -> FastAPI routes -> application services -> local model adapters / optional DeepSeek polish
```

- `src/api`: HTTP routes, schemas, application lifecycle, and compatibility facades
- `src/services`: classification and humanization policy
- `src/inference`: synchronized RoBERTa, GPT-2, and MiniLM adapters
- `src/integrations`: optional DeepSeek polish adapter
- `src/data`, `src/model`, `scripts`: offline data and training workflow
- `frontend`: Next.js interface
- `tests`: unit and API contract tests that do not download models

No database is used. Input text and results are not persisted by this application.

## Runtime setup

Python 3.10+ is required. Install the serving dependencies:

```bash
pip install -r requirements.txt
```

Classification requires a trained artifact at `models/roberta-classifier` by default. The directory must contain the Hugging Face model/tokenizer files. Newly trained artifacts also contain `textorigin-manifest.json`, which records dataset hash, labels, max length, code revision, training parameters, and test metrics.

The feature and similarity models use pinned Hugging Face revisions and are downloaded by their libraries on first use. For an offline deployment, pre-populate the Hugging Face cache during provisioning. Runtime imports never download NLTK or model resources.

Use `.env.example` as a reference and set variables in the process environment when defaults are unsuitable. The Python application does not implicitly load `.env` files. Relative model paths are resolved from the repository, independently of the shell working directory.

Set `DEEPSEEK_API_KEY` only when the optional polish stage should use an owner-provided key. Web users can instead provide a request-scoped key. A supplied user key takes precedence and is never silently replaced by the owner key after an authentication failure. Text is sent to DeepSeek only when polishing is enabled.

Start the API:

```bash
uvicorn src.api.app:app --reload
```

- `GET /health` checks process liveness.
- `GET /ready` verifies that the required local classifier files exist.
- `POST /classify` returns model scores, truncation metadata, and optional heuristic observations.
- `POST /humanize` returns a typed status, warnings, similarity values, and before/after metrics.

Start the frontend in another terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open `http://localhost:3000`. Set `NEXT_PUBLIC_TEXTORIGIN_API_URL` at frontend build time when the API is not at `http://localhost:8000`.

## CLI

```bash
pip install ./dist/textorigin_humanizer-0.2.0-py3-none-any.whl
humanizer rewrite --text "It is worth noting that we utilize this."
humanizer rewrite --file sample.txt --style formal --strength medium --output revised.txt
humanizer classify --file sample.txt --format json
humanizer compare --file sample.txt --style natural --strength light --format json
```

`rewrite` works without a trained classifier. `classify` and `compare` return a model-unavailable error until the trained artifact is installed. The displayed percentages are model scores, not calibrated confidence estimates. The original `python classify.py` command remains available as a compatibility interface.

## Build and train

Install training and development dependencies:

```bash
pip install -r requirements-training.txt
# or, for tests as well:
pip install -r requirements-dev.txt
```

Build the lineage-aware three-class dataset:

```bash
python -m src.data.pipeline --sample-size 5000
```

Paraphrases are appended to `data/raw/ai_paraphrased.checkpoint.csv` after every batch. Re-running the same command resumes the deterministic sample instead of discarding completed CPU/GPU work.

Train directly from the text dataset:

```bash
python -m scripts.train --data data/raw/combined.csv --output-dir models/roberta-classifier
```

Original AI samples and their paraphrases share a `source_id` and are kept in the same train, validation, or test partition. This prevents derivative content from leaking across evaluation boundaries.

Feature extraction is optional analysis and is not a prerequisite for RoBERTa training:

```bash
python -m scripts.extract_features --input data/raw/combined.csv
```

To replace the built-in observation thresholds with reproducible empirical bounds from a grouped validation split:

```bash
python -m scripts.calibrate_thresholds --data data/features/features.csv
```

The generated file records its dataset hash, quantiles, seed, and sample count. It affects only the separate heuristic observations, never RoBERTa scores.

Training reports accuracy, per-class and weighted F1, a confusion matrix, Brier score, and expected calibration error. Evaluate calibration before presenting model scores as probabilities.

## Tests and builds

```bash
pytest -q
cd frontend
npm run build
```

Build and smoke-test the local wheel:

```bash
python -m build --wheel
pip install --force-reinstall dist/textorigin_humanizer-0.2.0-py3-none-any.whl
humanizer rewrite --text "It is worth noting that we utilize this."
```

See `docs/IMPLEMENTATION_STATUS.md` for the current spec-driven backlog evidence and known incomplete work.

Unit and API contract tests use fake services, so they do not require a trained classifier, a DeepSeek key, or downloaded transformer weights.

## Deployment and security boundary

The default configuration is intended for a trusted local workstation. It does not provide user authentication or rate limiting. Do not bind it directly to an untrusted network. If remote access is required, terminate TLS and enforce authentication and request limits at a trusted reverse proxy or deployment edge.

The backend processes long rewrite inputs in bounded chunks, limits concurrent inference, uses a finite DeepSeek timeout, does not log submitted text or credentials, and restricts CORS to the configured frontend origins. Inputs are constrained by available memory and provider context limits rather than a product word limit. Keep `TEXTORIGIN_MAX_CONCURRENT_INFERENCES=1` unless measured GPU/CPU capacity supports more; each server worker owns its own model copies.
