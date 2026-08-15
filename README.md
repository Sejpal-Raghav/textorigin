# TextOrigin

TextOrigin is a local-first text-origin classifier and rewriting tool. A fine-tuned RoBERTa model produces scores for human-written, AI-written, and AI-paraphrased text. Separate linguistic observations describe the input but do not cause or explain the classifier score.

The optional humanizer applies conservative deterministic edits, verifies their semantic similarity, and can ask a local Ollama model for a polished candidate. Both stages must meet a server-controlled similarity threshold or the original text is returned.

## Architecture

```text
Offline: Hugging Face data -> source lineage -> paraphrases -> grouped split -> RoBERTa -> model manifest
Runtime: Next.js / CLI -> FastAPI routes -> application services -> model adapters / Ollama
```

- `src/api`: HTTP routes, schemas, application lifecycle, and compatibility facades
- `src/services`: classification and humanization policy
- `src/inference`: synchronized RoBERTa, GPT-2, and MiniLM adapters
- `src/integrations`: optional Ollama client
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
python classify.py --text "Text to inspect."
python classify.py --file sample.txt --json
```

The displayed percentages are model scores, not calibrated confidence estimates. Very short or truncated inputs include warnings.

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

Unit and API contract tests use fake services, so they do not require a trained classifier, Ollama, or downloaded transformer weights.

## Deployment and security boundary

The default configuration is intended for a trusted local workstation. It does not provide user authentication or rate limiting. Do not bind it directly to an untrusted network. If remote access is required, terminate TLS and enforce authentication and request limits at a trusted reverse proxy or deployment edge.

The backend limits text size and concurrent inference, uses a finite Ollama timeout, does not log submitted text, and restricts CORS to the configured frontend origins. Keep `TEXTORIGIN_MAX_CONCURRENT_INFERENCES=1` unless measured GPU/CPU capacity supports more; each server worker owns its own model copies.
