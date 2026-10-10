# API, jobs and runtime policy

The API is intended for a trusted local workstation. Use one Uvicorn process when using jobs: the queue and results live in that process's memory, expire, and disappear on restart. No database is used.

## Endpoints

| Endpoint | Behavior |
|---|---|
| GET /health | Cheap process liveness |
| GET /ready | Checks required artifact files, configuration JSON and label mapping; reports validation level, loaded state and inference verification |
| GET /ready?validate=true | Also loads local weights/tokenizer to check their compatibility; may take time |
| POST /classify | Synchronous document scores and optional separate observations |
| POST /humanize | Synchronous similarity-validated rewrite; optional DeepSeek |
| POST /compare | Shared-service rewrite and original/revised classification |
| POST /jobs | Returns HTTP 202 with an opaque job ID and status URL |
| GET /jobs/{id} | Returns current state, actual stage/counts, terminal result or safe error |
| POST /jobs/{id}/cancel | Idempotent server-side cancellation request |

The interactive web application uses jobs rather than keeping a long inference request open. Synchronous routes remain for compatibility. OpenAPI at /docs contains request/response schemas.

## Job request and result

Submit an operation, text, and supported options. Example body:

    {
      "operation": "humanize",
      "text": "It is worth noting that we utilize this method.",
      "style": "natural",
      "strength": "light",
      "use_llm": false
    }

Classify supports include_signals; humanize/compare support style, strength, optional similarity_threshold, seed, use_llm and deepseek_api_key. Keys are secret fields, excluded from results and errors. Unknown fields are rejected.

States transition from queued to running and then completed, failed or cancelled. Running cancellation goes through cancel_requested. Queued cancellation removes the job immediately. A cancelled operation has no successful result and complete=false. A late computation cannot overwrite cancellation. Cancelling a terminal job leaves it unchanged.

Progress reports an actual stage with completed_units and total_units where known. Counts reset when entering another stage; they are not an estimated overall percentage. Model loading and feature extraction have no invented percentage. Cancellation is cooperative between windows/chunks and provider calls; a current kernel or bounded network call may finish first.

A missing/expired job returns 404. Admission exhaustion returns structured 503 with Retry-After. Queue size, retained result count, terminal TTL and transient payload memory reservation are configurable. The operational byte budget rejects work explicitly rather than truncating text. Terminal jobs are also evicted oldest first when retention capacity is needed.

Defaults: one worker, four queued jobs, 32 retained records, 600-second terminal TTL, 32 MiB transient reservation budget. Shutdown requests cancellation and waits a bounded interval before releasing retained jobs.

## Document classification

RoBERTa processes disjoint content-token windows of manifest max_length minus special tokens. The default trained length is 256. Scores are the content-token-weighted mean of window softmax scores. Every content token receives one unit of weight; there is no overlap double counting.

Responses retain human, ai_written, ai_paraphrased, observed_signals, truncated, analyzed_tokens, model_version and warnings. Added fields: total_tokens, window_count, coverage, complete and aggregation. analyzed_tokens counts unique content tokens and excludes special tokens. Complete document results have truncated=false and coverage=1.0.

This policy trades cross-window context for predictable coverage and weighting. Its actual held-out performance is evaluated separately from the training single-window policy. Scores are uncalibrated and do not prove authorship.

## Rewrite acceptance

Whole fenced/inline code, URLs, quotes, numeric citations and numbers/units are protected. Prose edits retain indentation, list markers, paragraph/newline structure and protected content. There is no placeholder substitution into user input.

Chunks reconstruct the input losslessly. Oversized protected regions pass through unchanged with a warning. The semantic adapter divides both candidates into balanced corresponding token windows within the MiniLM limit and returns the minimum cosine similarity across them. Every segment must meet the threshold; a high average cannot hide a changed tail. This conservative positional comparison can reject valid major reordering.

Heuristic and optional provider candidates are checked against the original. Rejected candidates return the last accepted text. final_metrics describes humanized_text; post_heuristic_metrics and post_llm_metrics are explicitly intermediate-stage diagnostics. Cancellation raises an operation cancellation; it does not return an incomplete rewrite as successful.

The deterministic CLI local rewrite applies protected lexical rules without a similarity model. The HTTP/compare/provider paths additionally require provisioned feature/similarity models.

## Credential and data lifecycle

DeepSeek is only called when use_llm=true. An explicit request key takes precedence over the owner key, including failure behavior: no silent fallback. Web keys remain in page memory and have a clear action. CLI interactive key entry is masked; noninteractive calls require DEEPSEEK_API_KEY.

Text/results live in memory until job expiry/eviction and are not written to application storage. Training artifacts are a separate offline workflow. Logging records stage/duration/status, never submitted text or keys. Defaults are appropriate for a trusted local user; opaque job IDs are not a replacement for authentication on a public deployment.
