# Document classifier improvement plan

The completed GPU baseline achieved 79.1% first-window accuracy, but 58.4%
document accuracy and 4% AI-written recall. These are saved results, not new
measurements. Training used only document prefixes; inference averages all
nonoverlapping token windows. This mismatch is a hypothesis to investigate.

## Implementation

1. Use the inference window boundaries for training, covering every content
   token, including the final short window. Split source families before
   expanding documents into windows.
2. Give each document equal total loss weight within its class, distributing
   that weight by content-token count. Balance class totals using training
   document counts only. This avoids long documents dominating the objective.
3. Evaluate the real document aggregation policy each epoch. Select checkpoints
   using validation document macro F1, and record per-class precision/recall,
   human false positives, and calibration. Test data remains final evaluation.
4. Provide an existing-artifact diagnostic on the original validation split:
   first-window versus full-document metrics, length buckets and window-position
   errors. Verify dataset and split identity before comparison.
5. Record window/weight/selection policy in run identity and reports. Preserve
   the old artifact and use a new output directory; changed training cannot
   resume an old checkpoint.

## Experiment and acceptance

First diagnose the existing artifact on validation data. Then train a fresh
candidate on the same audited dataset and fixed source-family partitions.
Compare validation document macro F1, AI-written precision/recall, human false
positives, and errors by length. Choose tradeoffs on validation, not test.
Once the approach is fixed, report the complete held-out metrics and run actual
artifact online/offline verification. The existing test results have already
been inspected, so an additional untouched external evaluation is desirable
before claiming generalization.

More diverse human/AI domains and paraphrase generators, length-matched data,
and document-level objectives are later experiments if window training is
insufficient. More epochs, a different decision threshold, or more data alone
are not assumed to solve the current failure. No accuracy gain is promised.

## Running on the original GPU machine

Keep the original CSVs and audit report: they are excluded from Git. With the
updated source and installed GPU environment:

```powershell
.\.venv-gpu\Scripts\python.exe -m scripts.diagnose_classifier --model-path runs/gpu-fixed/model --data runs/gpu-fixed/combined.clean.csv --audit-report runs/gpu-fixed/audit.json --output runs/gpu-fixed/validation-diagnostic.json
.\.venv-gpu\Scripts\python.exe -m scripts.train --data runs/gpu-fixed/combined.clean.csv --audit-report runs/gpu-fixed/audit.json --output-dir runs/gpu-document-v2/model --epochs 3 --train-batch-size 16 --eval-batch-size 32 --gradient-accumulation-steps 1 --mixed-precision fp16 --no-gradient-checkpointing
.\.venv-gpu\Scripts\python.exe -m scripts.verify_runtime --model-path runs/gpu-document-v2/model
```

Full-window expansion and document validation increase runtime; the previous
three-epoch duration is not an estimate for this run. Do not overlap training
runs on a memory-constrained GPU. The diagnostic and training commands retain
the existing serving aggregation policy.

CUDA defaults now target throughput: training batch 16, evaluation batch 32,
one accumulation step, checkpointing off, fused AdamW, padding to multiples of
eight, and pinned/nonblocking data transfers. Inputs are cached before training;
Windows data-loader workers default to zero to avoid copying the cached corpus
and repeatedly importing the model stack. `--dataloader-workers 2` enables
persistent workers and prefetching if a measured CPU bottleneck remains.
Document evaluation batches windows across documents to reduce undersized GPU
batches. CPU defaults retain smaller batches and gradient checkpointing.

The effective GPU batch changes from 8 to 16. Use `--train-batch-size 8
--gradient-accumulation-steps 1` to retain effective batch 8 while eliminating
the old four-pass accumulation. If batch 16 exceeds available memory, lower the
training/evaluation batches in a fresh output directory and optionally enable
`--gradient-checkpointing`. VRAM capacity alone does not guarantee a batch fits.
The report records throughput, effective batch, and allocated/reserved peak
CUDA memory; gains must be benchmarked on the original GPU.
