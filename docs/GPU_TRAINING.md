# Unattended GPU training

Send your friend the entire project folder (or a ZIP of it), including run_gpu.py, scripts/, src/, pyproject.toml and README.md. The launcher is not a standalone copy of the project. Exclude .venv/, .venv-gpu/, frontend/node_modules/, frontend/.next/ and active CPU run locks/status files. Existing data is optional; the default GPU run uses its own runs/gpu/ folder.

## Before the first run

- NVIDIA CUDA-capable GPU; defaults target an 8GB VRAM machine. AMD and Apple GPUs are not supported by this launcher.
- Install Python 3.11 and a compatible NVIDIA driver before starting. Python 3.10–3.12 are supported by the Python launcher.
- Have internet access for packages, the dataset and pretrained models. No DeepSeek key or paid inference service is needed.
- Allow at least 25GB free disk space, including the Hugging Face cache and checkpoints. 8GB system RAM is tight: close browsers/games and leave the OS paging file enabled.
- Plug in the machine. Windows sleep is inhibited while the launcher runs; lid-close, shutdown and closing the terminal can still interrupt it. On Linux, disable sleep or use tmux/screen.

## One command on Windows

Extract/open the project folder, open a terminal there, and run:

~~~powershell
py -3.11 run_gpu.py
~~~

Alternatively double-click run_gpu.cmd (requires Python 3.11 and the Windows Python launcher). Leave its terminal open. It needs no input once started.

On Linux, from the project root:

~~~bash
python3.11 run_gpu.py
~~~

The launcher creates .venv-gpu/ automatically. You do not need to activate it or install packages manually. The default CUDA wheel channel is cu126; supported alternatives are cu130 and cu132. These channels are taken from the [official PyTorch 2.13 installation instructions](https://pytorch.org/get-started/previous-versions/). Driver/GPU compatibility is checked by executing a CUDA kernel before dataset generation. If your GPU requires another supported channel, choose it on the first run:

~~~powershell
py -3.11 run_gpu.py --cuda-index cu130
~~~

## What runs automatically

1. Create a separate Python environment, install pinned project/training dependencies and CUDA PyTorch, and run pip check.
2. Check CUDA with an actual GPU operation; record GPU, VRAM, free host RAM and disk space. There is no silent CPU fallback.
3. Stream 5,000 human and 5,000 AI-written originals from the pinned dataset.
4. Generate 5,000 T5 paraphrases, using batch size 2, four beams and existing source/output length limits. T5 generation stays in FP32. Each completed batch is checkpointed.
5. Save the combined dataset and provenance.
6. Audit/clean the data, removing malformed rows, duplicates and unchanged paraphrases; save the audit report and dataset hash.
7. Fine-tune pretrained RoBERTa for 3 epochs using FP16, max length 256, training batch 16, accumulation 1 (effective batch 16), evaluation batch 32 and gradient checkpointing off. CUDA uses fused AdamW, padding to multiples of eight, cached inputs and pinned/nonblocking transfers. Each stage runs in a separate process to release its model memory.
8. Run validation during training and held-out test evaluation afterward, including document-window aggregation, confusion matrices, baseline comparisons, error analysis and resource measurements. The split keeps source families together.
9. Save model, tokenizer, evaluation reports, manifest and checkpoint state.
10. Validate real HTTP/CLI classification, rewriting, comparison and background jobs, then repeat runtime validation offline after model caches have been populated.
11. Mark completion only when every stage succeeds. No frontend build/deployment or paid API calls are part of model training.

## Progress and outputs

All data, logs and outputs go to runs/gpu/ by default:

| Path | Contents |
|---|---|
| status.json | Current stage, PID, timestamp, failure reason and completion state |
| hardware.json | Actual GPU and memory information |
| setup-*.log | Dependency installation output |
| dataset.log | T5 progress |
| ai_paraphrased.checkpoint.csv | Completed paraphrases for restart |
| audit.log / audit.json | Audit execution and cleaning evidence |
| training.log | Training and evaluation progress |
| runtime-validation.log | HTTP, CLI and online/offline validation output |
| model/ | Model weights, tokenizer, evaluation reports, provenance, runtime reports and training checkpoints |

Watch progress in a second PowerShell terminal:

~~~powershell
Get-Content runs/gpu/status.json
Get-Content runs/gpu/dataset.log -Tail 20 -Wait
# Once training starts:
Get-Content runs/gpu/training.log -Tail 20 -Wait
~~~

Success prints ALL STAGES PASSED and sets status.json to state=completed and complete=true. Exit code 0 means success; 1 means failure; Ctrl+C exits 130. The launcher fails immediately when a command fails rather than treating incomplete work as a trained model.

## Restart and recovery

Rerun the exact same command after a normal failure or interruption. Successfully completed stages are skipped only after checking their inputs and output file hashes. T5 resumes completed batches. Training resumes from its latest checkpoint when one exists; optimizer/scheduler state is retained. Runtime validation reruns if its stage did not complete.

If training failed before its first checkpoint, the launcher stops with an explanation. After checking no worker is running, rename runs/gpu/model/ to preserve the failed attempt, then rerun the same command. Dataset generation and audit are reused. This avoids silently deleting an existing training identity.

After a power failure, pipeline.lock may remain. Read its recorded PID and verify the launcher and child workers have stopped before deleting that specific lock. Do not run two launchers against the same run folder/environment at once.

Settings and source are recorded. Changing them requires a new output folder. For a fresh more conservative run after a GPU memory error:

~~~powershell
py -3.11 run_gpu.py --run-dir runs/gpu-small --paraphrase-batch-size 1 --train-batch-size 1 --eval-batch-size 2 --gradient-accumulation-steps 8 --gradient-checkpointing
~~~

For a quick end-to-end engineering run (still downloads the required models):

~~~powershell
py -3.11 run_gpu.py --run-dir runs/gpu-small-check --sample-size 100 --epochs 1
~~~

This small run is not evidence of full-dataset model quality. Network/driver/disk failures still require fixing their cause; unattended execution cannot recover from a powered-off computer or missing hardware support.

## Throughput on an 8 GB GPU

The larger defaults are a starting point, not a measured memory guarantee. They
replace the earlier conservative batch-2/checkpointing setup. Gradient
checkpointing trades additional compute for lower activation memory; use it only
when needed to fit a useful batch. See the [Transformers explanation](https://huggingface.co/docs/transformers/grad_checkpointing).
Fused AdamW reduces optimizer overhead on CUDA; see [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

Use a fresh run folder after changing source or settings. To retain the previous
effective batch of eight while making each GPU pass larger:

~~~powershell
py -3.11 run_gpu.py --run-dir runs/gpu-throughput --train-batch-size 8 --eval-batch-size 32 --gradient-accumulation-steps 1
~~~

The default batch 16 changes effective batch size, so evaluate model quality
again. If memory is exhausted, lower training and evaluation batches, or enable
`--gradient-checkpointing`. More VRAM allocation alone is not a speed metric:
compare samples/second and epoch duration using the same dataset/window policy.
`evaluation.json` records trainer throughput, document-evaluation throughput,
effective batch, and allocated/reserved peak CUDA memory. The training timer
includes epoch validation and checkpoint writes.

Window inputs and prefix-evaluation tokens are cached before training. Workers
default to zero because Windows spawn can copy a large cached corpus into each
worker. `--dataloader-workers 2` enables persistent workers and prefetching if
profiling shows a loading bottleneck. Offline document evaluation fills batches
across document boundaries while retaining each document's token-weighted score.
Paraphrase generation keeps its existing FP32/batch-2 defaults; these throughput
changes target classifier training and evaluation.

## Return the results

After successful completion, ZIP the entire runs/gpu/model/ folder and also include audit.json, pipeline-config.json, hardware.json and status.json. The model folder includes model.safetensors, tokenizer files, config.json, textorigin-manifest.json, evaluation.json/.md and both runtime-verification reports. Keep the manifest and tokenizer with the weights.

To use the returned artifact, extract it to a model folder and point TEXTORIGIN_MODEL_PATH to that folder. Do not overwrite an active local training run's model directory. Evaluate the reports before calling the model accurate; the pipeline does not promise an accuracy threshold.
