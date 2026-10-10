"""One-command CUDA setup, dataset generation, audit, training and validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import venv

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = None
    try:
        # A unique, closed file avoids writer collisions and Windows handle locks.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.stem + '-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
        for attempt in range(6):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(0.1 * 2**attempt)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def save_status(path, status):
    """Status is advisory; a locked status file must not stop the pipeline."""
    try:
        save(path, status)
    except OSError as error:
        print(f'Warning: could not update {path}: {error}. Status may be stale; '
              'execution will continue.', file=sys.stderr, flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--sample-size', type=int, default=5000)
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--paraphrase-batch-size', type=int, default=2)
    result.add_argument('--train-batch-size', type=int, default=16)
    result.add_argument('--eval-batch-size', type=int, default=32)
    result.add_argument('--gradient-accumulation-steps', type=int, default=1)
    result.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False)
    result.add_argument('--dataloader-workers', type=int, default=0)
    result.add_argument('--optimizer', choices=['auto', 'adamw_torch', 'adamw_torch_fused'], default='auto')
    result.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    result.add_argument('--cuda-index', choices=['cu126', 'cu130', 'cu132'], default='cu126')
    result.add_argument('--run-dir', type=Path, default=Path('runs/gpu'))
    result.add_argument('--worker', choices=['preflight', 'dataset'], help=argparse.SUPPRESS)
    return result


def worker(args, run):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch build. CPU fallback is disabled.')
    if args.worker == 'preflight':
        import psutil
        # Actual kernel execution catches unsupported GPU architectures/drivers.
        probe = torch.ones((32, 32), device='cuda')
        (probe @ probe).sum().item()
        torch.cuda.synchronize()
        if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('GPU does not support BF16; select --precision fp16.')
        report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                  'cuda': torch.version.cuda, 'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
                  'available_ram_bytes': psutil.virtual_memory().available,
                  'free_disk_bytes': shutil.disk_usage(run).free}
        print(json.dumps(report, indent=2), flush=True)
        if report['free_disk_bytes'] < 15 * 1024**3:
            raise RuntimeError('At least 15 GiB free disk is required; 25 GiB or more is recommended.')
        save(run / 'hardware.json', report)
    else:
        import src.data.paraphrase as paraphrase
        from src.data.pipeline import build_dataset
        # Keep T5 in FP32 for generation stability; smaller batches bound beam-search memory.
        paraphrase.paraphrase_texts = functools.partial(
            paraphrase.paraphrase_texts, batch_size=args.paraphrase_batch_size)
        build_dataset(args.sample_size, str(run / 'combined.csv'),
                      str(run / 'ai_paraphrased.checkpoint.csv'))


def main():
    args = parser().parse_args()
    if not (3, 10) <= sys.version_info[:2] < (3, 13):
        raise SystemExit('Use Python 3.10, 3.11 or 3.12 (3.11 recommended).')
    if min(args.sample_size, args.epochs, args.paraphrase_batch_size,
           args.train_batch_size, args.eval_batch_size, args.gradient_accumulation_steps) < 1:
        raise SystemExit('Sample, epoch, batch and accumulation counts must be positive.')
    if args.dataloader_workers < 0:
        raise SystemExit('Data loader workers must be nonnegative.')
    os.chdir(ROOT)
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, run)
        return
    lock = run / 'pipeline.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'Run is locked: {lock}. After a crash, verify its PID and all child processes have stopped before removing this lock.')
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(str(os.getpid()))
    status_path = run / 'status.json'
    status = {'state': 'starting', 'pid': os.getpid(), 'complete': False}
    child = None
    sleep_prevented = False
    try:
        if os.name == 'nt':
            import ctypes
            sleep_prevented = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        environment = dict(os.environ)
        environment.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                           TOKENIZERS_PARALLELISM='false', TEXTORIGIN_CLASSIFIER_BATCH_SIZE=str(args.eval_batch_size))
        # Select a single visible GPU consistently for every stage.
        environment.setdefault('CUDA_VISIBLE_DEVICES', '0')
        environment.pop('HF_HUB_OFFLINE', None)
        environment.pop('TRANSFORMERS_OFFLINE', None)
        def execute(stage, command):
            nonlocal child
            status.update(state='running', stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
            print(f'\n[{stage}] Starting. Log: {run / (stage + ".log")}', flush=True)
            save_status(status_path, status)
            with (run / (stage + '.log')).open('a', encoding='utf-8') as log:
                log.write('\n' + datetime.now(timezone.utc).isoformat() + '\n' + json.dumps(command) + '\n')
                log.flush()
                child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                while child.poll() is None:
                    status['updated_at'] = datetime.now(timezone.utc).isoformat()
                    save_status(status_path, status)
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'{stage} failed (exit {child.returncode}). See {run / (stage + ".log")}. Rerun the same command after resolving the cause.')
            status.pop('child_pid', None)
            print(f'[{stage}] Completed.', flush=True)

        config = {key: value for key, value in vars(args).items() if key not in {'worker', 'run_dir'}}
        config['source_sha256'] = hashlib.sha256(''.join(
            str(path.relative_to(ROOT)) + sha(path)
            for path in [ROOT / 'run_gpu.py', ROOT / 'pyproject.toml'] +
            sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').rglob('*.py'))
        ).encode()).hexdigest()
        identity = run / 'pipeline-config.json'
        if identity.exists() and json.loads(identity.read_text()) != config:
            raise RuntimeError('Settings or source changed. Use a new --run-dir to avoid mixing incompatible datasets/checkpoints.')
        save(identity, config)
        environment_dir = ROOT / '.venv-gpu'
        python = environment_dir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(environment_dir)
        setup_marker = environment_dir / 'textorigin-setup.json'
        setup_identity = {'pyproject': sha(ROOT / 'pyproject.toml'), 'cuda_index': args.cuda_index}
        if not setup_marker.exists() or json.loads(setup_marker.read_text()) != setup_identity:
            execute('setup-pip', [str(python), '-m', 'pip', 'install', '--upgrade', 'pip'])
            execute('setup-cuda', [str(python), '-m', 'pip', 'install', '--force-reinstall',
                    'torch==2.13.0', '--index-url', 'https://download.pytorch.org/whl/' + args.cuda_index])
            execute('setup-project', [str(python), '-m', 'pip', 'install', '-e', '.[training]',
                    'httpx==0.28.1', 'psutil'])
            save(setup_marker, setup_identity)
        execute('dependency-check', [str(python), '-m', 'pip', 'check'])
        base = [str(python), str(ROOT / 'run_gpu.py'), '--run-dir', str(run),
                '--sample-size', str(args.sample_size), '--paraphrase-batch-size',
                str(args.paraphrase_batch_size), '--precision', args.precision]
        execute('preflight', base + ['--worker', 'preflight'])

        def stage(name, command, artifacts, inputs=()):
            marker = run / (name + '.completed.json')
            input_hashes = {str(path.relative_to(run)): sha(path) for path in inputs}
            if marker.exists():
                saved = json.loads(marker.read_text())
                if (saved['inputs'] == input_hashes and all(path.is_file() for path in artifacts)
                        and saved['outputs'] == {str(path.relative_to(run)): sha(path) for path in artifacts}):
                    print(f'[{name}] Previously completed; verified artifacts.', flush=True)
                    return
                raise RuntimeError(f'{name} artifacts changed. Use a new --run-dir.')
            execute(name, command)
            save(marker, {'inputs': input_hashes,
                          'outputs': {str(path.relative_to(run)): sha(path) for path in artifacts}})

        raw = run / 'combined.csv'
        clean = run / 'combined.clean.csv'
        audit = run / 'audit.json'
        model = run / 'model'
        stage('dataset', base + ['--worker', 'dataset'], [raw, run / 'combined.provenance.json'])
        stage('audit', [str(python), '-m', 'scripts.audit_dataset', '--input', str(raw),
              '--output', str(clean), '--report', str(audit)], [clean, audit], [raw])
        train = [str(python), '-m', 'scripts.train', '--data', str(clean), '--audit-report', str(audit),
                 '--output-dir', str(model), '--epochs', str(args.epochs), '--max-length', '256',
                 '--train-batch-size', str(args.train_batch_size), '--eval-batch-size', str(args.eval_batch_size),
                 '--gradient-accumulation-steps', str(args.gradient_accumulation_steps),
                 '--mixed-precision', args.precision,
                 '--dataloader-workers', str(args.dataloader_workers), '--optimizer', args.optimizer,
                 '--gradient-checkpointing' if args.gradient_checkpointing else '--no-gradient-checkpointing']
        if (model / 'training-identity.json').exists() and not (run / 'training.completed.json').exists():
            if not any(model.glob('checkpoint-*/trainer_state.json')):
                raise RuntimeError('Training stopped before its first checkpoint. Preserve/rename the model directory, then rerun; dataset and audit will be reused.')
            train += ['--resume-from-checkpoint', 'latest']
        stage('training', train, [model / name for name in
              ['model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'config.json',
               'training-identity.json', 'textorigin-manifest.json', 'evaluation.json', 'evaluation.md']], [clean, audit])
        stage('runtime-validation', [str(python), '-m', 'scripts.verify_runtime', '--model-path', str(model)],
              [model / 'runtime-verification.json', model / 'runtime-verification-offline.json'],
              [model / 'model.safetensors', model / 'textorigin-manifest.json'])
        status.update(state='completed', stage='completed', complete=True, model_path=str(model))
        print(f'\nALL STAGES PASSED. Send back the entire folder: {model}', flush=True)
    except KeyboardInterrupt:
        status.update(state='cancelled', complete=False)
        raise SystemExit(130)
    except Exception as error:
        status.update(state='failed', complete=False, error=str(error))
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()
        status.pop('child_pid', None)
        status['updated_at'] = datetime.now(timezone.utc).isoformat()
        save_status(status_path, status)
        lock.unlink(missing_ok=True)
        if sleep_prevented:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
