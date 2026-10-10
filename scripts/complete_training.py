"""Run the authorized local training sequence with durable progress and fail-fast stages."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import sleep

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()
    run = ROOT / "data" / "processed"
    run.mkdir(parents=True, exist_ok=True)
    status_path = run / "training-run.json"
    lock_path = run / "training-run.lock"
    try:
        handle = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit("A training-run lock exists. Verify the recorded PID has exited before removing that specific lock.")
    os.write(handle, str(os.getpid()).encode())
    os.close(handle)
    status = {"state":"running", "pid":os.getpid(), "sample_size_per_class":args.sample_size,
              "complete":False, "stages_completed":[], "started_at":datetime.now(timezone.utc).isoformat()}
    log = (run / "full-training.log").open("a", encoding="utf-8", buffering=1)
    child = None
    def save():
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
        temporary.replace(status_path)
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    train = ["-m","scripts.train","--data","data/processed/combined.clean.csv",
             "--audit-report","data/processed/audit.json","--output-dir","models/roberta-classifier",
             "--epochs","3","--train-batch-size","1","--eval-batch-size","1",
             "--gradient-accumulation-steps","8","--mixed-precision","none"]
    if args.resume_training:
        train.extend(["--resume-from-checkpoint","latest"])
    steps = [
        ("dataset", ["-m","src.data.pipeline","--sample-size",str(args.sample_size)]),
        ("audit", ["-m","scripts.audit_dataset"]),
        ("training", train),
        ("runtime_validation", ["-m","scripts.verify_runtime","--model-path","models/roberta-classifier"]),
    ]
    try:
        save()
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["OMP_NUM_THREADS"] = "4"
        environment["MKL_NUM_THREADS"] = "4"
        for stage, command in steps:
            status["stage"] = stage
            status["stage_started_at"] = datetime.now(timezone.utc).isoformat()
            print(json.dumps({"stage":stage,"state":"starting"}), flush=True)
            child = subprocess.Popen([sys.executable,*command],cwd=ROOT,env=environment, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = child.pid
            save()
            while child.poll() is None:
                sleep(5)
                save()
            if child.returncode:
                raise RuntimeError(f"Stage {stage} failed with exit code {child.returncode}")
            status["stages_completed"].append(stage)
            save()
        status.update(state="completed",stage="completed",complete=True)
    except KeyboardInterrupt:
        if child and child.poll() is None:
            child.terminate()
            child.wait()
        status.update(state="cancelled",stage="cancelled")
        raise SystemExit(130)
    except Exception as exc:
        status.update(state="failed",error=str(exc))
        raise
    finally:
        save()
        log.close()
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
