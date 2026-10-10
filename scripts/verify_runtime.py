"""Opt-in actual-artifact checks; never use this as evidence of model quality."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

from fastapi.testclient import TestClient
from src.api.app import create_app
from src.bootstrap import build_services
from src.config import get_settings
from src.utils.files import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true", help="Engineering verification only, never release acceptance")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    path = args.model_path.resolve()
    manifest = json.loads((path / "textorigin-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") == "smoke_only_not_release" and not args.allow_smoke:
        raise ValueError("Smoke artifacts cannot pass release validation")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"]="1"
        os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ["TEXTORIGIN_MODEL_PATH"]=str(path)
    get_settings.cache_clear()
    services = build_services(get_settings())
    # Loading verifies artifact files and weight/tokenizer compatibility.
    assert services.classifier.readiness(validate=True)["ready"]
    text = ("It is worth noting that we utilize this method. This is a second sentence. " * 75).strip()
    report = {"purpose":manifest.get("purpose"), "offline":args.offline, "checks":{}}
    with TestClient(create_app(services)) as client:
        assert client.get("/ready?validate=true").status_code==200
        analysis = client.post("/classify",json={"text":text,"include_signals":False})
        assert analysis.status_code==200, analysis.text
        result=analysis.json()
        assert result["window_count"]>1 and not result["truncated"] and result["coverage"]==1
        assert result["analyzed_tokens"]==result["total_tokens"]
        assert abs(sum(result[label] for label in ("human","ai_written","ai_paraphrased"))-1)<1e-5
        report["checks"]["http_long_classify"]={"windows":result["window_count"],"tokens":result["analyzed_tokens"]}
        rewriting=client.post("/humanize",json={"text":"It is worth noting that we utilize this method.","use_llm":False})
        assert rewriting.status_code==200,rewriting.text
        assert rewriting.json()["final_metrics"] is not None
        report["checks"]["http_rewrite"]=rewriting.json()["status"]
        compare=client.post("/compare",json={"text":"It is worth noting that we utilize this method.","use_llm":False})
        assert compare.status_code==200,compare.text
        assert compare.json()["complete"]
        report["checks"]["http_compare"]=True
        created=client.post("/jobs",json={"operation":"classify","text":text}).json()
        deadline=monotonic()+120
        while monotonic()<deadline:
            job=client.get(created["status_url"]).json()
            if job["state"] in {"completed","failed","cancelled"}:break
            sleep(.1)
        assert job["state"]=="completed",job
        assert job["result"]["total_tokens"]==result["total_tokens"]
        report["checks"]["http_job"]=True
    for operation in ("rewrite","classify","compare"):
        run=subprocess.run([sys.executable,"-m","src.cli",operation,"--text",
                            "It is worth noting that we utilize this method.","--format","json"],
                           capture_output=True,text=True,env=dict(os.environ),check=True)
        payload=json.loads(run.stdout)
        report["checks"]["cli_"+operation]=True
    report["artifact_sha256"]=file_sha256(path/"model.safetensors")
    if not args.offline:
        command = [sys.executable, "-m", "scripts.verify_runtime", "--model-path", str(path), "--offline"]
        if args.allow_smoke:
            command.append("--allow-smoke")
        subprocess.run(command, env=dict(os.environ), check=True)
        report["checks"]["offline_followup"] = True
    report["complete"]=True
    destination=path / ("runtime-verification-offline.json" if args.offline else "runtime-verification.json")
    destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
