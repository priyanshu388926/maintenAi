"""Background jobs: retraining the model bundles and re-running all experiments. One job runs at a time."""
from __future__ import annotations

import subprocess
import sys
import threading
import traceback
import uuid
from collections import deque
from datetime import datetime, timezone

from app import config


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Job:
    def __init__(self, kind: str, params: dict):
        self.id, self.kind, self.params = uuid.uuid4().hex[:12], kind, params
        self.state, self.started_at, self.finished_at, self.error = "queued", None, None, None
        self.log: deque[str] = deque(maxlen=400)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "params": self.params, "state": self.state,
                "started_at": self.started_at, "finished_at": self.finished_at, "error": self.error, "log": list(self.log)}


class JobManager:
    def __init__(self, on_trained):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._on_trained = on_trained

    def running(self):
        return next((j for j in self.jobs.values() if j.state in ("queued", "running")), None)

    def start(self, kind: str, params: dict) -> Job:
        with self._lock:
            if self.running():
                raise RuntimeError("Another job is already running; wait for it to finish.")
            job = Job(kind, params)
            self.jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job):
        job.state, job.started_at = "running", now()
        try:
            if job.kind == "train":
                from ml.inference.train import save_bundle, train_bundle
                for fs in job.params["feature_sets"]:
                    b = train_bundle(fs, progress=job.log.append)
                    save_bundle(b)
                    m = b["metrics_test"]["IEMAF"]
                    job.log.append(f"[{fs}] saved; test PR-AUC {m['pr_auc']:.3f}, recall {m['recall']:.3f} "
                                   f"({b['train_seconds']} s)")
                self._on_trained()
            elif job.kind == "experiments":
                proc = subprocess.Popen([sys.executable, "-u", "-m", "ml.experiments.run_all"], cwd=config.ROOT,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    job.log.append(line.rstrip())
                if proc.wait() != 0:
                    raise RuntimeError(f"run_all exited with code {proc.returncode}")
            else:
                raise ValueError(job.kind)
            job.state = "succeeded"
        except Exception as e:
            job.state, job.error = "failed", f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc(limit=3))
        finally:
            job.finished_at = now()
