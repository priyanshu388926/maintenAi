"""API tests. Prediction tests need trained bundles (python -m ml.inference.train); they are skipped otherwise."""
import io
import os
import tempfile

import pytest

os.environ["MAINTENAI_DATA_DIR"] = tempfile.mkdtemp()
os.environ["MONGODB_URI"] = ""
from fastapi.testclient import TestClient  # noqa: E402

from app.main import _PRED, app  # noqa: E402
from app.store import FileStore  # noqa: E402

client = TestClient(app)
needs_model = pytest.mark.skipif("engineered" not in _PRED, reason="models not trained")
NORMAL = dict(type="M", air_temperature=298.1, process_temperature=308.6, rotational_speed=1551, torque=42.8, tool_wear=0)
OVERSTRAIN = dict(type="L", air_temperature=300.0, process_temperature=310.0, rotational_speed=1500, torque=50, tool_wear=230)


def test_health_and_dataset():
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["dataset"] and h["store"] == "file"
    s = client.get("/api/dataset/summary").json()
    assert s["rows"] == 10000 and s["failures"] == 339
    assert sum(t["rows"] for t in s["types"]) == 10000
    hist = client.get("/api/dataset/histogram", params={"feature": "Torque [Nm]", "bins": 20}).json()
    assert sum(hist["normal"]) + sum(hist["failure"]) == 10000
    assert client.get("/api/dataset/histogram", params={"feature": "TWF"}).status_code == 400
    rows = client.get("/api/dataset/rows", params={"failures_only": True, "limit": 5}).json()
    assert rows["total"] == 339 and len(rows["rows"]) == 5


def test_figure_name_is_validated():
    assert client.get("/api/figures/..%2Fsecret.png").status_code in (400, 404)


def test_input_validation():
    bad = {**NORMAL, "torque": 999}
    assert client.post("/api/predict", json=bad).status_code == 422
    assert client.post("/api/predict", json={**NORMAL, "type": "X"}).status_code == 422


@needs_model
def test_predict_normal_and_failure_and_history():
    client.delete("/api/history")
    a = client.post("/api/predict", json={**NORMAL, "machine_id": "A1"}).json()
    assert a["risk"] == "Low" and a["probability"] < 0.05 and not a["predicted_failure"]
    b = client.post("/api/predict", json=OVERSTRAIN).json()
    assert b["probability"] > 0.5 and "OSF" in [r["mode"] for r in b["rules"] if r["triggered"]]
    assert len(b["models"]) == 7 and len(b["shap"]["contributions"]) == 9 and b["cart_path"]["steps"]
    hist = client.get("/api/history").json()
    assert len(hist) == 2 and hist[1]["machine_id"] == "A1"
    assert client.delete(f"/api/history/{hist[0]['id']}").json() == {"deleted": 1}
    assert client.get("/api/history/stats").json()["total"] == 1


@needs_model
def test_batch_with_labels():
    csv = ("Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min],Machine failure\n"
           "M1,M,298.1,308.6,1551,42.8,0,0\nL2,L,300,310,1500,50,230,1\n")
    r = client.post("/api/batch", files={"file": ("b.csv", io.BytesIO(csv.encode()), "text/csv")},
                    data={"feature_set": "engineered"}).json()
    assert r["summary"]["rows"] == 2 and r["rows"][0]["id"] == "M1"
    assert r["summary"]["evaluation"]["tp"] + r["summary"]["evaluation"]["fn"] == 1
    bad = client.post("/api/batch", files={"file": ("b.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")})
    assert bad.status_code == 400


@needs_model
def test_models_and_global_explanations():
    m = client.get("/api/models").json()
    assert abs(sum(m["bma_weights"].values()) - 1) < 1e-6 and "IEMAF" in m["thresholds"]
    g = client.get("/api/explain/global", params={"feature_set": "raw"}).json()
    assert "cart_rules" in g and len(g["gam_shapes"]) == 5


def test_file_store_roundtrip(tmp_path):
    s = FileStore(tmp_path / "h.jsonl")
    d = s.add({"risk": "Low"}); s.add({"risk": "High"})
    assert s.count() == 2 and s.list(risk="High")[0]["risk"] == "High"
    assert s.get(d["id"])["risk"] == "Low" and s.delete(d["id"]) and s.count() == 1
    assert s.clear() == 1
