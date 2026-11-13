"""Loads the CTG XGBoost model once and serves predictions, examples and the screening summary."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

from backend.app.core.config import REPO_ROOT, Settings
from backend.app.schemas.ctg import CTGExample, CTGPrediction, ScreeningRequest, ScreeningSummaryResponse
from ml.ctg import EXAMPLES_PATH, CTGModel
from ml.fusion import ScreeningInput, summarize

log = logging.getLogger("fetalai.ctg")
CTG_METRICS_PATH = REPO_ROOT / "outputs" / "ctg_metrics.json"
RULES = [
    "R1 CTG Pathological -> Urgent review (a pathological CTG always escalates)",
    "R2 CTG Suspect -> Review recommended",
    "R3 HC below the 3rd or above the 97th INTERGROWTH-21st centile (|z| > 1.88) -> Review recommended",
    "R4 Plane check did not confirm a trans-thalamic plane -> Review recommended (re-acquire)",
    "R5 HC measurement flagged low reliability by TTA -> Review recommended (re-scan)",
    "R6 Everything assessed is unremarkable -> Routine; inputs not provided are listed as not assessed",
]


class CTGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: CTGModel | None = None
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        path = Path(self.settings.ctg_model_path)
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            t0 = time.perf_counter()
            self.model = CTGModel(path)
            log.info("ctg model loaded", extra={"model": str(path), "seconds": round(time.perf_counter() - t0, 2)})
        except FileNotFoundError:
            self.load_error = f"CTG model not found: {path}"
            log.warning("ctg model not loaded", extra={"reason": self.load_error})
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("ctg model not loaded")

    def predict(self, features: dict[str, float]) -> CTGPrediction:
        r = self.model.predict(features)
        it = self.model.meta.get("best_iteration")
        log.info("ctg prediction", extra={"class": r.predicted_class})
        return CTGPrediction(predicted_class=r.predicted_class, probabilities=r.probabilities,
                             top_factors=r.top_factors, model_version=f"ctg-xgboost-it{it}")

    @staticmethod
    def examples() -> list[CTGExample]:
        if not EXAMPLES_PATH.is_file():
            return []
        return [CTGExample(**e) for e in json.loads(EXAMPLES_PATH.read_text())]

    def info(self) -> dict | None:
        if self.model is None:
            return None
        info = dict(self.model.meta)
        if CTG_METRICS_PATH.is_file():
            m = json.loads(CTG_METRICS_PATH.read_text())
            info["comparison_test_macro_f1"] = {k: v["test"]["macro_f1"] for k, v in m["models"].items()}
        return info

    @staticmethod
    def summary(req: ScreeningRequest) -> ScreeningSummaryResponse:
        s = summarize(ScreeningInput(**req.model_dump()))
        return ScreeningSummaryResponse(
            level=s.level, drivers=[asdict(d) for d in s.drivers], unremarkable=[asdict(d) for d in s.unremarkable],
            not_assessed=s.not_assessed, rules_version=s.rules_version, rules=RULES, caveats=s.caveats)
