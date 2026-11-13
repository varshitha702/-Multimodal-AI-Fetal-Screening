"""Decision-level fusion: one screening summary from the ultrasound and CTG branches (Feature C).

Rule-based and deliberately simple, so every output can be traced to a rule. The ultrasound models
(HC18, FETAL_PLANES_DB) and the CTG model (UCI Cardiotocography) were trained on DIFFERENT patients, so there is
no paired data to learn or calibrate the fused output: these rules are illustrative, and learned
(feature-level) fusion with calibration is Phase-2 work that needs paired clinical data.

Rules (most severe wins):
  R1  CTG = Pathological                               -> Urgent review   (a pathological CTG always escalates)
  R2  CTG = Suspect                                    -> Review recommended
  R3  HC below the 3rd or above the 97th centile       -> Review recommended (INTERGROWTH-21st, |z| > 1.88)
  R4  Plane check did not confirm a trans-thalamic plane (warn, incl. overridden or withheld)
                                                       -> Review recommended (re-acquire the image)
  R5  HC measurement flagged low reliability (TTA)     -> Review recommended (re-scan)
  R6  everything that was assessed is unremarkable     -> Routine
No input at all -> "Insufficient data". Inputs that were not provided are listed as "not assessed".
"""
from __future__ import annotations

from dataclasses import dataclass, field

LEVELS = ["Routine", "Review recommended", "Urgent review"]
Z_P3_P97 = 1.88       # INTERGROWTH-21st: 3rd / 97th centile = mean -/+ 1.88 SD (Papageorghiou et al., 2014)
RULES_VERSION = "fusion-rules-1"
CAVEATS = [
    "The ultrasound and CTG models were trained on different patients (HC18 / FETAL_PLANES_DB vs UCI CTG); "
    "there is no paired data, so these fusion rules are illustrative and the combined level is not calibrated. "
    "Learned fusion and its calibration are Phase-2 work requiring paired clinical data.",
    "Academic research prototype, not a medical device; not for clinical decisions.",
]


@dataclass
class ScreeningInput:
    plane_status: str | None = None              # "pass" | "warn" | "unavailable"
    plane_overridden: bool = False
    measurement_withheld: bool = False
    hc_mm: float | None = None
    hc_z_score: float | None = None              # needs gestational age + pixel spacing
    hc_centile: float | None = None
    low_reliability: bool | None = None          # TTA flag; None = TTA not run
    ctg_class: str | None = None                 # "Normal" | "Suspect" | "Pathological"
    ctg_probability: float | None = None         # probability of ctg_class


@dataclass
class Driver:
    input: str
    finding: str
    level: str          # level this input alone implies ("Routine" for unremarkable findings)
    rule: str


@dataclass
class ScreeningSummary:
    level: str
    drivers: list[Driver]                         # inputs that raised the level (empty for Routine)
    unremarkable: list[Driver] = field(default_factory=list)
    not_assessed: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=lambda: list(CAVEATS))
    rules_version: str = RULES_VERSION


def summarize(s: ScreeningInput) -> ScreeningSummary:
    findings: list[Driver] = []
    missing: list[str] = []

    if s.ctg_class is None:
        missing.append("CTG (no CTG features submitted)")
    else:
        p = f" (p = {s.ctg_probability:.2f})" if s.ctg_probability is not None else ""
        if s.ctg_class == "Pathological":
            findings.append(Driver("CTG", f"Pathological{p}", "Urgent review", "R1"))
        elif s.ctg_class == "Suspect":
            findings.append(Driver("CTG", f"Suspect{p}", "Review recommended", "R2"))
        elif s.ctg_class == "Normal":
            findings.append(Driver("CTG", f"Normal{p}", "Routine", "R6"))
        else:
            raise ValueError(f"Unknown CTG class: {s.ctg_class}")

    if s.plane_status in (None, "unavailable"):
        missing.append("Standard-plane check")
    elif s.plane_status == "warn":
        how = " (measurement withheld)" if s.measurement_withheld else (" (measured with override)"
                                                                       if s.plane_overridden else "")
        findings.append(Driver("Plane check", f"Trans-thalamic plane not confirmed{how}; re-acquire",
                               "Review recommended", "R4"))
    elif s.plane_status == "pass":
        findings.append(Driver("Plane check", "Trans-thalamic plane confirmed", "Routine", "R6"))
    else:
        raise ValueError(f"Unknown plane status: {s.plane_status}")

    if s.hc_z_score is None:
        missing.append("HC centile (needs HC in mm and gestational age)" if not s.measurement_withheld
                       else "HC centile (measurement withheld)")
    else:
        c = f"centile {s.hc_centile:.1f}, " if s.hc_centile is not None else ""
        if abs(s.hc_z_score) > Z_P3_P97:
            side = "below the 3rd" if s.hc_z_score < 0 else "above the 97th"
            findings.append(Driver("HC for gestational age", f"HC {side} centile ({c}z = {s.hc_z_score:+.2f})",
                                   "Review recommended", "R3"))
        else:
            findings.append(Driver("HC for gestational age", f"HC within 3rd-97th centile ({c}z = "
                                   f"{s.hc_z_score:+.2f})", "Routine", "R6"))

    if s.low_reliability is None:
        missing.append("Measurement uncertainty (TTA not run)")
    elif s.low_reliability:
        findings.append(Driver("Measurement reliability", "Low reliability across augmented passes; re-scan",
                               "Review recommended", "R5"))
    else:
        findings.append(Driver("Measurement reliability", "Consistent across augmented passes", "Routine", "R6"))

    if not findings:
        return ScreeningSummary(level="Insufficient data", drivers=[], not_assessed=missing)
    level = max((f.level for f in findings), key=LEVELS.index)
    drivers = [f for f in findings if f.level != "Routine"]
    drivers.sort(key=lambda f: -LEVELS.index(f.level))
    return ScreeningSummary(level=level, drivers=drivers, unremarkable=[f for f in findings if f.level == "Routine"],
                            not_assessed=missing)
