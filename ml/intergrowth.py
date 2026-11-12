"""INTERGROWTH-21st fetal head-circumference standard: HC-for-gestational-age z-scores and centiles.

Source (equations copied from Table 2, p. 876):
    A. T. Papageorghiou, E. O. Ohuma, D. G. Altman, et al., "International standards for fetal growth based
    on serial ultrasound measurements: the Fetal Growth Longitudinal Study of the INTERGROWTH-21st Project",
    Lancet 2014; 384: 869-79. doi:10.1016/S0140-6736(14)61490-2

    Mean(HC) = -28.2849 + 1.69267*GA^2 - 0.397485*GA^2*log(GA)
    SD(HC)   = 1.98735 + 0.0136772*GA^3 - 0.00726264*GA^3*log(GA) + 0.000976253*GA^3*log(GA)^2

    HC in mm, GA = exact gestational age in weeks, log = natural logarithm.
    Centile c is Mean + Z_c * SD (paper text, p. 876).

Verification: tests/test_intergrowth.py checks all 189 cells (14-40 weeks x z = -3..+3) of the official
INTERGROWTH-21st table "International Fetal Growth Standards - Head Circumference (mm), z-scores"
(University of Oxford, https://intergrowth21.com/sites/default/files/2023-01/grow_fetal-zs_hc_table.pdf);
every cell is reproduced within 0.06 mm.

The standard covers 14 to 40 exact weeks; outside that range nothing is computed.
This compares one measurement with a population standard. It is not a diagnosis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

GA_MIN_WEEKS = 14.0
GA_MAX_WEEKS = 40.0
REFERENCE = "INTERGROWTH-21st fetal growth standard (Papageorghiou et al., Lancet 2014;384:869-79)"


def _check_ga(ga_weeks: float) -> None:
    if not (GA_MIN_WEEKS <= ga_weeks <= GA_MAX_WEEKS):
        raise ValueError(f"Gestational age must be between {GA_MIN_WEEKS:g} and {GA_MAX_WEEKS:g} weeks")


def hc_mean_mm(ga_weeks: float) -> float:
    _check_ga(ga_weeks)
    ga = ga_weeks
    return -28.2849 + 1.69267 * ga**2 - 0.397485 * ga**2 * math.log(ga)


def hc_sd_mm(ga_weeks: float) -> float:
    _check_ga(ga_weeks)
    ga, log_ga = ga_weeks, math.log(ga_weeks)
    return 1.98735 + 0.0136772 * ga**3 - 0.00726264 * ga**3 * log_ga + 0.000976253 * ga**3 * log_ga**2


def hc_at_z(ga_weeks: float, z: float) -> float:
    return hc_mean_mm(ga_weeks) + z * hc_sd_mm(ga_weeks)


def z_to_centile(z: float) -> float:
    """Standard normal CDF x 100."""
    return 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))


# z-values for the centiles named in the paper (p. 876)
Z_P3, Z_P10, Z_P50, Z_P90, Z_P97 = -1.88, -1.28, 0.0, 1.28, 1.88


@dataclass(frozen=True)
class HCAssessment:
    ga_weeks: float
    hc_mm: float
    z_score: float
    centile: float
    median_mm: float
    p3_mm: float
    p97_mm: float
    within_p3_p97: bool


def assess_hc(hc_mm: float, ga_weeks: float) -> HCAssessment:
    """Compare a measured HC with the INTERGROWTH-21st standard at the given gestational age."""
    if hc_mm <= 0:
        raise ValueError("HC must be positive")
    mean, sd = hc_mean_mm(ga_weeks), hc_sd_mm(ga_weeks)
    z = (hc_mm - mean) / sd
    return HCAssessment(
        ga_weeks=ga_weeks, hc_mm=hc_mm, z_score=z, centile=z_to_centile(z), median_mm=mean,
        p3_mm=mean + Z_P3 * sd, p97_mm=mean + Z_P97 * sd, within_p3_p97=Z_P3 <= z <= Z_P97,
    )


def centile_curves(step_weeks: float = 0.5) -> list[dict]:
    """3rd/10th/50th/90th/97th centile curves from 14 to 40 weeks, for charts."""
    n = int(round((GA_MAX_WEEKS - GA_MIN_WEEKS) / step_weeks))
    points = []
    for i in range(n + 1):
        ga = GA_MIN_WEEKS + i * step_weeks
        points.append({"ga_weeks": ga, **{name: round(hc_at_z(ga, z), 2) for name, z in
                       (("p3", Z_P3), ("p10", Z_P10), ("p50", Z_P50), ("p90", Z_P90), ("p97", Z_P97))}})
    return points
