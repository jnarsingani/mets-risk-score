# ═══════════════════════════════════════════════════════════════════════════
# msss_calculator.py
# ───────────────────────────────────────────────────────────────────────────
# Metabolic Syndrome Severity Score (MSSS) Calculator
#
# Implements the Gurka-DeBoer MSSS formula for adults and adolescents,
# stratified by sex and race/ethnicity.
#
# Formula source : github.com/MetSCalc/metscalc (Apache 2.0 License)
#                  University of Florida
# Reference      : Gurka et al. (2014), Metabolism
#
# Race/Ethnicity Coverage:
#   ✅ Non-Hispanic White
#   ✅ Non-Hispanic Black
#   ✅ Hispanic (Mexican American + Other Hispanic)
#   ❌ Non-Hispanic Asian    → returns NaN (no validated equation exists)
#   ❌ Other / Multiracial   → returns NaN (no validated equation exists)
#
# Age Groups:
#   Adults      (age >= 20) → waist circumference version (mets_z_wc)
#   Adolescents (age <  20) → BMI z-score version (mets_z_bmi)
#
# NHANES Variable Inputs:
#   RIAGENDR     : sex (1 = Male, 2 = Female)
#   RIDRETH3     : race/ethnicity code
#   RIDAGEYR     : age in years
#   BMXWAIST     : waist circumference in cm (adults)
#   BMI_ZSCORE   : CDC BMI-for-age z-score (adolescents, precomputed)
#   LBDHDD       : HDL cholesterol in mg/dL
#   SYSTOLIC_AVG : average systolic BP in mmHg (mean of BPXOSY1, BPXOSY2)
#   LBXTR        : triglycerides in mg/dL (log-transformed internally)
#   LBXGLU       : fasting glucose in mg/dL
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


# ── Race/Ethnicity Mapping ──────────────────────────────────────────────────
# Maps NHANES RIDRETH3 codes to MetSCalc race labels.
# Codes 6 (Asian) and 7 (Other/Multiracial) map to None → excluded.

RACE_MAP = {
    1.0: 'HISPANIC',   # Mexican American
    2.0: 'HISPANIC',   # Other Hispanic
    3.0: 'WHITE',      # Non-Hispanic White
    4.0: 'BLACK',      # Non-Hispanic Black
    6.0: None,         # Non-Hispanic Asian — no validated equation
    7.0: None,         # Other / Multiracial — no validated equation
}


# ── Adult Coefficients (age >= 20, waist circumference version) ─────────────
# Format: intercept, waist, hdl, sbp, log_trig, glucose
# Coefficients copied verbatim from msscalc.js (mets_z_wc branch)

ADULT_COEFFICIENTS = {
    ('FEMALE', 'BLACK'):    (-7.1913, 0.0304, -0.0095, 0.0054, 0.4455, 0.0225),
    ('FEMALE', 'HISPANIC'): (-7.7641, 0.0162, -0.0157, 0.0084, 0.8872, 0.0206),
    ('FEMALE', 'WHITE'):    (-7.2591, 0.0254, -0.0120, 0.0075, 0.5800, 0.0203),
    ('MALE',   'BLACK'):    (-6.3767, 0.0232, -0.0175, 0.0040, 0.5400, 0.0203),
    ('MALE',   'HISPANIC'): (-5.5541, 0.0135, -0.0278, 0.0054, 0.8340, 0.0105),
    ('MALE',   'WHITE'):    (-5.4559, 0.0125, -0.0251, 0.0047, 0.8244, 0.0106),
}


# ── Adolescent Coefficients (age < 20, BMI z-score version) ─────────────────
# Format: intercept, bmi_z, hdl, sbp, log_trig, glucose
# Coefficients copied verbatim from msscalc.js (mets_z_bmi branch)

ADOLESCENT_COEFFICIENTS = {
    ('FEMALE', 'BLACK'):    (-3.7145, 0.5136, -0.0190, 0.0131, 0.4442, 0.0108),
    ('FEMALE', 'HISPANIC'): (-4.7637, 0.3520, -0.0263, 0.0152, 0.6910, 0.0133),
    ('FEMALE', 'WHITE'):    (-4.3757, 0.4849, -0.0176, 0.0257, 0.3172, 0.0083),
    ('MALE',   'BLACK'):    (-4.7544, 0.2401, -0.0284, 0.0134, 0.6773, 0.0179),
    ('MALE',   'HISPANIC'): (-3.2971, 0.2930, -0.0315, 0.0109, 0.6137, 0.0095),
    ('MALE',   'WHITE'):    (-4.9310, 0.2804, -0.0257, 0.0189, 0.6240, 0.0140),
}


def compute_msss(row):
    """
    Compute the Metabolic Syndrome Severity Score (MSSS) for a single
    participant row from a pandas DataFrame.

    Parameters
    ----------
    row : pd.Series
        One row from the NHANES cohort DataFrame. Must contain:
        RIAGENDR, RIDRETH3, RIDAGEYR, LBDHDD, SYSTOLIC_AVG,
        LBXTR, LBXGLU, BMXWAIST (adults), BMI_ZSCORE (adolescents).

    Returns
    -------
    float
        Continuous MSSS value, or np.nan if:
        - Race has no validated equation (Asian / Other-Multiracial)
        - Any required input is missing
    """

    # ── Extract grouping variables ──────────────────────────────────────────
    sex_code  = row['RIAGENDR']    # 1.0 = Male, 2.0 = Female
    race_code = row['RIDRETH3']
    age       = row['RIDAGEYR']

    # ── Map to string labels used in coefficient dictionaries ───────────────
    sex  = 'FEMALE' if sex_code == 2.0 else 'MALE'
    race = RACE_MAP.get(race_code, None)

    # Exclude races with no validated equation
    if race is None:
        return np.nan

    # ── Extract shared inputs (required for all formulas) ───────────────────
    hdl  = row['LBDHDD']
    sbp  = row['SYSTOLIC_AVG']
    trig = row['LBXTR']
    glu  = row['LBXGLU']

    # Return NaN if any shared input is missing
    if any(pd.isna(v) for v in [hdl, sbp, trig, glu]):
        return np.nan

    # Log-transform triglycerides (required by the formula)
    log_trig = np.log(trig)

    # ── ADOLESCENT FORMULA (age < 20) ──────────────────────────────────────
    if age < 20:
        bmi_z = row.get('BMI_ZSCORE', np.nan)
        if pd.isna(bmi_z):
            return np.nan

        coeffs = ADOLESCENT_COEFFICIENTS.get((sex, race), None)
        if coeffs is None:
            return np.nan

        intercept, c_bmi_z, c_hdl, c_sbp, c_log_trig, c_glu = coeffs
        return (intercept
                + c_bmi_z    * bmi_z
                + c_hdl      * hdl
                + c_sbp      * sbp
                + c_log_trig * log_trig
                + c_glu      * glu)

    # ── ADULT FORMULA (age >= 20) ───────────────────────────────────────────
    waist = row['BMXWAIST']
    if pd.isna(waist):
        return np.nan

    coeffs = ADULT_COEFFICIENTS.get((sex, race), None)
    if coeffs is None:
        return np.nan

    intercept, c_waist, c_hdl, c_sbp, c_log_trig, c_glu = coeffs
    return (intercept
            + c_waist    * waist
            + c_hdl      * hdl
            + c_sbp      * sbp
            + c_log_trig * log_trig
            + c_glu      * glu)
