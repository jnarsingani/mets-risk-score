# ============================================================
# MetS Risk Score - Clinical Prediction Tool
# src/app.py
#
# Model: ElasticNet (full multimodal), R2=0.736
# Data:  NHANES 2017-March 2020, n=1,203, ages 18-45
# Author: Jay Narsingani
# ============================================================

from collections import abc
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# ------------------------------------------------------------
# Page config - must be first Streamlit call
# ------------------------------------------------------------
st.set_page_config(
    page_title  = "MetS Risk Score",
    page_icon   = "assets/favicon.ico",
    layout      = "centered",
    initial_sidebar_state = "collapsed",
)

# ------------------------------------------------------------
# Custom CSS - responsive layout, input styling, no dark mode
# ------------------------------------------------------------
st.markdown("""
<style>
/* Force white background regardless of system theme */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
}

/* Hide BMI placeholder radio */
[data-testid="stRadio"][aria-disabled="true"] {
    visibility: hidden;
    height: 28px;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/* Max width container */
.block-container {
    max-width: 860px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
}

/* Section headers */
.sec-head {
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 6px;
    margin-bottom: 12px;
    margin-top: 20px;
}

/* Info bar */
.info-bar {
    background: #F8FAFC;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 12px;
    color: #475569;
    line-height: 1.5;
    margin-top: 12px;
}

/* Warning bar */
.warn-bar {
    background: #FEF3C7;
    border: 1px solid #FCD34D;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 12px;
    color: #92400E;
    line-height: 1.5;
    margin-top: 8px;
}

/* Error bar */
.error-bar {
    background: #FEE2E2;
    border: 1px solid #FCA5A5;
    border-radius: 8px;
    padding: 9px 14px;
    font-size: 12px;
    color: #991B1B;
    line-height: 1.5;
    margin-top: 8px;
}

/* Score display */
.score-hero {
    text-align: center;
    padding: 2rem 0 1.5rem;
}
.score-num {
    font-size: 64px;
    font-weight: 600;
    color: #0B6E8C;
    line-height: 1;
}
.score-lbl {
    font-size: 13px;
    color: #94A3B8;
    margin-top: 6px;
}
.score-stmt {
    font-size: 15px;
    color: #1E293B;
    margin-top: 12px;
    line-height: 1.6;
}
.risk-tag {
    display: inline-block;
    padding: 5px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 10px;
}
.risk-low      { background:#D1FAE5; color:#065F46; }
.risk-moderate { background:#FEF3C7; color:#92400E; }
.risk-high     { background:#FEE2E2; color:#991B1B; }
.risk-veryhigh { background:#FEE2E2; color:#7F1D1D; font-weight:700; }

/* Clinical note */
.note-box {
    background: #F0F9FF;
    border-left: 3px solid #0B6E8C;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.25rem;
    font-size: 13px;
    color: #1E293B;
    line-height: 1.7;
}

/* Recommendation item */
.reco {
    padding: 12px 0;
    border-bottom: 1px solid #E2E8F0;
}
.reco-title {
    font-size: 13px;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 4px;
}
.reco-body {
    font-size: 12px;
    color: #475569;
    line-height: 1.6;
    margin-bottom: 4px;
}
.reco-cite {
    font-size: 11px;
    color: #94A3B8;
}

/* Disclaimer */
.disc {
    font-size: 11px;
    color: #94A3B8;
    line-height: 1.6;
    padding-top: 1rem;
    border-top: 1px solid #E2E8F0;
    margin-top: 1.5rem;
}

/* Stepper */
.stepper {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    color: #94A3B8;
    margin-bottom: 1.5rem;
}
.step-active { color: #0B6E8C; font-weight: 600; }
.step-done   { color: #0B6E8C; }
.step-sep    { color: #E2E8F0; padding: 0 4px; }

/* Responsive columns */
@media (max-width: 640px) {
    [data-testid="column"] { width: 100% !important; flex: 100% !important; }
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load model files - cached so they only load once
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base, 'outputs', 'models')

    with open(os.path.join(model_dir, 'final_elasticnet.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'final_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, 'population_means.json')) as f:
        pop_means = json.load(f)
    with open(os.path.join(model_dir, 'feature_columns.json')) as f:
        feature_cols = json.load(f)

    return model, scaler, pop_means, feature_cols

model, scaler, pop_means, feature_cols = load_models()

# ------------------------------------------------------------
# Session state - track current page and inputs
# ------------------------------------------------------------
if 'page' not in st.session_state:
    st.session_state.page = 0
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def go_to(page):
    st.session_state.page = page
    st.rerun()

def risk_category(msss):
    if msss < -0.5:
        return "Low Risk", "risk-low"
    elif msss < 0.5:
        return "Moderate Risk", "risk-moderate"
    elif msss < 1.5:
        return "Moderate-High Risk", "risk-high"
    else:
        return "Very High Risk", "risk-veryhigh"

def percentile_from_msss(msss):
    # Empirical percentile from training cohort MSSS distribution
    # Based on normal approximation (MSSS is z-score by construction)
    from scipy.stats import norm
    return int(norm.cdf(msss) * 100)

def render_stepper(current):
    steps = ["Patient info", "Risk overview", "Drivers", "Summary"]
    parts = []
    for i, s in enumerate(steps):
        if i < current:
            parts.append(f'<span class="step-done">{i+1}. {s}</span>')
        elif i == current:
            parts.append(f'<span class="step-active">{i+1}. {s}</span>')
        else:
            parts.append(f'<span>{i+1}. {s}</span>')
        if i < len(steps) - 1:
            parts.append('<span class="step-sep">/</span>')
    st.markdown(
        f'<div class="stepper">{"".join(parts)}</div>',
        unsafe_allow_html=True
    )

def build_feature_vector(inputs):
    """
    Build the full 230-column feature vector from user inputs.
    Uses exact column order from training to ensure scaler alignment.
    Missing values filled with population means.
    """
    # Load exact column order from training
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    col_path = os.path.join(base, 'outputs', 'models',
                            'feature_columns.json')
    with open(col_path) as f:
        feature_cols = json.load(f)

    # Start with population means for all columns
    x = {col: pop_means.get(col, 0.0) for col in feature_cols}

    # Map user inputs to column names
    field_map = {
        'age'        : 'RIDAGEYR',
        'bmi'        : 'BMXBMI',
        'weight_kg'  : 'BMXWT',
        'alt'        : 'LBXSATSI',
        'ggt'        : 'LBXSGTSI',
        'alk_phos'   : 'LBXSAPSI',
        'albumin'    : 'LBXSAL',
        'cholesterol': 'LBXSCH',
        'chloride'   : 'LBXSCLSI',
        'wbc'        : 'LBXWBCSI',
        'monocyte'   : 'LBXMOPCT',
        'sedentary'  : 'PAD680',
        'met_minutes': 'total_MET_minutes',
    }

    for field, col in field_map.items():
        if field in inputs and inputs[field] is not None:
            val = inputs[field]
            if field in ['alt', 'ggt', 'alk_phos']:
                val = np.log1p(val)
            elif field == 'sedentary':
                val = np.log1p(min(val, 1440))
            elif field == 'met_minutes':
                val = np.log1p(val)
            if col in x:
                x[col] = val

    # Sex encoding
    if inputs.get('sex'):
        x['RIAGENDR'] = 1.0 if inputs['sex'] == 'Male' else 2.0

    # Race encoding
    race_map = {
        'Non-Hispanic White': 3.0,
        'Non-Hispanic Black': 4.0,
        'Hispanic'          : 1.0,
    }
    if inputs.get('race'):
        x['RIDRETH3'] = race_map.get(inputs['race'], 3.0)

    # Build vector in exact training column order
    vector = np.array([x[col] for col in feature_cols])
    return vector.reshape(1, -1), feature_cols

def run_prediction(inputs):
    import shap
    vector, feat_cols = build_feature_vector(inputs)
    vector_scaled = scaler.transform(vector)
    msss = float(model.predict(vector_scaled)[0])

    try:
        background = np.zeros((1, vector_scaled.shape[1]))
        explainer  = shap.LinearExplainer(model, background)
        shap_vals  = explainer.shap_values(vector_scaled)[0]
    except Exception as e:
        st.error(f"SHAP error: {e}")
        shap_vals = np.zeros(vector_scaled.shape[1])

    return msss, shap_vals, feat_cols

# ============================================================
# PAGE 1 - Patient Information
# ============================================================
def page_patient_info():
    render_stepper(0)

    st.markdown("## Patient information")
    st.markdown(
        "Enter available values. Fields marked with "
        "**\*** are required. Recommended fields improve prediction accuracy."
    )

    inputs = {}
    missing_required    = []
    missing_recommended = []

    # ── Demographics ────────────────────────────────────────
    st.markdown('<div class="sec-head">Demographics</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input(
            "Age (years) *", min_value=18, max_value=45,
            value=None, placeholder="e.g. 34",
            help="Required. Ages 18-45 only."
        )
        inputs['age'] = age
        if age is None:
            missing_required.append("Age")

    with col2:
        sex = st.selectbox(
            "Sex *",
            options=["", "Male", "Female"],
            index=0,
            help="Required."
        )
        inputs['sex'] = sex if sex else None
        if not sex:
            missing_required.append("Sex")

    with col3:
        race = st.selectbox(
            "Race / ethnicity *",
            options=["", "Non-Hispanic White",
                     "Non-Hispanic Black", "Hispanic"],
            index=0,
            help="Required. Model validated for these groups only."
        )
        inputs['race'] = race if race else None
        if not race:
            missing_required.append("Race / ethnicity")

    # ── Body measures ────────────────────────────────────────
    st.markdown('<div class="sec-head">Body measures</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        height_unit = st.radio(
            "Height unit", ["cm", "in", "ft"],
            horizontal=True, label_visibility="collapsed"
        )
        if height_unit == "ft":
            c1, c2 = st.columns(2)
            with c1:
                ht_ft = st.number_input(
                    "Height (ft) *", min_value=0,
                    max_value=8, value=None, placeholder="ft"
                )
            with c2:
                ht_in = st.number_input(
                    "Inches", min_value=0,
                    max_value=11, value=None, placeholder="in"
                )
            if ht_ft is not None:
                height_cm = ((ht_ft or 0) * 12 +
                             (ht_in or 0)) * 2.54
            else:
                height_cm = None
        elif height_unit == "in":
            ht = st.number_input(
                "Height (in) *", min_value=0.0,
                value=None, placeholder="e.g. 69"
            )
            height_cm = ht * 2.54 if ht else None
        else:
            height_cm = st.number_input(
                "Height (cm) *", min_value=0.0,
                value=None, placeholder="e.g. 175"
            )

        if height_cm is None:
            missing_required.append("Height")

    with col2:
        weight_unit = st.radio(
            "Weight unit", ["kg", "lbs"],
            horizontal=True, label_visibility="collapsed"
        )
        if weight_unit == "lbs":
            wt = st.number_input(
                "Weight (lbs) *", min_value=0.0,
                value=None, placeholder="e.g. 194"
            )
            weight_kg = wt * 0.453592 if wt else None
        else:
            weight_kg = st.number_input(
                "Weight (kg) *", min_value=0.0,
                value=None, placeholder="e.g. 88"
            )

        if weight_kg is None:
            missing_required.append("Weight")

    with col3:
        st.markdown(
            '<div style="height:42px;"></div>',
            unsafe_allow_html=True
        )
        st.markdown("**BMI (auto-calculated)**")
        if height_cm and weight_kg and height_cm > 0:
            bmi = weight_kg / ((height_cm / 100) ** 2)
            st.markdown(
                f'<div style="background:#F0F9FF;border:1px solid '
                f'#BAE6FD;border-radius:8px;padding:9px 12px;'
                f'font-size:22px;font-weight:600;color:#0B6E8C;">'
                f'{bmi:.1f} <span style="font-size:13px;'
                f'font-weight:400;">kg/m²</span></div>',
                unsafe_allow_html=True
            )
            inputs['bmi']       = bmi
            inputs['weight_kg'] = weight_kg
        else:
            st.markdown(
                '<div style="background:#F8FAFC;border:1px solid '
                '#E2E8F0;border-radius:8px;padding:9px 12px;'
                'font-size:13px;color:#94A3B8;">--</div>',
                unsafe_allow_html=True
            )
            inputs['bmi']       = None
            inputs['weight_kg'] = None

    # ── Comprehensive metabolic panel ────────────────────────
    st.markdown(
        '<div class="sec-head">Comprehensive metabolic panel</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        alt = st.number_input(
            "ALT (U/L) - recommended",
            min_value=0.0, value=None, placeholder="e.g. 48",
            help="Alanine aminotransferase. High SHAP importance."
        )
        inputs['alt'] = alt
        if alt is None:
            missing_recommended.append("ALT")

    with col2:
        ggt = st.number_input(
            "GGT (U/L) - recommended",
            min_value=0.0, value=None, placeholder="e.g. 35",
            help="Gamma-glutamyl transferase. High SHAP importance."
        )
        inputs['ggt'] = ggt
        if ggt is None:
            missing_recommended.append("GGT")

    with col3:
        alk_phos = st.number_input(
            "Alk.Phos.(U/L)-recommended",
            min_value=0.0, value=None, placeholder="e.g. 75",
            help="High SHAP importance."
        )
        inputs['alk_phos'] = alk_phos
        if alk_phos is None:
            missing_recommended.append("Alkaline phosphatase")

    col1, col2, col3 = st.columns(3)
    with col1:
        cholesterol = st.number_input(
            "Total.Chol(mg/dL)-recommended",
            min_value=0.0, value=None, placeholder="e.g. 185",
            help="Moderate SHAP importance."
        )
        inputs['cholesterol'] = cholesterol
        if cholesterol is None:
            missing_recommended.append("Total cholesterol")

    with col2:
        albumin = st.number_input(
            "Albumin (g/dL)",
            min_value=0.0, value=None, placeholder="e.g. 4.2"
        )
        inputs['albumin'] = albumin

    with col3:
        chloride = st.number_input(
            "Chloride (mmol/L)",
            min_value=0.0, value=None, placeholder="e.g. 102"
        )
        inputs['chloride'] = chloride

    # ── Complete blood count ─────────────────────────────────
    st.markdown(
        '<div class="sec-head">Complete blood count</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        wbc = st.number_input(
            "WBC (x10³/µL)",
            min_value=0.0, value=None, placeholder="e.g. 7.2"
        )
        inputs['wbc'] = wbc

    with col2:
        monocyte = st.number_input(
            "Monocyte (%)",
            min_value=0.0, max_value=100.0,
            value=None, placeholder="e.g. 8.1"
        )
        inputs['monocyte'] = monocyte

    # ── Lifestyle ────────────────────────────────────────────
    st.markdown(
        '<div class="sec-head">Lifestyle (patient-reported)</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)
    with col1:
        sedentary = st.number_input(
            "Sedentary time (min / day)",
            min_value=0, max_value=1440,
            value=None, placeholder="e.g. 420",
            help="Average daily sitting / screen time."
        )
        inputs['sedentary'] = sedentary

    with col2:
        met = st.number_input(
            "Physical activity (MET-min / week)",
            min_value=0.0, value=None, placeholder="e.g. 3240",
            help="Moderate activity: ~4 MET/min. Vigorous: ~8 MET/min."
        )
        inputs['met_minutes'] = met

    # ── Status messages ──────────────────────────────────────
    st.markdown("")

    if missing_required:
        st.markdown(
            f'<div class="error-bar">Required fields missing: '
            f'{", ".join(missing_required)}. '
            f'These fields are needed to generate a prediction.</div>',
            unsafe_allow_html=True
        )
    elif missing_recommended:
        st.markdown(
            f'<div class="warn-bar">Recommended fields missing: '
            f'{", ".join(missing_recommended)}. '
            f'Population averages will be used. Prediction accuracy '
            f'may be reduced for patients with hepatic involvement.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-bar">All recommended fields entered. '
            'Prediction will be based on complete clinical profile.</div>',
            unsafe_allow_html=True
        )

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Step 1 of 4 - NHANES 2017-2020 · Ages 18-45")
    with col2:
        predict_disabled = len(missing_required) > 0
        if st.button(
            "Predict MSSS",
            disabled=predict_disabled,
            use_container_width=True,
            type="primary"
        ):
            with st.spinner("Computing prediction..."):
                msss, shap_vals, feature_cols = run_prediction(inputs)
            st.session_state.inputs      = inputs
            st.session_state.prediction  = msss
            st.session_state.shap_values = shap_vals
            st.session_state.feature_cols = feature_cols
            st.session_state.missing_recommended = missing_recommended
            go_to(1)

            # ============================================================
# PAGE 2 - Risk Overview
# ============================================================
def page_risk_overview():
    render_stepper(1)

    st.markdown("## Risk overview")
    st.markdown(
        "Predicted MetS Severity Score based on secondary "
        "biochemistry and lifestyle data."
    )

    msss    = st.session_state.prediction
    missing = st.session_state.get('missing_recommended', [])

    # ── Score hero ───────────────────────────────────────────
    risk_label, risk_class = risk_category(msss)
    pct = percentile_from_msss(msss)

    st.markdown(
        f'<div class="score-hero">'
        f'<div class="score-num">{msss:.2f}</div>'
        f'<div class="score-lbl">Predicted MSSS (continuous z-score)</div>'
        f'<div class="score-stmt">This patient scores higher than '
        f'<strong>{pct}% of young adults aged 18-45</strong> '
        f'in the reference population.</div>'
        f'<div class="risk-tag {risk_class}">{risk_label} '
        f'&nbsp;·&nbsp; MSSS {msss:.2f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Data quality indicator ───────────────────────────────
    if not missing:
        st.markdown(
            '<div class="info-bar">Prediction based on complete '
            'clinical profile. All recommended fields were entered.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="warn-bar">Prediction based on partial '
            f'profile. Population averages used for: '
            f'{", ".join(missing)}. Results may underestimate risk '
            f'in patients with hepatic involvement.</div>',
            unsafe_allow_html=True
        )

    # ── Population distribution ──────────────────────────────
    st.markdown(
        '<div class="sec-head">Population comparison</div>',
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(8, 2.8))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    # Risk zone shading
    ax.axvspan(-3,   -0.5, alpha=0.15, color='#10B981', zorder=0)
    ax.axvspan(-0.5,  0.5, alpha=0.15, color='#F59E0B', zorder=0)
    ax.axvspan(0.5,   1.5, alpha=0.15, color='#EF4444', zorder=0)
    ax.axvspan(1.5,   3.5, alpha=0.20, color='#B91C1C', zorder=0)

    # Zone labels
    for x_pos, label, color in [
        (-1.75, 'Low',       '#065F46'),
        (0.0,   'Moderate',  '#92400E'),
        (1.0,   'High',      '#991B1B'),
        (2.5,   'Very high', '#7F1D1D'),
    ]:
        ax.text(x_pos, 0.42, label, ha='center', va='center',
                fontsize=8, color=color,
                transform=ax.get_xaxis_transform())

    # Density curve (approximated from training cohort)
    x_vals = np.linspace(-3, 3.5, 300)
    from scipy.stats import norm
    y_vals = norm.pdf(x_vals, loc=0.01, scale=1.10)
    ax.plot(x_vals, y_vals, color='#0B6E8C', linewidth=1.8, zorder=2)
    ax.fill_between(x_vals, y_vals, alpha=0.08,
                    color='#0B6E8C', zorder=1)

    # Patient marker
    patient_y = norm.pdf(msss, loc=0.01, scale=1.10)
    ax.axvline(msss, color='#D97706', linewidth=1.5,
               linestyle='--', zorder=3)
    ax.scatter([msss], [patient_y], color='#D97706',
               s=60, zorder=4)
    ax.text(msss + 0.08, patient_y + 0.01,
            f'This patient\nMSSS = {msss:.2f}',
            fontsize=8, color='#D97706', va='bottom')

    ax.set_xlim(-3, 3.5)
    ax.set_xlabel('MSSS', fontsize=10, color='#475569')
    ax.set_ylabel('Density', fontsize=10, color='#475569')
    ax.tick_params(colors='#94A3B8', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#E2E8F0')
    ax.set_ylim(bottom=0)

    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Model info ───────────────────────────────────────────
    st.markdown(
        '<div class="info-bar">Model: ElasticNet (full multimodal) '
        '&nbsp;·&nbsp; R² = 0.736 &nbsp;·&nbsp; 5-fold CV '
        '&nbsp;·&nbsp; n = 1,203 &nbsp;·&nbsp; '
        'NHANES 2017-March 2020</div>',
        unsafe_allow_html=True
    )

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            go_to(0)
    with col2:
        if st.button("See what's driving it",
                     use_container_width=True, type="primary"):
            go_to(2)


# ============================================================
# PAGE 3 - Drivers (SHAP)
# ============================================================
def page_drivers():
    st.write("DEBUG: page_drivers loaded")
    render_stepper(2)
    st.write("DEBUG: after stepper")
    
    msss       = st.session_state.prediction
    shap_vals  = st.session_state.shap_values
    inputs     = st.session_state.inputs
    
    st.write(f"DEBUG: msss={msss}, shap_vals is None={shap_vals is None}")
    st.write(f"DEBUG: feature_cols length={len(feature_cols)}, shap_vals length={len(shap_vals)}")
    st.write(f"DEBUG: label_map loaded, about to build top features")

    st.markdown("## What's driving the prediction")
    st.markdown(
        "Each factor shows how much it pushed the score up or down "
        "compared to an average patient. "
        "Select a factor to see clinical details."
    )
    # ── Top features ─────────────────────────────────────────
    # Human-readable labels for key features
label_map = {
    # Demographics
    'RIDAGEYR'         : 'Age',
    'RIAGENDR'         : 'Sex',
    'RIDRETH3'         : 'Race / ethnicity',
    'RIDEXMON'         : 'Exam month',
    'DMDBORN4'         : 'Country of birth',
    # Body measures
    'BMXBMI'           : 'BMI',
    'BMXWT'            : 'Weight',
    'BMXHT'            : 'Height',
    'BMXWAIST'         : 'Waist circumference',
    'BMXHIP'           : 'Hip circumference',
    'BMXLEG'           : 'Upper leg length',
    'BMXARML'          : 'Upper arm length',
    'BMXARMC'          : 'Arm circumference',
    'BMXSAD1'          : 'Abdominal depth 1',
    'BMXSAD2'          : 'Abdominal depth 2',
    'BMXSAD3'          : 'Abdominal depth 3',
    'BMXSAD4'          : 'Abdominal depth 4',
    'BMXRECUM'         : 'Recumbent length',
    # Blood pressure
    'BPXOSY1'          : 'Systolic BP (1)',
    'BPXOSY2'          : 'Systolic BP (2)',
    'BPXOSY3'          : 'Systolic BP (3)',
    'BPXODI1'          : 'Diastolic BP (1)',
    'BPXODI2'          : 'Diastolic BP (2)',
    'BPXODI3'          : 'Diastolic BP (3)',
    'BPXOPLS1'         : 'Pulse (1)',
    'BPXOPLS2'         : 'Pulse (2)',
    'BPAOCSZ'          : 'BP cuff size',
    'BPAARM'           : 'BP arm used',
    # Biochemistry
    'LBXSATSI'         : 'ALT (liver)',
    'LBXSGTSI'         : 'GGT (liver)',
    'LBXSAPSI'         : 'Alk. phosphatase',
    'LBXSAL'           : 'Albumin',
    'LBXSCH'           : 'Total cholesterol',
    'LBXSCLSI'         : 'Chloride',
    'LBXWBCSI'         : 'WBC count',
    'LBXMOPCT'         : 'Monocyte %',
    'LBXNRBC'          : 'Nucleated RBC',
    'LBXSUA'           : 'Uric acid',
    'LBXSCR'           : 'Creatinine',
    'LBXSGL'           : 'Glucose',
    'LBXSKSI'          : 'Potassium',
    'LBXSNASI'         : 'Sodium',
    'LBXSCA'           : 'Calcium',
    'LBXSPH'           : 'Phosphorus',
    'LBXSTB'           : 'Total bilirubin',
    'LBXSTP'           : 'Total protein',
    'LBXSIR'           : 'Iron',
    'LBDLDL'           : 'LDL cholesterol',
    'LBXRBCSI'         : 'RBC count',
    'LBXHGB'           : 'Hemoglobin',
    'LBXHCT'           : 'Hematocrit',
    'LBXMCVSI'         : 'MCV',
    'LBXPLTSI'         : 'Platelets',
    'LBXBPB'           : 'Blood lead',
    'LBXBCD'           : 'Blood cadmium',
    'LBXTHG'           : 'Blood mercury',
    # Lifestyle
    'total_MET_minutes': 'Physical activity',
    'PAD680'           : 'Sedentary time',
    'SLD012'           : 'Sleep (weekday)',
    'SLD013'           : 'Sleep (weekend)',
    'SMQ020'           : 'Smoking status',
    'ALQ130'           : 'Alcohol (drinks/day)',
    'DPQ050'           : 'Depression (PHQ item)',
}

    # Clinical explanations for panel
explanations = {
        'BMXBMI': {
            'body': 'Elevated BMI is the strongest predictor of MetS '
                    'severity in this model. Values above 30 consistently '
                    'push predicted MSSS upward across the training cohort.',
            'pv_key': 'bmi', 'pv_unit': 'kg/m²', 'mv': '30.0 kg/m²',
            'cite': 'Grundy et al. (2005), Circulation',
            'url' : 'https://www.ahajournals.org/doi/10.1161/CIRCULATIONAHA.105.169404'
        },
        'LBXSATSI': {
            'body': 'Elevated ALT suggests hepatic stress linked to insulin '
                    'resistance. ALT above 40 U/L in young adults is '
                    'associated with early metabolic dysfunction.',
            'pv_key': 'alt', 'pv_unit': 'U/L', 'mv': '22.9 U/L',
            'cite': 'Stefan et al. (2011), Diabetes Care',
            'url' : 'https://pubmed.ncbi.nlm.nih.gov/21179427/'
        },
        'LBXSGTSI': {
            'body': 'GGT reflects oxidative stress and early hepatic fat '
                    'accumulation. Elevated GGT is independently associated '
                    'with MetS incidence in prospective studies.',
            'pv_key': 'ggt', 'pv_unit': 'U/L', 'mv': '26.4 U/L',
            'cite': 'Lee et al. (2005), JAMA',
            'url' : 'https://pubmed.ncbi.nlm.nih.gov/15616165/'
        },
        'LBXSCH': {
            'body': 'Total cholesterol contributes to predicted MSSS. '
                    'The association strengthens above 200 mg/dL.',
            'pv_key': 'cholesterol', 'pv_unit': 'mg/dL', 'mv': '192.4 mg/dL',
            'cite': 'Stone et al. (2013), AHA/ACC Guidelines',
            'url' : 'https://www.ahajournals.org/doi/10.1161/01.cir.0000437738.63853.7a'
        },
        'total_MET_minutes': {
            'body': 'Physical activity above the population median is a '
                    'meaningful protective factor against MetS severity.',
            'pv_key': 'met_minutes', 'pv_unit': 'MET-min/wk',
            'mv': '3,240 MET-min/wk',
            'cite': '2018 Physical Activity Guidelines, HHS',
            'url' : 'https://health.gov/paguidelines/second-edition/'
        },
        'RIDAGEYR': {
            'body': 'Younger age is associated with lower MetS severity '
                    'in this cohort. Age is a consistent predictor across '
                    'all model types.',
            'pv_key': 'age', 'pv_unit': 'yrs', 'mv': '31.2 yrs',
            'cite': 'Gurka et al. (2017), Metabolism',
            'url' : 'https://pubmed.ncbi.nlm.nih.gov/28655017/'
        },
    }

    # Get top 10 features by absolute SHAP
    def clean_label(c):
        if c in label_map:
            return label_map[c]
        return c.replace('_', ' ').replace('LBX', '').replace('BMX', '').replace('BPX', '').strip()

    top_idx  = np.argsort(np.abs(shap_vals))[::-1][:10]
    top_cols = [feature_cols[i] for i in top_idx]
    top_vals = [shap_vals[i] for i in top_idx]
    top_lbls = [clean_label(c) for c in top_cols]
    st.write(f"DEBUG: top_lbls={top_lbls}")
    # ── Layout: chart + panel ────────────────────────────────
    col_chart, col_panel = st.columns([1.2, 1])

    with col_chart:
        # Lollipop chart
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')

        colors = ['#D97706' if v > 0 else '#0B6E8C' for v in top_vals]

        for i, (lbl, val, col) in enumerate(
            zip(top_lbls[::-1], top_vals[::-1], colors[::-1])
        ):
            ax.plot([0, val], [i, i], color=col,
                    linewidth=1.2, zorder=1)
            ax.scatter([val], [i], color=col,
                       s=50, zorder=2)

        ax.axvline(0, color='#E2E8F0', linewidth=1, zorder=0)
        ax.set_yticks(range(len(top_lbls)))
        ax.set_yticklabels(top_lbls[::-1], fontsize=9,
                           color='#475569')
        ax.set_xlabel('SHAP value (impact on MSSS)',
                      fontsize=9, color='#475569')
        ax.tick_params(colors='#94A3B8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#E2E8F0')

        amber_patch = mpatches.Patch(
            color='#D97706', label='Risk-increasing'
        )
        teal_patch  = mpatches.Patch(
            color='#0B6E8C', label='Protective'
        )
        ax.legend(handles=[amber_patch, teal_patch],
                  fontsize=8, loc='lower right',
                  framealpha=0.8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.caption("Click a factor name to see details")
        selected = st.selectbox(
            "Select factor for details",
            options=top_cols,
            format_func=lambda x: label_map.get(x, x),
            label_visibility="collapsed"
        )

    with col_panel:
        if selected and selected in explanations:
            exp = explanations[selected]
            pv_key  = exp.get('pv_key')
            pv_unit = exp.get('pv_unit', '')
            pv_raw  = inputs.get(pv_key)

            if pv_raw is not None:
                pv_display = f"{pv_raw:.1f} {pv_unit}"
            else:
                pv_display = f"Population mean"

            st.markdown(
                f"**{label_map.get(selected, selected)}**"
            )
            st.markdown(
                f'<div class="info-bar">{exp["body"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                f'<div style="display:grid;grid-template-columns:1fr 1fr;'
                f'gap:8px;margin-bottom:10px;">'
                f'<div style="background:white;border:1px solid #E2E8F0;'
                f'border-radius:7px;padding:8px 10px;text-align:center;">'
                f'<div style="font-size:15px;font-weight:600;color:#0B6E8C;">'
                f'{pv_display}</div>'
                f'<div style="font-size:10px;color:#94A3B8;margin-top:2px;">'
                f'This patient</div></div>'
                f'<div style="background:white;border:1px solid #E2E8F0;'
                f'border-radius:7px;padding:8px 10px;text-align:center;">'
                f'<div style="font-size:15px;font-weight:600;color:#0B6E8C;">'
                f'{exp["mv"]}</div>'
                f'<div style="font-size:10px;color:#94A3B8;margin-top:2px;">'
                f'Cohort mean</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(
                f"Reference: [{exp['cite']}]({exp['url']})"
            )
        else:
            st.markdown(
                f"**{label_map.get(selected, selected)}**"
            )
            shap_v = shap_vals[feature_cols.index(selected)]
            direction = "increases" if shap_v > 0 else "decreases"
            st.markdown(
                f'<div class="info-bar">This variable '
                f'{direction} the predicted MSSS by '
                f'{abs(shap_v):.3f} relative to the '
                f'population average.</div>',
                unsafe_allow_html=True
            )

    # ── Domain attribution donut ─────────────────────────────
    st.markdown(
        '<div class="sec-head">Attribution by domain</div>',
        unsafe_allow_html=True
    )

    domain_map = {
        'Biochemistry'  : [c for c in feature_cols
                           if c.startswith('LBX') or
                           c.startswith('LBD')],
        'Dietary'       : [c for c in feature_cols
                           if c.startswith('DR')],
        'Body measures' : [c for c in feature_cols
                           if c.startswith('BMX') or
                           c.startswith('BMI')],
        'Demographics'  : [c for c in feature_cols
                           if c.startswith('RIA') or
                           c.startswith('DMD') or
                           c in ['RIDAGEYR', 'RIDRETH3']],
        'Lifestyle'     : [c for c in feature_cols
                           if c.startswith('SL') or
                           c.startswith('DPQ') or
                           c.startswith('ALQ') or
                           c.startswith('SMQ') or
                           c.startswith('PAD') or
                           c == 'total_MET_minutes'],
        'Blood pressure': [c for c in feature_cols
                           if c.startswith('BPX') or
                           c.startswith('BPA')],
    }

    domain_shap = {}
    for domain, cols in domain_map.items():
        idxs = [feature_cols.index(c)
                for c in cols if c in feature_cols]
        if idxs:
            domain_shap[domain] = float(
                np.abs(shap_vals[idxs]).sum()
            )

    total = sum(domain_shap.values())
    domain_pct = {}
    if total > 0:
        domain_pct = {
            k: v / total * 100
            for k, v in sorted(
                domain_shap.items(),
                key=lambda x: x[1], reverse=True
            )
        }
    else:
        domain_pct = {k: 0.0 for k in domain_shap}
        colors_donut = [
            '#0B6E8C', '#5BA8C0', '#FAC775',
            '#94A3B8', '#CBD5E1', '#E2E8F0'
        ]

    fig2, ax2 = plt.subplots(figsize=(6, 2.5))
    fig2.patch.set_facecolor('#F8FAFC')
    ax2.set_facecolor('#F8FAFC')

    domains  = list(domain_pct.keys())
    pcts     = list(domain_pct.values())
    bars     = ax2.barh(
        domains[::-1], pcts[::-1],
        color=colors_donut[:len(domains)],
        height=0.55
    )

    for bar, pct in zip(bars, pcts[::-1]):
        ax2.text(
            bar.get_width() + 0.5, bar.get_y() +
            bar.get_height() / 2,
            f'{pct:.1f}%', va='center',
            fontsize=9, color='#475569'
        )

    ax2.set_xlabel('% of total SHAP attribution',
                   fontsize=9, color='#475569')
    ax2.tick_params(colors='#475569', labelsize=9)
    ax2.set_xlim(0, max(pcts) * 1.2)
    for spine in ax2.spines.values():
        spine.set_edgecolor('#E2E8F0')

    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            go_to(1)
    with col2:
        if st.button("Clinical summary",
                     use_container_width=True, type="primary"):
            go_to(3)


# ============================================================
# PAGE 4 - Clinical Summary
# ============================================================
def page_summary():
    render_stepper(3)

    st.markdown("## Clinical summary")
    st.markdown(
        "Auto-generated note based on predicted MSSS and SHAP "
        "analysis. Verify all values before clinical use."
    )

    msss    = st.session_state.prediction
    inputs  = st.session_state.inputs
    missing = st.session_state.get('missing_recommended', [])

    risk_label, _ = risk_category(msss)
    pct           = percentile_from_msss(msss)

    age  = inputs.get('age', 'unknown age')
    sex  = inputs.get('sex', 'unknown sex')
    bmi  = inputs.get('bmi')
    alt  = inputs.get('alt')
    ggt  = inputs.get('ggt')
    met  = inputs.get('met_minutes')

    # ── Auto-generated note ──────────────────────────────────
    bmi_str = f"BMI {bmi:.1f} kg/m²" if bmi else "BMI not entered"
    alt_str = f"ALT {alt:.0f} U/L" if alt else "ALT not entered"
    ggt_str = f"GGT {ggt:.0f} U/L" if ggt else "GGT not entered"
    met_str = (f"Physical activity ({met:.0f} MET-min/week) is "
               f"above the cohort median and functions as a "
               f"meaningful protective factor."
               if met and met > 3240
               else "Physical activity data not entered or "
                    "below cohort median.")

    missing_str = (
        f" Note: population averages were used for "
        f"{', '.join(missing)} - results may underestimate "
        f"risk in patients with hepatic involvement."
        if missing else ""
    )

    note = (
        f"This {age}-year-old {sex.lower() if sex else 'patient'} "
        f"presents with a predicted MetS Severity Score of "
        f"{msss:.2f} ({risk_label.lower()}, {pct}th percentile "
        f"for ages 18-45, NHANES reference). Primary anthropometric "
        f"input: {bmi_str}. Hepatic markers: {alt_str}, {ggt_str}. "
        f"{met_str}{missing_str}"
    )

    st.markdown(
        f'<div class="note-box">{note}</div>',
        unsafe_allow_html=True
    )

    if st.button("Copy note to clipboard"):
        st.write(
            f'<script>navigator.clipboard.writeText(`{note}`)'
            f'</script>',
            unsafe_allow_html=True
        )
        st.success("Copied to clipboard.")

    # ── Recommendations ──────────────────────────────────────
    st.markdown(
        '<div class="sec-head">Suggested next steps</div>',
        unsafe_allow_html=True
    )

    recos = [
        {
            'title': 'Order fasting metabolic panel at next visit',
            'body' : 'Fasting lipid panel (triglycerides, HDL), fasting '
                     'plasma glucose, and waist circumference measurement '
                     'would allow formal ATP III MetS criteria assessment. '
                     'Current prediction is based on secondary markers only.',
            'cite' : 'NHLBI - Metabolic Syndrome Diagnosis Criteria',
            'url'  : 'https://www.nhlbi.nih.gov/health/metabolic-syndrome',
            'cite2': 'Grundy et al. (2005), Circulation',
            'url2' : 'https://pubmed.ncbi.nlm.nih.gov/15723527/',
        },
        {
            'title': 'Assess dietary patterns',
            'body' : 'Dietary intake contributes substantially to predicted '
                     'risk attribution in this model. A brief dietary '
                     'screening (REAP-S) or referral to a registered '
                     'dietitian may be appropriate, particularly given '
                     'hepatic enzyme elevation patterns.',
            'cite' : '2020-2025 Dietary Guidelines for Americans',
            'url'  : 'https://www.dietaryguidelines.gov/',
            'cite2': 'Esposito et al. (2020), Nutrients',
            'url2' : 'https://pubmed.ncbi.nlm.nih.gov/31951552/',
        },
        {
            'title': 'Reinforce physical activity',
            'body' : 'Physical activity is the primary modifiable '
                     'protective factor in this model. Maintaining or '
                     'increasing moderate-vigorous activity supports '
                     'continued metabolic protection.',
            'cite' : '2018 Physical Activity Guidelines for Americans (HHS)',
            'url'  : 'https://health.gov/paguidelines/second-edition/',
            'cite2': None,
            'url2' : None,
        },
        {
            'title': 'Monitor liver enzymes at annual visit',
            'body' : 'Elevated ALT and GGT in young adults without known '
                     'liver disease may indicate early NAFLD, which is '
                     'strongly associated with MetS progression. Annual '
                     'liver function panel is appropriate.',
            'cite' : 'AASLD Practice Guidelines - NAFLD',
            'url'  : 'https://www.aasld.org/publications/practice-guidelines',
            'cite2': 'Chalasani et al. (2018), Hepatology',
            'url2' : 'https://pubmed.ncbi.nlm.nih.gov/28756680/',
        },
    ]

    for r in recos:
        cite_str = f"[{r['cite']}]({r['url']})"
        if r['cite2']:
            cite_str += f" &nbsp;·&nbsp; [{r['cite2']}]({r['url2']})"
        st.markdown(
            f'<div class="reco">'
            f'<div class="reco-title">{r["title"]}</div>'
            f'<div class="reco-body">{r["body"]}</div>'
            f'<div class="reco-cite">Source: {cite_str}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Disclaimer ───────────────────────────────────────────
    st.markdown(
        '<div class="disc">Research tool only. Model trained on '
        'NHANES 2017-March 2020 (n=1,203, ages 18-45, '
        'Non-Hispanic White / Black / Hispanic). ElasticNet '
        'regression, R² = 0.736 (5-fold CV across 5 multiply '
        'imputed datasets). Not validated for clinical '
        'decision-making. Not a substitute for clinical judgment. '
        'Jay Narsingani, 2025.</div>',
        unsafe_allow_html=True
    )

    # ── Footer ───────────────────────────────────────────────
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Back", use_container_width=True):
            go_to(2)
    with col2:
        if st.button("New patient",
                     use_container_width=True, type="primary"):
            st.session_state.page       = 0
            st.session_state.inputs     = {}
            st.session_state.prediction = None
            st.session_state.shap_values = None
            st.rerun()

st.write(f"DEBUG current page: {st.session_state.page}")
# ============================================================
# MAIN ROUTER
# ============================================================
page = st.session_state.page

if page == 0:
    page_patient_info()
elif page == 1:
    page_risk_overview()
elif page == 2:
    page_drivers()
elif page == 3:
    page_summary()