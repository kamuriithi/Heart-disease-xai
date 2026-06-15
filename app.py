"""
Heart Disease Prediction – Explainable AI Dashboard
CDAM . Chuka University | Certificate in Python for Data Science & Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, warnings, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #c9d1d9 !important;
    font-size: 0.95rem;
    padding: 6px 0;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }
.metric-card .metric-value {
    font-size: 2.2rem; font-weight: 700; margin: 6px 0 2px;
}
.metric-card .metric-label {
    font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px;
}
.metric-card .metric-delta { font-size: 0.75rem; margin-top: 4px; }

/* ── Section headers ── */
.section-header {
    font-size: 1.35rem; font-weight: 600;
    color: #e6edf3;
    border-left: 4px solid #238636;
    padding-left: 12px;
    margin: 24px 0 14px;
}

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: .5px;
}

/* ── Probability bar ── */
.prob-bar-wrap { background:#21262d; border-radius:999px; height:18px; width:100%; }
.prob-bar-fill { height:18px; border-radius:999px; transition: width .6s; }

/* ── Info box ── */
.info-box {
    background: #1c2128; border: 1px solid #30363d; border-radius: 10px;
    padding: 16px 20px; margin: 10px 0;
}

/* ── Feature table row alternating ── */
.dataframe tbody tr:nth-child(even) { background-color: #161b22 !important; }
.dataframe tbody tr:nth-child(odd)  { background-color: #0d1117 !important; }

/* ── Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white; font-weight: 600; border: none;
    border-radius: 8px; padding: 10px 28px;
    font-size: 1rem; cursor: pointer;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: .85; }

/* ── Select / inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div {
    background: #161b22 !important; border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* ── Tabs ── */
button[data-baseweb="tab"] { color: #8b949e !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
}

/* ── Divider ── */
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ── Artifact paths ─────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    models, metrics, roc_data = {}, {}, {}
    model_files = {
        "Logistic Regression": "Logistic_Regression.pkl",
        "KNN": "KNN.pkl",
        "SVM": "SVM.pkl",
        "Random Forest": "Random_Forest.pkl",
        "XGBoost": "XGBoost.pkl",
    }
    for name, fname in model_files.items():
        p = os.path.join(MODEL_DIR, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)

    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, "roc_data.json")) as f:
        roc_data = json.load(f)

    feature_names  = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    test_data      = joblib.load(os.path.join(MODEL_DIR, "test_data.pkl"))
    shap_values    = joblib.load(os.path.join(MODEL_DIR, "shap_values.pkl"))
    scaler         = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    le_chest       = joblib.load(os.path.join(MODEL_DIR, "le_chest.pkl"))
    le_thal        = joblib.load(os.path.join(MODEL_DIR, "le_thal.pkl"))
    df_orig        = pd.read_csv(os.path.join(MODEL_DIR, "original_data.csv"))

    return models, metrics, roc_data, feature_names, test_data, shap_values, scaler, le_chest, le_thal, df_orig

models, metrics, roc_data, feature_names, test_data, shap_values, scaler, le_chest, le_thal, df_orig = load_all()
X_test = test_data["X"]
y_test = test_data["y"]

# ── Helpers ────────────────────────────────────────────────────────────────────
MODEL_COLORS = {
    "Logistic Regression": "#58a6ff",
    "KNN":                 "#3fb950",
    "SVM":                 "#f78166",
    "Random Forest":       "#d2a8ff",
    "XGBoost":             "#ffa657",
}

FEATURE_INFO = {
    "Age":      "Patient age in years",
    "Sex":      "Biological sex (1=Male, 0=Female)",
    "ChestPain":"Type of chest pain (typical/atypical/nonanginal/asymptomatic)",
    "RestBP":   "Resting blood pressure (mmHg)",
    "Chol":     "Serum cholesterol (mg/dl)",
    "Fbs":      "Fasting blood sugar >120 mg/dl (1=True)",
    "RestECG":  "Resting electrocardiographic results (0-2)",
    "MaxHR":    "Maximum heart rate achieved (bpm)",
    "ExAng":    "Exercise-induced angina (1=Yes, 0=No)",
    "Oldpeak":  "ST depression induced by exercise",
    "Slope":    "Slope of peak exercise ST segment (1-3)",
    "Ca":       "Number of major vessels coloured by fluoroscopy (0-3)",
    "Thal":     "Thalassemia type (normal/fixed defect/reversable defect)",
}

def risk_info(prob):
    if prob < 0.25:
        return "🟢 Low Risk",      "#3fb950", "#0d4a1c"
    elif prob < 0.50:
        return "🟡 Mild Risk",     "#d29922", "#3d2e00"
    elif prob < 0.75:
        return "🟠 Moderate Risk", "#f0883e", "#4d2400"
    else:
        return "🔴 High Risk",     "#f85149", "#4d1117"

def recommendations(prob):
    if prob < 0.25:
        return [
            "✅ Continue routine annual cardiovascular screening",
            "🏃 Maintain regular physical activity (≥150 min/week moderate intensity)",
            "🥗 Follow a heart-healthy diet (Mediterranean or DASH pattern)",
            "🚭 Avoid smoking and limit alcohol consumption",
            "📊 Monitor blood pressure and cholesterol annually",
        ]
    elif prob < 0.50:
        return [
            "⚠️ Schedule follow-up with primary care physician within 3 months",
            "💊 Discuss preventive medications (aspirin, statins) with doctor",
            "📉 Focus on modifiable risk factors: weight, diet, exercise",
            "🩺 Consider stress ECG or echocardiogram based on symptoms",
            "📊 Increase monitoring frequency: BP & lipids every 6 months",
        ]
    elif prob < 0.75:
        return [
            "🏥 Prompt medical consultation recommended within 2–4 weeks",
            "🔬 Order comprehensive lipid panel, HbA1c, and cardiac biomarkers",
            "🫀 Consider cardiac stress testing or coronary CT angiography",
            "💊 Discuss dual antiplatelet or statin therapy with cardiologist",
            "📋 Lifestyle modifications: strict diet, structured cardiac rehab",
        ]
    else:
        return [
            "🚨 Urgent referral to cardiologist or emergency evaluation",
            "🏥 Consider hospital admission for comprehensive cardiac workup",
            "🔬 Immediate coronary angiography may be indicated",
            "💉 Prepare for possible interventional procedures (PCI / CABG)",
            "📞 Ensure patient has emergency contact plan and is educated on warning signs",
        ]

def apply_dark_style(fig, ax_list=None):
    fig.patch.set_facecolor('#0d1117')
    axes = ax_list or fig.axes
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#c9d1d9')
        ax.yaxis.label.set_color('#c9d1d9')
        ax.title.set_color('#e6edf3')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px 0;'>
    """, unsafe_allow_html=True)
    
    # Display Chuka University Logo
    st.image("logo.png", use_container_width=True)
    
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:2px;'>
            Heart Disease Prediction System
        </div>
        <hr style='border-color:#21262d; margin:14px 0 6px;'>
        <div style='font-size:0.72rem; color:#8b949e;'>
            CDAM AI Hub <br> Python for Data Science and ML
        </div>
    </div>
                
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard Overview",
            "🔮 Individual Prediction",
            "📊 Model Performance",
            "📈 ROC Analysis",
            "🧠 Explainable AI (SHAP)",
            "🗃️ Data Explorer",
            "📖 About & Methods",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#21262d;'>", unsafe_allow_html=True)
    best_model = max(metrics, key=lambda m: metrics[m]["roc_auc"])
    best_auc   = metrics[best_model]["roc_auc"]
    st.markdown(f"""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 14px;font-size:0.8rem;'>
        <div style='color:#8b949e; margin-bottom:6px;'>🏆 Top Model</div>
        <div style='color:#f0883e; font-weight:600;'>{best_model}</div>
        <div style='color:#3fb950; font-size:1rem; font-weight:700;'>AUC = {best_auc:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard Overview":
    # Header
    st.markdown("""
    <div style='text-align:center; padding: 30px 0 20px;'>
        <div style='font-size:5.5rem'>🫀</div>
        <h1 style='color:#e6edf3; font-size:2.4rem; margin:10px 0 6px; font-weight:700;'>
            Heart Disease Prediction Dashboard
        </h1>
        <p style='color:#8b949e; font-size:1.05rem; max-width:700px; margin:0 auto;'>
            An Explainable AI clinical decision-support system using machine learning
            for early detection of coronary artery disease
        </p>
        <div style='margin-top:10px;'>
            <span style='background:#161b22;border:1px solid #30363d;border-radius:20px;
                padding:4px 14px;font-size:0.78rem;color:#8b949e;margin:0 4px;'>
                🏫 CDAM - Chuka University
            </span>
            <span style='background:#161b22;border:1px solid #30363d;border-radius:20px;
                padding:4px 14px;font-size:0.78rem;color:#8b949e;margin:0 4px;'>
                📚 Python for Data Science and Machine Learning
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    best_acc = max(metrics[m]["accuracy"] for m in metrics)
    best_f1  = max(metrics[m]["f1"] for m in metrics)
    best_auc_v = max(metrics[m]["roc_auc"] for m in metrics)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Best ROC-AUC", f"{best_auc_v:.4f}", "#58a6ff", "Discriminative power"),
        (c2, "Best Accuracy", f"{best_acc:.1%}", "#3fb950", "Overall correct classifications"),
        (c3, "Best F1 Score", f"{best_f1:.4f}", "#f78166", "Precision-Recall balance"),
        (c4, "Models Trained", str(len(models)), "#d2a8ff", "Algorithms evaluated"),
    ]
    for col, label, val, color, delta in cards:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:{color};">{val}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-delta" style="color:#8b949e;">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">📋 Project Description</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
        <p style='color:#c9d1d9;line-height:1.7;'>
        This dashboard implements a <strong style='color:#58a6ff'>comparative machine learning framework</strong>
        for predicting coronary artery disease (CAD) using clinical and physiological features from
        the UCI Heart Disease dataset. Five algorithms — Logistic Regression, KNN, SVM, Random Forest,
        and XGBoost — are trained and evaluated under a unified preprocessing and validation pipeline.
        </p>
        <p style='color:#c9d1d9;line-height:1.7;margin-top:10px;'>
        <strong style='color:#3fb950'>Clinical Purpose:</strong> Early, data-driven identification of
        patients at elevated cardiovascular risk enables timely referral, targeted intervention, and
        improved patient outcomes — a critical need in resource-constrained health systems.
        </p>
        <p style='color:#c9d1d9;line-height:1.7;margin-top:10px;'>
        <strong style='color:#f0883e'>Explainable AI (XAI):</strong> SHAP (SHapley Additive exPlanations)
        is integrated to make every prediction interpretable at both global (population) and local
        (individual patient) levels — fostering clinician trust and regulatory compliance.
        </p>
        </div>
        """, unsafe_allow_html=True)

        # Features table
        st.markdown('<div class="section-header">🔬 Predictor Variables</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame([
            {"Feature": k, "Description": v,
             "Type": "Categorical" if k in ("Sex","ChestPain","Fbs","ExAng","RestECG","Thal") else "Continuous"}
            for k, v in FEATURE_INFO.items()
        ])
        st.dataframe(feat_df, use_container_width=True, hide_index=True,
                     column_config={"Feature": st.column_config.TextColumn(width="small")})

    with col_right:
        # Leaderboard
        st.markdown('<div class="section-header">🏆 Model Leaderboard</div>', unsafe_allow_html=True)
        lb = []
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
        for i, (m, v) in enumerate(sorted_models):
            lb.append({
                "Rank": medals[i] if i < 3 else str(i+1),
                "Model": m,
                "Accuracy": f"{v['accuracy']:.2%}",
                "Recall": f"{v['recall']:.2%}",
                "F1": f"{v['f1']:.4f}",
                "ROC-AUC": f"{v['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(lb), use_container_width=True, hide_index=True)

        # Risk categories
        st.markdown('<div class="section-header">🚦 Risk Stratification Guide</div>', unsafe_allow_html=True)
        risk_data = [
            ("🟢 Low Risk",      "< 25%",     "Routine monitoring; healthy lifestyle"),
            ("🟡 Mild Risk",     "25% – 50%", "Clinical follow-up; lifestyle modification"),
            ("🟠 Moderate Risk", "50% – 75%", "Additional cardiac tests; medication review"),
            ("🔴 High Risk",     "> 75%",     "Urgent referral; immediate cardiac workup"),
        ]
        for label, rng, action in risk_data:
            st.markdown(f"""
            <div class="info-box" style='margin:6px 0; padding:10px 16px;'>
                <span style='font-weight:600;font-size:.9rem;'>{label}</span>
                <span style='color:#8b949e;font-size:.8rem;margin-left:10px;'>{rng}</span>
                <div style='color:#c9d1d9;font-size:.82rem;margin-top:3px;'>{action}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — INDIVIDUAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Individual Prediction":
    st.markdown('<h2 style="color:#e6edf3;">🔮 Individual Patient Prediction</h2>', unsafe_allow_html=True)
 
    col_form, col_result = st.columns([2, 3])
 
    with col_form:
        st.markdown('<div class="section-header">⚙️ Model & Patient Input</div>', unsafe_allow_html=True)
        sel_model = st.selectbox("Select Model", list(models.keys()), index=0)
 
        with st.expander("👤 Patient Demographics", expanded=True):
            age    = st.slider("Age (years)", 20, 80, 55)
            sex    = st.selectbox("Sex", ["Male (1)", "Female (0)"])
            sex_v  = 1 if "Male" in sex else 0
 
        with st.expander("❤️ Cardiac Symptoms", expanded=True):
            chest_pain = st.selectbox("Chest Pain Type",
                                      ["asymptomatic", "nonanginal", "nontypical", "typical"])
            ex_ang = st.selectbox("Exercise-Induced Angina", ["No (0)", "Yes (1)"])
            ex_ang_v = 1 if "Yes" in ex_ang else 0
 
        with st.expander("🩺 Vital Signs & Labs", expanded=True):
            rest_bp  = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 130)
            chol     = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
            max_hr   = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
            oldpeak  = st.number_input("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, 0.1)
            fbs      = st.selectbox("Fasting Blood Sugar >120 mg/dl", ["No (0)", "Yes (1)"])
            fbs_v    = 1 if "Yes" in fbs else 0
 
        with st.expander("📊 ECG & Imaging", expanded=True):
            rest_ecg = st.selectbox("Resting ECG Result",
                                    ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"])
            rest_ecg_v = int(rest_ecg.split("(")[1].replace(")", ""))
            slope = st.selectbox("ST Slope", ["Upsloping (1)", "Flat (2)", "Downsloping (3)"])
            slope_v = int(slope.split("(")[1].replace(")", ""))
            ca    = st.selectbox("Major Vessels (Ca)", [0, 1, 2, 3])
            thal  = st.selectbox("Thalassemia", ["normal", "fixed", "reversable"])
 
        predict_btn = st.button("🚀 Predict Risk", use_container_width=True)
 
    with col_result:
        if predict_btn:
            # Encode categoricals
            chest_enc = le_chest.transform([chest_pain])[0]
            thal_enc  = le_thal.transform([thal])[0]
 
            raw = np.array([[age, sex_v, chest_enc, rest_bp, chol, fbs_v,
                             rest_ecg_v, max_hr, ex_ang_v, oldpeak, slope_v, ca, thal_enc]])
            X_scaled = scaler.transform(raw)
 
            clf   = models[sel_model]
            prob  = clf.predict_proba(X_scaled)[0][1]
            pred  = int(prob >= 0.5)
 
            label, color, bg_color = risk_info(prob)
            pct   = int(prob * 100)
 
            st.markdown('<div class="section-header">📋 Prediction Result</div>', unsafe_allow_html=True)
 
            # Risk badge
            st.markdown(f"""
            <div style='text-align:center; background:{bg_color};
                border:2px solid {color}; border-radius:14px; padding:20px 24px; margin-bottom:16px;'>
                <div style='font-size:1rem;color:#8b949e;'>Model: {sel_model}</div>
                <div class='risk-badge' style='background:{color}20;color:{color};
                    border:1.5px solid {color}; margin:10px auto; display:table;'>
                    {label}
                </div>
                <div style='font-size:3rem;font-weight:800;color:{color};'>{pct}%</div>
                <div style='color:#8b949e;font-size:.85rem;'>Probability of Heart Disease</div>
            </div>
            """, unsafe_allow_html=True)
 
            # Probability gauge bar
            st.markdown("**Probability Gauge**")
            bar_html = f"""
            <div style='position:relative;'>
                <div style='display:flex;justify-content:space-between;
                    font-size:.72rem;color:#8b949e;margin-bottom:4px;'>
                    <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100%</span>
                </div>
                <div class='prob-bar-wrap'>
                    <div class='prob-bar-fill' style='width:{pct}%;background:
                        {"#3fb950" if prob<.25 else "#d29922" if prob<.5 else "#f0883e" if prob<.75 else "#f85149"};'>
                    </div>
                </div>
                <div style='display:flex;gap:8px;margin-top:6px;font-size:.7rem;'>
                    <span style='color:#3fb950;'>■ Low</span>
                    <span style='color:#d29922;'>■ Mild</span>
                    <span style='color:#f0883e;'>■ Moderate</span>
                    <span style='color:#f85149;'>■ High</span>
                </div>
            </div>"""
            st.markdown(bar_html, unsafe_allow_html=True)
 
            # Recommendations
            st.markdown('<div class="section-header">💊 Clinical Recommendations</div>',
                        unsafe_allow_html=True)
            for rec in recommendations(prob):
                st.markdown(f"""
                <div style='background:#ffffff;border:1px solid #a0cfe8;border-radius:8px;
                    padding:9px 14px;margin:5px 0;color:#0a2540;font-size:.9rem;'>{rec}</div>
                """, unsafe_allow_html=True)
 
            # SHAP waterfall
            st.markdown('<div class="section-header">🧠 SHAP Explanation</div>', unsafe_allow_html=True)
            if sel_model in shap_values:
                sv_data = shap_values[sel_model]
                sv_arr  = np.array(sv_data["values"], dtype=float)
                if sv_arr.ndim == 3:
                    sv_arr = sv_arr[:, :, 1]
                elif sv_arr.ndim == 1:
                    sv_arr = sv_arr.reshape(1, -1)
                base_v  = float(sv_data["base_value"])
 
                # Use first test sample as reference proxy for SHAP waterfall
                # Flatten to 1-D array of scalars regardless of storage format
                sample_shap = np.array(sv_arr[0], dtype=float).flatten()
                # Ensure length matches feature_names
                n_feats = len(feature_names)
                if len(sample_shap) > n_feats:
                    sample_shap = sample_shap[:n_feats]
                elif len(sample_shap) < n_feats:
                    sample_shap = np.pad(sample_shap, (0, n_feats - len(sample_shap)))
 
                feat_shap = list(zip(feature_names, sample_shap.tolist()))
                feat_shap_sorted = sorted(feat_shap, key=lambda x: abs(float(x[1])), reverse=True)[:10]
 
                fnames_wf = [f[0] for f in feat_shap_sorted]
                fvals_wf  = [float(f[1]) for f in feat_shap_sorted]
 
                fig_wf, ax_wf = plt.subplots(figsize=(7, 4.5))
                colors_wf = ["#f85149" if v > 0 else "#3fb950" for v in fvals_wf]
                bars = ax_wf.barh(range(len(fnames_wf)), fvals_wf[::-1],
                                  color=colors_wf[::-1], edgecolor='none', height=0.6)
                ax_wf.set_yticks(range(len(fnames_wf)))
                ax_wf.set_yticklabels(fnames_wf[::-1], fontsize=10)
                ax_wf.axvline(0, color='#8b949e', linewidth=0.8, linestyle='--')
                ax_wf.set_xlabel("SHAP Value", fontsize=10)
                ax_wf.set_title(f"Feature Contributions – {sel_model}", fontsize=11, pad=10)
                p_pos = mpatches.Patch(color='#f85149', label='↑ Increases Risk')
                p_neg = mpatches.Patch(color='#3fb950', label='↓ Decreases Risk')
                ax_wf.legend(handles=[p_pos, p_neg], fontsize=8,
                             facecolor='#ffffff', edgecolor='#a0cfe8',
                             labelcolor='#0a2540')
                apply_dark_style(fig_wf)
                plt.tight_layout()
                st.pyplot(fig_wf, use_container_width=True)
                plt.close(fig_wf)
 
                st.markdown(f"""
                <div class='info-box' style='font-size:.82rem;color:#3a7ca5; background:#ffffff; border:1px solid #a0cfe8;'>
                    🔵 Base value (average model output): <strong style='color:#58a6ff;'>{base_v:.4f}</strong>
                    &nbsp;|&nbsp; Red bars push prediction higher &nbsp;|&nbsp; Green bars push prediction lower
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='text-align:center;padding:80px 40px;color:#8b949e;'>
                <div style='font-size:3rem;'>🔮</div>
                <div style='font-size:1.1rem;margin-top:12px;'>
                    Fill in patient information and click <strong style='color:#3fb950;'>Predict Risk</strong>
                </div>
            </div>""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown('<h2 style="color:#e6edf3;">📊 Model Performance Comparison</h2>', unsafe_allow_html=True)

    # Performance table
    st.markdown('<div class="section-header">📋 Performance Summary Table</div>', unsafe_allow_html=True)
    perf_rows = []
    for m, v in sorted(metrics.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        perf_rows.append({
            "Model": m,
            "Accuracy": f"{v['accuracy']:.2%}",
            "Precision": f"{v['precision']:.2%}",
            "Recall": f"{v['recall']:.2%}",
            "F1 Score": f"{v['f1']:.4f}",
            "ROC-AUC": f"{v['roc_auc']:.4f}",
        })
    st.dataframe(pd.DataFrame(perf_rows), use_container_width=True, hide_index=True)

    # Grouped bar chart
    st.markdown('<div class="section-header">📊 Metric Comparison Chart</div>', unsafe_allow_html=True)
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    model_names = list(metrics.keys())
    x = np.arange(len(metric_labels))
    width = 0.15

    fig_bar, ax_bar = plt.subplots(figsize=(12, 5))
    for i, m in enumerate(model_names):
        vals = [metrics[m][k] for k in metric_keys]
        offset = (i - len(model_names)/2 + 0.5) * width
        bars = ax_bar.bar(x + offset, vals, width*0.9, label=m,
                          color=MODEL_COLORS.get(m, "#58a6ff"), alpha=0.88, edgecolor='none')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metric_labels, fontsize=11)
    ax_bar.set_ylim(0.5, 1.02)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_title("Model Performance Across All Metrics", fontsize=13, pad=12)
    ax_bar.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    apply_dark_style(fig_bar)
    ax_bar.yaxis.grid(True, color='#21262d', linewidth=0.6)
    ax_bar.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close(fig_bar)

    # Confusion matrices
    st.markdown('<div class="section-header">🔲 Confusion Matrices</div>', unsafe_allow_html=True)
    cols_cm = st.columns(len(metrics))
    for idx, (mname, mval) in enumerate(metrics.items()):
        with cols_cm[idx]:
            cm_arr = np.array(mval["confusion_matrix"])
            fig_cm, ax_cm = plt.subplots(figsize=(3.2, 2.8))
            sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
                        ax=ax_cm, cbar=False, linewidths=.5,
                        annot_kws={"size": 14, "weight": "bold"},
                        xticklabels=["No CAD", "CAD"],
                        yticklabels=["No CAD", "CAD"])
            ax_cm.set_xlabel("Predicted", fontsize=9, color='#c9d1d9')
            ax_cm.set_ylabel("Actual", fontsize=9, color='#c9d1d9')
            ax_cm.set_title(mname, fontsize=9, pad=6, color='#e6edf3')
            fig_cm.patch.set_facecolor('#0d1117')
            ax_cm.set_facecolor('#161b22')
            ax_cm.tick_params(colors='#8b949e', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ROC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ROC Analysis":
    st.markdown('<h2 style="color:#e6edf3;">📈 ROC Curve Analysis</h2>', unsafe_allow_html=True)

    col_roc, col_rank = st.columns([3, 2])

    with col_roc:
        st.markdown('<div class="section-header">🔵 Combined ROC Curves</div>', unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(7, 5.5))

        # Random classifier
        ax_roc.plot([0,1],[0,1], linestyle='--', color='#8b949e',
                    linewidth=1.2, label='Random Classifier (AUC=0.50)')

        sorted_roc = sorted(roc_data.items(), key=lambda x: x[1]["auc"], reverse=True)
        for mname, rdata in sorted_roc:
            color = MODEL_COLORS.get(mname, "#58a6ff")
            ax_roc.plot(rdata["fpr"], rdata["tpr"],
                        color=color, linewidth=2.2,
                        label=f"{mname} (AUC={rdata['auc']:.4f})")

        ax_roc.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=11)
        ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax_roc.set_title("ROC Curves – All Models", fontsize=13, pad=10)
        ax_roc.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9',
                      loc='lower right')
        ax_roc.set_xlim(-0.02, 1.02)
        ax_roc.set_ylim(-0.02, 1.04)
        apply_dark_style(fig_roc)
        ax_roc.grid(True, color='#21262d', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        plt.close(fig_roc)

    with col_rank:
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        sorted_models = sorted(metrics.items(), key=lambda x: x[1]["roc_auc"], reverse=True)
        st.markdown('<div class="section-header">🏆 AUC Ranking</div>', unsafe_allow_html=True)
        for i, (mname, mval) in enumerate(sorted_models):
            color = MODEL_COLORS.get(mname, "#58a6ff")
            bar_w = int(mval["roc_auc"] * 100)
            st.markdown(f"""
            <div class='info-box' style='margin:6px 0; padding:12px 16px;'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;'>
                    <span style='font-size:1.1rem;'>{medals[i]} {mname}</span>
                    <span style='color:{color};font-weight:700;font-size:1.1rem;'>{mval['roc_auc']:.4f}</span>
                </div>
                <div style='background:#21262d;border-radius:999px;height:8px;'>
                    <div style='width:{bar_w}%;background:{color};height:8px;border-radius:999px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">📚 ROC Interpretation</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box' style='font-size:.84rem;line-height:1.8;color:#c9d1d9;'>
            <strong style='color:#58a6ff;'>ROC Curve</strong> plots Sensitivity vs. 1–Specificity
            at every decision threshold.<br><br>
            <strong style='color:#3fb950;'>AUC (Area Under Curve)</strong><br>
            • 1.00 = Perfect classifier<br>
            • 0.90–0.99 = Excellent discrimination<br>
            • 0.80–0.89 = Good discrimination<br>
            • 0.70–0.79 = Fair discrimination<br>
            • 0.50 = No discrimination (random)<br><br>
            <strong style='color:#f0883e;'>Clinical Interpretation:</strong> A high AUC indicates
            the model can reliably distinguish patients with and without coronary artery disease,
            making it suitable for clinical decision support.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABLE AI (SHAP)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Explainable AI (SHAP)":
    st.markdown('<h2 style="color:#e6edf3;">🧠 Explainable AI — SHAP Analysis</h2>', unsafe_allow_html=True)
 
    shap_model = st.selectbox("Select Model for SHAP Analysis", list(shap_values.keys()), index=0)
 
    sv_data  = shap_values[shap_model]
    sv_arr   = np.array(sv_data["values"], dtype=float)
    # If stored as 3D (samples, features, classes) squeeze to 2D
    if sv_arr.ndim == 3:
        sv_arr = sv_arr[:, :, 1]
    elif sv_arr.ndim == 1:
        sv_arr = sv_arr.reshape(1, -1)
    base_val = float(sv_data["base_value"])
 
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Feature Importance", "🐝 SHAP Beeswarm",
        "🌊 Individual Waterfall", "🔗 Dependence Plot"
    ])
 
    # ── Tab 1: Feature Importance ──────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">📊 Mean |SHAP| Feature Importance</div>',
                    unsafe_allow_html=True)
        mean_abs = np.abs(sv_arr).mean(axis=0).astype(float).flatten()
        fi_sorted = sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: float(x[1]))
 
        col_fi1, col_fi2 = st.columns([3, 2])
        with col_fi1:
            fig_fi, ax_fi = plt.subplots(figsize=(7, 5))
            colors_fi = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(fi_sorted)))
            ax_fi.barh([f[0] for f in fi_sorted], [f[1] for f in fi_sorted],
                       color=colors_fi, edgecolor='none', height=0.65)
            ax_fi.set_xlabel("Mean |SHAP Value|", fontsize=10)
            ax_fi.set_title(f"Global Feature Importance — {shap_model}", fontsize=11, pad=8)
            apply_dark_style(fig_fi)
            ax_fi.xaxis.grid(True, color='#b8d8ea', linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)
 
        with col_fi2:
            fi_df = pd.DataFrame(
                [(f, f"{float(v):.4f}") for f, v in sorted(zip(feature_names, mean_abs.tolist()),
                                                      key=lambda x: float(x[1]), reverse=True)],
                columns=["Feature", "Mean |SHAP|"]
            )
            fi_df.insert(0, "Rank", range(1, len(fi_df)+1))
            st.dataframe(fi_df, use_container_width=True, hide_index=True)
            st.markdown("""
            <div class='info-box' style='font-size:.82rem;color:#3a7ca5;margin-top:10px; background:#ffffff; border:1px solid #a0cfe8;'>
                <strong>Mean |SHAP|</strong> = average absolute contribution of each feature
                across all test patients. Higher = more influential in model decisions.
            </div>""", unsafe_allow_html=True)
 
    # ── Tab 2: Beeswarm ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-header">🐝 SHAP Beeswarm (Summary) Plot</div>',
                    unsafe_allow_html=True)
 
        # Sort features by mean|shap|
        order = np.argsort(np.abs(sv_arr).mean(axis=0))
        sv_ord = sv_arr[:, order]
        fn_ord = [feature_names[i] for i in order]
 
        fig_bs, ax_bs = plt.subplots(figsize=(9, 6))
        for j, (fname, sv_col) in enumerate(zip(fn_ord, sv_ord.T)):
            raw_feat = X_test[:, order[j]]
            # Align lengths — KernelExplainer pads zeros for rows beyond n=50
            n       = min(len(raw_feat), len(sv_col))
            rf_n    = raw_feat[:n]
            sv_n    = sv_col[:n]
            normed  = (rf_n - rf_n.min()) / ((rf_n.max() - rf_n.min()) + 1e-9)
            jitter  = np.random.uniform(-0.25, 0.25, size=n)
            ax_bs.scatter(sv_n, np.full(n, j, dtype=float) + jitter,
                          c=normed, cmap='coolwarm', vmin=0, vmax=1,
                          s=10, alpha=0.65, linewidths=0)
        ax_bs.set_yticks(range(len(fn_ord)))
        ax_bs.set_yticklabels(fn_ord, fontsize=10)
        ax_bs.axvline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax_bs.set_xlabel("SHAP Value", fontsize=11)
        ax_bs.set_title(f"SHAP Beeswarm Plot — {shap_model}", fontsize=12, pad=10)
        apply_dark_style(fig_bs)
        ax_bs.xaxis.grid(True, color='#b8d8ea', linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig_bs, use_container_width=True)
        plt.close(fig_bs)
 
        st.markdown("""
        <div class='info-box' style='font-size:.85rem;line-height:1.7;color:#0a2540; background:#ffffff; border:1px solid #a0cfe8;'>
            🔴 <strong>Red dots</strong> = high feature value &nbsp;|&nbsp;
            🔵 <strong>Blue dots</strong> = low feature value<br>
            Dots to the <strong>right of zero</strong> increase the predicted risk.
            Dots to the <strong>left</strong> decrease it.
            The vertical spread shows distribution over all test patients.
        </div>""", unsafe_allow_html=True)
 
    # ── Tab 3: Individual Waterfall ──────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">🌊 Individual Patient Waterfall</div>',
                    unsafe_allow_html=True)
        # Limit slider to rows where SHAP values are valid (non-padded)
        n_patients = min(sv_arr.shape[0], len(y_test))
        patient_idx = st.slider("Select Patient Index", 0, n_patients - 1, 0)
        true_label  = y_test[patient_idx]
        sample_sv   = sv_arr[patient_idx]
 
        pairs   = sorted(zip(feature_names, sample_sv), key=lambda x: x[1])
        f_names = [p[0] for p in pairs]
        f_vals  = [p[1] for p in pairs]
        colors_wfall = ["#3fb950" if v < 0 else "#f85149" for v in f_vals]
 
        fig_wfall, ax_wfall = plt.subplots(figsize=(8, 5.5))
        ax_wfall.barh(f_names, f_vals, color=colors_wfall, edgecolor='none', height=0.6)
        ax_wfall.axvline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax_wfall.set_xlabel("SHAP Value", fontsize=11)
        ax_wfall.set_title(
            f"Waterfall — Patient #{patient_idx}  |  True Label: {'CAD ✅' if true_label else 'No CAD ❌'}",
            fontsize=11, pad=10)
        p_pos = mpatches.Patch(color='#f85149', label='Increases Risk')
        p_neg = mpatches.Patch(color='#3fb950', label='Decreases Risk')
        ax_wfall.legend(handles=[p_pos, p_neg], fontsize=9,
                        facecolor='#ffffff', edgecolor='#a0cfe8', labelcolor='#0a2540')
        apply_dark_style(fig_wfall)
        ax_wfall.xaxis.grid(True, color='#b8d8ea', linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig_wfall, use_container_width=True)
        plt.close(fig_wfall)
 
        st.markdown(f"""
        <div class='info-box' style='font-size:.85rem;color:#3a7ca5; background:#ffffff; border:1px solid #a0cfe8;'>
            Model baseline (E[f(X)]): <strong style='color:#58a6ff;'>{base_val:.4f}</strong>
            &nbsp;|&nbsp; Sum of SHAP contributions shifts prediction from baseline to final output.
        </div>""", unsafe_allow_html=True)
 
    # ── Tab 4: Dependence Plot ────────────────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-header">🔗 SHAP Dependence Plot</div>', unsafe_allow_html=True)
        feat_sel = st.selectbox("Select Feature", feature_names, index=feature_names.index("Age"))
        fi       = feature_names.index(feat_sel)
 
        # Align lengths — sv_arr may be shorter than X_test due to KernelExplainer padding
        n_dep      = min(len(X_test), len(sv_arr))
        feat_vals  = X_test[:n_dep, fi]
        shap_for_f = sv_arr[:n_dep, fi]
 
        # Colour by highest-interaction feature (Ca)
        color_fi   = feature_names.index("Ca")
        color_vals = X_test[:n_dep, color_fi]
 
        fig_dep, ax_dep = plt.subplots(figsize=(8, 5))
        sc = ax_dep.scatter(feat_vals, shap_for_f, c=color_vals,
                            cmap='RdYlGn_r', s=35, alpha=0.75, linewidths=0)
        cb = fig_dep.colorbar(sc, ax=ax_dep)
        cb.set_label("Ca (# Vessels)", color='#c9d1d9', fontsize=9)
        cb.ax.yaxis.set_tick_params(color='#8b949e')
        plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color='#8b949e')
        ax_dep.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
        ax_dep.set_xlabel(f"{feat_sel} (scaled)", fontsize=11)
        ax_dep.set_ylabel("SHAP Value", fontsize=11)
        ax_dep.set_title(f"SHAP Dependence: {feat_sel} — {shap_model}", fontsize=12, pad=10)
        apply_dark_style(fig_dep)
        ax_dep.grid(True, color='#b8d8ea', linewidth=0.4)
        plt.tight_layout()
        st.pyplot(fig_dep, use_container_width=True)
        plt.close(fig_dep)
 
        st.markdown(f"""
        <div class='info-box' style='font-size:.85rem;line-height:1.7;color:#0a2540; background:#ffffff; border:1px solid #a0cfe8;'>
            X-axis: scaled values of <strong>{feat_sel}</strong>.
            Y-axis: SHAP contribution to predicted risk. Points above zero push the prediction
            higher; points below reduce it. Colour encodes a secondary interaction feature (Ca).
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗃️ Data Explorer":
    st.markdown('<h2 style="color:#e6edf3;">🗃️ Interactive Data Explorer</h2>', unsafe_allow_html=True)

    df_e = df_orig.copy()
    df_e["HD_Label"] = df_e["HD"].map({0: "No CAD", 1: "CAD"})

    # Dataset summary
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    n_cad   = int((df_e["HD"] == 1).sum())
    n_nocad = int((df_e["HD"] == 0).sum())
    prev    = n_cad / len(df_e)
    for col, val, lbl, col_hex in [
        (col_s1, str(len(df_e)),       "Total Patients", "#58a6ff"),
        (col_s2, str(n_cad),           "CAD Cases",      "#f85149"),
        (col_s3, str(n_nocad),         "No CAD",         "#3fb950"),
        (col_s4, f"{prev:.1%}",        "Prevalence",     "#f0883e"),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{col_hex};'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_tbl, col_dist = st.columns([3, 2])

    with col_tbl:
        st.markdown('<div class="section-header">🔎 Interactive Table</div>', unsafe_allow_html=True)
        outcome_filter = st.selectbox("Filter by Outcome", ["All", "CAD", "No CAD"])
        if outcome_filter == "CAD":
            df_view = df_e[df_e["HD"] == 1]
        elif outcome_filter == "No CAD":
            df_view = df_e[df_e["HD"] == 0]
        else:
            df_view = df_e
        st.dataframe(df_view.drop(columns=["HD_Label"]), use_container_width=True,
                     height=320, hide_index=True)

    with col_dist:
        st.markdown('<div class="section-header">📊 Feature Distribution</div>', unsafe_allow_html=True)
        num_feats = [c for c in feature_names if df_e[c].dtype in (float, np.float64, int, np.int64)]
        sel_feat  = st.selectbox("Select Feature", num_feats, index=num_feats.index("Age"))

        fig_dist, ax_dist = plt.subplots(figsize=(5.5, 4))
        for label, color in [("No CAD", "#3fb950"), ("CAD", "#f85149")]:
            sub = df_e[df_e["HD_Label"] == label][sel_feat].dropna()
            ax_dist.hist(sub, bins=18, alpha=0.7, color=color, label=label, edgecolor='none')
        ax_dist.set_xlabel(sel_feat, fontsize=10)
        ax_dist.set_ylabel("Count", fontsize=10)
        ax_dist.set_title(f"Distribution of {sel_feat}", fontsize=11, pad=8)
        ax_dist.legend(fontsize=9, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
        apply_dark_style(fig_dist)
        plt.tight_layout()
        st.pyplot(fig_dist, use_container_width=True)
        plt.close(fig_dist)

    # Correlation heatmap
    st.markdown('<div class="section-header">🔥 Correlation Heatmap</div>', unsafe_allow_html=True)
    num_cols_only = [c for c in feature_names
                     if df_e[c].dtype in (float, np.float64, int, np.int64)] + ["HD"]
    corr_df = df_e[num_cols_only].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(corr_df, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax_corr, cbar_kws={"shrink": .8},
                linewidths=.3, linecolor='#0d1117',
                annot_kws={"size": 8})
    ax_corr.set_title("Feature Correlation Matrix", fontsize=13, pad=10, color='#e6edf3')
    fig_corr.patch.set_facecolor('#0d1117')
    ax_corr.set_facecolor('#161b22')
    ax_corr.tick_params(colors='#8b949e', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig_corr, use_container_width=True)
    plt.close(fig_corr)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT & METHODS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖 About & Methods":
    st.markdown('<h2 style="color:#e6edf3;">📖 About This System & Methodology</h2>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    sections_left = [
        ("📁 Dataset Information", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li><strong>Source:</strong> UCI Heart Disease Dataset (Cleveland Clinic Foundation)</li>
            <li><strong>Sample size:</strong> 303 patients (299 after removing missing values)</li>
            <li><strong>Features:</strong> 13 clinical and physiological predictors</li>
            <li><strong>Target:</strong> HD — binary indicator of coronary artery disease (CAD)</li>
            <li><strong>Prevalence:</strong> ~46% positive cases</li>
        </ul>"""),
        ("⚙️ Preprocessing Pipeline", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li><strong>Missing values:</strong> Rows with missing Ca or Thal dropped (n=4)</li>
            <li><strong>Categorical encoding:</strong> LabelEncoder for ChestPain and Thal</li>
            <li><strong>Feature scaling:</strong> StandardScaler (zero mean, unit variance)</li>
            <li><strong>Train-test split:</strong> 80/20 stratified by outcome</li>
            <li><strong>No data leakage:</strong> Scaler fit only on training set</li>
        </ul>"""),
        ("🤖 Machine Learning Models", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li><strong>Logistic Regression:</strong> Baseline linear classifier with L2 regularisation</li>
            <li><strong>K-Nearest Neighbours:</strong> Non-parametric, distance-based classifier</li>
            <li><strong>Support Vector Machine:</strong> RBF kernel, probabilistic output via Platt scaling</li>
            <li><strong>Random Forest:</strong> Ensemble of 200 decision trees with bagging</li>
            <li><strong>XGBoost:</strong> Gradient-boosted trees with early stopping protection</li>
        </ul>"""),
    ]

    sections_right = [
        ("🧠 Explainable AI — SHAP", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li><strong>SHAP</strong> (SHapley Additive exPlanations) provides game-theoretically
            grounded feature attributions</li>
            <li><strong>TreeExplainer:</strong> Exact, fast SHAP for tree-based models (RF, XGB)</li>
            <li><strong>KernelExplainer:</strong> Model-agnostic SHAP for LR, KNN, SVM using
            background data via k-means clustering</li>
            <li>Both global (population) and local (individual) explanations are supported</li>
        </ul>"""),
        ("📏 Evaluation Framework", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li><strong>Primary metric:</strong> ROC-AUC (discrimination across all thresholds)</li>
            <li><strong>Secondary metrics:</strong> Accuracy, Precision, Recall, F1 Score</li>
            <li><strong>Threshold-free evaluation:</strong> ROC-AUC preferred for imbalanced outcomes</li>
            <li><strong>Confusion matrix:</strong> Visual inspection of error types per model</li>
        </ul>"""),
        ("🔬 Limitations", """
        <ul style='color:#c9d1d9;line-height:1.8;'>
            <li>Dataset is relatively small (n≈299); larger cohort needed for generalisation</li>
            <li>External validation on African/Kenyan populations not yet conducted</li>
            <li>KernelExplainer approximations computed on subset (n=50) for computational efficiency</li>
            <li>No hyperparameter tuning (GridSearchCV) applied in this version</li>
        </ul>"""),
    ]

    with col1:
        for title, content in sections_left:
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">{content}</div>', unsafe_allow_html=True)

    with col2:
        for title, content in sections_right:
            st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">{content}</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown('<div class="section-header">⚠️ Disclaimer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1c2128;border:2px solid #f0883e;border-radius:12px;padding:20px 24px;'>
        <p style='color:#f0883e;font-weight:700;margin-bottom:8px;font-size:1.05rem;'>
            ⚕️ Educational & Research Use Only
        </p>
        <p style='color:#c9d1d9;line-height:1.75;margin:0;'>
        This dashboard is developed for academic research and educational purposes as part of a
        <strong>Python for Data Science and Machine Learning Course </strong> project at CDAM-Chuka University.
        The predictions generated by the models are <strong>probabilistic estimates</strong> based on
        historical population data and <strong>must not be used as a substitute for professional
        clinical diagnosis, medical advice, or treatment decisions</strong>. All outputs should be
        interpreted and validated by a qualified healthcare professional. The developers accept no
        liability for clinical decisions made based on this system.
        </p>
    </div>
    """, unsafe_allow_html=True)
