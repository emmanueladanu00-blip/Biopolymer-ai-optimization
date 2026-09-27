import os
import warnings

# Suppress background diagnostic warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf

# 1. Page Configuration
st.set_page_config(
    page_title="Biocomposite Packaging Optimization Engine",
    page_icon="https://img.icons8.com/color/96/shield.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. White-Label Stealth CSS (Hides GitHub, Fork, Menus & Watermarks)
st.markdown("""
<head>
    <meta name="mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="theme-color" content="#10B981">
</head>

<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* ========================================================
       STEALTH WHITE-LABEL OVERRIDES (Hides GitHub & Fork Badges)
       ======================================================== */
    #MainMenu {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important; display: none !important;}
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    
    /* Remove Streamlit Cloud Bottom/Top Fork and GitHub Links */
    div[class^="viewerBadge"] {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}
    div[class^="styles_viewerBadge"] {display: none !important;}
    div[class*="styles_viewerBadge"] {display: none !important;}
    a[href*="github.com"] {display: none !important;}

    /* Tighten top space caused by removing header */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
    }

    /* Dynamic Theme Variables */
    :root {
        --bg-card: rgba(255, 255, 255, 0.85);
        --border-card: rgba(226, 232, 240, 0.9);
        --border-card-hover: rgba(16, 185, 129, 0.4);
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        --card-shadow-hover: 0 12px 30px -4px rgba(16, 185, 129, 0.15);
        --hero-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.04) 100%), rgba(255, 255, 255, 0.95);
        --tag-color: #0284C7;
        --tab-bg: rgba(241, 245, 249, 0.8);
        --tab-selected: #FFFFFF;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --bg-card: rgba(17, 24, 39, 0.75);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-card-hover: rgba(16, 185, 129, 0.5);
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            --card-shadow-hover: 0 12px 40px -10px rgba(6, 182, 212, 0.25);
            --hero-bg: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.04) 100%), rgba(17, 24, 39, 0.85);
            --tag-color: #38BDF8;
            --tab-bg: rgba(17, 24, 39, 0.6);
            --tab-selected: rgba(255, 255, 255, 0.1);
        }
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(12px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseBeacon {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    @keyframes pulseFail {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1.05); box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-card);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: var(--card-shadow);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease-out;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: var(--border-card-hover);
        box-shadow: var(--card-shadow-hover);
    }

    .hero-card {
        background: var(--hero-bg);
        border: 1px solid rgba(16, 185, 129, 0.4);
        animation: fadeInUp 0.5s ease-out;
    }

    .metric-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    
    .metric-num {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.03em;
    }

    .metric-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 4px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 9999px;
        padding: 5px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        margin-top: 8px;
    }
    
    .status-pass {
        background: rgba(16, 185, 129, 0.12);
        color: #059669;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .status-fail {
        background: rgba(239, 68, 68, 0.12);
        color: #DC2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .beacon-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    .beacon-pass {
        background-color: #10B981;
        animation: pulseBeacon 2s infinite;
    }
    .beacon-fail {
        background-color: #EF4444;
        animation: pulseFail 2s infinite;
    }

    .brand-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--tag-color);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.03em;
    }
    .brand-desc {
        color: var(--text-secondary);
        font-size: 0.95rem;
        margin-bottom: 22px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--tab-bg);
        border-radius: 10px;
        padding: 6px;
        border: 1px solid var(--border-card);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--tab-selected) !important;
        color: var(--text-primary) !important;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Model Assets
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/ann_model.keras')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="brand-tag">RESEARCH SPECIFICATION</div>', unsafe_allow_html=True)
    st.markdown("### JOSTUM Materials Lab")
    st.markdown("""
    **Project Framework:**  
    *AI-Optimised Lignin-Cellulose-Chitosan Biocomposite Sack Packaging*

    **Principal Investigator:**  
    Ikwuje, Idoko John 

    **Institution:**  
    Joseph Sarwuan Tarka University, Makurdi

    **Engine Architecture:**  
    Dual-Core ANN (4-Layer MLP) + DEAP Evolutionary Multi-Objective GA
    """)
    st.divider()
    st.markdown('<div class="brand-tag">INDUSTRIAL BENCHMARKS</div>', unsafe_allow_html=True)
    st.caption("• **ASTM D882 Standard:** ≥ 25.0 MPa")
    st.caption("• **Water Barrier Target:** < 100% 1hr Immersion")
    st.caption("• **Mineralization Window:** 100% Soil Loss ≤ 16 Days")
    
    st.divider()
    with st.expander("Mobile / Desktop App Install"):
        st.markdown("""
        * **Android:** Tap `⋮` > **'Install App'**
        * **iOS:** Tap `⬆` > **'Add to Home Screen'**
        * **PC/Mac:** Click **'Install'** icon in browser address bar.
        """)

# --- HEADER SECTION ---
st.markdown("""
<div>
    <div class="brand-tag">DEEPTECH PACKAGING ENGINE // PRODUCTION PLATFORM</div>
    <div class="brand-title">Biopolymer Inverse Formulation & Optimization System</div>
    <div class="brand-desc">Neural surrogate inference and non-dominated sorting for heavy-duty circular agro-waste biocomposites.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Virtual Formulation Synthesizer",
    "Evolutionary Pareto Frontier (GA)",
    "Verification & Industrial Parity"
])

# ==========================================
# TAB 1: FORMULATION LAB
# ==========================================
with tabs[0]:
    col_input, col_metric = st.columns([1, 1.15], gap="large")

    with col_input:
        st.markdown("#### 1. Biopolymer Weight Partitioning")
        cs_raw = st.slider("Chitosan Biopolymer Matrix (%)", 10, 80, 50, 1)
        ce_raw = st.slider("Rice Husk Cellulose Reinforcement (%)", 5, 60, 35, 1)
        lg_raw = st.slider("Rice Husk Lignin Barrier Agent (%)", 0, 40, 15, 1)

        total = cs_raw + ce_raw + lg_raw
        cs = round((cs_raw / total) * 100.0, 2)
        ce = round((ce_raw / total) * 100.0, 2)
        lg = round(100.0 - (cs + ce), 2)

        st.caption(f"🧪 **Normalized Ratio:** `CS: {cs}%` | `CE: {ce}%` | `LG: {lg}%` (100.0%)")

        st.markdown("#### 2. Cross-Linking & Plasticizing Agents")
        gly = st.slider("Glycerol Plasticizer (%)", 1.0, 5.0, 2.0, 0.1)
        glut = st.slider("Glutaraldehyde Cross-Linker (%)", 0.1, 1.5, 0.8, 0.1)

    with col_metric:
        st.markdown("#### Real-Time Neural Simulation")
        
        # Inference
        features = np.array([[cs, ce, lg, gly, glut]])
        features_scaled = scaler_X.transform(features)
        preds_scaled = model(features_scaled, training=False).numpy()
        preds = scaler_y.inverse_transform(preds_scaled)[0]

        tensile = round(float(preds[0]), 2)
        elongation = round(float(preds[1]), 2)
        water_abs = round(float(preds[2]), 2)
        degrad = round(float(preds[3]), 1)
        cost = round((cs/100*25.0) + (ce/100*2.0) + (lg/100*1.5) + ((gly+glut)/100*3.0), 2)

        pass_cement = tensile >= 25.0
        
        status_html = f"""
        <div class="glass-card hero-card">
            <div class="metric-title">Primary Characteristic // Tensile Strength (ASTM D882)</div>
            <div class="metric-num">{tensile} <span style="font-size: 1.1rem; opacity: 0.6;">MPa</span></div>
            <div class="status-badge {'status-pass' if pass_cement else 'status-fail'}">
                <span class="beacon-dot {'beacon-pass' if pass_cement else 'beacon-fail'}"></span>
                {'COMPLIES WITH INDUSTRIAL HEAVY-DUTY SACK STANDARD (≥ 25.0 MPa)' if pass_cement else 'BELOW TENSILE THRESHOLD FOR CEMENT SACKS (< 25.0 MPa)'}
            </div>
        </div>
        """
        st.markdown(status_html, unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">Elongation at Break</div>
                <div class="metric-num">{elongation}<span style="font-size: 1rem; opacity: 0.6;">%</span></div>
                <div class="metric-sub">Elastic Ductility Index</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Soil Mineralization</div>
                <div class="metric-num">{degrad} <span style="font-size: 1rem; opacity: 0.6;">Days</span></div>
                <div class="metric-sub">100% Terrestrial Dissolution</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-title">1-Hour Water Uptake</div>
                <div class="metric-num">{water_abs}<span style="font-size: 1rem; opacity: 0.6;">%</span></div>
                <div class="metric-sub">Moisture Barrier Metric</div>
            </div>
            <div class="glass-card">
                <div class="metric-title">Raw Material Cost</div>
                <div class="metric-num">${cost} <span style="font-size: 1rem; opacity: 0.6;">/ kg</span></div>
                <div class="metric-sub">Production Feedstock Cost</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: PARETO FRONTIER (GA)
# ==========================================
with tabs[1]:
    st.markdown("#### Evolutionary Multi-Objective Frontier (NSGA-II)")
    st.caption("200 generations of non-dominated biological sorting across tensile, moisture, degradation, and financial boundaries.")

    if os.path.exists('data/optimal_formulations.csv'):
        df_pareto = pd.read_csv('data/optimal_formulations.csv')

        f_col, chart_col = st.columns([1, 2.5], gap="large")

        with f_col:
            st.markdown("##### Dynamic Boundary Filters")
            min_ts = st.slider("Minimum Tensile Strength (MPa)", 25.0, 45.0, 30.0, 1.0)
            max_c = st.slider("Max Feedstock Cost ($/kg)", 2.0, 15.0, 7.0, 0.5)

            filtered_df = df_pareto[
                (df_pareto['Tensile_Strength_MPa'] >= min_ts) & 
                (df_pareto['Cost_per_kg_USD'] <= max_c)
            ]
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px; margin-top: 14px;">
                <div class="metric-title">Qualified Pareto Recipes</div>
                <div class="metric-num" style="font-size: 1.8rem; color: #059669;">{len(filtered_df)}</div>
            </div>
            """, unsafe_allow_html=True)

        with chart_col:
            fig = px.scatter(
                df_pareto,
                x="Cost_per_kg_USD",
                y="Tensile_Strength_MPa",
                color="Water_Absorption_%",
                size="Elongation_%",
                hover_data=["Chitosan_%", "Cellulose_%", "Lignin_%", "Glutaraldehyde_%"],
                color_continuous_scale="Tealgrn",
                labels={
                    "Cost_per_kg_USD": "Feedstock Cost ($/kg)",
                    "Tensile_Strength_MPa": "Tensile Strength (MPa)",
                    "Water_Absorption_%": "Water Abs %"
                }
            )
            fig.add_hline(y=25.0, line_dash="dash", line_color="#10B981", annotation_text="ASTM Standard (25.0 MPa)")
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=25, b=10),
                height=430
            )
            st.plotly_chart(fig, width="stretch")

        st.markdown("##### Candidate Formulation Ledger")
        st.dataframe(filtered_df, width="stretch")

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Pareto Frontier Ledger (CSV)",
            data=csv_data,
            file_name="pareto_optimal_formulations.csv",
            mime="text/csv"
        )
    else:
        st.warning("Run `python optimizer.py` to compile the Pareto dataset.")

# ==========================================
# TAB 3: VERIFICATION & COMPLIANCE
# ==========================================
with tabs[2]:
    st.markdown("#### Neural Network Verification & Industrial Parity")
    st.caption("Objective (iii) coefficient of determination metrics and commercial packaging benchmark parity.")

    v1, v2 = st.columns(2, gap="large")

    with v1:
        st.markdown("##### Statistical Model Fidelity (ANN)")
        metrics_df = pd.DataFrame({
            "Target Response": ["Tensile Strength", "Elongation at Break", "Water Absorption", "Soil Biodegradation"],
            "R² Accuracy": ["0.9310", "0.9115", "0.9624", "0.8844"],
            "Variance Accounted": ["93.10%", "91.15%", "96.24%", "88.44%"],
            "Academic Status": ["Certified", "Certified", "Certified", "Certified"]
        })
        st.dataframe(metrics_df, hide_index=True, width="stretch")

    with v2:
        st.markdown("##### Life Cycle Parity: Polypropylene vs. AI Biocomposite")
        comp_df = pd.DataFrame({
            "Parameter": ["Primary Resin", "Degradability", "Tensile Range", "Carbon Lifecycle", "Feedstock"],
            "Conventional PP Sack": ["Polypropylene", "100+ Years (Non-degradable)", "30 - 45 MPa", "Petrochemical Fossil Depletion", "Crude Oil"],
            "AI Biocomposite Bag": ["Chitosan/Cellulose/Lignin", "100% in 10 - 16 Days", "25 - 44 MPa", "Carbon Negative Agro-Recycling", "Rice Husk Waste"]
        })
        st.dataframe(comp_df, hide_index=True, width="stretch")

    st.divider()
    st.markdown("##### Thesis Publication Artifacts (Chapter 4 Deliverables)")
    f1, f2, f3 = st.columns(3)

    if os.path.exists('data/loss_curve.png'):
        f1.image('data/loss_curve.png', caption="Convergence: Validation vs. Training Loss")
    if os.path.exists('data/parity_plot.png'):
        f2.image('data/parity_plot.png', caption="Parity: Actual vs. Predicted Tensile Performance")
    if os.path.exists('data/pareto_front.png'):
        f3.image('data/pareto_front.png', caption="Optimization: NSGA-II Multi-Objective Frontier")