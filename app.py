import os
import warnings

# Silent background logs
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
    page_title="Biopolymer Packaging Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Responsive Mobile CSS (Restores Sidebar Button, Kills GitHub & Fork)
st.markdown("""
<style>
    /* =========================================================
       1. RESTORE SIDEBAR BUTTON, HIDE GITHUB / FORK / TOOLBAR
       ========================================================= */
    /* Keep the header area invisible but allow the sidebar arrow to show */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Ensure the sidebar toggle button is ALWAYS visible and clickable */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        display: flex !important;
        z-index: 1000001 !important;
    }

    /* Destroy GitHub, Fork, Deploy, and 3-dots menus */
    [data-testid="stToolbar"] {visibility: hidden !important; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden !important; display: none !important;}
    [data-testid="stStatusWidget"] {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden !important; display: none !important;}
    .stDeployButton {display: none !important;}

    /* Destroy "Created by" and bottom GitHub badges */
    footer {visibility: hidden !important; display: none !important;}
    [data-testid="stFooter"] {visibility: hidden !important; display: none !important;}
    div[class*="viewerBadge"] {visibility: hidden !important; display: none !important;}
    div[class*="ProfileBadge"] {visibility: hidden !important; display: none !important;}
    div[class*="StatusWidget"] {visibility: hidden !important; display: none !important;}
    div[class*="manage-app"] {visibility: hidden !important; display: none !important;}
    a[href*="github.com"] {display: none !important;}

    /* =========================================================
       2. MOBILE-FIRST RESPONSIVE CARDS & TYPOGRAPHY
       ========================================================= */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    .clean-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    
    .card-label {
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.75;
        margin-bottom: 2px;
        color: var(--text-color);
    }
    
    .card-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--text-color);
    }
    
    .card-hint {
        font-size: 0.75rem;
        opacity: 0.6;
        margin-top: 2px;
        color: var(--text-color);
    }

    /* Status Badges */
    .badge-pass {
        display: inline-block;
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-fail {
        display: inline-block;
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #FBD5D5;
        border-radius: 4px;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Mobile screen adjustments (max-width 768px) */
    @media (max-width: 768px) {
        .card-value {
            font-size: 1.35rem !important;
        }
        .clean-card {
            padding: 12px 14px !important;
        }
        h1 {
            font-size: 1.6rem !important;
        }
        h2, h3 {
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Trained Model
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/ann_model.keras')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

# --- SIDEBAR (The Left-Hand Dashboard) ---
with st.sidebar:
    st.markdown("### Research Project")
    st.markdown("""
    **Title:**  
    AI-Optimised Lignin-Cellulose-Chitosan Biodegradable Bags for Cement Packaging

    **Candidate:**  
    Ikwuje, Idoko John

    **Institution:**  
    Joseph Sarwuan Tarka University, Makurdi (JOSTUM)
    """)
    st.divider()
    st.markdown("### Target Standards")
    st.markdown("""
    * **Tensile Strength:** ≥ 25.0 MPa (ASTM D882)
    * **Water Absorption:** Minimize (< 100%)
    * **Biodegradation:** 100% loss ≤ 16 days
    """)
    st.divider()
    with st.expander("Install App on Mobile / PC"):
        st.caption("• **Android / PC:** Browser Menu > 'Install App'\n• **iPhone:** Share Button > 'Add to Home Screen'")

# --- MAIN PAGE HEADER ---
st.title("Biopolymer Packaging Optimization")
st.caption("Decision-support system for high-strength biodegradable agricultural-waste cement packaging.")

tabs = st.tabs(["Formulation Simulator", "Optimal Recipes (Pareto)", "Model Performance"])

# ==========================================
# TAB 1: FORMULATION SIMULATOR
# ==========================================
with tabs[0]:
    col_inputs, col_outputs = st.columns([1, 1.2], gap="large")

    with col_inputs:
        st.subheader("1. Polymer Blend")
        cs_raw = st.slider("Chitosan Matrix (%)", 10, 80, 50, 1)
        ce_raw = st.slider("Rice Husk Cellulose (%)", 5, 60, 35, 1)
        lg_raw = st.slider("Rice Husk Lignin (%)", 0, 40, 15, 1)

        total = cs_raw + ce_raw + lg_raw
        cs = round((cs_raw / total) * 100.0, 2)
        ce = round((ce_raw / total) * 100.0, 2)
        lg = round(100.0 - (cs + ce), 2)

        st.info(f"Normalized composition: {cs}% Chitosan, {ce}% Cellulose, {lg}% Lignin (Total: 100%)")

        st.subheader("2. Additives")
        gly = st.slider("Glycerol Plasticizer (%)", 1.0, 5.0, 2.0, 0.1)
        glut = st.slider("Glutaraldehyde Cross-Linker (%)", 0.1, 1.5, 0.8, 0.1)

    with col_outputs:
        st.subheader("Predicted Properties")

        # ANN Prediction
        features = np.array([[cs, ce, lg, gly, glut]])
        features_scaled = scaler_X.transform(features)
        preds_scaled = model(features_scaled, training=False).numpy()
        preds = scaler_y.inverse_transform(preds_scaled)[0]

        tensile = round(float(preds[0]), 2)
        elongation = round(float(preds[1]), 2)
        water_abs = round(float(preds[2]), 2)
        degrad = round(float(preds[3]), 1)
        cost = round((cs/100*25.0) + (ce/100*2.0) + (lg/100*1.5) + ((gly+glut)/100*3.0), 2)

        meets_spec = tensile >= 25.0

        # Primary Metric Card
        badge_html = '<div class="badge-pass">Meets cement packaging standard (≥ 25.0 MPa)</div>' if meets_spec else '<div class="badge-fail">Below required tensile standard (< 25.0 MPa)</div>'
        st.markdown(f"""
        <div class="clean-card">
            <div class="card-label">Tensile Strength (ASTM D882)</div>
            <div class="card-value">{tensile} MPa</div>
            {badge_html}
        </div>
        """, unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="clean-card">
                <div class="card-label">Elongation at Break</div>
                <div class="card-value">{elongation}%</div>
                <div class="card-hint">Material flexibility</div>
            </div>
            <div class="clean-card">
                <div class="card-label">Soil Biodegradation</div>
                <div class="card-value">{degrad} Days</div>
                <div class="card-hint">Time to complete mass loss</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="clean-card">
                <div class="card-label">Water Absorption (1 Hour)</div>
                <div class="card-value">{water_abs}%</div>
                <div class="card-hint">Lower indicates better barrier</div>
            </div>
            <div class="clean-card">
                <div class="card-label">Estimated Production Cost</div>
                <div class="card-value">${cost} / kg</div>
                <div class="card-hint">Raw material feedstock cost</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: OPTIMAL RECIPES (PARETO)
# ==========================================
with tabs[1]:
    st.subheader("AI-Optimized Formulation Candidates")
    st.caption("Formulations selected by the multi-objective genetic algorithm (NSGA-II) across 200 generations.")

    if os.path.exists('data/optimal_formulations.csv'):
        df_pareto = pd.read_csv('data/optimal_formulations.csv')

        f_col, chart_col = st.columns([1, 2.5], gap="large")

        with f_col:
            min_ts = st.slider("Minimum Tensile Strength (MPa)", 25.0, 45.0, 30.0, 1.0)
            max_c = st.slider("Maximum Cost ($/kg)", 2.0, 15.0, 7.0, 0.5)

            filtered_df = df_pareto[
                (df_pareto['Tensile_Strength_MPa'] >= min_ts) & 
                (df_pareto['Cost_per_kg_USD'] <= max_c)
            ]
            st.metric("Qualified Recipes", len(filtered_df))

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
                    "Cost_per_kg_USD": "Material Cost ($/kg)",
                    "Tensile_Strength_MPa": "Tensile Strength (MPa)",
                    "Water_Absorption_%": "Water Absorption (%)"
                }
            )
            fig.add_hline(y=25.0, line_dash="dash", line_color="#10B981", annotation_text="Standard Benchmark (25.0 MPa)")
            fig.update_layout(margin=dict(l=10, r=10, t=25, b=10), height=380)
            st.plotly_chart(fig, width="stretch")

        st.markdown("#### Top Recommended Formulations")
        st.dataframe(filtered_df, width="stretch")

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Formulations Table (CSV)",
            data=csv_data,
            file_name="optimal_formulations.csv",
            mime="text/csv"
        )
    else:
        st.warning("Pareto data not found. Run optimizer.py to generate it.")

# ==========================================
# TAB 3: MODEL PERFORMANCE & VALIDATION
# ==========================================
with tabs[2]:
    st.subheader("Model Validation & Industry Comparison")
    st.caption("Verification metrics required for Chapter 4 and Chapter 5.")

    v1, v2 = st.columns(2, gap="large")

    with v1:
        st.markdown("#### Neural Network Accuracy")
        metrics_df = pd.DataFrame({
            "Property": ["Tensile Strength", "Elongation at Break", "Water Absorption", "Biodegradation Days"],
            "R² Accuracy": ["0.9310", "0.9115", "0.9624", "0.8844"],
            "Quality Assessment": ["High Accuracy (>0.90)", "High Accuracy (>0.90)", "Exceptional (>0.95)", "Robust (>0.85)"]
        })
        st.dataframe(metrics_df, hide_index=True, width="stretch")

    with v2:
        st.markdown("#### Comparison with Polypropylene (PP) Sacks")
        comp_df = pd.DataFrame({
            "Feature": ["Base Material", "Degradability", "Tensile Strength", "Environmental Impact"],
            "Standard PP Sack": ["Polypropylene Plastic", "Non-biodegradable (100+ years)", "30 - 45 MPa", "Microplastic waste"],
            "Biopolymer Bag": ["Chitosan / Cellulose / Lignin", "100% loss in 10 - 16 days", "25 - 44 MPa", "Zero microplastics, fully compostable"]
        })
        st.dataframe(comp_df, hide_index=True, width="stretch")

    st.divider()
    st.markdown("#### Thesis Publication Figures")
    f1, f2, f3 = st.columns(3)

    if os.path.exists('data/loss_curve.png'):
        f1.image('data/loss_curve.png', caption="Figure 1: Training Loss Convergence")
    if os.path.exists('data/parity_plot.png'):
        f2.image('data/parity_plot.png', caption="Figure 2: Measured vs Predicted Tensile Strength")
    if os.path.exists('data/pareto_front.png'):
        f3.image('data/pareto_front.png', caption="Figure 3: Pareto Frontier Trade-Off")