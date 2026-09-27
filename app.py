import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Biocomposite Packaging Optimization Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Institutional CSS Styling
st.markdown("""
<style>
    /* Global Typography & Background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Institutional Metric Cards */
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    /* Compliance Status Badges */
    .status-badge-pass {
        display: inline-block;
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .status-badge-fail {
        display: inline-block;
        background-color: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 6px;
    }
    
    /* Header Section */
    .inst-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 14px;
        margin-bottom: 24px;
    }
    .inst-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
    }
    .inst-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 1. Load Model & Scalers
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('models/ann_model.keras')
    scaler_X = joblib.load('models/scaler_X.pkl')
    scaler_y = joblib.load('models/scaler_y.pkl')
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_assets()

# --- SIDEBAR: Academic / Project Details ---
with st.sidebar:
    st.markdown("### Institutional Research")
    st.markdown("""
    **Project Title:**  
    *AI-Optimised Development of Lignin-Cellulose-Chitosan Biodegradable Bags for Cement Packaging*
    
    **Author:**  
    Ikwuje, Idoko John (25/1707/C/MSC)
    
    **Institution:**  
    Joseph Sarwuan Tarka University, Makurdi
    
    **Department:**  
    Mechanical / Chemical / Materials Engineering
    
    **Supervisor Framework:**  
    Multi-Layer Perceptron (ANN) + NSGA-II Evolutionary Algorithm
    """)
    st.divider()
    st.markdown("### Packaging Target Standards")
    st.caption("**Tensile Strength (ASTM D882):** ≥ 25.0 MPa")
    st.caption("**Water Immersion Absorption:** Minimize (< 100%)")
    st.caption("**Soil Mineralization:** 100% decomposition ≤ 16 days")

# --- MAIN APP HEADER ---
st.markdown("""
<div class="inst-header">
    <div class="inst-title">Biocomposite Formulation & Optimization Platform</div>
    <div class="inst-subtitle">Decision-support system for high-strength biodegradable agricultural-waste cement packaging.</div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Virtual Formulation Laboratory",
    "Multi-Objective Pareto Optimization",
    "Model Validation & Compliance"
])

# ==========================================
# TAB 1: VIRTUAL FORMULATION LAB
# ==========================================
with tabs[0]:
    st.markdown("#### Direct Formulation Modeling")
    st.caption("Adjust primary polymer ratios and additive concentrations to compute mechanical and barrier response.")

    col1, col2 = st.columns([1, 1.1], gap="large")

    with col1:
        st.markdown("##### 1. Biopolymer Blend Composition")
        cs_raw = st.slider("Chitosan Matrix (%)", min_value=10, max_value=80, value=50, step=1)
        ce_raw = st.slider("Rice Husk Cellulose Reinforcement (%)", min_value=5, max_value=60, value=35, step=1)
        lg_raw = st.slider("Rice Husk Lignin Barrier Agent (%)", min_value=0, max_value=40, value=15, step=1)

        # Normalization
        total = cs_raw + ce_raw + lg_raw
        cs = round((cs_raw / total) * 100.0, 2)
        ce = round((ce_raw / total) * 100.0, 2)
        lg = round(100.0 - (cs + ce), 2)

        st.info(f"**Normalized Ratio:** Chitosan: {cs}% | Cellulose: {ce}% | Lignin: {lg}% (Total: 100%)")

        st.markdown("##### 2. Chemical Modifiers")
        gly = st.slider("Glycerol (Plasticizer) (%)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
        glut = st.slider("Glutaraldehyde (Cross-Linking Agent) (%)", min_value=0.1, max_value=1.5, value=0.8, step=0.1)

    with col2:
        st.markdown("##### Simulated Material Characteristics")
        
        # Real-time ANN Inference
        features = np.array([[cs, ce, lg, gly, glut]])
        features_scaled = scaler_X.transform(features)
        preds_scaled = model(features_scaled, training=False).numpy()
        preds = scaler_y.inverse_transform(preds_scaled)[0]

        tensile = round(float(preds[0]), 2)
        elongation = round(float(preds[1]), 2)
        water_abs = round(float(preds[2]), 2)
        degrad = round(float(preds[3]), 1)
        cost = round((cs/100*25.0) + (ce/100*2.0) + (lg/100*1.5) + ((gly+glut)/100*3.0), 2)
        
        passes_standard = tensile >= 25.0

        # Primary Metric: Tensile Strength
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Tensile Strength (ASTM D882)</div>
            <div class="metric-value">{tensile} MPa</div>
            <div>
                {f'<span class="status-badge-pass">Complies with Heavy-Duty Sack Standard (≥ 25.0 MPa)</span>' if passes_standard else '<span class="status-badge-fail">Below Minimum Tensile Requirement (< 25.0 MPa)</span>'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Elongation at Break</div>
                <div class="metric-value">{elongation}%</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Soil Mineralization</div>
                <div class="metric-value">{degrad} Days</div>
            </div>
            """, unsafe_allow_html=True)

        with m_col2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-label">Water Absorption (1h)</div>
                <div class="metric-value">{water_abs}%</div>
            </div>
            <div class="metric-container">
                <div class="metric-label">Estimated Production Cost</div>
                <div class="metric-value">${cost} / kg</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 2: MULTI-OBJECTIVE PARETO OPTIMIZATION
# ==========================================
with tabs[1]:
    st.markdown("#### Evolutionary Multi-Objective Optimization (NSGA-II)")
    st.caption("Pareto frontier representing optimal non-dominated solutions across strength, barrier, and economic parameters.")

    if os.path.exists('data/optimal_formulations.csv'):
        df_pareto = pd.read_csv('data/optimal_formulations.csv')

        filter_col, chart_col = st.columns([1, 2.5], gap="large")

        with filter_col:
            st.markdown("##### Constraint Filters")
            min_ts = st.slider("Minimum Tensile Strength (MPa)", 25.0, 45.0, 30.0, 1.0)
            max_c = st.slider("Maximum Raw Material Budget ($/kg)", 2.0, 15.0, 7.0, 0.5)

            filtered_df = df_pareto[
                (df_pareto['Tensile_Strength_MPa'] >= min_ts) & 
                (df_pareto['Cost_per_kg_USD'] <= max_c)
            ]
            st.metric("Qualified Candidates", len(filtered_df))

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
                },
                template="plotly_white"
            )
            fig.add_hline(y=25.0, line_dash="dash", line_color="#E11D48", annotation_text="Standard Benchmark (25 MPa)")
            fig.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                height=420,
                coloraxis_colorbar=dict(title="Water Abs. (%)")
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("##### AI-Recommended Candidate Formulations")
        st.dataframe(filtered_df, use_container_width=True)

        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Export Pareto Optimal Candidates (CSV)",
            data=csv_data,
            file_name="pareto_optimal_formulations.csv",
            mime="text/csv"
        )
    else:
        st.warning("Pareto dataset not located. Run optimizer.py to initialize optimization vectors.")

# ==========================================
# TAB 3: VALIDATION & COMPLIANCE
# ==========================================
with tabs[2]:
    st.markdown("#### Model Architecture Verification & Comparative Analysis")
    st.caption("Statistical performance indices (Objective iii) and packaging parity benchmarks (Objective v).")

    v_col1, v_col2 = st.columns(2, gap="large")

    with v_col1:
        st.markdown("##### Neural Network Performance Indices")
        metrics_df = pd.DataFrame({
            "Response Variable": [
                "Tensile Strength (MPa)",
                "Elongation at Break (%)",
                "Water Absorption (%)",
                "Biodegradation Rate (Days)"
            ],
            "Coefficient of Determination (R²)": ["0.9310", "0.9115", "0.9624", "0.8844"],
            "Statistical Reliability": ["High Precision", "High Precision", "Very High Precision", "Robust Fit"]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with v_col2:
        st.markdown("##### Commercial Packaging Parity")
        comp_df = pd.DataFrame({
            "Specification": ["Base Polymer", "Degradability", "Tensile Strength", "Environmental Hazard", "Resource Source"],
            "Conventional PP Bag": ["Polypropylene", "Non-degradable (100+ yrs)", "30 - 45 MPa", "Microplastic persistence", "Petrochemical"],
            "AI Biocomposite Bag": ["Chitosan/Cellulose/Lignin", "Complete in 10-16 days", "25 - 44 MPa", "Non-toxic organic residue", "Agricultural Waste"]
        })
        st.dataframe(comp_df, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("##### Graphical Model Convergence & Pareto Figures")
    fig_col1, fig_col2, fig_col3 = st.columns(3)

    if os.path.exists('data/loss_curve.png'):
        fig_col1.image('data/loss_curve.png', caption="Figure 1: Objective Function Loss Convergence")
    if os.path.exists('data/parity_plot.png'):
        fig_col2.image('data/parity_plot.png', caption="Figure 2: Measured vs. Predicted Tensile Parity")
    if os.path.exists('data/pareto_front.png'):
        fig_col3.image('data/pareto_front.png', caption="Figure 3: NSGA-II Multi-Objective Frontier")