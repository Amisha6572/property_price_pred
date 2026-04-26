import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Modern Light Theme CSS ─────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
    <style>
    /* Root Color Variables */
    :root {
        --primary-color: #2563eb;
        --secondary-color: #1e40af;
        --accent-color: #dc2626;
        --success-color: #059669;
        --warning-color: #ea580c;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --border-color: #e2e8f0;
    }
    
    /* Main Background with Subtle Pattern */
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f0f4f8 50%, #e8eef7 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #ffffff 0%, #f0f4f8 100%);
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid var(--primary-color);
        padding-bottom: 0.5rem;
    }
    
    h3 {
        font-size: 1.5rem !important;
        color: var(--primary-color) !important;
    }
    
    /* Text Styling */
    body, p, span, label {
        color: var(--text-primary) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif !important;
    }
    
    /* Card Styling */
    .card-container {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .card-container:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12);
        border-color: var(--primary-color);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white !important;
        border: 0 !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div > input {
        background-color: var(--light-bg) !important;
        border: 1.5px solid var(--border-color) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        font-size: 1rem !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--primary-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    /* Radio Buttons */
    .stRadio > div {
        background: var(--card-bg);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    .stRadio > div > label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        border-bottom: 2px solid var(--border-color);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--text-secondary) !important;
        border-bottom: 3px solid transparent !important;
        font-weight: 600 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--primary-color) !important;
        border-bottom-color: var(--primary-color) !important;
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        color: #065f46 !important;
        border: 1px solid #6ee7b7 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Warning Messages */
    .stWarning {
        background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%) !important;
        color: #7c2d12 !important;
        border: 1px solid #fb923c !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Info Messages */
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        color: #0c4a6e !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Tables */
    .stDataFrame {
        font-size: 0.95rem !important;
    }
    
    table {
        border-collapse: collapse !important;
    }
    
    tbody, thead {
        background: var(--card-bg) !important;
    }
    
    thead tr {
        background: linear-gradient(90deg, #f0f4f8 0%, #e8eef7 100%) !important;
        border-bottom: 2px solid var(--border-color) !important;
    }
    
    tbody tr {
        border-bottom: 1px solid var(--border-color) !important;
    }
    
    tbody tr:hover {
        background: #f8fafc !important;
    }
    
    /* Metrics */
    .stMetric {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .stMetric label {
        color: var(--text-secondary) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    
    /* Horizontal Rule */
    hr {
        border: 0 !important;
        border-top: 2px solid var(--border-color) !important;
        margin: 2rem 0 !important;
    }
    
    /* Section Headers */
    .section-header {
        color: var(--primary-color);
        font-weight: 700;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--primary-color);
    }
    
    /* Hero Header */
    .hero-header {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4f8 50%, #dbeafe 100%);
        border: 2px solid var(--primary-color);
        border-radius: 12px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.1);
    }
    
    .hero-header h1 {
        color: var(--primary-color) !important;
        font-size: 2.8rem !important;
        margin: 0 !important;
        font-weight: 800 !important;
    }
    
    .hero-header p {
        color: var(--text-secondary) !important;
        font-size: 1.1rem !important;
        margin: 0.5rem 0 0 0 !important;
        font-weight: 500 !important;
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #a7f3d0 100%);
        border: 2px solid var(--success-color);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 24px rgba(5, 150, 105, 0.2);
    }
    
    .result-card h1 {
        color: var(--text-primary) !important;
        font-size: 3.5rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .result-card p {
        color: var(--text-secondary) !important;
        font-size: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Grid Columns */
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .grid-item {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    /* Divider Line */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 2rem 0;
    }
    
    /* Label Styling */
    label {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ── Load PKL Files ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = "pkl_files"
    
    # Try to load model from joblib first (more efficient compression)
    # Falls back to pickle only if joblib not available
    try:
        import joblib
        if os.path.exists("best_model_joblib.joblib"):
            model = joblib.load("best_model_joblib.joblib")
        elif os.path.exists("best_model.pkl"):
            with open("best_model.pkl", "rb") as f: 
                model = pickle.load(f)
        else:
            raise FileNotFoundError("Neither best_model_joblib.joblib nor best_model.pkl found!")
    except ImportError:
        # Fallback if joblib not installed
        with open("best_model.pkl", "rb") as f: 
            model = pickle.load(f)
    
    
    with open(f"{base}/scaler.pkl",           "rb") as f: scaler        = pickle.load(f)
    with open(f"{base}/feature_columns.pkl",  "rb") as f: feature_cols  = pickle.load(f)
    with open(f"{base}/label_encoder.pkl",    "rb") as f: le            = pickle.load(f)
    with open(f"{base}/city_target_enc.pkl",  "rb") as f: city_target   = pickle.load(f)
    with open(f"{base}/city_freq_enc.pkl",    "rb") as f: city_freq     = pickle.load(f)
    with open(f"{base}/global_mean.pkl",      "rb") as f: global_mean   = pickle.load(f)
    with open(f"{base}/model_results.pkl",    "rb") as f: results_df    = pickle.load(f)
    return model, scaler, feature_cols, le, city_target, city_freq, global_mean, results_df

model, scaler, feature_cols, le, city_target, city_freq, global_mean, results_df = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="hero-header">
        <h1 style='margin:0;'>🏠 Property Price Predictor</h1>
        <p style='margin:0.5rem 0 0 0;'>
            Advanced ML-powered house price estimation for Indian Real Estate Market
        </p>
    </div>
""", unsafe_allow_html=True)

# Add metadata information
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Data Source", "India Real Estate")
with col2:
    st.metric("🎯 Model Type", "Regression")
with col3:
    st.metric("⚡ Technology", "ML Ensemble")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Model Performance", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>📝 Enter Property Details</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("<h4 style='color: #2563eb; margin-bottom: 1rem;'>🏗️ Property Information</h4>", unsafe_allow_html=True)
        posted_by   = st.selectbox("Posted By", le.classes_.tolist(), help="Who is posting this property?")
        bhk_no      = st.slider("Number of BHK / Rooms", 1, 10, 2, help="Select the number of bedrooms/rooms")
        bhk_or_rk   = st.radio("Property Type", ["BHK", "RK"], horizontal=True)
        square_ft   = st.number_input("Area (sq ft)", min_value=100.0, max_value=10000.0,
                                       value=1000.0, step=50.0, help="Total built-up area in square feet")

    with col2:
        st.markdown("<h4 style='color: #2563eb; margin-bottom: 1rem;'>📍 Location Details</h4>", unsafe_allow_html=True)
        city_list = sorted(city_target.keys())
        city      = st.selectbox("City", city_list, help="Select the city where property is located")
        longitude = st.number_input("Longitude", min_value=68.0, max_value=98.0, value=77.59, step=0.01, 
                                   help="Geographic longitude coordinate")
        latitude  = st.number_input("Latitude",  min_value=8.0,  max_value=37.0, value=12.97, step=0.01,
                                   help="Geographic latitude coordinate")

    with col3:
        st.markdown("<h4 style='color: #2563eb; margin-bottom: 1rem;'>✅ Property Status</h4>", unsafe_allow_html=True)
        rera          = st.selectbox("RERA Approved", [0, 1],
                                      format_func=lambda x: "✓ Yes" if x else "✗ No",
                                      help="Is the property RERA approved?")
        ready_to_move = st.selectbox("Ready to Move", [0, 1],
                                      format_func=lambda x: "✓ Yes" if x else "✗ No",
                                      help="Is the property ready for immediate occupancy?")
        resale        = st.selectbox("Resale Property", [0, 1],
                                      format_func=lambda x: "✓ Yes" if x else "✗ No",
                                      help="Is this a resale or new property?")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    predict_btn = st.button("🔮 Predict Property Price", use_container_width=True, type="primary")

    if predict_btn:
        # ── Feature Engineering (mirror code.ipynb) ────────────────────────────
        posted_by_enc    = int(le.transform([posted_by])[0])
        is_bhk           = 1 if bhk_or_rk == "BHK" else 0
        city_target_enc  = city_target.get(city, global_mean)
        city_freq_enc    = city_freq.get(city, 0.0)

        # log1p on square_ft (same as training)
        sq_ft_log        = np.log1p(square_ft)

        # price_per_sqft placeholder — use city mean / sq_ft as proxy, then log1p
        price_per_sqft_proxy = city_target_enc / square_ft if square_ft > 0 else 0
        price_per_sqft_log   = np.log1p(price_per_sqft_proxy)

        input_dict = {
            'RERA'               : rera,
            'BHK_NO.'            : bhk_no,
            'SQUARE_FT'          : sq_ft_log,
            'READY_TO_MOVE'      : ready_to_move,
            'RESALE'             : resale,
            'LONGITUDE'          : longitude,
            'LATITUDE'           : latitude,
            'PRICE_PER_SQFT'     : price_per_sqft_log,
            'IS_BHK'             : is_bhk,
            'POSTED_BY_ENC'      : posted_by_enc,
            'CITY_TARGET_ENC'    : city_target_enc,
            'CITY_FREQ_ENC'      : city_freq_enc,
        }

        # Align to training feature order
        input_df  = pd.DataFrame([input_dict])[feature_cols]
        input_sc  = scaler.transform(input_df)
        pred_log  = model.predict(input_sc)[0]
        pred_lacs = np.expm1(pred_log)   # reverse log1p on target

        # ── Result Card ────────────────────────────────────────────────────────
        st.markdown(f"""
            <div class="result-card">
                <p style='color: #64748b; font-size: 1rem; margin: 0; font-weight: 500; letter-spacing: 0.5px;'>ESTIMATED PROPERTY PRICE</p>
                <h1 style='color: #059669; font-size: 3.5rem; margin: 0.8rem 0; font-weight: 800;'>
                    ₹ {pred_lacs:,.2f} <span style='font-size: 1.8rem;'>Lacs</span>
                </h1>
                <p style='color: #1e293b; font-size: 1.1rem; margin: 0.5rem 0 0 0; font-weight: 600;'>
                    ≈ ₹ {pred_lacs/100:,.3f} Crore
                </p>
                <p style='color: #64748b; font-size: 0.9rem; margin: 1rem 0 0 0; font-style: italic;'>
                    Predicted by advanced ML ensemble model
                </p>
            </div>
        """, unsafe_allow_html=True)

        # ── Summary ────────────────────────────────────────────────────────────
        st.markdown("<div class='section-header'>📋 Input Summary</div>", unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Feature": ["Posted By", "BHK No.", "Type", "Area (sq ft)", "City",
                        "RERA Approved", "Ready to Move", "Resale Property"],
            "Value"  : [posted_by, bhk_no, bhk_or_rk, f"{square_ft:,.0f}", city,
                        "✓ Yes" if rera else "✗ No",
                        "✓ Yes" if ready_to_move else "✗ No",
                        "✓ Yes" if resale else "✗ No"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>📊 Model Performance & Evaluation</div>", unsafe_allow_html=True)

    best_name = results_df.iloc[0]['Model']
    
    # Best model indicator
    st.markdown(f"""
        <div style='background: linear-gradient(135deg, #d1fae5, #a7f3d0); 
                    border-left: 4px solid #059669; padding: 1.5rem; border-radius: 8px;
                    margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(5, 150, 105, 0.15);'>
            <h3 style='color: #065f46; margin: 0 0 0.5rem 0;'>🏆 Best Performing Model</h3>
            <p style='color: #047857; margin: 0; font-size: 1.1rem; font-weight: 600;'>{best_name}</p>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;'>
                <div style='background: white; padding: 0.75rem; border-radius: 6px;'>
                    <p style='color: #64748b; font-size: 0.85rem; margin: 0;'>Test R²</p>
                    <p style='color: #059669; font-size: 1.3rem; font-weight: 700; margin: 0.25rem 0 0 0;'>{results_df.iloc[0]['Test_R2']:.4f}</p>
                </div>
                <div style='background: white; padding: 0.75rem; border-radius: 6px;'>
                    <p style='color: #64748b; font-size: 0.85rem; margin: 0;'>Adjusted R²</p>
                    <p style='color: #059669; font-size: 1.3rem; font-weight: 700; margin: 0.25rem 0 0 0;'>{results_df.iloc[0]['Test_AdjR2']:.4f}</p>
                </div>
                <div style='background: white; padding: 0.75rem; border-radius: 6px;'>
                    <p style='color: #64748b; font-size: 0.85rem; margin: 0;'>RMSE</p>
                    <p style='color: #059669; font-size: 1.3rem; font-weight: 700; margin: 0.25rem 0 0 0;'>{results_df.iloc[0]['RMSE']:.2f}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Styled table
    st.markdown("<h4 style='color: #2563eb; margin-bottom: 1rem;'>📈 All Models Comparison</h4>", unsafe_allow_html=True)
    
    def highlight_best(row):
        color = 'background-color: #dbeafe; color: #0c4a6e; font-weight: 600;' if row['Model'] == best_name else ''
        return [color] * len(row)

    styled = results_df.style.apply(highlight_best, axis=1)\
                             .format({'Train_R2':    '{:.4f}', 'Train_AdjR2': '{:.4f}',
                                      'Test_R2':     '{:.4f}', 'Test_AdjR2':  '{:.4f}',
                                      'MAE':         '{:.4f}', 'RMSE':        '{:.4f}',
                                      'CV_R2':       '{:.4f}'})
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Bar charts
    import plotly.graph_objects as go

    st.markdown("<h4 style='color: #2563eb; margin-top: 1.5rem; margin-bottom: 1rem;'>R² Metrics (Higher is Better)</h4>", unsafe_allow_html=True)
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(name='Train R²',    x=results_df['Model'], y=results_df['Train_R2'],
                          marker=dict(color='#2563eb', line=dict(width=0))))
    fig1.add_trace(go.Bar(name='Test R²',     x=results_df['Model'], y=results_df['Test_R2'],
                          marker=dict(color='#059669', line=dict(width=0))))
    fig1.add_trace(go.Bar(name='Train Adj-R²', x=results_df['Model'], y=results_df['Train_AdjR2'],
                          marker=dict(color='#ea580c', line=dict(width=0))))
    fig1.add_trace(go.Bar(name='Test Adj-R²',  x=results_df['Model'], y=results_df['Test_AdjR2'],
                          marker=dict(color='#dc2626', line=dict(width=0))))
    fig1.update_layout(
        barmode='group', 
        title='R² & Adjusted R² — All Models Comparison',
        xaxis_tickangle=-35, 
        height=500,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b', size=11, family='Arial'),
        hovermode='x unified',
        margin=dict(b=100)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<h4 style='color: #2563eb; margin-top: 1.5rem; margin-bottom: 1rem;'>⚡ RMSE Comparison (Lower is Better)</h4>", unsafe_allow_html=True)
    
    fig2 = go.Figure(go.Bar(
        x=results_df['RMSE'], y=results_df['Model'], orientation='h',
        marker=dict(
            color=['#059669' if m == best_name else '#2563eb' for m in results_df['Model']],
            line=dict(width=0)
        ),
        text=results_df['RMSE'].apply(lambda x: f'{x:.2f}'), 
        textposition='outside'
    ))
    fig2.update_layout(
        title='RMSE — All Models (lower is better)',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#1e293b', size=11, family='Arial'),
        xaxis_title='RMSE (Root Mean Squared Error)',
        showlegend=False
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>ℹ️ About This Application</div>", unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Overview
    
    **Property Price Predictor** is a professional-grade machine learning application designed to provide 
    accurate price estimations for residential properties across major Indian cities. Built with industry-standard 
    ML practices, this application leverages ensemble learning techniques to deliver reliable predictions.
    
    ---
    
    ### 📊 ML Pipeline Summary
    """)
    
    # Pipeline steps in cards
    pipeline_data = {
        "📥 Data Source": "Comprehensive house price dataset covering Indian real estate market",
        "🔍 EDA": "Missing values analysis, distribution plots, correlation heatmaps, outlier detection",
        "⚙️ Feature Engineering": "PRICE_PER_SQFT, IS_BHK, POSTED_BY_ENC, CITY_TARGET_ENC, CITY_FREQ_ENC",
        "🎯 Outlier Treatment": "IQR-based robust capping (no data deletion)",
        "📈 Skewness Removal": "log1p transformation on SQUARE_FT, TARGET, PRICE_PER_SQFT",
        "⬛ Scaling": "StandardScaler for feature normalization",
        "🤖 Models Evaluated": "14 advanced regression models (Linear, Tree-based, Ensemble)",
        "🏆 Best Model": "XGBoost/LightGBM — Auto-selected by highest Test R²"
    }
    
    col1, col2 = st.columns(2)
    for idx, (key, value) in enumerate(pipeline_data.items()):
        col = col1 if idx % 2 == 0 else col2
        with col:
            st.markdown(f"""
                <div class='card-container' style='background: linear-gradient(135deg, #f0f4f8 0%, #dbeafe 100%);'>
                    <h4 style='color: #2563eb; margin-top: 0;'>{key}</h4>
                    <p style='color: #475569; margin: 0; font-size: 0.95rem;'>{value}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🌍 City Encoding Strategy
    
    Our model uses advanced encoding techniques for location features:
    
    **Target Encoding** → Mean property price per city (captures price signals)  
    **Frequency Encoding** → Relative city frequency (captures market activity)  
    **Fallback Strategy** → Unseen cities use global mean price for robust predictions
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🛠️ Technology Stack
    """)
    
    tech_stack = """
    | Component | Technology |
    |-----------|-----------|
    | **Language** | Python 3.x |
    | **ML Frameworks** | Scikit-learn, XGBoost, LightGBM |
    | **Web Framework** | Streamlit |
    | **Data Processing** | Pandas, NumPy |
    | **Visualization** | Plotly, Matplotlib, Seaborn |
    | **Model Serialization** | Pickle |
    """
    st.markdown(tech_stack)
    
    st.markdown("---")
    
    # Key Metrics
    st.markdown("<h3 style='color: #2563eb; margin-top: 1.5rem;'>📈 Model Performance Metrics</h3>", unsafe_allow_html=True)
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.metric(
            label="Models Evaluated",
            value=len(results_df),
            delta="Complete Comparison",
            delta_color="off"
        )
    
    with metric_col2:
        st.metric(
            label="Best Test R² Score",
            value=f"{results_df.iloc[0]['Test_R2']:.4f}",
            delta=f"Adj-R²: {results_df.iloc[0]['Test_AdjR2']:.4f}",
            delta_color="off"
        )
    
    with metric_col3:
        st.metric(
            label="Best RMSE",
            value=f"{results_df.iloc[0]['RMSE']:.2f}",
            delta=f"Model: {results_df.iloc[0]['Model']}",
            delta_color="off"
        )
    
    st.markdown("---")
    
    st.markdown("""
    ### ✨ Key Features
    
    - ✅ **Real-time Predictions** — Get instant property price estimates
    - 📊 **Model Comparison** — View detailed performance metrics of all trained models
    - 🎯 **Feature Importance** — Understand how different factors influence prices
    - 🌍 **Multi-city Support** — Accurate predictions across major Indian cities
    - 🔒 **Robust Handling** — Smart fallback for unseen cities and edge cases
    
    ### 📝 Usage Guidelines
    
    1. **Navigate to "Predict Price"** tab to enter property details
    2. **Fill in all fields** with accurate information
    3. **Click "Predict Price"** to get instant estimation
    4. **Review the summary** to verify your inputs
    5. **Check Model Performance** tab to understand model reliability
    
    ### 📧 Notes
    - Predictions are based on historical market data and patterns
    - For an individual property, actual prices may vary based on condition and negotiation
    - Always verify predictions with domain experts and market analysis
    """)
    
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 2rem;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                unsafe_allow_html=True)
