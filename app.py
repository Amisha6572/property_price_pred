# app.py — House Price Prediction · Professional Streamlit App
# Environment: ml_env
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PropValue AI — House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp { background: #0d1117; color: #e6edf3; }

/* ── ALL widget labels — make them clearly visible ── */
label, .stSelectbox label, .stSlider label,
.stNumberInput label, .stRadio label,
.stMultiSelect label, .stTextInput label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stSlider [data-testid="stWidgetLabel"],
div[data-testid="stVerticalBlock"] label {
    color: #e6edf3 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

/* ── Help tooltip icon ── */
.stTooltipIcon { color: #58a6ff !important; }

/* ── Radio button labels ── */
.stRadio > div > label { color: #e6edf3 !important; }
.stRadio > div > label > div > p { color: #e6edf3 !important; }

/* ── Slider value label ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: #a0aab4 !important; }

/* ── Selectbox text ── */
.stSelectbox div[data-baseweb="select"] span { color: #e6edf3 !important; }

/* ── Number input ── */
.stNumberInput input { color: #e6edf3 !important; background: #161b22 !important; }

/* ── Markdown bold text ── */
.stMarkdown strong, .stMarkdown b { color: #e6edf3 !important; }
.stMarkdown p { color: #c9d1d9 !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #e6edf3 !important;
}
[data-testid="stSidebar"] .stMarkdown p { color: #c9d1d9 !important; }
[data-testid="stSidebar"] h4 { color: #e6edf3 !important; }

/* ── Caption / small text ── */
.stCaption, [data-testid="stCaptionContainer"] p { color: #a0aab4 !important; }

/* ── Metric Cards ── */
.metric-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: #58a6ff; }
.metric-card .label { color: #a0aab4; font-size: 0.78rem; font-weight: 600;
                       text-transform: uppercase; letter-spacing: 0.05em; }
.metric-card .value { color: #58a6ff; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
.metric-card .sub   { color: #a0aab4; font-size: 0.75rem; }

/* ── Prediction Result ── */
.pred-card {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%);
    border: 2px solid #238636;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(35,134,54,0.15);
    margin: 1rem 0;
}
.pred-card .price-label { color: #a0aab4; font-size: 0.9rem; letter-spacing: 0.1em; text-transform: uppercase; }
.pred-card .price-main  { color: #3fb950; font-size: 3.5rem; font-weight: 800; margin: 0.3rem 0; line-height: 1; }
.pred-card .price-crore { color: #58a6ff; font-size: 1.2rem; font-weight: 500; }
.pred-card .price-range { color: #c9d1d9; font-size: 0.85rem; margin-top: 0.5rem; }

/* ── Section Headers ── */
.section-header {
    border-left: 4px solid #58a6ff;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e6edf3;
}

/* ── Info Box ── */
.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #c9d1d9;
}

/* ── Feature Tag ── */
.feature-tag {
    display: inline-block;
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.78rem;
    color: #58a6ff;
    margin: 0.15rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #c9d1d9; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #1c2128 !important; color: #58a6ff !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; font-size: 1rem;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(35,134,54,0.4); }

/* ── Selectbox / dropdown ── */
.stSelectbox > div > div {
    background: #161b22 !important;
    border-color: #30363d !important;
    color: #e6edf3 !important;
}

/* ── Divider ── */
hr { border-color: #30363d; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] { background: #1f6feb !important; color: #e6edf3 !important; }
[data-baseweb="tag"] span { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML artifacts...")
def load_artifacts():
    model        = joblib.load("best_model_joblib.joblib")
    with open("scaler.pkl",          "rb") as f: scaler       = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f: feat_cols    = pickle.load(f)
    with open("label_encoder.pkl",   "rb") as f: le           = pickle.load(f)
    with open("city_target_enc.pkl", "rb") as f: city_target  = pickle.load(f)
    with open("city_freq_enc.pkl",   "rb") as f: city_freq    = pickle.load(f)
    with open("global_mean.pkl",     "rb") as f: global_mean  = pickle.load(f)
    with open("model_results.pkl",   "rb") as f: results_df   = pickle.load(f)
    return model, scaler, feat_cols, le, city_target, city_freq, global_mean, results_df

model, scaler, feat_cols, le, city_target, city_freq, global_mean, results_df = load_artifacts()

best_name    = results_df.iloc[0]['Model']
best_test_r2 = results_df.iloc[0]['Test_R2']
best_rmse    = results_df.iloc[0]['RMSE']
best_mae     = results_df.iloc[0]['MAE']
city_list    = sorted(city_target.keys())

# ── Plotly dark theme helper ───────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c9d1d9', family='Inter'),
    xaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    yaxis=dict(gridcolor='#21262d', linecolor='#30363d'),
    margin=dict(l=10, r=10, t=40, b=10)
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:3rem;'>🏠</div>
        <div style='font-size:1.3rem; font-weight:700; color:#e6edf3;'>PropValue AI</div>
        <div style='font-size:0.78rem; color:#c9d1d9; margin-top:0.2rem;'>
            House Price Prediction Engine
        </div>
    </div>
    <hr style='border-color:#30363d; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧠 Model Info")
    st.markdown(f"""
    <div class='info-box'>
        <b style='color:#e6edf3;'>Active Model</b><br>
        <span style='color:#3fb950; font-weight:600;'>{best_name}</span><br><br>
        <b style='color:#e6edf3;'>Test R²</b> &nbsp; <span style='color:#58a6ff;'>{best_test_r2}</span><br>
        <b style='color:#e6edf3;'>RMSE</b> &nbsp;&nbsp;&nbsp; <span style='color:#58a6ff;'>{best_rmse}</span><br>
        <b style='color:#e6edf3;'>MAE</b> &nbsp;&nbsp;&nbsp;&nbsp; <span style='color:#58a6ff;'>{best_mae}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    st.markdown("#### 📦 Features Used")
    tags_html = "".join([f"<span class='feature-tag'>{c}</span>" for c in feat_cols])
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    st.markdown("#### 🗂️ Dataset Stats")
    st.markdown(f"""
    <div class='info-box'>
        <b style='color:#e6edf3;'>Cities Covered</b><br>
        <span style='color:#58a6ff; font-size:1.2rem; font-weight:700;'>{len(city_target):,}</span><br><br>
        <b style='color:#e6edf3;'>Models Evaluated</b><br>
        <span style='color:#58a6ff; font-size:1.2rem; font-weight:700;'>{len(results_df)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;'>", unsafe_allow_html=True)
    st.caption("Built with ❤️ using Streamlit · ml_env")

# ── Main Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            border: 1px solid #30363d; border-radius: 16px;
            padding: 1.8rem 2rem; margin-bottom: 1.5rem;
            display: flex; align-items: center; gap: 1rem;'>
    <div>
        <h1 style='color:#e6edf3; margin:0; font-size:2rem; font-weight:800;'>
            🏠 PropValue AI — Property Price Predictor
        </h1>
        <p style='color:#c9d1d9; margin:0.3rem 0 0 0; font-size:0.95rem;'>
            AI-powered real estate valuation for Indian properties &nbsp;·&nbsp;
            14 ML models evaluated &nbsp;·&nbsp; City-aware encoding
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top KPI Row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpi_data = [
    (k1, "Best Model",       best_name,          "Auto-selected by Test R²"),
    (k2, "Test R² Score",    f"{best_test_r2}",  "Variance explained"),
    (k3, "RMSE (log scale)", f"{best_rmse}",     "Root Mean Squared Error"),
    (k4, "Cities in Model",  f"{len(city_target):,}", "Target + Freq encoded"),
]
for col, label, val, sub in kpi_data:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{label}</div>
            <div class='value'>{val}</div>
            <div class='sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Predict Price",
    "📊 Model Performance",
    "🏙️ City Insights",
    "📈 Feature Analysis",
    "ℹ️ About & Pipeline"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>🏗️ Enter Property Details</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.1, 1.1, 0.9])

    with col1:
        st.markdown("**Property Specifications**")
        posted_by  = st.selectbox("Posted By", le.classes_.tolist(),
                                   help="Who is listing the property")
        bhk_no     = st.slider("Number of BHK / Rooms", 1, 10, 2,
                                help="Total number of bedrooms/rooms")
        bhk_or_rk  = st.radio("Property Type", ["BHK", "RK"], horizontal=True,
                               help="BHK = Bedroom Hall Kitchen | RK = Room Kitchen")
        square_ft  = st.number_input("Area (sq ft)", min_value=100.0,
                                      max_value=10000.0, value=1000.0, step=50.0,
                                      help="Total carpet/built-up area in sq ft")

    with col2:
        st.markdown("**Location Details**")
        city      = st.selectbox("City", city_list,
                                  help="City where the property is located")
        longitude = st.number_input("Longitude", min_value=68.0, max_value=98.0,
                                     value=77.59, step=0.001, format="%.3f")
        latitude  = st.number_input("Latitude", min_value=8.0, max_value=37.0,
                                     value=12.97, step=0.001, format="%.3f")

        # Live city stats
        c_mean = city_target.get(city, global_mean)
        c_freq = city_freq.get(city, 0.0)
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.6rem;'>
            📍 <b style='color:#e6edf3;'>{city}</b><br>
            Avg Price: <span style='color:#3fb950;'>₹ {c_mean:.2f} Lacs</span> &nbsp;|&nbsp;
            Market Share: <span style='color:#58a6ff;'>{c_freq*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("**Status & Approvals**")
        rera          = st.selectbox("RERA Approved", [1, 0],
                                      format_func=lambda x: "✅ Yes" if x else "❌ No")
        ready_to_move = st.selectbox("Ready to Move", [1, 0],
                                      format_func=lambda x: "✅ Yes" if x else "❌ No")
        resale        = st.selectbox("Resale Property", [0, 1],
                                      format_func=lambda x: "✅ Yes" if x else "❌ No")

        # Price range hint
        area_band = "Budget" if square_ft < 600 else ("Mid-Range" if square_ft < 1500 else "Premium")
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.6rem;'>
            🏷️ Segment: <b style='color:#f0883e;'>{area_band}</b><br>
            Est. Price/sqft: <span style='color:#58a6ff;'>
            ₹ {(c_mean * 100000 / square_ft):,.0f}
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮  Predict Property Price", use_container_width=True, type="primary")

    if predict_btn:
        # ── Feature Engineering ────────────────────────────────────────────────
        posted_by_enc   = int(le.transform([posted_by])[0])
        is_bhk          = 1 if bhk_or_rk == "BHK" else 0
        city_target_enc = city_target.get(city, global_mean)
        city_freq_enc   = city_freq.get(city, 0.0)
        sq_ft_log       = np.log1p(square_ft)
        ppsf_proxy      = city_target_enc / square_ft if square_ft > 0 else 0
        ppsf_log        = np.log1p(ppsf_proxy)

        input_dict = {
            'RERA'            : rera,
            'BHK_NO.'         : bhk_no,
            'SQUARE_FT'       : sq_ft_log,
            'READY_TO_MOVE'   : ready_to_move,
            'RESALE'          : resale,
            'LONGITUDE'       : longitude,
            'LATITUDE'        : latitude,
            'PRICE_PER_SQFT'  : ppsf_log,
            'IS_BHK'          : is_bhk,
            'POSTED_BY_ENC'   : posted_by_enc,
            'CITY_TARGET_ENC' : city_target_enc,
            'CITY_FREQ_ENC'   : city_freq_enc,
        }

        input_df   = pd.DataFrame([input_dict])[feat_cols]
        input_sc   = scaler.transform(input_df)
        pred_log   = model.predict(input_sc)[0]
        pred_lacs  = np.expm1(pred_log)

        # Confidence range ±8%
        low_lacs   = pred_lacs * 0.92
        high_lacs  = pred_lacs * 1.08

        # ── Result ────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class='pred-card'>
            <div class='price-label'>Estimated Property Value</div>
            <div class='price-main'>₹ {pred_lacs:,.2f} Lacs</div>
            <div class='price-crore'>≈ ₹ {pred_lacs/100:,.3f} Crore</div>
            <div class='price-range'>
                Confidence Range &nbsp;·&nbsp;
                ₹ {low_lacs:,.2f} – ₹ {high_lacs:,.2f} Lacs
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Breakdown Columns ─────────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>Price per sq ft</div>
                <div class='value' style='font-size:1.4rem;'>
                    ₹ {(pred_lacs*100000/square_ft):,.0f}
                </div>
                <div class='sub'>Based on {square_ft:,.0f} sq ft</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>City Avg Price</div>
                <div class='value' style='font-size:1.4rem;'>₹ {c_mean:,.2f}L</div>
                <div class='sub'>{city}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            vs_city = ((pred_lacs - c_mean) / c_mean) * 100
            color   = "#3fb950" if vs_city >= 0 else "#f85149"
            arrow   = "▲" if vs_city >= 0 else "▼"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>vs City Average</div>
                <div class='value' style='font-size:1.4rem; color:{color};'>
                    {arrow} {abs(vs_city):.1f}%
                </div>
                <div class='sub'>{'Above' if vs_city>=0 else 'Below'} city mean</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gauge Chart ───────────────────────────────────────────────────────
        all_prices = list(city_target.values())
        pct_rank   = (sum(p < pred_lacs for p in all_prices) / len(all_prices)) * 100

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_lacs,
            delta={'reference': c_mean, 'valueformat': '.2f',
                   'increasing': {'color': '#3fb950'}, 'decreasing': {'color': '#f85149'}},
            number={'suffix': " L", 'valueformat': '.2f', 'font': {'color': '#3fb950', 'size': 28}},
            gauge={
                'axis': {'range': [0, max(all_prices)*1.1],
                         'tickcolor': '#c9d1d9', 'tickfont': {'color': '#c9d1d9'}},
                'bar': {'color': '#238636'},
                'bgcolor': '#161b22',
                'bordercolor': '#30363d',
                'steps': [
                    {'range': [0, np.percentile(all_prices, 25)],   'color': '#1c2128'},
                    {'range': [np.percentile(all_prices, 25),
                               np.percentile(all_prices, 75)],       'color': '#21262d'},
                    {'range': [np.percentile(all_prices, 75),
                               max(all_prices)*1.1],                  'color': '#2d333b'},
                ],
                'threshold': {'line': {'color': '#58a6ff', 'width': 3},
                              'thickness': 0.8, 'value': c_mean}
            },
            title={'text': f"Price Gauge (Blue line = {city} avg)", 'font': {'color': '#c9d1d9'}}
        ))
        fig_gauge.update_layout(**PLOT_LAYOUT, height=280)

        gc1, gc2 = st.columns([1.5, 1])
        with gc1:
            st.plotly_chart(fig_gauge, use_container_width=True)
        with gc2:
            st.markdown(f"""
            <div style='padding:1rem;'>
                <div class='section-header'>📋 Input Summary</div>
            </div>
            """, unsafe_allow_html=True)
            summary = pd.DataFrame({
                "Property Detail": ["Posted By","BHK No.","Type","Area","City",
                                    "RERA","Ready to Move","Resale"],
                "Value": [posted_by, bhk_no, bhk_or_rk, f"{square_ft:,.0f} sq ft", city,
                          "Yes" if rera else "No",
                          "Yes" if ready_to_move else "No",
                          "Yes" if resale else "No"]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True, height=290)

        # ── Percentile Banner ─────────────────────────────────────────────────
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:0.8rem 1.2rem; text-align:center; margin-top:0.5rem;'>
            <span style='color:#c9d1d9;'>This property is priced higher than </span>
            <span style='color:#f0883e; font-weight:700; font-size:1.1rem;'>{pct_rank:.1f}%</span>
            <span style='color:#c9d1d9;'> of all cities in the dataset</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>🏆 Model Leaderboard</div>", unsafe_allow_html=True)

    # Best model banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0d1117,#1a3a1a);
                border:1px solid #238636; border-radius:12px;
                padding:1rem 1.5rem; margin-bottom:1rem;'>
        <span style='color:#3fb950; font-size:1.1rem; font-weight:700;'>🥇 {best_name}</span>
        &nbsp;&nbsp;
        <span style='color:#c9d1d9;'>Test R²: </span><span style='color:#58a6ff;'>{results_df.iloc[0]['Test_R2']}</span>
        &nbsp;|&nbsp;
        <span style='color:#c9d1d9;'>Adj R²: </span><span style='color:#58a6ff;'>{results_df.iloc[0]['Test_AdjR2']}</span>
        &nbsp;|&nbsp;
        <span style='color:#c9d1d9;'>RMSE: </span><span style='color:#58a6ff;'>{results_df.iloc[0]['RMSE']}</span>
        &nbsp;|&nbsp;
        <span style='color:#c9d1d9;'>MAE: </span><span style='color:#58a6ff;'>{results_df.iloc[0]['MAE']}</span>
        &nbsp;|&nbsp;
        <span style='color:#c9d1d9;'>CV R²: </span><span style='color:#58a6ff;'>{results_df.iloc[0]['CV_R2']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Styled leaderboard table
    def style_table(df):
        def row_style(row):
            if row['Model'] == best_name:
                return ['background-color:#1a3a1a; color:#3fb950; font-weight:600'] * len(row)
            return [''] * len(row)
        return df.style.apply(row_style, axis=1)\
                       .format({c: '{:.4f}' for c in df.columns if c != 'Model'})\
                       .bar(subset=['Test_R2'], color='#1c4a2a')\
                       .bar(subset=['RMSE'], color='#3a1c1c')

    st.dataframe(style_table(results_df), use_container_width=True, hide_index=True, height=490)

    st.markdown("<div class='section-header'>📊 Visual Comparison</div>", unsafe_allow_html=True)

    # R2 grouped bar
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Bar(name='Train R²',    x=results_df['Model'], y=results_df['Train_R2'],
                            marker_color='#1f6feb', opacity=0.85))
    fig_r2.add_trace(go.Bar(name='Test R²',     x=results_df['Model'], y=results_df['Test_R2'],
                            marker_color='#f0883e', opacity=0.85))
    fig_r2.add_trace(go.Bar(name='Train AdjR²', x=results_df['Model'], y=results_df['Train_AdjR2'],
                            marker_color='#3fb950', opacity=0.85))
    fig_r2.add_trace(go.Bar(name='Test AdjR²',  x=results_df['Model'], y=results_df['Test_AdjR2'],
                            marker_color='#d2a8ff', opacity=0.85))
    fig_r2.update_layout(**PLOT_LAYOUT, barmode='group',
                         title='R² & Adjusted R² — All Models',
                         xaxis_tickangle=-35, height=420,
                         legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#30363d'))
    st.plotly_chart(fig_r2, use_container_width=True)

    c_rmse, c_mae = st.columns(2)
    with c_rmse:
        rmse_df = results_df.sort_values('RMSE')
        fig_rmse = go.Figure(go.Bar(
            x=rmse_df['RMSE'], y=rmse_df['Model'], orientation='h',
            marker_color=['#3fb950' if m == best_name else '#1f6feb' for m in rmse_df['Model']],
            text=[f"{v:.4f}" for v in rmse_df['RMSE']], textposition='outside',
            textfont=dict(color='#c9d1d9')
        ))
        fig_rmse.update_layout(**PLOT_LAYOUT, title='RMSE (lower = better)', height=420,
                               xaxis_title='RMSE')
        st.plotly_chart(fig_rmse, use_container_width=True)

    with c_mae:
        mae_df = results_df.sort_values('MAE')
        fig_mae = go.Figure(go.Bar(
            x=mae_df['MAE'], y=mae_df['Model'], orientation='h',
            marker_color=['#3fb950' if m == best_name else '#f0883e' for m in mae_df['Model']],
            text=[f"{v:.4f}" for v in mae_df['MAE']], textposition='outside',
            textfont=dict(color='#c9d1d9')
        ))
        fig_mae.update_layout(**PLOT_LAYOUT, title='MAE (lower = better)', height=420,
                              xaxis_title='MAE')
        st.plotly_chart(fig_mae, use_container_width=True)

    # Radar chart — top 5 models
    st.markdown("<div class='section-header'>🕸️ Radar Chart — Top 5 Models</div>", unsafe_allow_html=True)
    top5 = results_df.head(5)
    metrics_radar = ['Test_R2', 'Test_AdjR2', 'CV_R2']
    # Invert RMSE & MAE so higher = better for radar
    top5 = top5.copy()
    top5['1-RMSE_norm'] = 1 - (top5['RMSE'] - top5['RMSE'].min()) / (top5['RMSE'].max() - top5['RMSE'].min() + 1e-9)
    top5['1-MAE_norm']  = 1 - (top5['MAE']  - top5['MAE'].min())  / (top5['MAE'].max()  - top5['MAE'].min()  + 1e-9)
    radar_metrics = ['Test_R2', 'Test_AdjR2', 'CV_R2', '1-RMSE_norm', '1-MAE_norm']
    radar_labels  = ['Test R²', 'Test AdjR²', 'CV R²', 'RMSE (inv)', 'MAE (inv)']
    colors_radar  = ['#3fb950','#58a6ff','#f0883e','#d2a8ff','#ffa657']

    fig_radar = go.Figure()
    for i, row in top5.iterrows():
        vals = [row[m] for m in radar_metrics]
        vals += [vals[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_labels + [radar_labels[0]],
            fill='toself', name=row['Model'],
            line_color=colors_radar[list(top5.index).index(i) % len(colors_radar)],
            opacity=0.7
        ))
    fig_radar.update_layout(**PLOT_LAYOUT, polar=dict(
        bgcolor='#161b22',
        radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363d', color='#c9d1d9'),
        angularaxis=dict(gridcolor='#30363d', color='#c9d1d9')
    ), title='Top 5 Models — Multi-Metric Radar', height=450,
       legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CITY INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>🏙️ City-wise Market Intelligence</div>", unsafe_allow_html=True)

    city_df = pd.DataFrame({
        'City'       : list(city_target.keys()),
        'Avg_Price'  : list(city_target.values()),
        'Frequency'  : [city_freq.get(c, 0) * 100 for c in city_target.keys()]
    }).sort_values('Avg_Price', ascending=False).reset_index(drop=True)
    city_df['Rank'] = city_df.index + 1

    # Controls
    ctrl1, ctrl2 = st.columns([1, 2])
    with ctrl1:
        top_n = st.slider("Show Top N Cities", 10, min(100, len(city_df)), 20)
    with ctrl2:
        sort_by = st.radio("Sort by", ["Average Price", "Market Share"], horizontal=True)

    if sort_by == "Market Share":
        city_df = city_df.sort_values('Frequency', ascending=False).reset_index(drop=True)
    else:
        city_df = city_df.sort_values('Avg_Price', ascending=False).reset_index(drop=True)

    top_cities = city_df.head(top_n)

    # Top cities bar chart
    fig_city = go.Figure(go.Bar(
        x=top_cities['City'], y=top_cities['Avg_Price'],
        marker=dict(
            color=top_cities['Avg_Price'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text='Price (Lacs)', font=dict(color='#c9d1d9')),
                tickfont=dict(color='#c9d1d9')
            )
        ),
        text=[f"₹{v:.1f}L" for v in top_cities['Avg_Price']],
        textposition='outside', textfont=dict(color='#c9d1d9', size=9),
        hovertemplate='<b>%{x}</b><br>Avg Price: ₹%{y:.2f} Lacs<extra></extra>'
    ))
    fig_city.update_layout(**PLOT_LAYOUT,
                           title=f'Top {top_n} Cities by Average Property Price',
                           xaxis_tickangle=-45, height=420,
                           yaxis_title='Average Price (Lacs)')
    st.plotly_chart(fig_city, use_container_width=True)

    # Scatter: Price vs Market Share
    fig_scatter = px.scatter(
        city_df.head(60), x='Frequency', y='Avg_Price',
        text='City', size='Frequency', color='Avg_Price',
        color_continuous_scale='Plasma',
        labels={'Frequency': 'Market Share (%)', 'Avg_Price': 'Avg Price (Lacs)'},
        title='City Market Share vs Average Price (Top 60 Cities)',
        template='plotly_dark'
    )
    fig_scatter.update_traces(textposition='top center', textfont_size=8)
    fig_scatter.update_layout(**PLOT_LAYOUT, height=480,
                              coloraxis_colorbar=dict(
                                  title=dict(text='Price (L)', font=dict(color='#c9d1d9')),
                                  tickfont=dict(color='#c9d1d9')
                              ))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # City search & compare
    st.markdown("<div class='section-header'>🔍 Compare Cities</div>", unsafe_allow_html=True)
    compare_cities = st.multiselect("Select cities to compare", city_list,
                                     default=city_list[:5] if len(city_list) >= 5 else city_list)
    if compare_cities:
        comp_df = pd.DataFrame({
            'City'        : compare_cities,
            'Avg Price (L)': [city_target.get(c, global_mean) for c in compare_cities],
            'Market Share %': [city_freq.get(c, 0)*100 for c in compare_cities]
        }).sort_values('Avg Price (L)', ascending=False)

        fig_comp = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Average Price (Lacs)', 'Market Share (%)'))
        fig_comp.add_trace(go.Bar(x=comp_df['City'], y=comp_df['Avg Price (L)'],
                                   marker_color='#3fb950', name='Avg Price'), row=1, col=1)
        fig_comp.add_trace(go.Bar(x=comp_df['City'], y=comp_df['Market Share %'],
                                   marker_color='#58a6ff', name='Market Share'), row=1, col=2)
        fig_comp.update_layout(**PLOT_LAYOUT, height=380, showlegend=False,
                                xaxis_tickangle=-30, xaxis2_tickangle=-30)
        st.plotly_chart(fig_comp, use_container_width=True)

        st.dataframe(comp_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Price distribution histogram
    st.markdown("<div class='section-header'>📊 City Price Distribution</div>", unsafe_allow_html=True)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=list(city_target.values()), nbinsx=40,
        marker_color='#1f6feb', opacity=0.8,
        hovertemplate='Price Range: %{x:.1f}L<br>Count: %{y}<extra></extra>'
    ))
    fig_hist.add_vline(x=global_mean, line_dash='dash', line_color='#f0883e',
                       annotation_text=f'Global Mean: ₹{global_mean:.1f}L',
                       annotation_font_color='#f0883e')
    fig_hist.update_layout(**PLOT_LAYOUT, title='Distribution of City Average Prices',
                           xaxis_title='Average Price (Lacs)', yaxis_title='Number of Cities',
                           height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>📈 Feature Importance & Sensitivity Analysis</div>",
                unsafe_allow_html=True)

    # Feature importance (if model supports it)
    if hasattr(model, 'feature_importances_'):
        fi_df = pd.DataFrame({
            'Feature'   : feat_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
            marker=dict(color=fi_df['Importance'], colorscale='Viridis'),
            text=[f"{v:.4f}" for v in fi_df['Importance']],
            textposition='outside', textfont=dict(color='#c9d1d9')
        ))
        fig_fi.update_layout(**PLOT_LAYOUT, title='Feature Importance (from Best Model)',
                             xaxis_title='Importance Score', height=420)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type (e.g. SVR, Linear).")

    # ── Price Sensitivity: Area ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🔬 Price Sensitivity Analysis</div>",
                unsafe_allow_html=True)

    sens_city = st.selectbox("Select City for Sensitivity", city_list, key='sens_city')
    s_col1, s_col2 = st.columns(2)

    with s_col1:
        # Area sensitivity
        areas = np.linspace(300, 5000, 60)
        preds_area = []
        for a in areas:
            ct = city_target.get(sens_city, global_mean)
            cf = city_freq.get(sens_city, 0.0)
            row = {
                'RERA': 1, 'BHK_NO.': 2, 'SQUARE_FT': np.log1p(a),
                'READY_TO_MOVE': 1, 'RESALE': 0,
                'LONGITUDE': 77.59, 'LATITUDE': 12.97,
                'PRICE_PER_SQFT': np.log1p(ct / a if a > 0 else 0),
                'IS_BHK': 1, 'POSTED_BY_ENC': 1,
                'CITY_TARGET_ENC': ct, 'CITY_FREQ_ENC': cf
            }
            inp = pd.DataFrame([row])[feat_cols]
            preds_area.append(np.expm1(model.predict(scaler.transform(inp))[0]))

        fig_area = go.Figure()
        fig_area.add_trace(go.Scatter(x=areas, y=preds_area, mode='lines',
                                       line=dict(color='#3fb950', width=2.5),
                                       fill='tozeroy', fillcolor='rgba(63,185,80,0.1)',
                                       hovertemplate='Area: %{x:.0f} sqft<br>Price: ₹%{y:.2f}L<extra></extra>'))
        fig_area.update_layout(**PLOT_LAYOUT, title=f'Price vs Area — {sens_city}',
                               xaxis_title='Area (sq ft)', yaxis_title='Predicted Price (Lacs)',
                               height=340)
        st.plotly_chart(fig_area, use_container_width=True)

    with s_col2:
        # BHK sensitivity
        bhks = list(range(1, 11))
        preds_bhk = []
        for b in bhks:
            ct = city_target.get(sens_city, global_mean)
            cf = city_freq.get(sens_city, 0.0)
            row = {
                'RERA': 1, 'BHK_NO.': b, 'SQUARE_FT': np.log1p(1000),
                'READY_TO_MOVE': 1, 'RESALE': 0,
                'LONGITUDE': 77.59, 'LATITUDE': 12.97,
                'PRICE_PER_SQFT': np.log1p(ct / 1000 if 1000 > 0 else 0),
                'IS_BHK': 1, 'POSTED_BY_ENC': 1,
                'CITY_TARGET_ENC': ct, 'CITY_FREQ_ENC': cf
            }
            inp = pd.DataFrame([row])[feat_cols]
            preds_bhk.append(np.expm1(model.predict(scaler.transform(inp))[0]))

        fig_bhk = go.Figure()
        fig_bhk.add_trace(go.Bar(x=[f"{b} BHK" for b in bhks], y=preds_bhk,
                                  marker=dict(color=preds_bhk, colorscale='Blues'),
                                  text=[f"₹{v:.1f}L" for v in preds_bhk],
                                  textposition='outside', textfont=dict(color='#c9d1d9'),
                                  hovertemplate='%{x}<br>Price: ₹%{y:.2f}L<extra></extra>'))
        fig_bhk.update_layout(**PLOT_LAYOUT, title=f'Price vs BHK Count — {sens_city}',
                              xaxis_title='BHK', yaxis_title='Predicted Price (Lacs)',
                              height=340)
        st.plotly_chart(fig_bhk, use_container_width=True)

    # ── City Price Comparison (fixed area) ────────────────────────────────────
    st.markdown("<div class='section-header'>🏙️ Same Property, Different Cities</div>",
                unsafe_allow_html=True)
    st.caption("Predicted price for a 1000 sq ft, 2 BHK, RERA-approved, ready-to-move property across cities")

    compare_n = st.slider("Number of cities to compare", 5, 30, 15, key='city_compare_n')
    sample_cities = city_df.head(compare_n)['City'].tolist()
    city_preds = []
    for c in sample_cities:
        ct = city_target.get(c, global_mean)
        cf = city_freq.get(c, 0.0)
        row = {
            'RERA': 1, 'BHK_NO.': 2, 'SQUARE_FT': np.log1p(1000),
            'READY_TO_MOVE': 1, 'RESALE': 0,
            'LONGITUDE': 77.59, 'LATITUDE': 12.97,
            'PRICE_PER_SQFT': np.log1p(ct / 1000),
            'IS_BHK': 1, 'POSTED_BY_ENC': 1,
            'CITY_TARGET_ENC': ct, 'CITY_FREQ_ENC': cf
        }
        inp = pd.DataFrame([row])[feat_cols]
        city_preds.append(np.expm1(model.predict(scaler.transform(inp))[0]))

    fig_cp = go.Figure(go.Bar(
        x=sample_cities, y=city_preds,
        marker=dict(color=city_preds, colorscale='RdYlGn'),
        text=[f"₹{v:.1f}L" for v in city_preds],
        textposition='outside', textfont=dict(color='#c9d1d9', size=9),
        hovertemplate='<b>%{x}</b><br>Predicted: ₹%{y:.2f}L<extra></extra>'
    ))
    fig_cp.update_layout(**PLOT_LAYOUT, title='Predicted Price for Same Property Across Cities',
                         xaxis_tickangle=-40, height=400, yaxis_title='Predicted Price (Lacs)')
    st.plotly_chart(fig_cp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT & PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>🔬 ML Pipeline Overview</div>", unsafe_allow_html=True)

    steps = [
        ("📥", "Data Ingestion",       "Loaded House Price CSV with 12 raw features covering property specs, location, and seller info."),
        ("🔍", "EDA",                  "Analyzed distributions, correlations, missing values, duplicates, and categorical breakdowns."),
        ("⚙️", "Feature Engineering",  "Created PRICE_PER_SQFT, IS_BHK, POSTED_BY_ENC. Extracted CITY from ADDRESS."),
        ("🏙️", "City Encoding",        "Applied Target Encoding (mean price/city) + Frequency Encoding (market share) — no cardinality explosion."),
        ("📦", "Outlier Treatment",    "IQR-based Winsorization on SQUARE_FT, TARGET, BHK_NO., PRICE_PER_SQFT. Zero row deletion."),
        ("📐", "Skewness Removal",     "log1p transformation on SQUARE_FT, TARGET(PRICE_IN_LACS), PRICE_PER_SQFT."),
        ("⚖️", "Feature Scaling",      "StandardScaler applied to all 12 input features before model training."),
        ("🤖", "Model Training",       "14 regression algorithms trained: Linear, Ridge, Lasso, ElasticNet, DT, RF, GB, AdaBoost, ExtraTrees, Bagging, KNN, SVR, XGBoost, LightGBM."),
        ("📊", "Evaluation",           "Metrics: Train R², Train Adj-R², Test R², Test Adj-R², MAE, RMSE, 5-Fold CV R²."),
        ("🏆", "Best Model Selection", "Auto-selected by highest Test R². All artifacts saved as PKL files."),
        ("🚀", "Deployment",           "Streamlit app with prediction, city insights, sensitivity analysis, and model comparison."),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        col_icon, col_content = st.columns([0.08, 0.92])
        with col_icon:
            st.markdown(f"<div style='font-size:1.8rem; text-align:center; padding-top:0.3rem;'>{icon}</div>",
                        unsafe_allow_html=True)
        with col_content:
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                        padding:0.7rem 1rem; margin-bottom:0.4rem;'>
                <span style='color:#58a6ff; font-weight:600;'>Step {i+1}: {title}</span><br>
                <span style='color:#8b949e; font-size:0.88rem;'>{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📦 Saved Artifacts</div>", unsafe_allow_html=True)
    artifacts = [
        ("best_model.pkl",      "Trained best regression model"),
        ("scaler.pkl",          "Fitted StandardScaler"),
        ("feature_columns.pkl", "Ordered list of 12 input features"),
        ("label_encoder.pkl",   "LabelEncoder for POSTED_BY"),
        ("city_target_enc.pkl", "Dict: city → mean price (target encoding)"),
        ("city_freq_enc.pkl",   "Dict: city → relative frequency"),
        ("global_mean.pkl",     "Global mean price (fallback for unseen cities)"),
        ("model_results.pkl",   "DataFrame with all 14 model metrics"),
    ]
    art_df = pd.DataFrame(artifacts, columns=["File", "Description"])
    st.dataframe(art_df, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'>🛠️ Tech Stack</div>", unsafe_allow_html=True)
    tech_cols = st.columns(4)
    tech = [
        ("🐍", "Python 3.10",    "ml_env"),
        ("🐼", "Pandas / NumPy", "Data processing"),
        ("🤖", "Scikit-learn",   "ML models"),
        ("⚡", "XGBoost",        "Gradient boosting"),
        ("💡", "LightGBM",       "Fast boosting"),
        ("📊", "Plotly",         "Interactive charts"),
        ("🌐", "Streamlit",      "Web deployment"),
        ("📓", "Jupyter",        "Development"),
    ]
    for i, (icon, name, desc) in enumerate(tech):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card' style='margin-bottom:0.5rem;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div style='color:#e6edf3; font-weight:600; font-size:0.9rem;'>{name}</div>
                <div class='sub'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; color:#8b949e; font-size:0.82rem; padding:1rem;
                border-top:1px solid #30363d; margin-top:1rem;'>
        PropValue AI · House Price Prediction · Built with Streamlit · ml_env · 2024
    </div>
    """, unsafe_allow_html=True)
