import streamlit as st

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

.stApp { background: linear-gradient(135deg,#1a2332,#243447,#2a3f5f) !important; font-family:'Source Sans Pro',sans-serif !important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:1rem !important; max-width:1200px; }

.stTabs [data-baseweb="tab-list"] { background:#112240 !important; border-radius:8px; padding:4px; gap:4px; border:1px solid #1e3a5f; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:#8892b0 !important; font-family:'Source Sans Pro',sans-serif !important; font-weight:600 !important; font-size:.88rem !important; border-radius:6px !important; padding:8px 16px !important; border:none !important; transition:all .25s !important; }
.stTabs [data-baseweb="tab"]:hover { color:#FFD700 !important; background:rgba(0,77,128,.3) !important; }
.stTabs [aria-selected="true"] { background:#003366 !important; color:#FFD700 !important; border:1px solid #FFD700 !important; }
.stTabs [data-baseweb="tab-border"] { display:none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top:14px !important; }

div[data-testid="stMetric"] { background:#112240 !important; border:1px solid #1e3a5f !important; border-radius:8px !important; padding:14px !important; }
div[data-testid="stMetric"] label { color:#ADD8E6 !important; font-weight:600 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#FFD700 !important; font-family:'JetBrains Mono',monospace !important; font-size:1.3rem !important; }
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { color:#ADD8E6 !important; }

.stSelectbox label,.stSlider label,.stRadio label,.stNumberInput label { color:#ADD8E6 !important; font-weight:600 !important; }
.stSelectbox [data-baseweb="select"]>div { background:#112240 !important; border-color:#1e3a5f !important; color:#e6f1ff !important; }
.stSelectbox [data-baseweb="select"] span { color:#e6f1ff !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color:#e6f1ff !important; }
.stRadio div[role="radiogroup"] label span { color:#e6f1ff !important; }
.stSlider p { color:#ADD8E6 !important; }
.stNumberInput input { background:#112240 !important; color:#e6f1ff !important; border-color:#1e3a5f !important; }
.stTextInput input { background:#112240 !important; color:#e6f1ff !important; border-color:#1e3a5f !important; }
.stButton button { background:#003366 !important; color:#FFD700 !important; border:2px solid #FFD700 !important; font-weight:700 !important; border-radius:6px !important; }
.stButton button:hover { background:#004d80 !important; }
.stCodeBlock pre { background:#0d1f3a !important; }
.stCodeBlock code { color:#64ffda !important; }
.stDataFrame { background:#112240 !important; }
div[data-testid="stDataFrameResizable"] { background:#112240 !important; border:1px solid #1e3a5f !important; border-radius:8px !important; }
.stCheckbox label span { color:#e6f1ff !important; }
.stMultiSelect label { color:#ADD8E6 !important; font-weight:600 !important; }
iframe { border:none !important; }
</style>
""", unsafe_allow_html=True)
