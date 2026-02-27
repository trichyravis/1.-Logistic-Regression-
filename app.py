"""
Logistic Regression in Finance — Streamlit App
The Mountain Path – World of Finance | Prof. V. Ravichandran
"""
import streamlit as st
from styles import inject_css
from tab_concepts   import tab_concepts
from tab_model      import tab_model
from tab_diagnostics import tab_diagnostics
from tab_finance    import tab_finance
from tab_code       import tab_code
from tab_vocab      import tab_vocab

st.set_page_config(
    page_title="Logistic Regression in Finance",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()

NO_SEL = "user-select:none;-webkit-user-select:none"
FH = "'Playfair Display',serif"
FB = "'Source Sans Pro',sans-serif"

st.html(f"""
<div style="text-align:center;padding:28px 20px 16px;
            border-bottom:2px solid #FFD700;margin-bottom:24px;{NO_SEL}">
  <div style="font-family:{FH};font-size:2.2rem;color:#FFD700;
              -webkit-text-fill-color:#FFD700;letter-spacing:1px;margin-bottom:6px;font-weight:700">
    Logistic Regression in Finance
  </div>
  <div style="color:#8892b0;-webkit-text-fill-color:#8892b0;
              font-family:{FB};font-size:1rem;margin-bottom:4px">
    MLE Estimation, Odds Ratios, Diagnostics &amp; Financial Applications
  </div>
  <div style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;
              font-family:{FB};font-size:0.85rem;font-style:italic">
    The Mountain Path – World of Finance &nbsp;|&nbsp; Prof. V. Ravichandran
  </div>
  <div style="display:flex;justify-content:center;gap:10px;margin-top:12px;flex-wrap:wrap">
    <span style="background:rgba(220,53,69,0.2);color:#dc3545;-webkit-text-fill-color:#dc3545;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #dc3545">Credit Default (PD)</span>
    <span style="background:rgba(255,159,67,0.15);color:#ff9f43;-webkit-text-fill-color:#ff9f43;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #ff9f43">Fraud Detection</span>
    <span style="background:rgba(255,215,0,0.12);color:#FFD700;-webkit-text-fill-color:#FFD700;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #FFD700">Rating Downgrade</span>
    <span style="background:rgba(40,167,69,0.18);color:#28a745;-webkit-text-fill-color:#28a745;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #28a745">Loan Approval</span>
    <span style="background:rgba(162,155,254,0.15);color:#a29bfe;-webkit-text-fill-color:#a29bfe;
                 padding:3px 12px;border-radius:20px;font-size:.78rem;font-family:{FB};
                 border:1px solid #a29bfe">Scorecard Models</span>
  </div>
</div>
""")

TABS = st.tabs([
    "🎯 Concepts & Theory",
    "🔬 Model Builder",
    "🔍 Diagnostics",
    "🏦 Finance Cases",
    "🐍 Python Code",
    "📚 Education Hub",
])

with TABS[0]: tab_concepts()
with TABS[1]: tab_model()
with TABS[2]: tab_diagnostics()
with TABS[3]: tab_finance()
with TABS[4]: tab_code()
with TABS[5]: tab_vocab()

st.html(f"""
<div style="text-align:center;padding:18px;color:#8892b0;-webkit-text-fill-color:#8892b0;
            font-family:{FB};font-size:.84rem;border-top:1px solid #1e3a5f;
            margin-top:28px;line-height:1.9;{NO_SEL}">
  <span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:700">
    The Mountain Path – World of Finance
  </span><br>
  <a href="https://www.linkedin.com/in/trichyravis" target="_blank"
     style="color:#FFD700;-webkit-text-fill-color:#FFD700;text-decoration:none">LinkedIn</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/trichyravis" target="_blank"
     style="color:#FFD700;-webkit-text-fill-color:#FFD700;text-decoration:none">GitHub</a><br>
  <span style="color:#8892b0;-webkit-text-fill-color:#8892b0">
    Prof. V. Ravichandran &nbsp;|&nbsp;
    28+ Years Corporate Finance &amp; Banking Experience &nbsp;|&nbsp;
    10+ Years Academic Excellence
  </span>
</div>
""")
