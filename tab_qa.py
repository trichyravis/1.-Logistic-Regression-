"""
tab_qa.py — Q&A / Self-Assessment for Logistic Regression
MCQ, Numerical Problems, AI Tutor
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col,
    table_html, metric_row, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

MCQ_BANK = [
    {"id":"LR-01","topic":"Concepts","level":"Foundation",
     "question":"In logistic regression, the sigmoid function maps the linear predictor z to:",
     "options":["Any real number from −∞ to +∞","A value between 0 and 1 exclusively","A value between −1 and 1","A normally distributed random variable"],
     "answer":1,"explanation":"σ(z) = 1/(1+e⁻ᶻ) always outputs (0,1), making it suitable for probability modelling. z itself is −∞ to +∞ (the linear predictor); the sigmoid squashes it into the probability range."},
    {"id":"LR-02","topic":"Concepts","level":"Foundation",
     "question":"If a logistic regression coefficient β₁ = 0.85 for D/E ratio in a credit default model, the odds ratio exp(β₁) is approximately:",
     "options":["0.427","2.340","1.850","0.573"],
     "answer":1,"explanation":"exp(0.85) = 2.340. This means each 1-unit increase in D/E ratio multiplies the default odds by 2.340. Odds > 1 → risk factor. If β were −0.85, OR = exp(−0.85) = 0.427 (protective factor)."},
    {"id":"LR-03","topic":"Concepts","level":"Foundation",
     "question":"Compared to OLS, logistic regression coefficients are estimated by:",
     "options":["Minimising Σ(Yᵢ − p̂ᵢ)² (sum of squared residuals)","Maximising the log-likelihood L = Σ[Yᵢln(p̂ᵢ) + (1−Yᵢ)ln(1−p̂ᵢ)]","Minimising the sum of absolute residuals (LAD)","Maximising R²"],
     "answer":1,"explanation":"MLE (Maximum Likelihood Estimation) is used because Y is binary — the Bernoulli likelihood is appropriate. OLS minimises SSE which gives invalid predictions (outside [0,1]) for binary Y. MLE is solved iteratively using Newton-Raphson / IRLS."},
    {"id":"LR-04","topic":"Model Fit","level":"Foundation",
     "question":"AUC-ROC = 0.50 in a fraud detection model means:",
     "options":["The model correctly classifies 50% of cases","The model has no discriminatory ability — equivalent to random guessing","The model detects 50% of frauds","The model has 50% precision"],
     "answer":1,"explanation":"AUC = 0.50 = random classifier. AUC measures P(score(Y=1) > score(Y=0)). At 0.50, the model ranks a random fraud case above a random legitimate case exactly 50% of the time — no better than a coin flip. AUC = 1.0 = perfect discrimination."},
    {"id":"LR-05","topic":"Model Fit","level":"Intermediate",
     "question":"A logistic model has null log-likelihood = −180 and full log-likelihood = −108. The McFadden R² is:",
     "options":["0.30","0.40","0.60","0.20"],
     "answer":1,"explanation":"McFadden R² = 1 − L_full/L_null = 1 − (−108)/(−180) = 1 − 0.60 = 0.40. This is excellent — values 0.20–0.40 indicate very good fit. The Likelihood Ratio test statistic = −2(−180−(−108)) = −2(−72) = 144, tested against χ²(k)."},
    {"id":"LR-06","topic":"Model Fit","level":"Intermediate",
     "question":"In a credit risk model, a Hosmer-Lemeshow test gives HL = 6.2 with p-value = 0.62 (χ²(8), α=5%). The conclusion is:",
     "options":["REJECT H₀ — model is poorly calibrated","FAIL TO REJECT H₀ — model is well-calibrated","The model has high discrimination","Multicollinearity is present"],
     "answer":1,"explanation":"HL H₀: the model is well-calibrated. Since p=0.62 > 0.05, we fail to reject — predicted probabilities match observed event rates across decile groups. Under IFRS 9 validation, banks require HL p > 0.10 for calibration approval."},
    {"id":"LR-07","topic":"Diagnostics","level":"Intermediate",
     "question":"Complete separation in logistic regression occurs when:",
     "options":["R² = 1 in the underlying OLS regression","A predictor (or combination) perfectly classifies all observations","All predicted probabilities equal 0.5","The Hosmer-Lemeshow test p-value is < 0.05"],
     "answer":1,"explanation":"Complete separation means ∃β: Xβ > 0 iff Y=1. MLE diverges — coefficients → ±∞ and standard errors → ∞. Common in small samples. Remedy: Firth's penalised logistic regression (adds a Jeffreys prior penalty to the log-likelihood)."},
    {"id":"LR-08","topic":"Diagnostics","level":"Advanced",
     "question":"A logistic regression has AUC = 0.84, but the Hosmer-Lemeshow test gives p = 0.02. This means:",
     "options":["The model has good discrimination but poor calibration","The model has poor discrimination and poor calibration","The model has perfect calibration but poor discrimination","Both AUC and calibration are satisfactory"],
     "answer":0,"explanation":"AUC measures discrimination (ranking) — 0.84 is good. HL tests calibration (does predicted 30% mean 30% actual defaults?). These are independent: a model can rank well (high AUC) but systematically under/overpredict probabilities (poor HL). Both are required for Basel-compliant PD models."},
    {"id":"LR-09","topic":"Finance","level":"Foundation",
     "question":"Under Basel III, the minimum AUC for an internal ratings-based (IRB) credit risk model to be acceptable is:",
     "options":["0.60","0.75","0.90","0.50"],
     "answer":1,"explanation":"Basel III guidance (BIS, 2005) suggests AUC ≥ 0.75 (Gini ≥ 0.50) for IRB credit models. Models below this threshold may not be approved by supervisors. In practice, most banks target AUC ≥ 0.80. The Gini coefficient = 2 × AUC − 1."},
    {"id":"LR-10","topic":"Finance","level":"Intermediate",
     "question":"In a fraud detection model with 0.5% fraud rate, accuracy = 99.5%. The most likely explanation is:",
     "options":["The model has excellent performance across all metrics","The model predicts 'no fraud' for all transactions (trivial classifier)","The model catches 99.5% of all frauds","The threshold τ has been optimally chosen"],
     "answer":1,"explanation":"With 0.5% fraud rate, predicting Y=0 always yields 99.5% accuracy — but recall = 0% (catches no fraud). This is the class imbalance trap. Always use F1, Recall, or AUC for imbalanced datasets. Accuracy is misleading when classes are severely skewed."},
    {"id":"LR-11","topic":"Finance","level":"Advanced",
     "question":"Under IFRS 9 Expected Credit Loss, the formula is ECL = PD × LGD × EAD. If logistic regression gives PD = 3.2% and LGD = 45%, EAD = ₹10 million, the 12-month ECL is:",
     "options":["₹144,000","₹320,000","₹450,000","₹1,440,000"],
     "answer":0,"explanation":"ECL = PD × LGD × EAD = 0.032 × 0.45 × 10,000,000 = 0.0144 × 10,000,000 = ₹144,000. The logistic model provides PD — the other components (LGD, EAD) require separate models. Stage 1 uses 12-month PD; Stage 2/3 use lifetime PD."},
    {"id":"LR-12","topic":"Concepts","level":"Advanced",
     "question":"The marginal effect of X₁ on P(Y=1) in logistic regression is:",
     "options":["Constant at β₁ regardless of X","β₁ × p̂ × (1−p̂) which varies with the predicted probability","exp(β₁) at every point","1/σ²"],
     "answer":1,"explanation":"Unlike OLS (constant marginal effect = β₁), in logistic regression the marginal effect is ∂P/∂X₁ = β₁ × σ(z) × (1−σ(z)) = β₁ × p̂ × (1−p̂). This is maximised at p̂ = 0.5 (decision boundary) and approaches 0 at the extremes (p̂ → 0 or 1). Always evaluate marginal effects at the mean or across the distribution."},
]

NUMERICAL_BANK = [
    {"id":"NUM-LR-1","topic":"Concepts","level":"Foundation",
     "title":"Sigmoid & Odds Calculation",
     "question":"A credit default model gives z = −3.5 + 0.8(D/E) − 0.6(ICR). For a firm with D/E=3.0, ICR=2.5: (a) Compute z, (b) Compute P(Default), (c) Compute the odds of default, (d) Interpret the OR for D/E.",
     "solution":"Step 1 — Compute z:\n  z = −3.5 + 0.8(3.0) − 0.6(2.5)\n    = −3.5 + 2.4 − 1.5 = −2.6\n\nStep 2 — P(Default) = σ(z):\n  P = 1/(1 + e^2.6) = 1/(1 + 13.46) = 1/14.46 = 0.0692 = 6.92%\n\nStep 3 — Odds of default:\n  Odds = P/(1−P) = 0.0692/0.9308 = 0.0743\n  Alternatively: Odds = e^z = e^(−2.6) = 0.0743\n\nStep 4 — Odds Ratio for D/E:\n  OR = exp(β_DE) = exp(0.8) = 2.226\n  Interpretation: 1-unit increase in D/E multiplies default odds by 2.226×\n  If D/E rises from 3.0 to 4.0 → new odds = 0.0743 × 2.226 = 0.1654\n  New P = 0.1654/(1+0.1654) = 14.2%",
     "key_results":[("z (log-odds)","−2.60"),("P(Default)","6.92%"),("Odds","0.0743"),("OR for D/E","2.226×")]},
    {"id":"NUM-LR-2","topic":"Model Fit","level":"Intermediate",
     "title":"McFadden R², LR Test & AIC",
     "question":"A logistic model for credit default has: n=300, k=4 predictors, null log-likelihood L_null=−185.4, full log-likelihood L_full=−126.8. Compute: (a) McFadden R², (b) LR test statistic, (c) AIC, (d) Is the model significant at α=1%? [χ²_crit(4,1%)=13.28]",
     "solution":"Step 1 — McFadden R²:\n  R² = 1 − L_full/L_null\n     = 1 − (−126.8)/(−185.4)\n     = 1 − 0.6839 = 0.3161\n  Interpretation: Excellent fit (0.20–0.40 = good)\n\nStep 2 — LR Test Statistic:\n  G² = −2[L_null − L_full]\n     = −2[(−185.4) − (−126.8)]\n     = −2(−58.6) = 117.2\n  G² ~ χ²(k) = χ²(4) under H₀: all β_j = 0\n\nStep 3 — AIC:\n  AIC = 2k − 2L_full = 2(4) − 2(−126.8) = 8 + 253.6 = 261.6\n  BIC = k·ln(n) − 2L_full = 4·ln(300) + 253.6 = 4(5.704)+253.6 = 276.4\n\nStep 4 — Significance:\n  G² = 117.2 >> χ²_crit(4,1%) = 13.28\n  REJECT H₀: All β_j = 0 at 1% significance level\n  At least one predictor significantly explains default",
     "key_results":[("McFadden R²","0.3161 — Excellent fit"),("LR Statistic G²","117.2"),("AIC","261.6"),("Decision","REJECT H₀ — model highly significant")]},
    {"id":"NUM-LR-3","topic":"Finance","level":"Intermediate",
     "title":"Confusion Matrix & Threshold Analysis",
     "question":"A PD model scores 400 borrowers: TP=36, TN=310, FP=22, FN=32. Compute: (a) Accuracy, (b) Sensitivity, (c) Specificity, (d) Precision, (e) F1 Score. Is this model suitable for a credit risk application?",
     "solution":"Given: TP=36, TN=310, FP=22, FN=32, N=400\n\nStep 1 — Accuracy:\n  Acc = (TP+TN)/N = (36+310)/400 = 346/400 = 86.5%\n\nStep 2 — Sensitivity (Recall/TPR):\n  Sens = TP/(TP+FN) = 36/(36+32) = 36/68 = 52.94%\n  ← Only 53% of actual defaults are caught\n\nStep 3 — Specificity:\n  Spec = TN/(TN+FP) = 310/(310+22) = 310/332 = 93.37%\n\nStep 4 — Precision (PPV):\n  Prec = TP/(TP+FP) = 36/(36+22) = 36/58 = 62.07%\n\nStep 5 — F1 Score:\n  F1 = 2×Prec×Recall/(Prec+Recall)\n     = 2×0.6207×0.5294/(0.6207+0.5294)\n     = 0.6575/1.1501 = 0.5718\n\nConclusion: Sensitivity = 53% is concerning for credit risk.\n  32 defaulters are classified as non-defaulting (FN=32).\n  Recommendation: Lower τ from 0.5 to ~0.3 to increase sensitivity,\n  accepting higher FP (more false alarms) to catch more defaults.",
     "key_results":[("Accuracy","86.5%"),("Sensitivity (Recall)","52.94% ← Concern"),("Specificity","93.37%"),("F1 Score","57.18%")]},
    {"id":"NUM-LR-4","topic":"Finance","level":"Advanced",
     "title":"IFRS 9 ECL Computation",
     "question":"A bank's logistic PD model produces: Stage 1 PD=1.8%, Stage 2 PD=12.5%. LGD=40%, EAD=₹5 crore. (a) Compute 12-month ECL (Stage 1), (b) Lifetime ECL approximation (Stage 2), (c) What triggers Stage 2 transfer?",
     "solution":"Step 1 — Stage 1: 12-month ECL:\n  ECL_1 = PD_12M × LGD × EAD\n         = 0.018 × 0.40 × 5,00,00,000\n         = 0.0072 × 5,00,00,000\n         = ₹3,60,000\n\nStep 2 — Stage 2: Lifetime ECL (simplified, 3-year avg):\n  ECL_2 ≈ PD_lifetime × LGD × EAD\n  PD_lifetime ≈ 1−(1−PD_annual)^3 = 1−(1−0.125)^3\n             = 1−(0.875)^3 = 1−0.6699 = 0.3301 = 33.01%\n  ECL_2 = 0.3301 × 0.40 × 5,00,00,000 = ₹66,02,000\n\n  ECL_2/ECL_1 = 66,02,000/3,60,000 = 18.3× increase\n  This illustrates the cliff effect of SICR reclassification\n\nStep 3 — Stage 2 Triggers (SICR — Significant Increase in Credit Risk):\n  • 30 days past due (backstop indicator)\n  • Internal rating downgrade (e.g., from BB to B)\n  • Watchlist placement\n  • Logistic PD increase > 3× initial PD at origination\n  • Macroeconomic stress indicators for portfolio segments",
     "key_results":[("Stage 1 ECL (12-month)","₹3,60,000"),("Stage 2 ECL (Lifetime)","₹66,02,000"),("ECL Multiplier","18.3× from SICR"),("SICR Trigger","30 DPD or rating downgrade or PD ≥ 3× origination")]},
]

FALLBACKS = {
    "sigmoid": "The sigmoid σ(z)=1/(1+e⁻ᶻ) maps z∈(−∞,+∞) → (0,1). Properties: σ(0)=0.5 (decision boundary), monotone increasing, symmetric: σ(−z)=1−σ(z). Derivative: σ'(z)=σ(z)(1−σ(z)), maximised at z=0. Key Takeaway: The sigmoid ensures all predicted probabilities lie in (0,1), making logistic regression valid for binary outcomes.",
    "odds ratio": "OR = exp(β). Interpretation: 1-unit increase in X multiplies odds by exp(β). OR > 1: risk factor (higher X → higher P). OR < 1: protective (higher X → lower P). 95% CI: exp(β ± 1.96×SE). Key Takeaway: Always interpret logistic coefficients as odds ratios, not probabilities.",
    "auc": "AUC = Area Under the ROC Curve = P(score(Y=1) > score(Y=0)). Range: 0.5 (random) to 1.0 (perfect). Interpretation: AUC=0.80 means model ranks a random positive above a random negative 80% of the time. Basel III requires AUC ≥ 0.75 for IRB models. Gini = 2×AUC−1. Key Takeaway: AUC measures discrimination, not calibration.",
    "hosmer": "Hosmer-Lemeshow test checks calibration: are predicted probabilities accurate? Groups observations into 10 deciles of predicted probability, then tests if observed event rates match. HL ~ χ²(8). H₀: well-calibrated. p > 0.05 = calibration OK. Key Takeaway: AUC measures ranking; HL measures accuracy of predicted probabilities. Both needed for Basel compliance.",
    "mle": "MLE maximises L(β)=Σ[Y·lnP+(1-Y)·ln(1-P)]. No closed form — solved iteratively using Newton-Raphson or IRLS. Information matrix = −∂²L/∂β² gives standard errors. MLE is consistent (converges to true β as n→∞) and asymptotically efficient. Key Takeaway: Unlike OLS, logistic MLE has no analytical solution — convergence should always be verified.",
    "mcfadden": "McFadden R² = 1 − L_full/L_null. Measures how much the full model improves over the intercept-only model. Range 0.20–0.40 = excellent (note: this is NOT comparable to OLS R²). Computed from log-likelihoods, not sum of squares. Key Takeaway: McFadden R²=0.25 is comparable in meaning to OLS R²≈0.50 — do not compare directly.",
}


def tab_qa():
    render_card("🎓 Self-Assessment — Logistic Regression in Finance",
        p(f'Test your understanding across {hl("Concepts")}, {hl("Model Fit")}, '
          f'{hl("Diagnostics")}, and {hl("Finance Applications")}. '
          f'Questions are CFA/FRM/MBA level.')
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📝 MCQ — {len(MCQ_BANK)} Questions</span><br>'
               + p(f'4 topics, 3 difficulty levels<br>Immediate explanations'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">🔢 Numericals — {len(NUMERICAL_BANK)} Problems</span><br>'
               + p(f'Sigmoid, ECL, Confusion Matrix<br>Step-by-step worked solutions'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">🤖 AI Tutor</span><br>'
               + p(f'Ask anything about logistic regression<br>Finance-focused explanations'), "green"),
        )
    )
    mode = st.radio("Mode", ["📝 MCQ Quiz","🔢 Numerical Problems","🤖 AI Tutor"], horizontal=True, key="qa_mode")
    if "MCQ" in mode:      _mcq_section()
    elif "Numerical" in mode: _num_section()
    else:                  _ai_section()


def _mcq_section():
    c1,c2,c3 = st.columns(3)
    topic_f = c1.selectbox("Topic",["All","Concepts","Model Fit","Diagnostics","Finance"], key="mcq_t")
    level_f = c2.selectbox("Level",["All","Foundation","Intermediate","Advanced"], key="mcq_l")
    mode_f  = c3.selectbox("Mode",["Study (show answer)","Quiz (hide answer)"], key="mcq_m")
    filtered = [q for q in MCQ_BANK
                if (topic_f=="All" or q["topic"]==topic_f)
                and (level_f=="All" or q["level"]==level_f)]
    if not filtered:
        render_ib(rt2("No questions match filters."), "red"); return
    if "mcq_score" not in st.session_state: st.session_state.mcq_score={}
    if "mcq_ans"   not in st.session_state: st.session_state.mcq_ans={}
    correct   = sum(1 for q in filtered if st.session_state.mcq_score.get(q["id"])==True)
    attempted = sum(1 for q in filtered if q["id"] in st.session_state.mcq_ans)
    if attempted > 0:
        pct = correct/attempted*100
        col = "#28a745" if pct>=70 else "#ff9f43" if pct>=50 else "#dc3545"
        st.html(f'<div style="background:rgba(0,51,102,0.5);border:1px solid #1e3a5f;border-radius:8px;padding:12px 18px;margin-bottom:14px;display:flex;align-items:center;gap:18px;{NO_SEL}">'
                f'<span style="color:{col};-webkit-text-fill-color:{col};font-family:{FM};font-size:1.5rem;font-weight:700">{correct}/{attempted}</span>'
                f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB}">{pct:.0f}% correct from {len(filtered)} available</span>'
                f'<span style="margin-left:auto">{bdg("Excellent","green") if pct>=80 else bdg("Good","gold") if pct>=60 else bdg("Keep practising","red")}</span>'
                f'</div>')
    if st.button("🔄 Reset", key="mcq_reset"):
        for q in filtered:
            st.session_state.mcq_score.pop(q["id"],None); st.session_state.mcq_ans.pop(q["id"],None)
        st.rerun()
    for idx, q in enumerate(filtered): _render_mcq(q, idx, "Quiz" in mode_f)


def _render_mcq(q, idx, hide):
    lc = {"Foundation":"#28a745","Intermediate":"#FFD700","Advanced":"#dc3545"}.get(q["level"],"#ADD8E6")
    answered = q["id"] in st.session_state.get("mcq_ans",{})
    is_corr  = st.session_state.get("mcq_score",{}).get(q["id"])
    hdr_bg   = ("rgba(40,167,69,0.15)" if is_corr else "rgba(220,53,69,0.12)") if answered else "#112240"
    st.html(
        f'<div style="background:{hdr_bg};border:1px solid #1e3a5f;border-radius:10px;padding:16px 18px;margin-bottom:4px;{NO_SEL}">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">'
        f'{bdg(q["topic"],"blue")} <span style="color:{lc};-webkit-text-fill-color:{lc};font-size:.78rem;font-weight:700;font-family:{FB}">{q["level"]}</span>'
        f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.75rem;font-family:{FB};margin-left:auto">{q["id"]}</span></div>'
        f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB};font-size:.97rem;line-height:1.6">{q["question"]}</div>'
        f'</div>'
    )
    choice = st.radio(f"Q{idx+1}", q["options"], index=None, key=f"mcq_{q['id']}", label_visibility="collapsed")
    if choice is not None:
        ci = q["options"].index(choice); corr = ci == q["answer"]
        st.session_state.setdefault("mcq_score",{})[q["id"]] = corr
        st.session_state.setdefault("mcq_ans",{})[q["id"]] = ci
        if not hide:
            if corr:
                st.html(ib(gt("✅ Correct! ") + txt_s(q["explanation"]), "green"))
            else:
                st.html(ib(rt2("✗ Incorrect. ") + f'<strong style="color:#FFD700;-webkit-text-fill-color:#FFD700">Correct: {q["options"][q["answer"]]}</strong><br><br>' + txt_s(q["explanation"]), "red"))
    st.html('<div style="margin-bottom:10px"></div>')


def _num_section():
    c1,c2 = st.columns(2)
    topic_n = c1.selectbox("Topic",["All","Concepts","Model Fit","Finance"], key="num_t")
    level_n = c2.selectbox("Level",["All","Foundation","Intermediate","Advanced"], key="num_l")
    filtered = [q for q in NUMERICAL_BANK
                if (topic_n=="All" or q["topic"]==topic_n)
                and (level_n=="All" or q["level"]==level_n)]
    if not filtered:
        render_ib(rt2("No problems match filters."), "red"); return
    for prob in filtered:
        lc = {"Foundation":"#28a745","Intermediate":"#FFD700","Advanced":"#dc3545"}.get(prob["level"],"#ADD8E6")
        st.html(
            f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;padding:16px 18px;margin-bottom:4px;{NO_SEL}">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:9px">'
            f'{bdg(prob["topic"],"blue")} <span style="color:{lc};-webkit-text-fill-color:{lc};font-size:.78rem;font-weight:700;font-family:{FB}">{prob["level"]}</span>'
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-family:{FH};font-size:1.0rem;margin-left:8px">{prob["title"]}</span>'
            f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.75rem;font-family:{FB};margin-left:auto">{prob["id"]}</span></div>'
            f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB};font-size:.95rem;line-height:1.65">{prob["question"]}</div>'
            f'</div>'
        )
        sk = f"show_{prob['id']}"
        if sk not in st.session_state: st.session_state[sk]=False
        if st.button("💡 Show Solution", key=f"btn_{prob['id']}"):
            st.session_state[sk] = not st.session_state[sk]
        if st.session_state[sk]:
            rows = [[hl(k), txt_s(v)] for k,v in prob["key_results"]]
            st.html('<div style="margin-top:10px">'+table_html(["Result","Value"],rows)+'</div>')
            st.html(ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📐 Full Solution:</span>' + fml(prob["solution"]), "gold"))
        st.html('<div style="margin-bottom:12px"></div>')


def _ai_section():
    render_card("🤖 AI Tutor — Ask Anything About Logistic Regression",
        ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">Powered by Claude AI.</span> '
           + txt_s("Finance-focused explanations at CFA/FRM/MBA level."), "blue")
    )
    quick_qs = ["Explain odds ratio interpretation","What is AUC and how is it calculated?",
                "How does MLE work in logistic regression?","Explain the confusion matrix metrics",
                "What is complete separation and how to fix it?","When to use logistic vs linear regression?"]
    cols = st.columns(3)
    for i, qq in enumerate(quick_qs):
        if cols[i%3].button(qq, key=f"qq_{i}", use_container_width=True):
            st.session_state["ai_q"] = qq; st.rerun()
    if "chat_lr" not in st.session_state: st.session_state.chat_lr=[]
    for msg in st.session_state.chat_lr:
        color = "#ADD8E6" if msg["role"]=="user" else "#FFD700"
        label = "YOU" if msg["role"]=="user" else "AI TUTOR"
        border = "#ADD8E6" if msg["role"]=="user" else "#FFD700"
        bg = "rgba(0,77,128,0.4)" if msg["role"]=="user" else "rgba(255,215,0,0.07)"
        st.html(f'<div style="background:{bg};border-left:4px solid {border};border-radius:8px;padding:12px 15px;margin:8px 0;{NO_SEL}">'
                f'<span style="color:{color};-webkit-text-fill-color:{color};font-weight:600;font-size:.8rem">{label}</span><br>'
                f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB};line-height:1.7">{msg["content"]}</span></div>')
    default_q = st.session_state.pop("ai_q","")
    question  = st.text_input("Ask about logistic regression...", value=default_q, key="ai_input_lr",
                               placeholder="e.g. What is the difference between AUC and F1 score?")
    c1,c2 = st.columns([1,5])
    if c1.button("🤖 Ask", key="ai_ask"): 
        if question.strip():
            st.session_state.chat_lr.append({"role":"user","content":question})
            with st.spinner("Thinking..."):
                ans = _call_claude(question, st.session_state.chat_lr[:-1])
            st.session_state.chat_lr.append({"role":"assistant","content":ans})
            st.rerun()
    if c2.button("🗑 Clear", key="ai_clear"):
        st.session_state.chat_lr=[]; st.rerun()


def _call_claude(question, history):
    import json, urllib.request
    system = """You are an expert finance professor specialising in logistic regression for financial applications.
Your students are MBA/CFA/FRM candidates. When answering:
- Be precise and exam-ready with formulas, conditions, interpretations
- Ground examples in finance: credit risk, PD models, fraud detection, Basel III, IFRS 9
- For numerical questions, show clear step-by-step workings
- Use plain text formatting (no markdown like ** or ##)
- Keep responses 150–300 words unless a worked example is needed
- End with a one-line 'Key Takeaway:' summary"""
    msgs = [{"role":h["role"],"content":h["content"]} for h in history[-6:]]
    msgs.append({"role":"user","content":question})
    try:
        payload = json.dumps({"model":"claude-sonnet-4-20250514","max_tokens":1000,"system":system,"messages":msgs}).encode()
        req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=payload,
            headers={"Content-Type":"application/json","anthropic-version":"2023-06-01"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["content"][0]["text"]
    except Exception as e:
        for kw, ans in FALLBACKS.items():
            if kw in question.lower(): return ans
        return f"AI service unavailable ({str(e)[:60]}). Check the MCQ explanations and worked solutions for this topic."
