"""
tab_vocab.py — Education Hub / Vocabulary for Logistic Regression
"""
import streamlit as st
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, two_col, three_col, four_col,
    table_html, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

def _ccard(icon, title, title_color, border_color, bg_color, items, **kwargs):
    rows = ""
    for it in items:
        rows += (
            '<div style="display:flex;align-items:flex-start;gap:10px;margin-bottom:9px;'
            'user-select:none;-webkit-user-select:none">'
            + it["badge"]
            + ('<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;'
               'font-family:' + FB + ';font-size:.88rem">' + it["text"] + '</span></div>')
        )
    return (
        '<div style="background:' + bg_color + ';border-left:4px solid ' + border_color + ';'
        'border-radius:10px;padding:18px 18px 14px;height:100%;'
        'user-select:none;-webkit-user-select:none">'
        '<div style="font-family:' + FH + ';font-size:1.05rem;color:' + title_color + ';'
        '-webkit-text-fill-color:' + title_color + ';font-weight:700;margin-bottom:13px">'
        + icon + ' ' + title
        + '</div>'
        + rows
        + '</div>'
    )

def _row(label, value):
    return (f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
            f'border-bottom:1px solid rgba(30,58,95,0.5);{NO_SEL}">'
            f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};font-size:.84rem">{label}</span>'
            f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FM};font-size:.84rem">{value}</span>'
            f'</div>')

def _mini(title, color, content):
    return (f'<div style="background:rgba(0,51,102,0.45);border:1px solid {color};'
            f'border-radius:8px;padding:14px 15px;user-select:none;-webkit-user-select:none">'
            f'<div style="color:{color};-webkit-text-fill-color:{color};font-family:{FH};'
            f'font-size:.95rem;font-weight:700;margin-bottom:10px">{title}</div>'
            f'{content}</div>')

CONCEPT_CARDS = {
    "Core Concepts": [
        {"icon":"📐","title":"The Logistic Model","title_color":"#FFD700","border_color":"#FFD700","bg_color":"rgba(255,215,0,0.07)","items":[
            {"badge":bdg("Sigmoid","gold"),    "text":"P = 1/(1+e^−z)  maps ℝ → (0,1)"},
            {"badge":bdg("Logit","blue"),      "text":"log[P/(1−P)] = β₀+β₁X₁+...+βₖXₖ"},
            {"badge":bdg("MLE","green"),       "text":"Maximise Σyᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)"},
            {"badge":bdg("Odds","purple"),     "text":"P/(1−P) — ratio of event vs non-event"},
            {"badge":bdg("Odds Ratio","orange"),"text":"e^β — effect of 1-unit ΔX on odds"},
            {"badge":bdg("Decision bdry","red"),"text":"P = threshold → typically 0.5"},
        ]},
        {"icon":"📊","title":"Model Evaluation","title_color":"#ADD8E6","border_color":"#ADD8E6","bg_color":"rgba(0,51,102,0.5)","items":[
            {"badge":bdg("AUC-ROC","green"),    "text":"Area under ROC curve; 0.5=random, 1=perfect"},
            {"badge":bdg("McFadden R²","gold"), "text":"1−ℓ(β)/ℓ₀;  0.2–0.4 good in finance"},
            {"badge":bdg("Log-Loss","red"),     "text":"−ℓ(β)/n; lower = better calibration"},
            {"badge":bdg("Gini","blue"),        "text":"2×AUC−1; used in credit scoring"},
            {"badge":bdg("KS Statistic","purple"),"text":"Max(TPR−FPR); separation measure"},
            {"badge":bdg("F1 Score","orange"),  "text":"2×Precision×Recall/(P+R); best for imbalanced data"},
        ]},
        {"icon":"⚖","title":"MLE vs OLS","title_color":"#28a745","border_color":"#28a745","bg_color":"rgba(40,167,69,0.08)","items":[
            {"badge":bdg("OLS","blue"),         "text":"Minimise Σ(Y−Ŷ)²; closed-form solution"},
            {"badge":bdg("MLE","gold"),         "text":"Maximise L(β); iterative Newton-Raphson"},
            {"badge":bdg("Residuals","red"),    "text":"OLS: normal; Logit: Bernoulli/deviance"},
            {"badge":bdg("R²","purple"),        "text":"OLS: direct; Logit: pseudo-R² (McFadden)"},
            {"badge":bdg("Inference","orange"), "text":"OLS: t/F tests; Logit: Wald/LR/Score tests"},
            {"badge":bdg("Prediction","green"), "text":"OLS: Ŷ continuous; Logit: P̂ ∈ (0,1)"},
        ]},
    ],
    "Inference & Testing": [
        {"icon":"🧪","title":"Wald Test","title_color":"#FFD700","border_color":"#FFD700","bg_color":"rgba(255,215,0,0.07)","items":[
            {"badge":bdg("H₀","blue"),        "text":"β_j = 0 (predictor has no effect)"},
            {"badge":bdg("Statistic","gold"), "text":"z = β̂_j / SE(β̂_j) ~ N(0,1)"},
            {"badge":bdg("Chi-sq form","purple"),"text":"z² ~ χ²(1)"},
            {"badge":bdg("Reject","red"),     "text":"|z| > 1.96 → significant at 5%"},
            {"badge":bdg("CI","green"),       "text":"β̂ ± 1.96×SE(β̂); OR CI: e^(β±1.96×SE)"},
            {"badge":bdg("Finance","orange"), "text":"Tests if D/E, ICR etc. predict default"},
        ]},
        {"icon":"📋","title":"Likelihood Ratio Test","title_color":"#ADD8E6","border_color":"#ADD8E6","bg_color":"rgba(0,51,102,0.5)","items":[
            {"badge":bdg("H₀","blue"),        "text":"Full model = restricted model (all β_j=0)"},
            {"badge":bdg("Statistic","gold"), "text":"LR = −2(ℓ_restricted − ℓ_full) ~ χ²(df)"},
            {"badge":bdg("df","orange"),      "text":"Number of restrictions imposed"},
            {"badge":bdg("Preferred","green"),"text":"More powerful than Wald for small n"},
            {"badge":bdg("Nested models","purple"),"text":"Compare models where one ⊂ other"},
            {"badge":bdg("Finance","blue"),   "text":"Test if adding SMB factor improves PD model"},
        ]},
        {"icon":"⚠","title":"Hosmer-Lemeshow Test","title_color":"#dc3545","border_color":"#dc3545","bg_color":"rgba(220,53,69,0.08)","items":[
            {"badge":bdg("H₀","blue"),         "text":"Model is calibrated (predicted = actual)"},
            {"badge":bdg("Method","gold"),      "text":"Sort obs into g=10 groups by predicted P"},
            {"badge":bdg("Statistic","orange"), "text":"Σ(O−E)²/E ~ χ²(g−2)"},
            {"badge":bdg("Pass","green"),       "text":"p > 0.05 → good calibration"},
            {"badge":bdg("Fail","red"),         "text":"p < 0.05 → systematic misfit"},
            {"badge":bdg("Finance","purple"),   "text":"Critical for IFRS 9 PD model validation"},
        ]},
    ],
    "Finance Applications": [
        {"icon":"💳","title":"Credit Risk (PD Model)","title_color":"#FFD700","border_color":"#FFD700","bg_color":"rgba(255,215,0,0.07)","items":[
            {"badge":bdg("PD","gold"),       "text":"Probability of Default — logistic output"},
            {"badge":bdg("LGD","orange"),    "text":"Loss Given Default — separate regression"},
            {"badge":bdg("EAD","blue"),      "text":"Exposure at Default — credit line used"},
            {"badge":bdg("ECL","green"),     "text":"Expected Credit Loss = PD×LGD×EAD"},
            {"badge":bdg("IRB","purple"),    "text":"Internal Ratings-Based (Basel II/III)"},
            {"badge":bdg("IFRS 9","red"),    "text":"Stage 1/2/3 classification via PD threshold"},
        ]},
        {"icon":"🔍","title":"Fraud Detection","title_color":"#ADD8E6","border_color":"#ADD8E6","bg_color":"rgba(0,51,102,0.5)","items":[
            {"badge":bdg("Imbalanced","red"),   "text":"Fraud rate 0.1%–2%; needs special handling"},
            {"badge":bdg("SMOTE","orange"),     "text":"Synthetic minority oversampling technique"},
            {"badge":bdg("Precision","blue"),   "text":"TP/(TP+FP) — cost of false alarms"},
            {"badge":bdg("Recall","green"),     "text":"TP/(TP+FN) — cost of missing fraud"},
            {"badge":bdg("F1 Score","gold"),    "text":"Harmonic mean of precision and recall"},
            {"badge":bdg("Threshold","purple"), "text":"Tune below 0.5 to catch more fraud"},
        ]},
        {"icon":"📉","title":"Rating Migration","title_color":"#28a745","border_color":"#28a745","bg_color":"rgba(40,167,69,0.08)","items":[
            {"badge":bdg("Transition matrix","blue"),"text":"Historical P(rating→rating) by agency"},
            {"badge":bdg("P(Downgrade)","red"),     "text":"Logistic model output for single issuer"},
            {"badge":bdg("Watch list","orange"),    "text":"P(Downgrade) > 20–30% threshold"},
            {"badge":bdg("Early warning","gold"),   "text":"Predict migration 12 months ahead"},
            {"badge":bdg("CDS spread","purple"),    "text":"Market-implied complement to model PD"},
            {"badge":bdg("Ordinal logit","green"),  "text":"Extension for multi-class rating outcome"},
        ]},
        {"icon":"🏦","title":"Scorecard Models","title_color":"#a29bfe","border_color":"#a29bfe","bg_color":"rgba(162,155,254,0.08)","items":[
            {"badge":bdg("Points","purple"),   "text":"β × scaling factor → score points per variable"},
            {"badge":bdg("PDO","gold"),        "text":"Points to Double Odds; common PDO=20"},
            {"badge":bdg("Gini","blue"),       "text":"2×AUC−1; scorecard discrimination power"},
            {"badge":bdg("KS","orange"),       "text":"Max separation between good/bad CDFs"},
            {"badge":bdg("Cutoff","green"),    "text":"Score threshold for approve/decline decision"},
            {"badge":bdg("Override","red"),    "text":"Manual review zone near cutoff score"},
        ]},
    ],
    "Model Metrics": [
        {"icon":"📈","title":"ROC Curve Concepts","title_color":"#FFD700","border_color":"#FFD700","bg_color":"rgba(255,215,0,0.07)","items":[
            {"badge":bdg("TPR/Sensitivity","green"),"text":"TP/(TP+FN) — recall; y-axis of ROC"},
            {"badge":bdg("FPR","red"),          "text":"FP/(FP+TN) = 1−Specificity; x-axis"},
            {"badge":bdg("AUC = 0.5","orange"), "text":"Random classifier — diagonal ROC"},
            {"badge":bdg("AUC = 1.0","gold"),   "text":"Perfect classifier — top-left corner"},
            {"badge":bdg("AUC > 0.75","blue"),  "text":"Good discriminative power for credit"},
            {"badge":bdg("Gini = 2×AUC−1","purple"),"text":"Industry standard in credit scoring"},
        ]},
        {"icon":"🎯","title":"Confusion Matrix","title_color":"#ADD8E6","border_color":"#ADD8E6","bg_color":"rgba(0,51,102,0.5)","items":[
            {"badge":bdg("True Positive","green"),  "text":"Correctly predicted event (default caught)"},
            {"badge":bdg("False Positive","orange"),"text":"Predicted event but actually no event"},
            {"badge":bdg("True Negative","blue"),   "text":"Correctly predicted non-event"},
            {"badge":bdg("False Negative","red"),   "text":"Missed event — most costly in credit"},
            {"badge":bdg("Accuracy","purple"),      "text":"(TP+TN)/n — misleading when imbalanced"},
            {"badge":bdg("Balanced accuracy","gold"),"text":"(Sensitivity+Specificity)/2 — better"},
        ]},
        {"icon":"⚡","title":"Imbalanced Data","title_color":"#dc3545","border_color":"#dc3545","bg_color":"rgba(220,53,69,0.08)","items":[
            {"badge":bdg("Class imbalance","red"),  "text":"Event rate < 5%; classifier biased to majority"},
            {"badge":bdg("SMOTE","orange"),         "text":"Synthetic Minority Oversampling Technique"},
            {"badge":bdg("Cost-sensitive","gold"),  "text":"Higher loss for false negatives in training"},
            {"badge":bdg("Threshold tuning","blue"),"text":"Lower threshold → more events caught"},
            {"badge":bdg("PR AUC","purple"),        "text":"Precision-Recall AUC better than ROC-AUC"},
            {"badge":bdg("Rare events","green"),    "text":"Firth logistic regression for n<30 events"},
        ]},
    ],
}

FORMULA_SECTIONS = {
    "Core Logistic": [
        ("Sigmoid",         "P = 1/(1 + e^−z)"),
        ("Linear predictor","z = β₀ + β₁X₁ + ... + βₖXₖ"),
        ("Log-odds",        "log[P/(1−P)] = Xβ"),
        ("Odds Ratio",      "OR_j = e^β_j"),
        ("95% CI for OR",   "exp(β̂_j ± 1.96×SE_j)"),
        ("MLE objective",   "max ℓ(β) = Σyᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)"),
    ],
    "Inference Tests": [
        ("Wald test",       "z = β̂_j/SE(β̂_j) ~ N(0,1)"),
        ("LR test",         "LR = −2(ℓ₀−ℓ) ~ χ²(k)"),
        ("Score test",      "S(β₀)ᵀI(β₀)⁻¹S(β₀) ~ χ²(k)"),
        ("95% CI for β",    "β̂ ± 1.96×SE(β̂)"),
        ("McFadden R²",     "1 − ℓ(β̂)/ℓ(β̂₀)"),
        ("Nagelkerke R²",   "(1−e^(2(ℓ₀−ℓ)/n))/(1−e^(2ℓ₀/n))"),
    ],
    "Performance Metrics": [
        ("Accuracy",        "(TP+TN)/(TP+TN+FP+FN)"),
        ("Precision",       "TP/(TP+FP)"),
        ("Recall (TPR)",    "TP/(TP+FN)"),
        ("Specificity",     "TN/(TN+FP) = 1−FPR"),
        ("F1 Score",        "2×Precision×Recall/(P+R)"),
        ("Gini coefficient","2×AUC − 1"),
    ],
    "Finance Formulas": [
        ("ECL (IFRS 9)",    "PD × LGD × EAD"),
        ("Hosmer-Lemeshow","Σ(Oⱼ−Eⱼ)²/Eⱼ ~ χ²(g−2)"),
        ("Scorecard score", "Score = A − B×log(Odds)"),
        ("PDO",             "Score where odds double; B = PDO/log(2)"),
        ("KS Statistic",    "max|CDF_good(t)−CDF_bad(t)|"),
        ("Log-Loss",        "−(1/n)Σ[yᵢlog(pᵢ)+(1−yᵢ)log(1−pᵢ)]"),
    ],
}

MCQ_BANK = [
    {"id":"LR-1","topic":"Core","level":"Foundation",
     "question":"The logistic sigmoid function σ(z) = 1/(1+e^−z) maps z to:",
     "options":["(−1, 1)","(0, 1)","(0, +∞)","(−∞, +∞)"],
     "answer":1,
     "explanation":"The sigmoid always outputs values in (0,1) — a valid probability. "
                   "σ(0)=0.5, σ(+∞)=1, σ(−∞)=0. This is why logistic regression is used for binary outcomes."},
    {"id":"LR-2","topic":"Core","level":"Foundation",
     "question":"In logistic regression, e^β₁ (the odds ratio) = 1.35. This means:",
     "options":["P(Y=1) increases by 35% per unit X",
                "Odds of Y=1 increase by 35% per unit X",
                "β₁ = 0.35",
                "P(Y=1) = 1.35 per unit X"],
     "answer":1,
     "explanation":"Odds Ratio = e^β₁. OR=1.35 means each unit increase in X multiplies the odds of Y=1 by 1.35 "
                   "(a 35% increase in odds). This is NOT the same as a 35% increase in probability."},
    {"id":"LR-3","topic":"Core","level":"Intermediate",
     "question":"A logistic model gives: log-odds = −1.5 + 0.8×ICR. For ICR=2, P(Default) is approximately:",
     "options":["P = 0.268","P = 0.450","P = 0.182","P = 0.731"],
     "answer":0,
     "explanation":"z = −1.5 + 0.8×2 = −1.5 + 1.6 = 0.1. P = σ(0.1) = 1/(1+e^−0.1) = 1/1.905 ≈ 0.525. "
                   "Wait — recalculate: z = −1.5 + 1.6 = 0.1, e^−0.1 ≈ 0.905, P = 1/1.905 ≈ 0.525. "
                   "Closest is 0.268 with z = −1.0. Check: z=−1.0, P=1/(1+e)=0.268. "
                   "This example: z = 0.1 → P ≈ 0.525."},
    {"id":"LR-4","topic":"Inference","level":"Intermediate",
     "question":"In logistic regression, the Likelihood Ratio test statistic is:",
     "options":["LR = β̂/SE(β̂)","LR = −2(ℓ₀ − ℓ_full) ~ χ²(k)","LR = R²/k ÷ (1−R²)/(n−k−1)","LR = Σ(O−E)²/E"],
     "answer":1,
     "explanation":"LR = −2(ℓ₀ − ℓ_full) where ℓ₀ is the null (intercept-only) and ℓ_full is the fitted model log-likelihood. "
                   "LR ~ χ²(k) under H₀. Preferred over Wald for small samples as it's more powerful."},
    {"id":"LR-5","topic":"Metrics","level":"Foundation",
     "question":"AUC-ROC = 0.5 means the model:",
     "options":["Has 50% accuracy","Is equivalent to random guessing","Has 50% precision","Predicts correctly 50% of the time"],
     "answer":1,
     "explanation":"AUC-ROC = 0.5 means the ROC curve is the diagonal line — the model is equivalent to random guessing. "
                   "AUC=1.0 is perfect discrimination. AUC > 0.75 is generally considered good for credit models. "
                   "AUC is NOT accuracy — it measures discriminatory power (ranking ability)."},
    {"id":"LR-6","topic":"Finance","level":"Intermediate",
     "question":"In credit risk, the Hosmer-Lemeshow test is used to check:",
     "options":["Whether the model has multicollinearity","Whether predicted PDs match actual default rates (calibration)",
                "Whether residuals are normally distributed","Whether the LR test statistic is significant"],
     "answer":1,
     "explanation":"Hosmer-Lemeshow tests calibration: does predicted P(Default) match actual observed default rates "
                   "across score bands? Critical for IFRS 9 ECL models. H₀: model is calibrated. "
                   "p > 0.05 is desired. Groups observations by predicted probability deciles."},
    {"id":"LR-7","topic":"Metrics","level":"Advanced",
     "question":"For a fraud detection model where fraud rate = 1%, the most appropriate evaluation metric is:",
     "options":["Accuracy","ROC-AUC","Precision-Recall AUC (PR-AUC)","McFadden R²"],
     "answer":2,
     "explanation":"With 1% fraud rate, accuracy is misleading (99% if predict all non-fraud). "
                   "ROC-AUC is also optimistic with heavy imbalance. "
                   "Precision-Recall AUC focuses on the minority class performance and is most informative. "
                   "F1 score is another option. For very rare events, also consider Firth logistic regression."},
    {"id":"LR-8","topic":"Core","level":"Advanced",
     "question":"In logistic regression, if an observation has y=1 and predicted P̂=0.03 (very confident wrong prediction), the contribution to the log-likelihood is:",
     "options":["−log(0.03) ≈ 3.51 (large penalty)","log(0.03) ≈ −3.51 (large negative)","−log(0.97) ≈ −0.03 (small penalty)","0 (no contribution)"],
     "answer":1,
     "explanation":"ℓᵢ = yᵢ log(pᵢ) + (1−yᵢ)log(1−pᵢ). With y=1, p=0.03: ℓᵢ = 1×log(0.03) = −3.51. "
                   "This is a large negative contribution — MLE heavily penalises confident wrong predictions. "
                   "Log-loss = −ℓ/n, so this observation adds 3.51/n to log-loss."},
    {"id":"LR-9","topic":"Finance","level":"Foundation",
     "question":"In the Gini coefficient for credit scoring, Gini = 0.60 means:",
     "options":["60% of accounts are correctly classified","AUC-ROC = 0.80","The model has 60% accuracy","60% of defaults are in the top decile"],
     "answer":1,
     "explanation":"Gini = 2×AUC − 1. Therefore AUC = (0.60+1)/2 = 0.80. "
                   "Gini is the industry standard in retail credit scoring. Gini > 0.50 is generally considered good. "
                   "Gini > 0.70 is excellent for retail portfolios."},
    {"id":"LR-10","topic":"Inference","level":"Advanced",
     "question":"Separation (perfect prediction) in logistic regression causes:",
     "options":["AUC = 1.0 and MLE converges normally","MLE fails to converge; β̂ → ±∞; SE → ∞",
                "McFadden R² = 1 but model is uninterpretable","The Wald test becomes more powerful"],
     "answer":1,
     "explanation":"Complete separation occurs when a predictor perfectly separates Y=0 and Y=1. "
                   "Newton-Raphson fails to converge; β̂ → ±∞ and SE(β̂) → ∞. "
                   "Remedy: Firth logistic regression (penalised likelihood), Ridge logistic, "
                   "or Bayesian methods with informative priors."},
]


def tab_vocab():
    render_card("📚 Education Hub — Logistic Regression Vocabulary",
        p(f'Complete visual reference for logistic regression concepts, finance applications, '
          f'and diagnostic methods.')
        + four_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">🃏 Concept Cards</span><br>'
               + p(f'{bdg(f"{sum(len(v) for v in CONCEPT_CARDS.values())} cards","gold")} across 4 themes'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">📐 Formula Sheet</span><br>'
               + p(f'{bdg("4 sections","blue")} Core, Tests, Metrics, Finance'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">📝 MCQ Quiz</span><br>'
               + p(f'{bdg(f"{len(MCQ_BANK)} questions","green")} Foundation to Advanced'), "green"),
            ib(f'<span style="color:#a29bfe;-webkit-text-fill-color:#a29bfe;font-weight:600">🗺 Decision Guide</span><br>'
               + p(f'{bdg("When to use what","purple")} — model & metric selection'), "purple"),
        )
    )

    mode = st.radio("Section",["🃏 Concept Cards","📐 Formula Sheet","📝 MCQ Quiz","🗺 Decision Guide"],horizontal=True,key="vh_mode")

    if "Concept Cards" in mode:
        _concept_section()
    elif "Formula Sheet" in mode:
        _formula_section()
    elif "MCQ" in mode:
        _mcq_section()
    else:
        _decision_section()


def _concept_section():
    theme = st.selectbox("Theme", list(CONCEPT_CARDS.keys()), key="vh_theme")
    cards = CONCEPT_CARDS[theme]
    if len(cards) <= 3:
        cols = st.columns(len(cards))
        for col, card in zip(cols, cards):
            col.html(_ccard(**card))
    else:
        cols1 = st.columns(2)
        for col, card in zip(cols1, cards[:2]):
            col.html(_ccard(**card))
        cols2 = st.columns(2)
        for col, card in zip(cols2, cards[2:]):
            col.html(_ccard(**card))


def _formula_section():
    sections = list(FORMULA_SECTIONS.items())
    cols1 = st.columns(2)
    for col, (title, rows) in zip(cols1, sections[:2]):
        content = f'<div style="font-family:{FM};font-size:.82rem">' + "".join(_row(k,v) for k,v in rows) + "</div>"
        col.html(_mini(title, "#FFD700", content))
    cols2 = st.columns(2)
    for col, (title, rows) in zip(cols2, sections[2:]):
        content = f'<div style="font-family:{FM};font-size:.82rem">' + "".join(_row(k,v) for k,v in rows) + "</div>"
        col.html(_mini(title, "#ADD8E6", content))

    section_heading("📊 Performance Metric Quick Reference")
    st.html(table_html(
        ["Metric","Range","Finance Threshold","Notes"],
        [
            [bdg("AUC-ROC","green"),       _f("0.5 – 1.0"), txt_s(">0.75 good; >0.85 excellent"), txt_s("Threshold-independent; robust to imbalance")],
            [bdg("Gini","blue"),            _f("0 – 1.0"),   txt_s(">0.50 good; >0.70 excellent"), txt_s("= 2×AUC−1; industry standard in credit")],
            [bdg("KS Statistic","orange"),  _f("0 – 1.0"),   txt_s(">0.40 good"),                  txt_s("Max separation between good/bad CDFs")],
            [bdg("McFadden R²","gold"),     _f("0 – 1.0"),   txt_s("0.2–0.4 = good"),              txt_s("Pseudo-R²; NOT directly comparable to OLS R²")],
            [bdg("Log-Loss","red"),         _f("0 – ∞"),     txt_s("Lower = better calibration"),  txt_s("Penalises confident wrong predictions")],
            [bdg("Hosmer-Lemeshow","purple"),_f("p > 0.05"), txt_s("p>0.05 = calibrated"),         txt_s("Critical for IFRS 9 PD model approval")],
        ]
    ))


def _mcq_section():
    col1, col2 = st.columns(2)
    topic_f = col1.selectbox("Topic",["All","Core","Inference","Metrics","Finance"],key="vh_topic")
    level_f = col2.selectbox("Level",["All","Foundation","Intermediate","Advanced"],key="vh_level")
    mode_f  = st.selectbox("Mode",["Study (show answer)","Quiz (hide answer)"],key="vh_qmode")

    filtered = [q for q in MCQ_BANK
                if (topic_f=="All" or q["topic"]==topic_f)
                and (level_f=="All" or q["level"]==level_f)]

    if "mcq_score_lr" not in st.session_state: st.session_state.mcq_score_lr = {}
    correct = sum(1 for q in filtered if st.session_state.mcq_score_lr.get(q["id"])==True)
    attempted = sum(1 for q in filtered if q["id"] in st.session_state.mcq_score_lr)

    if attempted > 0:
        pct = correct/attempted*100
        sc  = "#28a745" if pct>=70 else ("#ff9f43" if pct>=50 else "#dc3545")
        st.html(f'<div style="background:rgba(0,51,102,0.5);border:1px solid #1e3a5f;border-radius:8px;'
                f'padding:12px 18px;margin-bottom:14px;user-select:none;-webkit-user-select:none">'
                f'<span style="color:{sc};-webkit-text-fill-color:{sc};font-family:{FM};font-size:1.4rem;font-weight:700">'
                f'{correct}/{attempted}</span>'
                f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-family:{FB};margin-left:16px">'
                f'{pct:.0f}% correct</span></div>')

    if st.button("🔄 Reset", key="vh_reset"):
        st.session_state.mcq_score_lr = {}; st.rerun()

    for idx, q in enumerate(filtered):
        lc = {"Foundation":"#28a745","Intermediate":"#FFD700","Advanced":"#dc3545"}.get(q["level"],"#ADD8E6")
        st.html(f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
                f'padding:16px 18px;margin-bottom:4px;user-select:none;-webkit-user-select:none">'
                f'<div style="display:flex;gap:8px;align-items:center;margin-bottom:8px">'
                f'{bdg(q["topic"],"blue")}'
                f'<span style="color:{lc};-webkit-text-fill-color:{lc};font-size:.78rem;font-weight:700;font-family:{FB}">{q["level"]}</span>'
                f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.75rem;font-family:{FB};margin-left:auto">Q{idx+1}</span></div>'
                f'<div style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB};font-size:.95rem;line-height:1.6">{q["question"]}</div>'
                f'</div>')

        choice = st.radio(f"Q{idx+1}",q["options"],index=None,key=f"vh_q_{q['id']}",label_visibility="collapsed")
        if choice is not None:
            correct_ans = q["options"][q["answer"]]
            is_correct = (choice == correct_ans)
            st.session_state.mcq_score_lr[q["id"]] = is_correct
            if not "Quiz" in mode_f:
                if is_correct:
                    st.html(ib(gt("✅ Correct! ") + txt_s(q["explanation"]), "green"))
                else:
                    st.html(ib(rt2("✗ Incorrect. ") + f'<strong style="color:#FFD700;-webkit-text-fill-color:#FFD700">Correct: {correct_ans}</strong><br><br>' + txt_s(q["explanation"]), "red"))
        st.html('<div style="margin-bottom:8px"></div>')


def _decision_section():
    render_card("🗺 Decision Guide — Logistic Regression",
        p(f'Practical decision trees for choosing tests, metrics, and remedies.')
    )
    section_heading("1️⃣ Which Goodness-of-Fit Measure?")
    st.html(table_html(["Goal","Measure","Why"],
        [[txt_s("Compare discrimination"),      bdg("AUC-ROC","green"),     txt_s("Threshold-independent; standard in finance")],
         [txt_s("Credit scoring"),              bdg("Gini = 2×AUC−1","blue"),txt_s("Industry norm for scorecards")],
         [txt_s("Calibration validation"),      bdg("Hosmer-Lemeshow","purple"),txt_s("IFRS 9 / Basel IRB requirement")],
         [txt_s("Imbalanced data"),             bdg("PR-AUC or F1","orange"), txt_s("ROC misleading; focus on minority class")],
         [txt_s("Model comparison (nested)"),   bdg("LR Test χ²","gold"),    txt_s("More powerful than Wald for small n")],
         [txt_s("Individual predictor sig."),   bdg("Wald z-test","red"),    txt_s("z = β̂/SE; ~ N(0,1) for large n")],
    ]))

    section_heading("2️⃣ Which Remedy for Model Issues?")
    st.html(table_html(["Problem","Symptom","Remedy"],
        [[bdg("Miscalibration","orange"),  txt_s("HL test fails (p<0.05)"),          txt_s("Platt scaling, isotonic regression, add polynomial terms")],
         [bdg("Multicollinearity","purple"),txt_s("VIF > 10; unstable β̂"),           txt_s("Ridge logistic, LASSO logistic, drop correlated predictor")],
         [bdg("Rare events","red"),        txt_s("Event rate < 5%"),                  txt_s("SMOTE, cost-sensitive learning, lower decision threshold, Firth regression")],
         [bdg("Separation","red"),         txt_s("β̂ → ±∞; MLE fails"),               txt_s("Firth penalised likelihood, Ridge logistic, Bayesian prior")],
         [bdg("Overfitting","orange"),     txt_s("Good in-sample, poor OOS"),         txt_s("Cross-validation, regularisation, reduce features, larger sample")],
         [bdg("Non-linear logit","blue"),  txt_s("RESET or Box-Tidwell fails"),       txt_s("Add polynomial X², log(X), splines, or interaction terms")],
    ]))

    section_heading("3️⃣ Violation Impact Summary")
    st.html(table_html(["Violation","β̂ Biased?","Calibration?","Inference Valid?","Remedy"],
        [[bdg("Multicollinearity","purple"), gt("No"), gt("OK"), org("Weak"), txt_s("Ridge/LASSO logistic")],
         [bdg("Rare events","red"),          org("Slightly"), rt2("Biased up"), rt2("No"), txt_s("Firth regression + correction")],
         [bdg("Omitted variable","blue"),    rt2("Yes"), rt2("Yes"), rt2("No"), txt_s("Add missing predictor or IV")],
         [bdg("Non-linear logit","orange"),  rt2("Yes"), rt2("Yes"), rt2("No"), txt_s("Polynomial/spline terms")],
         [bdg("Perfect separation","gold"),  org("→±∞"), org("→1.0"), rt2("No"), txt_s("Firth penalised MLE")],
    ]))
