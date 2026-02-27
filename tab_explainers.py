"""
tab_explainers.py
Layman-friendly "How This Tab Works" explainer boxes.
Call render_explainer(tab_name) at the TOP of each tab function,
right after the main render_card() header.
"""
from components import render_ib, bdg, hl, p, FH, FB, FM, NO_SEL

# ── colour shortcuts ──────────────────────────────────────────────
def _gold(t):  return f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">{t}</span>'
def _blue(t):  return f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">{t}</span>'
def _green(t): return f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">{t}</span>'
def _red(t):   return f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">{t}</span>'
def _mute(t):  return f'<span style="color:#8892b0;-webkit-text-fill-color:#8892b0">{t}</span>'

def _box(steps, tip=None, variant="blue"):
    """
    Render a step-numbered explainer box.
    steps = list of (emoji, bold_title, plain_description) tuples
    tip   = optional "Plain English takeaway" string
    """
    rows = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:10px;'
        f'margin-bottom:10px;{NO_SEL}">'
        f'<span style="font-size:1.05rem;min-width:22px">{icon}</span>'
        f'<div style="font-family:{FB};font-size:.88rem;color:#e6f1ff;'
        f'-webkit-text-fill-color:#e6f1ff;line-height:1.6">'
        f'<span style="font-weight:700">{title}: </span>{desc}'
        f'</div></div>'
        for icon, title, desc in steps
    )
    tip_html = ""
    if tip:
        tip_html = (
            f'<div style="margin-top:12px;padding:9px 13px;'
            f'background:rgba(255,215,0,0.08);border-left:3px solid #FFD700;'
            f'border-radius:5px;font-family:{FB};font-size:.86rem;'
            f'color:#FFD700;-webkit-text-fill-color:#FFD700;line-height:1.6;{NO_SEL}">'
            f'💡 <span style="font-weight:700">Plain English: </span>{tip}</div>'
        )
    header = (
        f'<div style="font-family:{FH};font-size:.92rem;font-weight:700;'
        f'color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
        f'margin-bottom:11px;letter-spacing:.3px;{NO_SEL}">'
        f'🗺 How This Tab Works</div>'
    )
    return render_ib(header + rows + tip_html, variant)


# ════════════════════════════════════════════════════════
# TAB-SPECIFIC EXPLAINERS
# ════════════════════════════════════════════════════════

def explainer_concepts():
    """Tab 1 — Concepts & Theory"""
    _box([
        ("📌", "The Core Problem",
         "Ordinary linear regression (OLS) cannot model outcomes that are just "
         "yes/no, 0/1, default/no-default — it can predict negative probabilities or "
         "values above 100%, which are meaningless."),
        ("🔀", "The Logistic Solution",
         "Logistic regression passes any number through the "
         f"{_gold('sigmoid (S-curve) function')}, which squeezes every output into "
         "a valid probability between 0 and 1. You get P(event) — never below 0%, never above 100%."),
        ("📐", "The Log-Odds Bridge",
         f"Internally the model estimates {_gold('log-odds = β₀ + β₁X₁ + ...')}. "
         "Think of log-odds as a score — positive means more likely to happen, negative means less likely. "
         "The sigmoid converts that score to a probability."),
        ("💹", "Finance Connection",
         "Every slider and chart in this tab uses a real finance example — "
         "predicting corporate {_gold('default (PD)')} from accounting ratios. "
         "You will see exactly how leverage (D/E) and interest coverage (ICR) shift the default probability."),
    ],
    tip="Logistic regression answers: 'Given a company's financial ratios, what is the probability it defaults?' "
        "The S-curve ensures the answer is always a sensible number between 0% and 100%.")


def explainer_model():
    """Tab 2 — Model Builder"""
    _box([
        ("🎛", "Step 1 — Set Your Portfolio",
         f"Use the sliders to choose {_gold('sample size')} (how many companies) and "
         f"adjust {_gold('default rate')} (what fraction actually default). "
         "The app then simulates realistic financial ratios — D/E, ICR, Current Ratio, ROA — for each firm."),
        ("⚙", "Step 2 — MLE Estimation",
         f"Clicking Run fits the model using {_gold('Maximum Likelihood Estimation (MLE)')}. "
         "Think of MLE as asking: 'What coefficient values make the observed defaults most probable?' "
         "The algorithm iterates (Newton-Raphson) until it finds the best-fitting numbers."),
        ("📋", "Step 3 — Read the Coefficient Table",
         f"Each row shows: {_gold('β (log-odds impact)')}, {_gold('Odds Ratio (OR)')}, "
         f"{_gold('Standard Error')}, {_gold('Wald z-stat')}, and p-value. "
         f"An OR of 1.8 means: one unit increase in that ratio multiplies the odds of default by 1.8×."),
        ("🔮", "Step 4 - Point Prediction",
         "Enter a firm's specific ratios and the model computes P(Default) directly. "
         "This is the same calculation banks run inside credit scoring engines."),
    ],
    tip="Think of MLE like tuning a radio — you adjust dials (coefficients) until the signal (model predictions) "
        "best matches the actual data (who defaulted and who didn't).")


def explainer_diagnostics():
    """Tab 3 — Diagnostics"""
    _box([
        ("🔍", "Why Diagnostics?",
         "Fitting a model is only Step 1. Diagnostics check whether the model is "
         f"{_gold('well-calibrated')} (predicted probabilities match actual rates) and "
         f"whether any observations are {_gold('unduly influential')} (distorting the coefficients)."),
        ("📊", "Residual Plots",
         f"Unlike linear regression, logistic residuals are not symmetric. "
         f"{_gold('Pearson residuals')} = (actual − predicted)/SE. "
         f"{_gold('Deviance residuals')} are based on log-likelihood contribution of each observation. "
         "Large residuals flag companies the model mispredicts badly."),
        ("🎯", "Hosmer-Lemeshow GOF Test",
         f"Groups observations into {_gold('10 deciles')} by predicted probability, then compares "
         f"average predicted vs actual default rates within each group. "
         f"A p-value {_red('< 0.05')} means the model is poorly calibrated — predictions don't match reality."),
        ("🏔", "ROC Curve & AUC",
         f"The {_gold('ROC curve')} plots True Positive Rate vs False Positive Rate across all possible "
         f"cut-off thresholds. {_gold('AUC = 0.50')} = useless coin flip. "
         f"{_gold('AUC = 1.00')} = perfect model. AUC > 0.75 is generally good for credit scoring."),
        ("⚠", "Cook's Distance",
         f"Identifies {_gold('influential observations')} — companies that, if removed, would significantly "
         "change the coefficients. In finance, these often correspond to outlier firms during crisis periods."),
    ],
    tip="Diagnostics are like a health check for your model. A high AUC means it ranks risky firms above safe ones. "
        "Hosmer-Lemeshow checks whether the actual default rates in each risk bucket match what the model predicted.")


def explainer_finance():
    """Tab 4 — Finance Cases"""
    _box([
        ("🏦", "Three Real Finance Scenarios",
         f"This tab runs {_gold('three independent case studies')} using the same logistic framework, "
         "each representing a common application in banking and investment: "
         "corporate default, retail credit scoring, and sovereign risk."),
        ("💳", "Case 1 — Corporate Default (Altman-style)",
         f"Based on the classic {_gold('Altman Z-Score')} framework. Predictors are accounting ratios: "
         "D/E (leverage), ICR (interest coverage), Current Ratio (liquidity), ROA (profitability). "
         f"{_gold('Expected signs:')} D/E raises PD; ICR, CR, ROA reduce PD."),
        ("👤", "Case 2 — Retail Credit Scoring",
         f"Predicts {_gold('P(personal loan default)')} from CIBIL score, loan-to-income ratio, "
         "employment stability, and account age. The output is used to set credit limits and interest rates."),
        ("🌍", "Case 3 — Sovereign Risk",
         f"Predicts {_gold('P(sovereign credit downgrade)')} from macro variables: "
         "debt-to-GDP, current account deficit, FX reserves, GDP growth rate. "
         "Used by rating agencies and fixed income portfolio managers."),
        ("📈", "What the Output Shows",
         f"For each case: {_gold('coefficient table')}, {_gold('odds ratios')}, "
         f"{_gold('ROC + AUC')}, and a {_gold('point prediction')} for a representative entity. "
         "Compare AUC across cases to see which model is most discriminating."),
    ],
    tip="Each case study mirrors what a bank's credit risk team actually builds. "
        "The only difference between them is which financial ratios go into the model. "
        "The logistic regression machinery — MLE, odds ratios, AUC — is identical.")


def explainer_code():
    """Tab 5 — Python Code"""
    _box([
        ("📦", "What's Shown",
         f"Ready-to-run Python snippets covering the {_gold('full logistic regression workflow')}: "
         "data simulation, model fitting (statsmodels + sklearn), coefficient extraction, "
         "odds ratio calculation, ROC/AUC, calibration, and VIF check."),
        ("🔢", "Snippet 1 — Core MLE Fit",
         f"Uses {_gold('statsmodels.Logit')} — gives full statistical output (p-values, SE, Wald tests, "
         "log-likelihood). This is the academic/research approach, equivalent to R's glm()."),
        ("🤖", "Snippet 2 — sklearn Pipeline",
         f"Uses {_gold('sklearn.LogisticRegression')} — designed for machine learning workflows. "
         "Easier to plug into cross-validation, grid search, and ensemble models. "
         "Less detailed statistical output but faster for production."),
        ("🎯", "Snippet 3 — ROC & AUC",
         f"Shows how to compute and plot {_gold('sklearn.metrics.roc_curve')} and "
         f"{_gold('roc_auc_score')}. Copy-paste ready for any binary classification problem."),
        ("📐", "Snippet 4 — Calibration & VIF",
         f"Calibration plot checks if P(Default)=0.3 actually means 30% of firms default. "
         f"VIF detects multicollinearity among predictors before fitting the model."),
    ],
    tip="Copy any snippet directly into a Jupyter notebook or .py file — all imports are included. "
        "Start with Snippet 1 to understand the statistics, then use Snippet 2 for production code.")


def explainer_vocab():
    """Tab 6 — Education Hub"""
    _box([
        ("🃏", "Concept Cards",
         f"Visual badge-style cards grouping related terms together. "
         f"Choose a theme ({_gold('Core Model')}, {_gold('Model Fit')}, {_gold('Diagnostics')}, "
         f"{_gold('Finance Apps')}) from the dropdown. "
         "Each card shows the key idea, the formula badge, and the finance interpretation side-by-side."),
        ("📖", "Glossary",
         f"Searchable definitions for every important term — from {_gold('Odds Ratio')} to "
         f"{_gold('McFadden R²')} to {_gold('Hosmer-Lemeshow')}. "
         "Each entry includes: plain-English definition, formula, worked example, and finance context."),
        ("📐", "Formula Sheet",
         f"All key formulas in one place: sigmoid function, log-likelihood, Wald test, "
         "AUC, calibration. Use this as a quick reference during exams or when reviewing model output."),
        ("🗺", "Decision Guide",
         f"Answers practical questions: {_gold('Which GOF test should I use?')} "
         f"{_gold('What does an OR of 2.3 mean?')} {_gold('When is AUC insufficient?')} "
         "Structured as decision trees and comparison tables."),
        ("🎓", "MCQ Quiz",
         f"Test your understanding with {_gold('20 questions')} from Foundation to Advanced level, "
         "covering concepts, calculations, and finance interpretation. "
         "Immediate feedback with full explanations after each answer."),
    ],
    tip="Start with Concept Cards for a visual overview, use the Glossary when you encounter an unfamiliar term, "
        "and finish with the MCQ Quiz to confirm you can apply the concepts under exam conditions.")
