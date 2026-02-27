"""
tab_concepts.py — Logistic Regression Concepts & Theory
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tab_explainers import explainer_concepts
from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col, four_col,
    table_html, metric_row, section_heading, stat_box, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

def tab_concepts():
    render_card("🎯 Why Logistic Regression? The Problem with OLS for Binary Outcomes",
        p(f'When the dependent variable Y is {hl("binary (0 or 1)")}, OLS regression fails. '
          f'Logistic regression solves this with a bounded, probabilistic model.')
        + three_col(
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">❌ OLS Fails Because:</span><br>'
               + p(f'Predicted values can exceed [0,1]<br>Violates homoscedasticity<br>'
                   f'Residuals non-normal (only 0 or 1)<br>Linear PD model is nonsensical')
               + p(f'{rt2("Example:")} OLS predicts PD = −0.12 or PD = 1.34 — impossible!'), "red"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">✅ Logistic Solves:</span><br>'
               + p(f'Output always bounded in (0,1) — valid probability<br>'
                   f'Uses {hl("sigmoid")} to map ℝ → (0,1)<br>'
                   f'Estimated via {hl("Maximum Likelihood")}<br>'
                   f'Natural probability interpretation')
               + p(f'{gt("Finance:")} P(Default), P(Fraud), P(Downgrade), P(Approval)'), "green"),
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📈 Finance Applications:</span><br>'
               + p(f'{bdg("Credit Risk","red")} P(Default) from financials<br>'
                   f'{bdg("Fraud Detection","orange")} P(Fraud transaction)<br>'
                   f'{bdg("Rating Migration","blue")} P(Downgrade)<br>'
                   f'{bdg("M&A","purple")} P(Deal Completion)<br>'
                   f'{bdg("Loan Approval","green")} Binary credit decision'), "gold"),
        )
    )

    render_card("📐 The Logistic (Sigmoid) Function",
        p(f'The core transformation that maps any real number to a valid probability.')
        + two_col(
            fml("Model:   P(Y=1|X) = 1 / (1 + e^(−z))\n"
                "where:   z = β₀ + β₁X₁ + ... + βₖXₖ\n\n"
                "Log-Odds (Logit): log[P/(1−P)] = β₀ + β₁X₁ + ... + βₖXₖ\n\n"
                "Odds Ratio:  OR = e^β_j\n"
                "  OR > 1 → Xⱼ increases odds of Y=1\n"
                "  OR < 1 → Xⱼ decreases odds of Y=1"),
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Key Properties:</span><br>'
               + p(f'σ(0) = 0.5 — natural decision boundary<br>'
                   f'σ(z) → 1 as z → +∞; σ(z) → 0 as z → −∞<br>'
                   f'S-shaped, monotonically increasing<br>'
                   f"σ′(z) = σ(z)(1−σ(z)) — elegant derivative<br>"
                   f'Inverse of sigmoid = logit function<br>'
                   f'Threshold adjustable for imbalanced data'), "gold"),
        )
    )

    render_card("📊 Interactive Sigmoid Explorer",
        ib(f'Adjust β₀ and β₁ to see how the sigmoid maps X → P(Y=1). '
           f'{hl("Green region")} = predicted positive class. '
           f'{rt2("Red region")} = predicted negative class.', "blue")
    )
    explainer_concepts()
    col1, col2, col3 = st.columns(3)
    b0 = col1.slider("β₀ (Intercept)", -4.0, 4.0, 0.0, 0.25, key="sig_b0")
    b1 = col2.slider("β₁ (Slope)", -3.0, 3.0, 1.0, 0.25, key="sig_b1")
    threshold = col3.slider("Decision Threshold", 0.10, 0.90, 0.50, 0.05, key="sig_thresh")

    x = np.linspace(-6, 6, 400)
    z = b0 + b1 * x
    prob = 1 / (1 + np.exp(-z))
    log_odds = z  # logit is linear

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0a1628")

    def _sax(ax):
        ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.grid(color="#1e3a5f", alpha=0.35, lw=0.5)

    axes[0].plot(x, prob, color="#FFD700", lw=2.5)
    axes[0].axhline(threshold, color="#dc3545", lw=1.5, ls="--", label=f"Threshold={threshold:.2f}")
    axes[0].axhline(0.5, color="#64ffda", lw=1, ls=":", alpha=0.7)
    axes[0].fill_between(x, prob, threshold, where=(prob > threshold), alpha=0.2, color="#28a745")
    axes[0].fill_between(x, prob, threshold, where=(prob <= threshold), alpha=0.2, color="#dc3545")
    axes[0].set(xlabel="X", ylabel="P(Y=1|X)"); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title(f"Sigmoid σ(β₀={b0}, β₁={b1})", color="#FFD700", fontsize=10)
    axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
    axes[0].xaxis.label.set_color("#8892b0"); axes[0].yaxis.label.set_color("#8892b0")
    _sax(axes[0])

    axes[1].plot(x, log_odds, color="#ADD8E6", lw=2.5)
    axes[1].axhline(0, color="#FFD700", lw=1, ls="--")
    axes[1].set(xlabel="X", ylabel="Log-Odds = β₀ + β₁X")
    axes[1].set_title("Log-Odds (Linear in X)", color="#FFD700", fontsize=10)
    axes[1].xaxis.label.set_color("#8892b0"); axes[1].yaxis.label.set_color("#8892b0")
    _sax(axes[1])

    for b1v, col in [(-2,"#dc3545"),(-1,"#ff9f43"),(0,"#8892b0"),(1,"#ADD8E6"),(2,"#FFD700")]:
        axes[2].plot(x, 1/(1+np.exp(-(b0+b1v*x))), color=col, lw=1.8, label=f"β₁={b1v}", alpha=0.85)
    axes[2].axhline(0.5, color="#64ffda", lw=1, ls=":", alpha=0.7)
    axes[2].set(xlabel="X", ylabel="P(Y=1|X)")
    axes[2].set_title("Effect of β₁ on Steepness", color="#FFD700", fontsize=10)
    axes[2].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f", ncol=2)
    axes[2].xaxis.label.set_color("#8892b0"); axes[2].yaxis.label.set_color("#8892b0")
    _sax(axes[2])

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if b1 != 0:
        boundary = -b0 / b1
        p_at_0 = 1 / (1 + np.exp(-b0))
        st.html(four_col(
            stat_box("Decision Boundary X*", f"{boundary:.3f}", f"P={0.5:.1f} here always", "gold"),
            stat_box("P(Y=1) at X=0", f"{p_at_0:.4f}", "Intercept-driven", "blue"),
            stat_box("Odds Ratio e^β₁", f"{np.exp(b1):.4f}", "Per unit ΔX change", "orange"),
            stat_box("Max Slope ≈ β₁/4", f"{b1/4:.4f}", "At decision boundary", "purple"),
        ))

    render_card("⚖ OLS vs Logistic Regression — Complete Comparison",
        table_html(
            ["Feature", "OLS Regression", "Logistic Regression"],
            [
                [bdg("Dependent Y","blue"),    txt_s("Continuous ∈ (−∞,+∞)"),        txt_s("Binary ∈ {0,1}")],
                [bdg("Output","gold"),          txt_s("Predicted value Ŷ"),            txt_s("Probability P(Y=1|X) ∈ (0,1)")],
                [bdg("Link function","purple"), txt_s("Identity: E(Y) = Xβ"),         txt_s("Logit: log[P/(1−P)] = Xβ")],
                [bdg("Estimation","orange"),    txt_s("OLS — Minimise Σ(Y−Ŷ)²"),     txt_s("MLE — Maximise Σlog L(β)")],
                [bdg("Error dist.","red"),      txt_s("Normal (CLRM assumption)"),    txt_s("Bernoulli — not normal")],
                [bdg("Goodness of fit","green"),txt_s("R², Adj R², F-test"),          txt_s("McFadden R², AUC-ROC, Log-Loss")],
                [bdg("Coefficients","blue"),    txt_s("Direct ΔY per unit ΔX"),       txt_s("Log-odds change; exp(β) = OR")],
                [bdg("Inference","purple"),     txt_s("t-test, F-test"),              txt_s("Wald test, LR test, Score test")],
                [bdg("Finance use","orange"),   txt_s("Return forecast, factor models"),txt_s("PD, fraud, rating migration")],
            ]
        )
    )

    render_card("🔧 Maximum Likelihood Estimation (MLE)",
        p(f'Coefficients found by {hl("maximising the log-likelihood")} — making observed data most probable.')
        + steps_html([
            ("Log-Likelihood", "ℓ(β) = Σᵢ [yᵢ log(pᵢ) + (1−yᵢ) log(1−pᵢ)]  where pᵢ = σ(Xᵢβ)"),
            ("Optimisation", "Newton-Raphson / Fisher Scoring: β_new = β_old − H⁻¹∇ℓ (Hessian update)"),
            ("Standard Errors", "SE(β̂) = √[diag(−H⁻¹)] from the Fisher Information Matrix"),
            ("McFadden R²", "ρ² = 1 − ℓ(β)/ℓ₀  where ℓ₀ is intercept-only log-likelihood"),
        ])
        + fml("Log-Likelihood:   ℓ(β) = Σᵢ yᵢ log(pᵢ) + (1−yᵢ)log(1−pᵢ)\n"
              "Null LL:          ℓ₀ = n[p̄ log(p̄) + (1−p̄)log(1−p̄)]\n"
              "McFadden R²:      ρ² = 1 − ℓ(β)/ℓ₀    [0.2–0.4 = good fit in finance]")
    )

    render_card("📋 Logistic Regression Assumptions",
        two_col(
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">✅ Required:</span><br>'
               + table_html(["Assumption","Meaning"],[
                   [bdg("Binary Y","red"),           txt_s("Outcome must be 0 or 1")],
                   [bdg("Linearity in logit","gold"), txt_s("log[P/(1-P)] linear in X")],
                   [bdg("Independence","blue"),       txt_s("Observations independent")],
                   [bdg("No multicollinearity","purple"),txt_s("VIF < 10 among predictors")],
                   [bdg("Events per var ≥10","orange"),txt_s("EPV rule: avoids overfitting")],
               ]), "green"),
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">❌ NOT Required (unlike OLS):</span><br>'
               + p(f'{rt2("Normal residuals")} — Bernoulli errors<br><br>'
                   f'{rt2("Homoscedasticity")} — Var varies with p<br><br>'
                   f'{rt2("Linear Y-X relationship")} — only logit linear<br><br>'
                   f'{rt2("Equal group sizes")} — handles imbalance'), "red"),
        )
    )
