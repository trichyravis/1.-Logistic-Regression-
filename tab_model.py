"""
tab_model.py — Interactive Logistic Regression Model Builder
Credit Default prediction with full MLE output, odds ratios, Wald tests
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col, four_col,
    table_html, metric_row, section_heading, stat_box, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'


def _sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def _logistic_mle(X, y, max_iter=200, tol=1e-8):
    """Newton-Raphson MLE for logistic regression."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        p   = _sigmoid(X @ beta)
        W   = p * (1 - p) + 1e-12
        grad = X.T @ (y - p)
        H    = -(X.T * W) @ X
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta -= delta
        if np.max(np.abs(delta)) < tol:
            break
    p_final = _sigmoid(X @ beta)
    W_final = p_final * (1 - p_final) + 1e-12
    H_final = -(X.T * W_final) @ X
    try:
        cov = np.linalg.inv(-H_final)
        se  = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    ll = np.sum(y * np.log(p_final + 1e-12) + (1-y) * np.log(1 - p_final + 1e-12))
    return beta, se, ll, p_final


def tab_model():
    render_card("🔬 Interactive Credit Default Logistic Regression",
        p(f'Simulate a {hl("corporate credit portfolio")} and fit a full logistic regression model. '
          f'Adjust true default rates and financial ratios to see how MLE, odds ratios, and Wald tests respond.')
        + three_col(
            ib(f'{bdg("Model","gold")} P(Default) = σ(β₀ + β₁·D/E + β₂·ICR + β₃·CurrentRatio + β₄·ROA)', "gold"),
            ib(f'{bdg("Estimation","blue")} Newton-Raphson MLE — full Hessian-based standard errors', "blue"),
            ib(f'{bdg("Output","green")} Coefficients, Odds Ratios, Wald tests, McFadden R², AUC', "green"),
        )
    )

    # Controls
    col1, col2, col3, col4 = st.columns(4)
    n_firms   = col1.number_input("Number of firms", 50, 500, 200, 25, key="lr_n")
    base_rate = col2.slider("Base default rate (%)", 5, 40, 15, 1, key="lr_rate") / 100
    seed      = col3.number_input("Random seed", 1, 999, 42, key="lr_seed")
    threshold = col4.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05, key="lr_thresh")

    if st.button("🚀 Fit Logistic Regression", key="lr_fit", use_container_width=False):
        np.random.seed(int(seed))
        n = int(n_firms)

        # Simulate financial ratios
        de    = np.random.lognormal(0, 0.5, n)           # D/E ratio
        icr   = np.random.lognormal(1, 0.6, n)           # Interest Coverage
        cr    = np.random.uniform(0.5, 3.5, n)           # Current Ratio
        roa   = np.random.normal(0.06, 0.08, n)          # ROA

        # True data-generating logit
        z_true = (-1.5 + 0.6*de - 0.4*icr - 0.5*cr - 3.0*roa
                  + np.log(base_rate/(1-base_rate)))
        p_true = _sigmoid(z_true)
        y = (np.random.uniform(size=n) < p_true).astype(float)

        # Design matrix
        X = np.column_stack([np.ones(n), de, icr, cr, roa])
        names = ["Intercept", "D/E Ratio", "ICR", "Current Ratio", "ROA"]

        # Fit model
        beta, se, ll, p_hat = _logistic_mle(X, y)

        # Null log-likelihood
        p_bar = y.mean()
        ll0 = n * (p_bar * np.log(p_bar + 1e-12) + (1-p_bar)*np.log(1-p_bar + 1e-12))

        # Metrics
        mcf_r2 = 1 - ll / ll0
        lr_stat = -2 * (ll0 - ll)
        lr_p    = 1 - stats.chi2.cdf(lr_stat, df=len(beta)-1)
        y_pred  = (p_hat >= threshold).astype(int)
        acc     = np.mean(y_pred == y)

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y, p_hat)
        except Exception:
            auc = float('nan')

        # Display metrics
        st.html(four_col(
            stat_box("McFadden R²",   f"{mcf_r2:.4f}", "0.2–0.4 = good", "gold"),
            stat_box("AUC-ROC",       f"{auc:.4f}",    ">0.75 = good",    "green"),
            stat_box("LR Statistic",  f"{lr_stat:.2f}", f"p={lr_p:.4f}",  "blue"),
            stat_box("Accuracy",      f"{acc:.3f}",     f"Threshold={threshold:.2f}", "orange"),
        ))

        # Coefficient table
        wald_stats = beta / (se + 1e-12)
        wald_p     = 2 * (1 - stats.norm.cdf(np.abs(wald_stats)))
        ci_lo      = beta - 1.96 * se
        ci_hi      = beta + 1.96 * se
        or_vals    = np.exp(beta)

        def _sig(p):
            if p < 0.001: return gt("***")
            if p < 0.01:  return gt("**")
            if p < 0.05:  return gt("*")
            if p < 0.10:  return org(".")
            return rt2("ns")

        def _sign_badge(b, name):
            expected = {"D/E Ratio":"+","ICR":"-","Current Ratio":"-","ROA":"-"}
            exp = expected.get(name, "?")
            actual = "+" if b > 0 else "-"
            if exp == "?": return bdg("—","blue")
            return gt(f"✓ {actual}") if actual == exp else rt2(f"✗ {actual}")

        rows = []
        for i, name in enumerate(names):
            rows.append([
                hl(name),
                _f(f"{beta[i]:+.4f}"),
                _f(f"{se[i]:.4f}"),
                _f(f"{or_vals[i]:.4f}"),
                _f(f"{wald_stats[i]:+.3f}"),
                txt_s(f"{wald_p[i]:.4f}"),
                _sig(wald_p[i]),
                txt_s(f"[{ci_lo[i]:+.3f}, {ci_hi[i]:+.3f}]"),
                _sign_badge(beta[i], name) if i > 0 else txt_s("—"),
            ])

        section_heading("📋 MLE Coefficient Table")
        st.html(table_html(
            ["Variable", "β̂", "SE(β̂)", "OR = e^β", "Wald z", "p-value", "Sig", "95% CI", "Sign"],
            rows
        ))

        st.html(ib(
            f'{bdg("Significance codes","gold")} '
            + txt_s("*** p<0.001 &nbsp;&nbsp; ** p<0.01 &nbsp;&nbsp; * p<0.05 &nbsp;&nbsp; . p<0.10 &nbsp;&nbsp; ns not significant"),
            "gold"
        ))

        # Plots
        fig = _model_plots(y, p_hat, beta, se, names, threshold, n)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Interpretation
        section_heading("📖 Model Interpretation")
        significant = [(names[i], beta[i], or_vals[i], wald_p[i]) for i in range(1, len(names)) if wald_p[i] < 0.05]
        if significant:
            interp_rows = []
            for nm, b, orv, pv in significant:
                direction = "increases" if b > 0 else "decreases"
                pct_change = abs(orv - 1) * 100
                interp_rows.append([
                    hl(nm),
                    gt("↑ Risk") if b > 0 else rt2("↓ Risk"),
                    txt_s(f"Each unit increase {direction} odds of default by {pct_change:.1f}%"),
                    _f(f"OR={orv:.3f}"),
                    txt_s(f"p={pv:.4f}"),
                ])
            st.html(table_html(["Variable","Direction","Interpretation","Odds Ratio","p-value"], interp_rows))

        st.html(ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Model Assessment: </span>'
            + txt_s(f'McFadden R²={mcf_r2:.3f} '
                    + ("(Excellent fit — above 0.4)" if mcf_r2 > 0.4
                       else "(Good fit — above 0.2)" if mcf_r2 > 0.2
                       else "(Weak fit — below 0.2)")
                    + f'. AUC={auc:.3f} '
                    + ("(Strong discriminative power)" if auc > 0.75 else "(Moderate discriminative power)")),
            "gold"
        ))

    else:
        render_ib(ib(
            f'👆 Set parameters above and click {hl("Fit Logistic Regression")} to run the model.',
            "blue"
        ), "blue")

    # Reference: Goodness of Fit Measures
    render_card("📊 Logistic Regression Goodness-of-Fit Measures",
        table_html(
            ["Measure", "Formula", "Range", "Threshold", "Notes"],
            [
                [bdg("McFadden R²","gold"),  _f("1 − ℓ(β)/ℓ₀"),    txt_s("0 to 1"),  txt_s("0.2–0.4 = good"),   txt_s("Analogous to R² but not directly comparable")],
                [bdg("AUC-ROC","green"),      _f("∫ROC curve"),       txt_s("0.5 to 1"),txt_s(">0.75 good"),        txt_s("0.5 = random; 1.0 = perfect; robust to imbalance")],
                [bdg("Log-Loss","red"),       _f("−ℓ(β)/n"),          txt_s("0 to ∞"), txt_s("Lower = better"),    txt_s("Penalises confident wrong predictions heavily")],
                [bdg("LR χ² test","blue"),   _f("−2(ℓ₀−ℓ) ~ χ²(k)"),txt_s("0 to ∞"), txt_s("p < 0.05 = sig"),   txt_s("Tests if any predictor improves on null model")],
                [bdg("Nagelkerke R²","purple"),_f("1−(L₀/L)^(2/n)"), txt_s("0 to 1"),  txt_s("Scales to 1"),      txt_s("Adjusted pseudo-R² that reaches 1 for perfect fit")],
                [bdg("Hosmer-Lemeshow","orange"),_f("χ²(8) test"),    txt_s("p > 0.05"),txt_s("p > 0.05 = fit OK"),txt_s("Tests calibration: predicted vs observed rates")],
            ]
        )
    )


def _model_plots(y, p_hat, beta, se, names, threshold, n):
    fig = plt.figure(figsize=(16, 10), facecolor="#0a1628")
    gs  = GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.38)

    def _sax(ax):
        ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.grid(color="#1e3a5f", alpha=0.35, lw=0.5)

    # 1 — Predicted prob distribution
    ax1 = fig.add_subplot(gs[0, 0]); _sax(ax1)
    ax1.hist(p_hat[y==0], bins=30, alpha=0.7, color="#ADD8E6", label="Non-default (y=0)", density=True)
    ax1.hist(p_hat[y==1], bins=30, alpha=0.7, color="#dc3545", label="Default (y=1)", density=True)
    ax1.axvline(threshold, color="#FFD700", lw=2, ls="--", label=f"Threshold={threshold:.2f}")
    ax1.set_xlabel("Predicted P(Default)", color="#8892b0", fontsize=8)
    ax1.set_title("Score Distribution", color="#FFD700", fontsize=10)
    ax1.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=7, edgecolor="#1e3a5f")

    # 2 — ROC Curve
    ax2 = fig.add_subplot(gs[0, 1]); _sax(ax2)
    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for t in thresholds:
        yp = (p_hat >= t).astype(int)
        tp = np.sum((yp==1)&(y==1)); fp = np.sum((yp==1)&(y==0))
        tn = np.sum((yp==0)&(y==0)); fn = np.sum((yp==0)&(y==1))
        tprs.append(tp/(tp+fn+1e-9)); fprs.append(fp/(fp+tn+1e-9))
    ax2.plot(fprs, tprs, color="#FFD700", lw=2.5, label="ROC")
    ax2.plot([0,1],[0,1], color="#8892b0", lw=1, ls="--", label="Random")
    ax2.fill_between(fprs, tprs, alpha=0.12, color="#FFD700")
    ax2.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax2.set_title("ROC Curve", color="#FFD700", fontsize=10)
    ax2.xaxis.label.set_color("#8892b0"); ax2.yaxis.label.set_color("#8892b0")
    ax2.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")

    # 3 — Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2]); _sax(ax3)
    yp = (p_hat >= threshold).astype(int)
    tp = np.sum((yp==1)&(y==1)); fp = np.sum((yp==1)&(y==0))
    tn = np.sum((yp==0)&(y==0)); fn = np.sum((yp==0)&(y==1))
    cm = np.array([[tn, fp],[fn, tp]])
    im = ax3.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, str(cm[i,j]), ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    ax3.set_xticks([0,1]); ax3.set_yticks([0,1])
    ax3.set_xticklabels(["Pred 0","Pred 1"], color="#8892b0")
    ax3.set_yticklabels(["Actual 0","Actual 1"], color="#8892b0")
    ax3.set_title("Confusion Matrix", color="#FFD700", fontsize=10)
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    ax3.set_xlabel(f"Precision={prec:.3f}  Recall={rec:.3f}", color="#8892b0", fontsize=8)

    # 4 — Odds Ratio Plot
    ax4 = fig.add_subplot(gs[0, 3]); _sax(ax4)
    or_vals = np.exp(beta[1:])
    ci_lo   = np.exp(beta[1:] - 1.96*se[1:])
    ci_hi   = np.exp(beta[1:] + 1.96*se[1:])
    y_pos   = np.arange(len(names)-1)
    colors  = ["#dc3545" if o > 1 else "#28a745" for o in or_vals]
    ax4.barh(y_pos, or_vals-1, left=1, color=colors, alpha=0.7, height=0.5)
    for i,(lo,hi) in enumerate(zip(ci_lo, ci_hi)):
        ax4.plot([lo,hi],[i,i], color="white", lw=1.5)
        ax4.scatter([or_vals[i]],[i], color="white", s=40, zorder=5)
    ax4.axvline(1, color="#FFD700", lw=2, ls="--")
    ax4.set_yticks(y_pos); ax4.set_yticklabels(names[1:], color="#e6f1ff", fontsize=8)
    ax4.set_xlabel("Odds Ratio (OR)", color="#8892b0", fontsize=8)
    ax4.set_title("Odds Ratios + 95% CI", color="#FFD700", fontsize=10)

    # 5 — Calibration curve
    ax5 = fig.add_subplot(gs[1, 0:2]); _sax(ax5)
    bins = np.percentile(p_hat, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    obs_rates, pred_means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_hat >= lo) & (p_hat < hi)
        if mask.sum() > 0:
            obs_rates.append(y[mask].mean())
            pred_means.append(p_hat[mask].mean())
    ax5.plot([0,1],[0,1], color="#8892b0", lw=1.5, ls="--", label="Perfect calibration")
    ax5.scatter(pred_means, obs_rates, color="#FFD700", s=80, zorder=5)
    ax5.plot(pred_means, obs_rates, color="#ADD8E6", lw=1.5, label="Model calibration")
    ax5.fill_between(pred_means, obs_rates, pred_means, alpha=0.15, color="#dc3545")
    ax5.set(xlabel="Mean Predicted Probability", ylabel="Observed Default Rate")
    ax5.xaxis.label.set_color("#8892b0"); ax5.yaxis.label.set_color("#8892b0")
    ax5.set_title("Calibration Curve (Predicted vs Actual Default Rates)", color="#FFD700", fontsize=10)
    ax5.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")

    # 6 — Precision-Recall curve
    ax6 = fig.add_subplot(gs[1, 2:4]); _sax(ax6)
    precs, recs, f1s = [], [], []
    ths = np.linspace(0.01, 0.99, 100)
    for t in ths:
        yp2 = (p_hat >= t).astype(int)
        tp2 = np.sum((yp2==1)&(y==1)); fp2 = np.sum((yp2==1)&(y==0)); fn2 = np.sum((yp2==0)&(y==1))
        prec2 = tp2/(tp2+fp2+1e-9); rec2 = tp2/(tp2+fn2+1e-9)
        precs.append(prec2); recs.append(rec2)
        f1s.append(2*prec2*rec2/(prec2+rec2+1e-9))
    ax6.plot(recs, precs, color="#FFD700", lw=2.5, label="Precision-Recall")
    best_f1_idx = np.argmax(f1s)
    ax6.scatter([recs[best_f1_idx]],[precs[best_f1_idx]], color="#28a745", s=100, zorder=5,
                label=f"Best F1={f1s[best_f1_idx]:.3f} @ t={ths[best_f1_idx]:.2f}")
    ax6.axhline(y.mean(), color="#dc3545", lw=1.5, ls="--", label=f"Baseline={y.mean():.3f}")
    ax6.set(xlabel="Recall", ylabel="Precision")
    ax6.xaxis.label.set_color("#8892b0"); ax6.yaxis.label.set_color("#8892b0")
    ax6.set_title("Precision-Recall Curve", color="#FFD700", fontsize=10)
    ax6.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")

    return fig
