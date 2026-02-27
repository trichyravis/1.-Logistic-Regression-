"""
tab_diagnostics.py — Logistic Regression Diagnostics
Hosmer-Lemeshow, residuals, influence, multicollinearity, GOF tests
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
    lb_t, txt_s, p, steps_html, two_col, three_col,
    table_html, metric_row, section_heading, stat_box, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

def _fit_logistic(X, y):
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(200):
        p   = _sigmoid(X @ beta)
        W   = p*(1-p)+1e-12
        grad = X.T@(y-p)
        H    = -(X.T*W)@X
        try:
            delta = np.linalg.solve(H, grad)
        except Exception:
            break
        beta -= delta
        if np.max(np.abs(delta)) < 1e-8: break
    p_f = _sigmoid(X @ beta)
    W_f = p_f*(1-p_f)+1e-12
    H_f = -(X.T*W_f)@X
    try:
        cov = np.linalg.inv(-H_f)
        se  = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(k, np.nan)
    ll = np.sum(y*np.log(p_f+1e-12)+(1-y)*np.log(1-p_f+1e-12))
    return beta, se, ll, p_f


def tab_diagnostics():
    render_card("🔬 Logistic Regression Diagnostics Overview",
        p(f'After fitting a logistic model, validate it using these diagnostic tests. '
          f'Unlike OLS, residuals are {rt2("NOT")} normally distributed — use specialised logistic diagnostics.')
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">📐 Residual Types</span><br>'
               + p(f'{bdg("Pearson","blue")} (yᵢ−p̂ᵢ)/√(p̂ᵢ(1−p̂ᵢ))<br>'
                   f'{bdg("Deviance","gold")} ±√(−2yᵢlog(p̂ᵢ)−2(1−yᵢ)log(1−p̂ᵢ))<br>'
                   f'{bdg("Working","purple")} linearised pseudo-residuals<br>'
                   f'{bdg("Anscombe","green")} normalising transform'), "gold"),
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">🔍 Influence Measures</span><br>'
               + p(f'{bdg("Cook\'s Distance","red")} identifies high-influence obs<br>'
                   f'{bdg("Leverage hᵢᵢ","orange")} from hat matrix diagonal<br>'
                   f'{bdg("DFBetas","purple")} change in β̂ if obs removed<br>'
                   f'{bdg("DFITS","blue")} combined influence measure'), "red"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">✅ GOF Tests</span><br>'
               + p(f'{bdg("Hosmer-Lemeshow","green")} calibration χ²(8) test<br>'
                   f'{bdg("LR Test","blue")} −2(ℓ₀−ℓ) ~ χ²(k)<br>'
                   f'{bdg("Wald Test","gold")} β̂²/Var(β̂) ~ χ²(1)<br>'
                   f'{bdg("Score Test","orange")} based on gradient at H₀'), "green"),
        )
    )

    render_card("📋 Diagnostic Tests Reference Table",
        table_html(
            ["Test", "H₀", "Statistic", "Distribution", "Finance Context"],
            [
                [bdg("Hosmer-Lemeshow","green"), txt_s("Model calibrated"),
                 _f("Σ(Oᵢ−Eᵢ)²/Eᵢ"), txt_s("χ²(g−2)"),
                 txt_s("p > 0.05 → predicted PD matches actual default rates")],
                [bdg("LR Test","blue"), txt_s("All βⱼ = 0"),
                 _f("−2(ℓ₀−ℓ)"), txt_s("χ²(k)"),
                 txt_s("Reject → at least one financial ratio significantly predicts default")],
                [bdg("Wald Test","gold"), txt_s("β_j = 0"),
                 _f("(β̂_j/SE)² ~ χ²(1)"), txt_s("χ²(1)"),
                 txt_s("p < 0.05 → individual predictor is significant")],
                [bdg("VIF","purple"), txt_s("No multicollinearity"),
                 _f("1/(1−R²_j)"), txt_s("> 10 = problem"),
                 txt_s("High VIF among D/E, ICR, CurrentRatio is common in credit models")],
                [bdg("Box-Tidwell","orange"), txt_s("Linearity in logit"),
                 _f("Add X·ln(X) terms"), txt_s("Wald tests on interactions"),
                 txt_s("Tests if financial ratios have linear log-odds relationship")],
            ]
        )
    )

    # Interactive diagnostic suite
    render_card("🔬 Interactive Diagnostic Suite",
        ib(f'Choose a scenario to inject specific violations. The suite runs {hl("Hosmer-Lemeshow")}, '
           f'{hl("VIF")}, influence diagnostics, and deviance residual analysis.', "blue")
    )

    col1, col2 = st.columns(2)
    scenario = col1.selectbox("Scenario", [
        "Well-specified credit model",
        "Multicollinearity: correlated financial ratios",
        "Rare events: very low default rate (3%)",
        "Influential outliers: extreme leverage firms",
        "Non-linear log-odds: need polynomial terms",
    ], key="diag_scen")
    n_diag = col2.number_input("Sample size", 100, 500, 250, 50, key="diag_n")
    seed_d = st.number_input("Seed", 1, 999, 7, key="diag_seed")

    if st.button("🔬 Run Diagnostics", key="diag_run"):
        np.random.seed(int(seed_d))
        n = int(n_diag)

        if "Multicollinearity" in scenario:
            de  = np.random.lognormal(0, 0.5, n)
            icr = 5 - 0.9*de + np.random.normal(0, 0.3, n)  # correlated!
            cr  = np.random.uniform(0.5, 3.5, n)
            roa = np.random.normal(0.06, 0.08, n)
        elif "Rare events" in scenario:
            de  = np.random.lognormal(0, 0.5, n)
            icr = np.random.lognormal(1, 0.6, n)
            cr  = np.random.uniform(0.5, 3.5, n)
            roa = np.random.normal(0.06, 0.08, n)
        elif "Influential outliers" in scenario:
            de  = np.concatenate([np.random.lognormal(0, 0.4, n-5), np.array([15,20,18,25,22])])
            icr = np.concatenate([np.random.lognormal(1, 0.5, n-5), np.array([0.1,0.15,0.2,0.1,0.12])])
            cr  = np.random.uniform(0.5, 3.5, n)
            roa = np.random.normal(0.06, 0.08, n)
        elif "Non-linear" in scenario:
            de  = np.random.lognormal(0, 0.5, n)
            icr = np.random.lognormal(1, 0.6, n)
            cr  = np.random.uniform(0.5, 3.5, n)
            roa = np.random.normal(0.06, 0.08, n)
        else:
            de  = np.random.lognormal(0, 0.5, n)
            icr = np.random.lognormal(1, 0.6, n)
            cr  = np.random.uniform(0.5, 3.5, n)
            roa = np.random.normal(0.06, 0.08, n)

        rate_adj = 0.03 if "Rare" in scenario else 0.15
        z_true = -1.5 + 0.6*de - 0.4*icr - 0.5*cr - 3.0*roa
        if "Non-linear" in scenario:
            z_true += 0.1 * de**2  # quadratic effect
        z_true += np.log(rate_adj/(1-rate_adj))
        p_true = _sigmoid(z_true)
        y = (np.random.uniform(size=n) < p_true).astype(float)

        X = np.column_stack([np.ones(n), de, icr, cr, roa])
        names = ["Intercept","D/E Ratio","ICR","Current Ratio","ROA"]
        beta, se, ll, p_hat = _fit_logistic(X, y)

        # HL Test
        n_groups = 10
        q = np.percentile(p_hat, np.linspace(0, 100, n_groups+1))
        hl_stat = 0.0
        for lo, hi in zip(q[:-1], q[1:]):
            mask = (p_hat >= lo) & (p_hat <= hi)
            if mask.sum() > 0:
                O1 = y[mask].sum(); E1 = p_hat[mask].sum()
                O0 = (1-y[mask]).sum(); E0 = (1-p_hat[mask]).sum()
                hl_stat += (O1-E1)**2/(E1+1e-9) + (O0-E0)**2/(E0+1e-9)
        hl_p = 1 - stats.chi2.cdf(hl_stat, df=n_groups-2)

        # VIF
        vifs = []
        for j in range(1, X.shape[1]):
            Xother = np.delete(X[:, 1:], j-1, axis=1)
            Xother = np.column_stack([np.ones(n), Xother])
            b2 = np.linalg.lstsq(Xother, X[:,j], rcond=None)[0]
            r2_j = 1 - np.sum((X[:,j]-Xother@b2)**2)/(np.sum((X[:,j]-X[:,j].mean())**2)+1e-9)
            vifs.append(1/(1-r2_j+1e-9))

        # Null LL
        pb = y.mean()
        ll0 = n*(pb*np.log(pb+1e-12)+(1-pb)*np.log(1-pb+1e-12))
        mcf_r2 = 1 - ll/ll0
        lr_stat = -2*(ll0-ll)
        lr_p = 1 - stats.chi2.cdf(lr_stat, df=len(beta)-1)

        # Pearson residuals
        pearson = (y - p_hat) / np.sqrt(p_hat*(1-p_hat)+1e-9)
        # Deviance residuals
        dev_sign = np.where(y == 1, 1, -1)
        dev_res  = dev_sign * np.sqrt(-2*(y*np.log(p_hat+1e-9)+(1-y)*np.log(1-p_hat+1e-9)))

        # Leverage (hat matrix diagonal approximation)
        W = np.diag(p_hat*(1-p_hat))
        XtWX = X.T @ W @ X
        try:
            XtWX_inv = np.linalg.inv(XtWX)
            h = np.array([X[i] @ XtWX_inv @ (p_hat[i]*(1-p_hat[i])) * X[i] for i in range(n)])
        except Exception:
            h = np.zeros(n)

        # Display test results
        metric_row([
            ("Hosmer-Lemeshow χ²", f"{hl_stat:.3f}", None),
            ("HL p-value", f"{hl_p:.4f}", None),
            ("LR Statistic", f"{lr_stat:.2f}", None),
            ("McFadden R²", f"{mcf_r2:.4f}", None),
        ])

        st.html(table_html(
            ["Test","Statistic","Result","Interpretation"],
            [
                [bdg("Hosmer-Lemeshow","green"), _f(f"χ²={hl_stat:.3f}, p={hl_p:.4f}"),
                 gt("Calibrated ✓") if hl_p > 0.05 else rt2("Miscalibrated ✗"),
                 txt_s("Good calibration" if hl_p > 0.05 else "Predicted PD ≠ actual default rates")],
                [bdg("LR Test","blue"), _f(f"χ²={lr_stat:.2f}, p={lr_p:.4f}"),
                 gt("Significant ✓") if lr_p < 0.05 else rt2("Not significant"),
                 txt_s("Model improves on null" if lr_p < 0.05 else "No predictors help")],
            ]
        ))

        st.html(table_html(
            ["Variable","VIF","Severity"],
            [[hl(names[j+1]), _f(f"{vifs[j]:.2f}"),
              gt("OK") if vifs[j] < 5 else (org("Moderate") if vifs[j] < 10 else rt2("Severe"))]
             for j in range(len(vifs))]
        ))

        fig = _diagnostic_plots(y, p_hat, pearson, dev_res, h, scenario)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Violation alerts
        violations = []
        if hl_p < 0.05:
            violations.append(f'{rt2("Miscalibration")} — Consider {hl("Platt scaling")} or adding polynomial terms')
        if max(vifs) > 10:
            violations.append(f'{rt2("Multicollinearity")} (max VIF={max(vifs):.1f}) — Use {hl("Ridge logistic regression")} or drop correlated predictor')
        if "Rare" in scenario and y.mean() < 0.05:
            violations.append(f'{org("Rare events")} — Apply {hl("SMOTE oversampling")} or {hl("case-control sampling")} + correction')
        if "Outlier" in scenario:
            violations.append(f'{rt2("Influential outliers")} detected — Review {hl("Cook\'s D > 4/n")} threshold')

        if violations:
            render_ib(
                f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">⚠ Issues Detected:</span><br>'
                + "".join(f'<div style="margin-top:8px;color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">• {v}</div>'
                          for v in violations), "red"
            )
        else:
            render_ib(gt("✅ Diagnostics clear") + txt_s(" — Model appears well-specified and calibrated."), "green")


def _diagnostic_plots(y, p_hat, pearson, dev_res, h, scenario):
    fig = plt.figure(figsize=(14, 9), facecolor="#0a1628")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    def _sax(ax):
        ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.grid(color="#1e3a5f", alpha=0.35, lw=0.5)

    # Pearson residuals vs fitted
    ax1 = fig.add_subplot(gs[0,0]); _sax(ax1)
    ax1.scatter(p_hat, pearson, color="#ADD8E6", alpha=0.5, s=20)
    ax1.axhline(0, color="#FFD700", lw=1.5, ls="--")
    ax1.axhline(2, color="#dc3545", lw=1, ls=":", alpha=0.7); ax1.axhline(-2, color="#dc3545", lw=1, ls=":", alpha=0.7)
    ax1.set(xlabel="Fitted P(Y=1)", ylabel="Pearson Residual")
    ax1.xaxis.label.set_color("#8892b0"); ax1.yaxis.label.set_color("#8892b0")
    ax1.set_title("Pearson Residuals", color="#FFD700", fontsize=10)

    # Deviance residuals
    ax2 = fig.add_subplot(gs[0,1]); _sax(ax2)
    ax2.scatter(p_hat, dev_res, color="#ff9f43", alpha=0.5, s=20)
    ax2.axhline(0, color="#FFD700", lw=1.5, ls="--")
    ax2.axhline(2, color="#dc3545", lw=1, ls=":", alpha=0.7); ax2.axhline(-2, color="#dc3545", lw=1, ls=":", alpha=0.7)
    ax2.set(xlabel="Fitted P(Y=1)", ylabel="Deviance Residual")
    ax2.xaxis.label.set_color("#8892b0"); ax2.yaxis.label.set_color("#8892b0")
    ax2.set_title("Deviance Residuals", color="#FFD700", fontsize=10)

    # Calibration
    ax3 = fig.add_subplot(gs[0,2]); _sax(ax3)
    bins = np.percentile(p_hat, np.linspace(0,100,11))
    bins = np.unique(bins)
    obs_r, pred_m = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_hat>=lo)&(p_hat<hi)
        if mask.sum()>0:
            obs_r.append(y[mask].mean()); pred_m.append(p_hat[mask].mean())
    ax3.plot([0,1],[0,1], color="#8892b0", lw=1.5, ls="--")
    ax3.scatter(pred_m, obs_r, color="#FFD700", s=80, zorder=5)
    ax3.plot(pred_m, obs_r, color="#ADD8E6", lw=1.5)
    ax3.set(xlabel="Predicted P", ylabel="Observed Rate")
    ax3.xaxis.label.set_color("#8892b0"); ax3.yaxis.label.set_color("#8892b0")
    ax3.set_title("Calibration Plot", color="#FFD700", fontsize=10)

    # Leverage vs deviance
    ax4 = fig.add_subplot(gs[1,0]); _sax(ax4)
    ax4.scatter(h, np.abs(dev_res), color="#a29bfe", alpha=0.5, s=20)
    ax4.set(xlabel="Leverage hᵢᵢ", ylabel="|Deviance Residual|")
    ax4.xaxis.label.set_color("#8892b0"); ax4.yaxis.label.set_color("#8892b0")
    ax4.set_title("Influence Plot", color="#FFD700", fontsize=10)

    # Distribution of P(Y=1)
    ax5 = fig.add_subplot(gs[1,1]); _sax(ax5)
    ax5.hist(p_hat[y==0], bins=25, alpha=0.7, color="#ADD8E6", label="y=0", density=True)
    ax5.hist(p_hat[y==1], bins=25, alpha=0.7, color="#dc3545", label="y=1", density=True)
    ax5.set(xlabel="Predicted P(Default)", ylabel="Density")
    ax5.xaxis.label.set_color("#8892b0"); ax5.yaxis.label.set_color("#8892b0")
    ax5.set_title("Score Separation", color="#FFD700", fontsize=10)
    ax5.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")

    # Deviance residual Q-Q
    ax6 = fig.add_subplot(gs[1,2]); _sax(ax6)
    osm, osr = stats.probplot(dev_res, dist="norm")
    ax6.scatter(osm[0], osm[1], color="#64ffda", alpha=0.6, s=20)
    ax6.plot(osm[0], osm[0]*osr[0]+osr[1], color="#FFD700", lw=2)
    ax6.set(xlabel="Theoretical Quantiles", ylabel="Deviance Residual Quantiles")
    ax6.xaxis.label.set_color("#8892b0"); ax6.yaxis.label.set_color("#8892b0")
    ax6.set_title("Deviance Residual Q-Q", color="#FFD700", fontsize=10)

    return fig
