"""
tab_finance.py — Finance Case Studies for Logistic Regression
Credit Default, Fraud Detection, Rating Migration, Loan Approval
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2, org, pur,
    lb_t, txt_s, p, steps_html, two_col, three_col,
    table_html, metric_row, section_heading, stat_box, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

def _sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))

def _fit_logistic(X, y):
    n, k = X.shape; beta = np.zeros(k)
    for _ in range(200):
        p = _sigmoid(X@beta); W = p*(1-p)+1e-12
        grad = X.T@(y-p); H = -(X.T*W)@X
        try: delta = np.linalg.solve(H,grad)
        except Exception: break
        beta -= delta
        if np.max(np.abs(delta)) < 1e-8: break
    p_f = _sigmoid(X@beta)
    W_f = p_f*(1-p_f)+1e-12; H_f = -(X.T*W_f)@X
    try: cov=np.linalg.inv(-H_f); se=np.sqrt(np.diag(cov))
    except Exception: se=np.full(k,np.nan)
    ll = np.sum(y*np.log(p_f+1e-12)+(1-y)*np.log(1-p_f+1e-12))
    return beta, se, ll, p_f


def tab_finance():
    cases = ["💳 Credit Default Prediction", "🔍 Fraud Detection",
             "📉 Credit Rating Downgrade", "🏦 Loan Approval Model"]
    case = st.radio("Select Case Study", cases, horizontal=True, key="fin_case")

    if "Credit Default" in case:
        _credit_default()
    elif "Fraud" in case:
        _fraud_detection()
    elif "Downgrade" in case:
        _rating_downgrade()
    else:
        _loan_approval()


def _credit_default():
    render_card("💳 Case Study 1: Corporate Credit Default Prediction",
        p(f'Predict {hl("P(Corporate Default)")} using accounting ratios. Based on {hl("Altman Z-Score")} framework '
          f'recast as logistic regression. Used by banks for IFRS 9 Expected Credit Loss (ECL) computation.')
        + two_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Model:</span><br>'
               + fml("P(Default) = σ(β₀ + β₁·D/E + β₂·ICR + β₃·CurrentRatio\n"
                     "           + β₄·ROA + β₅·AssetTurnover)"),
               "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">Expected Signs:</span><br>'
               + table_html(["Variable","Expected β","Finance Logic"],[
                   [txt_s("D/E Ratio"),       rt2("+ positive"), txt_s("More leverage → higher default risk")],
                   [txt_s("ICR"),             gt("− negative"), txt_s("Higher coverage → lower risk")],
                   [txt_s("Current Ratio"),   gt("− negative"), txt_s("More liquidity → lower risk")],
                   [txt_s("ROA"),             gt("− negative"), txt_s("More profitable → lower risk")],
                   [txt_s("Asset Turnover"),  gt("− negative"), txt_s("More efficient → lower risk")],
               ]), "blue")
        )
    )
    col1, col2 = st.columns(2)
    n = col1.number_input("Firms", 100, 600, 300, 50, key="cd_n")
    seed = col2.number_input("Seed", 1, 999, 21, key="cd_seed")

    if st.button("▶ Run Credit Default Model", key="cd_run"):
        np.random.seed(int(seed))
        n = int(n)
        de  = np.random.lognormal(0, 0.6, n)
        icr = np.random.lognormal(1.2, 0.7, n)
        cr  = np.random.uniform(0.4, 4.0, n)
        roa = np.random.normal(0.07, 0.09, n)
        at  = np.random.uniform(0.3, 2.5, n)
        z   = -2.0 + 0.55*de - 0.45*icr - 0.6*cr - 3.5*roa - 0.4*at
        y   = (np.random.uniform(size=n) < _sigmoid(z)).astype(float)
        X   = np.column_stack([np.ones(n), de, icr, cr, roa, at])
        names = ["Intercept","D/E","ICR","Current Ratio","ROA","Asset Turnover"]

        beta, se, ll, p_hat = _fit_logistic(X, y)
        pb = y.mean(); ll0 = n*(pb*np.log(pb+1e-12)+(1-pb)*np.log(1-pb+1e-12))
        mcf_r2 = 1-ll/ll0; lr_stat = -2*(ll0-ll); lr_p = 1-stats.chi2.cdf(lr_stat, df=5)

        # AUC
        thresholds = np.linspace(0,1,100)
        tprs, fprs = [], []
        for t in thresholds:
            yp=(p_hat>=t).astype(int)
            tp=np.sum((yp==1)&(y==1)); fp=np.sum((yp==1)&(y==0))
            tn=np.sum((yp==0)&(y==0)); fn=np.sum((yp==0)&(y==1))
            tprs.append(tp/(tp+fn+1e-9)); fprs.append(fp/(fp+tn+1e-9))
        fprs_a = np.array(fprs); tprs_a = np.array(tprs)
        idx = np.argsort(fprs_a); auc = np.trapezoid(tprs_a[idx], fprs_a[idx])

        metric_row([
            ("Default Rate", f"{y.mean():.3f}", None),
            ("McFadden R²", f"{mcf_r2:.4f}", None),
            ("AUC-ROC", f"{auc:.4f}", None),
            ("LR p-value", f"{lr_p:.4f}", None),
        ])

        wald_z = beta/se; wald_p = 2*(1-stats.norm.cdf(np.abs(wald_z)))
        or_vals = np.exp(beta)
        expected = {"D/E":"+","ICR":"-","Current Ratio":"-","ROA":"-","Asset Turnover":"-"}

        rows = []
        for i,nm in enumerate(names):
            sign_ok = "?" if nm=="Intercept" else (
                gt("✓") if (beta[i]>0)==(expected.get(nm,"+")=="+") else rt2("✗"))
            rows.append([hl(nm), _f(f"{beta[i]:+.4f}"), _f(f"{or_vals[i]:.4f}"),
                         _f(f"{wald_z[i]:+.3f}"), txt_s(f"{wald_p[i]:.4f}"),
                         gt("***") if wald_p[i]<0.001 else gt("**") if wald_p[i]<0.01
                         else gt("*") if wald_p[i]<0.05 else rt2("ns"), sign_ok])
        section_heading("📋 Coefficient Table")
        st.html(table_html(["Variable","β̂","OR=e^β","Wald z","p-value","Sig","Sign OK?"], rows))

        _plot_credit(y, p_hat, fprs_a[idx], tprs_a[idx])

        # PD for example firm
        section_heading("🔮 Point Prediction: Example Firm")
        ex_de, ex_icr, ex_cr, ex_roa, ex_at = 2.5, 1.8, 1.2, 0.03, 0.8
        z_ex = beta[0]+beta[1]*ex_de+beta[2]*ex_icr+beta[3]*ex_cr+beta[4]*ex_roa+beta[5]*ex_at
        pd_ex = _sigmoid(z_ex)
        st.html(ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Example firm: </span>'
            + txt_s(f'D/E=2.5, ICR=1.8, CR=1.2, ROA=3%, AssetTurnover=0.8<br>')
            + f'<span style="color:{"#dc3545" if pd_ex>0.20 else "#28a745"};'
            + f'-webkit-text-fill-color:{"#dc3545" if pd_ex>0.20 else "#28a745"};font-size:1.4rem;font-weight:700">'
            + f'Predicted PD = {pd_ex:.4f} ({pd_ex*100:.2f}%) — '
            + f'{"⚠ HIGH RISK" if pd_ex>0.20 else "✓ MODERATE RISK" if pd_ex>0.10 else "✓ LOW RISK"}'
            + f'</span>', "gold"
        ))


def _plot_credit(y, p_hat, fprs, tprs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0a1628")
    def _sax(ax):
        ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#1e3a5f")
        ax.grid(color="#1e3a5f", alpha=0.35, lw=0.5)

    axes[0].hist(p_hat[y==0], bins=30, alpha=0.7, color="#ADD8E6", label="Non-default", density=True)
    axes[0].hist(p_hat[y==1], bins=30, alpha=0.7, color="#dc3545", label="Default", density=True)
    axes[0].set(xlabel="P(Default)"); axes[0].xaxis.label.set_color("#8892b0")
    axes[0].set_title("PD Score Distribution", color="#FFD700", fontsize=10)
    axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
    _sax(axes[0])

    auc = np.trapezoid(tprs, fprs)
    axes[1].plot(fprs, tprs, color="#FFD700", lw=2.5, label=f"AUC={auc:.3f}")
    axes[1].plot([0,1],[0,1], color="#8892b0", lw=1, ls="--"); axes[1].fill_between(fprs, tprs, alpha=0.12, color="#FFD700")
    axes[1].set(xlabel="FPR", ylabel="TPR"); axes[1].xaxis.label.set_color("#8892b0"); axes[1].yaxis.label.set_color("#8892b0")
    axes[1].set_title("ROC Curve", color="#FFD700", fontsize=10)
    axes[1].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
    _sax(axes[1])

    # PD by decile
    deciles = np.percentile(p_hat, np.arange(10,101,10))
    dec_rates = []
    for lo, hi in zip(np.percentile(p_hat, np.arange(0,100,10)), deciles):
        mask=(p_hat>=lo)&(p_hat<=hi); dec_rates.append(y[mask].mean() if mask.sum()>0 else 0)
    colors = ["#dc3545" if r>0.3 else "#ff9f43" if r>0.15 else "#28a745" for r in dec_rates]
    axes[2].bar(range(1,11), dec_rates, color=colors, edgecolor="#ADD8E6", alpha=0.85)
    axes[2].set(xlabel="Score Decile (1=low risk, 10=high risk)", ylabel="Actual Default Rate")
    axes[2].xaxis.label.set_color("#8892b0"); axes[2].yaxis.label.set_color("#8892b0")
    axes[2].set_title("Default Rate by Score Decile", color="#FFD700", fontsize=10)
    _sax(axes[2])

    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _fraud_detection():
    render_card("🔍 Case Study 2: Transaction Fraud Detection",
        p(f'Predict {hl("P(Transaction = Fraud)")} using transaction characteristics. '
          f'Critical in retail banking, e-commerce, and payment systems. '
          f'Highly {rt2("imbalanced")} — fraud rate typically 0.1%–2%.')
        + two_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Model:</span><br>'
               + fml("P(Fraud) = σ(β₀ + β₁·Amount + β₂·TimeOfDay\n"
                     "         + β₃·DiffFromAvg + β₄·ForeignTxn\n"
                     "         + β₅·NewMerchant)"),
               "gold"),
            ib(f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">Class Imbalance Challenge:</span><br>'
               + steps_html([
                   ("SMOTE", "Synthetic Minority Oversampling — creates synthetic fraud samples"),
                   ("Cost-sensitive learning", "Higher penalty for missing fraud (false negatives)"),
                   ("Threshold tuning", "Lower threshold to catch more fraud at cost of false alarms"),
                   ("Precision-Recall AUC", "Better than ROC-AUC for imbalanced data"),
               ]), "red")
        )
    )
    col1, col2, col3 = st.columns(3)
    n = col1.number_input("Transactions", 500, 5000, 2000, 250, key="fr_n")
    fraud_rate = col2.slider("Fraud rate (%)", 1, 20, 5, 1, key="fr_rate") / 100
    seed = col3.number_input("Seed", 1, 999, 55, key="fr_seed")

    if st.button("▶ Run Fraud Detection Model", key="fr_run"):
        np.random.seed(int(seed))
        n = int(n)
        amount   = np.random.lognormal(4, 1.2, n)
        tod      = np.random.uniform(0, 24, n)
        diff_avg = np.random.normal(0, 2, n)
        foreign  = (np.random.uniform(size=n) < 0.15).astype(float)
        new_merch= (np.random.uniform(size=n) < 0.20).astype(float)

        z = (np.log(fraud_rate/(1-fraud_rate))
             + 0.0008*amount + 0.15*(tod>22).astype(float) - 0.05*tod
             + 0.8*diff_avg + 2.5*foreign + 1.8*new_merch)
        y = (np.random.uniform(size=n) < _sigmoid(z)).astype(float)
        X = np.column_stack([np.ones(n), amount, tod, diff_avg, foreign, new_merch])
        names = ["Intercept","Amount","Time of Day","Diff from Avg","Foreign Txn","New Merchant"]

        beta, se, ll, p_hat = _fit_logistic(X, y)

        # Threshold sweep for F1
        ths = np.linspace(0.01, 0.99, 100)
        f1s = []
        for t in ths:
            yp = (p_hat>=t).astype(int)
            tp=np.sum((yp==1)&(y==1)); fp=np.sum((yp==1)&(y==0)); fn=np.sum((yp==0)&(y==1))
            prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
            f1s.append(2*prec*rec/(prec+rec+1e-9))
        best_t = ths[np.argmax(f1s)]

        yp_opt = (p_hat >= best_t).astype(int)
        tp=np.sum((yp_opt==1)&(y==1)); fp=np.sum((yp_opt==1)&(y==0))
        fn=np.sum((yp_opt==0)&(y==1)); tn=np.sum((yp_opt==0)&(y==0))
        prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)

        metric_row([
            ("Fraud Rate", f"{y.mean():.3f}", None),
            ("Optimal Threshold", f"{best_t:.3f}", None),
            ("Best F1 Score", f"{f1:.4f}", None),
            ("Recall (Sensitivity)", f"{rec:.4f}", None),
        ])

        wald_z = beta/se; wald_p = 2*(1-stats.norm.cdf(np.abs(wald_z)))
        rows = []
        for i,nm in enumerate(names):
            rows.append([hl(nm), _f(f"{beta[i]:+.4f}"), _f(f"{np.exp(beta[i]):.4f}"),
                         _f(f"{wald_z[i]:+.3f}"), txt_s(f"{wald_p[i]:.4f}"),
                         gt("***") if wald_p[i]<0.001 else gt("**") if wald_p[i]<0.01
                         else gt("*") if wald_p[i]<0.05 else rt2("ns")])

        section_heading("📋 Fraud Model Coefficients")
        st.html(table_html(["Variable","β̂","OR=e^β","Wald z","p-value","Sig"], rows))

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#0a1628")
        def _sax(ax):
            ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0", labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#1e3a5f")
            ax.grid(color="#1e3a5f", alpha=0.35, lw=0.5)

        axes[0].hist(p_hat[y==0], bins=40, alpha=0.7, color="#ADD8E6", label="Legit", density=True)
        axes[0].hist(p_hat[y==1], bins=40, alpha=0.7, color="#dc3545", label="Fraud", density=True)
        axes[0].axvline(best_t, color="#FFD700", lw=2, ls="--", label=f"Threshold={best_t:.3f}")
        axes[0].set_title("Fraud Score Distribution", color="#FFD700", fontsize=10)
        axes[0].set(xlabel="P(Fraud)"); axes[0].xaxis.label.set_color("#8892b0")
        axes[0].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=7, edgecolor="#1e3a5f")
        _sax(axes[0])

        precs, recs_plt = [], []
        for t in ths:
            yp2=(p_hat>=t).astype(int)
            tp2=np.sum((yp2==1)&(y==1)); fp2=np.sum((yp2==1)&(y==0)); fn2=np.sum((yp2==0)&(y==1))
            precs.append(tp2/(tp2+fp2+1e-9)); recs_plt.append(tp2/(tp2+fn2+1e-9))
        axes[1].plot(recs_plt, precs, color="#FFD700", lw=2.5)
        axes[1].axhline(y.mean(), color="#dc3545", lw=1.5, ls="--", label=f"Baseline={y.mean():.3f}")
        axes[1].scatter([rec],[prec], color="#28a745", s=120, zorder=5, label=f"Best F1={f1:.3f}")
        axes[1].set(xlabel="Recall", ylabel="Precision")
        axes[1].xaxis.label.set_color("#8892b0"); axes[1].yaxis.label.set_color("#8892b0")
        axes[1].set_title("Precision-Recall Curve", color="#FFD700", fontsize=10)
        axes[1].legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        _sax(axes[1])

        axes[2].bar(["True Neg","False Pos","False Neg","True Pos"],
                    [tn,fp,fn,tp], color=["#28a745","#ff9f43","#dc3545","#ADD8E6"], alpha=0.8)
        axes[2].set_title(f"Confusion Matrix (t={best_t:.2f})", color="#FFD700", fontsize=10)
        axes[2].tick_params(axis='x', colors="#8892b0", labelsize=8); axes[2].tick_params(axis='y', colors="#8892b0")
        _sax(axes[2])

        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def _rating_downgrade():
    render_card("📉 Case Study 3: Credit Rating Downgrade Prediction",
        p(f'Predict {hl("P(Rating Downgrade)")} for a corporate bond. Used by {hl("credit analysts")} '
          f'at rating agencies (Moody\'s, S&P) and portfolio managers for early warning systems.')
        + two_col(
            fml("P(Downgrade) = σ(β₀ + β₁·InterestCoverage + β₂·DebtToAssets\n"
                "             + β₃·RevenueGrowth + β₄·CashFlow + β₅·SectorDummy)"),
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Industry context:</span><br>'
               + p(f'{bdg("IFRS 9","blue")} Requires 12-month and lifetime PD<br>'
                   f'{bdg("Basel III","orange")} IRB approach uses internal PD models<br>'
                   f'{bdg("CDS spreads","purple")} Market-implied PD complement<br>'
                   f'{bdg("Transition matrix","gold")} Historical migration probabilities'), "gold")
        )
    )
    col1, col2 = st.columns(2)
    n = col1.number_input("Bonds", 100, 500, 250, 50, key="rg_n")
    seed = col2.number_input("Seed", 1, 999, 33, key="rg_seed")

    if st.button("▶ Run Rating Downgrade Model", key="rg_run"):
        np.random.seed(int(seed))
        n = int(n)
        icov = np.random.lognormal(1.5, 0.7, n)
        dta  = np.random.uniform(0.1, 0.8, n)
        revg = np.random.normal(0.05, 0.12, n)
        cf   = np.random.normal(0.08, 0.06, n)
        sector = (np.random.uniform(size=n)<0.35).astype(float)  # 1=cyclical

        z = -1.0 - 0.4*icov + 2.5*dta - 1.5*revg - 3.0*cf + 0.8*sector
        y = (np.random.uniform(size=n) < _sigmoid(z)).astype(float)
        X = np.column_stack([np.ones(n), icov, dta, revg, cf, sector])
        names = ["Intercept","Interest Coverage","Debt/Assets","Revenue Growth","Cash Flow","Cyclical Sector"]
        beta, se, ll, p_hat = _fit_logistic(X, y)
        pb=y.mean(); ll0=n*(pb*np.log(pb+1e-12)+(1-pb)*np.log(1-pb+1e-12))
        mcf_r2=1-ll/ll0

        metric_row([
            ("Downgrade Rate", f"{y.mean():.3f}", None),
            ("McFadden R²", f"{mcf_r2:.4f}", None),
        ])

        wald_z = beta/se; wald_p = 2*(1-stats.norm.cdf(np.abs(wald_z)))
        rows = [[hl(names[i]), _f(f"{beta[i]:+.4f}"), _f(f"{np.exp(beta[i]):.4f}"),
                 _f(f"{wald_z[i]:+.3f}"), txt_s(f"{wald_p[i]:.4f}"),
                 gt("***") if wald_p[i]<0.001 else gt("**") if wald_p[i]<0.01
                 else gt("*") if wald_p[i]<0.05 else rt2("ns")]
                for i in range(len(names))]
        section_heading("📋 Downgrade Model Coefficients")
        st.html(table_html(["Variable","β̂","OR=e^β","Wald z","p-value","Sig"], rows))

        # Early warning
        ex = [1.2, 0.65, -0.02, 0.03, 1.0]  # stressed firm
        z_ex = beta[0]+sum(beta[i+1]*v for i,v in enumerate(ex))
        pd_ex = _sigmoid(z_ex)
        st.html(ib(
            f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Early Warning Example:</span><br>'
            + txt_s(f'ICov=1.2, D/A=65%, RevGrowth=−2%, CF=3%, Cyclical sector<br>')
            + f'<span style="color:{"#dc3545" if pd_ex>0.40 else "#ff9f43" if pd_ex>0.20 else "#28a745"};'
            + f'-webkit-text-fill-color:{"#dc3545" if pd_ex>0.40 else "#ff9f43" if pd_ex>0.20 else "#28a745"};'
            + f'font-size:1.3rem;font-weight:700">P(Downgrade) = {pd_ex:.4f} — '
            + ("⚠ DOWNGRADE ALERT" if pd_ex>0.40 else "⚡ Watch List" if pd_ex>0.20 else "✓ Stable")
            + f'</span>', "gold"
        ))


def _loan_approval():
    render_card("🏦 Case Study 4: Retail Loan Approval Model",
        p(f'Binary logistic model for {hl("P(Loan Approved)")}. Used by retail banks for '
          f'automated credit decisioning (scorecard models). Regulated under {hl("Fair Lending")} and ECOA.')
        + two_col(
            fml("P(Approved) = σ(β₀ + β₁·CreditScore + β₂·IncomeLTI\n"
                "            + β₃·EmploymentYears + β₄·ExistingDebt\n"
                "            + β₅·CollateralValue)"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">Credit Scorecard Context:</span><br>'
               + p(f'{bdg("Scorecard","blue")} Points-based system from logistic β<br>'
                   f'{bdg("FICO equivalent","gold")} β × 40 + PDO offset<br>'
                   f'{bdg("Cutoff score","orange")} Optimal threshold = 0.50 (or custom)<br>'
                   f'{bdg("Override rate","red")} Manual review if score near cutoff<br>'
                   f'{bdg("Gini coefficient","green")} 2×AUC − 1 for model power'), "blue")
        )
    )
    col1, col2 = st.columns(2)
    n = col1.number_input("Loan applications", 200, 1000, 500, 100, key="la_n")
    seed = col2.number_input("Seed", 1, 999, 77, key="la_seed")

    if st.button("▶ Run Loan Approval Model", key="la_run"):
        np.random.seed(int(seed))
        n = int(n)
        cscore = np.random.normal(680, 80, n).clip(300, 850)
        lti    = np.random.uniform(1.0, 6.0, n)
        empyrs = np.random.exponential(5, n).clip(0, 35)
        debt   = np.random.uniform(0.1, 0.6, n)
        coll   = np.random.lognormal(4, 0.5, n)

        z = (-2.0 + 0.008*(cscore-650) - 0.4*lti + 0.08*empyrs - 2.0*debt + 0.001*coll)
        y = (np.random.uniform(size=n) < _sigmoid(z)).astype(float)
        X = np.column_stack([np.ones(n), cscore, lti, empyrs, debt, coll])
        names = ["Intercept","Credit Score","LTI Ratio","Employment Years","Existing Debt","Collateral Value"]
        beta, se, ll, p_hat = _fit_logistic(X, y)

        thresholds=np.linspace(0,1,100)
        tprs_a, fprs_a = [], []
        for t in thresholds:
            yp=(p_hat>=t).astype(int)
            tp=np.sum((yp==1)&(y==1)); fp=np.sum((yp==1)&(y==0))
            fn=np.sum((yp==0)&(y==1)); tn=np.sum((yp==0)&(y==0))
            tprs_a.append(tp/(tp+fn+1e-9)); fprs_a.append(fp/(fp+tn+1e-9))
        fprs_s=np.array(fprs_a); tprs_s=np.array(tprs_a)
        idx=np.argsort(fprs_s); auc=np.trapezoid(tprs_s[idx],fprs_s[idx])
        gini = 2*auc - 1

        metric_row([
            ("Approval Rate", f"{y.mean():.3f}", None),
            ("AUC-ROC", f"{auc:.4f}", None),
            ("Gini Coefficient", f"{gini:.4f}", None),
        ])

        wald_z = beta/se; wald_p=2*(1-stats.norm.cdf(np.abs(wald_z)))
        rows=[[hl(names[i]),_f(f"{beta[i]:+.4f}"),_f(f"{np.exp(beta[i]):.4f}"),
               _f(f"{wald_z[i]:+.3f}"),txt_s(f"{wald_p[i]:.4f}"),
               gt("***") if wald_p[i]<0.001 else gt("**") if wald_p[i]<0.01
               else gt("*") if wald_p[i]<0.05 else rt2("ns")]
              for i in range(len(names))]
        section_heading("📋 Loan Model Coefficients")
        st.html(table_html(["Variable","β̂","OR=e^β","Wald z","p-value","Sig"],rows))

        # Plot score distribution
        fig, axes = plt.subplots(1,2,figsize=(12,4.5),facecolor="#0a1628")
        def _sax(ax):
            ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0",labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#1e3a5f")
            ax.grid(color="#1e3a5f",alpha=0.35,lw=0.5)
        axes[0].hist(p_hat[y==0],bins=30,alpha=0.7,color="#dc3545",label="Rejected",density=True)
        axes[0].hist(p_hat[y==1],bins=30,alpha=0.7,color="#28a745",label="Approved",density=True)
        axes[0].axvline(0.5,color="#FFD700",lw=2,ls="--",label="Cutoff=0.5")
        axes[0].set_title("Approval Score Distribution",color="#FFD700",fontsize=10)
        axes[0].set(xlabel="P(Approved)"); axes[0].xaxis.label.set_color("#8892b0")
        axes[0].legend(facecolor="#112240",labelcolor="#e6f1ff",fontsize=8,edgecolor="#1e3a5f")
        _sax(axes[0])
        axes[1].plot(fprs_s[idx],tprs_s[idx],color="#FFD700",lw=2.5,label=f"AUC={auc:.3f}, Gini={gini:.3f}")
        axes[1].plot([0,1],[0,1],color="#8892b0",lw=1,ls="--")
        axes[1].fill_between(fprs_s[idx],tprs_s[idx],alpha=0.12,color="#FFD700")
        axes[1].set(xlabel="FPR",ylabel="TPR"); axes[1].xaxis.label.set_color("#8892b0"); axes[1].yaxis.label.set_color("#8892b0")
        axes[1].set_title("ROC Curve — Loan Model",color="#FFD700",fontsize=10)
        axes[1].legend(facecolor="#112240",labelcolor="#e6f1ff",fontsize=8,edgecolor="#1e3a5f")
        _sax(axes[1])
        plt.tight_layout(pad=1.5)
        st.pyplot(fig,use_container_width=True)
        plt.close(fig)
