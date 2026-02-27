"""
tab_model_fit.py — Model Fit, Pseudo-R², ROC/AUC, Confusion Matrix
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
    table_html, metric_row, section_heading, S, FH, FB, FM, TXT, NO_SEL
)

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'


def tab_model_fit():

    render_card("📊 Model Fit in Logistic Regression — No Single R²",
        p(f'OLS R² does not apply to logistic regression. Multiple {hl("pseudo-R²")} measures '
          f'and {hl("classification metrics")} are used instead.')
        + three_col(
            ib(f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">Pseudo-R² Measures</span><br>'
               + p(f'{bdg("McFadden","gold")} 1 − L(full)/L(null) → 0.2–0.4 is good<br>'
                   f'{bdg("Cox-Snell","blue")} 1 − (L₀/L)^(2/n)<br>'
                   f'{bdg("Nagelkerke","green")} Scaled Cox-Snell, max = 1<br>'
                   f'{bdg("Tjur R²","orange")} Mean(p̂|Y=1) − Mean(p̂|Y=0)'), "gold"),
            ib(f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">Log-Loss & Deviance</span><br>'
               + p(f'{bdg("Log-loss","blue")} −(1/n)Σ[Yᵢlnp̂ + (1−Yᵢ)ln(1−p̂)]<br>'
                   f'{bdg("Deviance","purple")} −2 × log-likelihood<br>'
                   f'{bdg("Null deviance","red")} Deviance with only intercept<br>'
                   f'{bdg("LR test","green")} χ²(k) = Null dev − Residual dev'), "blue"),
            ib(f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">Classification Metrics</span><br>'
               + p(f'{bdg("Accuracy","green")} (TP+TN)/(TP+TN+FP+FN)<br>'
                   f'{bdg("Precision","blue")} TP/(TP+FP)<br>'
                   f'{bdg("Recall/TPR","gold")} TP/(TP+FN)<br>'
                   f'{bdg("AUC-ROC","orange")} Area under ROC curve [0.5–1]'), "green"),
        )
    )

    render_card("📐 Pseudo-R² Formulas & Interpretation",
        fml("McFadden R²   = 1 − L(full) / L(null)        [good: 0.2–0.4]\n"
            "Cox-Snell R²  = 1 − (L₀/L_full)^(2/n)       [max < 1]\n"
            "Nagelkerke R² = Cox-Snell / max(Cox-Snell)   [max = 1]\n"
            "Tjur R²       = mean(p̂|Y=1) − mean(p̂|Y=0)  [discrimination]\n\n"
            "Likelihood Ratio Test (LR):\n"
            "  G² = −2[L(null) − L(full)] ~ χ²(k)  under H₀: all β = 0\n\n"
            "Hosmer-Lemeshow Test (calibration):\n"
            "  HL ~ χ²(8) → H₀: model is well-calibrated\n"
            "  p > 0.05 → calibration OK; p < 0.05 → poor fit")
        + table_html(
            ["Metric", "Formula", "Range", "Good Value"],
            [
                [bdg("McFadden R²","gold"),   _f("1 − L_full/L_null"),    txt_s("0 to 1"), gt("0.20 – 0.40")],
                [bdg("AUC-ROC","orange"),      _f("∫TPR d(FPR)"),           txt_s("0.5 to 1"), gt("> 0.75")],
                [bdg("Log-Loss","red"),        _f("−(1/n)ΣYlnp̂"),         txt_s("0 to ∞"), gt("Lower = better")],
                [bdg("Accuracy","green"),      _f("(TP+TN)/N"),             txt_s("0 to 1"), gt("> 0.80")],
                [bdg("Brier Score","blue"),    _f("(1/n)Σ(Yᵢ−p̂ᵢ)²"),     txt_s("0 to 1"), gt("< 0.25")],
            ]
        )
    )

    render_card("📊 Confusion Matrix & Classification Metrics",
        fml("               Predicted Y=0    Predicted Y=1\n"
            "Actual Y=0  |  TN (True Neg)  |  FP (False Pos) |  ← Specificity = TN/(TN+FP)\n"
            "Actual Y=1  |  FN (False Neg)  |  TP (True Pos) |  ← Sensitivity = TP/(TP+FN)\n\n"
            "Accuracy    = (TP + TN) / N\n"
            "Precision   = TP / (TP + FP)          [among predicted positives, how many are correct?]\n"
            "Recall/TPR  = TP / (TP + FN)          [among actual positives, how many are caught?]\n"
            "F1 Score    = 2 × Precision × Recall / (Precision + Recall)\n"
            "FPR         = FP / (FP + TN)          [false alarm rate — x-axis of ROC]\n\n"
            "Finance context:\n"
            "  Credit risk:  FN (missed default) >> FP (rejected good borrower) in cost\n"
            "  Fraud detect: FN (missed fraud)   >> FP (blocked legit tx) in cost")
    )

    # ── Interactive ROC & Confusion Matrix ─────────────────────
    render_card("🎮 Interactive: ROC Curve, AUC & Confusion Matrix",
        ib(f'{bdg("Simulate","gold")} '
           + txt_s("a credit default dataset and see ROC curve, AUC, and confusion matrix respond to threshold changes."),
           "gold")
    )

    col1, col2, col3 = st.columns(3)
    n_obs    = col1.number_input("Observations", 200, 2000, 500, 100, key="roc_n")
    auc_true = col2.slider("Model discrimination (AUC target)", 0.55, 0.98, 0.78, 0.01, key="roc_auc")
    thresh   = col3.slider("Classification threshold τ", 0.1, 0.9, 0.5, 0.05, key="roc_thr")
    seed_r   = st.number_input("Seed", 1, 999, 42, key="roc_seed")

    if st.button("📊 Generate ROC & Confusion Matrix", key="roc_btn"):
        np.random.seed(int(seed_r))
        n = int(n_obs)
        # Simulate scores that yield approximate target AUC
        y_true = np.random.binomial(1, 0.25, n)
        sep    = stats.norm.ppf(auc_true) * np.sqrt(2)
        scores = np.where(y_true == 1,
                          np.random.normal(sep/2, 1, n),
                          np.random.normal(-sep/2, 1, n))
        probs  = 1 / (1 + np.exp(-scores))
        y_pred = (probs >= thresh).astype(int)

        # Confusion matrix
        TP = int(np.sum((y_pred == 1) & (y_true == 1)))
        TN = int(np.sum((y_pred == 0) & (y_true == 0)))
        FP = int(np.sum((y_pred == 1) & (y_true == 0)))
        FN = int(np.sum((y_pred == 0) & (y_true == 1)))

        acc  = (TP+TN)/n
        prec = TP/(TP+FP) if (TP+FP) > 0 else 0
        rec  = TP/(TP+FN) if (TP+FN) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        spec = TN/(TN+FP) if (TN+FP) > 0 else 0

        # ROC curve
        thresholds  = np.linspace(0, 1, 200)
        tprs, fprs  = [], []
        for t in thresholds:
            yp   = (probs >= t).astype(int)
            tp_t = np.sum((yp==1)&(y_true==1))
            tn_t = np.sum((yp==0)&(y_true==0))
            fp_t = np.sum((yp==1)&(y_true==0))
            fn_t = np.sum((yp==0)&(y_true==1))
            tprs.append(tp_t/(tp_t+fn_t+1e-10))
            fprs.append(fp_t/(fp_t+tn_t+1e-10))
        tprs, fprs = np.array(tprs), np.array(fprs)
        # Sort by FPR for proper AUC
        idx  = np.argsort(fprs)
        fprs_s, tprs_s = fprs[idx], tprs[idx]
        auc_calc = np.trapezoid(tprs_s, fprs_s)

        # Precision-Recall
        precs_r, recs_r = [], []
        for t in thresholds:
            yp   = (probs >= t).astype(int)
            tp_t = np.sum((yp==1)&(y_true==1))
            fp_t = np.sum((yp==1)&(y_true==0))
            fn_t = np.sum((yp==0)&(y_true==1))
            precs_r.append(tp_t/(tp_t+fp_t+1e-10))
            recs_r.append(tp_t/(tp_t+fn_t+1e-10))

        fig = plt.figure(figsize=(14, 5), facecolor="#0a1628")
        gs  = GridSpec(1, 3, figure=fig, wspace=0.32)

        def sax(ax):
            ax.tick_params(colors="#8892b0", labelsize=8)
            for sp in ax.spines.values(): sp.set_color("#1e3a5f")
            ax.grid(color="#1e3a5f", alpha=0.3, lw=0.5)

        # Plot 1: ROC
        ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("#112240")
        ax1.plot(fprs_s, tprs_s, color="#FFD700", lw=2.5, label=f"AUC = {auc_calc:.4f}")
        ax1.fill_between(fprs_s, tprs_s, alpha=0.15, color="#FFD700")
        ax1.plot([0,1],[0,1], color="#8892b0", lw=1, ls="--", label="Random (AUC=0.5)")
        # Current threshold point
        curr_fpr = FP/(FP+TN+1e-10); curr_tpr = TP/(TP+FN+1e-10)
        ax1.scatter([curr_fpr],[curr_tpr], color="#dc3545", s=80, zorder=5, label=f"τ={thresh:.2f}")
        ax1.set_xlabel("FPR (1 - Specificity)", color="#8892b0", fontsize=9)
        ax1.set_ylabel("TPR (Sensitivity / Recall)", color="#8892b0", fontsize=9)
        ax1.set_title("ROC Curve", color="#FFD700", fontsize=11, fontweight="bold")
        ax1.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        sax(ax1)

        # Plot 2: Confusion Matrix heatmap
        ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("#112240")
        cm_vals = np.array([[TN, FP],[FN, TP]])
        cm_cols = np.array([["#28a745","#dc3545"],["#ff9f43","#004d80"]])
        for i in range(2):
            for j in range(2):
                ax2.add_patch(plt.Rectangle((j,1-i), 1, 1,
                    facecolor=cm_cols[i,j], alpha=0.75, edgecolor="#0a1628", lw=2))
                ax2.text(j+0.5, 1.5-i, str(cm_vals[i,j]),
                         ha="center", va="center",
                         color="white", fontsize=18, fontweight="bold",
                         fontfamily="monospace")
        ax2.set_xlim(0,2); ax2.set_ylim(0,2)
        ax2.set_xticks([0.5,1.5]); ax2.set_yticks([0.5,1.5])
        ax2.set_xticklabels(["Pred Y=0","Pred Y=1"], color="#e6f1ff", fontsize=9)
        ax2.set_yticklabels(["Actual Y=1","Actual Y=0"], color="#e6f1ff", fontsize=9)
        ax2.set_title(f"Confusion Matrix  (τ={thresh:.2f})", color="#FFD700", fontsize=10, fontweight="bold")
        ax2.tick_params(colors="#8892b0", labelsize=8, length=0)
        for sp in ax2.spines.values(): sp.set_color("#1e3a5f")
        # Labels
        labels = [["TN","FP"],["FN","TP"]]
        lbl_c  = [["#c8ffdc","#ffb3b3"],["#ffd9b3","#b3d9ff"]]
        for i in range(2):
            for j in range(2):
                ax2.text(j+0.5, 1.5-i+0.3, labels[i][j],
                         ha="center", va="center",
                         color=lbl_c[i][j], fontsize=9, fontfamily="monospace")

        # Plot 3: Precision-Recall
        ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor("#112240")
        ax3.plot(recs_r, precs_r, color="#64ffda", lw=2.5, label="PR Curve")
        ax3.axhline(np.mean(y_true), color="#8892b0", lw=1, ls="--", label=f"Baseline P={np.mean(y_true):.2f}")
        ax3.scatter([rec],[prec], color="#dc3545", s=80, zorder=5, label=f"τ={thresh:.2f}")
        ax3.set_xlabel("Recall (Sensitivity)", color="#8892b0", fontsize=9)
        ax3.set_ylabel("Precision", color="#8892b0", fontsize=9)
        ax3.set_title("Precision-Recall Curve", color="#FFD700", fontsize=11, fontweight="bold")
        ax3.legend(facecolor="#112240", labelcolor="#e6f1ff", fontsize=8, edgecolor="#1e3a5f")
        sax(ax3)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        metric_row([
            ("AUC-ROC",   f"{auc_calc:.4f}", None),
            ("Accuracy",  f"{acc:.4f}",      None),
            ("Precision", f"{prec:.4f}",     None),
            ("Recall",    f"{rec:.4f}",      None),
            ("F1 Score",  f"{f1:.4f}",       None),
            ("Specificity",f"{spec:.4f}",    None),
        ])

        st.html(table_html(
            ["Metric","Value","Finance Interpretation"],
            [
                [bdg("AUC-ROC","gold"),   hl(f"{auc_calc:.4f}"),
                 txt_s("Prob model ranks random default above random non-default. >0.75 = acceptable for credit models")],
                [bdg("Recall/TPR","orange"), hl(f"{rec:.4f}"),
                 txt_s("Of all actual defaults, % correctly flagged. Maximise to reduce missed defaults (costly FN)")],
                [bdg("Precision","blue"),  hl(f"{prec:.4f}"),
                 txt_s("Of all predicted defaults, % that truly defaulted. Maximise to reduce false alarms (FP)")],
                [bdg("F1 Score","green"),  hl(f"{f1:.4f}"),
                 txt_s("Harmonic mean of precision+recall. Good single metric when classes are imbalanced")],
            ]
        ))

        # Threshold sensitivity guidance
        if rec < 0.6:
            render_ib(rt2(f"⚠ Recall = {rec:.2f} — many defaults are missed at τ={thresh}. "
                          "Consider lowering threshold to reduce FN at cost of more FP."), "red")
        elif prec < 0.4:
            render_ib(org(f"⚠ Precision = {prec:.2f} — many false alarms at τ={thresh}. "
                          "Consider raising threshold to reduce FP at cost of more FN."), "orange")
        else:
            render_ib(gt(f"✅ Good balance at τ={thresh}: Recall={rec:.2f}, Precision={prec:.2f}, AUC={auc_calc:.3f}"), "green")
