"""
tab_code.py — Python Reference Code for Logistic Regression
MLE from scratch, sklearn, statsmodels, diagnostics, finance applications
"""
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from components import (
    render_card, ib, render_ib, fml, bdg, hl, gt, rt2,
    txt_s, p, two_col, section_heading, metric_row, stat_box, four_col,
    S, FH, FB, FM, TXT, NO_SEL
)

CODES = {
    "MLE from Scratch": '''import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_mle(X, y, max_iter=200, tol=1e-8):
    """Newton-Raphson MLE for logistic regression."""
    n, k = X.shape
    beta = np.zeros(k)
    for iteration in range(max_iter):
        p    = sigmoid(X @ beta)          # predicted probabilities
        W    = p * (1 - p)                # weight matrix diagonal
        grad = X.T @ (y - p)             # score (gradient)
        H    = -(X.T * W) @ X            # Hessian matrix
        try:
            delta = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            break
        beta -= delta                     # Newton step
        if np.max(np.abs(delta)) < tol:
            break
    p_final = sigmoid(X @ beta)
    H_final = -(X.T * (p_final*(1-p_final))) @ X
    cov     = np.linalg.inv(-H_final)
    se      = np.sqrt(np.diag(cov))      # Standard errors
    ll      = np.sum(y*np.log(p_final+1e-12) + (1-y)*np.log(1-p_final+1e-12))
    return beta, se, ll, p_final

# Example: Credit default prediction
np.random.seed(42)
n = 300
de  = np.random.lognormal(0, 0.5, n)   # D/E ratio
icr = np.random.lognormal(1, 0.6, n)   # Interest coverage
roa = np.random.normal(0.06, 0.08, n)  # ROA
z_true = -1.5 + 0.6*de - 0.4*icr - 3.0*roa
y = (np.random.uniform(size=n) < sigmoid(z_true)).astype(float)
X = np.column_stack([np.ones(n), de, icr, roa])

beta, se, ll, p_hat = logistic_mle(X, y)
wald_z = beta / se
wald_p = 2 * (1 - stats.norm.cdf(np.abs(wald_z)))
or_vals = np.exp(beta)

print("Variable      Beta      SE        OR      Wald-z    p-value")
for i, name in enumerate(["Intercept","D/E","ICR","ROA"]):
    print(f"{name:<14}{beta[i]:>+.4f}  {se[i]:.4f}  {or_vals[i]:.4f}  {wald_z[i]:>+.3f}  {wald_p[i]:.4f}")

# McFadden R²
p_bar = y.mean()
ll0   = n*(p_bar*np.log(p_bar+1e-12) + (1-p_bar)*np.log(1-p_bar+1e-12))
print(f"\\nMcFadden R²: {1-ll/ll0:.4f}")''',

    "statsmodels (Full Output)": '''import numpy as np
import statsmodels.api as sm
import pandas as pd

np.random.seed(42)
n = 300
de  = np.random.lognormal(0, 0.5, n)
icr = np.random.lognormal(1, 0.6, n)
cr  = np.random.uniform(0.5, 3.5, n)
roa = np.random.normal(0.06, 0.08, n)
z   = -1.5 + 0.6*de - 0.4*icr - 0.5*cr - 3.0*roa
y   = (np.random.uniform(size=n) < 1/(1+np.exp(-z))).astype(float)

X = pd.DataFrame({"DE":de, "ICR":icr, "CR":cr, "ROA":roa})
X = sm.add_constant(X)

# Fit logistic regression
model = sm.Logit(y, X)
result = model.fit(method="newton", maxiter=200, disp=False)
print(result.summary())

# Key outputs
print("\\n--- Odds Ratios ---")
print(np.exp(result.params))
print("\\n--- 95% CI for OR ---")
print(np.exp(result.conf_int()))

# Marginal effects (Average Marginal Effect = AME)
mfx = result.get_margeff()
print("\\n--- Average Marginal Effects ---")
print(mfx.summary())

# Predictions
p_hat = result.predict(X)
print(f"\\nMcFadden R²: {result.prsquared:.4f}")
print(f"LR statistic: {result.llr:.4f}, p={result.llr_pvalue:.6f}")''',

    "sklearn + ROC + Calibration": '''import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500
X = np.column_stack([
    np.random.lognormal(0,0.5,n),  # D/E
    np.random.lognormal(1,0.6,n),  # ICR
    np.random.uniform(0.5,3.5,n),  # CR
    np.random.normal(0.06,0.08,n)  # ROA
])
z = -1.5 + 0.6*X[:,0] - 0.4*X[:,1] - 0.5*X[:,2] - 3.0*X[:,3]
y = (np.random.uniform(size=n) < 1/(1+np.exp(-z))).astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Fit model
clf = LogisticRegression(max_iter=500, random_state=42)
clf.fit(X_train_sc, y_train)
p_hat = clf.predict_proba(X_test_sc)[:,1]

# Evaluation
auc  = roc_auc_score(y_test, p_hat)
gini = 2*auc - 1
cv_auc = cross_val_score(clf, scaler.transform(X), y, cv=5, scoring="roc_auc")

print(f"AUC-ROC:  {auc:.4f}")
print(f"Gini:     {gini:.4f}")
print(f"CV AUC:   {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
print("\\n", classification_report(y_test, (p_hat>=0.5).astype(int)))

# Calibration
frac_pos, mean_pred = calibration_curve(y_test, p_hat, n_bins=10)
print("Calibration — Predicted vs Observed:")
for mp, fp in zip(mean_pred, frac_pos):
    print(f"  Predicted: {mp:.3f}  |  Observed: {fp:.3f}")''',

    "Diagnostics & GOF": '''import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
def fit_logistic(X,y):
    n,k=X.shape; beta=np.zeros(k)
    for _ in range(200):
        p=sigmoid(X@beta); W=p*(1-p)+1e-12
        grad=X.T@(y-p); H=-(X.T*W)@X
        try: delta=np.linalg.solve(H,grad)
        except: break
        beta-=delta
        if np.max(np.abs(delta))<1e-8: break
    p_f=sigmoid(X@beta)
    try: cov=np.linalg.inv(-(X.T*(p_f*(1-p_f)))@X); se=np.sqrt(np.diag(cov))
    except: se=np.full(k,np.nan)
    ll=np.sum(y*np.log(p_f+1e-12)+(1-y)*np.log(1-p_f+1e-12))
    return beta,se,ll,p_f

def hosmer_lemeshow(y, p_hat, g=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    thresholds = np.percentile(p_hat, np.linspace(0,100,g+1))
    stat = 0.0
    for lo,hi in zip(thresholds[:-1], thresholds[1:]):
        mask=(p_hat>=lo)&(p_hat<=hi)
        if mask.sum()==0: continue
        O1=y[mask].sum(); E1=p_hat[mask].sum()
        O0=(1-y[mask]).sum(); E0=(1-p_hat[mask]).sum()
        stat += (O1-E1)**2/(E1+1e-9) + (O0-E0)**2/(E0+1e-9)
    pval = 1 - stats.chi2.cdf(stat, df=g-2)
    return stat, pval

def compute_vif(X_vars):
    """VIF for each predictor (excluding intercept column)."""
    vifs = []
    n,k = X_vars.shape
    for j in range(k):
        X_oth = np.delete(X_vars,j,axis=1)
        X_oth = np.column_stack([np.ones(n),X_oth])
        b=np.linalg.lstsq(X_oth,X_vars[:,j],rcond=None)[0]
        r2=1-np.sum((X_vars[:,j]-X_oth@b)**2)/(np.sum((X_vars[:,j]-X_vars[:,j].mean())**2)+1e-9)
        vifs.append(1/(1-r2+1e-9))
    return vifs

# Example
np.random.seed(42)
n=250; de=np.random.lognormal(0,0.5,n); icr=np.random.lognormal(1,0.6,n)
cr=np.random.uniform(0.5,3.5,n); roa=np.random.normal(0.06,0.08,n)
z=-1.5+0.6*de-0.4*icr-0.5*cr-3.0*roa
y=(np.random.uniform(size=n)<sigmoid(z)).astype(float)
X=np.column_stack([np.ones(n),de,icr,cr,roa])

beta,se,ll,p_hat = fit_logistic(X,y)
hl_stat,hl_p = hosmer_lemeshow(y, p_hat)
vifs = compute_vif(X[:,1:])

print(f"Hosmer-Lemeshow: χ²={hl_stat:.3f}, p={hl_p:.4f}")
print("Calibration:", "PASS (p>0.05)" if hl_p>0.05 else "FAIL")
for nm,v in zip(["D/E","ICR","CR","ROA"],vifs):
    print(f"VIF({nm}): {v:.2f} — {'OK' if v<5 else 'MODERATE' if v<10 else 'SEVERE'}")

# Deviance residuals
dev_sign = np.where(y==1,1,-1)
dev_res  = dev_sign*np.sqrt(-2*(y*np.log(p_hat+1e-9)+(1-y)*np.log(1-p_hat+1e-9)))
print(f"\\nDeviance residuals std: {dev_res.std():.4f}")
print(f"Outliers (|dev|>3): {(np.abs(dev_res)>3).sum()}")''',
}

def _f(t):
    return f'<span style="font-family:{FM};font-size:.83rem;color:#64ffda;-webkit-text-fill-color:#64ffda">{t}</span>'

def tab_code():
    render_card("🐍 Python Reference Code — Logistic Regression",
        p(f'Complete {hl("production-ready Python")} implementations. Choose from MLE from scratch, '
          f'statsmodels (full output), sklearn pipeline, and diagnostics.')
        + two_col(
            ib(f'{bdg("MLE from scratch","gold")} Newton-Raphson, Hessian SE, Wald tests<br>'
               f'{bdg("statsmodels","blue")} Full summary, OR CI, marginal effects<br>'
               f'{bdg("sklearn","green")} Pipeline, CV, calibration, scoring', "gold"),
            ib(f'{bdg("Hosmer-Lemeshow","purple")} Manual χ² calibration test<br>'
               f'{bdg("VIF","red")} Multicollinearity detection<br>'
               f'{bdg("Deviance residuals","orange")} Influence and outlier detection', "blue"),
        )
    )

    section = st.selectbox("Code Section", list(CODES.keys()), key="code_sec")
    st.html(fml(CODES[section]))

    # Live Quick Calculator
    render_card("⚡ Live Logistic Calculator",
        ib(f'Enter comma-separated {hl("binary Y (0/1)")} and a single {hl("X predictor")} to instantly fit a logistic regression.', "blue")
    )
    col1, col2 = st.columns(2)
    y_in = col1.text_area("Y values (0 or 1, comma-separated)", "1,0,1,1,0,0,1,0,1,1,0,1,0,0,1", height=80, key="lc_y")
    x_in = col2.text_area("X values (comma-separated)", "3.5,1.2,4.1,2.8,0.9,1.5,3.8,1.1,4.5,3.2,0.7,2.9,1.0,1.8,3.6", height=80, key="lc_x")

    if st.button("⚡ Fit & Plot", key="lc_run"):
        try:
            y_arr = np.array([float(v.strip()) for v in y_in.split(",")])
            x_arr = np.array([float(v.strip()) for v in x_in.split(",")])
            assert len(y_arr)==len(x_arr) and len(y_arr)>=10
            assert set(np.unique(y_arr)).issubset({0.0,1.0})

            def _sig(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
            X_mat = np.column_stack([np.ones(len(y_arr)), x_arr])
            n,k = X_mat.shape; beta=np.zeros(k)
            for _ in range(200):
                p=_sig(X_mat@beta); W=p*(1-p)+1e-12
                grad=X_mat.T@(y_arr-p); H=-(X_mat.T*W)@X_mat
                try: delta=np.linalg.solve(H,grad)
                except: break
                beta-=delta
                if np.max(np.abs(delta))<1e-8: break
            p_hat=_sig(X_mat@beta)
            try: se=np.sqrt(np.diag(np.linalg.inv(-(X_mat.T*(p_hat*(1-p_hat)))@X_mat)))
            except: se=np.array([np.nan,np.nan])
            ll=np.sum(y_arr*np.log(p_hat+1e-12)+(1-y_arr)*np.log(1-p_hat+1e-12))
            pb=y_arr.mean(); ll0=len(y_arr)*(pb*np.log(pb+1e-12)+(1-pb)*np.log(1-pb+1e-12))
            mcf=1-ll/ll0; wald_z=beta/se; wald_p=2*(1-stats.norm.cdf(np.abs(wald_z)))

            metric_row([
                ("β₀ (Intercept)", f"{beta[0]:+.4f}", None),
                ("β₁ (Slope)", f"{beta[1]:+.4f}", None),
                ("OR = e^β₁", f"{np.exp(beta[1]):.4f}", None),
                ("McFadden R²", f"{mcf:.4f}", None),
            ])

            st.html(fml(
                f"β₀ = {beta[0]:+.4f}  SE={se[0]:.4f}  z={wald_z[0]:+.3f}  p={wald_p[0]:.4f}\n"
                f"β₁ = {beta[1]:+.4f}  SE={se[1]:.4f}  z={wald_z[1]:+.3f}  p={wald_p[1]:.4f}\n"
                f"OR = e^β₁ = {np.exp(beta[1]):.4f}\n"
                f"95% CI OR: [{np.exp(beta[1]-1.96*se[1]):.4f}, {np.exp(beta[1]+1.96*se[1]):.4f}]"
            ))

            fig, axes = plt.subplots(1,2,figsize=(12,4),facecolor="#0a1628")
            def _sax(ax):
                ax.set_facecolor("#112240"); ax.tick_params(colors="#8892b0",labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#1e3a5f")
                ax.grid(color="#1e3a5f",alpha=0.35,lw=0.5)
            x_line=np.linspace(x_arr.min(),x_arr.max(),200)
            p_line=_sig(beta[0]+beta[1]*x_line)
            axes[0].scatter(x_arr[y_arr==0],y_arr[y_arr==0],color="#ADD8E6",alpha=0.6,s=50,label="y=0")
            axes[0].scatter(x_arr[y_arr==1],y_arr[y_arr==1],color="#dc3545",alpha=0.6,s=50,label="y=1")
            axes[0].plot(x_line,p_line,color="#FFD700",lw=2.5,label="Fitted sigmoid")
            axes[0].axhline(0.5,color="#64ffda",lw=1,ls=":",alpha=0.7)
            axes[0].set(xlabel="X",ylabel="P(Y=1)"); axes[0].xaxis.label.set_color("#8892b0"); axes[0].yaxis.label.set_color("#8892b0")
            axes[0].set_title("Logistic Fit",color="#FFD700",fontsize=10)
            axes[0].legend(facecolor="#112240",labelcolor="#e6f1ff",fontsize=8,edgecolor="#1e3a5f")
            _sax(axes[0])
            pearson=(y_arr-p_hat)/np.sqrt(p_hat*(1-p_hat)+1e-9)
            axes[1].scatter(p_hat,pearson,color="#ff9f43",alpha=0.7,s=50)
            axes[1].axhline(0,color="#FFD700",lw=1.5,ls="--")
            axes[1].axhline(2,color="#dc3545",lw=1,ls=":",alpha=0.7); axes[1].axhline(-2,color="#dc3545",lw=1,ls=":",alpha=0.7)
            axes[1].set(xlabel="Fitted P",ylabel="Pearson Residual"); axes[1].xaxis.label.set_color("#8892b0"); axes[1].yaxis.label.set_color("#8892b0")
            axes[1].set_title("Pearson Residuals",color="#FFD700",fontsize=10)
            _sax(axes[1])
            plt.tight_layout(pad=1.5)
            st.pyplot(fig,use_container_width=True); plt.close(fig)

        except Exception as e:
            st.html(ib(rt2(f"Error: {e}. Check that Y contains only 0/1 and both lists have ≥10 equal-length entries."), "red"))
