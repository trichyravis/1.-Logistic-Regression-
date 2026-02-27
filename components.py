"""
components.py — Mountain Path design system for Linear Regression app.
All HTML via st.html() with 100% inline styles + user-select:none.
"""
import streamlit as st

S = {
    "txt":  "#e6f1ff", "gold": "#FFD700", "lb":   "#ADD8E6",
    "grn":  "#28a745", "red":  "#dc3545", "acc":  "#64ffda",
    "mut":  "#8892b0", "card": "#112240", "blue": "#003366",
    "mid":  "#004d80", "dark": "#0a1628", "bdr":  "#1e3a5f",
    "org":  "#ff9f43", "pur":  "#a29bfe",
}
FH = "'Playfair Display',serif"
FB = "'Source Sans Pro',sans-serif"
FM = "'JetBrains Mono',monospace"
TXT = f"color:#e6f1ff;font-family:{FB};line-height:1.65;-webkit-text-fill-color:#e6f1ff"
NO_SEL = "user-select:none;-webkit-user-select:none"

_IB = {
    "blue":   ("rgba(0,51,102,0.6)",   "#ADD8E6"),
    "gold":   ("rgba(255,215,0,0.13)", "#FFD700"),
    "green":  ("rgba(40,167,69,0.2)",  "#28a745"),
    "red":    ("rgba(220,53,69,0.2)",  "#dc3545"),
    "orange": ("rgba(255,159,67,0.15)","#ff9f43"),
    "purple": ("rgba(162,155,254,0.15)","#a29bfe"),
}
_BADGE = {
    "blue":   ("#004d80","#ffffff"), "gold":   ("#FFD700","#0a1628"),
    "green":  ("#28a745","#ffffff"), "red":    ("#dc3545","#ffffff"),
    "orange": ("#ff9f43","#0a1628"), "purple": ("#a29bfe","#0a1628"),
}

def render_card(title: str, body_html: str):
    h2 = (f'<h2 style="font-family:{FH};font-size:1.35rem;color:#FFD700;'
          f'-webkit-text-fill-color:#FFD700;border-bottom:1px solid #1e3a5f;'
          f'padding-bottom:8px;margin:0 0 14px 0;{NO_SEL}">{title}</h2>')
    st.html(f'<div style="background:#112240;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:22px;margin-bottom:18px;{TXT};{NO_SEL}">{h2}{body_html}</div>')

def ib(content, variant="blue"):
    bg, bc = _IB.get(variant, _IB["blue"])
    return (f'<div style="background:{bg};border-left:4px solid {bc};border-radius:8px;'
            f'padding:13px 15px;margin:10px 0;{TXT};{NO_SEL}">{content}</div>')

def render_ib(content, variant="blue"): st.html(ib(content, variant))

def fml(content):
    return (f'<div style="background:#0d1f3a;border-left:4px solid #FFD700;border-radius:6px;'
            f'padding:13px 17px;margin:10px 0;font-family:{FM};font-size:.88rem;'
            f'color:#64ffda;-webkit-text-fill-color:#64ffda;line-height:1.85;'
            f'white-space:pre-wrap;overflow-x:auto;{NO_SEL}">{content}</div>')

def bdg(text, variant="blue"):
    bg, fg = _BADGE.get(variant, _BADGE["blue"])
    return (f'<span style="background:{bg};color:{fg};-webkit-text-fill-color:{fg};'
            f'display:inline-block;padding:2px 10px;border-radius:20px;font-size:.77rem;'
            f'font-weight:700;margin:2px;font-family:{FB};{NO_SEL}">{text}</span>')

def hl(t):   return f'<span style="color:#FFD700;-webkit-text-fill-color:#FFD700;font-weight:600">{t}</span>'
def gt(t):   return f'<span style="color:#28a745;-webkit-text-fill-color:#28a745;font-weight:600">{t}</span>'
def rt2(t):  return f'<span style="color:#dc3545;-webkit-text-fill-color:#dc3545;font-weight:600">{t}</span>'
def org(t):  return f'<span style="color:#ff9f43;-webkit-text-fill-color:#ff9f43;font-weight:600">{t}</span>'
def pur(t):  return f'<span style="color:#a29bfe;-webkit-text-fill-color:#a29bfe;font-weight:600">{t}</span>'
def lb_t(t): return f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6">{t}</span>'
def acc_t(t):return f'<span style="color:#64ffda;-webkit-text-fill-color:#64ffda;font-family:{FM}">{t}</span>'
def txt_s(t):return f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">{t}</span>'
def p(content): return f'<p style="{TXT};margin-bottom:7px">{content}</p>'

def steps_html(steps):
    rows = ""
    for i,(title,body) in enumerate(steps,1):
        rows += (f'<div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start;{NO_SEL}">'
                 f'<div style="background:#FFD700;color:#0a1628;-webkit-text-fill-color:#0a1628;'
                 f'border-radius:50%;min-width:28px;height:28px;display:flex;align-items:center;'
                 f'justify-content:center;font-weight:700;font-size:.85rem;font-family:{FB}">{i}</div>'
                 f'<div style="{TXT};flex:1">'
                 f'<span style="color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;font-weight:600">{title}</span><br>'
                 f'<span style="color:#e6f1ff;-webkit-text-fill-color:#e6f1ff">{body}</span>'
                 f'</div></div>')
    return rows

def two_col(left, right):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:10px 0">'
            f'<div>{left}</div><div>{right}</div></div>')

def three_col(a, b, c):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin:10px 0">'
            f'<div>{a}</div><div>{b}</div><div>{c}</div></div>')

def four_col(a, b, c, d):
    return (f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:12px;margin:10px 0">'
            f'<div>{a}</div><div>{b}</div><div>{c}</div><div>{d}</div></div>')

def table_html(headers, rows):
    ths = "".join(f'<th style="background:#003366;color:#FFD700;-webkit-text-fill-color:#FFD700;'
                  f'padding:9px 12px;text-align:left;font-weight:600;font-family:{FB}">{h}</th>'
                  for h in headers)
    trs = "".join(
        f'<tr>{"".join(f"<td style=\"padding:8px 12px;border-bottom:1px solid #1e3a5f;color:#e6f1ff;-webkit-text-fill-color:#e6f1ff;font-family:{FB}\">{c}</td>" for c in row)}</tr>'
        for row in rows)
    return (f'<table style="width:100%;border-collapse:collapse;margin:12px 0;font-size:.88rem;{NO_SEL}">'
            f'<tr>{ths}</tr>{trs}</table>')

def metric_row(metrics):
    cols = st.columns(len(metrics))
    for col,(label,value,*rest) in zip(cols,metrics):
        col.metric(label, value, rest[0] if rest else None)

def section_heading(title):
    st.html(f'<h3 style="font-family:{FH};color:#ADD8E6;-webkit-text-fill-color:#ADD8E6;'
            f'font-size:1.1rem;margin:18px 0 8px 0;{NO_SEL}">{title}</h3>')

def stat_box(label, value, sub="", variant="blue"):
    bg, bc = _IB.get(variant, _IB["blue"])
    val_col = {"blue":"#ADD8E6","gold":"#FFD700","green":"#28a745","red":"#dc3545","orange":"#ff9f43","purple":"#a29bfe"}.get(variant,"#ADD8E6")
    return (f'<div style="background:{bg};border:1px solid {bc};border-radius:8px;padding:14px 16px;{NO_SEL}">'
            f'<div style="color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.78rem;font-family:{FB};margin-bottom:4px">{label}</div>'
            f'<div style="color:{val_col};-webkit-text-fill-color:{val_col};font-family:{FM};font-size:1.3rem;font-weight:700">{value}</div>'
            f'{"<div style=\"color:#8892b0;-webkit-text-fill-color:#8892b0;font-size:.75rem;margin-top:3px\">" + sub + "</div>" if sub else ""}'
            f'</div>')
