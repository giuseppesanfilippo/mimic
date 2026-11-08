"""Streamlit port of the Mimic GUI by G. Sanfilippo.

The app constructs the four reference conditional-probability curves used in
Lad & Sanfilippo (2020), then inverts them to an exchangeable pmf and plots
the induced CDF and prior mass function.
"""
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Mimic GUI", layout="wide")
st.title("Mimic GUI")
st.caption("Python/Streamlit port of the original R/Shiny application")

@st.cache_data(show_spinner=False)
def curves(n, p_l, f_l, f_u, p_u, c1=0.0, c2=0.0):
    x = np.arange(n + 1, dtype=float) / n
    y = {}
    left, middle, right = x < f_l, (x >= f_l) & (x <= f_u), x > f_u
    # Exact translations of computepLinear, computepWeak and computepStrict.
    linear = np.empty_like(x); linear[left] = p_l + (f_l-p_l)/f_l*x[left]; linear[middle] = x[middle]; linear[right] = f_u + (p_u-f_u)/(1-f_u)*(x[right]-f_u)
    weak = np.empty_like(x); weak[left] = p_l; weak[middle] = x[middle]; weak[right] = p_u
    strict = np.empty_like(x); strict[left] = f_l; strict[middle] = x[middle]; strict[right] = f_u
    y.update(linear=linear, weak=weak, strict=strict)
    def quartic_side(xv, x0, x1, y1, c):
        e, d = y1, 0.0
        a = c/(x0-x1)**2 + (1+2*d)/(x0-x1)**3 - 3*(x0-e)/(x0-x1)**4
        b = -2*c/(x0-x1) - (1+3*d)/(x0-x1)**2 + 4*(x0-e)/(x0-x1)**3
        return np.clip(a*(xv-x1)**4+b*(xv-x1)**3+c*(xv-x1)**2+d*(xv-x1)+e, 0, 1)
    quartic = np.empty_like(x)
    quartic[left] = np.clip(quartic_side(x[left], f_l, 0, p_l, c1), p_l, f_l)
    quartic[middle] = x[middle]
    quartic[right] = np.clip(quartic_side(x[right], f_u, 1, p_u, c2), f_u, p_u)
    y["quartic"] = quartic
    return x, y

def pmf_from_conditional(n, p):
    # Exact stable equivalent of server.R::computeq: forward and backward
    # products are combined before normalization to avoid overflow.
    odds = np.clip(p, 1e-12, 1-1e-12) / np.clip(1-p, 1e-12, 1)
    logf = np.log((n + 1 - np.arange(n + 1)) / np.arange(1, n + 2)) + np.log(odds)
    forward = np.r_[0.0, np.cumsum(logf)]
    backward = np.r_[np.cumsum(logf[::-1])[::-1], 0.0]
    logs = np.logaddexp(forward, backward)
    logs -= np.max(logs)
    return np.exp(logs) / np.exp(logs).sum()

def reduce_to_n2(p, n2):
    """Translate server.R::fromptop to Python."""
    out = np.asarray(p, dtype=float).copy()
    current = len(out) - 1
    for k in range(current, n2, -1):
        out = out[:-1] / (1 - out[1:] + out[:-1])
    return out

with st.sidebar:
    st.header("Parameters")
    n1 = st.number_input("N₁", min_value=1, max_value=100000, value=1000, step=1)
    n2 = st.slider("N₂", min_value=5, max_value=1000, value=100, step=1)
    st.subheader("Assessed conditional probabilities")
    selected = st.multiselect("Curves", ["linear", "quartic", "weak", "strict"], default=["linear", "quartic", "weak", "strict"])
    st.subheader("N-type plots")
    plot_n1 = st.checkbox("Compute at N₁ (solid)", value=True)
    plot_reduced = st.checkbox("Reduce to N₂ (dash-dash)", value=True)
    plot_n2 = st.checkbox("Compute at N₂ (dash-dot)", value=True)
    st.subheader("Mimic interval")
    a1 = st.slider("a₁ / N", 0.005, 0.995, 0.25, 0.005)
    a2 = st.slider("a₂ / N", 0.005, 0.995, 0.60, 0.005)
    p_l = st.slider("pL(N)", 0.005, 0.995, 0.10, 0.005)
    p_u = st.slider("pU(N)", 0.005, 0.995, 0.70, 0.005)
    c1 = st.slider("c₁", -50, 50, 0)
    c2 = st.slider("c₂", -50, 50, 0)

if a2 < a1:
    st.error("a₂/N deve essere maggiore o uguale ad a₁/N.")
    st.stop()

x, ys = curves(int(n1), p_l, a1, a2, p_u, c1, c2)
reduced = {name: reduce_to_n2(values, int(n2)) for name, values in ys.items()}
x2, ys2 = curves(int(n2), p_l, a1, a2, p_u, c1, c2)
fig = go.Figure()
colors = {"linear":"#1f3fb4", "quartic":"#d95f02", "weak":"#65c43b", "strict":"#222222"}
for name in selected:
    if plot_n1:
        fig.add_trace(go.Scatter(x=x, y=ys[name], name=f"{name.title()} N₁", line=dict(color=colors[name], width=2)))
    if plot_reduced:
        fig.add_trace(go.Scatter(x=np.linspace(0, 1, len(reduced[name])), y=reduced[name], name=f"{name.title()} reduced N₂", line=dict(color=colors[name], width=2, dash="dash")))
    if plot_n2:
        fig.add_trace(go.Scatter(x=x2, y=ys2[name], name=f"{name.title()} N₂", line=dict(color=colors[name], width=2, dash="dashdot")))
fig.add_vline(x=a1, line_dash="dot", line_color="gray"); fig.add_vline(x=a2, line_dash="dot", line_color="gray")
fig.update_layout(title="Assessed conditional probability", xaxis_title="a/N", yaxis_title="P(Eₙ₊₁ | Sₙ=a)", yaxis_range=[0,1], height=430)
st.plotly_chart(fig, use_container_width=True)

tab_pdf, tab_cdf = st.tabs(["Plot Pdf", "Plot Cdf"])
with tab_pdf:
    fig2 = go.Figure()
    for name in selected:
        if plot_n1:
            q = pmf_from_conditional(int(n1), ys[name]); fig2.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=q*int(n1), name=f"{name.title()} N₁", line=dict(color=colors[name])))
        if plot_reduced:
            q = pmf_from_conditional(int(n2), reduced[name]); fig2.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=q*int(n2), name=f"{name.title()} reduced N₂", line=dict(color=colors[name], dash="dash")))
        if plot_n2:
            q = pmf_from_conditional(int(n2), ys2[name]); fig2.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=q*int(n2), name=f"{name.title()} N₂", line=dict(color=colors[name], dash="dashdot")))
    fig2.update_layout(title="Implied Prior Mass Functions for proportions of N+1", xaxis_title="a/N", yaxis_title="mass", height=480)
    st.plotly_chart(fig2, use_container_width=True)
with tab_cdf:
    fig3 = go.Figure()
    for name in selected:
        if plot_n1:
            q = pmf_from_conditional(int(n1), ys[name]); fig3.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=np.cumsum(q), name=f"{name.title()} N₁", line=dict(color=colors[name])))
        if plot_reduced:
            q = pmf_from_conditional(int(n2), reduced[name]); fig3.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=np.cumsum(q), name=f"{name.title()} reduced N₂", line=dict(color=colors[name], dash="dash")))
        if plot_n2:
            q = pmf_from_conditional(int(n2), ys2[name]); fig3.add_trace(go.Scatter(x=np.linspace(0,1,len(q)), y=np.cumsum(q), name=f"{name.title()} N₂", line=dict(color=colors[name], dash="dashdot")))
    fig3.update_layout(title="Implied CDF", xaxis_title="a/N", yaxis_title="CDF", yaxis_range=[0,1], height=480)
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("Reference: [Lad & Sanfilippo (2020)](https://doi.org/10.1007/s10203-020-00281-z)")
