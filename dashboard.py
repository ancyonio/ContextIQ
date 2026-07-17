"""ContextIQ savings dashboard.

A Streamlit view over the token-savings ledger written by tokengraph_all.py
at `.context/gain.ndjson`. Panels:

  1. Hero + scorecard — headline $ saved, tokens saved, reduction %, runs
  2. Waterfall        — baseline -> saved -> sent, the single clearest visual
  3. Savings over time — stacked-area trend by day
  4. Savings by operation — where the savings come from

Run it with:

    pip install streamlit plotly
    streamlit run dashboard.py

It reads the ledger directly, so no server or API layer is needed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from tokengraph_all import (gain_by_operation as by_op, gain_daily as daily,
                            gain_totals as totals, read_gain_ledger)

# Same pricing table tokengraph_all.py uses for its $ projection.
PRICES_PER_1M: dict[str, float] = {
    "claude-opus": 15.0, "claude-sonnet": 3.0, "claude-haiku": 0.80,
    "gpt-4o": 2.5, "gpt-4o-mini": 0.15, "gpt-4.1": 2.0,
    "gemini-1.5-pro": 1.25, "gemini-1.5-flash": 0.075,
}
RANGES = {"All time": None, "Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--path", default=".")
_dashboard_args, _ = _parser.parse_known_args(sys.argv[1:])
ROOT = Path(_dashboard_args.path).resolve()
LEDGER = ROOT / ".context" / "gain.ndjson"
LOGO = Path(__file__).parent / "ContextIQ.png"
NAVY, BLUE, GREEN = "#1a2b4a", "#2f6fed", "#2e9e5b"  # brand navy/blue; green = savings


def human(n: float) -> str:
    """Compact number: 1_234_567 -> '1.2M'."""
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + unit
    return f"{int(n):,}"


# --- page -------------------------------------------------------------------
try:                                    # use the logo as the browser favicon
    from PIL import Image
    ICON = Image.open(LOGO) if LOGO.exists() else "📊"
except Exception:
    ICON = "📊"

st.set_page_config(page_title="ContextIQ — token savings", page_icon=ICON,
                   layout="wide")
if LOGO.exists() and hasattr(st, "logo"):   # persistent top-left brand mark
    st.logo(str(LOGO))
st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 2.2rem;}
  .hero {background: linear-gradient(135deg, #1a2b4a, #2f6fed); color: #fff;
         padding: 1.5rem 1.9rem; border-radius: 14px; margin: .3rem 0 1.3rem;}
  .hero-label {font-size: .78rem; text-transform: uppercase;
               letter-spacing: .09em; opacity: .85;}
  .hero-value {font-size: 2.9rem; font-weight: 700; line-height: 1.15;}
  .hero-sub {font-size: .98rem; opacity: .92;}
</style>
""", unsafe_allow_html=True)

head_l, head_r = st.columns([1, 5], vertical_alignment="center")
with head_l:
    if LOGO.exists():
        st.image(str(LOGO), width=120)
with head_r:
    st.markdown("### Token savings dashboard")
    st.caption("Token savings from code-graph context packs vs. reading whole files.")

AUTO_REFRESH_SECONDS = 10  # how often the fragment below re-reads the ledger

# --- sidebar (regular, non-fragment widgets trigger a full app rerun) -------
_all_rows = read_gain_ledger(ROOT)
if not _all_rows:
    st.warning(
        f"No ledger yet at `{LEDGER}`.\n\n"
        "Run some context/ask/measure calls (or `python tokengraph_all.py "
        "context \"a task\"`) to populate it, then refresh."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Pricing model", list(PRICES_PER_1M),
                         index=list(PRICES_PER_1M).index("claude-sonnet"))
    rng = st.selectbox("Date range", list(RANGES))
    st.divider()
    st.caption(f"Ledger: `{LEDGER.name}`")
    st.caption(f"{len(_all_rows)} total run(s) recorded")
    st.caption(f"Last refreshed {datetime.now(timezone.utc):%H:%M:%S} UTC "
               f"· auto every {AUTO_REFRESH_SECONDS}s")
    if st.button("🔄 Refresh now"):
        st.rerun()


# The panels are a fragment so they re-read the ledger and redraw on their own
# every AUTO_REFRESH_SECONDS — otherwise Streamlit only reruns on a widget
# interaction, so new runs appended to the ledger by tokengraph_all.py would
# never show up until the user manually reloaded the page. Fragments can't
# write to the sidebar, so the controls above stay outside this function.
@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def render_dashboard(model: str, rng: str) -> None:
    all_rows = read_gain_ledger(ROOT)

    if not all_rows:
        st.warning(
            f"No ledger yet at `{LEDGER}`.\n\n"
            "Run some context/ask/measure calls (or `python tokengraph_all.py "
            "context \"a task\"`) to populate it, then refresh."
        )
        st.stop()

    price = PRICES_PER_1M[model]
    now = datetime.now(timezone.utc)

    # Slice to the selected window, and the prior window for deltas.
    days = RANGES[rng]
    if days is None:
        rows, prev_rows = all_rows, []
    else:
        span = days * 86400
        cutoff = now.timestamp() - span
        rows = [r for r in all_rows if float(r.get("ts", 0)) >= cutoff]
        prev_rows = [r for r in all_rows
                     if cutoff - span <= float(r.get("ts", 0)) < cutoff]

    if not rows:
        st.info(f"No runs in **{rng.lower()}**. Try a wider range in the sidebar.")
        st.stop()

    t = totals(rows)
    p = totals(prev_rows)
    saved_usd = t["saved"] / 1_000_000 * price

    def delta(cur: int, prev: int) -> str | None:
        """Signed compact delta vs. the prior window, or None when there's no prior."""
        if not prev_rows:
            return None
        return f"{'+' if cur >= prev else '-'}{human(abs(cur - prev))} vs prev"

    # --- Panel 1: hero + scorecard ---------------------------------------
    st.markdown(f"""
    <div class="hero">
      <div class="hero-label">Estimated cost saved · {model}</div>
      <div class="hero-value">${saved_usd:,.2f}</div>
      <div class="hero-sub">{human(t['saved'])} tokens never sent ·
           prompts {t['reduction_pct']}% smaller · {rng.lower()}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Tokens saved", human(t["saved"]), delta(t["saved"], p["saved"]))
    c2.metric("Reduction", f"{t['reduction_pct']}%",
              None if not prev_rows else f"{round(t['reduction_pct'] - p['reduction_pct'], 1):+} pts")
    c3.metric("Runs", f"{t['runs']:,}", delta(t["runs"], p["runs"]))

    if t["runs"] < 2:
        st.info("Early days — only one run in view. The charts below fill out as "
                "you use the graph across more tasks and days.")

    st.divider()

    # --- Panel 2: waterfall -----------------------------------------------
    st.subheader("Baseline → saved → sent")
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "total"],
        x=["Baseline (whole files)", "Saved by ContextIQ", "Actually sent"],
        y=[t["baseline"], -t["saved"], t["final"]],
        text=[human(t["baseline"]), f"-{human(t['saved'])}", human(t["final"])],
        textposition="outside",
        connector={"line": {"color": "#c8ced9"}},
        decreasing={"marker": {"color": GREEN}},
        totals={"marker": {"color": BLUE}},
    ))
    fig.update_layout(showlegend=False, yaxis_title="tokens", height=420,
                      margin=dict(t=30, b=20), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Sending {human(t['final'])} of {human(t['baseline'])} tokens "
        f"({100 - t['reduction_pct']:.1f}% of baseline) across {t['runs']} run(s)."
    )

    st.divider()

    # --- Panel 3: trend + Panel 4: by-op -----------------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("Savings over time")
        d = daily(rows)
        if len(d) < 2:
            st.info("Only one day of data in view — the trend fills in over more days.")
        xs = [r["day"] for r in d]
        trend = go.Figure()
        trend.add_trace(go.Scatter(
            x=xs, y=[r["final"] for r in d], name="Sent", mode="lines+markers",
            stackgroup="one", line={"color": BLUE}))
        trend.add_trace(go.Scatter(
            x=xs, y=[r["saved"] for r in d], name="Saved", mode="lines+markers",
            stackgroup="one", line={"color": GREEN}))
        trend.update_layout(height=380, yaxis_title="tokens", margin=dict(t=10, b=20),
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.12))
        st.plotly_chart(trend, use_container_width=True)

    with right:
        st.subheader("Savings by operation")
        ops = by_op(rows)
        bar = go.Figure(go.Bar(
            orientation="h",
            y=[o["op"] for o in ops][::-1],
            x=[o["saved"] for o in ops][::-1],
            text=[f"{human(o['saved'])} · {o['runs']} run(s)" for o in ops][::-1],
            textposition="auto", marker={"color": GREEN},
        ))
        bar.update_layout(height=380, xaxis_title="tokens saved", margin=dict(t=10, b=20),
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, use_container_width=True)

    st.divider()
    with st.expander("Raw ledger records"):
        st.dataframe(rows, use_container_width=True)


render_dashboard(model, rng)
