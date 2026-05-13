"""
AI Resolution Tracker  —  v4
Built for: Jason (Support Ops) + Product Owners + Operations Managers
No statistics jargon. Plain business language throughout.

HOW TO RUN — LOCAL
------------------
1. pip install streamlit pandas matplotlib numpy
2. streamlit run ai_resolution_forecast_app.py
3. Upload your CSV using the file uploader on the page.

HOW TO DEPLOY (free, no coding needed)
---------------------------------------
1. Push this file to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", pick your repo and this file.
4. Click Deploy — done. You get a shareable URL.

COST MODEL
----------
  1. Bot resolved + within the 50k limit  →  $0.50 / ticket  (contract rate)
  2. Bot resolved + already over the limit →  $2.00 / ticket  (overage penalty)
  3. Bot passed ticket to a human agent    →  $4.00 / ticket
Counterfactual: every ticket handled by a human from the start → $4.00 each.
"""

from datetime import date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import streamlit as st


# ─── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resolution Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricLabel"] { font-size:.72rem; text-transform:uppercase;
                                 letter-spacing:.05em; color:#555; }
[data-testid="stMetricValue"] { font-size:1.8rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)

PALETTE = [
    "#1B4F8A", "#E8873A", "#2E9E6B", "#C0392B", "#8E44AD",
    "#1A7A8A", "#D4AC0D", "#884EA0", "#2874A6", "#148F77",
]

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": "#EBEBEB", "grid.linewidth": .6,
    "font.size": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


# ─── Sidebar settings ─────────────────────────────────────────────────────────
st.sidebar.title("Settings")
st.sidebar.caption("Pre-filled with publicly available Zendesk pricing (2025). Adjust to match your actual contract.")

# ── PRIMARY: AI contract settings ─────────────────────────────────────────────
st.sidebar.markdown("#### 🤖 AI contract")

contract_start = st.sidebar.date_input(
    "Contract start date",
    value=date(2025, 1, 1),
    help=(
        "The first day of your current contract year. "
        "Your annual resolution allowance resets on this date every 12 months."
    ),
)

allowance = st.sidebar.number_input(
    "Annual AI resolution allowance",
    min_value=1_000, max_value=1_000_000,
    value=50_000, step=1_000,
    help=(
        "The total number of AI-resolved tickets included in your contract. "
        "Going above this triggers overage charges at the pay-as-you-go rate."
    ),
)

annual_ai_fee = st.sidebar.number_input(
    "Annual AI platform fee ($)",
    min_value=0.0, value=10_000.0, step=500.0,
    help=(
        "Fixed annual fee for access to the AI resolution feature — "
        "charged on top of per-resolution costs. Set to 0 if not applicable."
    ),
)

vendor_discount_pct = st.sidebar.number_input(
    "Vendor discount (% off contract fee)",
    min_value=0.0, max_value=100.0, value=10.0, step=1.0,
    help="Discount your vendor gave you on the contract fee. Does not apply to per-resolution usage charges.",
)

st.sidebar.markdown("#### 💲 Cost per resolution")

cost_ai_contract = st.sidebar.number_input(
    "Within allowance ($/resolution)",
    value=0.50, min_value=0.0, step=0.10,
    help=(
        "What you pay per AI-resolved ticket while still within your annual allowance. "
        "Pre-filled at $0.50 — adjust to match your committed rate."
    ),
)
cost_ai_overage = st.sidebar.number_input(
    "Over the limit — pay-as-you-go ($/resolution)",
    value=2.00, min_value=0.0, step=0.50,
    help=(
        "Penalty rate per AI-resolved ticket once you exceed the annual allowance. "
        "Pre-filled at $2.00 based on publicly available Zendesk PAYG pricing (2025)."
    ),
)

# Agent defaults — hidden from UI, used only in background calculations
num_agents             = 10
cost_per_seat          = 20.0
cost_human             = 4.00
agent_monthly_salary   = 3_000.0
agent_tickets_per_day  = 20
working_days_per_month = 22


# ─── Page header + CSV upload ─────────────────────────────────────────────────
st.title("🤖 AI Resolution Tracker")

uploaded_file = st.file_uploader(
    "Upload your support data CSV to get started",
    type=["csv"],
    help="Upload the master support dataset. "
         "It needs columns: Date and Time Contact, Source, "
         "Chatbot Resolved, Handover (Y/N), Brand, Issue.",
)

if uploaded_file is None:
    st.info(
        "👆 Upload your CSV file above to load the dashboard. "
        "Use the sidebar to set your contract dates and cost rates."
    )
    st.stop()


# ─── Load data ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df["Date and Time Contact"] = pd.to_datetime(
        df["Date and Time Contact"], errors="coerce")
    df["_source"]   = df["Source"].astype(str).str.lower().str.strip()
    df["_resolved"] = df["Chatbot Resolved"].astype(str).str.lower().str.strip()
    df["_handover"] = df["Handover (Y/N)"].astype(str).str.upper().str.strip()
    return df

df = load_data(uploaded_file)


# ─── Contract-year filter ─────────────────────────────────────────────────────
cs  = pd.Timestamp(contract_start)
ce  = cs + pd.DateOffset(years=1)
now = pd.Timestamp(date.today())

in_year     = (df["Date and Time Contact"] >= cs) & (df["Date and Time Contact"] < ce)
chatbot_df  = df.loc[in_year & (df["_source"] == "chatbot")].copy()
resolved_df = chatbot_df.loc[chatbot_df["_resolved"] == "resolved"].copy()
handover_df = chatbot_df.loc[chatbot_df["_handover"] == "Y"].copy()


# ─── Monthly counts ───────────────────────────────────────────────────────────
monthly = chatbot_df.resample("MS", on="Date and Time Contact").agg(
    chatbot_total=("Ticket ID",  "count"),
    ai_resolved  =("_resolved",  lambda x: (x == "resolved").sum()),
    handovers    =("_handover",  lambda x: (x == "Y").sum()),
)

if monthly.empty:
    st.warning(
        "No chatbot tickets found in the selected contract year. "
        "Try adjusting the contract start date in the sidebar."
    )
    st.stop()

monthly["ai_rate_pct"] = (
    monthly["ai_resolved"] / monthly["chatbot_total"] * 100
).round(1)
labels = monthly.index.strftime("%b %y").tolist()


# ─── Contract fee calculations ────────────────────────────────────────────────
months_elapsed     = max(len(monthly), 1)
gross_contract_fee = (cost_per_seat * num_agents * 12) + annual_ai_fee
discount_amount    = gross_contract_fee * (vendor_discount_pct / 100)
net_contract_fee   = gross_contract_fee - discount_amount   # annual discounted
contract_fee_ytd   = net_contract_fee * (months_elapsed / 12)  # pro-rated

agent_capacity_per_month = agent_tickets_per_day * working_days_per_month

# ─── Running cost model ───────────────────────────────────────────────────────
cum_resolved = 0
cost_rows = []

for idx, row in monthly.iterrows():
    res = int(row["ai_resolved"])
    ho  = int(row["handovers"])

    within   = min(res, max(allowance - cum_resolved, 0))
    overage  = res - within
    cum_resolved += res

    ai_spend    = within * cost_ai_contract + overage * cost_ai_overage
    human_spend = ho * cost_human
    month_total = ai_spend + human_spend
    payg_spend  = res * cost_ai_overage + ho * cost_human   # no contract, pay per resolution
    all_human   = (res + ho) * cost_human

    cost_rows.append({
        "month":        idx,
        "ai_resolved":  res,
        "handovers":    ho,
        "within":       within,
        "overage_res":  overage,
        "cum_resolved": cum_resolved,
        "ai_spend":     ai_spend,
        "human_spend":  human_spend,
        "month_total":  month_total,
        "payg_spend":   payg_spend,
        "all_human":    all_human,
        "saving":       all_human - month_total,
    })

cost_df = pd.DataFrame(cost_rows).set_index("month")
cost_df["cum_total"]    = cost_df["month_total"].cumsum()
cost_df["cum_allhuman"] = cost_df["all_human"].cumsum()
cost_df["cum_payg"]     = cost_df["payg_spend"].cumsum()
cost_df["cum_saving"]   = cost_df["saving"].cumsum()

total_resolved   = int(cost_df["ai_resolved"].sum())
total_handovers  = int(cost_df["handovers"].sum())
total_chatbot    = int(monthly["chatbot_total"].sum())
chatbot_only     = total_chatbot - total_handovers
ai_rate_overall  = total_resolved / total_chatbot * 100 if total_chatbot else 0
pct_of_allowance = total_resolved / allowance * 100
days_into_year   = max((min(now, ce) - cs).days, 0)

# Total spend = pro-rated contract fee + usage costs
ytd_usage   = cost_df["cum_total"].iloc[-1]
ytd_total   = contract_fee_ytd + ytd_usage
ytd_allhuman = cost_df["cum_allhuman"].iloc[-1]
ytd_payg    = cost_df["cum_payg"].iloc[-1]   # PAYG: no contract fee, just per-ticket
ytd_saving  = ytd_allhuman - ytd_total

# Agents needed to replace AI entirely
total_tickets_per_month_avg = total_chatbot / months_elapsed
agents_needed_no_ai    = int(np.ceil(total_tickets_per_month_avg / agent_capacity_per_month))
cost_all_agents_ytd    = agents_needed_no_ai * agent_monthly_salary * months_elapsed
one_agent_monthly_cost = agent_monthly_salary


# ─── Forecast ─────────────────────────────────────────────────────────────────
last3_avg  = monthly["ai_resolved"].tail(3).mean()
last3_std  = monthly["ai_resolved"].tail(3).std()
last3_high = last3_avg + last3_std

remaining_months = max(
    int(np.ceil((ce - monthly.index[-1]).days / 30)) - 1, 1
)

projected_total        = total_resolved + last3_avg  * remaining_months
projected_total_high   = total_resolved + last3_high * remaining_months
projected_overage      = max(projected_total      - allowance, 0)
projected_overage_high = max(projected_total_high - allowance, 0)
overage_cost_normal    = projected_overage      * cost_ai_overage
overage_cost_worst     = projected_overage_high * cost_ai_overage
headcount_cost         = projected_overage      * cost_human
safe_pace              = (
    max(allowance - total_resolved, 0) / remaining_months
    if remaining_months else 0
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"**Contract:** {cs.strftime('%d %b %Y')} to {ce.strftime('%d %b %Y')}"
    f"  |  **Annual limit:** {allowance:,} AI-resolved tickets"
    f"  |  **Today:** {now.strftime('%d %b %Y')}",
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1 — At a glance
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("At a glance")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("AI tickets used",        f"{total_resolved:,}",
          help="Tickets the chatbot fully resolved — these count against the contract limit.")
k2.metric("Remaining in contract",  f"{max(allowance - total_resolved, 0):,}",
          help="How many more AI resolutions are covered before overage kicks in.")
k3.metric("Contract used",          f"{pct_of_allowance:.1f}%",
          help=f"{total_resolved:,} out of {allowance:,} allowed resolutions.")
k4.metric("Bot success rate",       f"{ai_rate_overall:.1f}%",
          help="Out of every 100 tickets the chatbot touched, it fully resolved this many without a human.")
k5.metric("Escalated to humans",    f"{total_handovers:,}",
          help="Tickets the chatbot could not handle and passed to a human agent.")

bar_pct   = min(pct_of_allowance, 100)
bar_color = (
    "#C0392B" if pct_of_allowance > 95 else
    "#E8873A" if pct_of_allowance > 75 else
    "#2E9E6B"
)
status = (
    "🔴 OVER THE LIMIT"                    if total_resolved > allowance else
    "🟠 Getting close — keep an eye on this" if pct_of_allowance > 75    else
    "🟢 On track"
)

st.markdown(f"""
<div style="background:#eee;border-radius:8px;height:18px;margin:12px 0 4px">
  <div style="background:{bar_color};width:{bar_pct:.1f}%;height:18px;border-radius:8px"></div>
</div>
<p style="font-size:.8rem;color:#555;margin:0">
  {status} &nbsp;·&nbsp;
  {pct_of_allowance:.1f}% of the {allowance:,} annual limit used &nbsp;·&nbsp;
  {days_into_year} days into the contract year
</p>
""", unsafe_allow_html=True)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2 — Will we hit the limit?
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Will we hit the limit before the contract ends?")
st.caption(
    f"Your 3-month average is **{last3_avg:,.0f} AI resolutions/month**. "
    f"Use the slider below to stress-test what happens if volume goes up "
    f"(seasonal spike, big promo, new brand launch) or comes down."
)

# ── Volume adjustment slider ──────────────────────────────────────────────────
volume_adjustment = st.slider(
    f"Adjust forecast: what if monthly AI resolutions go up or down from your recent average ({int(last3_avg):,}/month)?",
    min_value=-50,
    max_value=100,
    value=0,
    step=5,
    format="%d%%",
    help=(
        f"Your recent 3-month average is {int(last3_avg):,} AI resolutions per month. "
        f"This slider adjusts that number up or down to simulate future scenarios. "
        f"Example: +30% means you expect 30% MORE chatbot-resolved tickets per month "
        f"than your recent average — e.g. during a holiday campaign or new brand launch. "
        f"-20% means fewer resolutions, e.g. a slower season. "
        f"Only affects the forecast. Your actual historical data stays unchanged."
    ),
)

# Adjusted forecast based on slider
adjusted_monthly    = last3_avg * (1 + volume_adjustment / 100)
projected_total     = total_resolved + adjusted_monthly * remaining_months
projected_overage   = max(projected_total   - allowance, 0)
overage_cost_normal = projected_overage     * cost_ai_overage
headcount_cost      = projected_overage     * cost_human

# Busier band stays as last3_avg + 1 std regardless of slider
projected_total_high   = total_resolved + last3_high * (1 + volume_adjustment / 100) * remaining_months
projected_overage_high = max(projected_total_high - allowance, 0)
overage_cost_worst     = projected_overage_high * cost_ai_overage

# Show what the slider means in plain numbers
if volume_adjustment != 0:
    direction = "more" if volume_adjustment > 0 else "fewer"
    diff      = abs(int(adjusted_monthly - last3_avg))
    st.caption(
        f"At **{volume_adjustment:+d}%** — forecast uses **{int(adjusted_monthly):,} AI resolutions/month** "
        f"({diff:,} {direction} chatbot-resolved tickets per month than your recent average of {int(last3_avg):,})."
    )

# Result cards
fc_l, fc_r = st.columns(2)

with fc_l:
    label = "At current pace" if volume_adjustment == 0 else f"At {volume_adjustment:+d}% volume"
    st.markdown(f"##### {label}")
    if projected_overage > 0:
        st.error(
            f"Projected year-end total: **{int(projected_total):,}** AI resolutions\n\n"
            f"That is **{int(projected_overage):,} over the limit**.\n\n"
            f"Estimated overage charge: **${overage_cost_normal:,.0f}**"
        )
    else:
        st.success(
            f"Projected year-end total: **{int(projected_total):,}** AI resolutions\n\n"
            f"Safely within the {allowance:,} limit. No overage expected."
        )

with fc_r:
    st.markdown("##### Worst-case (if it gets even busier on top of that)")
    if projected_overage_high > 0:
        st.warning(
            f"Could reach **{int(projected_total_high):,}** resolutions — "
            f"**{int(projected_overage_high):,} over the limit**.\n\n"
            f"Worst-case overage charge: **${overage_cost_worst:,.0f}**"
        )
    else:
        st.success(
            f"Even in a worst-case scenario: **{int(projected_total_high):,}** "
            f"— still within the limit."
        )

# Recalculate safe pace based on adjusted forecast
safe_pace = (
    max(allowance - total_resolved, 0) / remaining_months
    if remaining_months else 0
)
st.info(
    f"**Safe monthly pace:** {max(allowance - total_resolved, 0):,} resolutions left "
    f"over {remaining_months} month(s) → stay below **{safe_pace:,.0f}/month** to avoid overage. "
    f"Your adjusted forecast is **{int(adjusted_monthly):,}/month** — "
    + (
        "within the safe zone."
        if adjusted_monthly <= safe_pace
        else "above the safe zone. Overage is likely at this volume."
    )
)

# Cumulative chart — updates live with the slider
future_idx     = pd.date_range(
    start=monthly.index[-1] + pd.DateOffset(months=1),
    periods=remaining_months, freq="MS",
)
fc_series      = pd.Series([adjusted_monthly] * remaining_months, index=future_idx)
fc_series_high = pd.Series([last3_high * (1 + volume_adjustment / 100)] * remaining_months, index=future_idx)
hist_cum       = monthly["ai_resolved"].cumsum()
fc_cum         = pd.concat([monthly["ai_resolved"], fc_series]).cumsum()
fc_cum_high    = pd.concat([monthly["ai_resolved"], fc_series_high]).cumsum()
hist_len       = len(hist_cum)
all_labels     = fc_cum.index.strftime("%b %y").tolist()

fig_fc, ax_fc = plt.subplots(figsize=(11, 4))
ax_fc.plot(range(hist_len), hist_cum.values,
           marker="o", markersize=5, linewidth=2.5,
           color="#1B4F8A", label="Actual AI resolutions (running total)")
if remaining_months > 0:
    fc_x = list(range(hist_len - 1, len(fc_cum)))
    slider_label = (
        "Forecast — normal pace" if volume_adjustment == 0
        else f"Forecast — {volume_adjustment:+d}% volume adjustment"
    )
    ax_fc.plot(fc_x, fc_cum.values[hist_len - 1:],
               linestyle="--", linewidth=2, color="#E8873A",
               label=slider_label)
    ax_fc.fill_between(
        fc_x,
        fc_cum.values[hist_len - 1:],
        fc_cum_high.values[hist_len - 1:],
        alpha=0.12, color="#E8873A", label="Worst-case band",
    )
ax_fc.axhline(allowance, color="#C0392B", linewidth=1.5,
              linestyle="-.", label=f"Contract limit ({allowance:,})")
ax_fc.set_xticks(range(len(all_labels)))
ax_fc.set_xticklabels(all_labels, rotation=40, ha="right")
ax_fc.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_fc.set_ylabel("Running total of AI resolutions")
ax_fc.legend(fontsize=8.5, loc="upper left")
plt.tight_layout()
st.pyplot(fig_fc)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3 — Cost tracker
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("💰 What does this actually cost?")
st.caption(
    "Three scenarios compared: your actual contract spend, what pay-as-you-go "
    "would have cost, and what full human agent coverage would have cost."
)

# ── Contract fee breakdown ────────────────────────────────────────────────────
with st.expander("See contract fee breakdown"):
    cf1, cf2, cf3, cf4 = st.columns(4)
    cf1.metric("Gross contract fee (annual)",   f"${gross_contract_fee:,.0f}",
               help=f"({num_agents} agents × ${cost_per_seat:.0f}/month × 12) + ${annual_ai_fee:,.0f} AI fee")
    cf2.metric(f"Vendor discount ({vendor_discount_pct:.0f}%)", f"-${discount_amount:,.0f}")
    cf3.metric("Net contract fee (annual)",     f"${net_contract_fee:,.0f}")
    cf4.metric(f"Pro-rated YTD ({months_elapsed} months)", f"${contract_fee_ytd:,.0f}")

# ── Top-line cost comparison ──────────────────────────────────────────────────
st.markdown("#### YTD cost comparison")
m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Actual total spend",
    f"${ytd_total:,.0f}",
    help="Contract fee (pro-rated) + AI usage costs + human handover costs.",
)
m2.metric(
    "If pay-as-you-go (no contract)",
    f"${ytd_payg:,.0f}",
    delta=f"${ytd_payg - ytd_total:+,.0f} vs contract",
    delta_color="inverse" if ytd_payg > ytd_total else "normal",
    help=f"No contract fee. Every AI resolution at ${cost_ai_overage:.2f} (PAYG rate).",
)
m3.metric(
    "If all human agents",
    f"${ytd_allhuman:,.0f}",
    delta=f"${ytd_allhuman - ytd_total:+,.0f} vs contract",
    delta_color="inverse" if ytd_allhuman > ytd_total else "normal",
    help=f"Every chatbot-touched ticket handled by a human at ${cost_human:.2f}.",
)
m4.metric(
    "Agents needed to replace AI",
    f"{agents_needed_no_ai}",
    help=(
        f"If you removed AI entirely, you would need {agents_needed_no_ai} human agents "
        f"just to cover the same chatbot ticket volume. "
        f"Based on: {total_tickets_per_month_avg:,.0f} avg chatbot tickets/month "
        f"÷ {agent_capacity_per_month} tickets/agent/month "
        f"({agent_tickets_per_day} tickets/day × {working_days_per_month} working days). "
        f"At ${agent_monthly_salary:,.0f}/month per agent, "
        f"that's ${agents_needed_no_ai * agent_monthly_salary:,.0f}/month in salary alone — "
        f"vs your current AI contract cost of ~${ytd_total/months_elapsed:,.0f}/month."
    ),
)

st.caption(
    f"ℹ️ **'Agents needed to replace AI' explained:** "
    f"Your chatbot touches ~{total_tickets_per_month_avg:,.0f} tickets/month on average. "
    f"One agent handles {agent_capacity_per_month:,} tickets/month "
    f"({agent_tickets_per_day}/day × {working_days_per_month} days). "
    f"{total_tickets_per_month_avg:,.0f} ÷ {agent_capacity_per_month} = **{agents_needed_no_ai} agents** needed to cover the same volume with no AI. "
    f"At ${agent_monthly_salary:,.0f}/month each → **${agents_needed_no_ai * agent_monthly_salary:,.0f}/month** in salary, "
    f"vs your current all-in AI cost of ~**${ytd_total/months_elapsed:,.0f}/month**."
)

# ── Monthly cost chart ────────────────────────────────────────────────────────
fig_cost, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
x = np.arange(len(cost_df))

ax1.bar(x, cost_df["within"] * cost_ai_contract,
        label=f"AI within limit (${cost_ai_contract:.2f})", color="#2E9E6B")
ax1.bar(x, cost_df["overage_res"] * cost_ai_overage,
        bottom=cost_df["within"] * cost_ai_contract,
        label=f"AI over limit (${cost_ai_overage:.2f})", color="#C0392B")
ax1.bar(x, cost_df["human_spend"],
        bottom=(cost_df["within"] * cost_ai_contract + cost_df["overage_res"] * cost_ai_overage),
        label=f"Human agent (${cost_human:.2f})", color="#E8873A")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=40, ha="right")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.set_ylabel("Monthly usage cost ($)")
ax1.set_title("Monthly usage cost breakdown")
ax1.legend(fontsize=8)

ax2.plot(range(len(cost_df)), cost_df["cum_allhuman"].values,
         linestyle="--", linewidth=1.8, color="#C0392B", label="All human agents")
ax2.plot(range(len(cost_df)), cost_df["cum_payg"].values,
         linestyle="-.", linewidth=1.8, color="#8E44AD", label="Pay-as-you-go (no contract)")
ax2.plot(range(len(cost_df)), cost_df["cum_total"].values,
         marker="o", markersize=4, linewidth=2.5, color="#1B4F8A",
         label="Actual usage cost (with contract)")
ax2.set_xticks(range(len(cost_df)))
ax2.set_xticklabels(labels, rotation=40, ha="right")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.set_ylabel("Cumulative usage cost ($)")
ax2.set_title("3-way cost comparison (usage only, excl. contract fee)")
ax2.legend(fontsize=8.5)
plt.tight_layout()
st.pyplot(fig_cost)

st.caption(
    "Note: \"Actual total spend\" includes the pro-rated contract fee. "
    "The chart above shows usage costs only so the three scenarios are comparable."
)

# ── Add headcount vs overage decision ────────────────────────────────────────
st.markdown("#### When does adding a human agent make more sense than paying overage?")

projected_monthly_overage_cost = max(adjusted_monthly - safe_pace, 0) * cost_ai_overage if safe_pace > 0 else 0

hc1, hc2, hc3 = st.columns(3)
hc1.metric(
    "1 extra agent costs",
    f"${one_agent_monthly_cost:,.0f}/month",
    help="Monthly salary input from the sidebar.",
)
hc2.metric(
    "Projected overage cost/month",
    f"${projected_monthly_overage_cost:,.0f}/month",
    help="Based on your current volume vs. safe monthly pace.",
)
hc3.metric(
    "Agent handles per month",
    f"{agent_capacity_per_month:,} tickets",
    help=f"{agent_tickets_per_day} tickets/day × {working_days_per_month} days.",
)

if projected_monthly_overage_cost > 0:
    if one_agent_monthly_cost <= projected_monthly_overage_cost:
        st.success(
            f"Adding 1 human agent (${one_agent_monthly_cost:,.0f}/month) is **cheaper** than "
            f"the projected overage (${projected_monthly_overage_cost:,.0f}/month). "
            f"That agent can absorb ~{agent_capacity_per_month:,} tickets/month, "
            f"reducing AI resolution volume and keeping you within the contract limit."
        )
    else:
        st.info(
            f"Paying the overage (${projected_monthly_overage_cost:,.0f}/month) is still **cheaper** "
            f"than adding a human agent (${one_agent_monthly_cost:,.0f}/month). "
            f"If volume keeps growing, revisit this — the crossover point is when "
            f"overage costs exceed ${one_agent_monthly_cost:,.0f}/month."
        )
else:
    st.success("Volume is within the safe zone — no overage or headcount decision needed right now.")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4 — Monthly bot performance
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("How is the bot performing each month?")
st.caption(
    "Bot success rate: out of every ticket the chatbot touched, "
    "how many did it fully resolve without a human? "
    "Higher is better. A flat line means consistent performance."
)

fig_perf, (ax_rate, ax_vol) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

ax_rate.fill_between(labels, monthly["ai_rate_pct"].values, alpha=0.1, color="#1B4F8A")
ax_rate.plot(labels, monthly["ai_rate_pct"].values,
             marker="o", markersize=5, linewidth=2.2, color="#1B4F8A",
             label="Bot success rate")
avg_rate = monthly["ai_rate_pct"].mean()
ax_rate.axhline(avg_rate, linestyle="--", linewidth=1,
                color="#1B4F8A", alpha=0.5, label=f"Average: {avg_rate:.1f}%")
ax_rate.set_ylim(0, 60)
ax_rate.yaxis.set_major_formatter(mticker.PercentFormatter())
ax_rate.set_ylabel("Bot success rate")
ax_rate.legend(fontsize=8.5)
ax_rate.set_title("Bot success rate per month (% of chatbot tickets resolved without a human)")

ax_vol.bar(labels, monthly["ai_resolved"].values,
           label="Bot resolved", color="#2E9E6B")
ax_vol.bar(labels, monthly["handovers"].values,
           bottom=monthly["ai_resolved"].values,
           label="Passed to human agent", color="#E8873A")
ax_vol.set_ylabel("Number of tickets")
ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_vol.legend(fontsize=8.5)
ax_vol.set_title("Monthly volume: bot resolved vs. passed to human")
plt.xticks(rotation=40, ha="right")
plt.tight_layout()
st.pyplot(fig_perf)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5 — Brand breakdown
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Which brands are using up the contract?")
st.caption(
    "Every AI-resolved ticket counts against the annual limit, "
    "regardless of brand. The funnel shows how tickets flow through "
    "the chatbot for each brand."
)

brand_res     = (resolved_df.groupby("Brand").size()
                 .rename("Bot resolved").sort_values(ascending=False))
brand_ho      = handover_df.groupby("Brand").size().rename("Passed to human")
brand_tot     = chatbot_df.groupby("Brand").size().rename("Total chatbot tickets")
brand_cb_only = (chatbot_df[chatbot_df["_handover"] != "Y"]
                 .groupby("Brand").size().rename("Chatbot only (no escalation)"))

brand_tbl = pd.concat(
    [brand_res, brand_ho, brand_tot, brand_cb_only], axis=1
).fillna(0).astype(int)
brand_tbl["Bot success rate"]   = (
    brand_tbl["Bot resolved"] / brand_tbl["Total chatbot tickets"] * 100
).round(1)
brand_tbl["% of contract used"] = (
    brand_tbl["Bot resolved"] / allowance * 100
).round(1)
brand_tbl = brand_tbl.sort_values("Bot resolved", ascending=False)


# ── Left: Pie — share of AI resolutions ──────────────────────────────────────
pie_col, funnel_col = st.columns([1, 1])

with pie_col:
    st.markdown("##### Share of AI resolutions by brand")
    fig_p1, ax_p1 = plt.subplots(figsize=(5, 5))
    ax_p1.pie(
        brand_res.values, labels=brand_res.index,
        autopct="%1.1f%%", startangle=140,
        colors=PALETTE[:len(brand_res)], pctdistance=0.78,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    ax_p1.axis("equal")
    plt.tight_layout()
    st.pyplot(fig_p1)


# ── Right: Funnel — ticket flow ───────────────────────────────────────────────
with funnel_col:
    st.markdown("##### How tickets flow through the chatbot")
    st.caption(
        "Starting from every ticket the chatbot touched, down to what it actually resolved."
    )

    # Four funnel levels
    # Level 2: whichever is higher (chatbot_only vs handovers) goes on the LEFT
    if chatbot_only >= total_handovers:
        level2_left_label  = "Handled only by bot"
        level2_left_val    = chatbot_only
        level2_right_label = "Passed to human agent"
        level2_right_val   = total_handovers
    else:
        level2_left_label  = "Passed to human agent"
        level2_left_val    = total_handovers
        level2_right_label = "Handled only by bot"
        level2_right_val   = chatbot_only

    funnel_data = [
        ("Total chatbot interactions",  total_chatbot,   "#1B4F8A"),
        (f"{level2_left_label}  |  {level2_right_label}",
         None,  None),          # split row — handled separately
        ("Bot resolved",               total_resolved,  "#2E9E6B"),
    ]

    # Build funnel as a matplotlib figure
    fig_funnel, ax_f = plt.subplots(figsize=(6, 5))
    ax_f.axis("off")

    max_val    = total_chatbot
    bar_height = 0.55
    gap        = 0.18
    y_positions= [0.85, 0.52, 0.18]   # top, middle, bottom

    # ---- Row 1: Total chatbot interactions ----
    w1 = 0.80
    x1 = (1 - w1) / 2
    ax_f.add_patch(mpatches.FancyBboxPatch(
        (x1, y_positions[0]), w1, bar_height * 0.55,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#1B4F8A", transform=ax_f.transAxes, zorder=2,
    ))
    ax_f.text(0.5, y_positions[0] + bar_height * 0.55 / 2,
              f"Total chatbot interactions\n{total_chatbot:,}",
              ha="center", va="center", fontsize=10, fontweight="bold",
              color="white", transform=ax_f.transAxes, zorder=3)

    # Arrow down
    ax_f.annotate("", xy=(0.5, y_positions[1] + bar_height * 0.55 + 0.005),
                  xytext=(0.5, y_positions[0] - 0.01),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.5))

    # ---- Row 2: Split bar (chatbot only | handovers) ----
    left_frac  = level2_left_val  / max_val * 0.80
    right_frac = level2_right_val / max_val * 0.80
    gap_frac   = 0.80 - left_frac - right_frac   # tiny gap between halves
    x2_left    = (1 - 0.80) / 2
    x2_right   = x2_left + left_frac + gap_frac / 2

    # Left half
    ax_f.add_patch(mpatches.FancyBboxPatch(
        (x2_left, y_positions[1]), left_frac, bar_height * 0.55,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#E8873A", transform=ax_f.transAxes, zorder=2,
    ))
    ax_f.text(x2_left + left_frac / 2, y_positions[1] + bar_height * 0.55 / 2,
              f"{level2_left_label}\n{level2_left_val:,}",
              ha="center", va="center", fontsize=8.5, fontweight="bold",
              color="white", transform=ax_f.transAxes, zorder=3)

    # Right half
    ax_f.add_patch(mpatches.FancyBboxPatch(
        (x2_right, y_positions[1]), right_frac, bar_height * 0.55,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#8E44AD", transform=ax_f.transAxes, zorder=2,
    ))
    ax_f.text(x2_right + right_frac / 2, y_positions[1] + bar_height * 0.55 / 2,
              f"{level2_right_label}\n{level2_right_val:,}",
              ha="center", va="center", fontsize=8.5, fontweight="bold",
              color="white", transform=ax_f.transAxes, zorder=3)

    # Arrow down
    ax_f.annotate("", xy=(0.5, y_positions[2] + bar_height * 0.55 + 0.005),
                  xytext=(0.5, y_positions[1] - 0.01),
                  xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.5))

    # ---- Row 3: Bot resolved ----
    w3 = total_resolved / max_val * 0.80
    x3 = (1 - w3) / 2
    ax_f.add_patch(mpatches.FancyBboxPatch(
        (x3, y_positions[2]), w3, bar_height * 0.55,
        boxstyle="round,pad=0.01", linewidth=0,
        facecolor="#2E9E6B", transform=ax_f.transAxes, zorder=2,
    ))
    ax_f.text(0.5, y_positions[2] + bar_height * 0.55 / 2,
              f"Resolved by chatbot\n{total_resolved:,}  ({total_resolved/total_chatbot*100:.1f}% of all chatbot tickets)",
              ha="center", va="center", fontsize=9, fontweight="bold",
              color="white", transform=ax_f.transAxes, zorder=3)

    plt.tight_layout()
    st.pyplot(fig_funnel)


# ── Grouped bar: resolved vs handovers per brand ──────────────────────────────
fig_bb, ax_bb = plt.subplots(figsize=(11, 4))
bx = np.arange(len(brand_tbl))
bw = 0.38
ax_bb.bar(bx - bw / 2, brand_tbl["Bot resolved"].values,
          width=bw, label="Bot resolved", color="#1B4F8A")
ax_bb.bar(bx + bw / 2, brand_tbl["Passed to human"].values,
          width=bw, label="Passed to human agent", color="#E8873A")
ax_bb.set_xticks(bx)
ax_bb.set_xticklabels(brand_tbl.index, rotation=30, ha="right")
ax_bb.set_ylabel("Tickets")
ax_bb.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_bb.legend(fontsize=9)
ax_bb.set_title("Bot resolved vs. passed to human — per brand")
plt.tight_layout()
st.pyplot(fig_bb)

# ── Bot success rate per brand ────────────────────────────────────────────────
st.markdown("##### Bot success rate per brand")
st.caption("Green = strong, orange = okay, red = needs attention.")
rate_sorted = brand_tbl["Bot success rate"].sort_values()
bar_colors  = [
    "#2E9E6B" if v >= 35 else "#E8873A" if v >= 25 else "#C0392B"
    for v in rate_sorted.values
]
fig_br, ax_br = plt.subplots(figsize=(8, 3.5))
ax_br.barh(rate_sorted.index, rate_sorted.values, color=bar_colors)
ax_br.axvline(ai_rate_overall, linestyle="--", linewidth=1.2,
              color="#1B4F8A", label=f"Overall average: {ai_rate_overall:.1f}%")
ax_br.xaxis.set_major_formatter(mticker.PercentFormatter())
ax_br.set_xlabel("Bot success rate (%)")
ax_br.legend(fontsize=8.5)
plt.tight_layout()
st.pyplot(fig_br)

with st.expander("See full brand numbers"):
    st.dataframe(
        brand_tbl.style
        .format({
            "Bot resolved":             "{:,}",
            "Passed to human":          "{:,}",
            "Total chatbot tickets":    "{:,}",
            "Chatbot only (no escalation)": "{:,}",
            "Bot success rate":         "{:.1f}%",
            "% of contract used":       "{:.1f}%",
        })
        .background_gradient(subset=["% of contract used"], cmap="Oranges")
        .background_gradient(subset=["Bot success rate"],   cmap="Greens"),
        use_container_width=True,
    )
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6 — Issue types
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("What kinds of issues does the bot handle well?")
st.caption(
    "Where the chatbot is genuinely saving money versus "
    "where it mostly just escalates to a human anyway."
)

issue_res = resolved_df.groupby("Issue").size().rename("Bot resolved")
issue_ho  = handover_df.groupby("Issue").size().rename("Passed to human")
issue_tot = chatbot_df.groupby("Issue").size().rename("Total")
issue_tbl = pd.concat([issue_res, issue_ho, issue_tot], axis=1).fillna(0).astype(int)
issue_tbl["Bot success rate"] = (
    issue_tbl["Bot resolved"] / issue_tbl["Total"] * 100
).round(1)
issue_tbl["Escalation rate"]  = (
    issue_tbl["Passed to human"] / issue_tbl["Total"] * 100
).round(1)
issue_tbl = issue_tbl.sort_values("Bot resolved", ascending=False)

fig_iss, ax_iss = plt.subplots(figsize=(11, 4))
ix  = np.arange(len(issue_tbl))
bw3 = 0.38
ax_iss.bar(ix - bw3 / 2, issue_tbl["Bot resolved"].values,
           width=bw3, label="Bot resolved", color="#1B4F8A")
ax_iss.bar(ix + bw3 / 2, issue_tbl["Passed to human"].values,
           width=bw3, label="Passed to human agent", color="#E8873A")
ax_iss.set_xticks(ix)
ax_iss.set_xticklabels(issue_tbl.index, rotation=30, ha="right")
ax_iss.set_ylabel("Tickets")
ax_iss.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax_iss.legend(fontsize=9)
ax_iss.set_title("Bot resolved vs. escalated — by issue type")
plt.tight_layout()
st.pyplot(fig_iss)

with st.expander("See full issue breakdown"):
    st.dataframe(
        issue_tbl.style
        .format({
            "Bot resolved":    "{:,}",
            "Passed to human": "{:,}",
            "Total":           "{:,}",
            "Bot success rate":"{:.1f}%",
            "Escalation rate": "{:.1f}%",
        })
        .background_gradient(subset=["Bot success rate"], cmap="Greens")
        .background_gradient(subset=["Escalation rate"],  cmap="Reds"),
        use_container_width=True,
    )
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7 — Raw monthly numbers
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("Show raw monthly numbers"):
    display = pd.DataFrame({
        "Month":                   monthly.index.strftime("%b %Y"),
        "Total chatbot tickets":   monthly["chatbot_total"].astype(int),
        "Bot resolved":            monthly["ai_resolved"].astype(int),
        "Passed to human":         monthly["handovers"].astype(int),
        "Bot success rate":        monthly["ai_rate_pct"],
        "Running total resolved":  monthly["ai_resolved"].cumsum().astype(int),
    }).set_index("Month")

    st.dataframe(
        display.style.format({
            "Total chatbot tickets":  "{:,}",
            "Bot resolved":           "{:,}",
            "Passed to human":        "{:,}",
            "Bot success rate":       "{:.1f}%",
            "Running total resolved": "{:,}",
        }),
        use_container_width=True,
    )
