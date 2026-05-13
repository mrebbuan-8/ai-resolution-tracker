# 🤖 AI Resolution Tracker

A Streamlit dashboard for AI Product Owners and Support Ops teams to track chatbot resolution consumption against a contract allowance, forecast overages, and make cost-informed decisions — without a spreadsheet.

Sample dataset for testing: [download here](https://drive.google.com/file/d/1vHv5UK5E4cIOHSdF26X304OjgV3sToOB/view?usp=drive_link)

**Stack:** Anthropic (Claude) · Python · Streamlit · Pandas · Matplotlib · NumPy

---

## The Problem

Most CRM platforms show you how many AI resolutions you've used and how many are left. What they don't tell you:

- Will we exceed the contract allowance before the year ends?
- If we go over — is it cheaper to pay the overage or add a human agent?
- Which brands and issue types are consuming the allowance the fastest?
- Is the AI contract actually saving money vs. pure pay-as-you-go or full human coverage?

This dashboard answers all of that — from a CSV export.

---

## Who It's For

**Primary:** AI Product Owner — tracking contract health, forecasting risk, justifying AI investment

**Secondary:** Support Ops Manager — understanding bot performance trends and brand/issue breakdowns

---

## Features

### At a glance
Live snapshot: resolutions used, remaining allowance, % consumed, bot success rate, escalations — with a colour-coded progress bar.

### Forecast with volume slider
Projects year-end totals using a 3-month rolling average. A **±% slider** lets you stress-test seasonal spikes, promos, or slow periods — chart and overage estimates update instantly. Shows both a base forecast and a worst-case band.

### Cost tracker — 3-way comparison
Compares your actual contract spend against:
- **Pure pay-as-you-go** (no commitment, PAYG rate per resolution)
- **Full human agent coverage** (no AI at all)

Includes a contract fee breakdown (seat fees + AI platform fee − vendor discount) and a headcount decision helper: *at what point does adding a human agent beat paying overage?*

### Monthly bot performance
Bot success rate trend + monthly volume split (resolved vs. escalated), on a shared time axis.

### Brand breakdown
Pie chart of allowance consumption by brand, grouped bar chart (resolved vs. escalated per brand), bot success rate per brand (colour-coded), and a ticket flow funnel.

### Issue type breakdown
Which issue types the bot handles well vs. consistently escalates — so you know where to focus automation improvements.

---

## Forecasting Method

Monthly AI resolutions on this dataset have a coefficient of variation of ~4% — essentially flat with no meaningful trend. A linear regression returns R² ≈ 0.003 (p = 0.875), meaning there is no statistically significant trend to fit.

The forecast uses a **3-month rolling average**, which is both more accurate and more honest for stable, stationary data. The volume slider lets you layer in your own business context (campaigns, seasonality) on top of the baseline forecast.

*Next iteration: replacing the manual slider with Facebook Prophet for automatic seasonality detection.*

---

## Pricing Model

Pre-filled with publicly available Zendesk pricing (2025). Fully adjustable in the sidebar — works for any AI support vendor.

| Tier | Default rate | When it applies |
|---|---|---|
| Within contract allowance | $0.50 / resolution | Up to your committed bundle |
| Over the limit (PAYG) | $2.00 / resolution | After the allowance is exhausted |
| Human agent escalation | $4.00 / ticket | Bot couldn't resolve, passed to human |

> Pricing based on publicly available Zendesk documentation as of 2025. Actual rates vary by contract — adjust sidebar inputs to match your specific terms.

**What counts as a billable AI resolution (Zendesk):** A ticket fully handled by the AI agent from start to finish, no human intervention, followed by 72 hours of customer inactivity. Escalated tickets are not charged.

---

## CSV Requirements

Your support data export needs these column names:

| Column | Expected value |
|---|---|
| `Date and Time Contact` | Date/time of the interaction |
| `Source` | `chatbot` for bot-handled tickets |
| `Chatbot Resolved` | `Resolved` for successful AI resolutions |
| `Handover (Y/N)` | `Y` for escalated tickets |
| `Brand` | Brand name |
| `Issue` | Issue type / category |
| `Ticket ID` | Any unique ticket identifier |

---

## How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ai-resolution-tracker.git
cd ai-resolution-tracker

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run ai_resolution_forecast_app.py
```

Upload your CSV using the file uploader on the page. No file path configuration needed.

---

## Deploy for Free (Shareable URL)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. **New app** → select this repo → main file: `ai_resolution_forecast_app.py`
4. **Deploy** — you get a public URL to share with your team

---

## Sidebar Settings

| Setting | Default | Notes |
|---|---|---|
| Contract start date | Jan 1 2025 | Resets the 12-month window |
| Annual allowance | 50,000 | Your committed resolution bundle |
| Annual AI platform fee | $10,000 | Fixed fee on top of per-resolution costs |
| Vendor discount | 10% | Applied to contract fee only |
| Within allowance rate | $0.50 | Per-resolution cost inside the bundle |
| PAYG overage rate | $2.00 | Per-resolution cost above the bundle |

---

## Background

In a previous support ops role, tracking AI chatbot resolution consumption meant a daily spreadsheet and a lot of gut feel. The CRM (Zendesk) showed current usage, but nothing about whether we'd run out before the contract year ended, or what the right call was when we got close.

This dashboard was built to answer those questions automatically from the same data the CRM already collects.

Built as a side project. Tested on synthetic support data. Real-world performance will vary with seasonality and volume patterns — which is exactly what the slider is for.

---

*Built with Python + Streamlit. No external APIs. No data leaves your machine.*
