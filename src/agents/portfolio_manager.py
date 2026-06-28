from langchain_core.messages import HumanMessage
from config.analysis_weights import (
    TECHNICAL_ANALYSIS_WEIGHT,
    SENTIMENT_ANALYSIS_WEIGHT,
)

from agents.state import AgentState, show_agent_reasoning
from groq import Groq
import os
import json

GROQ_MODEL = "llama-3.1-8b-instant"
_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Minimum trend magnitude to override LLM direction.
TREND_OVERRIDE_THRESHOLD = 2.0

# Regime detection: 7-day absolute change < this % → market is RANGING
REGIME_THRESHOLD_PCT = 3.0

# Mean reversion: 168 hourly candles = 7 days of data
MEAN_REV_LOOKBACK = 168

# Bollinger Band: 1 sigma from 7-day mean triggers entry
MEAN_REV_BAND_STD = 1.0


def _compute_trend(prices_df):
    """Return (pct_7d, pct_1d, last_price). Returns (0, 0, 0) on error."""
    try:
        closes  = prices_df["close"].dropna()
        last     = float(closes.iloc[-1])
        week_ago = float(closes.iloc[max(0, len(closes) - 168)])
        day_ago  = float(closes.iloc[max(0, len(closes) - 24)])
        return (last - week_ago) / week_ago * 100, (last - day_ago) / day_ago * 100, last
    except Exception:
        return 0.0, 0.0, 0.0


def _detect_regime(prices_df) -> str:
    """Returns 'ranging' or 'trending' based on 7-day absolute price change."""
    try:
        closes = prices_df["close"].dropna()
        if len(closes) < MEAN_REV_LOOKBACK:
            return "trending"
        current  = float(closes.iloc[-1])
        week_ago = float(closes.iloc[-MEAN_REV_LOOKBACK])
        pct_7d   = abs((current - week_ago) / week_ago * 100)
        return "ranging" if pct_7d < REGIME_THRESHOLD_PCT else "trending"
    except Exception:
        return "trending"


def _mean_reversion_signal(prices_df, portfolio) -> str:
    """
    Mean-reversion signal for ranging markets (Bollinger Band approach).
    Buy at lower band, close long at mean.
    Short at upper band, cover short at mean.
    SL at 2.5% to protect against range breakout.
    """
    try:
        closes  = prices_df["close"].dropna()
        recent  = closes.iloc[-MEAN_REV_LOOKBACK:]
        mean    = float(recent.mean())
        std     = float(recent.std())
        current = float(closes.iloc[-1])
        upper   = mean + MEAN_REV_BAND_STD * std
        lower   = mean - MEAN_REV_BAND_STD * std

        has_long  = float(portfolio.get("collateral_long",  0)) > 0
        has_short = float(portfolio.get("collateral_short", 0)) > 0

        if has_long and current >= mean:
            action = "hold"
            reason = f"MR close long: {current:,.0f} returned to mean {mean:,.0f}"
        elif has_short and current <= mean:
            action = "hold"
            reason = f"MR close short: {current:,.0f} returned to mean {mean:,.0f}"
        elif has_long:
            action = "long"
            reason = f"MR hold long: {current:,.0f} still below mean {mean:,.0f}"
        elif has_short:
            action = "short"
            reason = f"MR hold short: {current:,.0f} still above mean {mean:,.0f}"
        elif current <= lower:
            action = "long"
            reason = f"MR long: {current:,.0f} <= lower band {lower:,.0f} (mean {mean:,.0f})"
        elif current >= upper:
            action = "short"
            reason = f"MR short: {current:,.0f} >= upper band {upper:,.0f} (mean {mean:,.0f})"
        else:
            action = "hold"
            reason = f"MR wait: {current:,.0f} inside bands ({lower:,.0f}–{upper:,.0f})"

        return json.dumps({
            "action":          action,
            "quantity":        5000,
            "stop_loss_pct":   0.025,
            "take_profit_pct": 0.015,
            "confidence":      0.75,
            "reasoning":       reason,
        })
    except Exception as e:
        return json.dumps({
            "action": "hold", "quantity": 0,
            "stop_loss_pct": 0.03, "take_profit_pct": 0.05,
            "confidence": 0.5, "reasoning": f"MR error: {e}",
        })


def _extract_max_margin(risk_content: str) -> float:
    try:
        return float(json.loads(risk_content).get("max_position_margin", 1000))
    except Exception:
        return 1000.0


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio  = state["data"]["portfolio"]
    prices_df  = state["data"]["prices"]

    technical_message = next(
        msg for msg in state["messages"] if msg.name == "technical_analyst_agent"
    )
    sentiment_message = next(
        msg for msg in state["messages"] if msg.name == "sentiment_agent"
    )
    risk_message = next(
        msg for msg in state["messages"] if msg.name == "risk_management_agent"
    )

    max_margin         = _extract_max_margin(risk_message.content)
    pct_7d, pct_1d, last_price = _compute_trend(prices_df)
    trend_dir          = "UP" if pct_7d > 0 else "DOWN"
    price_context      = (
        f"Current price: {last_price:,.2f}  |  "
        f"1-day change: {pct_1d:+.2f}%  |  "
        f"7-day change: {pct_7d:+.2f}% ({trend_dir})"
    )

    # ── Regime detection: ranging → mean reversion, trending → LLM ────────
    regime = _detect_regime(prices_df)
    print(f"  [Regime] {regime.upper()}  (7d |Δ|={abs(pct_7d):.1f}%, threshold={REGIME_THRESHOLD_PCT}%)")

    if regime == "ranging":
        content = _mean_reversion_signal(prices_df, portfolio)
        print(f"  [Mode] MEAN REVERSION  →  {json.loads(content).get('action','?').upper()}")
    else:
        # ── Trending: use LLM ──────────────────────────────────────────────
        system_prompt = f"""You are a crypto perpetual-futures portfolio manager. Output ONE valid JSON object only.
No markdown, no code fences, no text outside the JSON.

HOW PERPETUAL FUTURES WORK:
- "long"  = buy BTC futures → you profit when price RISES.
- "short" = sell BTC futures → you profit when price FALLS. quantity is still a positive USD number.
- "hold"  = stay in cash, no position.

Decision weights:
- Technical Analysis: {TECHNICAL_ANALYSIS_WEIGHT}%
- Sentiment Analysis: {SENTIMENT_ANALYSIS_WEIGHT}%

Sizing rule: quantity must be between 200 and {max_margin:.0f}.
For "hold" set quantity to 0.

Long example:  {{"action": "long",  "quantity": 4500, "stop_loss_pct": 0.035, "take_profit_pct": 0.05, "confidence": 0.75, "reasoning": "7-day trend up, MACD bullish"}}
Short example: {{"action": "short", "quantity": 4500, "stop_loss_pct": 0.035, "take_profit_pct": 0.05, "confidence": 0.70, "reasoning": "7-day trend down, MACD bearish"}}
Hold example:  {{"action": "hold",  "quantity": 0,    "stop_loss_pct": 0,     "take_profit_pct": 0,    "confidence": 0.50, "reasoning": "mixed signals"}}"""

        human_prompt = f"""PRICE CONTEXT: {price_context}

Technical Analysis: {technical_message.content}
Sentiment Analysis: {sentiment_message.content}
Risk Management: {risk_message.content}

Portfolio cash: {portfolio['cash']:.2f}

DECISION RULES (follow strictly):
1. If 7-day change is POSITIVE (price rising) → prefer "long"
2. If 7-day change is NEGATIVE (price falling) → prefer "short"
3. If signals directly contradict the price trend → "hold"

Reply with ONLY the JSON object. quantity must be > 0 for long/short."""

        response = _groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        print(f"  [Mode] AI AGENT (Groq LLM)")

        try:
            parsed = json.loads(content)
            action = parsed.get("action", "hold")
            qty    = float(parsed.get("quantity", 0))

            # Safety net: non-zero quantity for long/short
            if action in ("long", "short") and qty <= 0:
                parsed["quantity"] = max_margin
                print(f"  [PM] qty was 0 for {action}, defaulted to {max_margin:.0f}")

            # Trend override: fix clear directional contradiction
            if action == "short" and pct_7d > TREND_OVERRIDE_THRESHOLD:
                parsed["action"] = "long"
                parsed["reasoning"] = parsed.get("reasoning", "") + f" [override: 7d={pct_7d:+.1f}% UP]"
                print(f"  [PM] Direction override SHORT→LONG  (7d trend: {pct_7d:+.1f}%)")
            elif action == "long" and pct_7d < -TREND_OVERRIDE_THRESHOLD:
                parsed["action"] = "short"
                parsed["reasoning"] = parsed.get("reasoning", "") + f" [override: 7d={pct_7d:+.1f}% DOWN]"
                print(f"  [PM] Direction override LONG→SHORT  (7d trend: {pct_7d:+.1f}%)")

            content = json.dumps(parsed)
        except Exception:
            pass

    message = HumanMessage(content=content, name="portfolio_management")

    if show_reasoning:
        show_agent_reasoning(content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}
