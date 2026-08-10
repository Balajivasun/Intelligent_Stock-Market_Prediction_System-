"""
Sentiment Analysis Module
Analyzes financial news headlines from yfinance using a sentiment lexicon.
"""

from typing import Dict, Any
import re
import datetime
import numpy as np
import yfinance as yf


class StockSentimentAnalyzer:
    POSITIVE_WORDS = {
        "surge": 2.0, "jump": 1.8, "rally": 2.0, "gain": 1.5, "gains": 1.5,
        "soar": 2.2, "growth": 1.5, "profit": 1.8, "profits": 1.8, "profitable": 1.6,
        "record": 1.4, "bullish": 2.0, "upgrade": 1.8, "upgraded": 1.8, "beat": 1.7,
        "outperform": 1.9, "strong": 1.4, "positive": 1.3, "higher": 1.2, "highs": 1.4,
        "rebound": 1.5, "expansion": 1.4, "dividend": 1.3, "deal": 1.2, "partnership": 1.3,
        "buy": 1.4, "innovation": 1.3, "breakthrough": 1.9, "success": 1.5, "boom": 1.8,
        "upbeat": 1.6, "optimistic": 1.5, "revenue": 1.0, "target": 1.0, "upside": 1.6,
    }

    NEGATIVE_WORDS = {
        "plunge": 2.2, "slump": 2.0, "drop": 1.5, "drops": 1.5, "fall": 1.4, "falls": 1.4,
        "crash": 2.5, "loss": 1.8, "losses": 1.8, "down": 1.2, "bearish": 2.0,
        "downgrade": 1.9, "downgraded": 1.9, "miss": 1.7, "missed": 1.7, "underperform": 1.8,
        "weak": 1.5, "weakness": 1.6, "negative": 1.4, "lower": 1.2, "lows": 1.4,
        "slashed": 1.8, "lawsuit": 1.9, "investigation": 1.9, "fraud": 2.5, "risk": 1.3,
        "risks": 1.4, "warning": 1.7, "warns": 1.7, "decline": 1.5, "declines": 1.5,
        "debt": 1.4, "layoffs": 1.8, "inflation": 1.3, "recession": 1.9, "turmoil": 2.0,
        "selloff": 2.0, "sell": 1.3, "cut": 1.4, "cuts": 1.4, "crisis": 2.2,
    }

    NEGATIONS = {"not", "no", "never", "without", "hardly", "barely", "scarcely", "despite"}

    def __init__(self):
        pass

    def _score_text(self, text: str) -> float:
        """Compute polarity score for a headline with negation tracking."""
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if not words:
            return 0.0

        score = 0.0
        negation_countdown = 0

        for word in words:
            if word in self.NEGATIONS:
                negation_countdown = 4
                continue

            multiplier = -1.0 if negation_countdown > 0 else 1.0

            if word in self.POSITIVE_WORDS:
                score += self.POSITIVE_WORDS[word] * multiplier
            elif word in self.NEGATIVE_WORDS:
                score -= self.NEGATIVE_WORDS[word] * multiplier

            if negation_countdown > 0:
                negation_countdown -= 1

        return round(float(np.clip(score / 3.5, -1.0, 1.0)), 3)

    def analyze_ticker_news(self, ticker: str, max_articles: int = 10) -> Dict[str, Any]:
        """Fetch recent news for the ticker and calculate sentiment breakdown."""
        ticker = ticker.strip().upper()
        try:
            stock = yf.Ticker(ticker)
            raw_news = stock.news or []
        except Exception:
            raw_news = []

        articles = []
        scores = []
        positive_count = 0
        neutral_count = 0
        negative_count = 0

        for item in raw_news[:max_articles]:
            title = ""
            publisher = "Financial News"
            link = "#"
            pub_time_str = "Recently"

            if isinstance(item, dict):
                content = item.get("content", {})
                if isinstance(content, dict) and "title" in content:
                    title = content.get("title", "")
                    publisher = content.get("provider", {}).get("displayName", publisher)
                    link = content.get("canonicalUrl", {}).get("url", link)
                    pub_date = content.get("pubDate")
                    if pub_date:
                        pub_time_str = pub_date[:10]
                else:
                    title = item.get("title", "")
                    publisher = item.get("publisher", publisher)
                    link = item.get("link", link)
                    epoch_time = item.get("providerPublishTime")
                    if epoch_time:
                        pub_time_str = datetime.datetime.fromtimestamp(epoch_time).strftime("%Y-%m-%d")

            if not title:
                continue

            score = self._score_text(title)
            scores.append(score)

            if score > 0.15:
                label = "Bullish"
                tag_class = "positive"
                positive_count += 1
            elif score < -0.15:
                label = "Bearish"
                tag_class = "negative"
                negative_count += 1
            else:
                label = "Neutral"
                tag_class = "neutral"
                neutral_count += 1

            articles.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "date": pub_time_str,
                "score": score,
                "label": label,
                "tag_class": tag_class,
            })

        total_analyzed = len(scores)
        if total_analyzed > 0:
            avg_score = float(np.mean(scores))
            pos_pct = round((positive_count / total_analyzed) * 100.0, 1)
            neu_pct = round((neutral_count / total_analyzed) * 100.0, 1)
            neg_pct = round((negative_count / total_analyzed) * 100.0, 1)
        else:
            avg_score = 0.0
            pos_pct, neu_pct, neg_pct = 33.3, 33.4, 33.3

        if avg_score > 0.15:
            overall_sentiment = "BULLISH"
            sentiment_summary = f"News headlines lean positive ({pos_pct}% Bullish)."
        elif avg_score < -0.15:
            overall_sentiment = "BEARISH"
            sentiment_summary = f"News headlines lean negative ({neg_pct}% Bearish)."
        else:
            overall_sentiment = "NEUTRAL"
            sentiment_summary = f"News coverage is balanced or neutral ({neu_pct}% Neutral)."

        return {
            "symbol": ticker,
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(avg_score, 2),
            "sentiment_summary": sentiment_summary,
            "counts": {
                "positive": positive_count,
                "neutral": neutral_count,
                "negative": negative_count,
                "total": total_analyzed,
            },
            "breakdown_pct": {
                "positive": pos_pct,
                "neutral": neu_pct,
                "negative": neg_pct,
            },
            "articles": articles,
        }
