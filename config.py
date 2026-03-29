import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

LOOKBACK_DAYS: int = int(os.getenv("LOOKBACK_DAYS", "90"))

# SQLite database
DB_PATH: str = os.getenv(
    "INVESTMENT_DB_PATH", os.path.join(os.path.dirname(__file__), "investment.db")
)

# Report output directory
REPORT_DIR: str = os.getenv(
    "REPORT_DIR", os.path.join(os.path.dirname(__file__), "reports")
)

# Date range for data ingestion
START_DATE: str = (datetime.now() - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
END_DATE: str = datetime.now().strftime("%Y-%m-%d")

# FRED series — income/dividend-investing focused
# Dividend ETF valuations are highly sensitive to rate levels and inflation.
FRED_SERIES: dict[str, str] = {
    "FEDFUNDS": "Federal Funds Effective Rate",
    "DGS10": "10-Year Treasury Yield (benchmark vs dividend yields)",
    "DGS2": "2-Year Treasury Yield",
    "T10Y2Y": "Yield Curve Spread (10Y-2Y) — recession signal",
    "CPIAUCSL": "Consumer Price Index — inflation erodes real dividend value",
    "CPILFESL": "Core CPI (ex-food & energy)",
    "DFII10": "10-Year TIPS Yield (real rate — key for dividend vs bonds)",
    "BAMLH0A0HYM2": "US High Yield Spread — credit risk appetite",
    "DCOILWTICO": "WTI Crude Oil Price — energy sector dividend driver",
    "UNRATE": "Unemployment Rate — consumer spending, REIT health",
}

# Grouped by strategy for comparison within and across categories.
DIVIDEND_ETFS: dict[str, str] = {
    # Broad high-yield / dividend growth
    "VYM": "Vanguard High Dividend Yield ETF",
    "SCHD": "Schwab US Dividend Equity ETF",
    "DVY": "iShares Select Dividend ETF",
    "DGRO": "iShares Core Dividend Growth ETF",
    "HDV": "iShares Core High Dividend ETF",
    "SDY": "SPDR S&P Dividend ETF",
    "NOBL": "ProShares S&P 500 Dividend Aristocrats ETF",
    "SPHD": "Invesco S&P 500 High Dividend Low Volatility ETF",
    # Covered-call / enhanced income
    "JEPI": "JPMorgan Equity Premium Income ETF",
    "JEPQ": "JPMorgan Nasdaq Equity Premium Income ETF",
    "XYLD": "Global X S&P 500 Covered Call ETF",
    "QYLD": "Global X Nasdaq 100 Covered Call ETF",
    # Preferred stock income
    "PFF": "iShares Preferred & Income Securities ETF",
    "PGX": "Invesco Preferred ETF",
    # International dividend
    "VYMI": "Vanguard International High Dividend Yield ETF",
    "IDV": "iShares International Select Dividend ETF",
}

# Flat list of all tickers for easy iteration
DIVIDEND_ETF_TICKERS: list[str] = list(DIVIDEND_ETFS.keys())
