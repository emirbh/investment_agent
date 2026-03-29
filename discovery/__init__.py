"""
Automatic ETF peer discovery — finds dividend ETFs via screeners and yfinance.
"""

from discovery.etf_screener import discover_dividend_etfs
from discovery.universe import refresh_universe
