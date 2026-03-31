# Investment Agent

Automated portfolio optimization system for dividend ETFs. Collects market and macroeconomic data, engineers features, trains a deep learning model to predict forward returns, ranks the full ETF universe by expected performance, backtests the strategy against your actual holdings, and generates weekly reports with actionable buy/hold/sell recommendations.

Runs as a CLI tool (`main.py`), an MCP server for conversational use with Claude (`mcp_server.py`), or a scheduled daemon via APScheduler.

## Methodology

- **Universe construction** — Starts from a curated seed list of 40+ dividend ETFs across 7 categories (high yield, dividend growth, covered call, preferred stock, international, REITs, utilities). Expands automatically by scanning yfinance metadata, filtering by minimum AUM and dividend yield, and classifying into strategy categories.

- **Data pipeline** — Daily price/volume/dividend data from yfinance and 10 macroeconomic series from FRED (fed funds rate, 10Y/2Y Treasury yields, yield curve spread, CPI, core CPI, TIPS real yield, high-yield credit spread, WTI crude, unemployment). All stored in SQLite with WAL mode for concurrent reads.

- **Feature engineering** — 24 features per ticker per day:
  - Price-derived: log returns at 5 horizons (1d, 5d, 10d, 21d, 63d), rolling volatility (10d, 21d), SMA ratios (10/21/50-day), RSI-14, volume ratio vs 20-day average
  - Dividend: trailing 12-month yield, distance from 52-week high/low
  - Macro: all 10 FRED series forward-filled to align with trading dates
  - Target variable: forward 21-day log return

- **Model architecture** — 2-layer LSTM (hidden=64, dropout=0.2) with temporal attention that learns which days in the input window matter most, followed by a fully connected head (64 → 32 → 1). Input normalization via LayerNorm. Predicts a single scalar: expected forward return.

- **Training** — AdamW optimizer with weight decay 1e-4, cosine annealing learning rate schedule, gradient clipping at norm 1.0, early stopping on validation loss with configurable patience. Automatic device selection: MPS (Apple Silicon GPU) → CUDA → CPU. Checkpoint save/resume support.

- **Prediction and ranking** — Runs inference on the latest feature window for every ticker in the universe. Ranks by predicted return, assigns actions (BUY/OVERWEIGHT/HOLD/UNDERWEIGHT/SELL) based on percentile position within the ranked list, and persists results for reporting and backtesting.

- **Walk-forward backtesting** — Simulates weekly rebalancing over a configurable trailing period. Each week the strategy equal-weights the top-N tickers by model prediction. Baseline is equal-weight buy-and-hold of the user's actual portfolio positions. Computes cumulative return, annualized return, Sharpe ratio (annualized, excess over risk-free rate), maximum drawdown, and weekly win rate vs baseline.

- **Report generation** — Weekly Markdown report with executive summary, portfolio snapshot, full prediction ranking table, backtest comparison, and macro environment overview. Also exports predictions as CSV. Optional PDF generation via pandoc.

---

## API Keys Required

- [FRED API Key](https://fred.stlouisfed.org/docs/api/fred/)
- [NEWS API Key](https://mediastack.com/)

## Initiate

### 1. Add your current holdings
```
python main.py portfolio add SCHD 100 --cost 25.50
python main.py portfolio add VYM 50 --cost 45.00
python main.py portfolio add JEPI 200 --cost 55.00
```

### 2. Seed the ETF universe from config
```
python main.py universe seed
```

### 3. Auto-discover peer dividend ETFs
```
python main.py discover
```

### 4. Collect price + macro data into SQLite
```
python main.py collect
```

### 5. Compute ML feature vectors
```
python main.py featurize
```

### 6. Train the LSTM model
```
python main.py train
```

### 7. Generate ranked predictions
```
python main.py predict
```

### 8. Run walk-forward backtest
```
python main.py backtest
```

### 9. Generate the weekly report
```
python main.py report
```

### (Alternative) Steps 4-9 as a single pipeline
```
python main.py pipeline
```

## Weekly Execution

### Full pipeline: collect → featurize → train → predict → backtest → report
```
python main.py pipeline
```

### Scheduled daemon
```
python main.py scheduler
```

## MCP Server

Run conversationally with Claude Desktop or Claude Code:
```
python mcp_server.py          # stdio transport
python mcp_server.py --sse    # SSE transport on port 8000
```

Add to Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "investment-agent": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/investment-agent"
    }
  }
}
```

16 tools exposed: `portfolio_show`, `portfolio_add`, `portfolio_remove`, `portfolio_update`, `universe_show`, `universe_seed`, `discover`, `collect`, `featurize`, `train`, `predict`, `backtest`, `report`, `pipeline`, `stats`, `live_price`.

## Adjust Portfolio

```
python main.py portfolio add NOBL 75 --cost 98.00       # new position
python main.py portfolio add SCHD 50 --cost 26.00       # add to existing
python main.py portfolio update SCHD 200 --cost 26.50   # replace share count
python main.py portfolio remove VYM                      # remove entirely
python main.py portfolio show                            # view holdings
```

## Check Status

```
python main.py stats              # DB row counts
python main.py universe show      # full ETF universe
python main.py universe stats     # price/macro data coverage
python main.py predict            # latest ranked predictions
```
