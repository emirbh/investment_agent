## API KEYS required
- [FRED API KEY](https://fred.stlouisfed.org/docs/api/fred/)
- [NEWS API KEY](https://mediastack.com/)

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
### (Alternative) 4-9. Pipeline - all steps together
```
python main.py pipeline
```

## Weekly Execution
### Full pipeline: collect → featurize → train → predict → backtest → report
```
python main.py pipeline
```
### Run Scheduler
```
python main.py scheduler
```

## Adjust P/ortfolio
### Add a new position
```
python main.py portfolio add NOBL 75 --cost 98.00
```

### Add more shares to an existing position
```
python main.py portfolio add SCHD 50 --cost 26.00
```

### Update share count (replaces, doesn't add)
```
python main.py portfolio update SCHD 200 --cost 26.50
```

### Remove a position entirely
```
python main.py portfolio remove VYM
```

### View current portfolio
```
python main.py portfolio show
```


## Examples
### Discover
`python main.py discover`
```
[Discovery] Scanning 37 seed ETFs...
  ... scanned 10/37, found 10 so far
  ... scanned 20/37, found 20 so far
  ... scanned 30/37, found 30 so far
[Discovery] Found 37 dividend ETFs (min AUM: $100,000,000, min yield: 0.5%).

Discovered 37 dividend ETFs, 21 new additions to universe.

By category:
  high_yield                13 ETFs
  international_dividend    6 ETFs
  dividend_growth           5 ETFs
  covered_call              4 ETFs
  preferred_stock           4 ETFs
  reit                      3 ETFs
  utilities                 2 ETFs
```

### Ingest
`python main.py ingest`
```
─── Ingestion complete ──────────────────────────────────
ChromaDB document counts:
  macro_data          : 10 docs
  stock_data          : 96 docs
  insider_data        : 0 docs
  news_data           : 0 docs

SQLite: 16 tickers with price data, 10 macro series
```