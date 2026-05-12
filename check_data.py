#!/usr/bin/env python
"""Quick diagnostic script to check available training data."""

from src.data_loader import MarketDataLoader
from src.features import FeatureEngineer
from config import PAIR_CONFIGS

loader = MarketDataLoader()
market_panel = loader.fetch_universe(PAIR_CONFIGS, period='1y', interval='1d')
print(f'Market panel shape: {market_panel.shape}')
print(f'Market panel unique pairs: {market_panel["pair"].nunique()}')

engineer = FeatureEngineer(rolling_window=20, horizon=1)
feature_panel = engineer.transform_universe(market_panel)
print(f'Feature panel shape: {feature_panel.shape}')

clean = feature_panel.dropna(subset=['target_te'])
print(f'Feature panel after dropna(target_te): {clean.shape}')

print(f'\nPer-pair rows (clean):')
for pair, group in clean.groupby('pair'):
    print(f'  {pair}: {len(group)}')

# Check first few timestamps
timestamps = sorted(clean.index.unique())
print(f'\nTotal unique timestamps: {len(timestamps)}')
print(f'First 10 timestamps:')
for i, ts in enumerate(timestamps[:10]):
    ts_data = clean.loc[clean.index == ts]
    print(f'  {ts} ({i}): {len(ts_data)} pairs')
    if i > 0:
        prev_ts = timestamps[i-1]
        train_data = clean.loc[clean.index <= prev_ts]
        print(f'    -> train data up to {prev_ts}: {len(train_data)} rows')
