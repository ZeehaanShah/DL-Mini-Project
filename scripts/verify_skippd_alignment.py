"""Verify SKIPP'D parquet-npy timestamp alignment."""
import pyarrow.parquet as pq
import numpy as np
import pandas as pd

# Train
pf = pq.ParquetFile('data/raw/skippd/train-00000-of-00005.parquet')
first_batch = next(pf.iter_batches(batch_size=10, columns=['time']))
train_ts = first_batch.to_pandas()['time']

tv = np.load('data/raw/skippd/times_trainval.npy', allow_pickle=True)

print("Train parquet first 5 timestamps (tz-aware):")
for i in range(5):
    print(f"  [{i}] {train_ts.iloc[i]}")

print("\ntimes_trainval.npy first 5 (naive):")
for i in range(5):
    print(f"  [{i}] {tv[i]}")

print("\nAlignment check (strip tz from parquet):")
for i in range(10):
    parquet_naive = train_ts.iloc[i].tz_localize(None)
    npy_ts = pd.Timestamp(tv[i])
    match = parquet_naive == npy_ts
    print(f"  [{i}] parquet={parquet_naive}, npy={npy_ts}, match={match}")

# Test
pf_test = pq.ParquetFile('data/raw/skippd/test-00000-of-00001.parquet')
test_batch = next(pf_test.iter_batches(batch_size=10, columns=['time']))
test_ts = test_batch.to_pandas()['time']

te = np.load('data/raw/skippd/times_test.npy', allow_pickle=True)

print("\nTest parquet first 5 timestamps:")
for i in range(5):
    print(f"  [{i}] {test_ts.iloc[i]}")

print("\ntimes_test.npy first 5:")
for i in range(5):
    print(f"  [{i}] {te[i]}")

print("\nTest alignment check:")
for i in range(10):
    parquet_naive = test_ts.iloc[i].tz_localize(None)
    npy_ts = pd.Timestamp(te[i])
    match = parquet_naive == npy_ts
    print(f"  [{i}] parquet={parquet_naive}, npy={npy_ts}, match={match}")

# PV target statistics from full train shard
print("\n--- PV Target Statistics (Full Train Shard) ---")
all_pv = []
for batch in pf.iter_batches(batch_size=1000, columns=['pv']):
    all_pv.extend(batch.to_pandas()['pv'].tolist())

pv_series = pd.Series(all_pv, dtype='float32')
print(f"Count: {len(pv_series)}")
print(f"Min: {pv_series.min():.6f}")
print(f"Max: {pv_series.max():.6f}")
print(f"Mean: {pv_series.mean():.6f}")
print(f"Std: {pv_series.std():.6f}")
print(f"Median: {pv_series.median():.6f}")
print(f"NaN count: {pv_series.isna().sum()}")
print(f"Zeros: {(pv_series == 0).sum()}")
print(f"Negative: {(pv_series < 0).sum()}")
print(f"Q25: {pv_series.quantile(0.25):.6f}")
print(f"Q75: {pv_series.quantile(0.75):.6f}")

# Check temporal resolution
print("\n--- Temporal Resolution (Train) ---")
pf2 = pq.ParquetFile('data/raw/skippd/train-00000-of-00005.parquet')
time_batches = []
for batch in pf2.iter_batches(batch_size=1000, columns=['time']):
    time_batches.append(batch.to_pandas()['time'])
all_times = pd.concat(time_batches).sort_values().reset_index(drop=True)
diffs = all_times.diff().dropna()
print(f"Time diffs (seconds) - unique values (first 20): {sorted(diffs.dt.total_seconds().unique())[:20]}")
print(f"Most common diff: {diffs.dt.total_seconds().mode().values}")
print(f"Min diff: {diffs.min()}")
print(f"Max diff: {diffs.max()}")
