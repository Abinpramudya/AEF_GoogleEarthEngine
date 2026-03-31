import pandas as pd
import numpy as np
import json, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

# ═══════════════════════════════════════════════════════
#  CONFIG  — point to your latest boruta run
# ═══════════════════════════════════════════════════════
BORUTA_RUN_DIR  = 'data/boruta/10'          # ← change to your latest run ID
TRAIN_CSV       = f'{BORUTA_RUN_DIR}/train.csv'
TEST_CSV        = f'{BORUTA_RUN_DIR}/test.csv'
RANKINGS_CSV    = f'{BORUTA_RUN_DIR}/feature_rankings.csv'
OUTPUT_DIR      = 'data/band_sweep'

TARGET_COL      = 'veg_fraction'
RANDOM_STATE    = 42
N_ESTIMATORS    = 200
K_STEPS         = list(range(5, 55, 5))    # 5,10,15,20,25,30,35,40,45,50 + full
# ═══════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────
train    = pd.read_csv(TRAIN_CSV)
test     = pd.read_csv(TEST_CSV)
rankings = pd.read_csv(RANKINGS_CSV)

# Sort by boruta_ranking then pearson_abs descending → importance order
rankings = rankings.sort_values(['boruta_ranking', 'pearson_abs'],
                                 ascending=[True, False]).reset_index(drop=True)

# Only keep bands that exist in train (i.e. boruta selected)
selected_ranked = [f for f in rankings['feature'] if f in train.columns]
print(f"Ranked selected bands: {len(selected_ranked)}")

META_COLS = ['system:index', 'cell_x', 'cell_y']

def get_xy(df):
    drop = [TARGET_COL] + [c for c in META_COLS if c in df.columns]
    X = df.drop(columns=drop).values
    y = 1 - df[TARGET_COL].values   # invert: 0=bare, 1=dense
    return X, y

# Add full set as last step
k_list = sorted(set(K_STEPS + [len(selected_ranked)]))

results = []

for k in tqdm(k_list, desc='Band sweep'):
    subset   = selected_ranked[:k]
    X_train  = train[subset].values
    y_train  = 1 - train[TARGET_COL].values
    X_test   = test[subset].values
    y_test   = 1 - test[TARGET_COL].values

    rf = RandomForestRegressor(
        n_estimators = N_ESTIMATORS,
        max_features = 'sqrt',
        random_state = RANDOM_STATE,
        n_jobs       = -1
    )
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    test_pred  = rf.predict(X_test)

    results.append({
        'k':          k,
        'bands':      subset,
        'train_r2':   round(r2_score(y_train, train_pred), 4),
        'test_r2':    round(r2_score(y_test,  test_pred),  4),
        'test_rmse':  round(np.sqrt(mean_squared_error(y_test, test_pred)), 4),
        'overfit':    round(r2_score(y_train, train_pred) - r2_score(y_test, test_pred), 4),
    })
    tqdm.write(f"  k={k:3d}  train_R²={results[-1]['train_r2']:.4f}  "
               f"test_R²={results[-1]['test_r2']:.4f}  "
               f"RMSE={results[-1]['test_rmse']:.4f}  "
               f"overfit={results[-1]['overfit']:+.4f}")

# ── Save results ───────────────────────────────────────
summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'bands'} for r in results])
summary.to_csv(f'{OUTPUT_DIR}/sweep_results.csv', index=False)
json.dump(results, open(f'{OUTPUT_DIR}/sweep_results.json', 'w'), indent=2)

best = summary.loc[summary['test_r2'].idxmax()]
print(f"\n✅ Best: k={int(best['k'])}  test_R²={best['test_r2']:.4f}  RMSE={best['test_rmse']:.4f}")
print(f"Results saved → {OUTPUT_DIR}/")