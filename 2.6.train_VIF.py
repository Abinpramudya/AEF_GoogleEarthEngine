import pandas as pd
import numpy as np
import json, os, time, itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from math import comb

# ═══════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════
BORUTA_RUN_DIR  = 'data/boruta/5'          # ← your latest boruta run
TRAIN_CSV       = f'{BORUTA_RUN_DIR}/train.csv'
TEST_CSV        = f'{BORUTA_RUN_DIR}/test.csv'
RANKINGS_CSV    = f'{BORUTA_RUN_DIR}/feature_rankings.csv'
OUTPUT_DIR      = 'data/vif_sweep'

TARGET_COL      = 'veg_fraction'
META_COLS       = ['system:index', 'cell_x', 'cell_y']
RANDOM_STATE    = 42
N_ESTIMATORS    = 200
VIF_THRESHOLD   = 10     # standard threshold; raise to 10 if too aggressive
MAX_K           = 10      # exhaustive search up to this many bands
SEC_PER_RUN     = 0.5     # rough estimate per RF fit (seconds)
# ═══════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load ───────────────────────────────────────────────
train    = pd.read_csv(TRAIN_CSV)
test     = pd.read_csv(TEST_CSV)
rankings = pd.read_csv(RANKINGS_CSV)

rankings = rankings.sort_values(['boruta_ranking', 'pearson_abs'],
                                 ascending=[True, False]).reset_index(drop=True)
selected = [f for f in rankings['feature'] if f in train.columns]
print(f"Boruta selected bands : {len(selected)}")

y_train = 1 - train[TARGET_COL].values
y_test  = 1 - test[TARGET_COL].values


# ── Step 1: VIF filtering ──────────────────────────────
print(f"\nComputing VIF (threshold={VIF_THRESHOLD})...")

remaining = selected.copy()
iteration = 0
vif_log   = []

while True:
    X_vif = train[remaining].values.astype(float)
    vifs  = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
    vif_df = pd.DataFrame({'feature': remaining, 'vif': vifs}).sort_values('vif', ascending=False)

    max_vif  = vif_df.iloc[0]['vif']
    max_feat = vif_df.iloc[0]['feature']

    vif_log.append({'iteration': iteration, 'n_remaining': len(remaining), 'max_vif': round(max_vif, 2), 'removed': max_feat if max_vif > VIF_THRESHOLD else None})

    if max_vif <= VIF_THRESHOLD:
        print(f"  ✅ All VIF ≤ {VIF_THRESHOLD} after {iteration} removals")
        break

    tqdm.write(f"  iter {iteration:3d} — removing '{max_feat}' (VIF={max_vif:.2f})  remaining={len(remaining)-1}")
    remaining.remove(max_feat)
    iteration += 1

vif_survivors = remaining
vif_df_final  = pd.DataFrame({'feature': vif_survivors,
                               'vif': [variance_inflation_factor(train[vif_survivors].values.astype(float), i)
                                       for i in range(len(vif_survivors))]})
vif_df_final = vif_df_final.sort_values('vif', ascending=False)
vif_df_final.to_csv(f'{OUTPUT_DIR}/vif_survivors.csv', index=False)

print(f"\nVIF survivors : {len(vif_survivors)} bands")
print(vif_df_final.to_string(index=False))


# ── Step 2: Runtime estimate & confirm ────────────────
n = len(vif_survivors)
total_combos = sum(comb(n, k) for k in range(1, min(MAX_K, n) + 1))
est_seconds  = total_combos * SEC_PER_RUN
est_hours    = est_seconds / 3600

print(f"\n{'═'*50}")
print(f"  VIF survivors    : {n} bands")
print(f"  Combinations 1–{MAX_K}: {total_combos:,}")
print(f"  Est. runtime     : {est_hours:.1f} hrs  ({est_seconds/60:.0f} min)")
print(f"{'═'*50}")

ans = input("\nProceed with exhaustive search? (y/n): ").strip().lower()
if ans != 'y':
    print("Aborted. VIF survivors saved to vif_survivors.csv")
    exit()


# ── Step 3: Exhaustive combo search ───────────────────
print(f"\nStarting exhaustive search — {total_combos:,} combinations...\n")

results  = []
best_r2  = -np.inf
best_rec = None
start    = time.time()

# Save checkpoint every N results in case of crash
CHECKPOINT_EVERY = 500

for k in range(1, min(MAX_K, n) + 1):
    combos = list(itertools.combinations(vif_survivors, k))
    print(f"  k={k}  ({len(combos):,} combos)")

    for combo in tqdm(combos, desc=f'  k={k}', leave=False,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        combo = list(combo)
        X_tr  = train[combo].values
        X_te  = test[combo].values

        rf = RandomForestRegressor(
            n_estimators = N_ESTIMATORS,
            max_features = 'sqrt',
            random_state = RANDOM_STATE,
            n_jobs       = -1
        )
        rf.fit(X_tr, y_train)

        tr_pred = rf.predict(X_tr)
        te_pred = rf.predict(X_te)

        rec = {
            'k':         k,
            'bands':     combo,
            'train_r2':  round(r2_score(y_train, tr_pred), 4),
            'test_r2':   round(r2_score(y_test,  te_pred),  4),
            'test_rmse': round(np.sqrt(mean_squared_error(y_test, te_pred)), 4),
            'overfit':   round(r2_score(y_train, tr_pred) - r2_score(y_test, te_pred), 4),
        }
        results.append(rec)

        if rec['test_r2'] > best_r2:
            best_r2  = rec['test_r2']
            best_rec = rec
            tqdm.write(f"  ★ New best  k={k}  test_R²={best_r2:.4f}  RMSE={rec['test_rmse']:.4f}  {combo}")

    # Checkpoint after each k
    pd.DataFrame([{kk: vv for kk, vv in r.items() if kk != 'bands'} for r in results])\
      .to_csv(f'{OUTPUT_DIR}/sweep_results.csv', index=False)

elapsed = time.time() - start
print(f"\nTotal time: {elapsed/60:.1f} min")


# ── Save final results ─────────────────────────────────
summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'bands'} for r in results])
summary_sorted = summary.sort_values('test_r2', ascending=False)
summary_sorted.to_csv(f'{OUTPUT_DIR}/sweep_results.csv', index=False)
json.dump(results, open(f'{OUTPUT_DIR}/sweep_results.json', 'w'), indent=2)

print(f"\n✅ Best combination:")
print(f"   k          : {best_rec['k']}")
print(f"   test R²    : {best_rec['test_r2']}")
print(f"   test RMSE  : {best_rec['test_rmse']}")
print(f"   overfit    : {best_rec['overfit']:+.4f}")
print(f"   bands      : {best_rec['bands']}")
print(f"\nResults → {OUTPUT_DIR}/")