import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


# ══════════════════════════════════════════════════════════
#  CONFIG — edit here only
# ══════════════════════════════════════════════════════════
BORUTA_RUN         = 10
FULL_FIELD_CSV     = 'data/training_data_bistugliu.csv'
TARGET_COL         = 'veg_fraction'
META_COLS          = ['system:index', 'cell_x', 'cell_y']

N_ESTIMATORS       = 500
MAX_DEPTH          = 12
MIN_SAMPLES_LEAF   = 1
MAX_FEATURES       = 'sqrt'
RANDOM_STATE       = 42

CV_FOLDS           = 3
SPATIAL_SORT_COL   = 'cell_y'
# ══════════════════════════════════════════════════════════


# ── Paths ──────────────────────────────────────────────────
BORUTA_DIR = f'data/boruta/{BORUTA_RUN}'
TRAIN_CSV  = f'{BORUTA_DIR}/train.csv'
TEST_CSV   = f'{BORUTA_DIR}/test.csv'

for p in [TRAIN_CSV, TEST_CSV]:
    if not os.path.exists(p):
        raise FileNotFoundError(f'Not found: {p}')

# ── Auto run ID ────────────────────────────────────────────
os.makedirs('output/rf', exist_ok=True)
existing = [
    f for f in os.listdir('output/rf')
    if re.match(r'^\d+$', f) and os.path.isdir(f'output/rf/{f}')
]
run_id     = max([int(f) for f in existing], default=0) + 1
OUTPUT_DIR = f'output/rf/{run_id}'
GRAPH_DIR  = f'{OUTPUT_DIR}/graph'
os.makedirs(GRAPH_DIR, exist_ok=True)

tqdm.write(f'\n{"─"*52}')
tqdm.write(f'  RF run {run_id}  |  Boruta run {BORUTA_RUN}')
tqdm.write(f'  Output  →  {OUTPUT_DIR}/')
tqdm.write(f'{"─"*52}\n')


# ── Load data ──────────────────────────────────────────────
df_train  = pd.read_csv(TRAIN_CSV)
df_test   = pd.read_csv(TEST_CSV)
feat_cols = [c for c in df_train.columns if c not in META_COLS + [TARGET_COL]]

X_train = df_train[feat_cols].values
y_train = df_train[TARGET_COL].values
X_test  = df_test[feat_cols].values
y_test  = df_test[TARGET_COL].values

tqdm.write(f'  Features  : {len(feat_cols)}')
tqdm.write(f'  Train     : {len(X_train)}  |  Test : {len(X_test)}\n')


# ── Helpers ────────────────────────────────────────────────
def make_rf():
    return RandomForestRegressor(
        n_estimators     = N_ESTIMATORS,
        max_depth        = MAX_DEPTH,
        min_samples_leaf = MIN_SAMPLES_LEAF,
        max_features     = MAX_FEATURES,
        random_state     = RANDOM_STATE,
        n_jobs           = -1,
    )

def metrics(y_true, y_pred, label=''):
    m = {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':  float(mean_absolute_error(y_true, y_pred)),
        'r2':   float(r2_score(y_true, y_pred)),
        'bias': float(np.mean(y_pred - y_true)),
        'n':    int(len(y_true)),
    }
    if label:
        tqdm.write(
            f'  {label:<8}  RMSE={m["rmse"]:.4f}  MAE={m["mae"]:.4f}  '
            f'R2={m["r2"]:.4f}  bias={m["bias"]:+.4f}  n={m["n"]}'
        )
    return m


STEPS    = 5
step_bar = tqdm(
    total=STEPS, desc='Pipeline', unit='step', position=0,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
)

def advance(label):
    step_bar.set_postfix_str(label)
    step_bar.update(1)


# ══════════════════════════════════════════════════════════
#  STEP 1 — Spatial block cross-validation
# ══════════════════════════════════════════════════════════
step_bar.set_description('Spatial CV')

sort_col_present = SPATIAL_SORT_COL in df_train.columns
if sort_col_present:
    sort_order   = df_train[SPATIAL_SORT_COL].argsort().values
    X_train_sort = X_train[sort_order]
    y_train_sort = y_train[sort_order]
    tqdm.write(f'  Spatial CV  — sorted by {SPATIAL_SORT_COL}  ({CV_FOLDS} folds)')
else:
    sort_order   = np.arange(len(X_train))
    X_train_sort = X_train
    y_train_sort = y_train
    tqdm.write(f'  WARNING: {SPATIAL_SORT_COL} not found — falling back to random KFold')

kf           = KFold(n_splits=CV_FOLDS, shuffle=False)
cv_rmse, cv_mae, cv_r2, cv_bias = [], [], [], []
fold_details = []
oof_pred     = np.zeros(len(X_train))

fold_bar = tqdm(
    total=CV_FOLDS, desc='  CV folds', unit='fold', leave=False, position=1,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} folds  [{elapsed}<{remaining}]'
)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_sort), 1):
    X_tr, X_val = X_train_sort[tr_idx], X_train_sort[val_idx]
    y_tr, y_val = y_train_sort[tr_idx], y_train_sort[val_idx]

    rf_fold = make_rf()
    rf_fold.fit(X_tr, y_tr)
    y_val_pred = rf_fold.predict(X_val)

    oof_pred[val_idx] = y_val_pred

    m = metrics(y_val, y_val_pred)
    cv_rmse.append(m['rmse'])
    cv_mae.append(m['mae'])
    cv_r2.append(m['r2'])
    cv_bias.append(m['bias'])
    fold_details.append({
        'fold':          fold,
        'val_start_idx': int(val_idx[0]),
        'val_end_idx':   int(val_idx[-1]),
        **m,
    })

    fold_bar.set_postfix_str(f'fold {fold}  R2={m["r2"]:.3f}  RMSE={m["rmse"]:.4f}')
    fold_bar.update(1)

fold_bar.close()

# ── Fold spatial diagnostics ──────────────────────────────
tqdm.write('\n  Fold spatial breakdown:')
for fd in fold_details:
    val_mask  = sort_order[fd['val_start_idx']:fd['val_end_idx'] + 1]
    cy_vals   = df_train['cell_y'].values[val_mask]
    cx_vals   = df_train['cell_x'].values[val_mask]
    tqdm.write(
        f'  Fold {fd["fold"]}  '
        f'cell_y=[{cy_vals.min():.1f}–{cy_vals.max():.1f}]  '
        f'cell_x=[{cx_vals.min():.1f}–{cx_vals.max():.1f}]  '
        f'n={fd["n"]}  R²={fd["r2"]:+.3f}  RMSE={fd["rmse"]:.4f}'
    )
tqdm.write('')

oof_metrics = metrics(y_train_sort, oof_pred, label='OOF   ')  # ← already there

oof_metrics = metrics(y_train_sort, oof_pred, label='OOF   ')

cv_results = {
    'type':      'spatial_block' if sort_col_present else 'random',
    'sort_col':  SPATIAL_SORT_COL if sort_col_present else None,
    'rmse_mean': float(np.mean(cv_rmse)), 'rmse_std': float(np.std(cv_rmse)),
    'mae_mean':  float(np.mean(cv_mae)),  'mae_std':  float(np.std(cv_mae)),
    'r2_mean':   float(np.mean(cv_r2)),   'r2_std':   float(np.std(cv_r2)),
    'bias_mean': float(np.mean(cv_bias)), 'bias_std': float(np.std(cv_bias)),
    'oof':       oof_metrics,
    'folds':     fold_details,
}

tqdm.write(
    f'\n  CV mean   RMSE={cv_results["rmse_mean"]:.4f}±{cv_results["rmse_std"]:.4f}  '
    f'R2={cv_results["r2_mean"]:.4f}±{cv_results["r2_std"]:.4f}'
)
tqdm.write(
    f'  OOF       RMSE={oof_metrics["rmse"]:.4f}  '
    f'R2={oof_metrics["r2"]:.4f}  bias={oof_metrics["bias"]:+.4f}\n'
)
advance(f'CV  OOF_R2={oof_metrics["r2"]:.3f}  std={cv_results["r2_std"]:.3f}')


# ══════════════════════════════════════════════════════════
#  STEP 2 — Final RF on full train set
# ══════════════════════════════════════════════════════════
step_bar.set_description('Training RF')
tqdm.write('  Fitting final RF on full train set...')

rf = make_rf()
rf.fit(X_train, y_train)

train_metrics = metrics(rf.predict(X_train), y_train, label='Train ')
test_metrics  = metrics(rf.predict(X_test),  y_test,  label='Test  ')
overfit_gap   = round(train_metrics['r2'] - test_metrics['r2'], 6)
tqdm.write(f'  Overfit gap  {overfit_gap:+.4f}\n')
advance(f'RF  test_R2={test_metrics["r2"]:.3f}  gap={overfit_gap:.3f}')


# ══════════════════════════════════════════════════════════
#  STEP 3 — Full-field prediction
# ══════════════════════════════════════════════════════════
step_bar.set_description('Full-field prediction')

df_full   = pd.read_csv(FULL_FIELD_CSV)
meta_full = [c for c in META_COLS if c in df_full.columns]
missing   = [c for c in feat_cols if c not in df_full.columns]
if missing:
    tqdm.write(f'  WARNING: {len(missing)} features missing in full CSV — filling 0')

X_full      = df_full.reindex(columns=feat_cols, fill_value=0).values
y_full      = df_full[TARGET_COL].values if TARGET_COL in df_full.columns else None
y_pred_full = rf.predict(X_full)

pred_df = pd.DataFrame()
for c in meta_full:
    pred_df[c] = df_full[c].values
if y_full is not None:
    pred_df['veg_fraction_gt'] = y_full
pred_df['veg_fraction_pred']   = y_pred_full
if y_full is not None:
    pred_df['residual']   = y_pred_full - y_full
    pred_df['abs_error']  = np.abs(y_pred_full - y_full)

train_idx = set(df_train['system:index'].values) if 'system:index' in df_train.columns else set()
test_idx  = set(df_test['system:index'].values)  if 'system:index' in df_test.columns  else set()
if 'system:index' in df_full.columns:
    pred_df['split'] = df_full['system:index'].apply(
        lambda x: 'train' if x in train_idx else ('test' if x in test_idx else 'full')
    )

pred_df.to_csv(f'{OUTPUT_DIR}/prediction.csv', index=False)
tqdm.write(f'  prediction.csv  ({len(pred_df)} rows)\n')
advance('Full-field done')


# ══════════════════════════════════════════════════════════
#  STEP 4 — Plots
# ══════════════════════════════════════════════════════════
step_bar.set_description('Plotting')

y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

# Build fold_ids array from exact val indices (safe for uneven fold sizes)
fold_ids = np.zeros(len(y_train_sort), dtype=int)
for fd in fold_details:
    fold_ids[fd['val_start_idx']:fd['val_end_idx'] + 1] = fd['fold']

BG_MAIN = '#1a1a2e'
BG_CELL = '#16213e'
C_BLUE  = '#4a9eff'
C_GREEN = '#2ecc71'
C_ORG   = '#f39c12'
C_RED   = '#e74c3c'
C_PURP  = '#a855f7'
C_MUT   = '#aaaaaa'

tkw  = dict(color='white', fontsize=11, fontweight='bold', pad=8)
lkw  = dict(color=C_MUT,  fontsize=8)
txkw = dict(colors=C_MUT, labelsize=7)

def sx(ax):
    ax.set_facecolor(BG_CELL)
    ax.spines[:].set_color('#333355')
    ax.tick_params(axis='x', **txkw)
    ax.tick_params(axis='y', **txkw)

fig = plt.figure(figsize=(24, 16), facecolor=BG_MAIN)
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.38)


# ── Row 0 ──────────────────────────────────────────────────

# [0,0] Predicted vs Actual — test
ax = fig.add_subplot(gs[0, 0])
ax.scatter(y_test, y_pred_test, alpha=0.8, s=35, color=C_BLUE, edgecolors='none')
lim = [min(y_test.min(), y_pred_test.min()) - 0.03,
       max(y_test.max(), y_pred_test.max()) + 0.03]
ax.plot(lim, lim, color=C_GREEN, linewidth=1, linestyle='--')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title('Predicted vs Actual (test)', **tkw)
ax.set_xlabel('Actual',    **lkw)
ax.set_ylabel('Predicted', **lkw)
ax.text(0.05, 0.92,
        f'R² = {test_metrics["r2"]:.3f}\nRMSE = {test_metrics["rmse"]:.4f}',
        transform=ax.transAxes, color=C_MUT, fontsize=8, va='top')
sx(ax)

# [0,1] Residuals vs Predicted — test
ax = fig.add_subplot(gs[0, 1])
res_test = y_pred_test - y_test
ax.scatter(y_pred_test, res_test, alpha=0.8, s=35, color=C_ORG, edgecolors='none')
ax.axhline(0,               color=C_GREEN, linewidth=1, linestyle='--')
ax.axhline(res_test.mean(), color=C_RED,   linewidth=1, linestyle=':')
ax.set_title('Residuals vs Predicted (test)', **tkw)
ax.set_xlabel('Predicted', **lkw)
ax.set_ylabel('Residual',  **lkw)
ax.text(0.05, 0.92, f'bias = {test_metrics["bias"]:+.4f}',
        transform=ax.transAxes, color=C_MUT, fontsize=8)
sx(ax)

# [0,2] OOF predicted vs actual — coloured by fold
ax = fig.add_subplot(gs[0, 2])
ax.scatter(y_train_sort, oof_pred, alpha=0.6, s=25,
           c=fold_ids, cmap='tab10', edgecolors='none')
lim2 = [min(y_train_sort.min(), oof_pred.min()) - 0.03,
        max(y_train_sort.max(), oof_pred.max()) + 0.03]
ax.plot(lim2, lim2, color=C_GREEN, linewidth=1, linestyle='--')
ax.set_xlim(lim2); ax.set_ylim(lim2)
ax.set_title('OOF Predicted vs Actual', **tkw)
ax.set_xlabel('Actual',   **lkw)
ax.set_ylabel('OOF Pred', **lkw)
ax.text(0.05, 0.92,
        f'OOF R² = {oof_metrics["r2"]:.3f}\nOOF RMSE = {oof_metrics["rmse"]:.4f}',
        transform=ax.transAxes, color=C_MUT, fontsize=8, va='top')
sx(ax)

# [0,3] Metrics summary text box
ax = fig.add_subplot(gs[0, 3])
ax.axis('off')
ax.set_facecolor(BG_CELL)
rows = [
    ('── CV (spatial block) ──', ''),
    ('RMSE', f'{cv_results["rmse_mean"]:.4f} ± {cv_results["rmse_std"]:.4f}'),
    ('MAE',  f'{cv_results["mae_mean"]:.4f} ± {cv_results["mae_std"]:.4f}'),
    ('R²',   f'{cv_results["r2_mean"]:.4f} ± {cv_results["r2_std"]:.4f}'),
    ('Bias', f'{cv_results["bias_mean"]:+.4f} ± {cv_results["bias_std"]:.4f}'),
    ('── OOF ──', ''),
    ('R²',   f'{oof_metrics["r2"]:.4f}'),
    ('RMSE', f'{oof_metrics["rmse"]:.4f}'),
    ('── Test ──', ''),
    ('R²',   f'{test_metrics["r2"]:.4f}'),
    ('RMSE', f'{test_metrics["rmse"]:.4f}'),
    ('Bias', f'{test_metrics["bias"]:+.4f}'),
    ('── Overfit ──', ''),
    ('Train R²', f'{train_metrics["r2"]:.4f}'),
    ('Gap',      f'{overfit_gap:+.4f}'),
]
for i, (k, v) in enumerate(rows):
    is_header = k.startswith('──')
    ax.text(0.04, 0.97 - i * 0.063, k,
            transform=ax.transAxes,
            color=C_GREEN if is_header else C_MUT,
            fontsize=8, fontweight='bold' if is_header else 'normal')
    if v:
        ax.text(0.55, 0.97 - i * 0.063, v,
                transform=ax.transAxes, color='white', fontsize=8)
ax.set_title('Metrics Summary', **tkw)


# ── Row 1 ──────────────────────────────────────────────────

# [1,0] CV fold R² bar
ax = fig.add_subplot(gs[1, 0])
folds     = [f['fold'] for f in fold_details]
fold_r2   = [f['r2']   for f in fold_details]
fold_rmse = [f['rmse'] for f in fold_details]
colors_r2 = [C_GREEN if r >= cv_results['r2_mean'] else C_RED for r in fold_r2]
ax.bar(folds, fold_r2, color=colors_r2, edgecolor='none', width=0.6, alpha=0.85)
ax.axhline(cv_results['r2_mean'], color='white', linewidth=1, linestyle='--', alpha=0.5)
ax.axhline(oof_metrics['r2'],     color=C_PURP,  linewidth=1, linestyle=':',  alpha=0.8,
           label=f'OOF R²={oof_metrics["r2"]:.3f}')
ax.set_title(f'{CV_FOLDS}-Fold CV — R² per Fold', **tkw)
ax.set_xlabel('Fold', **lkw)
ax.set_ylabel('R²',   **lkw)
ax.set_xticks(folds)
ax.legend(fontsize=7, labelcolor='white', framealpha=0.2)
sx(ax)

# [1,1] CV fold RMSE bar
ax = fig.add_subplot(gs[1, 1])
colors_rmse = [C_RED if r >= cv_results['rmse_mean'] else C_GREEN for r in fold_rmse]
ax.bar(folds, fold_rmse, color=colors_rmse, edgecolor='none', width=0.6, alpha=0.85)
ax.axhline(cv_results['rmse_mean'], color='white', linewidth=1, linestyle='--', alpha=0.5)
ax.set_title(f'{CV_FOLDS}-Fold CV — RMSE per Fold', **tkw)
ax.set_xlabel('Fold', **lkw)
ax.set_ylabel('RMSE', **lkw)
ax.set_xticks(folds)
sx(ax)

# [1,2] Spatial fold map
ax = fig.add_subplot(gs[1, 2])
if sort_col_present:
    y_coords = df_train[SPATIAL_SORT_COL].values[sort_order]
    scatter  = ax.scatter(
        np.arange(len(y_coords)), y_coords,
        c=fold_ids, cmap='tab10', s=20, alpha=0.8, edgecolors='none'
    )
    cb = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cb.set_label('Fold', color=C_MUT, fontsize=8)
    cb.ax.tick_params(colors=C_MUT)
    ax.set_title(f'Spatial Fold Distribution ({SPATIAL_SORT_COL})', **tkw)
    ax.set_xlabel('Sample index (sorted)', **lkw)
    ax.set_ylabel(SPATIAL_SORT_COL,        **lkw)
else:
    ax.text(0.5, 0.5, f'{SPATIAL_SORT_COL} not available',
            ha='center', va='center', transform=ax.transAxes, color=C_MUT)
    ax.set_title('Spatial Fold Distribution', **tkw)
sx(ax)

# [1,3] Train / CV / Test error bars
ax = fig.add_subplot(gs[1, 3])
x = np.arange(2)
w = 0.22
ax.bar(x - w, [train_metrics['rmse'], train_metrics['mae']], w,
       label='Train', color=C_BLUE,  edgecolor='none', alpha=0.9)
ax.bar(x,     [cv_results['rmse_mean'], cv_results['mae_mean']], w,
       label=f'CV ({CV_FOLDS}-fold)', color=C_ORG, edgecolor='none', alpha=0.9,
       yerr=[cv_results['rmse_std'], cv_results['mae_std']],
       error_kw=dict(ecolor='white', capsize=3, linewidth=1))
ax.bar(x + w, [test_metrics['rmse'],  test_metrics['mae']],  w,
       label='Test',  color=C_GREEN, edgecolor='none', alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels(['RMSE', 'MAE'], color=C_MUT, fontsize=9)
ax.set_title('Train / CV / Test', **tkw)
ax.legend(fontsize=7, labelcolor='white', framealpha=0.2)
sx(ax)


# ── Row 2 — Feature importance (full width) ────────────────
ax = fig.add_subplot(gs[2, :])
importances = (
    pd.Series(rf.feature_importances_, index=feat_cols)
    .sort_values(ascending=True)
    .tail(30)
)
bar_colors = [C_GREEN if v >= importances.median() else C_BLUE
              for v in importances.values]
ax.barh(importances.index, importances.values,
        color=bar_colors, edgecolor='none', height=0.7)
ax.set_title('Feature Importance (top 30)', **tkw)
ax.set_xlabel('Importance', **lkw)
sx(ax)


fig.suptitle(
    f'Random Forest  |  RF run {run_id}  |  Boruta run {BORUTA_RUN}  |  '
    f'{len(feat_cols)} features  |  '
    f'Spatial CV R²={cv_results["r2_mean"]:.3f}±{cv_results["r2_std"]:.3f}  '
    f'OOF R²={oof_metrics["r2"]:.3f}  '
    f'Test R²={test_metrics["r2"]:.3f}  '
    f'Gap={overfit_gap:+.3f}',
    color='white', fontsize=12, fontweight='bold', y=0.995
)

plt.savefig(f'{GRAPH_DIR}/results.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
advance('Graphs saved')


# ══════════════════════════════════════════════════════════
#  STEP 5 — Save result.json
# ══════════════════════════════════════════════════════════
step_bar.set_description('Saving JSON')

result = {
    'rf_run_id':      run_id,
    'boruta_run_id':  BORUTA_RUN,
    'train_csv':      TRAIN_CSV,
    'test_csv':       TEST_CSV,
    'full_field_csv': FULL_FIELD_CSV,
    'n_features':     len(feat_cols),
    'features':       feat_cols,
    'rf_params': {
        'n_estimators':     N_ESTIMATORS,
        'max_depth':        MAX_DEPTH,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'max_features':     MAX_FEATURES,
        'random_state':     RANDOM_STATE,
    },
    'cv':          cv_results,
    'train':       train_metrics,
    'test':        test_metrics,
    'overfit_gap': overfit_gap,
}

with open(f'{OUTPUT_DIR}/result.json', 'w') as f:
    json.dump(result, f, indent=2)

advance('result.json saved')
step_bar.close()


tqdm.write(f'\n{"─"*52}')
tqdm.write(f'  RF run {run_id} complete  →  {OUTPUT_DIR}/')
tqdm.write(f'  ├── prediction.csv    ({len(pred_df)} rows)')
tqdm.write(f'  ├── result.json')
tqdm.write(f'  └── graph/results.png')
tqdm.write(f'{"─"*52}')
tqdm.write(f'  OOF R²  = {oof_metrics["r2"]:.4f}   (honest spatial estimate)')
tqdm.write(f'  Test R² = {test_metrics["r2"]:.4f}   (held-out)')
tqdm.write(f'  Overfit = {overfit_gap:+.4f}   (target < 0.15)')
tqdm.write(f'{"─"*52}\n')