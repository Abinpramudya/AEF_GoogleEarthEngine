import os, re, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from tqdm import tqdm


# ── CONFIG ────────────────────────────────────────────────
TRAIN_CSV         = 'data/training_data_bistugliu.csv'
TARGET_COL        = 'veg_fraction'
META_COLS         = ['system:index', 'cell_x', 'cell_y']
AEF_YEARS         = '2023, 2024, 2025'
S2_SEASONS        = 'win, spr, sum, aut, snap'

TEST_SIZE         = 0.2
RANDOM_STATE      = 42
N_ESTIMATORS      = 200
MAX_DEPTH         = None
MIN_SAMPLES_LEAF  = 1
MAX_FEATURES      = 'sqrt'
BORUTA_MAX_ITER   = 200
BORUTA_PERC       = 100
BORUTA_ALPHA      = 0.05
INCLUDE_TENTATIVE = True
TOP_N_PLOT        = 40
VALID_SEASONS     = {'win', 'spr', 'sum', 'aut', 'snap'}
# ─────────────────────────────────────────────────────────


# ── Auto run ID ───────────────────────────────────────────
os.makedirs('data/boruta', exist_ok=True)
existing = [
    f for f in os.listdir('data/boruta')
    if re.match(r'^\d+$', f) and os.path.isdir(os.path.join('data/boruta', f))
]
run_id     = max([int(f) for f in existing], default=0) + 1
OUTPUT_DIR = f'data/boruta/{run_id}'
GRAPH_DIR  = f'{OUTPUT_DIR}/graph'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR,  exist_ok=True)
tqdm.write(f'Run ID: {run_id}  ->  {OUTPUT_DIR}/')


# ── Parse config ─────────────────────────────────────────
year_list   = [int(y.strip()) for y in AEF_YEARS.split(',')  if y.strip()]
season_list = [s.strip()      for s in S2_SEASONS.split(',') if s.strip()]

invalid = [s for s in season_list if s not in VALID_SEASONS]
if invalid:
    raise ValueError(f'Invalid seasons: {invalid}')
if not year_list and not season_list:
    raise ValueError('Set at least one of AEF_YEARS or S2_SEASONS.')

tqdm.write(f'AEF years:  {year_list}')
tqdm.write(f'S2 seasons: {season_list}')


# ── Load & collect feature columns ───────────────────────
df = pd.read_csv(TRAIN_CSV)

feature_cols = []
for yr in year_list:
    cols = [c for c in df.columns if c.startswith('A') and c.endswith(f'_{yr}')]
    tqdm.write(f'  AEF {yr}: {len(cols)} bands')
    feature_cols.extend(cols)
for s in season_list:
    cols = [c for c in df.columns if c.endswith(f'_{s}')]
    tqdm.write(f'  S2 _{s}: {len(cols)} bands')
    feature_cols.extend(cols)

if not feature_cols:
    raise ValueError('No feature columns found — check column names.')

meta_present = [c for c in META_COLS if c in df.columns]
X_full   = df[feature_cols].values
y        = 1 - df[TARGET_COL].values   # invert: 0=bare, 1=dense veg
grid_idx = df['system:index'].values if 'system:index' in df.columns else np.arange(len(df))

tqdm.write(f'\nTotal candidate features: {len(feature_cols)}')
tqdm.write(f'Dataset: {X_full.shape[0]} samples x {X_full.shape[1]} features')
tqdm.write(f'y range: {y.min():.4f} - {y.max():.4f}  (mean={y.mean():.4f})')


# ── Train / test split ────────────────────────────────────
# ── Train / test split — spatially evenly spaced ──────────
STEPS    = 4
step_bar = tqdm(total=STEPS, desc='Pipeline', unit='step',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                position=0)

def advance(label):
    step_bar.set_postfix_str(label)
    step_bar.update(1)

# Sort entire dataset by cell_y so spacing is spatial, not row-order
if 'cell_y' in df.columns:
    sort_order = df['cell_y'].argsort().values
    tqdm.write('  Split strategy: spatially evenly spaced (sorted by cell_y)')
else:
    sort_order = np.arange(len(X_full))
    tqdm.write('  WARNING: cell_y not found — evenly spaced by row order')

X_sorted   = X_full[sort_order]
y_sorted   = y[sort_order]
idx_sorted = sort_order  # maps back to original df rows

n_total = len(X_sorted)
n_test  = max(1, round(n_total * TEST_SIZE))

# Pick n_test indices evenly spaced across the sorted range
test_positions = np.round(np.linspace(0, n_total - 1, n_test)).astype(int)
train_positions = np.array([i for i in range(n_total) if i not in set(test_positions)])

X_train_full = X_sorted[train_positions]
X_test_full  = X_sorted[test_positions]
y_train      = y_sorted[train_positions]
y_test       = y_sorted[test_positions]
idx_train    = idx_sorted[train_positions]
idx_test     = idx_sorted[test_positions]

tqdm.write(f'  Train: {len(X_train_full)} | Test: {len(X_test_full)}')

# Sanity check — show cell_y coverage of each split
if 'cell_y' in df.columns:
    cy_train = df['cell_y'].values[idx_train]
    cy_test  = df['cell_y'].values[idx_test]
    tqdm.write(f'  Train cell_y: [{cy_train.min():.0f} – {cy_train.max():.0f}]')
    tqdm.write(f'  Test  cell_y: [{cy_test.min():.0f} – {cy_test.max():.0f}]')

# ── STEP 1 — Boruta on train only ────────────────────────
step_bar.set_description('Boruta')
np.random.seed(RANDOM_STATE)

rf_boruta = RandomForestRegressor(
    n_estimators     = N_ESTIMATORS,
    max_depth        = MAX_DEPTH,
    min_samples_leaf = MIN_SAMPLES_LEAF,
    max_features     = MAX_FEATURES,
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)
boruta = BorutaPy(
    estimator    = rf_boruta,
    n_estimators = 'auto',
    max_iter     = BORUTA_MAX_ITER,
    perc         = BORUTA_PERC,
    alpha        = BORUTA_ALPHA,
    verbose      = 0,
    random_state = RANDOM_STATE,
)

boruta_bar = tqdm(total=BORUTA_MAX_ITER, desc='  Boruta iter', unit='iter', leave=False,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} iter  [{elapsed}<{remaining}]',
                  position=1)
boruta.fit(X_train_full, y_train)
boruta_bar.update(BORUTA_MAX_ITER)
boruta_bar.close()

confirmed_mask = boruta.support_
tentative_mask = boruta.support_weak_
selected_mask  = confirmed_mask | tentative_mask if INCLUDE_TENTATIVE else confirmed_mask
selected_idx   = np.where(selected_mask)[0]
selected_cols  = [feature_cols[i] for i in selected_idx]

tqdm.write(f'  Confirmed: {confirmed_mask.sum()} | Tentative: {tentative_mask.sum()} | Selected: {len(selected_cols)}')
advance(f'Boruta done  selected={len(selected_cols)}')


# ── STEP 2 — Save train.csv & test.csv ───────────────────
step_bar.set_description('Saving CSVs')

def build_split_df(X_split, y_split, raw_idx):
    d = pd.DataFrame(X_split[:, selected_idx], columns=selected_cols)
    d[TARGET_COL] = 1 - y_split   # un-invert back to original veg_fraction
    if meta_present:
        meta = df[meta_present].iloc[raw_idx].reset_index(drop=True)
        d = pd.concat([meta, d], axis=1)
    return d

build_split_df(X_train_full, y_train, idx_train).to_csv(f'{OUTPUT_DIR}/train.csv', index=False)
build_split_df(X_test_full,  y_test,  idx_test ).to_csv(f'{OUTPUT_DIR}/test.csv',  index=False)
advance(f'CSVs saved  ({len(selected_cols)} features)')


# ── STEP 3 — Pearson + Spearman rankings ─────────────────
step_bar.set_description('Correlations')
pearson_scores  = np.zeros(len(feature_cols))
spearman_scores = np.zeros(len(feature_cols))

for i in tqdm(range(X_train_full.shape[1]),
              desc='  Correlations', unit='feat', leave=False,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} feat  [{elapsed}<{remaining}]',
              position=1):
    pearson_scores[i]  = abs(pearsonr( X_train_full[:, i], y_train)[0])
    spearman_scores[i] = abs(spearmanr(X_train_full[:, i], y_train)[0])

rankings = pd.DataFrame({
    'feature':          feature_cols,
    'pearson_abs':      pearson_scores,
    'spearman_abs':     spearman_scores,
    'boruta_ranking':   boruta.ranking_,
    'boruta_confirmed': confirmed_mask.astype(int),
    'boruta_tentative': tentative_mask.astype(int),
    'boruta_selected':  selected_mask.astype(int),
})
rankings = rankings.sort_values(['boruta_ranking', 'pearson_abs'], ascending=[True, False])
rankings.to_csv(f'{OUTPUT_DIR}/feature_rankings.csv', index=False)
advance('Correlations done')


# ── STEP 4 — Plot ─────────────────────────────────────────
step_bar.set_description('Plotting')

top       = rankings.head(TOP_N_PLOT).copy().sort_values('pearson_abs', ascending=True)
colors    = {'confirmed': '#2ecc71', 'tentative': '#f39c12', 'rejected': '#e74c3c'}
bar_color = top.apply(
    lambda r: colors['confirmed'] if r['boruta_confirmed']
              else (colors['tentative'] if r['boruta_tentative'] else colors['rejected']),
    axis=1).tolist()

fig    = plt.figure(figsize=(20, 13), facecolor='#1a1a2e')
gs     = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
tkw    = dict(color='white',   fontsize=11, fontweight='bold', pad=8)
lkw    = dict(color='#aaaaaa', fontsize=8)
tickkw = dict(colors='#aaaaaa', labelsize=7)

def sx(ax):
    ax.set_facecolor('#16213e')
    ax.spines[:].set_color('#333355')
    ax.tick_params(axis='x', **tickkw)
    ax.tick_params(axis='y', **tickkw)

ax1 = fig.add_subplot(gs[:, 0])
ax1.barh(top['feature'], top['pearson_abs'], color=bar_color, edgecolor='none', height=0.7)
ax1.set_title(f'Pearson |r|  (top {TOP_N_PLOT})', **tkw)
ax1.set_xlabel('|Pearson r|', **lkw)
sx(ax1)

ax2 = fig.add_subplot(gs[0, 1])
ax2.barh(top['feature'], top['spearman_abs'], color=bar_color, edgecolor='none', height=0.7)
ax2.set_title(f'Spearman |rho|  (top {TOP_N_PLOT})', **tkw)
ax2.set_xlabel('|Spearman rho|', **lkw)
sx(ax2)

ax3 = fig.add_subplot(gs[0, 2])
rank_counts = rankings['boruta_ranking'].value_counts().sort_index()
ax3.bar(rank_counts.index, rank_counts.values, color='#4a9eff', edgecolor='none')
ax3.set_title('Boruta Ranking Distribution', **tkw)
ax3.set_xlabel('Boruta rank (1 = confirmed)', **lkw)
ax3.set_ylabel('# features', **lkw)
sx(ax3)

ax4 = fig.add_subplot(gs[1, 1])
sc   = ax4.scatter(rankings['pearson_abs'], rankings['spearman_abs'],
                   c=rankings['boruta_ranking'], cmap='RdYlGn_r',
                   alpha=0.75, s=22, edgecolors='none')
cbar = plt.colorbar(sc, ax=ax4)
cbar.ax.tick_params(colors='#aaaaaa', labelsize=7)
cbar.set_label('Boruta rank', color='#aaaaaa', fontsize=8)
ax4.set_title('Pearson vs Spearman', **tkw)
ax4.set_xlabel('|Pearson r|',    **lkw)
ax4.set_ylabel('|Spearman rho|', **lkw)
sx(ax4)

ax5 = fig.add_subplot(gs[1, 2])
status_counts = {
    'Confirmed': int(confirmed_mask.sum()),
    'Tentative': int(tentative_mask.sum()),
    'Rejected':  int((~confirmed_mask & ~tentative_mask).sum()),
}
ax5.bar(status_counts.keys(), status_counts.values(),
        color=[colors['confirmed'], colors['tentative'], colors['rejected']],
        edgecolor='none', width=0.5)
ax5.set_title('Boruta Outcome', **tkw)
ax5.set_ylabel('# features', **lkw)
for i, (k, v) in enumerate(status_counts.items()):
    ax5.text(i, v + 0.5, str(v), ha='center', va='bottom',
             color='white', fontsize=9, fontweight='bold')
sx(ax5)

fig.legend(handles=[
    Patch(facecolor=colors['confirmed'], label=f'Confirmed ({confirmed_mask.sum()})'),
    Patch(facecolor=colors['tentative'], label=f'Tentative ({tentative_mask.sum()})'),
    Patch(facecolor=colors['rejected'],  label=f'Rejected ({(~confirmed_mask & ~tentative_mask).sum()})'),
], loc='lower center', ncol=3, framealpha=0.2, fontsize=9,
   labelcolor='white', bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    f'Feature Selection  |  Run {run_id}  |  '
    f'{X_train_full.shape[0]} train samples x {X_train_full.shape[1]} candidates  |  '
    f'Boruta selected: {len(selected_cols)}',
    color='white', fontsize=12, fontweight='bold', y=0.98
)
plt.savefig(f'{GRAPH_DIR}/feature_selection.png', dpi=150,
            bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
advance('Graph saved')

step_bar.close()


# ── Save JSON ─────────────────────────────────────────────
json.dump({
    'run_id':             run_id,
    'training_file':      TRAIN_CSV,
    'aef_years':          year_list,
    's2_seasons':         season_list,
    'n_samples':          int(X_full.shape[0]),
    'n_train':            int(len(X_train_full)),
    'n_test':             int(len(X_test_full)),
    'test_size':          TEST_SIZE,
    'random_state':       RANDOM_STATE,
    'include_tentative':  INCLUDE_TENTATIVE,
    'n_candidates':       len(feature_cols),
    'n_selected':         len(selected_cols),
    'selected_features':  selected_cols,
    'boruta': {
        'max_iter':    BORUTA_MAX_ITER,
        'perc':        BORUTA_PERC,
        'alpha':       BORUTA_ALPHA,
        'n_confirmed': int(confirmed_mask.sum()),
        'n_tentative': int(tentative_mask.sum()),
        'n_selected':  len(selected_cols),
    },
    'rf_params': {
        'n_estimators':     N_ESTIMATORS,
        'max_depth':        MAX_DEPTH,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'max_features':     MAX_FEATURES,
    },
}, open(f'{OUTPUT_DIR}/boruta_results.json', 'w'), indent=2)

tqdm.write(f'\nRun {run_id} complete  ->  {OUTPUT_DIR}/')
tqdm.write(f'  +-- train.csv             ({len(X_train_full)} rows x {len(selected_cols)} features)')
tqdm.write(f'  +-- test.csv              ({len(X_test_full)} rows x {len(selected_cols)} features)')
tqdm.write(f'  +-- feature_rankings.csv  (all {len(feature_cols)} candidates ranked)')
tqdm.write(f'  +-- boruta_results.json')
tqdm.write(f'  +-- graph/feature_selection.png')