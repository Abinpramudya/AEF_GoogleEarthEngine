import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os, glob


# ═══════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════
OUTPUT_ROOT = 'output/rf'
GT_CSV      = 'data/training_data_bistugliu.csv'
# ═══════════════════════════════════════════════════════════


PRED_COL_CANDIDATES = [
    'veg_fraction_pred',
    'ensemble_pred',
    'stage2_final_pred',
    'stage2_pred',
    'pred'
]


def get_pred_col(df):
    for col in PRED_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No prediction column found! Columns: {list(df.columns)}")


def build_grid(df, value_col):
    d = df.copy()
    idx = d['system:index'].str.extract(r'^(\d+),(\d+)')
    d['idx_x'] = idx[0].astype(int)
    d['idx_y'] = idx[1].astype(int)
    d['col'] = d['idx_x'].rank(method='dense').astype(int) - 1
    d['row'] = d['idx_y'].rank(method='dense').astype(int) - 1
    return d, d.pivot(index='row', columns='col', values=value_col).values


def save_map(z, title, path, cmap='gray_r', vmin=0, vmax=1, cbar_label='Veg Fraction'):
    nrows, ncols = z.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.03, pad=0.04)
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('Col (W → E)'); ax.set_ylabel('Row (S → N)')
    ax.set_xticks(range(ncols)); ax.set_yticks(range(nrows))
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def save_annotated(z, title, path, cmap='gray_r', vmin=0, vmax=1, fmt='.2f'):
    nrows, ncols = z.shape
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(z, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal', interpolation='nearest')
    for r in range(nrows):
        for c in range(ncols):
            val = z[r, c]
            if np.isnan(val): continue
            ax.text(c, r, f'{val:{fmt}}', ha='center', va='center',
                    fontsize=7.5, color='white' if val > 0.5 else 'black')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('Col (W → E)'); ax.set_ylabel('Row (S → N)')
    ax.set_xticks(range(ncols)); ax.set_xticklabels([f'C{i}' for i in range(ncols)], fontsize=8)
    ax.set_yticks(range(nrows)); ax.set_yticklabels([f'R{i}' for i in range(nrows)], fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def save_residual(z_res, title, path):
    nrows, ncols = z_res.shape
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(z_res, cmap='RdBu_r', vmin=-0.5, vmax=0.5,
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Residual (Pred − GT)', fraction=0.03, pad=0.04)
    for r in range(nrows):
        for c in range(ncols):
            val = z_res[r, c]
            if np.isnan(val): continue
            ax.text(c, r, f'{val:+.2f}', ha='center', va='center',
                    fontsize=7.5, color='white' if abs(val) > 0.25 else 'black')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('Col (W → E)'); ax.set_ylabel('Row (S → N)')
    ax.set_xticks(range(ncols)); ax.set_xticklabels([f'C{i}' for i in range(ncols)], fontsize=8)
    ax.set_yticks(range(nrows)); ax.set_yticklabels([f'R{i}' for i in range(nrows)], fontsize=8)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def save_error_pct(z_gt, z_pred, title, path):
    denom = np.where(np.abs(z_gt) < 0.01, 0.01, np.abs(z_gt))
    z_ape = np.abs(z_pred - z_gt) / denom * 100
    z_ape = np.clip(z_ape, 0, 100)
    nrows, ncols = z_ape.shape
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(z_ape, cmap='YlOrRd', vmin=0, vmax=100,
                   aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Absolute % Error (clipped at 100%)',
                 fraction=0.03, pad=0.04)
    for r in range(nrows):
        for c in range(ncols):
            val = z_ape[r, c]
            if np.isnan(val): continue
            ax.text(c, r, f'{val:.0f}%', ha='center', va='center',
                    fontsize=7.5, color='white' if val > 60 else 'black')
    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel('Col (W → E)'); ax.set_ylabel('Row (S → N)')
    ax.set_xticks(range(ncols)); ax.set_xticklabels([f'C{i}' for i in range(ncols)], fontsize=8)
    ax.set_yticks(range(nrows)); ax.set_yticklabels([f'R{i}' for i in range(nrows)], fontsize=8)
    mean_ape = np.nanmean(z_ape)
    ax.text(0.01, 0.99, f'MAPE = {mean_ape:.1f}%',
            transform=ax.transAxes, fontsize=10, color='black', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def save_scatter(merged, pred, pred_col, run_name, path):
    test_ids = set(pred[pred['split'] == 'test']['system:index']) if 'split' in pred.columns else set()
    colors   = ['red' if sid in test_ids else 'steelblue' for sid in merged['system:index']]
    fig, ax  = plt.subplots(figsize=(6, 6))
    ax.scatter(merged['veg_gt'], merged[pred_col],
               c=colors, alpha=0.7, edgecolors='k', linewidths=0.3, s=50)
    lim = [0, max(merged['veg_gt'].max(), merged[pred_col].max()) + 0.05]
    ax.plot(lim, lim, 'k--', linewidth=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Ground Truth Veg Fraction')
    ax.set_ylabel('Predicted Veg Fraction')
    ax.set_title(f'GT vs Predicted — {run_name}', fontsize=11)
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue', markersize=8, label='Train'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',       markersize=8, label='Test'),
        Line2D([0],[0], linestyle='--', color='k', label='Perfect fit')
    ]
    ax.legend(handles=handles)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


# ── Load ground truth ──────────────────────────────────────
gt_raw      = pd.read_csv(GT_CSV)
gt_df, z_gt = build_grid(gt_raw.assign(veg_gt=1 - gt_raw['veg_fraction']), 'veg_gt')


# ── Scour output/rf/** for predictions.csv ─────────────────
pred_files = sorted(glob.glob(os.path.join(OUTPUT_ROOT, '**', 'prediction.csv'), recursive=True))
print(f"Found {len(pred_files)} run(s) under '{OUTPUT_ROOT}/'\n")

if len(pred_files) == 0:
    print("  No predictions.csv files found. Check OUTPUT_ROOT path.")
    exit()


for pred_path in pred_files:
    run_dir    = os.path.dirname(pred_path)
    run_name   = os.path.relpath(run_dir, OUTPUT_ROOT)   # e.g. "1", "2"
    graphs_dir = os.path.join(run_dir, 'graph')

    print(f"── rf/{run_name}  [{pred_path}]")

    # ── Skip / re-run prompt ───────────────────────────────
    if os.path.isdir(graphs_dir) and len(os.listdir(graphs_dir)) > 0:
        ans = input(f"   graphs/ already exists. Re-run? (y/n): ").strip().lower()
        if ans != 'y':
            print(f"   → Skipped.\n")
            continue
        print(f"   → Re-running...")

    try:
        pred     = pd.read_csv(pred_path)
        pred_col = get_pred_col(pred)
        pred[pred_col] = pred[pred_col]
        print(f"   Using prediction column: '{pred_col}'")

        pred_d, z_pred = build_grid(pred, pred_col)

        merged = gt_df[['row', 'col', 'system:index', 'veg_gt']].merge(
            pred_d[['row', 'col', pred_col]], on=['row', 'col'])
        merged['residual'] = merged[pred_col] - merged['veg_gt']

        z_res          = merged.pivot(index='row', columns='col', values='residual').values
        z_pred_aligned = merged.pivot(index='row', columns='col', values=pred_col).values
        z_gt_aligned   = merged.pivot(index='row', columns='col', values='veg_gt').values

        os.makedirs(graphs_dir, exist_ok=True)

        save_map      (z_gt,           'Ground Truth — Veg Fraction',              os.path.join(graphs_dir, '1_gt_map.png'))
        save_annotated(z_gt,           'Ground Truth — Annotated Grid',            os.path.join(graphs_dir, '1_gt_annotated.png'))
        save_map      (z_pred,         f'Predicted — rf/{run_name}',               os.path.join(graphs_dir, '2_pred_map.png'))
        save_annotated(z_pred,         f'Predicted — rf/{run_name}',               os.path.join(graphs_dir, '2_pred_annotated.png'))
        save_residual (z_res,          f'Residual (Pred − GT) — rf/{run_name}',    os.path.join(graphs_dir, '3_residual.png'))
        save_scatter  (merged, pred, pred_col, f'rf/{run_name}',                   os.path.join(graphs_dir, '4_scatter.png'))
        save_error_pct(z_gt_aligned, z_pred_aligned,
                                       f'Absolute % Error — rf/{run_name}',        os.path.join(graphs_dir, '5_error_pct.png'))

        print(f"   → 7 images saved in {graphs_dir}/\n")

    except Exception as e:
        print(f"   ⚠️  SKIPPED rf/{run_name}: {e}\n")


print("✅ Done.")