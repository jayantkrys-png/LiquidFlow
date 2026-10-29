"""Report figures. Reads artifacts/ and data/processed/; writes PNGs to artifacts/figures/."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'artifacts/figures'; OUT.mkdir(parents=True, exist_ok=True)
for f in (BASE / 'assets/fonts').glob('*.ttf'):
    if f.name.startswith(('Inter-', 'JetBrains')): fm.fontManager.addfont(str(f))
SANS, MONO = 'Inter', 'JetBrains Mono'
NAVY, BLUE, ICE, TEXT, MUTED = '#142C46', '#1769B5', '#EEF5FC', '#243B53', '#52677D'
GRID, B80, B95, OUTC, OUTD = '#E2E9F1', '#A9C8EA', '#DCE8F6', '#D9822B', '#A8541A'
plt.rcParams.update({
    'font.family': SANS, 'font.size': 8, 'text.color': TEXT, 'axes.labelcolor': MUTED,
    'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelsize': 7.6, 'ytick.labelsize': 7.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID,
    'grid.linewidth': .6, 'axes.axisbelow': True, 'xtick.major.size': 0, 'ytick.major.size': 0,
    'savefig.dpi': 300, 'figure.dpi': 100, 'svg.fonttype': 'none'})
W = 511 / 72  # full text width in inches, so chart type sizes are true point sizes on the page

def mono(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_fontfamily(MONO)
def kfmt(v, _=None):
    return '0' if abs(v) < 1 else f'{v/1000:+,.0f}k'.replace('+', '') if v > 0 else f'−{abs(v)/1000:,.0f}k'
def save(fig, name):
    fig_path = OUT / f'{name}.png'
    if not fig_path.exists():
        fig.savefig(fig_path, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    plt.close(fig)

flows = pd.read_csv(BASE / 'data/processed/amfi_flows.csv').sort_values(['category', 'month'])
flows['prev_aum'] = flows.groupby('category').aum_crore.shift(1)
flows['pct'] = flows.net_flow_crore / flows.prev_aum * 100
flows['date'] = pd.to_datetime(flows.month)
fc = pd.read_csv(BASE / 'artifacts/forecasts.csv'); fc['date'] = pd.to_datetime(fc.target_month)
pm = pd.read_csv(BASE / 'artifacts/policy_months.csv'); pm['date'] = pd.to_datetime(pm.target_month)
mo = pd.read_csv(BASE / 'artifacts/metrics_overall.csv')
ab = pd.read_csv(BASE / 'artifacts/driver_ablations.csv')
ps = pd.read_csv(BASE / 'artifacts/policy_sanity_checks.csv')
facts = {}

# A. Cover: seven years of liquid-fund flows, March bars called out.
g = flows[flows.category == 'Liquid']
fig, ax = plt.subplots(figsize=(W, 2.75))
colr = [OUTD if (v < 0 and d.month == 3) else OUTC if v < 0 else BLUE for v, d in zip(g.net_flow_crore, g.date)]
ax.bar(g.date, g.net_flow_crore, width=22, color=colr, linewidth=0)
ax.axhline(0, color=NAVY, lw=.8)
ax.yaxis.set_major_formatter(FuncFormatter(kfmt)); ax.set_ylim(-190000, 190000)
ax.set_xticks(pd.to_datetime([f'{y}-01-01' for y in range(2020, 2027)]))
ax.set_xticklabels([str(y) for y in range(2020, 2027)])
for d, v in zip(g.date, g.net_flow_crore):
    if d.month == 3: ax.text(d, v - 9000, 'Mar', ha='center', va='top', fontsize=6, color=OUTD, fontfamily=MONO)
ax.set_ylabel('₹ crore, net flow per month', fontsize=7)
mono(ax); save(fig, 'cover_flows')

# B. Recommended buffer by month (q95 positive-outflow buffer, % of known AUM).
live = fc[fc.split == 'live'].sort_values('horizon')
fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), sharey=True, gridspec_kw={'wspace': .12})
labels = ['Oct', 'Nov', 'Dec', 'Jan']
for ax, cat in zip(axes, ['Liquid', 'Overnight']):
    r = live[live.category == cat]
    x = np.arange(4)
    cols = [OUTC if (cat == 'Liquid' and m == 12) else BLUE for m in r.date.dt.month]
    ax.bar(x, r.buffer_aum_pct, width=.62, color=cols)
    for xi, p, b in zip(x, r.buffer_aum_pct, r.outflow95_buffer_crore):
        ax.text(xi, p + 1.2, f'{p:.1f}%', ha='center', va='bottom', fontsize=8, fontfamily=MONO, color=NAVY, fontweight='bold')
        ax.text(xi, p + 5.6, f'{b:,.0f} cr', ha='center', va='bottom', fontsize=6.3, fontfamily=MONO, color=MUTED)
    ax.set_xticks(x, [f'{l}\n{d.year}' for l, d in zip(labels, r.date)])
    ax.set_ylim(0, 58); ax.set_yticks([0, 20, 40]); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_title(f'{cat} funds', loc='left', fontsize=9, fontweight='bold', color=NAVY, pad=6)
    mono(ax)
save(fig, 'buffer_outlook')

# C. Calendar heatmap: liquid net flow as % of previous month AUM.
h = g.dropna(subset=['pct']).copy(); h['y'] = h.date.dt.year; h['m'] = h.date.dt.month
piv = h.pivot(index='y', columns='m', values='pct').reindex(columns=range(1, 13))
cmap = LinearSegmentedColormap.from_list('flow', [OUTD, OUTC, '#FBE3CC', '#FFFFFF', '#D7E6F7', '#5B9AD6', BLUE])
fig, ax = plt.subplots(figsize=(W, 2.75))
ax.imshow(piv.values, cmap=cmap, norm=TwoSlopeNorm(0, -32, 32), aspect='auto')
for i, y in enumerate(piv.index):
    for j in range(12):
        v = piv.values[i, j]
        if np.isnan(v): continue
        ax.text(j, i, ('0' if round(v)==0 else f'{v:+.0f}'.replace('-', '−')), ha='center', va='center', fontsize=6.6, fontfamily=MONO,
                color='white' if abs(v) > 19 else TEXT)
ax.set_xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
ax.set_yticks(range(len(piv.index)), piv.index); ax.grid(False)
ax.xaxis.tick_top()
for j in [2, 5, 8, 11]:
    ax.add_patch(plt.Rectangle((j - .5, -.5), 1, len(piv.index), fill=False, ec=NAVY, lw=1.1))
neg = h.assign(n=h.pct < 0).groupby('m').n.agg(['sum', 'size'])
for j in range(12):
    s, n = neg.loc[j + 1]
    ax.text(j, len(piv.index) - .2, f'{s}/{n}', ha='center', va='top', fontsize=6.6, fontfamily=MONO,
            color=OUTD if s / n >= .8 else MUTED, fontweight='bold' if s / n >= .8 else 'normal')
ax.text(-.62, len(piv.index) - .2, 'Outflow years', ha='right', va='top', fontsize=6.6, color=MUTED)
ax.set_ylim(len(piv.index) + .5, -.5)
for s in ax.spines.values(): s.set_visible(False)
mono(ax)
for t in ax.get_xticklabels(): t.set_fontfamily(SANS); t.set_fontsize(7.2)
save(fig, 'calendar_heatmap')
facts['liquid_outflow_years_by_month'] = {int(k): [int(a), int(b)] for k, (a, b) in neg.iterrows()}
mar = h[h.m == 3][['month', 'net_flow_crore', 'prev_aum', 'pct']]
facts['liquid_march'] = mar.round(2).to_dict('records')

# D. Drivers: change in development error after removing each feature group.
names = {'calendar': 'Calendar and tax dates', 'rates': 'Interest rates', 'equity': 'Equity market', 'liquidity': 'Banking liquidity'}
a = ab.iloc[::-1]
fig, ax = plt.subplots(figsize=(3.45, 1.75))
ax.barh([names[x] for x in a.group_removed], a.change_vs_full_error_pct_origin_aum,
        color=[BLUE if v > 0 else '#B8C4D0' for v in a.change_vs_full_error_pct_origin_aum], height=.56)
for yi, v in enumerate(a.change_vs_full_error_pct_origin_aum):
    ax.text(v + (.04 if v > 0 else -.04), yi, f'{v:+.2f}'.replace('-', '−'), va='center', ha='left' if v > 0 else 'right',
            fontsize=8, fontfamily=MONO, color=NAVY)
ax.axvline(0, color=NAVY, lw=.8); ax.set_xlim(-.45, 1.45); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
ax.set_xticks([0, .5, 1]); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.1f}' if v else '0'))
ax.set_xlabel('Extra error when removed, % of AUM', fontsize=7.4)
mono(ax)
for t in ax.get_yticklabels(): t.set_fontfamily(SANS); t.set_fontsize(8.2); t.set_color(TEXT)
save(fig, 'drivers')

# E/F. Actual history next to discrete forecast ranges.
for cat in ['Liquid', 'Overnight']:
    hist = flows[(flows.category == cat) & (flows.date >= '2024-10-01')]
    r = live[live.category == cat]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(W, 2.95), sharey=True, gridspec_kw={'width_ratios': [1.8, 1], 'wspace': .05})
    ax.plot(hist.date, hist.net_flow_crore, color=NAVY, lw=1.2, zorder=2)
    ax.scatter(hist.date, hist.net_flow_crore, s=11, color=[OUTC if v < 0 else NAVY for v in hist.net_flow_crore], zorder=3, lw=0)
    for a_ in (ax, bx): a_.axhline(0, color=NAVY, lw=.7, zorder=1); a_.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ticks = pd.to_datetime(['2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-01'])
    ax.set_xticks(ticks, [t.strftime('%b %y') for t in ticks])
    ax.set_xlim(hist.date.min() - pd.Timedelta(days=20), hist.date.max() + pd.Timedelta(days=20))
    ax.set_ylabel('₹ crore, net flow per month', fontsize=7)
    ax.set_title('Last 24 months, actual', loc='left', fontsize=8, fontweight='bold', color=NAVY, pad=7)
    x = np.arange(4)
    bx.set_facecolor('#F6F9FC')
    bx.vlines(x, r.q025, r.q975, color=B95, lw=10, zorder=2)
    bx.vlines(x, r.q100, r.q900, color=B80, lw=10, zorder=3)
    bx.scatter(x, r.point, s=34, color='white', ec=BLUE, lw=1.6, zorder=4)
    bx.scatter(x, r.q050, marker='_', s=130, color=OUTD, lw=1.6, zorder=4)
    for xi, p in zip(x, r.point):
        bx.text(xi + .15, p, f'{p:,.0f}'.replace('-', '−'), va='center', ha='left', fontsize=6.6, fontfamily=MONO, color=BLUE, fontweight='bold', path_effects=[pe.withStroke(linewidth=2.4, foreground='white')])
    bx.set_xticks(x, [d.strftime('%b %y') for d in r.date]); bx.set_xlim(-.55, 3.75)
    bx.set_title('Next 4 months, forecast', loc='left', fontsize=8, fontweight='bold', color=BLUE, pad=7)
    bx.spines['left'].set_visible(True); bx.spines['left'].set_color(GRID)
    for a_ in (ax, bx):
        a_.tick_params(axis='x', pad=7)
        mono(a_)
        for t in a_.get_xticklabels(): t.set_fontfamily(SANS)
    save(fig, f'fan_{cat.lower()}')

# G. Backtest: actual outflow against the two buffers, horizon 1.
fig, axes = plt.subplots(2, 1, figsize=(W, 3.9), sharex=True, gridspec_kw={'hspace': .42})
for ax, cat in zip(axes, ['Liquid', 'Overnight']):
    r = pm[pm.category == cat].sort_values('date')
    miss = r.outflow_crore > r.q95_buffer_crore + 1e-9
    ax.bar(r.date, r.outflow_crore, width=20, color=[OUTC if m else '#C9D5E2' for m in miss], lw=0, label='Actual net outflow')
    ax.step(r.date, r.q95_buffer_crore, where='mid', color=BLUE, lw=1.5, label='LiquidFlow buffer')
    ax.step(r.date, r.naive_buffer_crore, where='mid', color=NAVY, lw=1, ls=(0, (2, 1.6)), label="Last year's outflow rule")
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    nm = int(miss.sum()); nn = int((r.outflow_crore > r.naive_buffer_crore + 1e-9).sum())
    ax.set_title(f'{cat} funds   ', loc='left', fontsize=10, fontweight='bold', color=NAVY, pad=5)
    ax.text(1, 1.04, f'LiquidFlow short {nm}/24   ·   last-year rule short {nn}/24', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=7.6, fontfamily=MONO, color=MUTED)
    mono(ax)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower left', frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(0.06, .94), handlelength=1.8)
from matplotlib.patches import Patch

fig.legend([Patch(color=OUTC)], ['Outflow larger than the buffer'], loc='lower right', frameon=False, fontsize=8, bbox_to_anchor=(1.0, .94), handlelength=1.2)
ticks = pd.to_datetime(['2024-10-01', '2025-01-01', '2025-04-01', '2025-07-01', '2025-10-01', '2026-01-01', '2026-04-01', '2026-07-01'])
axes[1].set_xticks(ticks, [t.strftime('%b %y') for t in ticks])
for t in axes[1].get_xticklabels(): t.set_fontfamily(SANS)
save(fig, 'backtest')
facts['backtest_misses'] = {c: int((pm[pm.category == c].outflow_crore > pm[pm.category == c].q95_buffer_crore + 1e-9).sum()) for c in ['Liquid', 'Overnight']}

# H. Policy cost comparison under the brief's literal penalty.
lit = ps[ps.interpretation == 'literal_3_rupees_per_rupee'].set_index('policy')
order = [('prior_year_outflow', "Last year's same-month outflow"), ('selected_model_q95', 'LiquidFlow buffer'), ('seasonal_naive_q95', 'Seasonal 95% band')]
fig, (ax, tx) = plt.subplots(1, 2, figsize=(W, 1.65), sharey=True, gridspec_kw={'width_ratios': [2.9, 1], 'wspace': .02})
ys = np.arange(3)[::-1]
for yv, (k, lab_) in zip(ys, order):
    v = lit.loc[k]
    ax.barh(yv, v.cost_crore, color=NAVY if k == 'prior_year_outflow' else BLUE if k == 'selected_model_q95' else '#7FA9D6', height=.55)
    ax.text(v.cost_crore + 14000, yv, f'{v.cost_crore:,.0f}', va='center', fontsize=8.2, fontfamily=MONO, color=NAVY, fontweight='bold')
    tx.text(.28, yv, f'{int(v.shortfall_category_months)}', va='center', ha='right', fontsize=8.2, fontfamily=MONO, color=OUTD if v.shortfall_category_months > 5 else TEXT, fontweight='bold')
    tx.text(.97, yv, f'{v.mean_buffer_aum_pct:.1f}%', va='center', ha='right', fontsize=8.2, fontfamily=MONO, color=TEXT)
tx.text(.28, 2.62, 'months short', ha='right', fontsize=7.2, color=MUTED); tx.text(.97, 2.62, 'avg. buffer', ha='right', fontsize=7.2, color=MUTED)
tx.set_xlim(0, 1); tx.axis('off')
ax.set_yticks(ys, [l for _, l in order]); ax.set_xlim(0, 1050000); ax.grid(axis='x'); ax.grid(axis='y', visible=False)
ax.xaxis.set_major_formatter(FuncFormatter(kfmt)); ax.set_xticks([0, 250000, 500000, 750000, 1000000])
ax.set_ylim(-.6, 2.9)
mono(ax)
for t in ax.get_yticklabels(): t.set_fontfamily(SANS); t.set_fontsize(8.4); t.set_color(TEXT)
save(fig, 'policy_cost')

# I. Model comparison.
cands = json.loads((BASE / 'protocol.json').read_text())['candidates'] if (BASE / 'protocol.json').exists() else None
lab = {'seasonal_naive': 'Seasonal naive', 'seasonal_scaled': 'Scaled seasonal', 'sarimax': 'SARIMAX', 'ridge_core': 'Ridge',
       'ridge_full': 'Ridge + macro', 'catboost_core': 'CatBoost', 'catboost_full': 'CatBoost + macro'}
t = mo[(mo.split == 'test') & mo.model.isin(lab)].set_index('model').loc[list(lab)]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.55), sharey=True, gridspec_kw={'wspace': .08})
yy = np.arange(len(t))[::-1]
specs = [('aum_scaled_mae_pct', 'Average error, % of AUM', 'lower is better', '{:.1f}%', (11.8, 14.6), [12, 13, 14]),
         ('mape_pct', 'Average % error (MAPE)', 'lower is better', '{:.0f}%', (100, 345), [100, 200, 300]),
         ('coverage95_pct', 'Hit rate of 95% range', 'closer to 95% is better', '{:.0f}%', (78, 98), [80, 85, 90, 95])]
for ax, (colm, title, sub, fmtv, lim, tk) in zip(axes, specs):
    ax.set_xticks(tk)
    vals = t[colm].values
    ax.hlines(yy, lim[0], vals, color=GRID, lw=1.2)
    ax.scatter(vals, yy, s=30, color=[BLUE if m == 'sarimax' else NAVY if m == 'seasonal_naive' else '#9FB3C8' for m in t.index], zorder=3)
    left = colm == 'coverage95_pct'
    for yv, v in zip(yy, vals): ax.text(v + (lim[1] - lim[0]) * (-.04 if left else .04), yv, fmtv.format(v), va='center', ha='right' if left else 'left', fontsize=7, fontfamily=MONO, color=TEXT)
    if colm == 'coverage95_pct': ax.axvline(95, color=OUTC, lw=1, ls=(0, (2, 1.5))); ax.text(95.4, yy[0] + .1, 'target', fontsize=6.6, color=OUTD, va='center')
    ax.set_xlim(*lim); ax.grid(axis='y', visible=False); ax.grid(axis='x')
    ax.set_title(title, loc='left', fontsize=8.4, fontweight='bold', color=NAVY, pad=12)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=7, color=MUTED)
    ax.tick_params(axis='x', labelsize=7); mono(ax)
axes[0].set_yticks(yy, [lab[m] for m in t.index])
for tl, m in zip(axes[0].get_yticklabels(), t.index):
    tl.set_fontfamily(SANS); tl.set_fontsize(8); tl.set_color(BLUE if m == 'sarimax' else TEXT); tl.set_fontweight('bold' if m in ('sarimax',) else 'normal')
save(fig, 'models')

facts_path = OUT / 'facts.json'
new_facts = json.dumps(facts, indent=1, default=str)
if not facts_path.exists() or facts_path.read_text() != new_facts:
    facts_path.write_text(new_facts)
print('ok', sorted(p.name for p in OUT.glob('*.png')))
