"""Journal-style figures. Okabe-Ito palette, minimal ink, 300 dpi."""
import pandas as pd, numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':0.8,'figure.dpi':300})
BLUE, ORANGE, GREY, VERM = '#0072B2', '#E69F00', '#7f7f7f', '#D55E00'

h12 = json.load(open('h1h2_results.json'))
h3  = json.load(open('h3_results.json'))
h4  = json.load(open('h4_results.json'))
rob = json.load(open('robustness_results.json'))
LAB = {'worry':'Climate worry','attrib':'Human attribution','resp':'Personal responsibility'}

# ---------- Figure 1: forest plot, sexism -> worry, per country ----------
t = pd.read_csv('twostep_worry.csv').sort_values('b').reset_index(drop=True)
pool = rob['twostep']['worry']
fig, ax = plt.subplots(figsize=(5.2, 6.6))
y = np.arange(len(t))
ax.errorbar(t['b'], y, xerr=1.96*t['se'], fmt='o', color=BLUE, ecolor=BLUE,
            elinewidth=1, ms=3.2, capsize=0)
ax.axvline(0, color='k', lw=0.8)
ax.set_yticks(y); ax.set_yticklabels(t['cntry'], fontsize=7.5)
# pooled diamond
pb, pse = pool['pooled_b'], pool['pooled_se']
dy = len(t) + 1.0
ax.fill([pb-1.96*pse, pb, pb+1.96*pse, pb], [dy, dy+0.45, dy, dy-0.45], color=VERM)
ax.text(pb, dy+0.9, f'RE-pooled β = {pb:.2f} (95% CI {pb-1.96*pse:.2f}, {pb+1.96*pse:.2f}); I² = {pool["I2"]:.0f}%',
        ha='center', fontsize=8)
ax.set_ylim(-1, dy+2.2)
ax.set_xlabel('β of sexism on climate worry (SD per SD), country-specific OLS')
ax.set_ylabel('')
fig.tight_layout(); fig.savefig('fig1_forest_worry.png', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 2: sexism coefficient across outcomes and blocks ----------
fig, ax = plt.subplots(figsize=(5.4, 2.9))
blocks = ['M2a','M2b','M3']; boff = {'M2a':-0.18,'M2b':0.0,'M3':0.18}
cols = {'M2a':GREY,'M2b':BLUE,'M3':ORANGE}
for i, o in enumerate(['worry','attrib','resp']):
    for b in blocks:
        est, se, p = h12[o][b]['sexism']
        ax.errorbar(i+boff[b], est, yerr=1.96*se, fmt='o', ms=4.5, color=cols[b], elinewidth=1.4)
ax.axhline(0, color='k', lw=0.8)
ax.axhline(-0.05, color=GREY, lw=0.8, ls=':')
ax.text(2.42, -0.052, 'benchmark −0.05', fontsize=7, color=GREY, va='top')
ax.set_xticks([0,1,2]); ax.set_xticklabels([LAB[o] for o in ['worry','attrib','resp']])
ax.set_ylabel('β of sexism (SD per SD)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=cols[b],label=l) for b,l in
                   [('M2a','+ demographics (M2a)'),('M2b','+ gendered identity (M2b)'),('M3','+ left–right (M3)')]],
          frameon=False, fontsize=7.5, loc='upper right', bbox_to_anchor=(1.0, 1.12), ncols=3)
fig.tight_layout(); fig.savefig('fig2_sexism_blocks.png', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 3: gender-gap decomposition ----------
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for i, o in enumerate(['worry','attrib','resp']):
    d = h3[o]
    ax.bar(i-0.16, d['gap_M1'], width=0.3, color=GREY, label='Before adjustment (M1)' if i==0 else None)
    ax.bar(i+0.16, d['gap_M2'], width=0.3, color=BLUE, label='After sexism + gendered identity (M2)' if i==0 else None)
    share = 100*d['share_explained']; lo, hi = 100*d['share_ci'][0], 100*d['share_ci'][1]
    ax.text(i, max(d['gap_M1'], d['gap_M2'])+0.012, f'{share:.0f}% explained\n[{lo:.0f}, {hi:.0f}]',
            ha='center', fontsize=7.5)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks([0,1,2]); ax.set_xticklabels([LAB[o] for o in ['worry','attrib','resp']])
ax.set_ylabel('Female–male gap (SD units)')
ax.set_ylim(top=max(h3[o]['gap_M1'] for o in h3)+0.09)
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig('fig3_decomposition.png', bbox_inches='tight'); plt.close(fig)

# ---------- Figure 4: H4 marginal slopes ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
ax = axes[0]
for i, o in enumerate(['worry','attrib','resp']):
    m = h4[o]
    bm, sem = m['sexism_men'][0], m['sexism_men'][1]
    bw, sew = m['sexism_women'][0], m['sexism_women'][1]
    ax.errorbar(i-0.12, bm, yerr=1.96*sem, fmt='s', color=BLUE, ms=4.5, label='Men' if i==0 else None)
    ax.errorbar(i+0.12, bw, yerr=1.96*sew, fmt='o', color=ORANGE, ms=4.5, label='Women' if i==0 else None)
    pint = m['interaction'][2]
    ax.text(i, -0.155, ('interaction\np<0.001' if pint<0.001 else f'interaction\np={pint:.2f}'), ha='center', fontsize=7, color=GREY)
ax.axhline(0, color='k', lw=0.8); ax.set_xticks([0,1,2]); ax.set_ylim(-0.175, 0.01)
ax.set_xticklabels(['Worry','Attribution','Responsibility'], fontsize=8)
ax.set_ylabel('β of sexism (SD per SD)')
ax.legend(frameon=False, fontsize=7.5, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncols=2)
ax.set_title('(a) By gender', fontsize=9, pad=26)
ax = axes[1]
for i, o in enumerate(['worry','attrib','resp']):
    m = h4[o]
    lo, selo = m['men_slope_centr_lo']; hi, sehi = m['men_slope_centr_hi']
    ax.errorbar(i-0.12, lo, yerr=1.96*selo, fmt='^', color=GREY, ms=4.5, label='Low centrality (−1 SD)' if i==0 else None)
    ax.errorbar(i+0.12, hi, yerr=1.96*sehi, fmt='v', color=VERM, ms=4.5, label='High centrality (+1 SD)' if i==0 else None)
    pint = m['men_inter_centr'][2]
    ax.text(i, -0.155, ('interaction\np<0.001' if pint<0.001 else f'interaction\np={pint:.2f}'), ha='center', fontsize=7, color=GREY)
ax.axhline(0, color='k', lw=0.8); ax.set_xticks([0,1,2]); ax.set_ylim(-0.175, 0.01)
ax.set_xticklabels(['Worry','Attribution','Responsibility'], fontsize=8)
ax.set_title('(b) Within men, by gender-identity centrality', fontsize=9, pad=26)
ax.legend(frameon=False, fontsize=7.5, loc='lower left', bbox_to_anchor=(0.0, 1.02), ncols=2)
fig.tight_layout(); fig.savefig('fig4_interactions.png', bbox_inches='tight'); plt.close(fig)
print('figures done')
