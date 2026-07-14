"""Figure 5: country-level moderation of the sexism-worry slope (backlash vs material)."""
import pandas as pd, numpy as np, json
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':0.8,'figure.dpi':300})
BLUE, VERM, GREY = '#0072B2', '#D55E00', '#7f7f7f'

mods = pd.DataFrame(json.load(open('country_moderators.json'))).T
mods.index.name='cntry'; mods=mods.reset_index()
for c in ['eige2024','fossil_share']: mods[c]=pd.to_numeric(mods[c], errors='coerce')

def meta_fit(y, se, x):
    k=len(y); X1=np.column_stack([np.ones(k),x]); p=2
    w=1/se**2; W=np.diag(w)
    XtWX_inv=np.linalg.inv(X1.T@W@X1); b_fe=XtWX_inv@X1.T@W@y
    r=y-X1@b_fe; Q=float(r.T@W@r)
    trP=np.trace(W@X1@XtWX_inv@X1.T@W)
    tau2=max(0.0,(Q-(k-p))/(w.sum()-trP))
    ws=1/(se**2+tau2); Ws=np.diag(ws)
    XtWX_inv_s=np.linalg.inv(X1.T@Ws@X1); b=XtWX_inv_s@X1.T@Ws@y
    r=y-X1@b; s2=float(r.T@Ws@r)/(k-p); Vb=s2*XtWX_inv_s
    return b, Vb, k

t = pd.read_csv('twostep_worry.csv').merge(mods, on='cntry')
h5 = json.load(open('h5_results.json'))

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, xcol, xlab, spec, col in [
    (axes[0], 'eige2024', 'EIGE Gender Equality Index 2024', 'A_eige', BLUE),
    (axes[1], 'fossil_share', 'Fossil-fuel share of energy, 2023 (%)', 'B_fossil', VERM)]:
    d = t[t[xcol].notna()]
    ax.errorbar(d[xcol], d['b'], yerr=1.96*d['se'], fmt='o', ms=3.5, color=col, ecolor='#bbbbbb', elinewidth=0.8, zorder=3)
    for _, row in d.iterrows():
        ax.annotate(row['cntry'], (row[xcol], row['b']), textcoords='offset points', xytext=(4,2), fontsize=6, color='#444444')
    b, Vb, k = meta_fit(d['b'].to_numpy(), d['se'].to_numpy(), d[xcol].to_numpy())
    xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
    fit = b[0]+b[1]*xs
    sef = np.sqrt(Vb[0,0] + xs**2*Vb[1,1] + 2*xs*Vb[0,1])
    tcrit = stats.t.ppf(0.975, k-2)
    ax.plot(xs, fit, color=col, lw=1.6)
    ax.fill_between(xs, fit-tcrit*sef, fit+tcrit*sef, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color='k', lw=0.7)
    g = h5['meta']['worry'][spec]
    ax.set_title(f"γ = {g['b'][1]:.3f} per s.d.; P = {g['p'][1]:.3f}; R²(meta) = {100*g['R2meta']:.0f}%" if g['p'][1]>=0.001
                 else f"γ = {g['b'][1]:.3f} per s.d.; P < 0.001; R²(meta) = {100*g['R2meta']:.0f}%", fontsize=8.5)
    ax.set_xlabel(xlab); ax.set_ylabel('Sexism–worry slope (β per s.d.)')
axes[0].text(0.02, 0.04, 'a', transform=axes[0].transAxes, fontweight='bold', fontsize=11)
axes[1].text(0.02, 0.04, 'b', transform=axes[1].transAxes, fontweight='bold', fontsize=11)
fig.tight_layout(); fig.savefig('fig5_h5_moderation.png', bbox_inches='tight'); plt.close(fig)
print('fig5 done')
