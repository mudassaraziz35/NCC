"""H5: contextual moderation of the sexism-climate slope.
Backlash hypothesis: slope more negative where gender equality higher (EIGE / reversed GII).
Material hypothesis: slope more negative where fossil-fuel share higher.
Two-step: random-effects meta-regression (method-of-moments tau2, Knapp-Hartung inference)
of country-specific M2b slopes on standardized moderators.
One-step validation: random-slope mixed models with cross-level interaction.
"""
import pandas as pd, numpy as np, json, warnings
from scipy import stats
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

mods = pd.DataFrame(json.load(open('country_moderators.json'))).T
mods.index.name = 'cntry'
mods = mods.reset_index()
for c in ['eige2024','fossil_share','gdp_pc_ppp','gii']:
    mods[c] = pd.to_numeric(mods[c], errors='coerce')
mods['equality_gii'] = -mods['gii']            # higher = more gender-equal
mods['log_gdp'] = np.log(mods['gdp_pc_ppp'])

OUT = ['worry','attrib','resp']

# ---------- meta-regression machinery ----------
def meta_reg(y, se, X):
    """RE meta-regression: MoM tau2 + Knapp-Hartung. X without intercept col; adds one.
    Returns dict with coefs, KH se, t, p, ci, tau2, tau2_null, R2meta, k."""
    k = len(y)
    X1 = np.column_stack([np.ones(k), X])
    p = X1.shape[1]
    w = 1/se**2
    W = np.diag(w)
    XtWX_inv = np.linalg.inv(X1.T @ W @ X1)
    b_fe = XtWX_inv @ X1.T @ W @ y
    resid = y - X1 @ b_fe
    Q = float(resid.T @ W @ resid)
    trP = np.trace(W @ X1 @ XtWX_inv @ X1.T @ W)   # = trace of W H
    tau2 = max(0.0, (Q - (k - p)) / (w.sum() - trP))
    # null model tau2 (intercept only) for R2_meta
    w0 = 1/se**2
    ybar = (w0*y).sum()/w0.sum()
    Q0 = ((y-ybar)**2*w0).sum()
    C0 = w0.sum() - (w0**2).sum()/w0.sum()
    tau2_0 = max(0.0, (Q0-(k-1))/C0)
    # RE weights + KH
    ws = 1/(se**2 + tau2)
    Ws = np.diag(ws)
    XtWX_inv_s = np.linalg.inv(X1.T @ Ws @ X1)
    b = XtWX_inv_s @ X1.T @ Ws @ y
    r = y - X1 @ b
    s2 = float(r.T @ Ws @ r) / (k - p)          # KH scale
    Vb = s2 * XtWX_inv_s
    seb = np.sqrt(np.diag(Vb))
    tval = b/seb
    df = k - p
    pval = 2*stats.t.sf(np.abs(tval), df)
    tcrit = stats.t.ppf(0.975, df)
    return {'b': b.tolist(), 'se': seb.tolist(), 'p': pval.tolist(),
            'ci': [[float(b[i]-tcrit*seb[i]), float(b[i]+tcrit*seb[i])] for i in range(p)],
            'tau2': tau2, 'tau2_null': tau2_0,
            'R2meta': float(max(0.0, 1 - tau2/tau2_0)) if tau2_0 > 0 else np.nan,
            'k': k, 'df': df}

def zcol(d, col):
    v = d[col]
    return (v - v.mean())/v.std(ddof=1)

R = {'moderator_correlations': {}, 'meta': {}, 'onestep': {}}

# ---------- moderator correlations (documentation of collinearity) ----------
cc = mods[['eige2024','equality_gii','fossil_share','log_gdp']].corr()
R['moderator_correlations'] = cc.round(3).to_dict()
print('moderator correlations:\n', cc.round(2))

for o in OUT:
    t = pd.read_csv(f'twostep_{o}.csv').merge(mods, on='cntry')
    res_o = {}
    # specs: name -> (columns, subset filter)
    specs = {
        'A_eige':        (['eige2024'], t['eige2024'].notna()),
        'B_fossil':      (['fossil_share'], t['fossil_share'].notna()),
        'B2_fossil_eurostat': (['fossil_share'], t['fossil_src'].eq('eurostat_ffgae') if 'fossil_src' in t else t['fossil_share'].notna()),
        'C_equality_gii':(['equality_gii'], t['equality_gii'].notna()),
        'D_eige_gdp':    (['eige2024','log_gdp'], t['eige2024'].notna()),
        'E_equality_fossil': (['equality_gii','fossil_share'], t['equality_gii'].notna() & t['fossil_share'].notna()),
        'F_gdp':         (['log_gdp'], t['log_gdp'].notna()),
        'G_equality_gdp':(['equality_gii','log_gdp'], t['equality_gii'].notna()),
    }
    for name, (cols, mask) in specs.items():
        d = t[mask].copy()
        X = np.column_stack([zcol(d, c) for c in cols])
        m = meta_reg(d['b'].to_numpy(), d['se'].to_numpy(), X)
        m['vars'] = cols
        res_o[name] = m
    R['meta'][o] = res_o
    a = res_o['A_eige']; bfo = res_o['B_fossil']; cgi = res_o['C_equality_gii']
    print(f"\n{o}: EIGE γ={a['b'][1]:.4f} (p={a['p'][1]:.4f}, k={a['k']}, R²={a['R2meta']:.2f}) | "
          f"fossil γ={bfo['b'][1]:.4f} (p={bfo['p'][1]:.4f}) | GIIeq γ={cgi['b'][1]:.4f} (p={cgi['p'][1]:.4f})")
    d2 = res_o['D_eige_gdp']; e2 = res_o['E_equality_fossil']
    print(f"   D eige|gdp: {d2['b'][1]:.4f} (p={d2['p'][1]:.3f}) / gdp {d2['b'][2]:.4f} (p={d2['p'][2]:.3f}) | "
          f"E eq|fossil: {e2['b'][1]:.4f} (p={e2['p'][1]:.3f}) / fossil {e2['b'][2]:.4f} (p={e2['p'][2]:.3f})")

json.dump(R, open('h5_results.json','w'), indent=1, default=float)
print('\nmeta-regressions done -> h5_results.json (one-step models run separately in 09)')
