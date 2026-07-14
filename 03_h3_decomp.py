"""H3 decomposition, fast bootstrap: design matrices built once, numpy lstsq per replicate.
Linear KHB (difference method == product method). Country fixed effects OLS.
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from patsy import dmatrix
warnings.filterwarnings('ignore')
rng = np.random.default_rng(20260710)

df = pd.read_parquet('analysis.parquet')
CTRL = "z_age + age2 + C(edu3, Treatment('mid')) + z_income + C(urban3, Treatment('town')) + z_relig + C(activity, Treatment('work')) + parent + C(cntry)"
MED = ['z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c']
VARS = ['female'] + MED + ['z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry']
OUT = {'worry':'z_worry','attrib':'z_attrib','resp':'z_resp'}
B = 1000

res = {}
for oname, y in OUT.items():
    sub = df[[y]+VARS].dropna().reset_index(drop=True)
    # design matrices (built once)
    X1 = np.asarray(dmatrix("female + " + CTRL, sub))                       # M1
    X2 = np.asarray(dmatrix("female + " + " + ".join(MED) + " + " + CTRL, sub))  # M2
    yv = sub[y].to_numpy()
    j1 = 1  # 'female' is first term after intercept in both designs (patsy orders numeric terms after categoricals?) -> verify by column names
    d1 = dmatrix("female + " + CTRL, sub, return_type='dataframe')
    d2 = dmatrix("female + " + " + ".join(MED) + " + " + CTRL, sub, return_type='dataframe')
    c1 = list(d1.columns).index('female'); c2 = list(d2.columns).index('female')
    med_ix = {k: list(d2.columns).index(k) for k in MED}
    def coefs(Xa, ya):
        return np.linalg.lstsq(Xa, ya, rcond=None)[0]
    b1 = coefs(X1, yv); b2 = coefs(X2, yv)
    g1, g2 = b1[c1], b2[c2]
    # product-of-coefficients contributions (a-paths via same X1 design with mediator as outcome)
    contrib = {}
    for k in MED:
        a = coefs(X1, sub[k].to_numpy())[c1]
        contrib[k] = float(a * b2[med_ix[k]])
    res[oname] = {'N': int(len(sub)), 'gap_M1': float(g1), 'gap_M2': float(g2),
                  'reduction': float(g1-g2), 'share_explained': float((g1-g2)/g1),
                  'contrib': contrib, 'contrib_sum_check': float(sum(contrib.values()))}
    # bootstrap (stratified within country)
    idx_by_c = [sub.index[sub['cntry']==c].to_numpy() for c in sub['cntry'].unique()]
    stats = np.empty((B,2))
    for bi in range(B):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_c])
        bb1 = coefs(X1[take], yv[take]); bb2 = coefs(X2[take], yv[take])
        stats[bi] = [bb1[c1], bb2[c2]]
    red = stats[:,0]-stats[:,1]
    share = np.where(stats[:,0]!=0, red/stats[:,0], np.nan)
    res[oname]['reduction_ci'] = [float(np.percentile(red,2.5)), float(np.percentile(red,97.5))]
    res[oname]['share_ci'] = [float(np.nanpercentile(share,2.5)), float(np.nanpercentile(share,97.5))]
    res[oname]['gapM2_ci'] = [float(np.percentile(stats[:,1],2.5)), float(np.percentile(stats[:,1],97.5))]
    res[oname]['gapM1_ci'] = [float(np.percentile(stats[:,0],2.5)), float(np.percentile(stats[:,0],97.5))]
    print(oname, 'N=%d gapM1=%.4f gapM2=%.4f share=%.1f%% CI[%.1f, %.1f]%%' %
          (len(sub), g1, g2, 100*(g1-g2)/g1, 100*res[oname]['share_ci'][0], 100*res[oname]['share_ci'][1]))
    print('  contrib:', {k: round(v,4) for k,v in contrib.items()}, 'sum=', round(sum(contrib.values()),4), 'vs reduction', round(g1-g2,4))

json.dump(res, open('h3_results.json','w'), indent=1)
print('done')
