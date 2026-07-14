"""Robustness: (1) per-country two-step + RE meta (all outcomes; also ordinal for worry),
(2) denial logit, (3) weighted country-FE OLS, (4) country exclusions, (5) alt sexism codings,
(6) income-missingness sensitivity."""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
warnings.filterwarnings('ignore')

df = pd.read_parquet('analysis.parquet')
CTRL = "z_age + age2 + C(edu3, Treatment('mid')) + z_income + C(urban3, Treatment('town')) + z_relig + C(activity, Treatment('work')) + parent"
M2b = "female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + " + CTRL
VARS = ['female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_lrscale_c',
        'z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry','anweight']
OUT = {'worry':'z_worry','attrib':'z_attrib','resp':'z_resp'}
R = {}

def re_meta(bs, ses):
    bs, ses = np.asarray(bs), np.asarray(ses)
    w = 1/ses**2
    bf = np.sum(w*bs)/np.sum(w)
    Q = np.sum(w*(bs-bf)**2); dfq = len(bs)-1
    C = np.sum(w) - np.sum(w**2)/np.sum(w)
    tau2 = max(0, (Q-dfq)/C)
    wr = 1/(ses**2+tau2)
    br = np.sum(wr*bs)/np.sum(wr); ser = np.sqrt(1/np.sum(wr))
    I2 = max(0, (Q-dfq)/Q)*100 if Q>0 else 0
    return br, ser, tau2, I2

# ---- (1) two-step per-country OLS slopes (M2b spec, no country FE needed within) ----
twostep = {}
for oname, y in OUT.items():
    rows = []
    for c, d in df.groupby('cntry'):
        d = d[[y]+[v for v in VARS if v not in ('cntry','anweight','z_lrscale_c')]].dropna()
        if len(d) < 300: continue
        m = smf.ols(f"{y} ~ {M2b}", d).fit()
        rows.append((c, m.params['z_sexism3'], m.bse['z_sexism3'], len(d)))
    t = pd.DataFrame(rows, columns=['cntry','b','se','n'])
    br, ser, tau2, I2 = re_meta(t['b'], t['se'])
    t.to_csv(f'twostep_{oname}.csv', index=False)
    twostep[oname] = {'pooled_b': float(br), 'pooled_se': float(ser), 'tau2': float(tau2), 'I2': float(I2),
                      'n_countries': int(len(t)), 'n_negative': int((t['b']<0).sum()),
                      'n_sig_negative': int(((t['b']<0)&(t['b']/t['se']<-1.96)).sum()),
                      'n_sig_positive': int(((t['b']>0)&(t['b']/t['se']>1.96)).sum())}
    print('two-step', oname, twostep[oname])
R['twostep'] = twostep

# ---- ordinal robustness for worry: per-country ordered logit, RE meta ----
rows = []
Xvars = ['female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_age','age2','z_income','z_relig','parent']
for c, d in df.groupby('cntry'):
    d = d[['worry']+Xvars+['edu3','urban3','activity']].dropna()
    if len(d) < 300: continue
    X = pd.get_dummies(d[Xvars+['edu3','urban3','activity']], columns=['edu3','urban3','activity'], drop_first=True).astype(float)
    try:
        m = OrderedModel(d['worry'], X, distr='logit').fit(method='bfgs', maxiter=300, disp=False)
        rows.append((c, m.params['z_sexism3'], m.bse['z_sexism3']))
    except Exception as e:
        print('ordinal fail', c, e)
t = pd.DataFrame(rows, columns=['cntry','b','se'])
br, ser, tau2, I2 = re_meta(t['b'], t['se'])
R['ordinal_worry'] = {'pooled_logOR': float(br), 'pooled_se': float(ser), 'OR': float(np.exp(br)), 'I2': float(I2), 'n_countries': int(len(t))}
print('ordinal worry pooled logOR', round(br,4), 'OR', round(np.exp(br),3))

# ---- (2) denial logit: country FE restricted to countries with >=10 events (separation guard),
#         plus all-country pooled logit without FE, both cluster-robust ----
d = df[['deny']+[v for v in VARS if v!='z_lrscale_c']].dropna()
ev = d.groupby('cntry')['deny'].sum()
keep = ev[ev >= 10].index.tolist()
d_fe = d[d['cntry'].isin(keep)]
m = smf.logit(f"deny ~ female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + {CTRL} + C(cntry)", d_fe)\
      .fit(disp=False, method='bfgs', maxiter=500, cov_type='cluster', cov_kwds={'groups': d_fe['cntry']})
m_all = smf.logit(f"deny ~ female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + {CTRL}", d)\
      .fit(disp=False, method='bfgs', maxiter=500, cov_type='cluster', cov_kwds={'groups': d['cntry']})
R['denial'] = {'N_fe': int(len(d_fe)), 'events_fe': int(d_fe['deny'].sum()), 'countries_fe': keep,
               'sexism_logOR_fe': [float(m.params['z_sexism3']), float(m.bse['z_sexism3']), float(m.pvalues['z_sexism3'])],
               'sexism_OR_fe': float(np.exp(m.params['z_sexism3'])),
               'female_logOR_fe': [float(m.params['female']), float(m.bse['female']), float(m.pvalues['female'])],
               'N_all': int(len(d)), 'events_all': int(d['deny'].sum()),
               'sexism_OR_all': float(np.exp(m_all.params['z_sexism3'])),
               'sexism_logOR_all': [float(m_all.params['z_sexism3']), float(m_all.bse['z_sexism3']), float(m_all.pvalues['z_sexism3'])]}
print('denial FE: N', len(d_fe), 'events', int(d_fe['deny'].sum()), 'sexism OR', round(np.exp(m.params['z_sexism3']),3), 'p', m.pvalues['z_sexism3'])
print('denial ALL: OR', round(np.exp(m_all.params['z_sexism3']),3), 'p', m_all.pvalues['z_sexism3'])

# ---- (3) weighted country-FE OLS (anweight), cluster-robust by country ----
wtd = {}
for oname, y in OUT.items():
    d = df[[y]+VARS].dropna()
    m = smf.wls(f"{y} ~ {M2b} + C(cntry)", d, weights=d['anweight'])\
          .fit(cov_type='cluster', cov_kwds={'groups': d['cntry']})
    wtd[oname] = {'sexism': [float(m.params['z_sexism3']), float(m.bse['z_sexism3']), float(m.pvalues['z_sexism3'])], 'N': int(len(d))}
    print('weighted', oname, wtd[oname])
R['weighted'] = wtd

# ---- (4) country exclusions (mixedlm, M2b) ----
excl = {}
for tag, drop in [('no_IL', ['IL']), ('no_UA', ['UA']), ('no_IL_UA', ['IL','UA'])]:
    sub = df[~df['cntry'].isin(drop)]
    row = {}
    for oname, y in OUT.items():
        d = sub[[y]+[v for v in VARS if v not in ('anweight','z_lrscale_c')]].dropna()
        m = smf.mixedlm(f"{y} ~ {M2b}", d, groups=d['cntry']).fit(reml=False)
        row[oname] = [float(m.params['z_sexism3']), float(m.bse['z_sexism3'])]
    excl[tag] = row
    print(tag, {k: round(v[0],4) for k,v in row.items()})
R['exclusions'] = excl

# ---- (5) alternative sexism codings ----
alt = {}
# (a) complete-case 3-item index
df['z_sexism3cc'] = (df['sexism3_cc'] - df['sexism3_cc'].mean())/df['sexism3_cc'].std()
# run
for tag, formula_x in [('sexism3_completecase', 'z_sexism3cc'), ('benevolent_added', 'z_sexism3 + z_benev2')]:
    row = {}
    for oname, y in OUT.items():
        need = [y,'female','z_mascfel_c','z_femifel_c','z_impbemw_c','z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry'] + \
               (['z_sexism3cc'] if 'completecase' in tag else ['z_sexism3','z_benev2'])
        d = df[need].dropna()
        m = smf.mixedlm(f"{y} ~ female + {formula_x} + z_mascfel_c + z_femifel_c + z_impbemw_c + {CTRL}", d, groups=d['cntry']).fit(reml=False)
        key = 'z_sexism3cc' if 'completecase' in tag else 'z_sexism3'
        row[oname] = {'sexism': [float(m.params[key]), float(m.bse[key])],
                      'benev': [float(m.params['z_benev2']), float(m.bse['z_benev2'])] if 'benev' in tag else None,
                      'N': int(len(d))}
    alt[tag] = row
    print(tag, {k: (round(v['sexism'][0],4), round(v['benev'][0],4) if v['benev'] else None) for k,v in row.items()})
R['alt_sexism'] = alt

# ---- (6) income-missingness sensitivity: median-impute income + missing dummy ----
df['inc_miss'] = df['income'].isna().astype(float)
med = df.groupby('cntry')['income'].transform('median')
inc_f = df['income'].fillna(med)
df['z_income_f'] = (inc_f - inc_f.mean())/inc_f.std()
CTRL_f = CTRL.replace('z_income','z_income_f + inc_miss')
inc = {}
for oname, y in OUT.items():
    need = [y,'female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_age','age2','edu3','z_income_f','inc_miss','urban3','z_relig','activity','parent','cntry']
    d = df[need].dropna()
    m = smf.mixedlm(f"{y} ~ female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + {CTRL_f}", d, groups=d['cntry']).fit(reml=False)
    inc[oname] = {'sexism': [float(m.params['z_sexism3']), float(m.bse['z_sexism3'])], 'N': int(len(d)),
                  'female': [float(m.params['female']), float(m.bse['female'])]}
    print('income-sens', oname, 'N', len(d), 'sexism b', round(m.params['z_sexism3'],4))
R['income_sensitivity'] = inc

json.dump(R, open('robustness_results.json','w'), indent=1)
print('done')
