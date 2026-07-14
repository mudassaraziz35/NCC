"""ESS11 petro-masculinity project - data preparation.
Input: ../data.csv (ESS11 integrated file, edition 4.2, 30 countries)
Output: analysis.parquet + prep_report.json
"""
import pandas as pd, numpy as np, json

usecols = ['idno','cntry','anweight','pspwght','gndr','agea','eisced','hinctnta','domicil',
           'rlgdgr','lrscale','mnactic','chldhhe','wrclmch','ccnthum','ccrdprs',
           'wsekpwr','weasoff','wexashr','wlespdm','wprtbym','wbrgwrm',
           'mascfel','femifel','impbemw','likrisk','liklead','sothnds','actcomp','nobingnd'] + \
          [f'rshipa{i}' for i in range(2,13)]
df = pd.read_csv('../data.csv', usecols=usecols, low_memory=False)
rep = {'n_raw': len(df), 'n_countries': df['cntry'].nunique()}

def clean(s, valid_min, valid_max, missing=()):
    s = pd.to_numeric(s, errors='coerce')
    s = s.where(~s.isin(list(missing)))
    return s.where((s >= valid_min) & (s <= valid_max))

# ---- outcomes ----
df['worry'] = clean(df['wrclmch'], 1, 5, missing=(6,7,8,9))            # 1 not at all - 5 extremely
cc = pd.to_numeric(df['ccnthum'], errors='coerce')
df['attrib'] = cc.where(cc.between(1,5))                               # 1 entirely natural - 5 entirely human
df['deny']   = np.where(cc == 55, 1.0, np.where(cc.between(1,5), 0.0, np.nan))
df['resp']   = clean(df['ccrdprs'], 0, 10, missing=(66,77,88,99))      # 0-10

# ---- sexism ----
for v in ['wsekpwr','weasoff','wexashr','wlespdm']:
    df[v+'_c'] = clean(df[v], 1, 5, missing=(7,8,9))
for v in ['wprtbym','wbrgwrm']:                                        # 1 agree strongly..5 disagree strongly -> reverse
    df[v+'_r'] = 6 - clean(df[v], 1, 5, missing=(7,8,9))
sex3 = df[['wsekpwr_c','weasoff_c','wexashr_c']]
df['sexism3'] = sex3.mean(axis=1).where(sex3.notna().sum(axis=1) >= 2)
df['sexism3_cc'] = sex3.mean(axis=1).where(sex3.notna().sum(axis=1) == 3)  # complete-case version
df['wlespdm_rev'] = 6 - df['wlespdm_c']                                # sexist pole = 'never paid less'
sex4 = df[['wsekpwr_c','weasoff_c','wexashr_c','wlespdm_rev']]
df['sexism4'] = sex4.mean(axis=1).where(sex4.notna().sum(axis=1) >= 3)
ben = df[['wprtbym_r','wbrgwrm_r']]
df['benev2'] = ben.mean(axis=1).where(ben.notna().sum(axis=1) == 2)

# ---- gendered identity ----
for v in ['mascfel','femifel','likrisk','liklead','sothnds','actcomp']:
    df[v+'_c'] = clean(df[v], 0, 6, missing=(7,8,9))
df['impbemw_c'] = clean(df['impbemw'], 0, 6, missing=(66,77,88,99))

# ---- controls ----
df['female'] = np.where(df['gndr']==2, 1.0, np.where(df['gndr']==1, 0.0, np.nan))
df['age'] = clean(df['agea'], 15, 120, missing=(999,))
ei = clean(df['eisced'], 1, 7, missing=(0,55,77,88,99))
df['edu3'] = pd.cut(ei, bins=[0,2,5,7], labels=['low','mid','high'])
df['income'] = clean(df['hinctnta'], 1, 10, missing=(77,88,99))
dom = clean(df['domicil'], 1, 5, missing=(7,8,9))
df['urban3'] = pd.cut(dom, bins=[0,2,3,5], labels=['urban','town','rural'])
df['relig'] = clean(df['rlgdgr'], 0, 10, missing=(77,88,99))
df['lrscale_c'] = clean(df['lrscale'], 0, 10, missing=(77,88,99))
mn = clean(df['mnactic'], 1, 9, missing=(66,77,88,99))
df['activity'] = mn.map({1:'work',2:'education',3:'unemployed',4:'unemployed',
                         5:'sick_disabled',6:'retired',7:'other',8:'homework',9:'other'})
# parenthood: children currently in household (grid rshipa: 2 = son/daughter etc.) OR ever (chldhhe)
kids_now = np.zeros(len(df))
for i in range(2,13):
    r = pd.to_numeric(df[f'rshipa{i}'], errors='coerce')
    kids_now = np.maximum(kids_now, (r==2).astype(float))
chl = pd.to_numeric(df['chldhhe'], errors='coerce')
ever = np.where(chl==1, 1.0, np.where(chl==2, 0.0, np.nan))
df['parent'] = np.where(kids_now==1, 1.0, ever)   # currently with kids -> 1; else chldhhe
rep['parent_missing_pct'] = float(np.mean(pd.isna(df['parent']))*100)

# ---- sample restriction ----
df = df[df['age'] >= 18].copy()
rep['n_18plus'] = len(df)

# ---- reliability (Cronbach alpha) ----
def alpha(frame):
    frame = frame.dropna()
    k = frame.shape[1]
    if len(frame) < 30: return np.nan
    return k/(k-1) * (1 - frame.var(ddof=1).sum()/frame.sum(axis=1).var(ddof=1))
rep['alpha_sexism3_pooled'] = float(alpha(sex3.loc[df.index]))
rep['alpha_sexism4_pooled'] = float(alpha(sex4.loc[df.index]))
percty = {c: float(alpha(sex3.loc[df.index][df['cntry']==c])) for c in sorted(df['cntry'].unique())}
rep['alpha_sexism3_by_country'] = percty
rep['alpha_sexism3_min'] = float(np.nanmin(list(percty.values())))
rep['alpha_sexism3_max'] = float(np.nanmax(list(percty.values())))
# inter-item correlations pooled
rep['sexism3_iic'] = sex3.loc[df.index].corr().round(3).to_dict()

# ---- z-standardisation on pooled 18+ sample ----
for v in ['sexism3','sexism4','benev2','mascfel_c','femifel_c','impbemw_c','age','income','relig','lrscale_c',
          'worry','attrib','resp']:
    m, s = df[v].mean(), df[v].std()
    df['z_'+v] = (df[v]-m)/s
    rep[f'msd_{v}'] = [float(m), float(s)]
df['age2'] = (df['age']-df['age'].mean())**2 / 100.0

# ---- weighted descriptives ----
w = df['anweight']
def wmean(x):
    ok = x.notna() & w.notna()
    return float(np.average(x[ok], weights=w[ok]))
desc = {}
for v in ['worry','attrib','resp','deny','sexism3','mascfel_c','femifel_c','impbemw_c']:
    desc[v] = {'wmean': wmean(df[v]), 'mean': float(df[v].mean()), 'sd': float(df[v].std()),
               'n': int(df[v].notna().sum())}
# weighted gender gaps (female - male), raw units
for v in ['worry','attrib','resp','sexism3']:
    f = df[df['female']==1]; m = df[df['female']==0]
    wf = f['anweight']; wm = m['anweight']
    gf = np.average(f[v][f[v].notna()], weights=wf[f[v].notna()])
    gm = np.average(m[v][m[v].notna()], weights=wm[m[v].notna()])
    desc[v]['gap_raw_w'] = float(gf-gm)
rep['descriptives'] = desc
rep['deny_pct_weighted'] = float(wmean(df['deny'])*100)
rep['deny_pct_by_cntry'] = {c: float(df.loc[df['cntry']==c,'deny'].mean()*100) for c in sorted(df['cntry'].unique())}

df.to_parquet('analysis.parquet')
json.dump(rep, open('prep_report.json','w'), indent=1, default=str)
print(json.dumps({k:v for k,v in rep.items() if k not in ('alpha_sexism3_by_country','deny_pct_by_cntry','sexism3_iic')}, indent=1))
print('alpha by country min/max:', rep['alpha_sexism3_min'], rep['alpha_sexism3_max'])
print('deny% top:', sorted(rep['deny_pct_by_cntry'].items(), key=lambda x:-x[1])[:5])
