"""H5 one-step validation: random-slope mixed models with cross-level interactions.
y ~ M2b covariates + sexism*moderator, random intercept + random sexism slope by country.
"""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_parquet('analysis.parquet')
mods = pd.DataFrame(json.load(open('country_moderators.json'))).T
mods.index.name = 'cntry'; mods = mods.reset_index()
for c in ['eige2024','fossil_share','gdp_pc_ppp','gii']:
    mods[c] = pd.to_numeric(mods[c], errors='coerce')
mods['equality_gii'] = -mods['gii']
df = df.merge(mods[['cntry','eige2024','fossil_share','equality_gii']], on='cntry', how='left')

CTRL = "z_age + age2 + C(edu3, Treatment('mid')) + z_income + C(urban3, Treatment('town')) + z_relig + C(activity, Treatment('work')) + parent"
BASE = "female + z_mascfel_c + z_femifel_c + z_impbemw_c + " + CTRL
VARS = ['female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry']
OUT = {'worry':'z_worry','attrib':'z_attrib','resp':'z_resp'}

res = {}
for mod_name, mod_col in [('eige','eige2024'), ('equality_gii','equality_gii'), ('fossil','fossil_share')]:
    res[mod_name] = {}
    for oname, y in OUT.items():
        d = df[[y, mod_col]+VARS].dropna().copy()
        cz = d.groupby('cntry')[mod_col].first()
        zmap = (cz - cz.mean())/cz.std(ddof=1)              # standardize across countries
        d['z_mod'] = d['cntry'].map(zmap)
        m = smf.mixedlm(f"{y} ~ z_sexism3*z_mod + {BASE}", d,
                        groups=d['cntry'], re_formula="~z_sexism3").fit(reml=False)
        key = 'z_sexism3:z_mod'
        res[mod_name][oname] = {'inter': [float(m.params[key]), float(m.bse[key]), float(m.pvalues[key])],
                                'sexism_at_mean': [float(m.params['z_sexism3']), float(m.bse['z_sexism3'])],
                                'N': int(len(d)), 'countries': int(d['cntry'].nunique()),
                                'converged': bool(m.converged)}
        print(mod_name, oname, 'inter b=%.4f se=%.4f p=%.4g' % (m.params[key], m.bse[key], m.pvalues[key]),
              '| slope at mean=%.4f' % m.params['z_sexism3'], '| N=%d k=%d conv=%s' % (len(d), d['cntry'].nunique(), m.converged))

json.dump(res, open('h5_onestep.json','w'), indent=1)
print('done')
