"""H1/H2: multilevel random-intercept models, blocks M1-M3, three outcomes."""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_parquet('analysis.parquet')

CTRL = "z_age + age2 + C(edu3, Treatment('mid')) + z_income + C(urban3, Treatment('town')) + z_relig + C(activity, Treatment('work')) + parent"
BLOCKS = {
 'M1':  "female + " + CTRL,
 'M2a': "female + z_sexism3 + " + CTRL,
 'M2b': "female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + " + CTRL,
 'M3':  "female + z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + z_lrscale_c + " + CTRL,
}
VARS_ALL = ['female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_lrscale_c',
            'z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry','anweight']
OUT = {'worry':'z_worry','attrib':'z_attrib','resp':'z_resp'}

results = {}
for oname, y in OUT.items():
    sub = df[[y]+VARS_ALL].dropna()          # common sample across M1-M3 (incl. lrscale)
    res_o = {'N': int(len(sub)), 'countries': int(sub['cntry'].nunique())}
    # ICC from null model
    m0 = smf.mixedlm(f"{y} ~ 1", sub, groups=sub['cntry']).fit(reml=True)
    icc = m0.cov_re.iloc[0,0]/(m0.cov_re.iloc[0,0]+m0.scale)
    res_o['ICC'] = float(icc)
    for bname, rhs in BLOCKS.items():
        m = smf.mixedlm(f"{y} ~ {rhs}", sub, groups=sub['cntry']).fit(reml=False)
        tab = pd.DataFrame({'b': m.params, 'se': m.bse, 'p': m.pvalues})
        tab.to_csv(f'coef_{oname}_{bname}.csv')
        res_o[bname] = {'loglik': float(m.llf),
                        'female': [float(m.params.get('female',np.nan)), float(m.bse.get('female',np.nan)), float(m.pvalues.get('female',np.nan))],
                        'sexism': [float(m.params.get('z_sexism3',np.nan)), float(m.bse.get('z_sexism3',np.nan)), float(m.pvalues.get('z_sexism3',np.nan))],
                        'masc':   [float(m.params.get('z_mascfel_c',np.nan)), float(m.bse.get('z_mascfel_c',np.nan)), float(m.pvalues.get('z_mascfel_c',np.nan))],
                        'femi':   [float(m.params.get('z_femifel_c',np.nan)), float(m.bse.get('z_femifel_c',np.nan)), float(m.pvalues.get('z_femifel_c',np.nan))],
                        'centr':  [float(m.params.get('z_impbemw_c',np.nan)), float(m.bse.get('z_impbemw_c',np.nan)), float(m.pvalues.get('z_impbemw_c',np.nan))],
                        'lr':     [float(m.params.get('z_lrscale_c',np.nan)), float(m.bse.get('z_lrscale_c',np.nan)), float(m.pvalues.get('z_lrscale_c',np.nan))]}
    # attenuation of sexism coefficient by left-right
    b2, b3 = res_o['M2b']['sexism'][0], res_o['M3']['sexism'][0]
    res_o['sexism_attenuation_pct_M2b_to_M3'] = float(100*(1-b3/b2)) if b2 else np.nan
    # larger-sample check for H1 (no lrscale requirement)
    sub2 = df[[y]+[v for v in VARS_ALL if v!='z_lrscale_c']].dropna()
    m = smf.mixedlm(f"{y} ~ {BLOCKS['M2a']}", sub2, groups=sub2['cntry']).fit(reml=False)
    res_o['M2a_maxN'] = {'N': int(len(sub2)), 'sexism': [float(m.params['z_sexism3']), float(m.bse['z_sexism3']), float(m.pvalues['z_sexism3'])]}
    results[oname] = res_o
    print(oname, 'N=', res_o['N'], 'ICC=', round(res_o['ICC'],3),
          '| sexism M2a b=', round(res_o['M2a']['sexism'][0],4),
          'M2b b=', round(res_o['M2b']['sexism'][0],4),
          'M3 b=', round(res_o['M3']['sexism'][0],4),
          '| atten%=', round(res_o['sexism_attenuation_pct_M2b_to_M3'],1))
    print('   female M1=', round(res_o['M1']['female'][0],4), 'M2b=', round(res_o['M2b']['female'][0],4),
          '| masc=', round(res_o['M2b']['masc'][0],4), 'femi=', round(res_o['M2b']['femi'][0],4),
          'centr=', round(res_o['M2b']['centr'][0],4), '| lr M3=', round(res_o['M3']['lr'][0],4))

json.dump(results, open('h1h2_results.json','w'), indent=1)
print('done')
