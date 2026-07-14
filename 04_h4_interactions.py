"""H4: sexism x gender interaction (full sample); sexism x centrality within men; simple slopes."""
import pandas as pd, numpy as np, json, warnings
import statsmodels.formula.api as smf
warnings.filterwarnings('ignore')

df = pd.read_parquet('analysis.parquet')
CTRL = "z_age + age2 + C(edu3, Treatment('mid')) + z_income + C(urban3, Treatment('town')) + z_relig + C(activity, Treatment('work')) + parent"
BASE = "z_sexism3 + z_mascfel_c + z_femifel_c + z_impbemw_c + z_lrscale_c + " + CTRL
VARS = ['female','z_sexism3','z_mascfel_c','z_femifel_c','z_impbemw_c','z_lrscale_c',
        'z_age','age2','edu3','z_income','urban3','z_relig','activity','parent','cntry']
OUT = {'worry':'z_worry','attrib':'z_attrib','resp':'z_resp'}

res = {}
for oname, y in OUT.items():
    sub = df[[y]+VARS].dropna()
    r = {'N': int(len(sub))}
    # (a) sexism x female, full sample
    m = smf.mixedlm(f"{y} ~ female*z_sexism3 + {BASE}", sub, groups=sub['cntry']).fit(reml=False)
    r['sexism_men']   = [float(m.params['z_sexism3']), float(m.bse['z_sexism3']), float(m.pvalues['z_sexism3'])]
    r['interaction']  = [float(m.params['female:z_sexism3']), float(m.bse['female:z_sexism3']), float(m.pvalues['female:z_sexism3'])]
    r['sexism_women'] = [float(m.params['z_sexism3'] + m.params['female:z_sexism3'])]
    # women slope SE via covariance
    V = m.cov_params()
    se_w = np.sqrt(V.loc['z_sexism3','z_sexism3'] + V.loc['female:z_sexism3','female:z_sexism3'] + 2*V.loc['z_sexism3','female:z_sexism3'])
    r['sexism_women'] += [float(se_w), float(2*(1-__import__('scipy.stats', fromlist=['norm']).norm.cdf(abs(r['sexism_women'][0]/se_w))))]
    # (b) sexism x centrality within men
    men = sub[sub['female']==0]
    mm = smf.mixedlm(f"{y} ~ z_sexism3*z_impbemw_c + z_mascfel_c + z_femifel_c + z_lrscale_c + {CTRL}", men, groups=men['cntry']).fit(reml=False)
    r['men_N'] = int(len(men))
    r['men_sexism'] = [float(mm.params['z_sexism3']), float(mm.bse['z_sexism3']), float(mm.pvalues['z_sexism3'])]
    r['men_inter_centr'] = [float(mm.params['z_sexism3:z_impbemw_c']), float(mm.bse['z_sexism3:z_impbemw_c']), float(mm.pvalues['z_sexism3:z_impbemw_c'])]
    # simple slopes at centrality -1SD / +1SD for men
    b, bi = mm.params['z_sexism3'], mm.params['z_sexism3:z_impbemw_c']
    Vm = mm.cov_params()
    for lab, cv in [('lo', -1.0), ('hi', 1.0)]:
        sl = b + bi*cv
        se = np.sqrt(Vm.loc['z_sexism3','z_sexism3'] + cv**2*Vm.loc['z_sexism3:z_impbemw_c','z_sexism3:z_impbemw_c'] + 2*cv*Vm.loc['z_sexism3','z_sexism3:z_impbemw_c'])
        r[f'men_slope_centr_{lab}'] = [float(sl), float(se)]
    # (c) symmetry: within women
    wom = sub[sub['female']==1]
    mw = smf.mixedlm(f"{y} ~ z_sexism3*z_impbemw_c + z_mascfel_c + z_femifel_c + z_lrscale_c + {CTRL}", wom, groups=wom['cntry']).fit(reml=False)
    r['women_inter_centr'] = [float(mw.params['z_sexism3:z_impbemw_c']), float(mw.bse['z_sexism3:z_impbemw_c']), float(mw.pvalues['z_sexism3:z_impbemw_c'])]
    res[oname] = r
    print(oname, 'men slope=', round(r['sexism_men'][0],4), 'inter=', round(r['interaction'][0],4), 'p=', round(r['interaction'][2],4),
          '| women slope=', round(r['sexism_women'][0],4),
          '| men sexism x centr =', round(r['men_inter_centr'][0],4), 'p=', round(r['men_inter_centr'][2],4),
          'slopes lo/hi:', round(r['men_slope_centr_lo'][0],4), round(r['men_slope_centr_hi'][0],4))

json.dump(res, open('h4_results.json','w'), indent=1)
print('done')
