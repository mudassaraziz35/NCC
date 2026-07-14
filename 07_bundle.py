"""Collect every number the Results section needs into results_bundle.json."""
import pandas as pd, numpy as np, json

df = pd.read_parquet('analysis.parquet')
prep = json.load(open('prep_report.json'))
h12 = json.load(open('h1h2_results.json'))
h3 = json.load(open('h3_results.json'))
h4 = json.load(open('h4_results.json'))
rob = json.load(open('robustness_results.json'))

B = {'prep': prep, 'h12': h12, 'h3': h3, 'h4': h4, 'rob': rob}

# Table 1: weighted means by gender
w = df['anweight']
def wm(x, mask=None):
    ok = x.notna() & w.notna()
    if mask is not None: ok &= mask
    return float(np.average(x[ok], weights=w[ok]))
t1 = {}
for v, lab in [('worry','worry'), ('attrib','attrib'), ('resp','resp'), ('deny','deny'),
               ('sexism3','sexism3'), ('mascfel_c','masc'), ('femifel_c','femi'), ('impbemw_c','centr'),
               ('lrscale_c','lr')]:
    t1[lab] = {'all': wm(df[v]), 'men': wm(df[v], df['female']==0), 'women': wm(df[v], df['female']==1),
               'sd': float(df[v].std()), 'n': int(df[v].notna().sum())}
B['table1'] = t1

# alpha min country
ac = prep['alpha_sexism3_by_country']
B['alpha_min_cntry'] = min(ac, key=ac.get); B['alpha_max_cntry'] = max(ac, key=ac.get)

# missingness snapshot (18+ sample)
miss = {v: float(df[v].isna().mean()*100) for v in ['income','lrscale_c','sexism3','worry','attrib','resp','impbemw_c']}
B['missing_pct'] = miss

# per-country two-step extremes for worry
tw = pd.read_csv('twostep_worry.csv'); tw['z'] = tw['b']/tw['se']
B['worry_most_neg'] = tw.nsmallest(5,'b')[['cntry','b']].values.tolist()
B['worry_sig_pos'] = tw[tw['z']>1.96][['cntry','b']].values.tolist()

# n countries with >= alpha 0.5
B['alpha_below_50'] = [c for c,a in ac.items() if a==a and a < 0.50]

json.dump(B, open('results_bundle.json','w'), indent=1)
print('bundle written')
print('alpha min', B['alpha_min_cntry'], 'below50:', B['alpha_below_50'])
print('missing%', {k: round(v,1) for k,v in miss.items()})
print('t1 sexism men', round(t1['sexism3']['men'],3), 'women', round(t1['sexism3']['women'],3))
print('t1 lr men', round(t1['lr']['men'],2), 'women', round(t1['lr']['women'],2))
