import os, numpy as np, pandas as pd
from common import load_all, CACHE, LABELS, macro_f1
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
al = load_all(); F = pd.read_pickle(os.path.join(CACHE, 'feats.pkl'))
yc = np.load(os.path.join(CACHE, 'y_c1.npy')); y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
oof = np.load(os.path.join(CACHE, 'oof_C1.npy'))
tr = (~al.is_test.values) & (al.timestamp.str[:7].values < '2026-10')
print('ceiling: score of the cleaned labels against the given labels:', round(macro_f1(y[tr], yc[tr]), 4))
print(dict(zip(LABELS, macro_f1(y[tr], yc[tr], per_class=True).round(3))))
# ip risk for ordinary-looking fraud rows
ordn = tr & (y > 0) & (yc == 0); patt = tr & (y > 0) & (yc > 0); lg = tr & (y == 0) & (yc == 0); unrep = tr & (y == 0) & (yc > 0)
for nm, m in [('legit', lg), ('ordinary-looking fraud', ordn), ('patterned fraud', patt), ('unreported fraud', unrep)]:
    print(f'{nm:24s} n={m.sum():8d} ip_risk mean {al.ip_risk_score.values[m].mean():.3f} median {np.median(al.ip_risk_score.values[m]):.3f}')
te = al.is_test.values; pt = np.load(os.path.join(CACHE, 'test_C1.npy'))
print('test: ip_risk mean for predicted fraud', al.ip_risk_score.values[te][pt.argmax(1) > 0].mean().round(3), ' predicted legit', al.ip_risk_score.values[te][pt.argmax(1) == 0].mean().round(3))
print('test ip quantiles', np.quantile(al.ip_risk_score.values[te], [.1, .5, .9, .99]).round(3), ' train', np.quantile(al.ip_risk_score.values[tr], [.1, .5, .9, .99]).round(3))
# sim_swap vs account_takeover
ss, at = LABELS.index('sim_swap'), LABELS.index('account_takeover')
m = tr & np.isin(yc, [ss, at]) & np.isin(y, [ss, at])
d = pd.DataFrame({'given': np.array(LABELS)[y[m]], 'simf': (al.sim_age_hours.values[m] < 72), 'devn': (al.device_age_days.values[m] < 2), 'away': F.away.values[m] == 1,
                  'own': F.own_dev.values[m] == 1, 'night': ((F.hour.values[m] >= 22) | (F.hour.values[m] < 5))})
print('\n== sim_swap vs account_takeover (given label) by row signals'); print(d.groupby(['simf', 'devn', 'away', 'own']).given.value_counts().unstack(fill_value=0).to_string())
# session view: sender within 1 day
key = al.sender_id.values[m] + '_' + (al.t.values[m] // 86400).astype(str)
d['key'] = key; g = d.groupby('key')
s = pd.DataFrame({'n': g.size(), 'n_ss': g.given.apply(lambda x: (x == 'sim_swap').sum()), 'simf': g.simf.mean(), 'devn': g.devn.mean(), 'away': g.away.mean(), 'own': g.own.mean()})
s['mix'] = np.where(s.n_ss == 0, 'all ATO', np.where(s.n_ss == s.n, 'all SIM', 'mixed'))
print('\nsessions (sender-day) by label mix'); print(s.groupby('mix').agg(sessions=('n', 'size'), rows=('n', 'sum'), simf=('simf', 'mean'), devn=('devn', 'mean'), away=('away', 'mean'), own=('own', 'mean')).round(3).to_string())
print('session size distribution', s.n.value_counts().sort_index().head(10).to_dict())
# example sessions
ex = d.assign(ts=al.timestamp.values[m], amt=al.amount.values[m], sim=al.sim_age_hours.values[m], dag=al.device_age_days.values[m], dev=al.device_id.values[m], snd=al.sender_id.values[m], city=al.device_city.values[m])
for k in list(s[s.mix == 'mixed'].index[:3]) + list(s[(s.mix == 'all SIM') & (s.n >= 3)].index[:2]) + list(s[(s.mix == 'all ATO') & (s.n >= 3)].index[:2]):
    print(ex[ex.key == k].sort_values('ts')[['ts', 'snd', 'dev', 'city', 'amt', 'sim', 'dag', 'away', 'own', 'given']].to_string(index=False)); print()
