"""Plan A: the organisers' single-row baseline. Writes submissions/sub_A_baseline.csv."""
import os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from common import load_all, DATA, SUBS

al = load_all()
home = pd.read_csv(os.path.join(DATA, 'accounts.csv')).set_index('account_id').home_city
X = pd.DataFrame(index=al.index)
X['hour'] = al.ts.dt.hour; X['weekday'] = al.ts.dt.weekday
for c in ['amount', 'sim_age_hours', 'device_age_days', 'sender_account_age_days', 'new_payee', 'pin_retries',
          'login_to_txn_sec', 'balance_before', 'qr_vpa_mismatch']:
    X[c] = al[c]
X['amount_share_of_balance'] = al.amount / al.balance_before
X['away_from_home'] = (al.device_city != al.sender_id.map(home)).astype(int)
for c in ['receiver_type', 'txn_type', 'merchant_category', 'app']:
    X[c] = al[c].fillna('none').astype('category').cat.codes
tr = ~al.is_test.values
sep = tr & (al.timestamp.str[:7] == '2026-09').values
fit = tr & ~sep
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
m.fit(X[fit], al.label[fit])
print('macro-F1 Jul-Aug -> Sep:', round(f1_score(al.label[sep], m.predict(X[sep]), average='macro'), 4), flush=True)
m.fit(X[tr], al.label[tr])
os.makedirs(SUBS, exist_ok=True)
sub = pd.DataFrame({'id': al.txn_id[~tr].values, 'label': m.predict(X[~tr])})
sub.to_csv(os.path.join(SUBS, 'sub_A_baseline.csv'), index=False)
print(sub.label.value_counts()); print('wrote', sub.shape)
