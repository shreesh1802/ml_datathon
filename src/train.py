"""Leave-one-month-out LightGBM. Writes out-of-fold and test probabilities, a report, and a submission.
usage: python src/train.py TAG [--drop file_with_feature_names] [--weights npy] [--extra pkl] [--rounds N]"""
import os, sys, time, argparse
import numpy as np, pandas as pd, lightgbm as lgb
from common import load_all, CACHE, SUBS, LABELS

ap = argparse.ArgumentParser()
ap.add_argument('tag'); ap.add_argument('--drop', default=None); ap.add_argument('--weights', default=None)
ap.add_argument('--extra', default=None); ap.add_argument('--rounds', type=int, default=0)
ap.add_argument('--legit_rate', type=float, default=0.12); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--full', type=int, default=0); ap.add_argument('--base', default='feats.pkl'); ap.add_argument('--labels', default=None); ap.add_argument('--leaves', type=int, default=63); ap.add_argument('--evalref', default=None)
ap.add_argument('--lr', type=float, default=0.06); ap.add_argument('--driftfix', action='store_true')
a = ap.parse_args()
T0 = time.time()
def log(*x): print(f'[{time.time() - T0:6.0f}s]', *x, flush=True)

from common import macro_f1, tune_weights
K = len(LABELS)


if __name__ == '__main__':
    al = load_all()
    # load feature tables one at a time into a single float32 array (the laptop cannot hold two copies)
    BAD = ['ip_risk_rk', 'dev_n_rows', 'snd_cnt_7d', 'snd_amt_7d', 'snd_tot', 'dev_cnt_1d', 'snd_cnt_1d', 'day_of_month', 'weekday']
    # ip_risk separates fraud in Jul-Sep (0.41 vs 0.20) but not in October (0.24 vs 0.22): a trap, so it goes
    drop = set(l.strip() for l in open(a.drop) if l.strip()) if a.drop else set()
    cols, feats = [], []
    for fi, fn in enumerate([a.base] + (a.extra.split(',') if a.extra else [])):
        Fp = pd.read_pickle(os.path.join(CACHE, fn))
        for c in Fp.columns:
            if c in drop: continue
            v = Fp[c].values.astype(np.float32)
            if a.driftfix and fi == 0:
                if c in BAD or c.startswith('rcv_lamt_'): continue
                # October QR swaps arrive with the mismatch flag blank; treat blank as "not flagged", never as its own signal
                if c == 'qr_checked': continue
                if c == 'qr_vpa_mismatch': v = (v == 1).astype(np.float32)
                if '_gap_' in c:   # half-volume October stretches long gaps
                    v = np.where(np.isnan(v), 7200, np.minimum(v, 7200)).astype(np.float32)
            cols.append(v); feats.append(c)
        del Fp
    if a.driftfix:
        cols.append(((al.receiver_type.values == 'P2P') & (al.txn_type.values == 'QR')).astype(np.float32)); feats.append('p2p_qr')
    log('features', len(feats))
    X = np.empty((len(al), len(feats)), dtype=np.float32)
    for i in range(len(cols)):
        X[:, i] = cols[i]; cols[i] = None
    del cols
    y = al.label.map({l: i for i, l in enumerate(LABELS)}).fillna(-1).astype(int).values
    y_given = y.copy()
    if a.labels:   # cleaned labels (see clean.py); saved under this tag so the next cleaning round can chain
        y = np.load(os.path.join(CACHE, f'y_{a.labels}.npy')); np.save(os.path.join(CACHE, f'y_{a.tag}.npy'), y)
    is_test = al.is_test.values
    month = al.timestamp.str[:7].values
    sw = np.ones(len(al)) if not a.weights else np.load(os.path.join(CACHE, a.weights))
    rng = np.random.default_rng(a.seed)
    keep = (y != 0) | (rng.random(len(al)) < a.legit_rate)   # subsample legit for training speed
    wt = np.where(y == 0, 1.0 / a.legit_rate, 1.0) * sw
    params = dict(objective='multiclass', num_class=K, learning_rate=a.lr, num_leaves=a.leaves, min_data_in_leaf=30,
                  feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, max_bin=127,
                  verbose=-1, num_threads=os.cpu_count(), seed=a.seed)
    cat = [c for c in ['txn_type', 'merchant_category'] if c in feats]
    oof = np.zeros((len(al), K), dtype=np.float32); ptest = np.zeros((is_test.sum(), K), dtype=np.float32)
    folds = ['2026-07', '2026-08', '2026-09']; imp = np.zeros(len(feats)); iters = []
    for m in folds:
        va = (~is_test) & (month == m); tr = (~is_test) & (month != m) & keep & (wt > 0)
        dtr = lgb.Dataset(X[tr], y[tr], weight=wt[tr], feature_name=feats, categorical_feature=cat, free_raw_data=False)
        vs = va & keep
        dva = lgb.Dataset(X[vs], y[vs], weight=np.where(y[vs] == 0, 1.0 / a.legit_rate, 1.0), reference=dtr)
        if a.rounds:
            bst = lgb.train(params, dtr, a.rounds)
        else:
            bst = lgb.train(params, dtr, 3000, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
        it = bst.best_iteration or a.rounds; iters.append(it)
        oof[va] = bst.predict(X[va], num_iteration=it)
        ptest += bst.predict(X[is_test], num_iteration=it) / len(folds)
        imp += bst.feature_importance('gain')
        log(f'fold {m}: iters {it}  argmax macro-F1 {macro_f1(y[va], oof[va].argmax(1)):.4f}')
    if a.full:   # refit on every labelled row with several seeds; the folds above only supply out-of-fold probabilities
        rounds = int(np.mean(iters) * 1.15); pf = np.zeros_like(ptest)
        for sd in range(a.full):
            r2 = np.random.default_rng(100 + sd)
            kp = (~is_test) & (wt > 0) & ((y != 0) | (r2.random(len(al)) < a.legit_rate * 1.5))
            w2 = np.where(y == 0, 1.0 / (a.legit_rate * 1.5), 1.0) * sw
            d = lgb.Dataset(X[kp], y[kp], weight=w2[kp], feature_name=feats, categorical_feature=cat)
            b = lgb.train({**params, 'seed': 100 + sd}, d, rounds)
            pf += b.predict(X[is_test]) / a.full
            log(f'full fit seed {sd}: rounds {rounds}  agreement with fold average {((pf * a.full / (sd + 1)).argmax(1) == ptest.argmax(1)).mean():.5f}')
        np.save(os.path.join(CACHE, f'test_{a.tag}_foldavg.npy'), ptest)
        ptest = (pf * 0.7 + ptest * 0.3).astype(np.float32)
    tr_all = (~is_test) & (month < '2026-10')
    np.save(os.path.join(CACHE, f'oof_{a.tag}.npy'), oof); np.save(os.path.join(CACHE, f'test_{a.tag}.npy'), ptest)
    pd.Series(imp, index=feats).sort_values(ascending=False).to_csv(os.path.join(CACHE, f'imp_{a.tag}.csv'))
    # per-class multipliers: tune on two of the months, check on the third, then tune on all for the submission
    log('argmax OOF macro-F1', round(macro_f1(y[tr_all], oof[tr_all].argmax(1)), 4),
        ' on given (noisy) labels:', round(macro_f1(y_given[tr_all], oof[tr_all].argmax(1)), 4))
    if a.evalref:   # fixed reference labels so different cleaning rounds are comparable
        y = np.load(os.path.join(CACHE, f'y_{a.evalref}.npy'))
        log('argmax OOF macro-F1 on reference labels', a.evalref, round(macro_f1(y[tr_all], oof[tr_all].argmax(1)), 4))
    sep = tr_all & (month == '2026-09'); ja = tr_all & ~sep
    w_ja, s_ja = tune_weights(oof[ja], y[ja])
    log(f'tuned on Jul-Aug: {s_ja:.4f}   same weights on Sep (honest): {macro_f1(y[sep], (oof[sep] * w_ja).argmax(1)):.4f}')
    w, s = tune_weights(oof[tr_all], y[tr_all])
    log(f'tuned on all OOF: {s:.4f}'); np.save(os.path.join(CACHE, f'w_{a.tag}.npy'), w)
    f = macro_f1(y[tr_all], (oof[tr_all] * w).argmax(1), per_class=True)
    pred = (oof[tr_all] * w).argmax(1); cm = np.bincount(y[tr_all] * K + pred, minlength=K * K).reshape(K, K)
    rep = pd.DataFrame({'f1': f.round(3), 'prec': (np.diag(cm) / np.maximum(cm.sum(0), 1)).round(3),
                        'rec': (np.diag(cm) / np.maximum(cm.sum(1), 1)).round(3), 'n': cm.sum(1), 'pred': cm.sum(0), 'w': w.round(2)}, index=LABELS)
    print(rep.to_string())
    tp = (ptest * w).argmax(1)
    os.makedirs(SUBS, exist_ok=True)
    sub = pd.DataFrame({'id': al.txn_id[is_test].values, 'label': np.array(LABELS)[tp]})
    sub.to_csv(os.path.join(SUBS, f'sub_{a.tag}.csv'), index=False)
    print('test predicted share per class (x1000):'); print((sub.label.value_counts(normalize=True) * 1000).round(2).to_string())
    print('train share per class (x1000):'); print((pd.Series(np.array(LABELS)[y[tr_all]]).value_counts(normalize=True) * 1000).round(2).to_string())
    log('done', iters)
