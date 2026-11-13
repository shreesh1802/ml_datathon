"""Shared loading and window helpers."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'upi-fraud-rings')
CACHE = os.path.join(ROOT, 'cache')
SUBS = os.path.join(ROOT, 'submissions')
LABELS = ['legit', 'mule_account', 'velocity_bot', 'layering_chain', 'sim_swap', 'account_takeover',
          'phishing_collect', 'qr_swap', 'refund_abuse', 'fake_cashback', 'loan_app_extortion', 'merchant_collusion']
SHIFT = 1 << 25  # > number of seconds in the data span


def load_all():
    """Train and test stacked, with `t` (seconds since first row) and `is_test`."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, 'all.pkl')
    if os.path.exists(p):
        al = pd.read_pickle(p)
    else:
        tr = pd.read_csv(os.path.join(DATA, 'train.csv'))
        te = pd.read_csv(os.path.join(DATA, 'test.csv'))
        te['label'] = 'TEST'
        al = pd.concat([tr, te], ignore_index=True)
        al['ts'] = pd.to_datetime(al.timestamp)
        al['t'] = (al.ts.astype('int64') // 10**9).astype('int64')
        al['is_test'] = al.label == 'TEST'
        al.to_pickle(p)
    al = al.reset_index(drop=True)
    al['t'] = al.t - al.t.min()
    return al


class Win:
    """Windowed counts/sums over rows sharing a key. Windows are relative to each query row's time."""

    def __init__(self, key, t):
        key = np.asarray(key, dtype=np.int64)
        self.comp = key * SHIFT + np.asarray(t, dtype=np.int64)
        self.order = np.argsort(self.comp, kind='stable')
        self.sorted = self.comp[self.order]
        self._cs = {}
        self._bc = {}   # bounds cache; query arrays must be long-lived (keyed by id)
        self._kc = {}

    def _bounds(self, qkey, qt, lo, hi):
        ck = (id(qkey), id(qt), lo, hi)
        if ck in self._bc:
            return self._bc[ck]
        q = np.asarray(qkey, dtype=np.int64) * SHIFT + np.asarray(qt, dtype=np.int64)
        kk = (id(qkey), id(qt))
        if kk not in self._kc:   # sort queries once so searchsorted walks memory in order
            qo = np.argsort(q, kind='stable')
            k0 = np.empty(len(q), dtype=np.int64); k1 = np.empty(len(q), dtype=np.int64)
            k0[qo] = np.searchsorted(self.sorted, (q[qo] // SHIFT) * SHIFT, 'left')
            k1[qo] = np.searchsorted(self.sorted, (q[qo] // SHIFT + 1) * SHIFT, 'left')
            self._kc[kk] = (qo, k0, k1)
        qo, k0, k1 = self._kc[kk]
        a = np.empty(len(q), dtype=np.int64); b = np.empty(len(q), dtype=np.int64)
        a[qo] = np.searchsorted(self.sorted, q[qo] + lo, 'left')
        b[qo] = np.searchsorted(self.sorted, q[qo] + hi, 'right')
        out = (np.clip(a, k0, k1).astype(np.int32), np.clip(b, k0, k1).astype(np.int32))
        self._bc[ck] = out
        return out

    def count(self, qkey, qt, lo, hi):
        a, b = self._bounds(qkey, qt, lo, hi)
        return (b - a).astype(np.float32)

    def sum(self, vals, qkey, qt, lo, hi, name=None):
        name = name or id(vals)
        if name not in self._cs:
            v = np.nan_to_num(np.asarray(vals, dtype=np.float64))[self.order]
            self._cs[name] = np.concatenate([[0.0], np.cumsum(v)])
        cs = self._cs[name]
        a, b = self._bounds(qkey, qt, lo, hi)
        return (cs[b] - cs[a]).astype(np.float32)

    def nxt(self, qkey, qt, strict=True):
        """Index (into original rows) of the first row with same key and time > qt (or >=). -1 if none."""
        q = np.asarray(qkey, dtype=np.int64) * SHIFT + np.asarray(qt, dtype=np.int64)
        i = np.searchsorted(self.sorted, q, 'right' if strict else 'left')
        ok = (i < len(self.sorted))
        ii = np.where(ok, i, 0)
        ok &= (self.sorted[ii] // SHIFT) == (q // SHIFT)
        return np.where(ok, self.order[ii], -1)

    def prv(self, qkey, qt, strict=True):
        q = np.asarray(qkey, dtype=np.int64) * SHIFT + np.asarray(qt, dtype=np.int64)
        i = np.searchsorted(self.sorted, q, 'left' if strict else 'right') - 1
        ok = i >= 0
        ii = np.where(ok, i, 0)
        ok &= (self.sorted[ii] // SHIFT) == (q // SHIFT)
        return np.where(ok, self.order[ii], -1)


K = len(LABELS)


def macro_f1(y, pred, per_class=False):
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float)
    f = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return f if per_class else f.mean()


def tune_weights(p, y, passes=2):
    """Coordinate ascent on per-class multipliers to maximise macro-F1 of argmax(p * w).
    Ties go to the multiplier closest to 1, so a class that is already separable keeps a neutral weight."""
    w = np.ones(K)
    grid = np.exp(np.linspace(-2.0, 2.0, 17))
    best = macro_f1(y, p.argmax(1))
    for _ in range(passes):
        for k in range(1, K):
            base = w[k]
            scores = np.empty(len(grid))
            for i, gk in enumerate(grid):
                w[k] = base * gk
                scores[i] = macro_f1(y, (p * w).argmax(1))
            ok = np.where(scores >= scores.max() - 2e-4)[0]
            pick = ok[np.argmin(np.abs(np.log(base * grid[ok])))]
            w[k] = base * grid[pick]
            best = scores[pick]
    return w, best
