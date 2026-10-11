
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def binary_metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-12)
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else float("nan")
    return dict(acc=(tp + tn) / len(y), auc=auc, f1=f1, precision=prec, sensitivity=sens,
                specificity=spec, bal_acc=(sens + spec) / 2, tp=tp, tn=tn, fp=fp, fn=fn)


def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x)
    t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n); out[j] = t
    return out


def _delong_cov(y, preds):

    pos, neg = preds[:, y == 1], preds[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = preds.shape[0]
    tx = np.stack([_midrank(r) for r in pos]); ty = np.stack([_midrank(r) for r in neg])
    tz = np.stack([_midrank(r) for r in np.concatenate([pos, neg], 1)])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01).reshape(k, k) / m + np.cov(v10).reshape(k, k) / n
    return aucs, cov


def delong_test(y, p1, p2):

    y = np.asarray(y)
    aucs, cov = _delong_cov(y, np.vstack([p1, p2]).astype(float))
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(max(var, 1e-12))
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def delong_ci(y, p, alpha=0.05):
    aucs, cov = _delong_cov(np.asarray(y), np.asarray(p, float)[None])
    se = np.sqrt(cov[0, 0]); zc = stats.norm.ppf(1 - alpha / 2)
    return aucs[0], max(0, aucs[0] - zc * se), min(1, aucs[0] + zc * se)


def mcnemar_test(y, yh1, yh2):

    c1 = np.asarray(yh1) == np.asarray(y); c2 = np.asarray(yh2) == np.asarray(y)
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def bootstrap_ci(y, p, fn, n=2000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return np.quantile(vals, [alpha / 2, 1 - alpha / 2])
