"""Train every model and run every experiment. Writes artifacts/ (models + metrics.json).

    python train.py          # ~15 min on a laptop GPU

Experiments (all on CIC-IDS2017 improved, temporal split, held-out test):
  models        6 classifiers: macro-F1, false-alarm rate, confusion matrix, speed
  protocol      random split vs temporal split (how much leakage inflates scores)
  features      47 portable features vs all 82
  learning      learning curves (train vs val macro-F1 as data grows)
  overfit       MLP: overfit -> L2 / dropout / early stopping / more data
  perceptron    PLA vs pocket PLA on 2-D projections (separable vs not)
  novelty       autoencoder on benign only: false alarms, detection per family
  zeroday       leave-one-family-out: what the classifier misses, what the AE catches
  crossdataset  2017-trained models on CSE-CIC-IDS2018 days (same extractor, new network)
  evasion       detection vs attacker slowing down / padding packets
"""
import copy
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from torch import nn

from core import (ART, CLASSES, DATA, FEATURES, META, MODELS, Prep, autoencoder, family, load_2017, mlp,
                  perturb, recon_error, torch_predict)

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
K = len(CLASSES)
CI = {c: i for i, c in enumerate(CLASSES)}
BENIGN_TRAIN_CAP = 300_000  # benign is ~80% of flows; cap keeps training fast, test keeps the real ratio
SVM_TRAIN = 100_000          # kernel features are n x 1000; subsample to fit in RAM
MLP_CFG = {'width': [256, 128], 'p': 0.2}
M = {}                       # metrics.json


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- data
def clean(df):
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    n_nan = n0 - len(df)
    df['fam'] = df['Label'].map({l: family(l) for l in df['Label'].unique()})
    dup = df.duplicated(subset=FEATURES + ['fam'])
    df = df[~dup]
    conflict = df.duplicated(subset=FEATURES, keep=False)  # identical inputs, different family
    df = df[~conflict]
    return df, {'raw': n0, 'nan_inf': n_nan, 'duplicates': int(dup.sum()),
                'conflicting': int(conflict.sum()), 'clean': len(df)}


def temporal_split(df):
    """Per (day, raw label): first 60% of flows in time -> train, next 20% val, last 20% test.
    Tests on the *later* part of every attack and every day, never interleaved with training."""
    ts = pd.to_datetime(df['Timestamp'])
    r = ts.groupby([df['day'], df['Label']]).rank(pct=True, method='first')
    return np.where(r <= .6, 'train', np.where(r <= .8, 'val', 'test'))


def cap_benign(df):
    b = df['fam'] == 'BENIGN'
    keep = df[b].sample(n=min(BENIGN_TRAIN_CAP, b.sum()), random_state=SEED)
    return pd.concat([keep, df[~b]]).sample(frac=1, random_state=SEED)


def sqrt_balanced(y):
    """Class weights sqrt(N / (K n_c)): helps the 62-flow WebAttack class without letting it dominate."""
    cnt = np.bincount(y, minlength=K).clip(min=1)
    return np.sqrt(len(y) / (K * cnt))


# ---------------------------------------------------------------- metrics
def report(y, p, secs_train=None, secs_pred=None):
    labels = list(range(K))
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    benign = y == 0
    return {
        'macro_f1': float(f1[sup > 0].mean()),
        'accuracy': float((y == p).mean()),
        'false_alarm_rate': float((p[benign] != 0).mean()),
        'attack_detection': float((p[~benign] != 0).mean()),
        'per_class': {CLASSES[i]: {'precision': float(pr[i]), 'recall': float(rc[i]), 'f1': float(f1[i]),
                                   'support': int(sup[i])} for i in labels},
        'confusion': confusion_matrix(y, p, labels=labels).tolist(),
        **({'train_s': round(secs_train, 1)} if secs_train is not None else {}),
        **({'us_per_flow': round(secs_pred / len(y) * 1e6, 2)} if secs_pred is not None else {}),
    }


# ---------------------------------------------------------------- models
def sklearn_models():
    return {
        'PLA': Perceptron(max_iter=50, tol=1e-4, random_state=SEED),
        'LinReg': RidgeClassifier(alpha=1.0),  # least-squares on one-hot targets = linear regression as a classifier
        'LogReg': LogisticRegression(max_iter=500),
        'SVM': make_pipeline(Nystroem(kernel='rbf', gamma=0.02, n_components=1000, random_state=SEED),
                             LinearSVC(C=1.0)),  # RBF kernel via Nystroem features: scales to 100k flows
        'LGBM': lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED),
    }


def fit_sklearn(name, m, X, y, sw, Xv=None, yv=None):
    if name == 'SVM':
        idx = rng.choice(len(X), min(SVM_TRAIN, len(X)), replace=False)
        m.fit(X[idx], y[idx], linearsvc__sample_weight=sw[idx])
    elif name == 'LGBM':
        m.fit(X, y, sample_weight=sw, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(40, verbose=False)])
    else:
        m.fit(X, y, sample_weight=sw)
    return m


def train_net(net, X, y, Xv, yv, epochs, lr=1e-3, wd=0.0, bs=2048, patience=None, weight=None, eval_n=50_000):
    net = net.to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(weight=None if weight is None else torch.tensor(weight, dtype=torch.float32, device=DEV))
    Xt, yt = torch.from_numpy(X).to(DEV), torch.from_numpy(y).long().to(DEV)
    Xvt, yvt = torch.from_numpy(Xv).to(DEV), torch.from_numpy(yv).long().to(DEV)
    sub = torch.randperm(len(Xt), device=DEV)[:eval_n]
    hist, best = [], (np.inf, None, 0)
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            lossf(net(Xt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr = lossf(net(Xt[sub]), yt[sub]).item()
            out = net(Xvt)
            va = lossf(out, yvt).item()
            vf = f1_score(yv, out.argmax(1).cpu().numpy(), average='macro', labels=np.unique(yv), zero_division=0)
        hist.append({'epoch': ep, 'train_loss': round(tr, 5), 'val_loss': round(va, 5), 'val_f1': round(float(vf), 4)})
        if va < best[0]:
            best = (va, copy.deepcopy(net.state_dict()), ep)
        elif patience and ep - best[2] >= patience:
            break
    if patience:
        net.load_state_dict(best[1])
    return net.cpu().eval(), hist, best[2]


def overfit_experiment(Xtr, ytr, Xva, yva):
    """Big MLP on 100 flows per class -> it memorises them. Remedies: weight decay, more data
    (3000 per class); early stopping = keep the epoch with the lowest validation loss.
    Fixed seeds per run so the curves are reproducible."""
    r = np.random.default_rng(SEED)
    vi = r.choice(len(Xva), 20000, replace=False)
    per_class = lambda n: np.concatenate([r.choice(np.flatnonzero(ytr == c), min(n, (ytr == c).sum()), replace=False)
                                          for c in range(K)])
    small, large = per_class(100), per_class(3000)
    runs = {'overfit': (small, 0.0), 'weight_decay': (small, 2.0), 'more_data': (large, 0.0)}
    out = {}
    for k, (idx, wd) in runs.items():
        torch.manual_seed(SEED)
        _, h, b = train_net(mlp(len(FEATURES), K, width=[512, 512, 256], p=0), Xtr[idx], ytr[idx], Xva[vi], yva[vi],
                            epochs=150, wd=wd, bs=128, eval_n=len(idx))
        out[k] = {'history': h, 'best_epoch': b, 'train_n': len(idx), 'weight_decay': wd}
    return out


def sample_2018(x18):
    """Benign background + up to 1500 flows of every real attack label: enough for the app's replay."""
    ben = x18[x18['Label'] == 'BENIGN']
    atk = x18[(x18['fam'] != 'BENIGN') & ~x18['Label'].str.contains('Attempted')]
    parts = [ben.sample(n=min(5000, len(ben)), random_state=SEED)] +             [g.sample(n=min(1500, len(g)), random_state=SEED) for _, g in atk.groupby('Label')]
    cols = FEATURES + ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label', 'fam']
    return pd.concat(parts)[cols]


def pla(X, y, epochs=40, pocket=False):
    """Rosenblatt perceptron (y in {-1,+1}, bias column in X). Pocket keeps the best weights
    seen so far, so it still converges to something sensible on non-separable data."""
    w = np.zeros(X.shape[1])
    best_w, best_err, errs = w.copy(), 1.0, []
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            if np.sign(X[i] @ w) != y[i]:
                w = w + y[i] * X[i]
                if pocket:
                    e = np.mean(np.sign(X @ w) != y)
                    if e < best_err:
                        best_w, best_err = w.copy(), e
        errs.append(float(np.mean(np.sign(X @ (best_w if pocket else w)) != y)))
        if errs[-1] == 0:
            break
    return (best_w if pocket else w), errs


# ---------------------------------------------------------------- main
def main():
    ART.mkdir(exist_ok=True)
    t_all = time.time()

    log('load 2017')
    df, M['dataset'] = clean(load_2017())
    log('clean', M['dataset'])
    unseen = df[df['fam'] == 'UNSEEN']
    df = df[df['fam'] != 'UNSEEN'].copy()
    df['split'] = temporal_split(df)
    tr = cap_benign(df[df['split'] == 'train'])
    va, te = df[df['split'] == 'val'], df[df['split'] == 'test']
    M['dataset'].update({'train': len(tr), 'val': len(va), 'test': len(te), 'unseen': len(unseen),
                         'class_counts': {s: d['fam'].value_counts().to_dict() for s, d in
                                          [('train', tr), ('val', va), ('test', te)]},
                         'raw_labels': df['Label'].value_counts().to_dict()})

    prep = Prep().fit(tr)
    joblib.dump(prep, ART / 'prep.joblib')
    Xtr, Xva, Xte = prep(tr), prep(va), prep(te)
    ytr, yva, yte = (d['fam'].map(CI).to_numpy() for d in (tr, va, te))
    w = sqrt_balanced(ytr)
    sw = w[ytr]

    # ---- classifiers
    M['models'], preds, sk = {}, {}, {}
    for name, m in sklearn_models().items():
        log('fit', name)
        t = time.time()
        sk[name] = fit_sklearn(name, m, Xtr, ytr, sw, Xva, yva)
        ttrain = time.time() - t
        t = time.time()
        preds[name] = np.asarray(sk[name].predict(Xte))
        M['models'][name] = report(yte, preds[name], ttrain, time.time() - t)
        log(name, 'macro-F1 %.4f' % M['models'][name]['macro_f1'])
    joblib.dump(sk, ART / 'sklearn_models.joblib')

    log('fit MLP')
    t = time.time()
    net, hist, best_ep = train_net(mlp(len(FEATURES), K, **MLP_CFG), Xtr, ytr, Xva, yva, epochs=60, wd=1e-4,
                                   patience=8, weight=w)
    ttrain = time.time() - t
    torch.save(net.state_dict(), ART / 'mlp.pt')
    t = time.time()
    preds['MLP'] = torch_predict(net, Xte).argmax(1)
    M['models']['MLP'] = report(yte, preds['MLP'], ttrain, time.time() - t)
    M['models']['MLP'].update({'history': hist, 'best_epoch': best_ep})
    log('MLP macro-F1 %.4f' % M['models']['MLP']['macro_f1'])
    M['models'] = {m: M['models'][m] for m in MODELS}

    # ---- autoencoder (novelty): trained on benign only, threshold = 99th pct of benign val error
    log('fit AE')
    Xb = Xtr[ytr == 0]
    ae = autoencoder(len(FEATURES)).to(DEV)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xbt = torch.from_numpy(Xb).to(DEV)
    for ep in range(25):
        perm = torch.randperm(len(Xbt), device=DEV)
        for i in range(0, len(Xbt), 2048):
            x = Xbt[perm[i:i + 2048]]
            opt.zero_grad()
            ((ae(x) - x) ** 2).mean().backward()
            opt.step()
    ae = ae.cpu().eval()
    torch.save(ae.state_dict(), ART / 'ae.pt')
    thr = float(np.percentile(recon_error(ae, Xva[yva == 0]), 99))
    (ART / 'torch_cfg.json').write_text(json.dumps({'mlp': MLP_CFG, 'ae_threshold': thr}))
    err_te = recon_error(ae, Xte) / thr
    err_un = recon_error(ae, prep(unseen)) / thr
    M['novelty'] = {
        'threshold': thr,
        'false_alarm_rate': float((err_te[yte == 0] > 1).mean()),
        'detection': {c: float((err_te[yte == i] > 1).mean()) for i, c in enumerate(CLASSES) if i},
        'unseen': {l: {'n': int((unseen['Label'] == l).sum()), 'flagged': int((err_un[(unseen['Label'] == l).to_numpy()] > 1).sum())}
                   for l in unseen['Label'].unique()},
        'unseen_lgbm_says_benign': float((sk['LGBM'].predict(prep(unseen)) == 0).mean()),
    }
    log('AE', M['novelty'])

    # ---- protocol: same LGBM, random stratified split instead of temporal
    log('protocol: random split')
    rtr, rte = train_test_split(df, test_size=0.2, stratify=df['fam'], random_state=SEED)
    rtr = cap_benign(rtr)
    rtr, rva = train_test_split(rtr, test_size=0.2, stratify=rtr['fam'], random_state=SEED)
    yr = rtr['fam'].map(CI).to_numpy()
    m = fit_sklearn('LGBM', sklearn_models()['LGBM'], prep(rtr), yr, sqrt_balanced(yr)[yr],
                    prep(rva), rva['fam'].map(CI).to_numpy())
    M['protocol'] = {'temporal': M['models']['LGBM']['macro_f1'],
                     'random': report(rte['fam'].map(CI).to_numpy(), m.predict(prep(rte)))['macro_f1']}
    log('protocol', M['protocol'])

    # ---- features: all numeric columns vs the 47 portable ones
    log('features: all columns')
    allf = [c for c in df.columns if c not in META + ['day', 'fam', 'split']]
    A = lambda d: np.nan_to_num(d[allf].to_numpy(np.float32), posinf=0, neginf=0)
    m = fit_sklearn('LGBM', sklearn_models()['LGBM'], A(tr), ytr, sw, A(va), yva)
    M['features'] = {'portable': {'n': len(FEATURES), 'macro_f1': M['models']['LGBM']['macro_f1']},
                     'all': {'n': len(allf), 'macro_f1': report(yte, m.predict(A(te)))['macro_f1']}}
    log('features', M['features'])

    # ---- learning curves
    log('learning curves')
    M['learning'] = {}
    sizes = [500, 2000, 8000, 32000, 128000, len(Xtr)]
    for name in ['LogReg', 'LGBM']:
        rows = []
        for n in sizes:
            idx = rng.choice(len(Xtr), n, replace=False)
            m = sklearn_models()[name]
            if name == 'LGBM':
                m.set_params(n_estimators=200)
            m.fit(Xtr[idx], ytr[idx], sample_weight=sw[idx])
            f = lambda X, y: float(f1_score(y, m.predict(X), average='macro', labels=np.unique(y), zero_division=0))
            rows.append({'n': n, 'train_f1': f(Xtr[idx], ytr[idx]), 'val_f1': f(Xva, yva)})
        M['learning'][name] = rows
    log('learning', M['learning'])

    # ---- overfitting and remedies (MLP on a small sample)
    log('overfit experiment')
    M['overfit'] = overfit_experiment(Xtr, ytr, Xva, yva)
    log('overfit done')

    # ---- perceptron on 2-D projections
    log('perceptron 2D')
    M['perceptron'] = {}
    for a, b in [('BENIGN', 'DDoS'), ('BENIGN', 'PortScan'), ('BENIGN', 'Bot'), ('DoS', 'DDoS')]:
        ia, ib = np.flatnonzero(ytr == CI[a]), np.flatnonzero(ytr == CI[b])
        idx = np.concatenate([rng.choice(ia, min(300, len(ia)), replace=False),
                              rng.choice(ib, min(300, len(ib)), replace=False)])
        P = PCA(2, random_state=SEED).fit_transform(Xtr[idx])
        P = (P - P.mean(0)) / P.std(0)
        y2 = np.where(ytr[idx] == CI[a], -1, 1)
        Xb2 = np.c_[np.ones(len(P)), P]
        w_pla, e_pla = pla(Xb2, y2)
        w_pk, e_pk = pla(Xb2, y2, pocket=True)
        M['perceptron'][f'{a}_vs_{b}'] = {'points': np.round(P, 3).tolist(), 'y': y2.tolist(),
                                          'pla': {'w': w_pla.tolist(), 'errors': e_pla},
                                          'pocket': {'w': w_pk.tolist(), 'errors': e_pk}}
    log('perceptron', {k: (v['pla']['errors'][-1], v['pocket']['errors'][-1]) for k, v in M['perceptron'].items()})

    # ---- zero-day: leave one family out of training
    log('zero-day')
    M['zeroday'] = {}
    for fam in CLASSES[1:]:
        keep = ytr != CI[fam]
        m = sklearn_models()['LGBM'].set_params(n_estimators=200)
        m.fit(Xtr[keep], ytr[keep], sample_weight=sw[keep])
        sel = yte == CI[fam]
        p, nov = m.predict(Xte[sel]), err_te[sel] > 1
        M['zeroday'][fam] = {'n': int(sel.sum()), 'classifier_alerts': float((p != 0).mean()),
                             'ae_alerts': float(nov.mean()), 'either': float(((p != 0) | nov).mean()),
                             'predicted_as': {CLASSES[c]: float((p == c).mean()) for c in np.unique(p)}}
    log('zero-day', M['zeroday'])

    # ---- cross-dataset: CSE-CIC-IDS2018 days, same improved extractor
    days = sorted(DATA.glob('cic2018_*.parquet'))
    if days:
        log('cross-dataset', [d.name for d in days])
        x18, _ = clean(pd.concat([pd.read_parquet(d) for d in days], ignore_index=True))
        x18 = x18[x18['fam'] != 'UNSEEN']
        sample_2018(x18).to_parquet(ART / 'cic2018_sample.parquet', index=False)  # small copy for the app's replay
        X18, y18 = prep(x18), x18['fam'].map(CI).to_numpy()
        M['crossdataset'] = {'days': [d.stem for d in days], 'n': len(x18),
                             'class_counts': x18['fam'].value_counts().to_dict(), 'models': {}}
        for name in MODELS:
            p = torch_predict(net, X18).argmax(1) if name == 'MLP' else np.asarray(sk[name].predict(X18))
            M['crossdataset']['models'][name] = report(y18, p)
        e18 = recon_error(ae, X18) / thr
        M['crossdataset']['novelty'] = {'false_alarm_rate': float((e18[y18 == 0] > 1).mean()),
                                        'detection': float((e18[y18 != 0] > 1).mean())}
        log('cross', {k: round(v['macro_f1'], 3) for k, v in M['crossdataset']['models'].items()})

    # ---- evasion: attacker slows down / pads packets
    log('evasion')
    atk = te.loc[np.concatenate([rng.choice(g.index, min(len(g), 2000), replace=False)
                                 for f, g in te.groupby('fam') if f != 'BENIGN'])]
    M['evasion'] = {}
    for knob, values in [('delay', [1, 2, 4, 8, 16, 32]), ('pad', [0, 64, 256, 512, 1024])]:
        rows = []
        for v in values:
            Xp = prep(perturb(atk, **{knob: v}))
            row = {knob: v}
            for name in MODELS:
                p = torch_predict(net, Xp).argmax(1) if name == 'MLP' else np.asarray(sk[name].predict(Xp))
                row[name] = float((p != 0).mean())
            row['AE'] = float((recon_error(ae, Xp) / thr > 1).mean())
            rows.append(row)
        M['evasion'][knob] = rows
    log('evasion', M['evasion'])

    # test flows for the app's replay / dissect views
    te[FEATURES + ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label', 'fam', 'day']] \
        .to_parquet(ART / 'test.parquet', index=False)
    unseen[FEATURES + ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label', 'fam', 'day']] \
        .to_parquet(ART / 'unseen.parquet', index=False)

    M['meta'] = {'seed': SEED, 'device': DEV, 'minutes': round((time.time() - t_all) / 60, 1),
                 'features': FEATURES, 'classes': CLASSES, 'models': MODELS}
    (ART / 'metrics.json').write_text(json.dumps(M, indent=1))
    log('done in %.1f min' % M['meta']['minutes'])


if __name__ == '__main__':
    main()
