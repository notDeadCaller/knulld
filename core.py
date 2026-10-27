"""Shared pieces: feature list, label mapping, preprocessing, model definitions, inference."""
import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = pathlib.Path(__file__).parent
DATA, ART = ROOT / 'data', ROOT / 'artifacts'

# Features we can recompute ourselves from raw packets (flows.py), so the same models
# work on any pcap. Names are the improved-CICFlowMeter column names.
FEATURES = [
    'Protocol', 'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd Header Length', 'Bwd Header Length',
    'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Down/Up Ratio', 'FWD Init Win Bytes', 'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
]
# never model inputs: identifiers and time are spurious shortcuts (Arp et al., "Dos and Don'ts")
META = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Label', 'Attempted Category']

CLASSES = ['BENIGN', 'DoS', 'DDoS', 'PortScan', 'BruteForce', 'WebAttack', 'Bot']
MODELS = ['PLA', 'LinReg', 'LogReg', 'SVM', 'MLP', 'LGBM']


def family(label: str) -> str:
    """Raw dataset label -> attack family. 'Attempted' flows carry no attack behaviour -> BENIGN
    (dataset authors' recommendation). Heartbleed/Infiltration are too rare to learn: kept out
    of training entirely and used as the 'never seen' set for the novelty detector."""
    l = label.lower()
    if 'attempted' in l or l == 'benign':
        return 'BENIGN'
    if 'web' in l or 'xss' in l or 'sql' in l:
        return 'WebAttack'
    if 'portscan' in l:
        return 'PortScan'
    if 'ddos' in l or 'loic' in l or 'hoic' in l:
        return 'DDoS'
    if 'dos' in l:
        return 'DoS'
    if 'patator' in l or 'brute' in l:
        return 'BruteForce'
    if 'bot' in l:
        return 'Bot'
    if 'heartbleed' in l or 'infiltration' in l:
        return 'UNSEEN'
    raise ValueError(f'unmapped label {label!r}')


def slog(x):
    """Signed log: counts/rates span 0..1e9, log squashes the tails; sign keeps -1 sentinels."""
    return np.sign(x) * np.log1p(np.abs(x))


class Prep:
    """log -> standardise for the models; log -> robust 0..1 for display."""

    def fit(self, df):
        L = slog(df[FEATURES].to_numpy(np.float64))
        self.mu, self.sd = L.mean(0), L.std(0) + 1e-9
        self.lo, self.hi = np.percentile(L, 1, axis=0), np.percentile(L, 99, axis=0) + 1e-9
        return self

    def __call__(self, df):
        return ((slog(df[FEATURES].to_numpy(np.float64)) - self.mu) / self.sd).astype(np.float32)

    def unit(self, df):
        L = slog(df[FEATURES].to_numpy(np.float64))
        return np.clip((L - self.lo) / (self.hi - self.lo), 0, 1).astype(np.float32)


def mlp(d_in, d_out, width=(256, 128), p=0.2):
    layers, d = [], d_in
    for w in width:
        layers += [nn.Linear(d, w), nn.ReLU()] + ([nn.Dropout(p)] if p else [])
        d = w
    return nn.Sequential(*layers, nn.Linear(d, d_out))


def autoencoder(d_in):
    return nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Linear(32, 12), nn.ReLU(),
                         nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, d_in))


def recon_error(ae, X, batch=65536):
    with torch.no_grad():
        out = []
        for i in range(0, len(X), batch):
            x = torch.from_numpy(X[i:i + batch])
            out.append(((ae(x) - x) ** 2).mean(1).numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def torch_predict(net, X, batch=65536):
    with torch.no_grad():
        return np.concatenate([torch.softmax(net(torch.from_numpy(X[i:i + batch])), 1).numpy()
                               for i in range(0, len(X), batch)])


class Bundle:
    """Everything train.py saved, loaded for inference (CPU)."""

    def __init__(self, art=ART):
        self.prep = joblib.load(art / 'prep.joblib')
        self.sk = joblib.load(art / 'sklearn_models.joblib')  # PLA, LinReg, LogReg, SVM, LGBM
        cfg = json.loads((art / 'torch_cfg.json').read_text())
        self.mlp = mlp(len(FEATURES), len(CLASSES), **cfg['mlp'])
        self.mlp.load_state_dict(torch.load(art / 'mlp.pt', map_location='cpu'))
        self.ae = autoencoder(len(FEATURES))
        self.ae.load_state_dict(torch.load(art / 'ae.pt', map_location='cpu'))
        self.mlp.eval(), self.ae.eval()
        self.ae_thr = cfg['ae_threshold']

    def predict(self, df):
        """-> dict with per-model labels, LGBM probabilities, novelty ratio, SHAP top features."""
        X = self.prep(df)
        votes = {m: np.asarray(self.sk[m].predict(X)) for m in ['PLA', 'LinReg', 'LogReg', 'SVM']}
        votes['MLP'] = torch_predict(self.mlp, X).argmax(1)
        proba = self.sk['LGBM'].predict_proba(X)
        votes['LGBM'] = proba.argmax(1)
        contrib = self.sk['LGBM'].booster_.predict(X, pred_contrib=True)
        k = len(FEATURES) + 1
        contrib = contrib.reshape(len(X), len(CLASSES), k)[np.arange(len(X)), votes['LGBM'], :-1]
        return {
            'votes': {m: v.astype(int) for m, v in votes.items()},
            'proba': proba,
            'novelty': recon_error(self.ae, X) / self.ae_thr,  # >1 = beyond the benign 99th pct
            'contrib': contrib,
            'unit': self.prep.unit(df),
        }


IAT = [c for c in FEATURES if 'IAT' in c] + ['Flow Duration']


MSS = 1460  # largest TCP payload on Ethernet; padding cannot push a packet past it


def perturb(df, delay=1.0, pad=0.0):
    """Feature-level what-if: the same flow k x slower (every gap and the duration scaled by
    `delay`; an approximation, since server response latency would not really stretch) and
    with `pad` extra payload bytes per forward packet, capped at the MSS. Dependent features
    are recomputed; Std columns are unchanged by a constant shift."""
    d = df.copy()
    d[IAT] = d[IAT] * delay
    nf, nb = d['Total Fwd Packet'], d['Total Bwd packets']
    pad = np.clip(MSS - d['Fwd Packet Length Max'], 0, pad)
    for c in ['Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean']:
        d[c] = d[c] + pad
    d['Total Length of Fwd Packet'] = d['Total Length of Fwd Packet'] + pad * nf
    d['Packet Length Max'] = np.maximum(d['Packet Length Max'], d['Fwd Packet Length Max'])
    d['Packet Length Min'] = np.minimum(d['Fwd Packet Length Min'], np.where(nb > 0, d['Bwd Packet Length Min'], np.inf))
    tot = d['Total Length of Fwd Packet'] + d['Total Length of Bwd Packet']
    d['Packet Length Mean'] = tot / (nf + nb).clip(lower=1)
    sec = (d['Flow Duration'] / 1e6).replace(0, np.nan)
    d['Flow Bytes/s'] = (tot / sec).fillna(0)
    d['Flow Packets/s'] = ((nf + nb) / sec).fillna(0)
    return d


def load_2017():
    """All five days, numeric features as float32 plus label/meta. Cached as parquet."""
    cache = DATA / 'cic2017.parquet'
    if cache.exists():
        return pd.read_parquet(cache)
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    parts = []
    for d in days:
        df = pd.read_csv(DATA / f'{d}.csv', low_memory=False)
        df['day'] = d
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    num = [c for c in df.columns if c not in META + ['day']]
    df[num] = df[num].astype(np.float32)
    df.to_parquet(cache, index=False)
    return df
