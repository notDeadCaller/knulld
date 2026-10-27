"""KNULLD console: API + static UI.

    python app.py            # http://localhost:8765
"""
import json
import os
import pathlib
import tempfile
import threading
import webbrowser

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import flows
from core import ART, CLASSES, DATA, FEATURES, MODELS, Bundle, family, perturb

WEB = pathlib.Path(__file__).parent / 'web'
SAMPLES = pathlib.Path(__file__).parent / 'samples'
MAX_PCAP = 50 << 20
MAX_FLOWS = 5000  # keeps the response and the table small
SAMPLE_INFO = {  # what each bundled capture contains (TitaniumOrca repo) and what to expect
    'portscan.pcap': 'nmap-style SYN scan of one host. Expected: PortScan.',
    'hping_syn.pcap': 'hping3 SYN flood, one packet per flow. Per-flow, a lone SYN looks exactly like a scan probe -> PortScan.',
    'syn_attack.pcap': 'SYN flood mixed with normal browsing. Flood flows -> PortScan, browsing -> mostly BENIGN.',
    'Attack.pcap': 'HTTP SYN flood from many spoofed hosts (3.7k flows). Same per-flow look as a scan; tiny 1-packet flows also look novel.',
}
META_COLS = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp']

app = FastAPI(title='KNULLD')
B = Bundle()
METRICS = json.loads((ART / 'metrics.json').read_text())
rng = np.random.default_rng(0)


def mix(benign, attacks, n_benign, bursts):
    """Synthetic mix of held-out flows for the replay: a benign background (real benign only,
    no relabelled 'Attempted' flows) with contiguous attack bursts dropped in at random points.
    Each burst takes an equal time-ordered slice of every raw label in the family."""
    benign = benign[benign['Label'] == 'BENIGN']
    bg = benign.sample(n=min(n_benign, len(benign)), random_state=0).sort_values('Timestamp')
    chunks = [bg.iloc[i:i + 40] for i in range(0, len(bg), 40)]
    for fam, n in bursts:
        a = attacks[(attacks['fam'] == fam) & ~attacks['Label'].str.contains('Attempted')]
        labels = a['Label'].unique()
        for lab in labels:
            g = a[a['Label'] == lab].sort_values('Timestamp')
            k = max(n // len(labels), 1)
            start = int(rng.integers(0, max(len(g) - k, 0) + 1))
            chunks.insert(int(rng.integers(1, len(chunks))), g.iloc[start:start + k])
    return pd.concat(chunks, ignore_index=True)


def build_scenarios():
    te = pd.read_parquet(ART / 'test.parquet')
    un = pd.read_parquet(ART / 'unseen.parquet')
    day = lambda d: te[te['day'] == d]
    ben = lambda d: day(d)[day(d)['fam'] == 'BENIGN']  # mix() keeps only true BENIGN
    sc = {
        'tue-bruteforce': ('Tuesday · brute force', 'FTP and SSH password guessing (Patator)',
                           mix(ben('tuesday'), day('tuesday'), 1600, [('BruteForce', 450)])),
        'wed-dos': ('Wednesday · DoS', 'Slowloris, Slowhttptest, Hulk, GoldenEye (one burst each)',
                    mix(ben('wednesday'), day('wednesday'), 1600, [('DoS', 600)])),
        'thu-web': ('Thursday · web + infiltration', 'XSS / SQL injection / web login brute force, scan from an infected PC',
                    mix(ben('thursday'), day('thursday'), 1600, [('WebAttack', 30), ('PortScan', 300)])),
        'fri-mixed': ('Friday · DDoS, scan, botnet', 'LOIC DDoS, nmap scan, Ares botnet beacons',
                      mix(ben('friday'), day('friday'), 1800, [('DDoS', 300), ('PortScan', 300), ('Bot', 120)])),
        'zero-day': ('Zero-day · never trained on', 'Heartbleed and infiltration: no model saw these classes',
                     mix(ben('thursday'), un.assign(fam='UNSEEN'), 1200, [('UNSEEN', 47)])),
    }
    # small sample written by train.py (shipped with the project); full days only if fetched
    days18 = [ART / 'cic2018_sample.parquet'] if (ART / 'cic2018_sample.parquet').exists() else sorted(DATA.glob('cic2018_*.parquet'))
    if days18:
        d18 = pd.concat([pd.read_parquet(p, columns=FEATURES + META_COLS + ['Label']) for p in days18])
        d18 = d18.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
        d18['fam'] = d18['Label'].map({l: family(l) for l in d18['Label'].unique()})
        d18 = d18.drop_duplicates(subset=FEATURES + ['fam'])  # same hygiene as train.clean()
        sc['new-network'] = ('2018 · unseen network', 'CSE-CIC-IDS2018, never trained on. Expect the models to fail here (see BENCH 2017→2018)',
                             mix(d18[d18['fam'] == 'BENIGN'], d18, 1600, [('BruteForce', 300), ('DoS', 300), ('DDoS', 300)]))
    return sc


def analyse(df, top=6):
    """Run every model on a frame of flows -> list of JSON-ready dicts."""
    out = B.predict(df)
    proba, nov, contrib, unit = out['proba'], out['novelty'], out['contrib'], out['unit']
    pred = out['votes']['LGBM']
    order = np.argsort(-np.abs(contrib), axis=1)[:, :top]
    rows = []
    for i in range(len(df)):
        r = df.iloc[i]
        p = CLASSES[pred[i]]
        novel = bool(nov[i] > 1 and p == 'BENIGN')
        rows.append({
            'i': i,
            'src': r['Src IP'], 'sport': int(r['Src Port']), 'dst': r['Dst IP'], 'dport': int(r['Dst Port']),
            'proto': int(r['Protocol']), 'ts': str(r['Timestamp']),
            'truth': r.get('fam'), 'raw_label': r.get('Label'),
            'pred': p, 'label': 'NOVEL?' if novel else p, 'novel': novel,
            'conf': round(float(proba[i].max()), 4), 'novelty': round(float(nov[i]), 4),
            'votes': {m: CLASSES[out['votes'][m][i]] for m in MODELS},
            'x': np.round(unit[i], 3).tolist(),
            'contrib': [{'f': FEATURES[j], 'v': round(float(contrib[i, j]), 4)} for j in order[i]],
        })
    return rows


SCEN = build_scenarios()
CACHE = {}


@app.get('/api/meta')
def meta():
    return {'classes': CLASSES, 'models': MODELS, 'features': FEATURES,
            'scenarios': [{'id': k, 'title': t, 'desc': d, 'n': len(df)} for k, (t, d, df) in SCEN.items() if k != 'pcap']}


@app.get('/api/metrics')
def metrics():
    return METRICS


@app.get('/api/replay/{sid}')
def replay(sid: str):
    if sid not in SCEN:
        raise HTTPException(404, 'unknown scenario')
    if sid not in CACHE:
        CACHE[sid] = analyse(SCEN[sid][2])
    return CACHE[sid]


@app.get('/api/flow/{sid}/{i}')
def flow(sid: str, i: int, delay: float = 1.0, pad: float = 0.0):
    """One flow, optionally with attacker-side perturbations (evasion test)."""
    if sid not in SCEN or not 0 <= i < len(SCEN[sid][2]):
        raise HTTPException(404, 'unknown flow')
    if not (1 <= delay <= 100 and 0 <= pad <= 1400):
        raise HTTPException(422, 'delay in [1, 100], pad in [0, 1400]')
    df = SCEN[sid][2].iloc[[i]]
    row = analyse(perturb(df, delay=delay, pad=pad), top=len(FEATURES))[0]
    row['i'] = i
    row['raw'] = {f: float(v) for f, v in perturb(df, delay=delay, pad=pad)[FEATURES].iloc[0].items()}
    return row


PCAP_LOCK = threading.Lock()  # one capture slot; serialise concurrent uploads


def classify_capture(path):
    with PCAP_LOCK:
        return _classify(path)


def _classify(path):
    try:
        df = flows.extract(path)
    except Exception as e:  # malformed capture: report, don't crash the server
        raise HTTPException(400, f'could not parse capture: {e}')
    if df.empty:
        raise HTTPException(400, 'no IP flows found')
    df = df.head(MAX_FLOWS)
    # ponytail: single slot, last upload wins; fine for a one-person local demo
    SCEN['pcap'] = ('Uploaded capture', '', df)
    CACHE['pcap'] = analyse(df)
    return CACHE['pcap']


@app.get('/api/samples')
def samples():
    return [{'name': p.name, 'desc': SAMPLE_INFO.get(p.name, '')} for p in sorted(SAMPLES.glob('*.pcap*'))]


@app.post('/api/pcap/sample/{name}')
def pcap_sample(name: str):
    path = SAMPLES / name
    if name not in {s['name'] for s in samples()} or not path.is_file():
        raise HTTPException(404, 'unknown sample')
    return classify_capture(path)


@app.post('/api/pcap')
def pcap(file: UploadFile):  # sync def: runs in the threadpool, the CPU-heavy parse no longer blocks the server
    data = file.file.read(MAX_PCAP + 1)
    if len(data) > MAX_PCAP:
        raise HTTPException(413, f'pcap larger than {MAX_PCAP >> 20} MB')
    with tempfile.NamedTemporaryFile(suffix='.pcap', delete=False) as tmp:
        tmp.write(data)
    try:
        return classify_capture(tmp.name)
    finally:
        pathlib.Path(tmp.name).unlink(missing_ok=True)


@app.get('/')
def index():
    return FileResponse(WEB / 'index.html')


app.mount('/', StaticFiles(directory=WEB), name='web')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8765))
    if not os.environ.get('NO_BROWSER'):  # models are loaded by now; the server is up a moment later
        threading.Timer(1.5, webbrowser.open, [f'http://localhost:{port}']).start()
    uvicorn.run(app, host='127.0.0.1', port=port)
