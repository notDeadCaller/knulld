"""Direction-mirror audit of the improved CIC-IDS2017 labels.

For every `label_flows(...)` rule in the official labelling notebook (DistriNet CNS2022_Code), swap the
source and destination IP lists and count BENIGN flows that match the swapped rule inside the same time
window. Rules are read from the notebook source with `ast`, so nothing is retyped by hand.

    python audit/mirror_audit.py <path to CICIDS2017_labelling_fixed_CICFlowMeter.ipynb>
"""
import ast
import json
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent  # audit/ (outputs) ; repo root holds core.py and data/
sys.path.insert(0, str(HERE.parent))
from core import load_2017  # noqa: E402

def _const(n):
    """literal_eval plus the integer arithmetic the 2018 notebook uses (1518618806*(10**9))."""
    if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Pow, ast.Add)):
        a, b = _const(n.left), _const(n.right)
        return a * b if isinstance(n.op, ast.Mult) else a ** b if isinstance(n.op, ast.Pow) else a + b
    return ast.literal_eval(n)


def rules(notebook):
    nb = json.loads(pathlib.Path(notebook).read_text(encoding='utf-8'))
    src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'label_flows':
            a = node.args
            kw = {k.arg: k.value for k in node.keywords}
            lit = lambda n: _const(n) if n is not None else None  # noqa: E731
            pos = lambda i, name: a[i] if len(a) > i else kw.get(name)  # noqa: E731
            out.append({
                'day': a[0].id.replace('_df', ''), 'label': lit(a[1]), 'start_ns': lit(a[2]), 'end_ns': lit(a[3]),
                'src_ip': lit(pos(4, 'src_ip_list')), 'dst_ip': lit(pos(5, 'dst_ip_list')),
                'has_port_or_extra_filters': any(k in kw for k in ('src_port_list', 'dst_port_list', 'additional_filters'))
                or len(a) > 6, 'payload_filter': 'payload_filter' in kw,
            })
    return out


def audit(df, rs):
    ts = pd.to_datetime(df['Timestamp'])
    rows = []
    for r in rs:
        if not r['src_ip'] or not r['dst_ip']:
            continue
        win = (ts >= pd.to_datetime(r['start_ns'], unit='ns')) & (ts <= pd.to_datetime(r['end_ns'], unit='ns'))  # windows are dated
        fwd = win & df['Src IP'].isin(r['src_ip']) & df['Dst IP'].isin(r['dst_ip'])
        rev = win & df['Src IP'].isin(r['dst_ip']) & df['Dst IP'].isin(r['src_ip'])
        rb = rev & (df['Label'] == 'BENIGN')
        pay = df['Total Length of Fwd Packet'] + df['Total Length of Bwd Packet']
        rows.append({'day': r['day'], 'label': r['label'], 'src': ','.join(r['src_ip']), 'dst': f'{len(r["dst_ip"])} ip(s)',
                     'fwd_flows': int(fwd.sum()), 'fwd_labelled_as_rule': int((fwd & (df['Label'] == r['label'])).sum()),
                     'rev_flows': int(rev.sum()), 'rev_benign': int(rb.sum()), 'rev_benign_zero_payload': int((rb & (pay == 0)).sum()),
                     'rev_benign_with_payload': int((rb & (pay > 0)).sum()),
                     'rev_benign_zero_payload_nmap_sig': int((rb & (pay == 0) & (df['FWD Init Win Bytes'] == 0)
                                                             & df['Bwd Init Win Bytes'].isin([1024, 29200])).sum()), 'rule_has_extra_filters': r['has_port_or_extra_filters']})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    rs = rules(sys.argv[1])
    year = '2018' if '2018' in pathlib.Path(sys.argv[1]).name else '2017'
    # 2018: only the three days we hold, benign sampled at 5% (multiply benign counts by ~20 for the full release)
    df = load_2017() if year == '2017' else pd.concat(
        [pd.read_parquet(p) for p in sorted((HERE.parent / 'data').glob('cic2018_*.parquet'))], ignore_index=True)
    res = audit(df, rs)
    res = res[(res['fwd_flows'] > 0) | (res['rev_flows'] > 0)]
    # one row per distinct (day, label, window, ips): Attempted sub-rules share windows with their parent
    res = res.drop_duplicates(subset=['day', 'label', 'src', 'dst', 'fwd_flows', 'rev_flows'])
    pd.set_option('display.width', 250)
    print(res.to_string(index=False))
    pass
    res.to_csv(HERE / f'mirror_audit_{year}.csv', index=False)
