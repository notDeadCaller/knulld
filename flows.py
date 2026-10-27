"""pcap/pcapng -> one row per flow with the 47 features in core.FEATURES.

Follows the (improved) CICFlowMeter conventions the models were trained on:
bidirectional flows keyed by 5-tuple, direction = first packet, 120 s flow timeout,
TCP flows end on RST or once both sides sent FIN, 'packet length' = transport payload
bytes, times in microseconds, sample (n-1) standard deviations.
Not reproduced: CICFlowMeter's bulk/subflow/active-idle features (not model inputs).

    python flows.py capture.pcap      # prints a feature table
    python flows.py                   # runs the self-check
"""
import socket
import sys

import dpkt
import numpy as np
import pandas as pd

from core import FEATURES

TIMEOUT = 120_000_000  # microseconds
FIN, SYN, RST, PSH, ACK, URG = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20


def read_packets(path):
    """Yield (ts, ip_packet) from pcap or pcapng; Ethernet/VLAN, raw IP, Linux SLL, loopback."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        f.seek(0)
        rd = dpkt.pcapng.Reader(f) if magic == b'\x0a\x0d\x0d\x0a' else dpkt.pcap.Reader(f)
        dl = rd.datalink()
        for ts, buf in rd:
            try:
                if dl == dpkt.pcap.DLT_EN10MB:
                    ip = dpkt.ethernet.Ethernet(buf).data
                elif dl == dpkt.pcap.DLT_LINUX_SLL:
                    ip = dpkt.sll.SLL(buf).data
                elif dl in (dpkt.pcap.DLT_NULL, dpkt.pcap.DLT_LOOP):
                    ip = dpkt.loopback.Loopback(buf).data
                else:
                    ip = dpkt.ip.IP(buf)
            except (dpkt.UnpackError, ValueError, IndexError):
                continue
            if isinstance(ip, (dpkt.ip.IP, dpkt.ip6.IP6)):
                yield float(ts), ip


def parse(ip):
    """-> (src, sport, dst, dport, proto, payload_len, header_len, flags, window)"""
    v6 = isinstance(ip, dpkt.ip6.IP6)
    af = socket.AF_INET6 if v6 else socket.AF_INET
    src, dst = socket.inet_ntop(af, ip.src), socket.inet_ntop(af, ip.dst)
    proto = ip.nxt if v6 else ip.p
    ip_payload = ip.plen if v6 else ip.len - ip.hl * 4  # from headers: robust to snaplen truncation
    l4 = ip.data
    if isinstance(l4, dpkt.tcp.TCP):
        h = l4.off * 4
        return src, l4.sport, dst, l4.dport, proto, max(ip_payload - h, 0), h, l4.flags, l4.win
    if isinstance(l4, dpkt.udp.UDP):
        return src, l4.sport, dst, l4.dport, proto, max(l4.ulen - 8, 0), 8, 0, None
    return src, 0, dst, 0, proto, max(ip_payload, 0), 0, 0, None


def _stats(a):
    a = np.asarray(a, np.float64)
    if len(a) == 0:
        return 0.0, 0.0, 0.0, 0.0
    return a.max(), a.min(), a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)


def _iat(ts):
    d = np.diff(np.asarray(ts, np.float64))
    mx, mn, mean, std = _stats(d)
    return d.sum() if len(d) else 0.0, mean, std, mx, mn


def features(key, pkts):
    """pkts: list of (ts_us, dir, plen, hlen, flags, win); dir 0 = forward."""
    src, sport, dst, dport, proto = key
    ts = [p[0] for p in pkts]
    fwd = [p for p in pkts if p[1] == 0]
    bwd = [p for p in pkts if p[1] == 1]
    flen, blen, alen = [p[2] for p in fwd], [p[2] for p in bwd], [p[2] for p in pkts]
    dur = ts[-1] - ts[0]
    fmax, fmin, fmean, fstd = _stats(flen)
    bmax, bmin, bmean, bstd = _stats(blen)
    pmax, pmin, pmean, pstd = _stats(alen)
    _, fi_mean, fi_std, fi_max, fi_min = _iat(ts)
    ftot, fwd_mean, fwd_std, fwd_max, fwd_min = _iat([p[0] for p in fwd])
    btot, bwd_mean, bwd_std, bwd_max, bwd_min = _iat([p[0] for p in bwd])
    nbytes, npk = sum(alen), len(pkts)
    cnt = lambda bit: sum(1 for p in pkts if p[4] & bit)
    win = lambda side: next((p[5] for p in side if p[5] is not None), 0)
    row = {
        'Protocol': proto, 'Flow Duration': dur, 'Total Fwd Packet': len(fwd), 'Total Bwd packets': len(bwd),
        'Total Length of Fwd Packet': sum(flen), 'Total Length of Bwd Packet': sum(blen),
        'Fwd Packet Length Max': fmax, 'Fwd Packet Length Min': fmin, 'Fwd Packet Length Mean': fmean,
        'Fwd Packet Length Std': fstd, 'Bwd Packet Length Max': bmax, 'Bwd Packet Length Min': bmin,
        'Bwd Packet Length Mean': bmean, 'Bwd Packet Length Std': bstd,
        'Flow Bytes/s': nbytes / (dur / 1e6) if dur else 0.0, 'Flow Packets/s': npk / (dur / 1e6) if dur else 0.0,
        'Flow IAT Mean': fi_mean, 'Flow IAT Std': fi_std, 'Flow IAT Max': fi_max, 'Flow IAT Min': fi_min,
        'Fwd IAT Total': ftot, 'Fwd IAT Mean': fwd_mean, 'Fwd IAT Std': fwd_std, 'Fwd IAT Max': fwd_max,
        'Fwd IAT Min': fwd_min, 'Bwd IAT Total': btot, 'Bwd IAT Mean': bwd_mean, 'Bwd IAT Std': bwd_std,
        'Bwd IAT Max': bwd_max, 'Bwd IAT Min': bwd_min,
        'Fwd Header Length': sum(p[3] for p in fwd), 'Bwd Header Length': sum(p[3] for p in bwd),
        'Packet Length Min': pmin, 'Packet Length Max': pmax, 'Packet Length Mean': pmean, 'Packet Length Std': pstd,
        'FIN Flag Count': cnt(FIN), 'SYN Flag Count': cnt(SYN), 'RST Flag Count': cnt(RST),
        'PSH Flag Count': cnt(PSH), 'ACK Flag Count': cnt(ACK), 'URG Flag Count': cnt(URG),
        'Down/Up Ratio': len(bwd) / len(fwd) if fwd else 0.0,
        'FWD Init Win Bytes': win(fwd), 'Bwd Init Win Bytes': win(bwd),
        'Fwd Act Data Pkts': sum(1 for x in flen if x > 0), 'Fwd Seg Size Min': min(p[3] for p in fwd) if fwd else 0,
    }
    meta = {'Src IP': src, 'Src Port': sport, 'Dst IP': dst, 'Dst Port': dport,
            'Timestamp': pd.Timestamp(ts[0], unit='us').strftime('%Y-%m-%d %H:%M:%S.%f')}
    return {**meta, **row}


def extract(path, max_packets=None):
    """Read a capture, return a DataFrame: meta columns + FEATURES, one row per flow."""
    active, rows = {}, []

    def close(k):
        f = active.pop(k)
        rows.append(features(k, f['pkts']))

    for n, (ts, ip) in enumerate(read_packets(path)):
        if max_packets and n >= max_packets:
            break
        ts = round(ts * 1e6)  # integer microseconds, like CICFlowMeter
        src, sport, dst, dport, proto, plen, hlen, flags, win = parse(ip)
        k, rk = (src, sport, dst, dport, proto), (dst, dport, src, sport, proto)
        key = k if k in active else rk if rk in active else None
        if key and ts - active[key]['start'] > TIMEOUT:
            close(key)
            key = None
        if key is None:
            key = k
            active[k] = {'start': ts, 'pkts': [], 'fin': [False, False]}
        f, d = active[key], 0 if key == k else 1
        f['pkts'].append((ts, d, plen, hlen, flags, win))
        if proto == 6:
            if flags & FIN:
                f['fin'][d] = True
            if flags & RST or all(f['fin']):
                close(key)
    for k in list(active):
        close(k)
    cols = ['Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp'] + FEATURES
    return pd.DataFrame(rows, columns=cols)


def _selfcheck():
    """Write a tiny pcap with a known TCP exchange and a lone UDP packet, check the numbers."""
    import os
    import tempfile
    cli, srv = socket.inet_aton('10.0.0.1'), socket.inet_aton('10.0.0.2')

    def pkt(src, dst, sp, dp, flags, data=b'', win=8192):
        tcp = dpkt.tcp.TCP(sport=sp, dport=dp, flags=flags, win=win, data=data)
        ip = dpkt.ip.IP(src=src, dst=dst, p=6, data=tcp)
        ip.len = 20 + len(tcp)
        return bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_IP, data=ip))

    t0 = 1_500_000_000.0
    seq = [  # (dt seconds, packet)
        (0.000, pkt(cli, srv, 5000, 80, SYN, win=1000)),
        (0.001, pkt(srv, cli, 80, 5000, SYN | ACK, win=2000)),
        (0.002, pkt(cli, srv, 5000, 80, ACK)),
        (0.010, pkt(cli, srv, 5000, 80, PSH | ACK, b'x' * 100)),
        (0.030, pkt(srv, cli, 80, 5000, PSH | ACK, b'y' * 300)),
        (0.040, pkt(cli, srv, 5000, 80, FIN | ACK)),
        (0.050, pkt(srv, cli, 80, 5000, FIN | ACK)),
    ]
    udp = dpkt.ip.IP(src=cli, dst=srv, p=17, data=dpkt.udp.UDP(sport=53000, dport=53, data=b'q' * 30))
    udp.data.ulen = 8 + 30
    udp.len = 20 + 8 + 30
    fd, path = tempfile.mkstemp(suffix='.pcap')
    with os.fdopen(fd, 'wb') as fh:
        w = dpkt.pcap.Writer(fh)
        for dt, b in seq:
            w.writepkt(b, t0 + dt)
        w.writepkt(bytes(dpkt.ethernet.Ethernet(type=dpkt.ethernet.ETH_TYPE_IP, data=udp)), t0 + 0.06)
    df = extract(path)
    os.unlink(path)
    assert len(df) == 2, df
    r = df.iloc[0]
    assert (r['Total Fwd Packet'], r['Total Bwd packets']) == (4, 3)
    assert (r['Total Length of Fwd Packet'], r['Total Length of Bwd Packet']) == (100, 300)
    assert r['Flow Duration'] == 50_000
    assert abs(r['Fwd Packet Length Std'] - 50.0) < 1e-6  # [0,0,100,0] sample std
    assert abs(r['Flow IAT Mean'] - 50_000 / 6) < 1e-3 and r['Flow IAT Max'] == 20_000
    assert (r['Fwd IAT Total'], r['Fwd IAT Min']) == (40_000, 2_000)
    assert (r['SYN Flag Count'], r['FIN Flag Count'], r['PSH Flag Count'], r['ACK Flag Count']) == (2, 2, 2, 6)
    assert (r['FWD Init Win Bytes'], r['Bwd Init Win Bytes']) == (1000, 2000)
    assert (r['Fwd Header Length'], r['Fwd Seg Size Min'], r['Fwd Act Data Pkts']) == (80, 20, 1)
    assert abs(r['Flow Bytes/s'] - 400 / 0.05) < 1e-6 and r['Down/Up Ratio'] == 0.75
    u = df.iloc[1]
    assert (u['Protocol'], u['Total Length of Fwd Packet'], u['Flow Duration']) == (17, 30, 0)
    print('flows.py self-check ok')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        out = extract(sys.argv[1])
        print(out.to_string(max_cols=12, max_rows=40))
    else:
        _selfcheck()
