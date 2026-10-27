"""Fetch the datasets into data/.

- CIC-IDS2017 (improved release, Liu, Engelen et al. 2022): full, ~330 MB zip.
- CSE-CIC-IDS2018 (same release, same fixed extractor): only the days used for the
  cross-dataset test, pulled out of the 9.7 GB zip with HTTP range requests and
  shrunk to attacks + a benign sample.

The server throttles each connection (~25 KB/s), so everything is fetched as many
parallel ranges. Safe to re-run: finished outputs are skipped.
"""
import concurrent.futures as cf
import pathlib
import struct
import urllib.request
import zipfile
import zlib

import pandas as pd
from remotezip import RemoteZip

BASE = 'https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/'
DATA = pathlib.Path(__file__).parent / 'data'
# days whose attacks overlap with 2017's families: brute force, DoS, DDoS (LOIC)
DAYS_2018 = ['Wednesday-14-02-2018.csv', 'Thursday-15-02-2018.csv', 'Tuesday-20-02-2018.csv']
BENIGN_FRAC_2018 = 0.05
CHUNK = 2 << 20


def get_range(url, start, end, tries=10):
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == end - start + 1:
                return data
        except OSError:
            pass
    raise RuntimeError(f'{url} range {start}-{end} failed after {tries} tries')


def download(url, start, length, out, workers=48):
    ranges = [(s, min(s + CHUNK, start + length) - 1) for s in range(start, start + length, CHUNK)]
    with open(out, 'wb') as f, cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(get_range, url, a, b): a for a, b in ranges}
        for n, fu in enumerate(cf.as_completed(futs), 1):
            f.seek(futs[fu] - start)
            f.write(fu.result())
            print(f'\r  {out.name}: {n}/{len(ranges)} chunks', end='', flush=True)
    print()


def fetch_2017():
    if (DATA / 'cic2017.parquet').exists() or \
            all((DATA / f'{d}.csv').exists() for d in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):
        return print('2017: present')
    url = BASE + 'CICIDS2017_improved.zip'
    size = int(urllib.request.urlopen(urllib.request.Request(url, method='HEAD')).headers['Content-Length'])
    zp = DATA / 'CICIDS2017_improved.zip'
    if not zp.exists() or zp.stat().st_size != size:
        download(url, 0, size, zp)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None, 'corrupt zip, delete it and re-run'
        z.extractall(DATA)
    zp.unlink()
    print('2017: extracted')


def fetch_2018_day(name):
    out = DATA / f"cic2018_{name.split('.')[0]}.parquet"
    if out.exists():
        return print(f'2018 {name}: present')
    url = BASE + 'CSECICIDS2018_improved.zip'
    with RemoteZip(url) as z:
        info = z.getinfo(name)
    assert info.compress_type == zipfile.ZIP_DEFLATED
    hdr = get_range(url, info.header_offset, info.header_offset + 29)
    n, m = struct.unpack('<HH', hdr[26:30])
    raw = DATA / (name + '.deflate')
    if not raw.exists() or raw.stat().st_size != info.compress_size:
        download(url, info.header_offset + 30 + n + m, info.compress_size, raw)

    csv = DATA / name
    dec = zlib.decompressobj(-15)
    with open(raw, 'rb') as src, open(csv, 'wb') as dst:
        while chunk := src.read(16 << 20):
            dst.write(dec.decompress(chunk))
        dst.write(dec.flush())
    assert csv.stat().st_size == info.file_size, 'size mismatch after inflate'

    # keep every attack row, sample benign; the full day is ~3 GB of mostly benign
    parts = []
    for df in pd.read_csv(csv, chunksize=500_000, low_memory=False):
        benign = df['Label'] == 'BENIGN'
        parts += [df[~benign], df[benign].sample(frac=BENIGN_FRAC_2018, random_state=0)]
    pd.concat(parts).to_parquet(out, index=False)
    csv.unlink()
    raw.unlink()
    print(f'2018 {name}: {out.name}')


if __name__ == '__main__':
    DATA.mkdir(exist_ok=True)
    fetch_2017()
    for day in DAYS_2018:
        fetch_2018_day(day)
