# KNULLD — flow-level cyber attack classification

An ML project that classifies network flows as benign or one of six attack
families, flags traffic unlike anything seen in training, explains each verdict, and
tests how the models hold up on a new network and against an attacker who adapts.

It is a learning and benchmarking tool, not a production IDS. The headline result is the
honest one: LightGBM scores 98 macro-F1 on held-out CIC-IDS2017 traffic. On a different network
(CSE-CIC-IDS2018) it misses most DoS and all brute-force attacks in every training run we tried, and
whether it catches DDoS there depends on the random seed: 0% for the shipped model, ~99% for four
retrained seeds. The console lets you see why.

![KNULLD console: live flow bitmap, verdict log and model votes](docs/img/tap.png)

## Quick start (trained models included)

Needs Python 3.12+ (python.org, tick "Add to PATH"). First run installs packages (~2–5 min, ~1 GB), later runs start in seconds.

- **Windows:** double-click `run.bat`
- **Mac / Linux:** `sh run.sh`

The browser opens at http://localhost:8765. New to the project? Read [GUIDE.md](GUIDE.md).

## Rebuild everything from scratch (optional)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/mac: source .venv/bin/activate)
pip install -r requirements.txt   # GPU torch: see the comment at the top of the file

python fetch_data.py   # ~1.5 GB download, resumable; builds data/
python train.py        # ~12 min on a laptop GPU (CPU works, slower); builds artifacts/
python app.py          # open http://localhost:8765
python flows.py        # optional: extractor self-check
```

## What is in the console

| Screen | What it shows |
|---|---|
| 01 TAP | Replays held-out test flows as a 1-bit bitmap (1 row = 1 flow, 47 feature columns). Lanes show prediction vs ground truth; verdict log, model votes, SHAP reasons, novelty score, running accuracy. |
| 02 DISSECT | One flow in full: all 47 features with SHAP impact. Evasion lab: slow the attack down / pad its packets and watch which models still catch it. |
| 03 BENCH | Model comparison (macro-F1, false alarms, speed, 2018 cross-dataset), confusion matrices, learning curves, overfitting + remedies, perceptron PLA vs pocket, evasion curves, evaluation checks. |
| 04 ZERO-DAY | Leave-one-family-out: what the classifier misses on an unknown attack type and what the autoencoder catches. |
| 05 PCAP | Upload a capture (or use a bundled sample); flows are rebuilt locally and classified. |

## Pipeline

```
fetch_data.py  CIC-IDS2017 (improved, Liu, Engelen et al. 2022) + 3 days of CSE-CIC-IDS2018 (same extractor)
core.py        47 features, label → family mapping, preprocessing, model definitions, inference
train.py       clean → temporal split → 6 classifiers + autoencoder → 10 experiments → artifacts/
flows.py       pcap → the same 47 features (CICFlowMeter conventions), pure Python (dpkt)
app.py         FastAPI: replay scenarios, single-flow + perturbation, pcap upload, static UI
web/           plain HTML/CSS/JS, no build step, fonts bundled (works offline)
```

Models: Perceptron, linear regression (least squares on one-hot targets), logistic
regression, RBF SVM (Nyström features), PyTorch MLP, LightGBM; autoencoder (PyTorch)
trained on benign traffic only for novelty detection.

## Choices worth defending

- **Dataset:** the *improved* CIC-IDS2017 release. The original has known labelling and
  feature-extraction errors. "Attempted" flows (attack tools that produced no attack
  behaviour) are relabelled benign, as the dataset authors recommend. The improved release (2023-04-27)
  still labels ~2.2k scan probes as BENIGN; KNULLD trains and scores on the official labels (see Label audit).
- **Hygiene:** exact duplicates and contradictory rows removed; IPs, ports and timestamps are
  never model inputs (shortcut features; Arp et al., *Dos and Don'ts of ML in Computer Security*).
- **Split:** time-ordered per (day, label): first 60% train, next 20% validation, last 20% test.
  Training benign traffic is capped at 300k flows; the test set keeps the real benign/attack ratio.
- **Metrics:** macro-F1 (rare classes count equally) and false-alarm rate, not accuracy.
- **Cross-dataset:** 2017-trained models evaluated on CSE-CIC-IDS2018 days (same extractor,
  different network). Expect a large drop; that drop is the honest result.
- **Portable features:** only features our own extractor can rebuild from raw packets, so pcap
  upload uses the same models. Costs ~0 macro-F1 versus all 82 columns (see BENCH).

## Limitations

- `flows.py` follows CICFlowMeter's rules but could not be compared byte-for-byte against it:
  the raw CIC-IDS2017 pcaps sit behind a registration form. Its self-check uses hand-computed values.
- Models trained on one network fail on another (2017 → 2018, LightGBM over 5 seeds: ~23% of DoS and 0% of brute force
  caught; the shipped model also misses all DDoS, which other seeds catch), and the autoencoder does not
  reliably flag foreign traffic either. Treat labels on your own pcaps as indicative only.
- Classification is per flow: a lone SYN looks the same in a port scan and in a SYN flood.
- Very rare classes (WebAttack: 104 flows after relabelling) give noisy per-class numbers.
- Evasion is a feature-level what-if (timing stretched, payload padded up to 1460 B, dependent features recomputed),
  not replayed packets; stretching also stretches the server's reply times, which an attacker would not fully control.
- The shipped artifacts are one training run (`SEED = 0` in `train.py`). Retraining with seeds 0–4 on the same split:
  LightGBM 2017 macro-F1 98.4 (range 97.1–99.3), MLP 95.5 (92.7–97.7), LightGBM ahead in 5/5 runs;
  Perceptron 66 (40–80), so the shipped 40 is its worst run. On 2018, LightGBM catches 57% of attacks on
  average (1.7–71%); the shipped model is the 1.7% outlier. Evasion at 8× slow-down is stable (73–74% caught).
- The temporal split cuts inside each (day, label) run, so train and test can share an attack session.
  A per-session or per-day split would be stricter.
- Base rates: the 2017 test set is 16% attacks and the 2018 benign traffic is subsampled to 5%, both far more
  attack-heavy than a real network. False-alarm rates carry over; precision and F1 do not.
- 2017 false alarms depend on 97 likely mislabelled test flows (see Label audit): LightGBM seeds report 10–89 false
  alarms on official labels and 7–10 on corrected labels, with PortScan recall 0.948 → 0.940 (0.894 for two seeds).
- 2018 false-alarm rates depend on label policy: 75–98% of LightGBM's 2018 false alarms are inbound internet
  RDP/SMB traffic that the dataset labels BENIGN by default (see Label audit). Detection numbers are unaffected.
- `Fwd/Bwd Init Win Bytes` and `Fwd Seg Size Min` are set by each host's OS and may act as testbed shortcuts (untested).

## Label audit

Reproduce with `python audit/mirror_audit.py <official 2017 labelling notebook>` (rules are read from
[`CNS2022_Code`](https://github.com/GintsEngelen/CNS2022_Code)); results in `audit/`.

- **CIC-IDS2017 (improved release).** Swapping source and destination in the five official scan-labelling rules
  (Thursday Infiltration–Portscan rounds 1–3, Friday Portscan rounds 1–2) finds 2,228 BENIGN zero-payload flows,
  victim → scanner, inside the scan windows, all carrying the mirrored nmap window signature (Fwd Init Win 0,
  Bwd Init Win 1024 or 29200). None duplicates a labelled probe, and the same host pairs have no flows with this
  signature outside the windows. They look like scan probes whose flow direction was flipped. This is inferred from flow features, not
  from packet captures, and it is a lower bound (the labelling code drops rows with infinite rates). It is a residual
  instance of a known problem: Engelen et al. 2021 describe the direction risk, Lanvin et al. 2022 report this
  Thursday scan, and [CNS2022_Code PR #3](https://github.com/GintsEngelen/CNS2022_Code/pull/3) proposes labelling every
  rule in both directions (Friday included); it is not merged and the distributed release predates it. 75 of the shipped
  LightGBM's 89 test false alarms are these flows (79 counting all 2,471 zero-payload reverse flows in the windows);
  0.031% → 0.0035% when those 2,471 are relabelled. The effect is seed-dependent (see Limitations).
- **CSE-CIC-IDS2018 (our three days).** No such reverse flows. But 27% of the benign rows in our 5% sample (191,969)
  are inbound TCP 3389 (132,038) or 445 (59,931) from ~19k public IPs to internal hosts; the labelling code has no rule
  for these ports, so they are BENIGN by default. 86% carry payload and their intent is not established, so this is a
  label-policy observation, not an error claim, and the set was defined after inspecting false alarms.

## Data

`artifacts/` ships the trained models plus the held-out test split of the improved CIC-IDS2017
(`test.parquet`, `unseen.parquet`) and a small CSE-CIC-IDS2018 sample, so the console works right
after cloning. The IPs in them belong to the dataset testbeds. If you use these files, cite
Sharafaldin et al. (2018) and Liu, Engelen et al. (2022) below. Neither distributor states a
licence; open an issue if you hold rights and want the files removed.

## Sources

- Cantone, Marrocco, Bria — *Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study*,
  IEEE Access 12 (2024), https://doi.org/10.1109/ACCESS.2024.3472907 — the same 2017 ↔ 2018 collapse, measured earlier.
- Liu, Engelen, Lynar, Essam, Joosen — *Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and
  CSE-CIC-IDS-2018* (IEEE CNS 2022) and the improved release used here: https://intrusion-detection.distrinet-research.be/CNS2022/
- Engelen, Rimmer, Joosen — *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study* (2021)
- Sharafaldin et al. — CIC-IDS2017: https://www.unb.ca/cic/datasets/ids-2017.html
- Arp et al. — *Dos and Don'ts of Machine Learning in Computer Security* (USENIX Security 2022)
- Mirsky et al. — *Kitsune* (NDSS 2018), autoencoder-ensemble NIDS (idea for the novelty detector)
- Sample pcaps in `samples/`: https://github.com/ganesh-k13/TitaniumOrca (MIT, see `samples/LICENSE-TitaniumOrca.txt`)

## Licence

Code: MIT (see `LICENSE`). Fonts in `web/fonts/`: SIL OFL 1.1 (see `web/fonts/OFL.txt`).
