'use strict';
// KNULLD front end: plain JS, no build step. Talks to app.py (/api/*).

const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
const kfmt = n => n >= 1e6 ? (n / 1e6).toFixed(2) + 'M' : n >= 1e4 ? Math.round(n / 1e3) + 'k' : n >= 1e3 ? (n / 1e3).toFixed(1) + 'k' : String(n);
const raw = v => Math.abs(v) >= 1e6 ? v.toExponential(2) : Number.isInteger(v) ? String(v) : v.toFixed(2);
async function api(path, opt) {
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
const NAMES = { PLA: 'Perceptron', LinReg: 'Linear reg.', LogReg: 'Logistic reg.', SVM: 'SVM (RBF)', MLP: 'MLP', LGBM: 'LightGBM' };
const ABBR = { BENIGN: 'Benign', DoS: 'DoS', DDoS: 'DDoS', PortScan: 'Scan', BruteForce: 'Brute', WebAttack: 'Web', Bot: 'Bot' };
const short = f => f.replace('Total Length of ', 'Tot Len ').replace('Packet Length', 'Pkt Len').replace(/ [Pp]ackets?$/, ' Pkts').replace('Total ', 'Tot ');
const onTap = () => (location.hash.slice(1) || 'tap') === 'tap';
// keyboard: rows that act on click also act on Enter / Space
const clickable = (el, fn) => { el.tabIndex = 0; el.onclick = fn; el.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } }; };

// ------------------------------------------------------------- tooltip
const tipEl = $('#tip');
function tip(e, text) {
  tipEl.textContent = text; tipEl.style.display = 'block';
  const x = Math.min(e.clientX + 14, innerWidth - tipEl.offsetWidth - 8);
  tipEl.style.left = x + 'px'; tipEl.style.top = e.clientY + 14 + 'px';
}
const untip = () => { tipEl.style.display = 'none'; };

// ------------------------------------------------------------- 1-bit flow bitmap
// 2 css px per bitmap pixel, 3 bitmap px per flow row; ordered (Bayer) dither of each 0..1 feature.
const BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
const PX = 2, RH = 3, LANE = 12;
function bitmap(cv, rows, withTruth) {
  const PW = Math.max(1, Math.floor(cv.clientWidth / PX)), PH = Math.max(1, Math.floor(cv.clientHeight / PX));
  if (cv.width !== PW || cv.height !== PH) { cv.width = PW; cv.height = PH; }
  const ctx = cv.getContext('2d'), img = ctx.createImageData(PW, PH), d = img.data;
  const lanes = withTruth ? 2 : 1, fw = PW - lanes * (LANE + 2), nf = META.features.length, cw = (fw - 2) / nf;
  const RED = [255, 42, 0], BLK = [0, 0, 0], WHT = [255, 255, 255];
  for (let y = 0; y < PH; y++) {
    const r = Math.floor(y / RH), f = rows[r], by = (y - r * RH + (f ? f.i * RH : 0)) & 3;
    for (let x = 0; x < PW; x++) {
      let c = WHT;
      if (f) {
        if (x < fw - 2) {
          const i = Math.floor(x / cw);
          if (x - i * cw >= 1 && f.x[i] > (BAYER[by][x & 3] + 0.5) / 16) c = BLK;
        } else if (x >= fw) {
          const lane = Math.floor((x - fw) / (LANE + 2)), lx = (x - fw) % (LANE + 2);
          if (lx < LANE) {
            if (lane === 0) {
              if (f.novel) c = ((x >> 1) + (y >> 1)) & 1 ? RED : BLK;
              else if (f.pred !== 'BENIGN') c = RED;
              else if (lx === 1 && y % RH === 1) c = BLK;
            } else if (f.truth && f.truth !== 'BENIGN') c = (y % RH === 2) ? WHT : BLK;
            else if (lx === 1 && y % RH === 1) c = BLK;
          }
        }
      }
      const o = (y * PW + x) * 4; d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  cv.dataset.lanes = lanes;
}
const verdictText = f => f.novel ? `NOVEL? (LightGBM: BENIGN p=${f.conf.toFixed(2)}, novelty ${f.novelty.toFixed(2)})` : `${f.label} p=${f.conf.toFixed(2)}`;
function hookBitmap(cv, getRows, onPick) {
  const at = e => {
    const r = cv.getBoundingClientRect(), rows = getRows();
    const f = rows[Math.floor((e.clientY - r.top) / PX / RH)];
    const lanes = cv.dataset.lanes | 0, fw = cv.width - lanes * (LANE + 2);
    const c = Math.floor((e.clientX - r.left) / PX / ((fw - 2) / META.features.length));
    return [f, c];
  };
  cv.addEventListener('mousemove', e => {
    const [f, c] = at(e);
    if (!f) return untip();
    const truth = f.truth ? `\ntruth ${f.truth}` : '';
    tip(e, c < META.features.length ? `#${f.i}  ${META.features[c]} = ${f.x[c].toFixed(2)} (scaled)\n${verdictText(f)}${truth}` : `#${f.i}  ${verdictText(f)}${truth}`);
  });
  cv.addEventListener('mouseleave', untip);
  cv.addEventListener('click', e => { const [f] = at(e); if (f) onPick(f); });
}

// ------------------------------------------------------------- state
let META, METRICS;
const S = { sid: null, flows: [], pos: 0, shown: [], playing: true, speed: 1, pinned: null, sel: null,
  alerts: 0, novel: 0, tally: {}, pcap: [] };

// right = known attack named correctly; benign left alone (no class, no NOVEL? flag); unseen attack alerted somehow
const correct = f => f.truth == null ? null
  : f.truth === 'UNSEEN' ? f.label !== 'BENIGN'
    : f.truth === 'BENIGN' ? f.label === 'BENIGN' : f.pred === f.truth;

// ------------------------------------------------------------- 01 TAP
async function loadScenario(sid) {
  S.sid = sid; S.flows = []; S.pos = 0; S.shown = []; S.pinned = null; S.sel = null;
  S.alerts = 0; S.novel = 0; S.tally = {};
  $$('#scen button').forEach(b => b.classList.toggle('on', b.dataset.sid === sid));
  const sc = META.scenarios.find(s => s.id === sid);
  $('#scen-desc').textContent = 'loading ' + sc.title + '…';
  $('#log').textContent = '';
  const flows = await api('/api/replay/' + sid);
  if (S.sid !== sid) return;  // another scenario was picked meanwhile
  S.flows = flows;
  $('#scen-desc').textContent = sc.desc;
  setPlaying(true);
}

function setPlaying(on) { S.playing = on; $('#play').textContent = on ? 'PAUSE' : S.pos >= S.flows.length ? 'REPLAY' : 'PLAY'; }

function push(f) {
  S.shown.unshift(f);
  S.shown.length = Math.min(S.shown.length, 120);
  const alert = f.label !== 'BENIGN';
  if (alert) S.alerts++;
  if (f.novel) S.novel++;
  if (f.truth) {
    const t = S.tally[f.truth] || (S.tally[f.truth] = { n: 0, ok: 0 });
    t.n++; if (correct(f)) t.ok++;
  }
  const s = document.createElement('span');
  s.className = (f.novel ? 'n' : alert ? 'a' : '') + (correct(f) === false ? ' x' : '');
  s.textContent = `${String(f.i).padStart(4)} ${f.src.padEnd(15).slice(0, 15)} ${String(f.dport).padStart(5)} ${f.label.padEnd(10)} ${f.conf.toFixed(2)}`;
  clickable(s, () => select(f, true));
  const log = $('#log'); log.prepend(s); while (log.children.length > 22) log.lastChild.remove();
  if (!S.sel || ((alert || correct(f) === false) && !S.pinned)) select(f);
}

function renderTap() {
  bitmap($('#bm'), S.shown, true);
  $('#prog').style.width = (S.flows.length ? S.pos / S.flows.length * 100 : 0) + '%';
  $('#pos').textContent = `${S.pos} / ${S.flows.length}`;
  $('#nflows').textContent = S.pos; $('#nalerts').textContent = S.alerts; $('#nnovel').textContent = S.novel;
  const fams = Object.entries(S.tally).sort((a, b) => (a[0] === 'BENIGN') - (b[0] === 'BENIGN') || b[1].n - a[1].n);
  const all = fams.reduce((a, [, t]) => [a[0] + t.ok, a[1] + t.n], [0, 0]);
  $('#tally').innerHTML = fams.map(([k, t]) =>
    `<div class="trow"><span>${k === 'BENIGN' ? 'benign ok' : k === 'UNSEEN' ? 'unseen' : k}</span><div class="hbar ${k === 'BENIGN' ? '' : 'r'}"><div style="width:${t.ok / t.n * 100}%"></div></div><span class="num">${t.ok}/${t.n}</span></div>`).join('') +
    (all[1] ? `<div class="note" style="margin-top:6px">${pct(all[0] / all[1])} of flows right so far. Red bars: share of each attack family caught.
      "benign ok": benign flows left alone (no attack class and no NOVEL? flag).</div>` : '');
}

function select(f, pin) {
  if (pin) S.pinned = S.pinned === f ? null : f;  // clicking the pinned flow again unpins it
  S.sel = { sid: f.sid || S.sid, i: f.i, row: f };
  const agree = META.models.filter(m => f.votes[m] === f.pred).length;
  $('#votes').innerHTML = META.models.map(m =>
    `<div class="cell ${f.votes[m] === f.pred ? 'agree' : 'disagree'}"><small>${NAMES[m]}</small><b>${f.votes[m]}</b></div>`).join('');
  $('#cons').innerHTML = `<b>${agree}/6</b><small>agree with LightGBM → ${esc(f.pred)}${S.pinned === f ? ' · PINNED (click it again to unpin)' : ''}</small>`;
  $('#attr-h').textContent = `log-odds · red = toward ${f.pred}`;
  const mx = Math.max(...f.contrib.map(c => Math.abs(c.v)), 1e-9);
  $('#attr').innerHTML = f.contrib.map(c => `<div class="arow"><span title="${c.f}">${short(c.f)}</span><div class="track"><div class="${c.v > 0 ? 'pos' : 'neg'}"
    style="${c.v > 0 ? 'left' : 'right'}:50%;width:${Math.abs(c.v) / mx * 50}%"></div></div><span class="num">${c.v > 0 ? '+' : '−'}${Math.abs(c.v).toFixed(2)}</span></div>`).join('') +
    `<div class="note">flow #${f.i} · truth: <b>${esc(f.truth ?? 'unknown')}</b>${f.raw_label ? ' (' + esc(f.raw_label) + ')' : ''}</div>`;
  const hot = f.novelty > 1;
  $('#nov').innerHTML = `<div class="big ${hot ? 'hot' : ''}">${f.novelty.toFixed(2)}</div>
    <div class="meter ${hot ? 'hot' : ''}"><div style="width:${Math.min(f.novelty / 3, 1) * 100}%"></div><i style="left:33.3%"></i></div>
    <div class="note">reconstruction error ÷ alarm level. The autoencoder only ever saw benign traffic: above 1 = unlike normal training traffic.
    ${f.novel ? '<br><b class="red">LightGBM says benign, autoencoder disagrees → NOVEL?</b>' : ''}</div>`;
}

let timer;
function startLoop() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (!S.playing || !S.flows.length || !onTap()) return;  // the replay only advances while it is on screen
    for (let k = 0; k < S.speed && S.pos < S.flows.length; k++) push(S.flows[S.pos++]);
    if (S.pos >= S.flows.length) setPlaying(false);
    renderTap();
  }, 110);
}

// ------------------------------------------------------------- 02 DISSECT
const DELAYS = [1, 2, 4, 8, 16, 32];
let D = { base: null, sort: 'imp', sid: null, i: null, req: 0 };
async function openDissect() {
  if (!S.sel) {
    const f = S.flows.find(x => x.truth && x.truth !== 'BENIGN') || S.flows[0];
    if (!f) { $('#dhead').innerHTML = '<span>Load a replay in TAP first.</span>'; return; }
    S.sel = { sid: S.sid, i: f.i, row: f };
  }
  const { sid, i } = S.sel, req = ++D.req;
  $('#delay').value = 0; $('#padb').value = 0; $('#delay-v').textContent = '×1'; $('#pad-v').textContent = '+0 B';
  const base = await api(`/api/flow/${sid}/${i}`);
  if (req !== D.req) return;  // a newer request superseded this one
  Object.assign(D, { sid, i, base });
  renderDissect(base, base);
}
function renderDissect(b, p) {
  const atk = b.label !== 'BENIGN';
  $('#dhead').innerHTML = `<span class="dot">#${b.i}</span>
    <span><b>${esc(b.src)}:${b.sport}</b> → <b>${esc(b.dst)}:${b.dport}</b> · proto ${b.proto} · ${esc(b.ts)}</span>
    <span>verdict <b class="${atk ? 'red' : ''}">${esc(verdictText(b))}</b></span>
    <span>truth <b>${esc(b.truth ?? 'unknown')}</b>${b.raw_label ? ' · ' + esc(b.raw_label) : ''}</span>`;
  $('#shap-h').textContent = `SHAP → ${p.pred} (log-odds)`;
  const idx = Object.fromEntries(META.features.map((f, j) => [f, j]));
  let rows = p.contrib.map(c => ({ ...c, j: idx[c.f] }));
  if (D.sort === 'orig') rows.sort((a, z) => a.j - z.j);
  const mx = Math.max(...rows.map(r => Math.abs(r.v)), 1e-9);
  $('#ftab').innerHTML = rows.map(r => {
    const changed = Math.abs((p.raw[r.f] ?? 0) - (b.raw[r.f] ?? 0)) > 1e-9;
    return `<tr class="${changed ? 'hl' : ''}"><td>${r.f}</td><td class="num">${raw(p.raw[r.f])}</td>
    <td><div class="ubar"><div style="width:${p.x[r.j] * 100}%"></div></div></td>
    <td><div class="track" style="width:140px"><div class="${r.v > 0 ? 'pos' : 'neg'}" style="${r.v > 0 ? 'left' : 'right'}:50%;width:${Math.abs(r.v) / mx * 50}%"></div></div></td>
    <td class="num">${r.v > 0 ? '+' : '−'}${Math.abs(r.v).toFixed(2)}</td></tr>`;
  }).join('');
  $('#cmp').innerHTML = META.models.map(m => `<tr><td>${NAMES[m]}</td><td>${b.votes[m]}</td>
    <td class="${p.votes[m] !== b.votes[m] ? 'flip' : ''}">${p.votes[m]}</td></tr>`).join('') +
    `<tr><td>Autoencoder</td><td>${b.novelty.toFixed(2)}</td><td class="${(p.novelty > 1) !== (b.novelty > 1) ? 'flip' : ''}">${p.novelty.toFixed(2)}</td></tr>`;
  const ev = $('#evres');
  const changed = p !== b;
  let head, cls = '';
  if (!atk) head = 'BENIGN FLOW';
  else if (!changed) head = 'MOVE A KNOB';
  else if (p.label === 'BENIGN') { head = 'EVADED'; cls = 'evaded'; }
  else head = 'STILL CAUGHT';
  const flips = META.models.filter(m => p.votes[m] === 'BENIGN' && b.votes[m] !== 'BENIGN').map(m => NAMES[m]);
  ev.className = 'verdict ' + cls;
  ev.innerHTML = `<b>${head}</b><span>${!atk ? 'Nothing to evade: LightGBM already calls it benign. Use "next attack →".'
    : !changed ? 'Stretch the timing or pad the packets, then watch which models still catch it.'
      : flips.length ? 'now called BENIGN by: ' + flips.join(', ') : 'no model flipped to benign'}</span>`;
}
let evT;
function onKnob() {
  const delay = DELAYS[$('#delay').value], pad = +$('#padb').value;
  $('#delay-v').textContent = '×' + delay; $('#pad-v').textContent = '+' + pad + ' B';
  clearTimeout(evT);
  evT = setTimeout(async () => {
    if (!D.base) return;
    const req = ++D.req;
    const p = delay === 1 && pad === 0 ? D.base : await api(`/api/flow/${D.sid}/${D.i}?delay=${delay}&pad=${pad}`);
    if (req === D.req) renderDissect(D.base, p);
  }, 140);
}
function stepAttack(dir) {
  const sid = D.sid || S.sid, list = sid === 'pcap' ? S.pcap : S.flows;
  if (!list.length) return;
  let i = D.i ?? -1;
  for (let k = 0; k < list.length; k++) {
    i = (i + dir + list.length) % list.length;
    const f = list[i];
    if ((f.truth && f.truth !== 'BENIGN') || (!f.truth && f.label !== 'BENIGN')) { S.sel = { sid, i, row: f }; if (sid !== 'pcap') S.pinned = f; break; }
  }
  openDissect();
}

// ------------------------------------------------------------- charts (SVG, 1-bit)
// One y axis, thin marks, legend + direct end labels, crosshair tooltip.
function lineChart(el, { series, xlog, xfmt = String, yfmt = v => v.toFixed(2), ymin = 0, ymax, xlabel, ylabel, vline, H = 230 }) {
  const W = Math.max(320, el.clientWidth - 20), m = { l: 48, r: 110, t: 12, b: 30 };  // viewBox = real px: text stays 9px
  const xs = [...new Set(series.flatMap(s => s.pts.map(p => p[0])))].sort((a, b) => a - b);
  const tx = v => xlog ? Math.log10(v) : v;
  const x0 = tx(xs[0]), x1 = tx(xs[xs.length - 1]);
  ymax = ymax ?? Math.max(...series.flatMap(s => s.pts.map(p => p[1])));
  const X = v => m.l + (tx(v) - x0) / (x1 - x0 || 1) * (W - m.l - m.r);
  const Y = v => m.t + (1 - (Math.min(Math.max(v, ymin), ymax) - ymin) / (ymax - ymin || 1)) * (H - m.t - m.b);
  const yt = [0, .25, .5, .75, 1].map(k => ymin + k * (ymax - ymin));
  const step = (() => { const r = (xs[xs.length - 1] - xs[0]) / 6, p = 10 ** Math.floor(Math.log10(r || 1)); return [1, 2, 5, 10].map(k => k * p).find(k => k >= r) || p; })();
  const xt = xs.length <= 8 || xlog ? xs.filter((_, i) => xs.length <= 8 || i % Math.ceil(xs.length / 8) === 0)
    : xs.filter(v => v % step === 0 || v === xs[0]);
  // end labels, nudged apart
  const ends = series.map(s => ({ s, y: Y(s.pts[s.pts.length - 1][1]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 11);
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel)} by ${esc(xlabel)}">
    <g class="grid">${yt.map(v => `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}"/>`).join('')}</g>
    <g class="ax"><line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}"/><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H - m.b}"/></g>
    ${yt.map(v => `<text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${yfmt(v)}</text>`).join('')}
    ${xt.map(v => `<text x="${X(v)}" y="${H - m.b + 13}" text-anchor="middle">${xfmt(v)}</text>`).join('')}
    <text x="${(m.l + W - m.r) / 2}" y="${H - 2}" text-anchor="middle">${xlabel}</text>
    <text x="10" y="${m.t + 4}" transform="rotate(-90 10 ${m.t + 4})" text-anchor="end">${ylabel}</text>
    ${vline ? `<line x1="${X(vline.x)}" x2="${X(vline.x)}" y1="${m.t}" y2="${H - m.b}" stroke="#FF2A00" stroke-dasharray="3 3"/>
      <text x="${X(vline.x) + 4}" y="${m.t + 10}" style="fill:#FF2A00">${vline.label}</text>` : ''}
    ${series.map(s => `<path d="${s.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join('')}"
      fill="none" stroke="${s.red ? '#FF2A00' : '#000'}" stroke-width="2" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`).join('')}
    ${ends.map(({ s, y }) => `<text x="${W - m.r + 6}" y="${y + 3}" ${s.red ? 'style="fill:#FF2A00"' : ''}>${s.name}</text>`).join('')}
    <line class="xh" y1="${m.t}" y2="${H - m.b}" stroke="#000" stroke-width="1" visibility="hidden"/>
    <rect x="${m.l}" y="${m.t}" width="${W - m.l - m.r}" height="${H - m.t - m.b}" fill="transparent"/>
  </svg>`;
  const svg = el.querySelector('svg'), xh = svg.querySelector('.xh');
  svg.querySelector('rect').addEventListener('mousemove', e => {
    const r = svg.getBoundingClientRect(), px = (e.clientX - r.left) / r.width * W;
    const nx = xs.reduce((a, v) => Math.abs(X(v) - px) < Math.abs(X(a) - px) ? v : a, xs[0]);
    xh.setAttribute('x1', X(nx)); xh.setAttribute('x2', X(nx)); xh.setAttribute('visibility', 'visible');
    tip(e, `${xlabel} ${xfmt(nx)}\n` + series.map(s => { const p = s.pts.find(q => q[0] === nx); return p ? `${s.name}: ${yfmt(p[1])}` : null; }).filter(Boolean).join('\n'));
  });
  svg.querySelector('rect').addEventListener('mouseleave', () => { xh.setAttribute('visibility', 'hidden'); untip(); });
}

const panel = (cls, title, right, body) => `<div class="${cls}"><div class="bar"><span>${title}</span><span>${right}</span></div>${body}</div>`;
const legend = items => `<div class="legend">${items.map(([t, c]) => `<span><i class="${c}"></i>${t}</span>`).join('')}</div>`;

// ------------------------------------------------------------- 03 BENCH
let B = { model: 'LGBM', ds: '2017', pair: null, ev: 'delay', done: false };
function benchSkeleton() {
  const M = METRICS, cross = M.crossdataset?.models || {}, d = M.dataset, mm = M.models;
  const web = mm.LGBM.per_class.WebAttack.support, ben = mm.LGBM.per_class.BENIGN.support;
  const cc = d.class_counts;
  $('#bench').innerHTML =
    panel('s7', 'Models · 2017 held-out test', 'click a row → per-class + confusion matrix',
      `<div class="scrollx"><table class="mtab"><thead><tr><th>MODEL</th><th class="num">MACRO-F1</th><th class="num">FALSE ALARMS</th><th class="num">ATTACKS CAUGHT</th>
       <th class="num">2018 F1</th><th class="num">µs / FLOW</th><th class="num">TRAIN</th></tr></thead><tbody id="mtab"></tbody></table></div>
       <div class="pad note">Macro-F1 = F1 averaged over the 7 classes, so the ${web}-flow WebAttack test class counts as much as ${kfmt(ben)} benign flows.
       False alarms = benign flows flagged as an attack. 2018 F1 = same models on CSE-CIC-IDS2018 (different network, never trained on). F1 × 100.</div>
       <div class="bar"><span>Per class · <span id="pc-model"></span></span><span id="pc-ds"></span></div>
       <div class="scrollx"><table class="mtab"><thead><tr><th>CLASS</th><th class="num">FLOWS</th><th class="num">PRECISION</th><th class="num">RECALL</th><th class="num">F1</th></tr></thead><tbody id="pcls"></tbody></table></div>`) +
    panel('s5 end', 'Confusion matrix', '<span id="cm-title"></span>',
      `<div class="seg"><button data-ds="2017" class="on">2017 test</button><button data-ds="2018">2018 unseen network</button></div>
       <div class="scrollx"><div class="cm" id="cm"></div></div>
       <div class="pad note">Rows = truth, columns = prediction, shading = share of the row. Red outline = mistakes. Hover a cell for counts.</div>`) +
    panel('s12 end', 'Data & models', 'flows per class after cleaning · what each model is',
      `<div class="dm"><div>${`<div class="scrollx"><table class="mtab"><thead><tr><th>CLASS</th><th class="num">TRAIN</th><th class="num">VALIDATION</th><th class="num">TEST</th></tr></thead>
       <tbody>${META.classes.map(c => `<tr class="nohover"><td>${c}</td>${['train', 'val', 'test'].map(s => `<td class="num">${(cc[s][c] || 0).toLocaleString('en-US')}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
       <div class="pad note">Benign training flows capped at 300k (speed); validation and test keep the real ratio. Heartbleed (11) and Infiltration (36) are
       too rare to learn and are kept out entirely as the never-seen set (ZERO-DAY). Splits: first 60% of every (day, label) in time → train, next 20% → validation, last 20% → test.</div>`}</div><div>${`<div class="pad note card">
       <p><b>Perceptron</b> — Rosenblatt rule, one-vs-rest, ≤ 50 epochs.</p>
       <p><b>Linear reg.</b> — least squares on one-hot class targets (+ small L2), predict the largest output.</p>
       <p><b>Logistic reg.</b> — softmax over 7 classes, L-BFGS.</p>
       <p><b>SVM (RBF)</b> — RBF kernel approximated with 1000 Nyström features + linear SVM; trained on 100k flows (a full kernel SVM does not fit in memory).</p>
       <p><b>MLP</b> — 47 → 256 → 128 → 7, ReLU, dropout 0.2, AdamW, class-weighted cross-entropy, early stopping (epoch ${mm.MLP.best_epoch}). PyTorch.</p>
       <p><b>LightGBM</b> — gradient-boosted trees, 63 leaves, ≤ 600 trees with early stopping on validation. The reference model.</p>
       <p><b>Autoencoder</b> — 47 → 32 → 12 → 32 → 47, trained on benign flows only; alarm above the 99th percentile of benign error.</p>
       <p><b>ANN vs SVM:</b> MLP ${Math.round(mm.MLP.macro_f1 * 100)} vs SVM ${Math.round(mm.SVM.macro_f1 * 100)} macro-F1, ${mm.MLP.us_per_flow} vs ${mm.SVM.us_per_flow} µs per flow.</p></div>`}</div></div>`) +
    panel('s6', 'Learning curves', 'more data → ?', `${legend([['train', 'dash'], ['validation', '']])}<div class="chart" id="lc"></div>
       <div class="pad note">Gap between train and validation = overfitting. LightGBM fits its training set perfectly (train F1 = 1.00) yet keeps improving on validation with more data;
       logistic regression is too simple — both curves stall lower (underfitting).</div>`) +
    panel('s6 end', 'Overfitting & remedies · MLP', 'big network, 100 flows per class',
      `${legend([['train loss', 'dot'], ['validation · no remedy', 'red'], ['validation · weight decay', 'dash'], ['validation · 30× more data', '']])}
       <div class="chart" id="of"></div><div class="pad note" id="of-note"></div>`) +
    panel('s4', 'Perceptron · 2-D projection', 'PLA vs pocket', `<div class="seg" id="pairs"></div>${legend([['PLA line', 'dash'], ['pocket line', 'red']])}<div class="chart" id="pc"></div><div class="pad note" id="pc-note"></div>`) +
    panel('s8 end', 'Evasion · feature-level what-if', '<span id="ev-title"></span>', `<div class="seg"><button data-ev="delay" class="on">stretch timing</button><button data-ev="pad">pad packets</button></div>
       <div class="chart" id="ev"></div><div class="pad note" id="ev-note"></div>`) +
    (M.crossdataset ? panel('s12 end', '2017 → 2018 · same extractor, unseen network', `${kfmt(M.crossdataset.n)} flows from ${M.crossdataset.days.length} days of CSE-CIC-IDS2018`,
      `<div class="xgrid">${META.models.map(m => {
        const a = mm[m], c = cross[m];
        return `<div class="xcol"><b>${NAMES[m]}</b><div class="xf"><span class="dot">${Math.round(a.macro_f1 * 100)}</span> → <span class="dot red">${Math.round(c.macro_f1 * 100)}</span> <small>F1</small></div>
          <div class="xl">attacks caught</div><div class="hbar r"><div style="width:${c.attack_detection * 100}%"></div><span>${pct(c.attack_detection, 0)}</span></div>
          <div class="xl">benign falsely flagged</div><div class="hbar"><div style="width:${c.false_alarm_rate * 100}%"></div><span>${pct(c.false_alarm_rate, 0)}</span></div></div>`;
      }).join('')}</div>
      <div class="pad note">Every model trained on the 2017 network breaks on the 2018 one. The simple models keep a higher F1 only by alarming on huge
      shares of benign traffic; this LightGBM stays quiet on benign traffic and misses almost every attack. Retrained with other seeds it catches ~99% of
      DDoS, but DoS (~23%) and brute force (0%) fail in every run. Read F1 together with the two bars.
      2018 F1 averages the 4 classes present there (benign, DoS, DDoS, brute force). Cross-network generalisation is an open problem: Cantone et al. (IEEE Access 2024)
      measured the same collapse on this dataset pair.</div>`) : '') +
    panel('s12 end', 'Honest evaluation', 'what we checked before trusting any number',
      `<div class="facts"><div class="fact"><span class="bigm">${d.raw.toLocaleString('en-US')} → ${d.clean.toLocaleString('en-US')}</span><span>flows after removing ${kfmt(d.duplicates)} duplicates
       (same 47 features, same class) and ${d.conflicting} contradictory rows (same features, different class).</span></div>
       <div class="fact"><span class="dot">${Math.round(M.protocol.random * 100)} vs ${Math.round(M.protocol.temporal * 100)}</span><span>macro-F1, random split vs our time-ordered split.
       The gap is small because duplicates were removed first; on raw data a random split puts copies of training flows into the test set.</span></div>
       <div class="fact"><span class="dot">${M.features.portable.n} vs ${M.features.all.n}</span><span>features: ${Math.round(M.features.portable.macro_f1 * 100)} vs ${Math.round(M.features.all.macro_f1 * 100)} macro-F1.
       We keep only features our own extractor can rebuild from any pcap — it costs nothing.</span></div></div>`);
  $('#mtab').innerHTML = META.models.map(m => {
    const r = mm[m];
    return `<tr data-m="${m}"><td><b>${NAMES[m]}</b></td><td class="num dot">${Math.round(r.macro_f1 * 100)}</td><td class="num">${pct(r.false_alarm_rate, 2)}</td>
      <td class="num">${pct(r.attack_detection)}</td><td class="num dot red">${cross[m] ? Math.round(cross[m].macro_f1 * 100) : '–'}</td>
      <td class="num">${r.us_per_flow}</td><td class="num">${r.train_s}s</td></tr>`;
  }).join('');
  $$('#mtab tr').forEach(tr => clickable(tr, () => { B.model = tr.dataset.m; renderBench(); }));
  $$('[data-ds]').forEach(b => b.onclick = () => { B.ds = b.dataset.ds; renderBench(); });
  const pairs = Object.keys(M.perceptron).sort((a, b) => (b === 'DoS_vs_DDoS') - (a === 'DoS_vs_DDoS'));
  B.pair = pairs[0];
  $('#pairs').innerHTML = pairs.map(k => `<button data-pair="${k}">${k.replace('_vs_', ' vs ')}</button>`).join('');
  $$('[data-pair]').forEach(b => b.onclick = () => { B.pair = b.dataset.pair; renderBench(); });
  $$('[data-ev]').forEach(b => b.onclick = () => { B.ev = b.dataset.ev; renderBench(); });
  renderStatic();
}

function renderStatic() {  // charts that never change with a control
  const M = METRICS, L = M.learning;
  lineChart($('#lc'), {
    xlog: true, xfmt: kfmt, ymax: 1, xlabel: 'training flows', ylabel: 'macro-F1',
    series: [
      { name: 'LGBM val', red: true, pts: L.LGBM.map(r => [r.n, r.val_f1]) },
      { name: 'LGBM train', red: true, dash: '5 4', pts: L.LGBM.map(r => [r.n, r.train_f1]) },
      { name: 'LogReg val', pts: L.LogReg.map(r => [r.n, r.val_f1]) },
      { name: 'LogReg train', dash: '5 4', pts: L.LogReg.map(r => [r.n, r.train_f1]) },
    ],
  });
  const O = M.overfit, o = O.overfit.history, best = O.overfit.best_epoch;
  // 5-epoch moving average: small-batch training makes single epochs spiky
  const ma = pts => pts.map((p, i) => { const w = pts.slice(Math.max(0, i - 4), i + 1); return [p[0], w.reduce((a, q) => a + q[1], 0) / w.length]; });
  const vl = k => ma(O[k].history.map(r => [r.epoch, r.val_loss])), end = k => O[k].history[O[k].history.length - 1];
  lineChart($('#of'), {
    xlabel: 'epoch (5-epoch moving average)', ylabel: 'loss', yfmt: v => v.toFixed(2), ymax: 0.6,  // first epochs (loss > 1) and a weight-decay blip run off the top
    series: [{ name: 'train', dash: '1 3', pts: ma(o.map(r => [r.epoch, r.train_loss])) },
             { name: 'no remedy', red: true, pts: vl('overfit') },
             { name: 'weight decay', dash: '5 4', pts: vl('weight_decay') },
             { name: 'more data', pts: vl('more_data') }],
    vline: { x: best, label: 'early stop' },
  });
  $('#of-note').innerHTML = `<b>No remedy:</b> training loss falls to ${end('overfit').train_loss.toFixed(4)} (the ${O.overfit.train_n} flows are memorised)
    while validation loss is lowest at epoch ${best} (${o[best - 1].val_loss.toFixed(2)}) and then climbs to ${end('overfit').val_loss.toFixed(2)} — overfitting.
    Validation F1 drops from ${pct(o[best - 1].val_f1, 0)} to ${pct(end('overfit').val_f1, 0)}.<br>
    <b>Early stopping</b> keeps the epoch-${best} weights. <b>Weight decay</b> keeps the loss lower (${end('weight_decay').val_loss.toFixed(2)}) but does not raise F1
    here (${pct(end('weight_decay').val_f1, 0)}). <b>More data</b> (${kfmt(O.more_data.train_n)} flows) works best: loss ${end('more_data').val_loss.toFixed(2)}, F1 ${pct(end('more_data').val_f1, 0)}.`;
}

function renderBench() {
  const M = METRICS, cross = M.crossdataset?.models || {};
  if (!B.done) { benchSkeleton(); B.done = true; }
  $$('#mtab tr').forEach(tr => tr.classList.toggle('on', tr.dataset.m === B.model));
  $$('[data-ds]').forEach(b => b.classList.toggle('on', b.dataset.ds === B.ds));
  $$('[data-pair]').forEach(b => b.classList.toggle('on', b.dataset.pair === B.pair));
  $$('[data-ev]').forEach(b => b.classList.toggle('on', b.dataset.ev === B.ev));

  // per-class table + confusion matrix
  const src = B.ds === '2018' ? cross[B.model] : M.models[B.model];
  $('#pc-model').textContent = NAMES[B.model]; $('#pc-ds').textContent = B.ds === '2018' ? '2018 unseen network' : '2017 test';
  $('#pcls').innerHTML = src ? META.classes.filter(c => src.per_class[c].support).map(c => { const r = src.per_class[c];
    return `<tr class="nohover"><td>${c}</td><td class="num">${r.support.toLocaleString('en-US')}</td><td class="num">${pct(r.precision)}</td><td class="num">${pct(r.recall)}</td><td class="num"><b>${pct(r.f1)}</b></td></tr>`; }).join('') : '';
  $('#cm-title').textContent = NAMES[B.model];
  const C = META.classes, cm = src?.confusion, el = $('#cm');
  el.style.gridTemplateColumns = `84px repeat(${C.length}, 1fr)`;
  if (!cm) el.innerHTML = '<div class="h">no 2018 data</div>';
  else {
    el.innerHTML = '<div class="h">truth ↓ pred →</div>' + C.map(c => `<div class="h">${ABBR[c]}</div>`).join('') +
      cm.map((row, i) => {
        const n = row.reduce((a, b) => a + b, 0);
        return `<div class="h">${C[i]}</div>` + row.map((v, j) => {
          const s = n ? v / n : 0, lv = v === 0 ? 0 : s < .05 ? 1 : s < .3 ? 2 : s < .7 ? 3 : 4;
          return `<div class="v${lv} ${i !== j ? 'off' : ''}" data-t="${C[i]} → ${C[j]}: ${v.toLocaleString('en-US')} (${n ? pct(s) : '–'} of ${C[i]})"><span>${v ? kfmt(v) : ''}</span></div>`;
        }).join('');
      }).join('');
    el.querySelectorAll('[data-t]').forEach(c => { c.onmousemove = e => tip(e, c.dataset.t); c.onmouseleave = untip; });
  }

  // perceptron 2-D (points beyond the frame are drawn on its edge)
  const P = M.perceptron[B.pair], W = 300, Hh = 230, lim = 3.2;
  const cl = v => Math.max(-lim, Math.min(lim, v));
  const sx = v => (cl(v) + lim) / (2 * lim) * W, sy = v => Hh - (cl(v) + lim) / (2 * lim) * Hh;
  const line = (w, attrs) => {
    if (Math.abs(w[2]) < 1e-9) return '';
    const y = x => -(w[0] + w[1] * x) / w[2];
    return `<line x1="${(-lim + lim) / (2 * lim) * W}" y1="${Hh - (y(-lim) + lim) / (2 * lim) * Hh}" x2="${W}" y2="${Hh - (y(lim) + lim) / (2 * lim) * Hh}" ${attrs}/>`;
  };
  const [ca, cb] = B.pair.split('_vs_');
  $('#pc').innerHTML = `<svg viewBox="0 0 ${W} ${Hh}" role="img" aria-label="perceptron decision lines, ${ca} vs ${cb}"><defs><clipPath id="clp"><rect width="${W}" height="${Hh}"/></clipPath></defs>
    <rect width="${W}" height="${Hh}" fill="none" stroke="#000"/>
    <g clip-path="url(#clp)">${P.points.map((p, i) => P.y[i] > 0
      ? `<rect x="${sx(p[0]) - 2}" y="${sy(p[1]) - 2}" width="4" height="4" fill="#FF2A00"/>`
      : `<rect x="${sx(p[0]) - 1.5}" y="${sy(p[1]) - 1.5}" width="3" height="3" fill="none" stroke="#000"/>`).join('')}
    ${line(P.pla.w, 'stroke="#000" stroke-width="2" stroke-dasharray="5 4"')}${line(P.pocket.w, 'stroke="#FF2A00" stroke-width="2"')}</g></svg>`;
  const ep = P.pla.errors, pk = P.pocket.errors;
  $('#pc-note').innerHTML = `□ ${ca} · <span class="red">■ ${cb}</span>, 300 training flows each, projected to 2-D with PCA. ${ep[ep.length - 1] === 0
    ? `Linearly separable in this projection: PLA reached <b>0 errors</b> after ${ep.length} epoch(s).`
    : `Not separable in this projection: PLA still makes <b>${pct(ep[ep.length - 1])}</b> errors after ${ep.length} epochs and keeps jumping;
       pocket PLA keeps the best weights seen: <b>${pct(pk[pk.length - 1])}</b>.`}`;

  // evasion
  const E = M.evasion[B.ev], key = B.ev, isD = key === 'delay';
  const xf = v => isD ? '×' + v : '+' + v + ' B';
  $('#ev-title').textContent = isD ? 'every gap and the duration × k' : 'extra payload bytes per forward packet (≤ 1460 B)';
  const show = [['LGBM', true, ''], ['MLP', false, '8 2 2 2'], ['SVM', false, '5 4'], ['LogReg', false, ''], ['AE', true, '1 3']];
  lineChart($('#ev'), {
    xlog: isD, xfmt: xf, ymax: 1, yfmt: v => pct(v, 0), xlabel: isD ? 'slow-down factor' : 'padding', ylabel: 'attack flows flagged',
    series: show.map(([m, red, dash]) => ({ name: m === 'AE' ? 'Autoencoder' : NAMES[m], red, dash, pts: E.map(r => [r[key], r[m]]) })),
  });
  const lg = E.map(r => r.LGBM), lo = Math.min(...lg), at = E[lg.indexOf(lo)][key];
  const bestM = ['LGBM', 'MLP', 'SVM', 'LogReg'].reduce((a, m) => E[E.length - 1][m] > E[E.length - 1][a] ? m : a, 'LGBM');
  $('#ev-note').innerHTML = `Share of the same attack test flows still flagged after the change. LightGBM drops to <b>${pct(lo, 0)}</b> at ${xf(at)};
    at the largest change ${NAMES[bestM]} holds best (${pct(E[E.length - 1][bestM], 0)}). ${isD ? '' : 'Big padding makes flows look strange, so the autoencoder starts flagging them. '}
    What-if on features, not replayed packets; stretching also stretches the server's replies, which a real attacker could not fully control.`;
}

// ------------------------------------------------------------- 04 ZERO-DAY
function renderZD() {
  const Z = METRICS.zeroday, N = METRICS.novelty;
  const hb = (v, c) => `<div class="hbar ${c}"><div style="width:${v * 100}%"></div><span>${pct(v, 0)}</span></div>`;
  const said = r => r.predicted_as ? Object.entries(r.predicted_as).sort((a, b) => b[1] - a[1])[0] : null;
  const tot = Object.values(N.unseen).reduce((a, v) => a + v.n, 0);
  $('#zd').innerHTML =
    `<div class="rule-r"><div class="bar"><span>Leave one family out</span><span>retrain LightGBM without it, then show it that attack</span></div>
      <div class="pad note">Simulates a new attack type the classifier has never seen. It can only answer with a family it knows or BENIGN — does anything still raise an alarm?</div>
      <div class="scrollx"><div class="zrow head"><span>FAMILY LEFT OUT</span><span class="num">TEST N</span><span>CLASSIFIER ALARMS</span><span>AUTOENCODER ALARMS</span><span>EITHER</span><span>CLASSIFIER MOSTLY SAID</span></div>
      ${Object.entries(Z).map(([f, r]) => { const s = said(r);
        return `<div class="zrow"><b>${f}</b><span class="num">${kfmt(r.n)}</span>${hb(r.classifier_alerts, '')}${hb(r.ae_alerts, 'c')}${hb(r.either, 'r')}<span>${s ? `${s[0]} (${pct(s[1], 0)})` : ''}</span></div>`; }).join('')}</div>
      <div class="pad note">The autoencoder only flags traffic that looks unlike normal flows (scans, Heartbleed). Attacks that look like ordinary sessions —
      HTTP floods (DoS here is mostly Hulk), password guessing, web attacks, botnet beacons — pass both detectors once the classifier has never seen them.
      Novelty detection is a safety net with holes, not a cure.</div></div>
    <div><div class="bar"><span>Never seen at all</span><span>not in any training class</span></div>
      <div class="never">
        ${Object.entries(N.unseen).map(([k, v]) => `<div><span class="dot">${v.flagged}/${v.n}</span><br><b>${k}</b> flows flagged by the autoencoder</div>`).join('')}
        <div><span class="dot red">${Math.round(N.unseen_lgbm_says_benign * tot)}/${tot}</span><br>of the same flows called <b>BENIGN</b> by LightGBM</div>
        <div class="note">Tiny samples: 11 Heartbleed and 36 Infiltration flows exist in the whole dataset. (Infiltration's later scanning stage was labelled
          "Infiltration - Portscan" and trained as PortScan.) Autoencoder alarm level = 99th percentile of benign reconstruction error, so it falsely flags
          <b>${pct(N.false_alarm_rate)}</b> of benign test flows.</div>
        <button id="zd-replay">replay zero-day scenario →</button>
      </div></div>`;
  $('#zd-replay').onclick = () => { location.hash = '#tap'; loadScenario('zero-day'); };
}

// ------------------------------------------------------------- 05 PCAP
let pcapBusy = false;
async function classifyPcap(req) {
  if (pcapBusy) return;
  pcapBusy = true;
  $$('#samples button, #file').forEach(b => { b.disabled = true; });
  $('#pres').innerHTML = '<div class="loading">EXTRACTING FLOWS</div>';
  try { S.pcap = (await req()).map(f => ({ ...f, sid: 'pcap' })); }
  catch (e) { $('#pres').innerHTML = `<div class="warn">${esc(e.message)}</div>`; return; }
  finally { pcapBusy = false; $$('#samples button, #file').forEach(b => { b.disabled = false; }); }
  const rows = S.pcap, by = {};
  rows.forEach(f => { by[f.label] = (by[f.label] || 0) + 1; });
  const odd = rows.filter(f => f.novelty > 1).length;
  $('#pres').innerHTML = `<div class="sumrow"><div><small>FLOWS${rows.length >= 5000 ? ' (first 5000)' : ''}</small><b>${rows.length}</b></div>${Object.entries(by).sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `<div class="${k === 'BENIGN' ? '' : 'red'}"><small>${esc(k)}</small><b>${v}</b></div>`).join('')}</div>
    ${odd / rows.length > 0.3 ? `<div class="warn"><b>${pct(odd / rows.length, 0)} of these flows have novelty above 1</b>: they look unlike the benign training traffic.</div>` : ''}
    <div class="bitmap"><canvas class="bm" id="pbm" role="img" aria-label="flow bitmap of the capture" style="height:${Math.min(440, Math.max(60, rows.length * RH * PX))}px"></canvas></div>
    <div class="ptab"><table><thead><tr><th>#</th><th>SOURCE</th><th>DESTINATION</th><th>PROTO</th><th>VERDICT</th><th class="num">P</th><th class="num">NOVELTY</th></tr></thead>
    <tbody>${rows.slice(0, 500).map(f => `<tr data-i="${f.i}"><td>${f.i}</td><td>${esc(f.src)}:${f.sport}</td><td>${esc(f.dst)}:${f.dport}</td><td>${f.proto}</td>
      <td class="${f.label === 'BENIGN' ? '' : 'red'}"><b>${esc(f.label)}</b></td><td class="num">${f.conf.toFixed(2)}</td><td class="num ${f.novelty > 1 ? 'red' : ''}">${f.novelty.toFixed(2)}</td></tr>`).join('')}</tbody></table></div>`;
  const cv = $('#pbm'), open = f => { S.sel = { sid: 'pcap', i: f.i, row: f }; location.hash = '#dissect'; };
  bitmap(cv, rows, false);
  hookBitmap(cv, () => rows, open);
  $$('.ptab tr[data-i]').forEach(tr => clickable(tr, () => open(rows[+tr.dataset.i])));
}
function setupPcap(samples) {
  $('#samples').innerHTML = '<div class="note"><b>Bundled samples</b> (small public attack captures) — what is inside and what to expect:</div>' +
    samples.map(s => `<div class="srow"><button data-sample="${esc(s.name)}">${esc(s.name)}</button><span class="note">${esc(s.desc)}</span></div>`).join('');
  $$('[data-sample]').forEach(b => b.onclick = () => classifyPcap(() => api('/api/pcap/sample/' + encodeURIComponent(b.dataset.sample), { method: 'POST' })));
  const send = file => { const fd = new FormData(); fd.append('file', file); classifyPcap(() => api('/api/pcap', { method: 'POST', body: fd })); };
  $('#file').onchange = e => { e.target.files[0] && send(e.target.files[0]); e.target.value = ''; };
  const drop = $('#drop');
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = e => { e.preventDefault(); drop.classList.remove('over'); e.dataTransfer.files[0] && send(e.dataTransfer.files[0]); };
  const c = METRICS.crossdataset?.models?.LGBM;
  $('#pcap-caveat').innerHTML = `<b>Read the labels with care.</b> The models learned one lab network (CIC-IDS2017). On a different network
    (CSE-CIC-IDS2018) LightGBM caught only ${c ? pct(c.attack_detection, 1) : 'a fraction'} of attacks, and the autoencoder did not reliably flag that
    traffic as novel either. Classification is per flow: a single SYN looks the same whether it is part of a scan or a flood.`;
}

// ------------------------------------------------------------- router + boot
function route() {
  const id = location.hash.slice(1) || 'tap';
  $$('section[data-screen]').forEach(s => s.classList.toggle('on', s.dataset.screen === id));
  $$('nav a').forEach(a => a.classList.toggle('on', a.dataset.nav === id));
  $('.ctr').style.display = id === 'tap' ? '' : 'none';  // replay counters belong to TAP only
  untip();
  if (id === 'tap') renderTap();
  if (id === 'dissect') openDissect().catch(e => { $('#dhead').innerHTML = `<span class="red">${esc(e.message)}</span>`; });
  if (id === 'bench') renderBench();
  if (id === 'zeroday') renderZD();
}

(async function boot() {
  [META, METRICS] = await Promise.all([api('/api/meta'), api('/api/metrics')]);
  $('#cols').style.gridTemplateColumns = `repeat(${META.features.length}, 1fr) ${(2 * (LANE + 2)) * PX}px`;
  $('#cols').innerHTML = META.features.map((_, i) => `<span>${i % 5 === 0 ? i : ''}</span>`).join('') + '<span>P&nbsp;&nbsp;T</span>';
  $('#scen').innerHTML = META.scenarios.map(s => `<button data-sid="${s.id}">${esc(s.title)}<small>${s.n} flows</small></button>`).join('');
  $$('#scen button').forEach(b => b.onclick = () => loadScenario(b.dataset.sid));
  $('#play').onclick = () => { if (S.pos >= S.flows.length) loadScenario(S.sid); else setPlaying(!S.playing); };
  $$('[data-speed]').forEach(b => b.onclick = () => { S.speed = +b.dataset.speed; $$('[data-speed]').forEach(x => x.classList.toggle('on', x === b)); });
  $('#to-dissect').onclick = () => { location.hash = '#dissect'; };
  $('#prev-atk').onclick = () => stepAttack(-1);
  $('#next-atk').onclick = () => stepAttack(1);
  $('#sort-imp').onclick = () => { D.sort = 'imp'; $('#sort-imp').classList.add('on'); $('#sort-orig').classList.remove('on'); onKnob(); };
  $('#sort-orig').onclick = () => { D.sort = 'orig'; $('#sort-orig').classList.add('on'); $('#sort-imp').classList.remove('on'); onKnob(); };
  $('#delay').oninput = onKnob; $('#padb').oninput = onKnob;
  hookBitmap($('#bm'), () => S.shown, f => select(f, true));
  setupPcap(await api('/api/samples'));
  $('#foot-meta').textContent = `models trained in ${METRICS.meta.minutes} min on ${METRICS.meta.device} · seed ${METRICS.meta.seed}`;
  addEventListener('hashchange', route);
  addEventListener('resize', () => { if (onTap()) renderTap(); });
  route();
  startLoop();
  await loadScenario(META.scenarios.find(s => s.id === 'fri-mixed') ? 'fri-mixed' : META.scenarios[0].id);
})();
