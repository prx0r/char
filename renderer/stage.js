// Pogtown 2D stage: layered painted rig + vector face, deterministic per-frame renderer.
// window.STAGE_INIT(timeline) -> Promise; window.STAGE_FRAME(i) -> dataURL (jpeg)
'use strict';
const W = 1080, H = 1920;
const cv = document.getElementById('c');
const ctx = cv.getContext('2d');
let TL = null, IMG = {}, PERF = [];

// ---------- math helpers ----------
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
const easeOutBack = t => { t = clamp(t, 0, 1); const c1 = 1.5, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
const easeInOut = t => { t = clamp(t, 0, 1); return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
function hash(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
function noise1(x) { const i = Math.floor(x), f = x - i; return lerp(hash(i), hash(i + 1), smooth(f)) * 2 - 1; }
// window envelope: 0 before t0, eases to 1 over a, holds, eases back over r after t1
function win(t, t0, t1, a = .25, r = .3) {
  if (t < t0 || t > t1 + r) return 0;
  if (t < t0 + a) return smooth((t - t0) / a);
  if (t <= t1) return 1;
  return 1 - smooth((t - t1) / r);
}
function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(src); i.src = src; }); }

// ---------- expression targets ----------
const EXPR = {
  neutral:  { lid: .16, tilt: 0,    bot: .08, brow: 0,   browTilt: 0,   smile: .08, pup: 1 },
  deadpan:  { lid: .44, tilt: 0,    bot: .14, brow: -3,  browTilt: 0,   smile: 0,   pup: 1 },
  annoyed:  { lid: .38, tilt: .26,  bot: .18, brow: -6,  browTilt: 7,   smile: -.45, pup: .95 },
  surprised:{ lid: 0,   tilt: -.05, bot: 0,   brow: 12,  browTilt: -3,  smile: 0,   pup: .78 },
  confused: { lid: .2,  tilt: .1,   bot: .1,  brow: 4,   browTilt: -5,  smile: -.15, pup: 1, asym: .28 },
  smug:     { lid: .36, tilt: -.12, bot: .22, brow: 2,   browTilt: -2,  smile: .6,  pup: 1, asym: .1 },
  happy:    { lid: .18, tilt: -.06, bot: .34, brow: 5,   browTilt: -3,  smile: .95, pup: 1 },
  content:  { lid: .55, tilt: -.1,  bot: .25, brow: 2,   browTilt: -2,  smile: .55, pup: 1 },
  sad:      { lid: .3,  tilt: -.28, bot: .1,  brow: 3,   browTilt: -8,  smile: -.5, pup: 1 },
};
const VIS = { // Rhubarb mouth shapes -> params
  X: { open: 0,   wide: .96, round: 0,  teeth: 0, tongue: 0 },
  A: { open: 0,   wide: .9,  round: 0,  teeth: 0, tongue: 0, press: 1 },
  B: { open: .3,  wide: 1.0, round: 0,  teeth: 1, tongue: 0 },
  C: { open: .58, wide: 1.04,round: 0,  teeth: 1, tongue: .3 },
  D: { open: .9,  wide: 1.08,round: 0,  teeth: 1, tongue: .6 },
  E: { open: .66, wide: .82, round: .45,teeth: 0, tongue: .4 },
  F: { open: .34, wide: .62, round: .95,teeth: 0, tongue: 0 },
  G: { open: .16, wide: .96, round: 0,  teeth: 2, tongue: 0 },
  H: { open: .6,  wide: 1.0, round: 0,  teeth: 1, tongue: 1, tongueUp: 1 },
};

// ---------- track sampling ----------
function exprAt(P, t) {
  // ease between expression keys over 0.22 s
  const ks = P.ev.expr; let cur = EXPR[ks.length ? ks[0].expr : 'neutral'] || EXPR.neutral, prev = cur, tk = -9;
  for (const k of ks) { if (k.t <= t) { prev = cur; cur = EXPR[k.expr] || EXPR.neutral; tk = k.t; } }
  const a = smooth((t - tk) / .22), o = {};
  for (const key of ['lid', 'tilt', 'bot', 'brow', 'browTilt', 'smile', 'pup', 'asym']) o[key] = lerp(prev[key] || 0, cur[key] || 0, a);
  return o;
}
function mouthAt(P, t) {
  const cues = P.mouth; if (!cues || !cues.length) return VIS.X;
  let i = 0; while (i + 1 < cues.length && cues[i + 1].t <= t) i++;
  const c = cues[i], p = i > 0 ? cues[i - 1] : c;
  const a = smooth((t - c.t) / .055);
  const A = VIS[p.v] || VIS.X, B = VIS[c.v] || VIS.X, o = {};
  for (const k of ['open', 'wide', 'round', 'teeth', 'tongue', 'tongueUp', 'press']) o[k] = lerp(A[k] || 0, B[k] || 0, a);
  o.teethMode = B.teeth;
  return o;
}
function lookAt(P, t) {
  let x = P.baseLook ? P.baseLook[0] : 0, y = P.baseLook ? P.baseLook[1] : 0, px = x, py = y, tk = -9;
  for (const l of P.ev.looks) if (l.t <= t) { px = x; py = y; x = l.x + (P.baseLook ? P.baseLook[0] * (l.x === 0 ? 1 : 0) : 0); y = l.y; tk = l.t; }
  const a = smooth((t - tk) / .09);
  return [lerp(px, x, a), lerp(py, y, a)];
}
function blinkAt(P, t) {
  let v = 0;
  for (const b of P.ev.blinks) {
    const d = t - b.t, c = b.slow ? 2.2 : 1;
    if (d < 0 || d > .26 * c) continue;
    if (d < .06 * c) v = Math.max(v, d / (.06 * c)); else if (d < .1 * c) v = 1; else v = Math.max(v, 1 - (d - .1 * c) / (.16 * c));
  }
  return smooth(v);
}
function gestureAt(P, t, name) {
  for (const g of P.ev.gestures) if (g.name === name && t >= g.t0 - .001 && t <= g.t1 + .6) return { p: (t - g.t0) / (g.t1 - g.t0), d: g.t1 - g.t0, t0: g.t0, t1: g.t1, el: t - g.t0 };
  return null;
}
function stillAt(P, t) { let s = 0; for (const w of P.ev.still) s = Math.max(s, win(t, w.t0, w.t1, .3, .35)); return s; }
function envAt(P, t) { const e = P.env; if (!e) return 0; const i = t * P.envFps; const i0 = Math.floor(i); if (i0 < 0 || i0 + 1 >= e.length) return 0; return lerp(e[i0], e[i0 + 1], i - i0); }
function nodAt(P, t) { let y = 0; for (const n of P.ev.nods) { const d = t - n.t + .05; if (d < 0 || d > .9) continue; y += n.amp * Math.sin(d * 9) * Math.exp(-d * 5.5); } return y; }

// ---------- face drawing ----------
function reptileEye(c, E, cx, cy, side, st) {
  const r = E.r;
  c.save();
  c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.clip();
  // eyeball
  let g = c.createRadialGradient(cx - r * .25, cy - r * .35, r * .1, cx, cy, r * 1.05);
  g.addColorStop(0, E.iris_hi); g.addColorStop(.6, E.iris); g.addColorStop(1, E.iris_rim);
  c.fillStyle = g; c.fillRect(cx - r, cy - r, 2 * r, 2 * r);
  // pupil
  const ps = st.pupScale, sl = st.slit;
  const prx = lerp(E.pupil_rx, 2.6, sl) * ps, pry = lerp(E.pupil_ry, E.pupil_ry * 1.45, sl) * ps;
  const px = cx + st.lx * r * .32 + 4 * side * 0, py = cy + st.ly * r * .25 + 3;
  if (sl > .05) { // hot iris ring when slit
    const rg = c.createRadialGradient(px, py, 2, px, py, r * .9);
    rg.addColorStop(0, `rgba(230,120,40,${.55 * sl})`); rg.addColorStop(1, 'rgba(230,120,40,0)');
    c.fillStyle = rg; c.fillRect(cx - r, cy - r, 2 * r, 2 * r);
  }
  c.fillStyle = E.pupil; c.beginPath(); c.ellipse(px, py, prx, pry, 0, 0, Math.PI * 2); c.fill();
  // spec
  c.fillStyle = 'rgba(255,255,250,.92)'; c.beginPath(); c.ellipse(cx + r * .3, cy - r * .48, r * .2, r * .15, -.5, 0, Math.PI * 2); c.fill();
  c.fillStyle = 'rgba(255,255,250,.5)'; c.beginPath(); c.arc(cx - r * .38, cy + r * .42, r * .07, 0, Math.PI * 2); c.fill();
  // upper lid (tilt: + => inner corner lower; inner side is toward face centre = -side)
  const lidY = cy - r + st.lid * 2 * r, tilt = st.tilt * side * -1;
  const x0 = cx - r - 4, x1 = cx + r + 4;
  const y0 = lidY + tilt * r * -1, y1 = lidY + tilt * r;
  const sag = 7 * (1 - st.lid);
  c.beginPath(); c.moveTo(x0, cy - r - 10); c.lineTo(x1, cy - r - 10); c.lineTo(x1, y1);
  c.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0); c.closePath();
  let lg = c.createLinearGradient(0, cy - r, 0, Math.max(y0, y1) + 2);
  lg.addColorStop(0, E.lid); lg.addColorStop(1, E.lid_shade);
  c.fillStyle = lg; c.fill();
  // lid shadow on eyeball
  c.save(); c.globalAlpha = .22; c.beginPath(); c.moveTo(x1, y1); c.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0);
  c.lineTo(x0, y0 + 9); c.quadraticCurveTo(cx, (y0 + y1) / 2 + sag + 9, x1, y1 + 9); c.closePath(); c.fillStyle = '#5a3a10'; c.fill(); c.restore();
  c.strokeStyle = E.ink; c.lineWidth = E.outline; c.lineCap = 'round';
  c.beginPath(); c.moveTo(x1, y1); c.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0); c.stroke();
  // lower lid
  const by = cy + r - st.bot * r;
  c.beginPath(); c.moveTo(x0, cy + r + 10); c.lineTo(x1, cy + r + 10); c.lineTo(x1, by); c.quadraticCurveTo(cx, by - 6 * st.bot, x0, by); c.closePath();
  c.fillStyle = E.lid; c.fill();
  if (st.bot > .04) { c.lineWidth = E.outline * .8; c.beginPath(); c.moveTo(x1, by); c.quadraticCurveTo(cx, by - 6 * st.bot, x0, by); c.stroke(); }
  c.restore();
  c.strokeStyle = E.ink; c.lineWidth = E.outline; c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.stroke();
}

function almond(c, cx, cy, hw, ht, hb) {
  c.beginPath(); c.moveTo(cx - hw, cy + 1);
  c.bezierCurveTo(cx - hw * .55, cy - ht * 1.35, cx + hw * .45, cy - ht * 1.4, cx + hw, cy - 1);
  c.bezierCurveTo(cx + hw * .5, cy + hb * 1.35, cx - hw * .5, cy + hb * 1.3, cx - hw, cy + 1); c.closePath();
}
function humanEye(c, E, cx, cy, side, st) {
  const hw = E.hw, ht = E.ht, hb = E.hb;
  const htt = ht * (1 + (st.pupScale < .9 ? .35 : 0) * 1);  // surprised opens wider
  c.save(); almond(c, cx, cy, hw, htt, hb); c.clip();
  c.fillStyle = E.white; c.fillRect(cx - hw - 2, cy - htt * 2, hw * 2 + 4, htt * 4);
  const ix = cx + st.lx * hw * .42, iy = cy + st.ly * ht * .4 + 1, ir = E.iris_r;
  let g = c.createRadialGradient(ix, iy, ir * .2, ix, iy, ir);
  g.addColorStop(0, E.iris_hi); g.addColorStop(1, E.iris);
  c.fillStyle = g; c.beginPath(); c.arc(ix, iy, ir, 0, Math.PI * 2); c.fill();
  c.strokeStyle = E.ink; c.lineWidth = 1.6; c.stroke();
  c.fillStyle = '#1d120c'; c.beginPath(); c.arc(ix, iy, ir * .45 * st.pupScale, 0, Math.PI * 2); c.fill();
  c.fillStyle = 'rgba(255,255,255,.95)'; c.beginPath(); c.arc(ix + ir * .35, iy - ir * .38, ir * .26, 0, Math.PI * 2); c.fill();
  // top shadow
  const sg = c.createLinearGradient(0, cy - htt * 1.2, 0, cy + 2); sg.addColorStop(0, 'rgba(90,50,30,.35)'); sg.addColorStop(1, 'rgba(90,50,30,0)');
  c.fillStyle = sg; c.fillRect(cx - hw, cy - htt * 2, hw * 2, htt * 2.2);
  // upper lid descends: edge = lerp(top curve, bottom curve, lid)
  const L = clamp(st.lid >= .95 ? st.lid : st.lid * (E.lid_gain || .6), 0, 1), ti = st.tilt * side * -1 * .7;
  const topY = k => lerp(cy - htt * 1.05, cy + hb * .95, L) + ti * hw * .35 * k;
  c.beginPath(); c.moveTo(cx - hw - 3, cy - htt * 3); c.lineTo(cx + hw + 3, cy - htt * 3);
  c.lineTo(cx + hw + 3, topY(1)); c.quadraticCurveTo(cx, topY(0) - (1 - L) * 2, cx - hw - 3, topY(-1)); c.closePath();
  c.fillStyle = E.skin; c.fill();
  // lower lid push (cheek smile)
  const by = cy + hb * 1.05 - st.bot * hb * 1.6;
  c.beginPath(); c.moveTo(cx - hw - 3, cy + hb * 3); c.lineTo(cx + hw + 3, cy + hb * 3); c.lineTo(cx + hw + 3, by); c.quadraticCurveTo(cx, by - st.bot * 3, cx - hw - 3, by); c.closePath();
  c.fillStyle = E.skin; c.fill();
  c.restore();
  // lash line (thick top), following lid
  const L2 = L;
  c.strokeStyle = E.ink; c.lineCap = 'round'; c.lineJoin = 'round';
  c.lineWidth = E.lash;
  const oy = (k) => lerp(cy - htt * 1.0, cy + hb * .9, L2) + ti * hw * .35 * k;
  c.beginPath(); c.moveTo(cx - hw * side * -1 * 1.0 - (side > 0 ? 0 : 0), cy);
  // draw from inner to outer corner
  const xi = cx - side * hw, xo = cx + side * hw;
  c.moveTo(xi, (xi < cx ? oy(-1) : oy(1)) * .55 + cy * .45);
  c.quadraticCurveTo(cx, oy(0) - 1, xo, (xo < cx ? oy(-1) : oy(1)) * .6 + cy * .4);
  c.lineTo(xo + side * 4, cy - 3 + L2 * 2); c.stroke();
  // lower line
  c.lineWidth = 2.2; c.globalAlpha = .85;
  c.beginPath(); c.moveTo(cx - hw * .85, cy + hb * .5); c.quadraticCurveTo(cx, by + 2.5, cx + hw * .85, cy + hb * .5); c.stroke();
  // crease
  c.globalAlpha = .45; c.lineWidth = 2;
  c.beginPath(); c.moveTo(cx - hw * .7, cy - htt * 1.25 - 3 - st.browLift * .3); c.quadraticCurveTo(cx, cy - htt * 2.0 - 3 - st.browLift * .35, cx + hw * .75, cy - htt * 1.3 - 3 - st.browLift * .3); c.stroke();
  c.globalAlpha = 1;
}
function brow(c, B, side, st) {
  // B: inner [x,y], peak [x,y], outer [x,y]; thickness
  const lift = st.brow, it = st.browTilt;  // browTilt + => inner end down (angry)
  const asym = (st.asym || 0) * (side > 0 ? 1 : -1) * 10;
  const ix = B.inner[0], iy = B.inner[1] - lift + it + asym, px = B.peak[0], py = B.peak[1] - lift * 1.1 + asym, ox = B.outer[0], oy = B.outer[1] - lift * .8 - it * .3 + asym;
  const th = B.th;
  c.beginPath();
  c.moveTo(ix, iy - th * .55); c.quadraticCurveTo(px, py - th * .7, ox, oy - th * .15);
  c.quadraticCurveTo(ox + side * 2, oy + th * .1, ox, oy + th * .2);
  c.quadraticCurveTo(px, py + th * .45, ix, iy + th * .5); c.closePath();
  c.fillStyle = B.color; c.fill(); c.strokeStyle = B.ink; c.lineWidth = 2.2; c.stroke();
  c.save(); c.clip(); c.globalAlpha = .35; c.strokeStyle = '#2a2723'; c.lineWidth = 1.2;
  for (let k = 0; k < 7; k++) { const f = k / 7; const x = lerp(ix, ox, f); c.beginPath(); c.moveTo(x, lerp(iy, oy, f) + th * .4); c.lineTo(x + side * 6, lerp(iy, oy, f) - th * .6 - (1 - Math.abs(f - .45)) * 3); c.stroke(); }
  c.restore();
}
function vectorMouth(c, M, m, e) {
  const cx = M.cx, cy = M.cy, hw = M.hw * m.wide * (1 - m.round * .12);
  const smile = e.smile, open = m.open * M.max_open;
  const lx = cx - hw, rx = cx + hw;
  const ly = cy - smile * 7 + (M.smirk || 0) * -1, ry = cy - smile * 7 + (M.smirk || 0) * 1 * 0 - (M.smirk || 0);
  const upC = cy - open * .18 - smile * 2 - m.round * 3, loC = cy + open + smile * 4 + m.round * 2;
  c.lineCap = 'round'; c.lineJoin = 'round'; c.strokeStyle = M.ink;
  if (open < 1.2) {
    // closed: curved line with corner ticks (smirk support)
    c.lineWidth = M.line;
    c.beginPath(); c.moveTo(lx, ly); c.quadraticCurveTo(cx, cy + smile * 8 - (m.press || 0) * 1.5, rx, ry); c.stroke();
    c.lineWidth = M.line * .7;
    if (smile > .2) { c.beginPath(); c.moveTo(lx - 3, ly - 7 * smile); c.quadraticCurveTo(lx - 1, ly + 1, lx + 4, ly + 3); c.stroke();
      c.beginPath(); c.moveTo(rx + 3, ry - 7 * smile); c.quadraticCurveTo(rx + 1, ry + 1, rx - 4, ry + 3); c.stroke(); }
  } else {
    c.beginPath(); c.moveTo(lx, ly);
    c.bezierCurveTo(lx + hw * .4, upC, rx - hw * .4, upC, rx, ry);
    c.bezierCurveTo(rx - hw * .25, loC + open * .25, lx + hw * .25, loC + open * .25, lx, ly); c.closePath();
    c.fillStyle = M.interior; c.fill();
    c.save(); c.clip();
    if (m.teethMode >= 1) { c.fillStyle = M.teeth; c.beginPath(); c.moveTo(lx, ly - 4); c.bezierCurveTo(lx + hw * .4, upC - 4, rx - hw * .4, upC - 4, rx, ry - 4); c.lineTo(rx, ry + 2); c.bezierCurveTo(rx - hw * .4, upC + Math.min(9, open * .45) + 2, lx + hw * .4, upC + Math.min(9, open * .45) + 2, lx, ly + 2); c.fill(); }
    if (m.teethMode >= 2) { c.fillStyle = M.teeth; c.fillRect(lx, loC - 6, hw * 2, 8); }
    if (m.tongue > .05 && open > 8) { c.fillStyle = M.tongue; c.beginPath(); c.ellipse(cx + 2, loC + open * .25 - (m.tongueUp || 0) * open * .35, hw * .42, open * .3 + 2, 0, 0, Math.PI * 2); c.fill(); }
    c.restore();
    c.lineWidth = M.line; c.stroke();
    // lower lip shading
    c.globalAlpha = .35; c.lineWidth = M.line * .8; c.beginPath(); c.moveTo(cx - hw * .4, loC + open * .25 + 9); c.quadraticCurveTo(cx, loC + open * .25 + 13, cx + hw * .4, loC + open * .25 + 9); c.stroke(); c.globalAlpha = 1;
  }
}
// lizard jaw: interior lens + jaw layer sliced into columns displaced by a tapered curve
function jawMouth(c, P, M, m, e) {
  const D = m.open * M.max_open + (m.round ? 0 : 0), cx = M.cx, hw = M.hw * (1 - m.round * .25), cy = M.cy, cu = M.curve;
  const lineY = x => { const u = (x - cx) / M.hw; return cy + cu * u * u; };
  const dy = x => { const u = (x - cx) / hw; if (Math.abs(u) >= 1) return 0; return D * Math.pow(1 - u * u, .75); };
  if (D > .6) {
    c.save();
    c.beginPath(); c.moveTo(cx - hw, lineY(cx - hw));
    for (let x = cx - hw; x <= cx + hw; x += 4) c.lineTo(x, lineY(x) - 1);
    for (let x = cx + hw; x >= cx - hw; x -= 4) c.lineTo(x, lineY(x) + dy(x) + 2);
    c.closePath(); c.fillStyle = M.interior; c.fill(); c.clip();
    // throat shade
    const tg = c.createRadialGradient(cx, cy + D * .9, 2, cx, cy + D * .9, hw * .8); tg.addColorStop(0, '#1a0606'); tg.addColorStop(1, 'rgba(26,6,6,0)');
    c.fillStyle = tg; c.fillRect(cx - hw, cy - 20, hw * 2, D + 60);
    // tongue
    if (m.tongue > .05 || D > 10) { c.fillStyle = M.tongue; c.beginPath(); c.ellipse(cx + 6, cy + D * (1.0 - .45 * (m.tongueUp || 0)) + 6, hw * .42, D * .38 + 4, 0, 0, Math.PI * 2); c.fill();
      c.strokeStyle = 'rgba(120,30,40,.6)'; c.lineWidth = 2; c.beginPath(); c.moveTo(cx + 6, cy + D * .8); c.lineTo(cx + 6, cy + D * 1.1 + 4); c.stroke(); }
    // little teeth row
    if (D > 11) { c.fillStyle = M.teeth; const tk = Math.min(1, (D - 11) / 8); for (let x = cx - hw * .7; x <= cx + hw * .71; x += hw * .35) { const y = lineY(x) - 1; c.beginPath(); c.moveTo(x - 6, y); c.lineTo(x + 6, y); c.lineTo(x, y + 9 * tk); c.closePath(); c.fill(); } }
    c.restore();
    // upper lip ink
    c.strokeStyle = M.ink; c.lineWidth = 5; c.lineCap = 'round';
    c.beginPath(); for (let x = cx - hw; x <= cx + hw; x += 4) { const y = lineY(x) - 1; if (x === cx - hw) c.moveTo(x, y); else c.lineTo(x, y); } c.stroke();
  }
  // jaw columns
  const J = IMG[P.key + ':jaw'];
  const x0 = Math.floor(cx - M.hw - 30), x1 = Math.ceil(cx + M.hw + 30), step = 3;
  for (let x = x0; x < x1; x += step) { const d = dy(x + step / 2); if (d < .3) continue; c.drawImage(J, x, 0, step, P.size[1], x, d, step, P.size[1]); }
}

// ---------- character ----------
function drawCharacter(c, P, t) {
  const R = P.rig, S = R.place.scale;
  const e = exprAt(P, t), m = mouthAt(P, t), look = lookAt(P, t), blink = blinkAt(P, t);
  const still = stillAt(P, t), env = envAt(P, t), mv = P.ev.profile.movement, en = P.ev.profile.energy;
  const ph = P.phase || 0;
  // gestures
  const gl = gestureAt(P, t, 'eye_glitch');
  const gs = gestureAt(P, t, 'suit_adjust') || gestureAt(P, t, 'collar_tug');
  const gw = gestureAt(P, t, 'warm_rock_lean') || gestureAt(P, t, 'lean_in') || gestureAt(P, t, 'lean_in_conspiratorial');
  const gr = gestureAt(P, t, 'shrug');
  const gp = gestureAt(P, t, 'golf_swing_mime');
  // idle + talk motion
  const breathe = Math.sin((t + ph) * 2 * Math.PI / 3.8);
  const damp = 1 - .85 * still;
  let hRot = (0.9 * noise1((t + ph) * .45) + .35 * Math.sin((t + ph) * 1.7)) * (0.5 + mv) * damp;
  let hx = 3 * noise1((t + ph) * .3 + 7) * damp, hy = -2.2 * breathe;
  hRot += env * 1.6 * en * damp * Math.sin(t * 5.3 + ph);
  hy += nodAt(P, t) * 7 * damp + env * 2.5 * damp;
  let bodyY = 0, bodyX = 0, bodySY = 1 + .007 * breathe, lean = 0, shoulder = 0;
  let lidAdd = 0, slit = 0, pupScale = e.pup, extraLook = [0, 0];
  if (gr) { const k = Math.sin(clamp(gr.p, 0, 1) * Math.PI); shoulder = -14 * k; hRot += 5 * k * (P.side || 1); e.brow += 8 * k; }
  if (gs) { // suit adjust: chin up, tug, head turns away, body jitter
    const p = gs.el / Math.max(gs.d, .6);
    const up = smooth(p / .25) * (1 - smooth((p - .8) / .2));
    hy -= 9 * up; hRot += -4.5 * up * (P.side || 1);
    const tug = (p > .3 && p < .75) ? Math.sin((p - .3) / .45 * Math.PI * 4) : 0;
    bodyX += 2.5 * tug * up; bodyY += -3 * Math.abs(tug) * up; shoulder += -6 * up;
  }
  if (gp) { // 'watches the drive slice': head follows an imaginary ball up and off to the right, wince
    const k = win(t, gp.t0, gp.t1, .3, .4), p = clamp(gp.el / gp.d, 0, 1);
    hRot += (-3 + 9 * smooth(p)) * k; hx += 10 * smooth(p) * k; hy -= 6 * k;
    extraLook[0] = (.3 + .7 * smooth(p)) * k; extraLook[1] = (-.7 + .4 * smooth(p)) * k; lidAdd += .2 * k; e.brow -= 3 * k; e.browTilt += 4 * k; e.smile -= .4 * k;
  }
  if (gw) { const k = win(t, gw.t0, gw.t1, .5, .5); lean = k; lidAdd += .25 * k; hRot += 2.5 * k; }
  if (gl) { const p = gl.el;
    slit = p < .07 ? p / .07 : (p > gl.d - .15 ? clamp(1 - (p - (gl.d - .15)) / .2, 0, 1) : 1);
    pupScale *= 1 + .1 * slit; lidAdd -= .35 * slit * (p < .5 ? 1 : 0);
    if (p < .3) { hx += (hash(Math.floor(t * 30)) - .5) * 6; }
  }
  // place character
  c.save();
  const leanS = 1 + .085 * lean;
  c.translate(R.place.x + P.offset[0], R.place.y + P.offset[1] + 40 * lean);
  c.scale(S, S);
  c.translate(P.size[0] / 2, P.size[1]); c.scale(leanS, leanS); c.translate(-P.size[0] / 2, -P.size[1]);
  // shadow behind (precomputed silhouette)
  c.save(); c.globalAlpha = .28; c.drawImage(IMG[P.key + ':shadow'], 26, 18); c.restore();
  // head transform relative to body pivot
  const piv = R.cut.pivot;
  const bodyPivY = P.size[1] - (P.size[1] - piv[1]) * bodySY;
  // head group to offscreen if glitching
  const glitchNow = gl && gl.el < .32;
  const hc = glitchNow ? P.headCanvas.getContext('2d') : c;
  if (glitchNow) { hc.setTransform(1, 0, 0, 1, 0, 0); hc.clearRect(0, 0, P.size[0], P.size[1]); }
  hc.save();
  hc.translate(piv[0] + hx + bodyX, bodyPivY + hy + bodyY + shoulder * .6);
  hc.rotate(hRot * Math.PI / 180);
  hc.translate(-piv[0], -piv[1]);
  hc.drawImage(IMG[P.key + ':head'], 0, 0);
  // mouth
  const M = R.mouth;
  if (M.type === 'jaw') jawMouth(hc, P, M, m, e); else vectorMouth(hc, M, m, e);
  // eyes
  const E = R.eyes;
  const lx = clamp(look[0] + extraLook[0], -1, 1), ly = clamp(look[1] + extraLook[1], -1, 1);
  for (const side of [-1, 1]) {
    const pos = side < 0 ? E.L : E.R;
    const asym = (e.asym || 0) * (side > 0 ? 1 : -1);
    const st = { lid: clamp(Math.max(e.lid + lidAdd + asym, blink * 1.02), -0.05, 1.02), tilt: e.tilt, bot: e.bot, lx, ly, slit, pupScale, browLift: e.brow };
    if (blink > .01) st.lid = clamp(lerp(e.lid + lidAdd + asym, 1.0, blink), 0, 1.0);
    if (E.type === 'reptile') reptileEye(hc, E, pos[0], pos[1], side, st); else humanEye(hc, E, pos[0], pos[1], side, st);
  }
  if (R.brows) for (const side of [-1, 1]) brow(hc, side < 0 ? R.brows.L : R.brows.R, side, { brow: e.brow + blink * -2, browTilt: e.browTilt, asym: e.asym });
  hc.restore();
  if (glitchNow) { // RGB split + slice displacement
    const p = gl.el / .32, amp = (1 - p) * 14;
    const seed = Math.floor(t * 30);
    c.save(); c.globalAlpha = .55; c.globalCompositeOperation = 'lighter';
    c.filter = 'url(#none)';
    c.drawImage(P.headCanvas, -amp * .8, 0); c.restore();
    for (let y = 0; y < piv[1] + 60; y += 18) {
      const off = (hash(seed * 13 + y) > .7) ? (hash(seed + y) - .5) * amp * 3 : 0;
      c.drawImage(P.headCanvas, 0, y, P.size[0], 18, off, y, P.size[0], 18);
    }
    c.save(); c.globalAlpha = .25 * (1 - p); c.fillStyle = '#7dffb0';
    for (let k = 0; k < 4; k++) { const yy = hash(seed * 7 + k) * piv[1]; c.fillRect(150, yy, 470, 3); } c.restore();
  }
  // body over head (collar covers neck)
  c.save(); c.translate(bodyX, bodyY + shoulder * .5);
  c.translate(0, P.size[1]); c.scale(1, bodySY); c.translate(0, -P.size[1]);
  c.drawImage(IMG[P.key + ':body'], 0, 0);
  if (IMG[P.key + ':collar']) c.drawImage(IMG[P.key + ':collar'], 0, 0);
  c.restore();
  // hand prop for suit adjust
  if (gs && IMG[P.key + ':hand']) {
    const p = gs.el / Math.max(gs.d, .6), Hd = R.hand, hi = IMG[P.key + ':hand'];
    const k = p < .28 ? easeOutBack(p / .28) : (p > .78 ? 1 - smooth((p - .78) / .22) : 1);
    const tug = (p > .3 && p < .75) ? Math.sin((p - .3) / .45 * Math.PI * 4) : 0;
    const tx = lerp(Hd.rest[0], Hd.tie[0], k) + tug * 3, ty = lerp(Hd.rest[1], Hd.tie[1], k) - Math.abs(tug) * 6;
    c.save(); c.translate(tx + bodyX, ty + bodyY); c.rotate((-12 + 10 * (1 - k)) * Math.PI / 180); c.scale(Hd.scale, Hd.scale);
    c.drawImage(hi, -hi.width * Hd.grip[0], -hi.height * Hd.grip[1]); c.restore();
  }
  c.restore();
  return { headScreen: P.headScreen };
}

// ---------- stage chrome ----------
function drawBackground(c, t) {
  const bg = IMG['bg'];
  const s = Math.max(W / bg.width, H / bg.height) * (TL.bgScale || 1.06);
  const drift = Math.sin(t * .12) * 6;
  c.drawImage(bg, (W - bg.width * s) / 2 + drift, (H - bg.height * s) / 2, bg.width * s, bg.height * s);
  // soft focus: overlay darkened top + vignette applied later
}
function drawVignette(c) {
  const g = c.createRadialGradient(W / 2, H * .45, H * .25, W / 2, H * .5, H * .78);
  g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(8,6,14,.55)');
  c.fillStyle = g; c.fillRect(0, 0, W, H);
  const w = c.createRadialGradient(W * .92, H * .38, 10, W * .92, H * .38, W * .9);
  w.addColorStop(0, 'rgba(255,190,110,.16)'); w.addColorStop(1, 'rgba(255,190,110,0)');
  c.save(); c.globalCompositeOperation = 'screen'; c.fillStyle = w; c.fillRect(0, 0, W, H); c.restore();
}
function grain(c, t) {
  const n = IMG['grain']; if (!n) return;
  c.save(); c.globalAlpha = .05; c.globalCompositeOperation = 'overlay';
  const ox = Math.floor(hash(Math.floor(t * 24)) * 256), oy = Math.floor(hash(Math.floor(t * 24) + 9) * 256);
  c.fillStyle = c.createPattern(n, 'repeat'); c.translate(-ox, -oy); c.fillRect(0, 0, W + 256, H + 256); c.restore();
}
function roundRect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
function drawCaptions(c, t) {
  const caps = TL.captions || [];
  const cap = caps.find(k => t >= k.t0 && t <= k.t1 + .05); if (!cap) return;
  const y0 = TL.captionY || 1540;
  c.save(); c.font = '78px Lilita'; c.textBaseline = 'alphabetic'; c.lineJoin = 'round';
  // layout words into up to 2 lines
  const space = 20, maxW = 920; const lines = [[]]; let lw = 0;
  for (const w of cap.words) { const ww = c.measureText(w.w).width; if (lw + ww > maxW && lines[lines.length - 1].length) { lines.push([]); lw = 0; } lines[lines.length - 1].push({ ...w, ww }); lw += ww + space; }
  const pop = smooth((t - cap.t0) / .12);
  lines.forEach((ln, li) => {
    const tw = ln.reduce((a, w) => a + w.ww, 0) + space * (ln.length - 1); let x = (W - tw) / 2; const y = y0 + li * 92;
    for (const w of ln) {
      const active = t >= w.t0 - .02 && t <= w.t1 + .08, said = t > w.t1;
      const sc = active ? 1.08 : 1;
      c.save(); c.translate(x + w.ww / 2, y - 26); c.scale(sc * (.9 + .1 * pop), sc * (.9 + .1 * pop)); c.translate(-(x + w.ww / 2), -(y - 26));
      c.lineWidth = 14; c.strokeStyle = '#1b1020'; c.strokeText(w.w, x, y);
      c.fillStyle = active ? '#ffd23f' : (said || TL.capStatic ? '#ffffff' : 'rgba(255,255,255,.92)'); c.fillText(w.w, x, y);
      c.restore(); x += w.ww + space;
    }
  });
  c.restore();
}
function drawNameTag(c, t) {
  for (const tag of TL.tags || []) {
    const k = win(t, tag.t0, tag.t1, .35, .35); if (k <= 0) continue;
    c.save(); c.globalAlpha = k; c.translate(lerp(-40, 0, easeOutBack(k)), 0);
    c.font = '52px Lilita'; const w1 = c.measureText(tag.name).width; c.font = '600 32px Nunito'; const w2 = c.measureText(tag.sub).width;
    const bw = Math.max(w1, w2) + 70, x = tag.x || 60, y = tag.y || 150;
    roundRect(c, x, y, bw, 128, 26); c.fillStyle = 'rgba(22,16,34,.82)'; c.fill();
    c.fillStyle = tag.color || '#7ed957'; roundRect(c, x, y, 14, 128, 7); c.fill();
    c.fillStyle = '#fff'; c.font = '52px Lilita'; c.fillText(tag.name, x + 36, y + 62);
    c.fillStyle = 'rgba(255,255,255,.75)'; c.font = '600 32px Nunito'; c.fillText(tag.sub, x + 36, y + 104);
    c.restore();
  }
}
function cameraAt(t) {
  // base framing + per-performer pushes + shot list (two-hander)
  let z = 1, fx = W / 2, fy = 760, ox = 0, oy = 0;
  if (TL.shots && TL.shots.length) {
    let s = TL.shots[0]; for (const sh of TL.shots) if (sh.t <= t) s = sh;
    z = s.zoom; fx = s.fx; fy = s.fy;
  }
  for (const P of PERF) {
    for (const cm of P.ev.camera) { const k = win(t, cm.t0, cm.t1, cm.t1 - cm.t0, .35); z *= 1 + cm.amt * easeInOut(k); }
    const gl = gestureAt(P, t, 'eye_glitch'); if (gl) { const k = gl.el < .1 ? gl.el / .1 : clamp(1 - (gl.el - gl.d) / .4, 0, 1); z *= 1 + .1 * k; }
  }
  ox = noise1(t * .25) * 3; oy = noise1(t * .21 + 4) * 3; // handheld drift
  return { z, fx, fy, ox, oy };
}

// ---------- frame ----------
function frame(i) {
  const t = i / TL.fps;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  const cam = cameraAt(t);
  ctx.save();
  ctx.translate(W / 2 + cam.ox, cam.fy + cam.oy); ctx.scale(cam.z, cam.z); ctx.translate(-cam.fx, -cam.fy);
  drawBackground(ctx, t);
  for (const P of PERF) drawCharacter(ctx, P, t);
  ctx.restore();
  drawVignette(ctx);
  grain(ctx, t);
  drawNameTag(ctx, t);
  drawCaptions(ctx, t);
  if (TL.fadeIn) { const k = 1 - smooth(t / TL.fadeIn); if (k > 0) { ctx.fillStyle = `rgba(0,0,0,${k})`; ctx.fillRect(0, 0, W, H); } }
  const tEnd = TL.duration; if (t > tEnd - .4) { ctx.fillStyle = `rgba(0,0,0,${smooth((t - (tEnd - .4)) / .4)})`; ctx.fillRect(0, 0, W, H); }
}

window.STAGE_INIT = async function (tl) {
  TL = tl;
  await document.fonts.load('78px Lilita'); await document.fonts.load('600 32px Nunito');
  IMG['bg'] = await loadImg(tl.background);
  // grain tile
  const g = document.createElement('canvas'); g.width = g.height = 256; const gc = g.getContext('2d'); const id = gc.createImageData(256, 256);
  for (let k = 0; k < id.data.length; k += 4) { const v = Math.floor(hash(k * .37) * 255); id.data[k] = id.data[k + 1] = id.data[k + 2] = v; id.data[k + 3] = 255; } gc.putImageData(id, 0, 0); IMG['grain'] = g;
  PERF = [];
  for (const [idx, p] of tl.performers.entries()) {
    const key = 'p' + idx; const base = p.dir;
    for (const L of ['head', 'body', 'jaw', 'hand', 'collar']) if (p.layers[L]) IMG[key + ':' + L] = await loadImg(base + '/' + p.layers[L]);
    // silhouette shadow
    const b = IMG[key + ':body'], sz = p.layers.size; const sc = document.createElement('canvas'); sc.width = sz[0] + 60; sc.height = sz[1] + 60;
    const s2 = sc.getContext('2d'); s2.filter = 'blur(14px)'; s2.drawImage(b, 0, 0); s2.drawImage(IMG[key + ':head'], 0, 0);
    s2.globalCompositeOperation = 'source-in'; s2.fillStyle = '#0c0816'; s2.fillRect(0, 0, sc.width, sc.height); IMG[key + ':shadow'] = sc;
    const hcv = document.createElement('canvas'); hcv.width = sz[0]; hcv.height = sz[1];
    PERF.push({ key, rig: p.rig, ev: p.events, mouth: p.mouth, env: p.env, envFps: p.env_fps, size: sz, offset: p.offset || [0, 0], side: p.side || 1, phase: idx * 1.7, baseLook: p.base_look, headCanvas: hcv });
  }
  return true;
};
window.STAGE_FRAME = function (i, q) { frame(i); return cv.toDataURL('image/jpeg', q || 0.93); };
