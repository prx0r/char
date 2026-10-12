// Pogtown stage v2: timeline (performance.v2) -> clip-driven OddRig performers, deterministic per frame.
// window.STAGE_INIT(timeline) -> Promise; window.STAGE_FRAME(i, q) -> jpeg dataURL
'use strict';
const W = 1080, H = 1920;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const glc = document.getElementById('gl');
const { GLStage, Rig, clamp, lerp, smooth } = OddRig;
let TL = null, IMG = {}, PERF = [], CLIPS = {}, GLS = null, SHADOW = null;

const easeOutBack = t => { t = clamp(t, 0, 1); const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
const EASE = {
  l: t => t, i: t => t * t * t, o: t => 1 - Math.pow(1 - t, 3), io: t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2,
  b: easeOutBack, s: t => 0,
};
function hash(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }
function noise1(x) { const i = Math.floor(x), f = x - i; return lerp(hash(i), hash(i + 1), smooth(f)) * 2 - 1; }
function win(t, t0, t1, a = .25, r = .3) { if (t < t0 || t > t1 + r) return 0; if (t < t0 + a) return smooth((t - t0) / a); if (t <= t1) return 1; return 1 - smooth((t - t1) / r); }
function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(src); i.src = src; }); }

// ---------- expressions / visemes ----------
const EXPR = {
  neutral:  { lid: .12, tilt: 0,    bot: .04, brow: 0,    browTilt: 0,    smile: .05, pup: 1 },
  deadpan:  { lid: .40, tilt: 0,    bot: .12, brow: -.4,  browTilt: 0,    smile: 0,   pup: 1 },
  annoyed:  { lid: .36, tilt: .24,  bot: .16, brow: -.8,  browTilt: .9,   smile: -.45, pup: .95 },
  surprised:{ lid: -.05,tilt: -.05, bot: 0,   brow: 1.4,  browTilt: -.3,  smile: 0,   pup: .78 },
  confused: { lid: .18, tilt: .1,   bot: .08, brow: .5,   browTilt: -.6,  smile: -.15, pup: 1, asym: .25 },
  smug:     { lid: .34, tilt: -.1,  bot: .2,  brow: .2,   browTilt: -.2,  smile: .6,  pup: 1, asym: .1 },
  grin:     { lid: .2,  tilt: -.06, bot: .3,  brow: .6,   browTilt: -.3,  smile: .95, pup: 1 },
  self_satisfied: { lid: .42, tilt: -.12, bot: .24, brow: .3, browTilt: -.2, smile: .7, pup: 1 },
  confident:{ lid: .22, tilt: -.04, bot: .1,  brow: .4,   browTilt: -.1,  smile: .4,  pup: 1 },
  suspicious:{ lid: .5, tilt: .16,  bot: .3,  brow: -.6,  browTilt: .5,   smile: -.1, pup: .9, asym: .15 },
  embarrassed:{ lid: .25, tilt: -.2, bot: .1, brow: .7,   browTilt: -.7,  smile: -.1, pup: 1 },
  sad:      { lid: .3,  tilt: -.26, bot: .08, brow: .3,   browTilt: -.9,  smile: -.5, pup: 1 },
  happy:    { lid: .18, tilt: -.06, bot: .3,  brow: .5,   browTilt: -.3,  smile: .9,  pup: 1 },
};
const VIS = {
  X: { open: 0, wide: .96, round: 0, teeth: 0, tongue: 0 }, A: { open: 0, wide: .9, round: 0, teeth: 0, tongue: 0, press: 1 },
  B: { open: .3, wide: 1.0, round: 0, teeth: 1, tongue: 0 }, C: { open: .58, wide: 1.04, round: 0, teeth: 1, tongue: .3 },
  D: { open: .9, wide: 1.08, round: 0, teeth: 1, tongue: .6 }, E: { open: .66, wide: .82, round: .45, teeth: 0, tongue: .4 },
  F: { open: .34, wide: .62, round: .95, teeth: 0, tongue: 0 }, G: { open: .16, wide: .96, round: 0, teeth: 2, tongue: 0 },
  H: { open: .6, wide: 1.0, round: 0, teeth: 1, tongue: 1, tongueUp: 1 },
};

// ---------- clip evaluation ----------
function keyAt(keys, t) {
  if (!keys.length) return 0;
  if (t <= keys[0][0]) return keys[0][1];
  for (let i = 0; i + 1 < keys.length; i++) {
    const [t0, v0, e] = keys[i], [t1, v1] = keys[i + 1];
    if (t < t1) {
      if (typeof v0 === 'string' || e === 's') return v0;
      const u = (t - t0) / Math.max(1e-6, t1 - t0);
      return lerp(v0, v1, (EASE[e || 'io'] || EASE.io)(u));
    }
  }
  return keys[keys.length - 1][1];
}
function clipLocalTime(c, inst, t) {
  const el = (t - inst.t0) * (inst.speed || 1);
  if (c.loop) return ((el + (inst.phase || 0)) % c.duration + c.duration) % c.duration;
  const dur = inst.dur || c.duration;
  if (c.hold_at != null && dur > c.duration) {        // stretch by holding the pose at hold_at
    const extra = dur - c.duration;
    if (el < c.hold_at) return el;
    if (el < c.hold_at + extra) return c.hold_at + (el - c.hold_at) * 0.0;
    return el - extra;
  }
  if (c.hold_at == null && dur !== c.duration && !c.loop) return el * c.duration / dur;   // time-scale
  return el;
}
function instWeight(c, inst, t) {
  const dur = c.loop ? (inst.dur || 1e9) : (inst.dur || c.duration);
  const fi = inst.fade_in != null ? inst.fade_in : (c.fade_in != null ? c.fade_in : .12);
  const fo = inst.fade_out != null ? inst.fade_out : (c.fade_out != null ? c.fade_out : .2);
  const el = t - inst.t0;
  if (el < 0 || el > dur + (c.loop ? fo : 0)) return 0;
  let w = fi > 0 ? smooth(el / fi) : 1;
  if (c.loop) { if (el > dur) w *= 1 - smooth((el - dur) / fo); }
  else if (fo > 0 && el > dur - fo) w *= smooth((dur - el) / fo);
  return w * (inst.w == null ? 1 : inst.w);
}
const SWAP = s => s.replace(/_r(\b|\.)/g, '\u0000$1').replace(/_l(\b|\.)/g, '_r$1').replace(/\u0000/g, '_l');
function compileClip(c) {
  // expand `*` into both sides; returns [{bone, ch, side, keys}] + face/ik/fx tracks
  const out = [];
  for (const [name, keys] of Object.entries(c.tracks || {})) {
    const [head, ch] = [name.slice(0, name.lastIndexOf('.')), name.slice(name.lastIndexOf('.') + 1)];
    const targets = head.includes('*') ? [head.replace('*', 'r'), head.replace('*', 'l')] : [head];
    for (const tg of targets) out.push({ target: tg, ch, keys });
  }
  return out;
}
function sideOf(b) { return /_r$/.test(b) ? 'r' : /_l$/.test(b) ? 'l' : null; }

function evalPose(P, t) {
  const R = P.rig, acc = { bones: {}, hands: { r: 'rest', l: 'rest' }, face: {}, ik: {}, fx: { glitch: 0 }, exprOv: null, handW: { r: 0, l: 0 } };
  const B = n => acc.bones[n] || (acc.bones[n] = { rot: 0, tx: 0, ty: 0, sx: 1, sy: 1 });
  const still = P.stillAt(t);
  for (const inst of P.clips) {
    const c = P.clipset[inst.clip]; if (!c) continue;
    let w = instWeight(c, inst, t); if (w <= 0) continue;
    if (c.layer === 'idle') w *= 1 - .75 * still;
    if (c.layer === 'talk') w *= 1 - .9 * still;
    const lt = clipLocalTime(c, inst, t), mir = !!inst.mirror;
    const amp = (c.layer === 'base' ? 1 : P.amp) * (inst.amp || 1);
    for (const tr of c._compiled) {
      let target = mir ? SWAP(tr.target) : tr.target;
      const v = keyAt(tr.keys, lt);
      if (target === 'face') {
        if (tr.ch === 'expr') { if (w > .5) acc.exprOv = v; continue; }
        let x = v; if (mir && (tr.ch === 'gaze_x' || tr.ch === 'yaw')) x = -x;
        acc.face[tr.ch] = (acc.face[tr.ch] || 0) + x * w; continue;
      }
      if (target === 'fx') { acc.fx[tr.ch] = Math.max(acc.fx[tr.ch] || 0, v * w); continue; }
      if (target.startsWith('ik.')) {
        const side = target.slice(-1), k = acc.ik[side] || (acc.ik[side] = { w: 0, ox: 0, oy: 0, target: null });
        const spec = (c.ik || {})['hand_' + (mir ? (side === 'r' ? 'l' : 'r') : side)] || {};
        k.target = R.j.anchors[(inst.ik_target || spec.target)] ? (inst.ik_target || spec.target) : 'chest_center';
        if (tr.ch === 'w') k.w = Math.max(k.w, v * w); else if (tr.ch === 'ox') k.ox += (mir ? -v : v) * w; else if (tr.ch === 'oy') k.oy += v * w;
        continue;
      }
      if (!R.bi[target]) continue;
      if (tr.ch === 'pose') { const s = sideOf(target); if (w >= acc.handW[s]) { acc.hands[s] = v; acc.handW[s] = w; } continue; }
      const side = sideOf(target), b = B(target);
      if (tr.ch === 'rot') { let x = v * amp * w; if (side === 'l') x = -x; if (!side && mir) x = -x; b.rot += x; }
      else if (tr.ch === 'tx') { let x = v * R.unit * amp * w; if (side === 'r') x = -x; if (!side && mir) x = -x; b.tx += x; }
      else if (tr.ch === 'ty') b.ty += v * R.unit * amp * w;
      else if (tr.ch === 'sx') b.sx *= 1 + (v - 1) * amp * w;
      else if (tr.ch === 'sy') b.sy *= 1 + (v - 1) * amp * w;
    }
  }
  if (acc.handW.r < .5) acc.hands.r = acc.hands.r === 'rest' ? 'rest' : acc.hands.r;
  return acc;
}

// two-bone IK in bind-world space -> local rotations for upper/lower arm
function solveIK(P, pose, side) {
  const R = P.rig, k = pose.ik[side]; if (!k || k.w <= .001) return;
  const up = R.bi['upper_arm_' + side], lo = R.bi['lower_arm_' + side], ha = R.bi['hand_' + side];
  const Wd = R.world(pose);
  const S = OddRig.apply(Wd.D[up.i], up.x, up.y);
  let T0 = R.anchor(Wd, k.target); T0 = [T0[0] + k.ox * R.unit, T0[1] + k.oy * R.unit];
  // aim the hand's grip point (not the wrist) at the target: iterate wrist target = T0 - R(forearm) * grip
  const hp = (pose.hands && pose.hands[side]) || 'rest', ha_att = R.j.attachments[(R.j.hands.by_side[side] || {})[hp]];
  const grip = (ha_att && ha_att.grip) || [0, 0];
  let T = T0.slice(), upAng = 0, loAng = 0, E = null;
  for (let it = 0; it < 3; it++) {
  const L1 = Math.hypot(lo.x - up.x, lo.y - up.y), L2 = Math.hypot(ha.x - lo.x, ha.y - lo.y);
  const sgn = side === 'r' ? 1 : -1;   // elbow out: for the right arm (screen-left) the elbow goes to -x
  let dx = T[0] - S[0], dy = T[1] - S[1]; let d = Math.hypot(dx, dy); d = clamp(d, Math.abs(L1 - L2) + 1, L1 + L2 - 1);
  const base = Math.atan2(dy, dx);
  const a1 = Math.acos(clamp((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1));
  // two solutions; take the one with the elbow lower on screen (elbows hang), ties broken toward "out"
  const c1 = base + a1, c2 = base - a1;
  const e1 = Math.sin(c1), e2 = Math.sin(c2);
  upAng = Math.abs(e1 - e2) < .15 ? base + sgn * a1 : (e1 > e2 ? c1 : c2);
  E = [S[0] + Math.cos(upAng) * L1, S[1] + Math.sin(upAng) * L1];
  loAng = Math.atan2(T[1] - E[1], T[0] - E[0]);
  const bl0 = Math.atan2(ha.y - lo.y, ha.x - lo.x), rot = loAng - bl0 + ((pose.bones['hand_' + side] || {}).rot || 0) * Math.PI / 180;
  const gx = grip[0] * Math.cos(rot) - grip[1] * Math.sin(rot), gy = grip[0] * Math.sin(rot) + grip[1] * Math.cos(rot);
  T = [T0[0] - gx * .45, T0[1] - gy * .45];
  }
  const parentRot = Math.atan2(Wd.D[R.bi['shoulder_' + side].i][1], Wd.D[R.bi['shoulder_' + side].i][0]);
  const bindUp = Math.atan2(lo.y - up.y, lo.x - up.x), bindLo = Math.atan2(ha.y - lo.y, ha.x - lo.x);
  const wrap = a => { while (a > Math.PI) a -= 2 * Math.PI; while (a < -Math.PI) a += 2 * Math.PI; return a; };
  const rUp = wrap(upAng - bindUp - parentRot) * 180 / Math.PI;
  const rLo = wrap(loAng - bindLo - (upAng - bindUp)) * 180 / Math.PI;
  const bu = pose.bones['upper_arm_' + side] || (pose.bones['upper_arm_' + side] = { rot: 0, tx: 0, ty: 0, sx: 1, sy: 1 });
  const bl = pose.bones['lower_arm_' + side] || (pose.bones['lower_arm_' + side] = { rot: 0, tx: 0, ty: 0, sx: 1, sy: 1 });
  bu.rot = lerp(bu.rot, rUp, k.w); bl.rot = lerp(bl.rot, rLo, k.w);
}

// ---------- performer ----------
class Performer {
  constructor(p, idx) { Object.assign(this, p); this.idx = idx; }
  async init() {
    const j = await (await fetch(this.rig_url)).json();
    this.rig = await new Rig(j, this.rig_url.replace(/\/rig\.json$/, ''), GLS).load();
    const mv = this.movement == null ? .4 : this.movement;
    this.amp = clamp(.55 + .9 * mv, .6, 1.25);
    this.stills = this.still || [];
    // per-character clip overrides (rig.json clips.overrides, then timeline) on top of the shared library
    this.clipset = { ...CLIPS };
    const ovs = { ...((j.clips || {}).overrides || {}), ...(this.clip_overrides || {}) };
    for (const [n, ov] of Object.entries(ovs)) { const c = { ...(CLIPS[n] || {}), ...ov, tracks: { ...((CLIPS[n] || {}).tracks || {}), ...(ov.tracks || {}) } }; c._compiled = compileClip(c); this.clipset[n] = c; }
    // base layers always on
    this.clips = [{ clip: 'rest_pose', t0: -1, dur: 1e9, w: 1 }, ...(this.clips || [])];
    this.precompute();
  }
  stillAt(t) { let s = 0; for (const w of this.stills) s = Math.max(s, win(t, w.t0, w.t1, .3, .35)); return s; }
  exprAt(t, ov, ovW) {
    const ks = this.expr || []; let cur = EXPR[ks.length ? ks[0].expr : 'neutral'] || EXPR.neutral, prev = cur, tk = -9;
    for (const k of ks) if (k.t <= t) { prev = cur; cur = EXPR[k.expr] || EXPR.neutral; tk = k.t; }
    const a = smooth((t - tk) / .22), o = {};
    for (const key of ['lid', 'tilt', 'bot', 'brow', 'browTilt', 'smile', 'pup', 'asym']) o[key] = lerp(prev[key] || 0, cur[key] || 0, a);
    if (ov && EXPR[ov]) for (const key of Object.keys(o)) o[key] = lerp(o[key], EXPR[ov][key] || 0, .85);
    return o;
  }
  mouthAt(t) {
    const cues = this.mouth; if (!cues || !cues.length) return { ...VIS.X, teethMode: 0 };
    let i = 0; while (i + 1 < cues.length && cues[i + 1].t <= t) i++;
    const c = cues[i], p = i > 0 ? cues[i - 1] : c, a = smooth((t - c.t) / .055);
    const A = VIS[p.v] || VIS.X, Bv = VIS[c.v] || VIS.X, o = {};
    for (const k of ['open', 'wide', 'round', 'teeth', 'tongue', 'tongueUp', 'press']) o[k] = lerp(A[k] || 0, Bv[k] || 0, a);
    o.teethMode = Bv.teeth; return o;
  }
  gazeAt(t) {
    let x = 0, y = 0, px = 0, py = 0, tk = -9;
    for (const l of this.looks || []) if (l.t <= t) { px = x; py = y; x = l.x; y = l.y; tk = l.t; }
    const a = smooth((t - tk) / .09); return [lerp(px, x, a), lerp(py, y, a)];
  }
  blinkAt(t) {
    let v = 0;
    for (const b of this.blinks || []) { const d = t - b.t, c = b.slow ? 2.2 : 1; if (d < 0 || d > .26 * c) continue;
      if (d < .06 * c) v = Math.max(v, d / (.06 * c)); else if (d < .1 * c) v = 1; else v = Math.max(v, 1 - (d - .1 * c) / (.16 * c)); }
    return smooth(v);
  }
  envAt(t) { const e = this.env; if (!e) return 0; const i = t * (this.env_fps || 30); const i0 = Math.floor(i); if (i0 < 0 || i0 + 1 >= e.length) return 0; return lerp(e[i0], e[i0 + 1], i - i0); }

  // deterministic sequential pass: clips -> IK -> head spring (eyes lead) -> secondary physics
  precompute() {
    const R = this.rig, n = Math.ceil(TL.duration * TL.fps) + 2, dt = 1 / TL.fps, sub = 4, h = dt / sub;
    this.frames = new Array(n);
    const phys = (R.j.physics || []).filter(p => R.bi[p.bone]).map(p => ({ ...p, th: 0, om: 0 }));
    for (const s of ['r', 'l']) if (R.bi['hand_' + s]) phys.push({ bone: 'hand_' + s, stiffness: 120, damping: 11, gain: .55, galpha: .35, th: 0, om: 0 });
    phys.push({ bone: 'head', stiffness: 160, damping: 13, gain: .25, galpha: .5, th: 0, om: 0, lagOnly: true });
    const hist = {};
    let yaw = 0, yawV = 0, pitch = 0, pitchV = 0;
    for (let f = 0; f < n; f++) {
      const t = f * dt, pose = evalPose(this, t);
      solveIK(this, pose, 'r'); solveIK(this, pose, 'l');
      // talking energy: tiny head bob + chest
      const env = this.envAt(t), still = this.stillAt(t), en = this.energy == null ? .5 : this.energy;
      const hb = pose.bones.head || (pose.bones.head = { rot: 0, tx: 0, ty: 0, sx: 1, sy: 1 });
      hb.rot += env * 1.4 * en * (1 - .85 * still) * Math.sin(t * 5.1 + this.idx * 1.3);
      hb.ty += env * .006 * R.unit * (1 - .85 * still);
      // gaze + head yaw/pitch springs (eyes lead, head follows with overshoot)
      const g = this.gazeAt(t);
      const gx = clamp(g[0] + (pose.face.gaze_x || 0) + (this.base_look ? this.base_look[0] : 0), -1, 1), gy = clamp(g[1] + (pose.face.gaze_y || 0), -1, 1);
      const yt = clamp(gx * .58 + (pose.face.yaw || 0), -1, 1), pt = clamp(gy * .3 + (pose.face.pitch || 0), -1, 1);
      for (let s = 0; s < sub; s++) {
        const w0 = 13, z = .62;
        yawV += (-2 * z * w0 * yawV - w0 * w0 * (yaw - yt)) * h; yaw += yawV * h;
        pitchV += (-2 * z * w0 * pitchV - w0 * w0 * (pitch - pt)) * h; pitch += pitchV * h;
      }
      if (f === 0) { yaw = yt; pitch = pt; }
      pose.yaw = yaw; pose.pitch = pitch;
      // secondary physics on bones (driven by parent motion of the kinematic pose)
      const Wd = R.world(pose);
      for (const p of phys) {
        const b = R.bi[p.bone], par = b.p;
        const pos = OddRig.apply(Wd.D[b.i], b.x, b.y);
        const pr = par ? Math.atan2(Wd.D[par.i][1], Wd.D[par.i][0]) : 0;
        const H0 = hist[p.bone] || (hist[p.bone] = { p: [pos, pos], r: [pr, pr] });
        const ax = (pos[0] - 2 * H0.p[1][0] + H0.p[0][0]) / (dt * dt), ay = (pos[1] - 2 * H0.p[1][1] + H0.p[0][1]) / (dt * dt);
        const alpha = (pr - 2 * H0.r[1] + H0.r[0]) / (dt * dt);
        H0.p = [H0.p[1], pos]; H0.r = [H0.r[1], pr];
        if (f < 2) continue;
        // direction of the bone (to its child, else parent's direction)
        const ch = R.bones.find(c => c.parent === b.name);
        let dx = 0, dy = 1; if (ch) { dx = ch.x - b.x; dy = ch.y - b.y; } else if (par) { dx = b.x - par.x; dy = b.y - par.y; }
        const len = Math.max(30, Math.hypot(dx, dy)); const c0 = Math.cos(pr), s0 = Math.sin(pr);
        const wx = dx * c0 - dy * s0, wy = dx * s0 + dy * c0;
        const nx = -wy / len, ny = wx / len;   // +theta moves the tip along n
        const drive = -(ax * nx + ay * ny) / len * p.gain - alpha * (p.galpha == null ? .9 : p.galpha);
        for (let s = 0; s < sub; s++) { p.om += (-p.stiffness * p.th - p.damping * p.om + clamp(drive, -400, 400)) * h; p.th += p.om * h; }
        p.th = clamp(p.th, -.5, .5);
        const bb = pose.bones[p.bone] || (pose.bones[p.bone] = { rot: 0, tx: 0, ty: 0, sx: 1, sy: 1 });
        bb.rot += p.th * 180 / Math.PI;
      }
      // face state
      const e = this.exprAt(t, pose.exprOv), m = this.mouthAt(t), bl = this.blinkAt(t);
      if (pose.face.open) m.open = Math.max(m.open, pose.face.open);
      const eyeX = clamp(gx - yaw * 1.1, -1, 1), eyeY = clamp(gy - pitch * .8, -1, 1);
      const lid = e.lid + (pose.face.lid || 0);
      const F = { lid: bl > .01 ? lerp(lid, 1.0, bl) : lid, blink: bl, tilt: e.tilt, bot: e.bot, smile: e.smile + (pose.face.smile || 0), pup: e.pup, asym: e.asym || 0,
        browLift: (e.brow + (pose.face.brow || 0)) * R.unit * .028, browTilt: (e.browTilt + (pose.face.brow_tilt || 0)) * R.unit * .02,
        slit: clamp(pose.face.slit || 0, 0, 1), look: [eyeX, eyeY], mouth: m };
      this.frames[f] = { bones: pose.bones, hands: pose.hands, yaw: pose.yaw, pitch: pose.pitch, F, glitch: pose.fx.glitch || 0 };
    }
  }
  placeMatrix(cam) {
    const R = this.rig, root = R.bi.root, pl = this.place;
    const Pm = [pl.scale, 0, 0, pl.scale, pl.x - root.x * pl.scale, pl.y - root.y * pl.scale];
    return OddRig.mul(cam, Pm);
  }
}

// ---------- stage chrome ----------
function drawBackground(c, t) {
  const bg = IMG.bg; const s = Math.max(W / bg.width, H / bg.height) * (TL.bgScale || 1.06); const drift = Math.sin(t * .12) * 6;
  c.drawImage(bg, (W - bg.width * s) / 2 + drift, (H - bg.height * s) / 2, bg.width * s, bg.height * s);
}
function drawVignette(c) {
  const g = c.createRadialGradient(W / 2, H * .45, H * .25, W / 2, H * .5, H * .78); g.addColorStop(0, 'rgba(0,0,0,0)'); g.addColorStop(1, 'rgba(8,6,14,.55)');
  c.fillStyle = g; c.fillRect(0, 0, W, H);
  const w = c.createRadialGradient(W * .92, H * .38, 10, W * .92, H * .38, W * .9); w.addColorStop(0, 'rgba(255,190,110,.14)'); w.addColorStop(1, 'rgba(255,190,110,0)');
  c.save(); c.globalCompositeOperation = 'screen'; c.fillStyle = w; c.fillRect(0, 0, W, H); c.restore();
}
function grain(c, t) { const n = IMG.grain; c.save(); c.globalAlpha = .05; c.globalCompositeOperation = 'overlay'; const ox = Math.floor(hash(Math.floor(t * 24)) * 256), oy = Math.floor(hash(Math.floor(t * 24) + 9) * 256); c.fillStyle = c.createPattern(n, 'repeat'); c.translate(-ox, -oy); c.fillRect(0, 0, W + 256, H + 256); c.restore(); }
function roundRect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }
function drawCaptions(c, t) {
  const cap = (TL.captions || []).find(k => t >= k.t0 && t <= k.t1 + .05); if (!cap) return;
  const y0 = TL.captionY || 1560; c.save(); c.font = '78px Lilita'; c.lineJoin = 'round';
  const space = 20, maxW = 920, lines = [[]]; let lw = 0;
  for (const w of cap.words) { const ww = c.measureText(w.w).width; if (lw + ww > maxW && lines[lines.length - 1].length) { lines.push([]); lw = 0; } lines[lines.length - 1].push({ ...w, ww }); lw += ww + space; }
  const pop = smooth((t - cap.t0) / .12);
  lines.forEach((ln, li) => { const tw = ln.reduce((a, w) => a + w.ww, 0) + space * (ln.length - 1); let x = (W - tw) / 2; const y = y0 + li * 92;
    for (const w of ln) { const active = t >= w.t0 - .02 && t <= w.t1 + .08; const sc = (active ? 1.08 : 1) * (.9 + .1 * pop);
      c.save(); c.translate(x + w.ww / 2, y - 26); c.scale(sc, sc); c.translate(-(x + w.ww / 2), -(y - 26));
      c.lineWidth = 14; c.strokeStyle = '#1b1020'; c.strokeText(w.w, x, y); c.fillStyle = active ? '#ffd23f' : '#ffffff'; c.fillText(w.w, x, y); c.restore(); x += w.ww + space; } });
  c.restore();
}
function drawNameTag(c, t) {
  for (const tag of TL.tags || []) { const k = win(t, tag.t0, tag.t1, .35, .35); if (k <= 0) continue;
    c.save(); c.globalAlpha = k; c.translate(lerp(-40, 0, easeOutBack(k)), 0);
    c.font = '52px Lilita'; const w1 = c.measureText(tag.name).width; c.font = '600 32px Nunito'; const w2 = c.measureText(tag.sub).width;
    const bw = Math.max(w1, w2) + 70, x = tag.x || 60, y = tag.y || 150;
    roundRect(c, x, y, bw, 128, 26); c.fillStyle = 'rgba(22,16,34,.82)'; c.fill(); c.fillStyle = tag.color || '#7ed957'; roundRect(c, x, y, 14, 128, 7); c.fill();
    c.fillStyle = '#fff'; c.font = '52px Lilita'; c.fillText(tag.name, x + 36, y + 62); c.fillStyle = 'rgba(255,255,255,.75)'; c.font = '600 32px Nunito'; c.fillText(tag.sub, x + 36, y + 104); c.restore(); }
  // persistent small disclosure labels (e.g. "AI voice: Kokoro stand-in"), bottom of frame
  for (const lb of TL.labels || []) {
    c.save(); c.font = '600 26px Nunito'; const w = c.measureText(lb.text).width, x = lb.x || (W - w) / 2, y = lb.y || 1858;
    roundRect(c, x - 16, y - 30, w + 32, 42, 14); c.fillStyle = 'rgba(22,16,34,.62)'; c.fill();
    c.fillStyle = 'rgba(255,255,255,.85)'; c.fillText(lb.text, x, y); c.restore(); }
}
function cameraAt(t) {
  let z = 1, fx = W / 2, fy = 800;
  if (TL.shots && TL.shots.length) { let s = TL.shots[0]; for (const sh of TL.shots) if (sh.t <= t) s = sh; z = s.zoom; fx = s.fx; fy = s.fy; }
  for (const P of PERF) {
    for (const cm of P.camera || []) { const k = win(t, cm.t0, cm.t1, Math.min(1.2, cm.t1 - cm.t0), .45); z *= 1 + cm.amt * EASE.io(k); }
    const fr = P.frames[Math.min(P.frames.length - 1, Math.round(t * TL.fps))]; if (fr && fr.glitch) z *= 1 + .05 * fr.glitch;
  }
  const ox = noise1(t * .25) * 3, oy = noise1(t * .21 + 4) * 3;
  return { z, fx, fy, ox, oy, m: [z, 0, 0, z, W / 2 + ox - fx * z, fy + oy - fy * z] };
}

function frame(i) {
  const t = i / TL.fps, cam = cameraAt(t);
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H);
  ctx.save(); ctx.setTransform(cam.m[0], 0, 0, cam.m[3], cam.m[4], cam.m[5]); drawBackground(ctx, t); ctx.restore();
  GLS.clear();
  for (const P of PERF) {
    const fr = P.frames[Math.min(P.frames.length - 1, i)];
    P.rig.draw({ bones: fr.bones, hands: fr.hands, yaw: fr.yaw, pitch: fr.pitch }, fr.F, P.placeMatrix(cam.m), { glitch: fr.glitch });
  }
  // soft contact shadow from the rendered silhouettes
  const sc = SHADOW.getContext('2d'); sc.setTransform(1, 0, 0, 1, 0, 0); sc.clearRect(0, 0, SHADOW.width, SHADOW.height);
  sc.filter = 'blur(7px) brightness(0)'; sc.drawImage(glc, 0, 0, SHADOW.width, SHADOW.height); sc.filter = 'none';
  ctx.save(); ctx.globalAlpha = .32; ctx.drawImage(SHADOW, 22, 16, W, H); ctx.restore();
  ctx.drawImage(glc, 0, 0);
  drawVignette(ctx); grain(ctx, t); drawNameTag(ctx, t); drawCaptions(ctx, t);
  if (TL.fadeIn) { const k = 1 - smooth(t / TL.fadeIn); if (k > 0) { ctx.fillStyle = `rgba(0,0,0,${k})`; ctx.fillRect(0, 0, W, H); } }
  if (t > TL.duration - .4) { ctx.fillStyle = `rgba(0,0,0,${smooth((t - (TL.duration - .4)) / .4)})`; ctx.fillRect(0, 0, W, H); }
}

window.STAGE_INIT = async function (tl) {
  TL = tl;
  await document.fonts.load('78px Lilita'); await document.fonts.load('600 32px Nunito');
  IMG.bg = await loadImg(tl.background);
  const g = document.createElement('canvas'); g.width = g.height = 256; const gc = g.getContext('2d'); const id = gc.createImageData(256, 256);
  for (let k = 0; k < id.data.length; k += 4) { const v = Math.floor(hash(k * .37) * 255); id.data[k] = id.data[k + 1] = id.data[k + 2] = v; id.data[k + 3] = 255; } gc.putImageData(id, 0, 0); IMG.grain = g;
  SHADOW = document.createElement('canvas'); SHADOW.width = W / 2; SHADOW.height = H / 2;
  GLS = new GLStage(glc);
  const names = await (await fetch('/clips/index.json')).json();
  for (const n of names) CLIPS[n] = await (await fetch('/clips/' + n + '.json')).json();
  for (const c of Object.values(CLIPS)) if (c.base && CLIPS[c.base]) { c.tracks = { ...CLIPS[c.base].tracks, ...c.tracks }; c.ik = c.ik || CLIPS[c.base].ik; }
  for (const c of Object.values(CLIPS)) c._compiled = compileClip(c);
  PERF = [];
  for (const [idx, p] of tl.performers.entries()) { const P = new Performer(p, idx); await P.init(); PERF.push(P); }
  return true;
};
window.STAGE_FRAME = function (i, q) { frame(i); return cv.toDataURL('image/jpeg', q || 0.93); };
