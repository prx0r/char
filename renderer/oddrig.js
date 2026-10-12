// OddRig runtime (schema oddrig/1): skeletal 2D rig with skinned meshes, rendered with WebGL2.
// - bones: bind pose = world positions with zero rotation; animation = local (rot, tx, ty, sx, sy) deltas
// - meshes: per-vertex weights; parent/child pairs use rotational blend (no candy-wrapper collapse at
//   elbows/neck), anything else uses linear blend skinning
// - head: several view attachments (front / 3/4 / profile, mirrored), crossfaded at yaw thresholds,
//   with per-vertex depth parallax so painted features slide inside the silhouette
// - face: painted art + procedural eyes (clipped to the painted eye mask), lids, jaw/vector mouth,
//   brow-region mesh warp; redrawn per frame into a canvas that is uploaded as the head texture
'use strict';
(function (global) {
const D2R = Math.PI / 180;
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };

// affine m = [a, b, c, d, e, f]:  x' = a*x + c*y + e,  y' = b*x + d*y + f
function mul(m, n) { return [m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1], m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3], m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5]]; }
function about(px, py, tx, ty, rot, sx, sy) { // T(p + t) R S T(-p)
  const c = Math.cos(rot * D2R), s = Math.sin(rot * D2R);
  const a = c * sx, b = s * sx, cc = -s * sy, d = c * sy;
  return [a, b, cc, d, px + tx - (a * px + cc * py), py + ty - (b * px + d * py)];
}
function apply(m, x, y) { return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]]; }

const VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv; uniform vec2 u_res; out vec2 v_uv;
void main(){ v_uv = a_uv; vec2 p = a_pos / u_res * 2.0 - 1.0; gl_Position = vec4(p.x, -p.y, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float; in vec2 v_uv; uniform sampler2D u_tex; uniform vec4 u_tint; uniform float u_flat; out vec4 o;
void main(){ vec4 c = texture(u_tex, v_uv); vec3 rgb = mix(c.rgb * u_tint.rgb, u_tint.rgb * c.a, u_flat); o = vec4(rgb, c.a) * u_tint.a; }`;

class GLStage {
  constructor(canvas) {
    const gl = this.gl = canvas.getContext('webgl2', { premultipliedAlpha: true, alpha: true, antialias: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error('webgl2 unavailable');
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = this.prog = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    this.loc = { pos: gl.getAttribLocation(p, 'a_pos'), uv: gl.getAttribLocation(p, 'a_uv'), res: gl.getUniformLocation(p, 'u_res'), tex: gl.getUniformLocation(p, 'u_tex'), tint: gl.getUniformLocation(p, 'u_tint'), flat: gl.getUniformLocation(p, 'u_flat') };
    gl.useProgram(p); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    this.posBuf = gl.createBuffer(); this.uvBuf = gl.createBuffer(); this.idxBuf = gl.createBuffer();
    this.canvas = canvas;
  }
  texture(src) {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }
  update(t, src) { const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, t); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); gl.generateMipmap(gl.TEXTURE_2D); }
  clear() { const gl = this.gl; gl.viewport(0, 0, this.canvas.width, this.canvas.height); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); }
  draw(tex, pos, uv, idx, tint = [1, 1, 1, 1], flat = 0) {
    const gl = this.gl, L = this.loc;
    gl.uniform2f(L.res, this.canvas.width, this.canvas.height);
    gl.uniform4fv(L.tint, tint); gl.uniform1f(L.flat, flat);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(L.tex, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STREAM_DRAW);
    gl.enableVertexAttribArray(L.pos); gl.vertexAttribPointer(L.pos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvBuf); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STREAM_DRAW);
    gl.enableVertexAttribArray(L.uv); gl.vertexAttribPointer(L.uv, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.idxBuf); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STREAM_DRAW);
    gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_INT, 0);
  }
}

function loadImg(src) { return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error('img ' + src)); i.src = src; }); }
const rgb = c => `rgb(${c[0]},${c[1]},${c[2]})`;
const shade = (c, k) => [Math.round(c[0] * k), Math.round(c[1] * k), Math.round(c[2] * k)];

class Rig {
  constructor(json, base, stage) { this.j = json; this.base = base; this.st = stage; }
  async load() {
    const J = this.j, st = this.st;
    this.bones = J.skeleton.bones.map((b, i) => ({ ...b, i }));
    this.bi = {}; for (const b of this.bones) this.bi[b.name] = b;
    for (const b of this.bones) b.p = b.parent ? this.bi[b.parent] : null;
    this.imgs = {}; this.tex = {};
    const want = new Set();
    for (const [k, a] of Object.entries(J.attachments)) { want.add(a.image); if (a.eye_mask) want.add(a.eye_mask); }
    for (const src of want) this.imgs[src] = await loadImg(this.base + '/' + src);
    this.att = {};
    for (const [k, a] of Object.entries(J.attachments)) this.att[k] = this.prepMesh(k, a);
    // face views (incl. virtual mirrored views)
    this.views = [];
    const neckX = this.bi.neck.x;
    for (const v of J.face.views) {
      if (v.mirror_of) continue;
      const a = this.att[v.attachment];
      const cv = document.createElement('canvas'); cv.width = a.src.size[0]; cv.height = a.src.size[1];
      const view = { ...v, a, canvas: cv, ctx: cv.getContext('2d'), img: this.imgs[a.src.image], mask: this.imgs[a.src.eye_mask], mirror: false };
      view.tex = st.texture(cv);
      this.prepFace(view);
      this.views.push(view);
    }
    for (const v of J.face.views) {
      if (!v.mirror_of) continue;
      const srcv = this.views.find(x => x.view === v.mirror_of);
      // mirrored geometry about the neck bone x; same uvs/texture canvas
      const a = srcv.a, m = { ...a, bind: a.bind.slice() };
      for (let k = 0; k < m.bind.length; k += 2) m.bind[k] = 2 * neckX - a.bind[k];
      m.browW = a.browW;
      this.views.push({ ...srcv, view: v.view, yaw: v.yaw, a: m, mirror: true });
    }
    this.views.sort((p, q) => p.yaw - q.yaw);
    for (const k of Object.keys(J.attachments)) if (!k.startsWith('head_')) this.tex[k] = st.texture(this.imgs[J.attachments[k].image]);
    // hand textures are shared by both sides
    this.unit = J.unit;
    this.scratch = new Float32Array(0);
    return this;
  }
  prepMesh(name, a) {
    const n = a.vertices.length, bind = new Float32Array(n * 2), uv = new Float32Array(n * 2);
    a.vertices.forEach((v, i) => { bind[2 * i] = v[0]; bind[2 * i + 1] = v[1]; });
    a.uvs.forEach((v, i) => { uv[2 * i] = v[0]; uv[2 * i + 1] = v[1]; });
    const idx = new Uint32Array(a.triangles);
    // skin modes
    const sk = a.weights.map(ws => {
      if (ws.length === 1) return { m: 1, b: this.bi[ws[0][0]].i };
      if (ws.length === 2) {
        const A = this.bi[ws[0][0]], B = this.bi[ws[1][0]];
        if (B.parent === A.name) return { m: 2, P: A.i, C: B.i, w: ws[1][1] };
        if (A.parent === B.name) return { m: 2, P: B.i, C: A.i, w: ws[0][1] };
      }
      return { m: 3, l: ws.map(([b, w]) => [this.bi[b].i, w]) };
    });
    return { name, src: a, bind, uv, idx, sk, depth: a.depth ? new Float32Array(a.depth) : null, n };
  }
  prepFace(v) {
    // brow-region warp weights: soft blobs above each eye, signed by side
    const a = v.a, n = a.n, bw = new Float32Array(n * 2);
    const eyes = v.eyes || [];
    const mx = eyes.length ? eyes.reduce((s, e) => s + e.cx, 0) / eyes.length : this.bi.head.x;
    eyes.forEach(e => {
      const side = eyes.length > 1 ? Math.sign(e.cx - mx) : 1;
      const bx = e.cx, by = e.cy - e.ry * 1.25, rx = e.rx * 1.7, ry = e.ry * 1.1;
      for (let i = 0; i < n; i++) {
        const dx = (a.bind[2 * i] - bx) / rx, dy = (a.bind[2 * i + 1] - by) / ry;
        const g = Math.exp(-(dx * dx + dy * dy) * 1.6);
        if (g > bw[2 * i]) { bw[2 * i] = g; bw[2 * i + 1] = side * clamp(dx, -1, 1); }
      }
    });
    a.browW = bw;
  }
  // pose: {bones: {name: {rot,tx,ty,sx,sy}}} in screen-space degrees / rig px
  world(pose) {
    const W = [], Dl = [], L = [];
    for (const b of this.bones) {
      const q = pose.bones[b.name] || {};
      const ox = b.p ? b.x - b.p.x : b.x, oy = b.p ? b.y - b.p.y : b.y;
      // local transform about the bone's own bind position, expressed in bind-world coords
      const loc = about(b.x, b.y, q.tx || 0, q.ty || 0, q.rot || 0, q.sx == null ? 1 : q.sx, q.sy == null ? 1 : q.sy);
      L[b.i] = { q, loc };
      Dl[b.i] = b.p ? mul(Dl[b.p.i], loc) : loc;   // delta = world * inv(bindWorld); with zero bind rotation it chains directly
    }
    return { D: Dl, L };
  }
  skin(att, W, pre) {
    const n = att.n, out = new Float32Array(n * 2), bind = att.bind, sk = att.sk;
    for (let i = 0; i < n; i++) {
      let x = bind[2 * i], y = bind[2 * i + 1];
      if (pre) { const o = pre(i, x, y); x = o[0]; y = o[1]; }
      const s = sk[i];
      let p;
      if (s.m === 1) p = apply(W.D[s.b], x, y);
      else if (s.m === 2) {
        const C = this.bones[s.C], q = W.L[s.C].q, w = s.w;
        const part = about(C.x, C.y, (q.tx || 0) * w, (q.ty || 0) * w, (q.rot || 0) * w, lerp(1, q.sx == null ? 1 : q.sx, w), lerp(1, q.sy == null ? 1 : q.sy, w));
        p = apply(mul(W.D[s.P], part), x, y);
      } else {
        let px = 0, py = 0;
        for (const [b, w] of s.l) { const r = apply(W.D[b], x, y); px += r[0] * w; py += r[1] * w; }
        p = [px, py];
      }
      out[2 * i] = p[0]; out[2 * i + 1] = p[1];
    }
    return out;
  }
  boneWorld(W, name) { const b = this.bi[name]; return apply(W.D[b.i], b.x, b.y); }
  anchor(W, name) { const a = this.j.anchors[name]; if (!a) return null; return apply(W.D[this.bi[a[2]].i], a[0], a[1]); }

  // ---------- face drawing into a view canvas (texture space) ----------
  drawFace(v, F, mirror) {
    const c = v.ctx, J = this.j, ox = v.origin[0], oy = v.origin[1];
    c.setTransform(1, 0, 0, 1, 0, 0); c.clearRect(0, 0, v.canvas.width, v.canvas.height);
    c.drawImage(v.img, 0, 0);
    const lookX = mirror ? -F.look[0] : F.look[0];
    if (v.mouth) this.drawMouth(c, v, F, ox, oy, mirror);
    if (v.eyes && v.eyes.length) this.drawEyes(c, v, F, ox, oy, lookX, mirror);
  }
  drawEyes(c, v, F, ox, oy, lookX, mirror) {
    const E = this.j.face.eyes, skin = v.skin, ink = v.ink;
    const tmp = this._etmp || (this._etmp = document.createElement('canvas'));
    if (tmp.width !== v.canvas.width || tmp.height !== v.canvas.height) { tmp.width = v.canvas.width; tmp.height = v.canvas.height; }
    const t = tmp.getContext('2d'); t.setTransform(1, 0, 0, 1, 0, 0); t.clearRect(0, 0, tmp.width, tmp.height);
    const mx = v.eyes.reduce((s, e) => s + e.cx, 0) / v.eyes.length;
    for (const e of v.eyes) {
      const cx = e.cx - ox, cy = e.cy - oy, rx = e.rx + 3, ry = e.ry + 3;
      let side = v.eyes.length > 1 ? Math.sign(e.cx - mx) : 1; if (mirror) side = -side;
      const asym = (F.asym || 0) * side;
      const lid = clamp(Math.max(F.lid + asym * .5, F.blink), -0.05, 1.0);
      if (E.type === 'human') {
        t.fillStyle = E.white || '#fbf8f2'; t.fillRect(cx - rx, cy - ry, 2 * rx, 2 * ry);
        const ir = Math.min(e.rx, e.ry) * (E.iris_scale || 0.62);
        const ix = cx + lookX * e.rx * .38, iy = cy + F.look[1] * e.ry * .3;
        t.fillStyle = E.iris || '#6b4a2e'; t.beginPath(); t.ellipse(ix, iy, ir * Math.min(1, e.rx / e.ry * 1.1), ir, 0, 0, 7); t.fill();
        t.fillStyle = '#1a100a'; t.beginPath(); t.ellipse(ix, iy, ir * .48 * F.pup, ir * .48 * F.pup, 0, 0, 7); t.fill();
        t.fillStyle = 'rgba(255,255,255,.9)'; t.beginPath(); t.arc(ix + ir * .35, iy - ir * .4, ir * .25, 0, 7); t.fill();
      } else {
        const col = e.color;
        const g = t.createRadialGradient(cx - rx * .25, cy - ry * .35, 2, cx, cy, Math.max(rx, ry) * 1.05);
        g.addColorStop(0, rgb(shade(col, 1.04).map(x => Math.min(255, x)))); g.addColorStop(.62, rgb(col)); g.addColorStop(1, rgb(shade(col, .86)));
        t.fillStyle = g; t.fillRect(cx - rx, cy - ry, 2 * rx, 2 * ry);
        const sl = F.slit || 0, foreshort = clamp(e.rx / e.ry, .3, 1.2);
        const prx = lerp(e.rx * .24, 2.2, sl) * F.pup * foreshort, pry = lerp(e.ry * .5, e.ry * .78, sl) * F.pup;
        const px = cx + lookX * e.rx * .38, py = cy + F.look[1] * e.ry * .26 + e.ry * .04;
        if (sl > .05) { const rg = t.createRadialGradient(px, py, 2, px, py, e.ry * .95); rg.addColorStop(0, `rgba(235,120,35,${.6 * sl})`); rg.addColorStop(1, 'rgba(235,120,35,0)'); t.fillStyle = rg; t.fillRect(cx - rx, cy - ry, 2 * rx, 2 * ry); }
        t.fillStyle = E.pupil_color || '#2e140c'; t.beginPath(); t.ellipse(px, py, Math.max(1.5, prx), pry, 0, 0, 7); t.fill();
        t.fillStyle = 'rgba(255,255,250,.92)'; t.beginPath(); t.ellipse(cx + e.rx * .28 * foreshort, cy - e.ry * .45, e.rx * .2 * foreshort, e.ry * .14, -.5, 0, 7); t.fill();
      }
      // upper lid: skin coloured, descends; tilt: + => inner corner lower (angry); inner side is toward face centre (-side)
      const lidY = cy - ry + lid * 2 * ry * 1.02, tl = (F.tilt || 0) * -side * ry;
      const x0 = cx - rx - 4, x1 = cx + rx + 4, y0 = lidY - tl, y1 = lidY + tl, sag = ry * .16 * (1 - lid);
      const lidc = shade(skin, E.type === 'human' ? 1 : .92);
      t.beginPath(); t.moveTo(x0, cy - ry - 30); t.lineTo(x1, cy - ry - 30); t.lineTo(x1, y1); t.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0); t.closePath();
      const lg = t.createLinearGradient(0, cy - ry, 0, Math.max(y0, y1)); lg.addColorStop(0, rgb(lidc)); lg.addColorStop(1, rgb(shade(lidc, .88)));
      t.fillStyle = lg; t.fill();
      if (lid > -0.02) {
        t.save(); t.globalAlpha = .2; t.beginPath(); t.moveTo(x1, y1); t.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0); t.lineTo(x0, y0 + ry * .2); t.quadraticCurveTo(cx, (y0 + y1) / 2 + sag + ry * .2, x1, y1 + ry * .2); t.closePath(); t.fillStyle = '#3a2410'; t.fill(); t.restore();
        t.strokeStyle = rgb(ink); t.lineWidth = Math.max(2.5, ry * .11); t.lineCap = 'round';
        t.beginPath(); t.moveTo(x1, y1); t.quadraticCurveTo(cx, (y0 + y1) / 2 + sag, x0, y0); t.stroke();
      }
      // lower lid
      const bot = F.bot || 0; if (bot > .01) {
        const by = cy + ry - bot * ry * 1.1;
        t.beginPath(); t.moveTo(x0, cy + ry + 30); t.lineTo(x1, cy + ry + 30); t.lineTo(x1, by); t.quadraticCurveTo(cx, by - ry * .1 * bot, x0, by); t.closePath();
        t.fillStyle = rgb(lidc); t.fill(); t.strokeStyle = rgb(ink); t.lineWidth = Math.max(2, ry * .08); t.beginPath(); t.moveTo(x1, by); t.quadraticCurveTo(cx, by - ry * .1 * bot, x0, by); t.stroke();
      }
    }
    // clip to the painted eye interiors (keeps occlusion by the snout and the painted outline)
    t.globalCompositeOperation = 'destination-in'; t.drawImage(v.mask, 0, 0); t.globalCompositeOperation = 'source-over';
    c.drawImage(tmp, 0, 0);
  }
  drawMouth(c, v, F, ox, oy, mirror) {
    const M = this.j.face.mouth, mo = v.mouth, m = F.mouth;
    const [A, B, C] = mo.coef; const lineY = x => { const X = x + ox; return A * X * X + B * X + C - oy; };
    const x0 = mo.x0 - ox, x1 = mo.x1 - ox;
    const Dmax = (M.max_open || .12) * this.unit;
    const D = clamp(m.open, 0, 1.2) * Dmax;
    if (M.type === 'jaw') {
      if (D < 0.6) return;
      const span = x1 - x0, wide = clamp(m.wide * (1 - m.round * .3), .4, 1.2);
      let hinge = mo.hinge; if (mirror && hinge !== 'both') hinge = hinge === 'left' ? 'right' : 'left';
      let a0 = x0, a1 = x1;
      if (hinge === 'both') { const cm = (x0 + x1) / 2, h = span / 2 * wide; a0 = cm - h; a1 = cm + h; }
      else if (hinge === 'right') a1 = x0 + span * clamp(wide, .5, 1); else a0 = x1 - span * clamp(wide, .5, 1);
      const prof = x => {
        let u; if (hinge === 'both') u = 1 - Math.abs((x - (a0 + a1) / 2) / ((a1 - a0) / 2));
        else if (hinge === 'right') u = (a1 - x) / (a1 - a0); else u = (x - a0) / (a1 - a0);
        if (u <= 0) return 0;
        return hinge === 'both' ? Math.pow(Math.sin(Math.PI / 2 * clamp(u, 0, 1)), .8) : Math.pow(clamp(u, 0, 1), .7) * (u > .93 ? 1 - (u - .93) * 3 : 1);
      };
      const depth = mo.jaw_depth;
      // interior
      c.save(); c.beginPath();
      for (let x = a0; x <= a1; x += 3) c.lineTo(x, lineY(x) - 1);
      for (let x = a1; x >= a0; x -= 3) c.lineTo(x, lineY(x) + D * prof(x) + 3);
      c.closePath(); c.fillStyle = M.interior || '#3d1414'; c.fill(); c.clip();
      const xm = hinge === 'both' ? (a0 + a1) / 2 : (hinge === 'right' ? a0 + (a1 - a0) * .3 : a1 - (a1 - a0) * .3);
      const tg = c.createRadialGradient(xm, lineY(xm) + D * .9, 2, xm, lineY(xm) + D, (a1 - a0) * .45); tg.addColorStop(0, '#160404'); tg.addColorStop(1, 'rgba(22,4,4,0)');
      c.fillStyle = tg; c.fillRect(a0, lineY(xm) - 30, a1 - a0, D + 80);
      if (m.tongue > .05 || D > 8) { c.fillStyle = M.tongue || '#c8626a'; c.beginPath(); c.ellipse(xm + 4, lineY(xm) + D * prof(xm) * (1 - .45 * (m.tongueUp || 0)) + 5, (a1 - a0) * .2, D * .36 + 3, 0, 0, 7); c.fill(); }
      if (D > 9) { c.fillStyle = M.teeth || '#f4efd9'; const k = Math.min(1, (D - 9) / 8); for (let x = a0 + (a1 - a0) * .18; x <= a1 - (a1 - a0) * .14; x += (a1 - a0) * .16) { const y = lineY(x) - 1; c.beginPath(); c.moveTo(x - 5, y); c.lineTo(x + 5, y); c.lineTo(x, y + 8 * k); c.closePath(); c.fill(); } }
      c.restore();
      // lower jaw columns slide down (rigid per column, tapered by the hinge profile)
      for (let x = Math.floor(a0) - 2; x < a1 + 2; x += 2) {
        const d = D * prof(x + 1); if (d < .3) continue;
        const y = lineY(x) + 1;
        c.drawImage(v.img, x, y, 2, depth, x, y + d, 2, depth);
      }
      // re-ink upper lip line across the opening
      c.strokeStyle = rgb(v.ink); c.lineWidth = Math.max(3, this.unit * .013); c.lineCap = 'round';
      c.beginPath(); for (let x = a0; x <= a1; x += 3) { const y = lineY(x); if (x === a0) c.moveTo(x, y); else c.lineTo(x, y); } c.stroke();
    } else {
      // vector mouth: paint out the drawn mouth with skin, then draw the shape
      const cx = (x0 + x1) / 2, cy = lineY(cx), hw0 = (x1 - x0) / 2;
      c.save(); c.fillStyle = rgb(v.skin); c.beginPath(); c.ellipse(cx, cy, hw0 * 1.12, Math.max(10, hw0 * .38), 0, 0, 7); c.filter = 'blur(2px)'; c.fill(); c.restore();
      const hw = hw0 * m.wide * (1 - m.round * .14), open = m.open * Dmax, smile = F.smile || 0;
      const lx = cx - hw, rx = cx + hw, ly = cy - smile * hw * .12, ry = ly;
      c.strokeStyle = rgb(v.ink); c.lineCap = 'round'; c.lineJoin = 'round'; c.lineWidth = Math.max(3, hw * .08);
      if (open < 1.2) { c.beginPath(); c.moveTo(lx, ly); c.quadraticCurveTo(cx, cy + smile * hw * .2, rx, ry); c.stroke(); return; }
      const upC = cy - open * .2 - smile * 2 - m.round * 3, loC = cy + open + smile * 4;
      c.beginPath(); c.moveTo(lx, ly); c.bezierCurveTo(lx + hw * .4, upC, rx - hw * .4, upC, rx, ry); c.bezierCurveTo(rx - hw * .25, loC + open * .25, lx + hw * .25, loC + open * .25, lx, ly); c.closePath();
      c.fillStyle = M.interior || '#4a1b1b'; c.fill(); c.save(); c.clip();
      if (m.teethMode >= 1) { c.fillStyle = M.teeth || '#f6f1e6'; c.fillRect(lx, upC - 4, hw * 2, Math.min(10, open * .45) + 4); }
      if (m.tongue > .05 && open > 6) { c.fillStyle = M.tongue || '#c46a6a'; c.beginPath(); c.ellipse(cx, loC + open * .2, hw * .45, open * .3 + 2, 0, 0, 7); c.fill(); }
      c.restore(); c.stroke();
    }
  }

  // ---------- full draw ----------
  // V: rig->screen affine. F: face state. pose: bones + hands + yaw/pitch. fx: {glitch}
  draw(pose, F, V, opts = {}) {
    const st = this.st, W = this.world(pose);
    const toScreen = (arr) => { const o = new Float32Array(arr.length); for (let i = 0; i < arr.length; i += 2) { o[i] = V[0] * arr[i] + V[2] * arr[i + 1] + V[4]; o[i + 1] = V[1] * arr[i] + V[3] * arr[i + 1] + V[5]; } return o; };
    const tint = opts.tint || [1, 1, 1, 1], flat = opts.flat || 0;
    const yaw = clamp(pose.yaw || 0, -1, 1), pitch = clamp(pose.pitch || 0, -1, 1);
    // which head views, with alpha
    const vs = this.views; let A = vs[0], Bv = vs[0], k = 0;
    for (let i = 0; i + 1 < vs.length; i++) if (yaw >= vs[i].yaw && yaw <= vs[i + 1].yaw) { A = vs[i]; Bv = vs[i + 1]; k = (yaw - A.yaw) / (Bv.yaw - A.yaw); break; }
    if (yaw > vs[vs.length - 1].yaw) { A = Bv = vs[vs.length - 1]; }
    const band = .1; const kk = smooth((k - (.5 - band)) / (2 * band));
    const headLayers = kk <= 0 ? [[A, 1]] : kk >= 1 ? [[Bv, 1]] : (kk < .5 ? [[A, 1], [Bv, kk * 2]] : [[Bv, 1], [A, (1 - kk) * 2]]);
    for (const slot of this.j.slots) {
      if (slot.kind === 'head_views') {
        for (const [v, alpha] of headLayers) {
          if (!opts.noFace) { this.drawFace(v, F, v.mirror); st.update(v.tex, v.canvas); }
          const res = yaw - v.yaw, hw = v.a.src.size[0];
          const bw = v.a.browW, lift = (F.browLift || 0), btilt = (F.browTilt || 0);
          const pre = (i, x, y) => {
            const d = v.a.depth ? v.a.depth[i] : 0;
            x += res * d * hw * .22; y += pitch * d * hw * .12;
            if (bw && bw[2 * i] > .01) y += bw[2 * i] * (-lift - btilt * bw[2 * i + 1]);
            return [x, y];
          };
          const pos = toScreen(this.skin(v.a, W, pre));
          st.draw(v.tex, pos, v.a.uv, v.a.idx, [tint[0], tint[1], tint[2], tint[3] * alpha], flat);
          if (opts.glitch && !flat) {
            st.draw(v.tex, pos.map((p, i) => i % 2 === 0 ? p - opts.glitch * 10 : p), v.a.uv, v.a.idx, [1, .25, .3, .35 * opts.glitch], 0);
            st.draw(v.tex, pos.map((p, i) => i % 2 === 0 ? p + opts.glitch * 10 : p), v.a.uv, v.a.idx, [.25, 1, .7, .35 * opts.glitch], 0);
          }
        }
        continue;
      }
      let an = slot.attachment;
      if (slot.kind === 'hand') { const side = slot.name.slice(-1); const pz = (pose.hands && pose.hands[side]) || 'rest'; an = this.j.hands.by_side[side][pz] || an; }
      const att = this.att[an]; if (!att) continue;
      const pos = toScreen(this.skin(att, W));
      const texKey = an.startsWith('hand_') ? an : an;
      let tx = this.tex[texKey];
      st.draw(tx, pos, att.uv, att.idx, tint, flat);
    }
    return W;
  }
}

global.OddRig = { GLStage, Rig, mul, about, apply, clamp, lerp, smooth };
})(window);
