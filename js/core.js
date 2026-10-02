// Core helpers shared by every chapter: colors, random data, math, UI controls,
// a Three.js stage wrapper and a small 2D plotting canvas.
window.ML = window.ML || {};
ML.chapters = {};
ML.register = (id, init) => { ML.chapters[id] = init; };

/* ---------------- DOM + colors ---------------- */
ML.el = (tag, attrs = {}, children = []) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c != null) e.append(c);
  return e;
};
ML.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
ML.colors = () => {
  const g = ML.css;
  return {
    bg: g("--bg"), surface: g("--surface"), surface2: g("--surface-2"), stage: g("--stage"),
    ink: g("--ink"), muted: g("--muted"), line: g("--line"), accent: g("--accent"),
    good: g("--good"), warn: g("--warn"), bad: g("--bad"), noise: g("--noise"),
    c: [g("--c0"), g("--c1"), g("--c2"), g("--c3")],
  };
};
ML.isDark = () => {
  const t = document.documentElement.getAttribute("data-theme");
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
};
// Distinct colors for an arbitrary number of clusters (DBSCAN can find many).
ML.clusterColor = (i) => {
  const base = ML.colors().c;
  if (i < 4) return base[i];
  const h = (i * 137.508) % 360;
  return `hsl(${h.toFixed(0)}, 62%, ${ML.isDark() ? 62 : 45}%)`;
};
ML.fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d));
ML.clamp = (v, a, b) => Math.max(a, Math.min(b, v));
ML.lerp = (a, b, t) => a + (b - a) * t;
ML.ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
ML.reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------- Random numbers + toy datasets ---------------- */
ML.rng = (seed = 1) => {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
ML.randn = (r) => {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
ML.makeMoons = (n, noise, seed = 3) => {
  const r = ML.rng(seed), X = [], y = [];
  const nOut = Math.floor(n / 2);
  for (let i = 0; i < n; i++) {
    const outer = i < nOut;
    const t = Math.PI * (outer ? i / Math.max(1, nOut - 1) : (i - nOut) / Math.max(1, n - nOut - 1));
    const x = outer ? Math.cos(t) : 1 - Math.cos(t);
    const yy = outer ? Math.sin(t) : 1 - Math.sin(t) - 0.5;
    X.push([x + noise * ML.randn(r), yy + noise * ML.randn(r)]);
    y.push(outer ? 0 : 1);
  }
  return { X, y };
};
ML.makeCircles = (n, noise, factor = 0.45, seed = 5) => {
  const r = ML.rng(seed), X = [], y = [];
  for (let i = 0; i < n; i++) {
    const inner = i % 2 === 1;
    const t = r() * Math.PI * 2, rad = inner ? factor : 1;
    X.push([rad * Math.cos(t) + noise * ML.randn(r), rad * Math.sin(t) + noise * ML.randn(r)]);
    y.push(inner ? 1 : 0);
  }
  return { X, y };
};
ML.makeBlobs = (centers, nPer, std, seed = 7) => {
  const r = ML.rng(seed), X = [], y = [];
  centers.forEach((c, k) => {
    for (let i = 0; i < nPer; i++) {
      X.push(c.map((v) => v + std * ML.randn(r)));
      y.push(k);
    }
  });
  return { X, y };
};
ML.shuffleIdx = (n, r) => {
  const a = [...Array(n).keys()];
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
};

/* ---------------- Linear algebra + stats ---------------- */
ML.mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
ML.std = (a) => { const m = ML.mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
ML.dist = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s); };
ML.colMeans = (X) => X[0].map((_, j) => ML.mean(X.map((r) => r[j])));
ML.colStds = (X) => X[0].map((_, j) => ML.std(X.map((r) => r[j])));
// StandardScaler: population standard deviation, like scikit-learn.
ML.scaler = (Xtrain) => {
  const mu = ML.colMeans(Xtrain), sd = ML.colStds(Xtrain).map((s) => (s === 0 ? 1 : s));
  return { mu, sd, transform: (X) => X.map((r) => r.map((v, j) => (v - mu[j]) / sd[j])) };
};
ML.solve = (A, b) => {
  const n = A.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / (M[i][i] || 1e-12));
};
// Least squares with an optional tiny ridge for numerical stability.
ML.lstsq = (Phi, y, ridge = 0) => {
  const p = Phi[0].length;
  const A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
  for (let i = 0; i < Phi.length; i++) {
    const r = Phi[i];
    for (let a = 0; a < p; a++) {
      b[a] += r[a] * y[i];
      for (let c = 0; c < p; c++) A[a][c] += r[a] * r[c];
    }
  }
  for (let a = 1; a < p; a++) A[a][a] += ridge;
  return ML.solve(A, b);
};
// Symmetric eigen-decomposition (Jacobi). Returns values sorted descending with matching vectors.
ML.eigSym = (S) => {
  const n = S.length, A = S.map((r) => r.slice());
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += A[i][j] ** 2;
    if (off < 1e-14) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-15) continue;
      const th = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [...Array(n).keys()].sort((a, b) => A[b][b] - A[a][a]);
  return { values: order.map((i) => A[i][i]), vectors: order.map((i) => V.map((r) => r[i])) };
};
ML.pca = (X) => {
  const mu = ML.colMeans(X), n = X.length, p = mu.length;
  const Xc = X.map((r) => r.map((v, j) => v - mu[j]));
  const S = Array.from({ length: p }, () => new Array(p).fill(0));
  for (const r of Xc) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) S[a][b] += (r[a] * r[b]) / (n - 1);
  const { values, vectors } = ML.eigSym(S);
  // Fix sign so the largest loading is positive (deterministic orientation).
  vectors.forEach((v) => {
    let m = 0; v.forEach((x, i) => { if (Math.abs(x) > Math.abs(v[m])) m = i; });
    if (v[m] < 0) v.forEach((x, i) => (v[i] = -x));
  });
  const total = values.reduce((s, v) => s + Math.max(v, 0), 0);
  const project = (Xn, k) => Xn.map((r) => vectors.slice(0, k).map((v) => v.reduce((s, vj, j) => s + vj * (r[j] - mu[j]), 0)));
  return { mu, values, vectors, ratio: values.map((v) => Math.max(v, 0) / total), project };
};

/* ---------------- UI controls ---------------- */
ML.ui = {};
let uid = 0;
ML.ui.id = (p = "c") => `${p}-${++uid}`;
ML.ui.slider = (parent, o) => {
  const id = o.id || ML.ui.id("s");
  const toVal = (pos) => (o.log ? Math.pow(10, pos) : pos);
  const toPos = (v) => (o.log ? Math.log10(v) : v);
  const input = ML.el("input", { type: "range", id, min: toPos(o.min), max: toPos(o.max), step: o.log ? 0.01 : o.step || 1, value: toPos(o.value) });
  const out = ML.el("output", { for: id });
  const fmt = o.fmt || ((v) => String(v));
  const wrap = ML.el("div", { class: "ctrl" }, [
    ML.el("div", { class: "ctrl-row" }, [ML.el("label", { for: id, html: o.label }), out]),
    input,
    o.note ? ML.el("div", { class: "ctrl-note", html: o.note }) : null,
  ]);
  const get = () => {
    let v = toVal(parseFloat(input.value));
    if (o.log) { const m = Math.pow(10, Math.floor(Math.log10(v)) - 1); v = Math.round(v / m) * m; }
    return +v.toPrecision(6);
  };
  const sync = () => { out.textContent = fmt(get()); };
  input.addEventListener("input", () => { sync(); o.onInput && o.onInput(get()); });
  sync();
  parent.append(wrap);
  const set = (v, fire = false) => { input.value = toPos(v); sync(); if (fire && o.onInput) o.onInput(get()); };
  input.mlSet = (v) => set(v, true);
  return { el: wrap, input, get, set };
};
ML.ui.seg = (parent, o) => {
  let value = o.value;
  const btns = o.options.map((op) => ML.el("button", { type: "button", class: op.v === value ? "on" : "", text: op.t, "aria-pressed": op.v === value ? "true" : "false" }));
  const seg = ML.el("div", { class: "seg", role: "group", "aria-label": o.label || "options" }, btns);
  btns.forEach((b, i) => b.addEventListener("click", () => set(o.options[i].v, true)));
  const set = (v, fire) => {
    value = v;
    btns.forEach((b, i) => { const on = o.options[i].v === v; b.classList.toggle("on", on); b.setAttribute("aria-pressed", on); });
    if (fire && o.onChange) o.onChange(v);
  };
  const wrap = ML.el("div", { class: "ctrl" }, [o.label ? ML.el("div", { class: "ctrl-row" }, [ML.el("label", { html: o.label })]) : null, seg, o.note ? ML.el("div", { class: "ctrl-note", html: o.note }) : null]);
  seg.mlSet = (v) => set(v, true);
  parent.append(wrap);
  return { el: wrap, get: () => value, set };
};
ML.ui.check = (parent, o) => {
  const id = o.id || ML.ui.id("k");
  const input = ML.el("input", { type: "checkbox", id });
  input.checked = !!o.checked;
  input.addEventListener("change", () => o.onChange && o.onChange(input.checked));
  const lab = ML.el("label", { class: "check", for: id }, [input, ML.el("span", { html: o.label })]);
  parent.append(lab);
  return { el: lab, input, get: () => input.checked, set: (v) => { input.checked = v; } };
};
ML.ui.button = (parent, text, onClick, ghost = false) => {
  const b = ML.el("button", { type: "button", class: "btn" + (ghost ? " ghost" : ""), text, onclick: onClick });
  parent.append(b);
  return b;
};
ML.ui.row = (parent) => { const r = ML.el("div", { class: "btn-row" }); parent.append(r); return r; };
ML.ui.heading = (parent, t) => parent.append(ML.el("h4", { text: t }));
ML.ui.stats = (parent, rows) => {
  const box = ML.el("div", { class: "readout" }), refs = {};
  rows.forEach((r) => {
    const b = ML.el("b", { text: "–" });
    refs[r.key] = b;
    box.append(ML.el("div", { class: "stat" + (r.big ? " big" : "") }, [ML.el("span", { html: r.label }), b]));
  });
  parent.append(box);
  return { el: box, set: (key, v, color) => { if (refs[key]) { refs[key].innerHTML = v; refs[key].style.color = color || ""; } } };
};
ML.ui.bars = (parent, items) => {
  const box = ML.el("div", { class: "bars" }), refs = [];
  items.forEach((it) => {
    const fill = ML.el("div", { class: "bar-fill" });
    const v = ML.el("span", { class: "v", text: "0" });
    const lab = ML.el("span", { html: it.label });
    box.append(ML.el("div", { class: "bar-row" }, [lab, ML.el("div", { class: "bar-track" }, [fill]), v]));
    refs.push({ fill, v, lab, color: it.color });
  });
  parent.append(box);
  return {
    el: box,
    set: (vals, fmt = (x) => x.toFixed(2), max = 1) => vals.forEach((val, i) => {
      const r = refs[i]; if (!r) return;
      r.fill.style.width = `${ML.clamp((val / max) * 100, 0, 100)}%`;
      r.fill.style.background = typeof r.color === "function" ? r.color() : r.color;
      r.v.textContent = fmt(val);
    }),
    label: (i, html) => { if (refs[i]) refs[i].lab.innerHTML = html; },
  };
};
ML.ui.verdict = (parent) => { const v = ML.el("div", { class: "verdict" }); parent.append(v); return v; };

/* ---------------- Three.js stage ---------------- */
// Data coordinates (x, y, z) with z pointing up map to Three.js (x, z, -y).
ML.P = (x, y, z = 0) => new THREE.Vector3(x, z, -y);

ML.Stage3D = class {
  constructor(container, o = {}) {
    this.container = container;
    this.frameFns = [];
    this.disposed = false;
    this.ok = true;
    const col = ML.colors();
    try {
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: false });
    } catch (e) {
      this.ok = false;
      container.append(ML.el("div", { class: "webgl-fail", text: "3D view needs WebGL, which this browser has turned off. The controls and numbers on the right still work." }));
      return;
    }
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.append(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(col.stage);
    this.baseFov = o.fov || 40;
    this.camera = new THREE.PerspectiveCamera(this.baseFov, 1, 0.05, 500);
    const cp = o.cam || [6, 5, 7];
    this.camera.position.set(cp[0], cp[1], cp[2]);
    this.target = new THREE.Vector3(...(o.target || [0, 0, 0]));
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.target);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.09;
    this.controls.rotateSpeed = 0.8;
    this.controls.minDistance = o.minDist || 2;
    this.controls.maxDistance = o.maxDist || 60;
    if (o.noPan) this.controls.enablePan = false;
    this.controls.update();
    this.scene.add(new THREE.HemisphereLight(0xffffff, ML.isDark() ? 0x334455 : 0x99a4ad, 0.85));
    const dl = new THREE.DirectionalLight(0xffffff, 0.55);
    dl.position.set(5, 10, 6);
    this.scene.add(dl);
    this.autoRotate = !!o.autoRotate;
    this.controls.autoRotate = this.autoRotate && !ML.reducedMotion();
    this.controls.autoRotateSpeed = 0.6;
    this.renderer.domElement.addEventListener("pointerdown", () => { this.controls.autoRotate = false; });
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(container);
    this.visible = true;
    this.io = new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; });
    this.io.observe(container);
    this.resize();
    this.anim = null;
    const loop = (t) => {
      if (this.disposed) return;
      this.raf = requestAnimationFrame(loop);
      if (!this.visible) return;
      if (this.anim) this.stepAnim(t);
      this.controls.update();
      for (const f of this.frameFns) f(t);
      this.renderer.render(this.scene, this.camera);
    };
    this.raf = requestAnimationFrame(loop);
  }
  resize() {
    if (!this.ok) return;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Narrow (phone) stages: widen the field of view so the whole scene stays in frame.
    this.camera.fov = w / h < 1.2 ? Math.min(75, this.baseFov * Math.pow(1.2 / (w / h), 0.75)) : this.baseFov;
    this.camera.updateProjectionMatrix();
    // Optional: keep a minimum horizontal half-width in view by moving the camera back.
    if (this.minHalfWidth) {
      const hfov = 2 * Math.atan(Math.tan((this.camera.fov * Math.PI) / 360) * this.camera.aspect);
      const need = this.minHalfWidth / Math.tan(hfov / 2);
      const dir = this.camera.position.clone().sub(this.controls.target);
      if (dir.length() < need) this.camera.position.copy(this.controls.target).add(dir.setLength(need));
    }
  }
  onFrame(f) { this.frameFns.push(f); }
  add(o) { this.scene.add(o); return o; }
  // Smooth camera move to a preset view.
  flyTo(pos, target, ms = 900) {
    const from = this.camera.position.clone(), to = new THREE.Vector3(...pos);
    const tf = this.controls.target.clone(), tt = target ? new THREE.Vector3(...target) : tf.clone();
    if (ML.reducedMotion()) { this.camera.position.copy(to); this.controls.target.copy(tt); return; }
    this.anim = { from, to, tf, tt, t0: null, ms };
    this.controls.autoRotate = false;
  }
  stepAnim(t) {
    const a = this.anim;
    if (a.t0 === null) a.t0 = t;
    const k = ML.ease(ML.clamp((t - a.t0) / a.ms, 0, 1));
    this.camera.position.lerpVectors(a.from, a.to, k);
    this.controls.target.lerpVectors(a.tf, a.tt, k);
    if (k >= 1) this.anim = null;
  }
  views(list) {
    const box = ML.el("div", { class: "stage-views" });
    list.forEach((v) => box.append(ML.el("button", { type: "button", text: v.t, onclick: () => this.flyTo(v.pos, v.target) })));
    this.container.append(box);
  }
  // Pointer position in normalized device coordinates.
  ndc(ev) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
  }
  raycast(ev, objects) {
    const rc = new THREE.Raycaster();
    rc.params.Points = { threshold: 0.1 };
    rc.setFromCamera(this.ndc(ev), this.camera);
    return rc.intersectObjects(objects, true);
  }
  // Intersect pointer ray with a horizontal plane (Three.js y = h).
  planeHit(ev, h = 0) {
    const rc = new THREE.Raycaster();
    rc.setFromCamera(this.ndc(ev), this.camera);
    const p = new THREE.Vector3();
    return rc.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -h), p) ? p : null;
  }
  clearGroup(g) {
    while (g.children.length) {
      const c = g.children.pop();
      ML.disposeObj(c);
    }
  }
  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.raf);
    if (!this.ok) return;
    this.ro.disconnect();
    this.io.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => ML.disposeObj(o, true));
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.renderer.domElement.remove();
  }
};
ML.disposeObj = (o, shallow = false) => {
  const go = (x) => {
    if (x.geometry) x.geometry.dispose();
    if (x.material) [].concat(x.material).forEach((m) => { if (m.map) m.map.dispose(); m.dispose(); });
  };
  if (shallow) go(o); else o.traverse(go);
};

// Text label as a camera-facing sprite. `h` is the world-space height.
ML.label = (text, o = {}) => {
  const col = ML.colors();
  const fs = 64, pad = 18;
  const c = document.createElement("canvas"), ctx = c.getContext("2d");
  const font = `${o.weight || 600} ${fs}px ${o.mono ? '"IBM Plex Mono", monospace' : '"IBM Plex Sans", sans-serif'}`;
  ctx.font = font;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2, h = fs + pad * 1.4;
  c.width = w; c.height = h;
  ctx.font = font;
  if (o.bg) {
    ctx.fillStyle = o.bg;
    const r = 18;
    ctx.beginPath(); ctx.moveTo(r, 0); ctx.arcTo(w, 0, w, h, r); ctx.arcTo(w, h, 0, h, r); ctx.arcTo(0, h, 0, 0, r); ctx.arcTo(0, 0, w, 0, r); ctx.fill();
  }
  ctx.fillStyle = o.color || col.ink;
  ctx.textBaseline = "middle";
  ctx.fillText(text, pad, h / 2 + 3);
  const tex = new THREE.CanvasTexture(c);
  tex.anisotropy = 4;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: o.depthTest !== undefined ? o.depthTest : true, depthWrite: false });
  const s = new THREE.Sprite(mat);
  const H = o.h || 0.28;
  s.scale.set((H * w) / h, H, 1);
  if (o.center === "left") s.center.set(0, 0.5);
  if (o.order) s.renderOrder = o.order;
  return s;
};
ML.line = (pts, color, o = {}) => {
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = o.dashed
    ? new THREE.LineDashedMaterial({ color, dashSize: o.dash || 0.12, gapSize: o.gap || 0.08, transparent: true, opacity: o.opacity ?? 1 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity: o.opacity ?? 1 });
  const l = new THREE.Line(g, m);
  if (o.dashed) l.computeLineDistances();
  return l;
};
ML.segments = (pairs, color, opacity = 1) => {
  const pos = [];
  pairs.forEach(([a, b]) => pos.push(a.x, a.y, a.z, b.x, b.y, b.z));
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity }));
};
// Box frame with floor grid and labeled axes. Ranges are in data coordinates.
ML.axes = (stage, o) => {
  const col = ML.colors();
  const g = new THREE.Group();
  const [x0, x1] = o.x, [y0, y1] = o.y, [z0, z1] = o.z || [0, 0];
  const gridCol = new THREE.Color(col.line);
  const div = o.div || 8;
  const lines = [];
  for (let i = 0; i <= div; i++) {
    const tx = x0 + ((x1 - x0) * i) / div, ty = y0 + ((y1 - y0) * i) / div;
    lines.push([ML.P(tx, y0, z0), ML.P(tx, y1, z0)], [ML.P(x0, ty, z0), ML.P(x1, ty, z0)]);
  }
  g.add(ML.segments(lines, gridCol, 0.9));
  const axCol = new THREE.Color(col.muted);
  const ax = [[ML.P(x0, y0, z0), ML.P(x1, y0, z0)], [ML.P(x0, y0, z0), ML.P(x0, y1, z0)]];
  if (o.z) ax.push([ML.P(x0, y1, z0), ML.P(x0, y1, z1)]);
  g.add(ML.segments(ax, axCol, 1));
  const lh = o.labelH || 0.26;
  if (o.labels) {
    const [lx, ly, lz] = o.labels;
    if (lx) { const s = ML.label(lx, { h: lh, color: col.muted }); s.position.copy(ML.P((x0 + x1) / 2, y0 - (y1 - y0) * 0.09, z0)); g.add(s); }
    if (ly) { const s = ML.label(ly, { h: lh, color: col.muted }); s.position.copy(ML.P(x0 - (x1 - x0) * 0.1, (y0 + y1) / 2, z0)); g.add(s); }
    if (lz && o.z) { const s = ML.label(lz, { h: lh, color: col.muted }); s.position.copy(ML.P(x0, y1, z1 + (z1 - z0) * 0.08)); g.add(s); }
  }
  if (o.ticks) {
    const tk = (v) => (Math.abs(v) < 1e-9 ? "0" : +v.toFixed(2) + "");
    const th = lh * 0.75;
    for (const v of o.ticks.x || []) { const s = ML.label(tk(v), { h: th, color: col.muted, weight: 500, mono: true }); s.position.copy(ML.P(v, y0 - (y1 - y0) * 0.035, z0)); g.add(s); }
    for (const v of o.ticks.y || []) { const s = ML.label(tk(v), { h: th, color: col.muted, weight: 500, mono: true }); s.position.copy(ML.P(x0 - (x1 - x0) * 0.04, v, z0)); g.add(s); }
    for (const v of o.ticks.z || []) { const s = ML.label(tk(v), { h: th, color: col.muted, weight: 500, mono: true }); s.position.copy(ML.P(x0 - (x1 - x0) * 0.03, y1, v)); g.add(s); }
  }
  stage.add(g);
  return g;
};
// Many spheres in one draw call. Positions are Three.js vectors.
ML.Spheres = class {
  constructor(stage, count, radius = 0.07, o = {}) {
    this.geo = new THREE.SphereGeometry(1, o.seg || 16, o.seg ? Math.ceil(o.seg * 0.75) : 12);
    this.mat = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.05, transparent: !!o.opacity, opacity: o.opacity ?? 1 });
    this.mesh = new THREE.InstancedMesh(this.geo, this.mat, Math.max(1, count));
    this.mesh.count = count;
    this.r = radius;
    this.dummy = new THREE.Object3D();
    this.color = new THREE.Color();
    this.pos = [];
    stage.add(this.mesh);
  }
  set(i, pos, color, scale = 1) {
    this.pos[i] = pos.clone();
    this.dummy.position.copy(pos);
    this.dummy.scale.setScalar(this.r * scale);
    this.dummy.updateMatrix();
    this.mesh.setMatrixAt(i, this.dummy.matrix);
    if (color !== undefined && color !== null) { this.color.set(color); this.mesh.setColorAt(i, this.color); }
  }
  commit() {
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }
  setCount(n) { this.mesh.count = n; }
};
// A colored image (from a 2D function) laid on a horizontal plane.
ML.texturePlane = (w, h, res, fill) => {
  const c = document.createElement("canvas");
  c.width = res; c.height = res;
  const tex = new THREE.CanvasTexture(c);
  const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
  mesh.rotation.x = -Math.PI / 2;
  const redraw = (f) => {
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(res, res);
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
      const [r, g, b, a] = f((i + 0.5) / res, 1 - (j + 0.5) / res);
      const k = (j * res + i) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = a;
    }
    ctx.putImageData(img, 0, 0);
    tex.needsUpdate = true;
  };
  if (fill) redraw(fill);
  return { mesh, redraw, canvas: c };
};
ML.rgb = (css) => { const c = new THREE.Color(css); return [Math.round(c.r * 255), Math.round(c.g * 255), Math.round(c.b * 255)]; };
ML.tooltip = (container) => {
  const t = ML.el("div", { class: "tooltip" });
  t.hidden = true;
  container.append(t);
  return {
    show: (x, y, html) => {
      t.innerHTML = html; t.hidden = false;
      const W = container.clientWidth;
      t.style.left = `${Math.min(x + 14, W - t.offsetWidth - 6)}px`;
      t.style.top = `${Math.max(6, y - 10 - t.offsetHeight)}px`;
    },
    hide: () => { t.hidden = true; },
  };
};
ML.legend = (container, items) => {
  const l = ML.el("div", { class: "legend" });
  items.forEach((it) => l.append(ML.el("span", {}, [ML.el("i", { style: `background:${it.color}${it.ring ? `;border:2px solid ${it.ring};background:transparent` : ""}` }), it.label])));
  container.append(l);
  return l;
};

/* ---------------- 2D plot canvas ---------------- */
ML.Plot2D = class {
  constructor(container, o = {}) {
    this.o = Object.assign({ height: 260, pad: [16, 16, 34, 46], xlim: [0, 1], ylim: [0, 1] }, o);
    this.box = ML.el("div", { class: "canvas-box" });
    this.canvas = ML.el("canvas", { role: "img", "aria-label": o.aria || "chart" });
    this.box.append(this.canvas);
    container.append(this.box);
    this.ctx = this.canvas.getContext("2d");
    this.drawFn = null;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(this.box);
    this.resize();
  }
  resize() {
    const w = this.box.clientWidth || 300, h = this.o.height;
    const d = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = w * d; this.canvas.height = h * d;
    this.canvas.style.height = h + "px";
    this.ctx.setTransform(d, 0, 0, d, 0, 0);
    this.W = w; this.H = h;
    if (this.drawFn) this.drawFn();
  }
  get iw() { return this.W - this.o.pad[1] - this.o.pad[3]; }
  get ih() { return this.H - this.o.pad[0] - this.o.pad[2]; }
  X(v) { const [a, b] = this.o.xlim; return this.o.pad[3] + ((v - a) / (b - a)) * this.iw; }
  Y(v) { const [a, b] = this.o.ylim; return this.o.pad[0] + (1 - (v - a) / (b - a)) * this.ih; }
  invX(px) { const [a, b] = this.o.xlim; return a + ((px - this.o.pad[3]) / this.iw) * (b - a); }
  invY(py) { const [a, b] = this.o.ylim; return a + (1 - (py - this.o.pad[0]) / this.ih) * (b - a); }
  clear() { const c = ML.colors(); this.ctx.fillStyle = c.surface; this.ctx.fillRect(0, 0, this.W, this.H); }
  frame({ xticks = [], yticks = [], xlabel = "", ylabel = "", xfmt = (v) => v, yfmt = (v) => v, grid = true } = {}) {
    const c = ML.colors(), ctx = this.ctx;
    ctx.save();
    ctx.font = '11.5px "IBM Plex Mono", monospace';
    ctx.fillStyle = c.muted;
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of xticks) {
      const x = this.X(t);
      if (grid) { ctx.beginPath(); ctx.moveTo(x, this.o.pad[0]); ctx.lineTo(x, this.H - this.o.pad[2]); ctx.stroke(); }
      ctx.fillText(xfmt(t), x, this.H - this.o.pad[2] + 5);
    }
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (const t of yticks) {
      const y = this.Y(t);
      if (grid) { ctx.beginPath(); ctx.moveTo(this.o.pad[3], y); ctx.lineTo(this.W - this.o.pad[1], y); ctx.stroke(); }
      ctx.fillText(yfmt(t), this.o.pad[3] - 6, y);
    }
    ctx.font = '500 12px "IBM Plex Sans", sans-serif';
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    if (xlabel) ctx.fillText(xlabel, this.o.pad[3] + this.iw / 2, this.H - 1);
    if (ylabel) { ctx.translate(12, this.o.pad[0] + this.ih / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = "middle"; ctx.fillText(ylabel, 0, 0); }
    ctx.restore();
  }
  dot(x, y, r, fill, stroke, lw = 1.5) {
    const ctx = this.ctx;
    ctx.beginPath(); ctx.arc(this.X(x), this.Y(y), r, 0, Math.PI * 2);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
  }
  path(pts, stroke, lw = 2, dash = null) {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    let first = true;
    for (const [x, y] of pts) {
      if (!Number.isFinite(y)) { first = true; continue; }
      if (first) { ctx.moveTo(this.X(x), this.Y(y)); first = false; } else ctx.lineTo(this.X(x), this.Y(y));
    }
    ctx.strokeStyle = stroke; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash);
    ctx.stroke();
    ctx.restore();
  }
  text(s, x, y, color, align = "left", font = '500 12px "IBM Plex Sans", sans-serif') {
    const ctx = this.ctx;
    ctx.save(); ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = "middle";
    ctx.fillText(s, x, y); ctx.restore();
  }
  clip(fn) {
    const ctx = this.ctx;
    ctx.save(); ctx.beginPath(); ctx.rect(this.o.pad[3], this.o.pad[0], this.iw, this.ih); ctx.clip(); fn(); ctx.restore();
  }
  dispose() { this.ro.disconnect(); }
};
ML.ticks = (a, b, n = 5) => {
  const span = b - a, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
};

/* ---------------- Text decoration ---------------- */
ML.highlight = (src) => {
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const KW = new Set("from import as def return for in if else elif True False None and or not lambda with".split(" "));
  const re = /(#[^\n]*)|("(?:[^"\\\n]|\\.)*"|'(?:[^'\\\n]|\\.)*')|\b(\d+\.?\d*(?:e-?\d+)?)\b|\b([A-Za-z_]\w*)(?=\()|\b([A-Za-z_]\w*)\b/g;
  let out = "", last = 0, m;
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    if (m[1]) out += `<span class="tok-c">${esc(m[1])}</span>`;
    else if (m[2]) out += `<span class="tok-s">${esc(m[2])}</span>`;
    else if (m[3]) out += `<span class="tok-n">${m[3]}</span>`;
    else if (m[4]) out += KW.has(m[4]) ? `<span class="tok-k">${m[4]}</span>` : `<span class="tok-f">${m[4]}</span>`;
    else out += KW.has(m[5]) ? `<span class="tok-k">${m[5]}</span>` : m[5];
    last = re.lastIndex;
  }
  return out + esc(src.slice(last));
};
ML.decorate = (root) => {
  root.querySelectorAll("pre.py").forEach((pre) => {
    const raw = pre.textContent.replace(/^\n+|\s+$/g, "");
    pre.dataset.raw = raw;
    pre.innerHTML = ML.highlight(raw);
  });
  root.querySelectorAll(".code .copy").forEach((b) => {
    b.addEventListener("click", () => {
      const pre = b.closest(".code").querySelector("pre.py");
      const done = () => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400); };
      if (navigator.clipboard) navigator.clipboard.writeText(pre.dataset.raw).then(done, () => window.getSelection().selectAllChildren(pre));
      else window.getSelection().selectAllChildren(pre);
    });
  });
};
// MathJax must not run two typeset passes at once, so calls are chained.
let mjChain = Promise.resolve();
ML.typeset = (root) => {
  if (!(window.MathJax && MathJax.typesetPromise)) return Promise.resolve();
  mjChain = mjChain.then(() => (root.isConnected ? MathJax.typesetPromise([root]) : null)).catch(() => {});
  return mjChain;
};

/* ---------------- Chapter markup helpers ---------------- */
ML.esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const CHIP = { cls: ["task-cls", "Classification"], reg: ["task-reg", "Regression"], unsup: ["task-unsup", "Unsupervised"], sup: ["task-cls", "Supervised"], sel: ["", "Model selection"] };
ML.chHead = (meta, idea, chips = []) => `
  <header class="ch-head">
    <div class="eyebrow"><span>Section ${meta.n}</span><span class="pdfref">${meta.pdf}</span></div>
    <h1>${meta.title}</h1>
    <p class="idea">${idea}</p>
    <div class="chips">${chips.map((c) => (CHIP[c] ? `<span class="chip ${CHIP[c][0]}">${CHIP[c][1]}</span>` : `<span class="chip">${c}</span>`)).join("")}</div>
  </header>`;
ML.code = (src, out, title = "python") => `
  <div class="code">
    <div class="code-head"><span>${title}</span><button class="copy" type="button">Copy</button></div>
    <pre class="py">${ML.esc(src.replace(/^\n/, ""))}</pre>
    ${out ? `<div class="out"><div class="code-head"><span>Output (scikit-learn 1.9)</span></div><pre>${ML.esc(out.replace(/^\n/, ""))}</pre></div>` : ""}
  </div>`;
ML.bench = (id, o = {}) => `
  <div class="bench ${o.cls || ""}">
    <div class="stage ${o.stage || ""}" id="${id}-stage">
      <span class="stage-tag">${o.tag || "3D view"}</span>
      ${o.help === false ? "" : `<span class="stage-help">${o.help || "Drag to rotate, scroll to zoom"}</span>`}
    </div>
    <div class="panel" id="${id}-panel"></div>
  </div>`;
ML.callout = (kind, title, body) => `<div class="callout ${kind}"><div class="c-title">${title}</div><div>${body}</div></div>`;
ML.formula = (label, tex, note = "", right = "") => `
  <div class="formula">
    <div class="f-label"><span>${label}</span><span>${right}</span></div>
    <div class="f-body">\\[${tex}\\]</div>
    ${note ? `<div class="f-note">${note}</div>` : ""}
  </div>`;


/* ---------------- Step-by-step tours ---------------- */
// Drive a control panel by visible labels (used by tours).
ML.drive = (panel) => {
  const ctrlBy = (label) => [...panel.querySelectorAll(".ctrl")].find((c) => {
    const l = c.querySelector(".ctrl-row label");
    return l && l.textContent.trim().toLowerCase().startsWith(label.toLowerCase());
  });
  return {
    slider(label, v) { const c = ctrlBy(label); const i = c && c.querySelector('input[type="range"]'); if (i) i.mlSet(v); },
    seg(text) { const b = [...panel.querySelectorAll(".seg button")].find((x) => x.textContent.trim() === text); if (b && !b.classList.contains("on")) b.click(); },
    btn(text) { const b = [...panel.querySelectorAll("button")].find((x) => x.textContent.trim().startsWith(text)); if (b) b.click(); },
    check(label, v) { const l = [...panel.querySelectorAll("label.check")].find((x) => x.textContent.trim().startsWith(label)); const i = l && l.querySelector("input"); if (i && i.checked !== v) { i.checked = v; i.dispatchEvent(new Event("change")); } },
  };
};
// Thick arrow (cylinder shaft + cone head) drawn on top of the scene.
ML.arrow = (from, to, color, head = 0.18, radius = 0.025) => {
  const dir = to.clone().sub(from), len = dir.length();
  const g = new THREE.Group();
  const mat = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
  const hl = Math.min(head, len * 0.4);
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, Math.max(len - hl, 0.001), 12), mat);
  shaft.position.y = (len - hl) / 2;
  const cone = new THREE.Mesh(new THREE.ConeGeometry(radius * 3, hl, 16), mat);
  cone.position.y = len - hl / 2;
  shaft.renderOrder = cone.renderOrder = 12;
  g.add(shaft, cone);
  g.position.copy(from);
  g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  return g;
};
// A guided sequence of steps for one 3D stage. Each step: { t, d, cam: [pos, target], run(ann, stage) }.
ML.tour = (bench, stage, steps, o = {}) => {
  const col = ML.colors();
  const box = ML.el("div", { class: "tour" });
  bench.parentNode.insertBefore(box, bench);
  const ann = new THREE.Group();
  ann.renderOrder = 10;
  if (stage && stage.ok) stage.add(ann);
  let idx = -1, timer = null, playing = false, stepCleanup = null;
  const head = ML.el("div", { class: "tour-head" });
  const text = ML.el("div", { class: "tour-text" });
  const dots = ML.el("div", { class: "tour-dots" });
  const prev = ML.el("button", { type: "button", class: "btn ghost", text: "Back" });
  const next = ML.el("button", { type: "button", class: "btn", text: "Start step by step" });
  const play = ML.el("button", { type: "button", class: "btn ghost", text: "Auto-play" });
  const exit = ML.el("button", { type: "button", class: "btn ghost", text: "Explore freely" });
  const ctr = ML.el("div", { class: "btn-row tour-ctrl" }, [prev, next, play, exit]);
  box.append(head, text, ctr, dots);
  steps.forEach((s, i) => dots.append(ML.el("button", { type: "button", class: "tour-dot", "aria-label": `Step ${i + 1}: ${s.t}`, title: s.t, onclick: () => go(i) })));
  const intro = () => {
    head.innerHTML = `<span class="tour-badge">Step by step</span><b>${o.title || "Guided tour"}</b><span class="tour-count">${steps.length} steps</span>`;
    text.innerHTML = `<ol class="tour-list">${steps.map((s) => `<li>${s.t}</li>`).join("")}</ol>`;
    prev.hidden = true; play.hidden = false; exit.hidden = true;
    next.textContent = "Start step by step";
    dots.querySelectorAll(".tour-dot").forEach((d) => d.classList.remove("on", "seen"));
  };
  function clearAnn() {
    if (stepCleanup) { try { stepCleanup(); } catch (e) { console.error(e); } stepCleanup = null; }
    if (stage && stage.ok) stage.clearGroup(ann);
  }
  function go(i) {
    if (i < 0 || i >= steps.length) return;
    clearAnn();
    idx = i;
    const s = steps[i];
    head.innerHTML = `<span class="tour-badge">Step ${i + 1} / ${steps.length}</span><b>${s.t}</b>`;
    try { stepCleanup = s.run ? s.run(ann, stage) || null : null; } catch (e) { console.error(e); }
    text.innerHTML = typeof s.d === "function" ? s.d() : s.d;
    ML.typesetLater && ML.typesetLater(text);
    prev.hidden = false; prev.disabled = i === 0; exit.hidden = false;
    next.textContent = i === steps.length - 1 ? "Finish" : "Next step";
    dots.querySelectorAll(".tour-dot").forEach((d, k) => { d.classList.toggle("on", k === i); d.classList.toggle("seen", k < i); });
    const cam = stage && stage.ok ? (typeof s.cam === "function" ? s.cam() : s.cam) : null;
    if (cam && stage && stage.ok) stage.flyTo(cam[0], cam[1], s.camMs || 1200);
    const r = bench.getBoundingClientRect();
    if (o.scroll !== false && (r.top < 0 || r.top > window.innerHeight * 0.5)) box.scrollIntoView({ block: "start", behavior: ML.reducedMotion() ? "auto" : "smooth" });
  }
  const stop = () => { playing = false; clearTimeout(timer); play.textContent = "Auto-play"; };
  const schedule = () => {
    clearTimeout(timer);
    if (!playing) return;
    const ms = (steps[idx] && steps[idx].ms) || 7000;
    timer = setTimeout(() => { if (idx < steps.length - 1) { go(idx + 1); schedule(); } else stop(); }, ms);
  };
  next.addEventListener("click", () => { if (idx === steps.length - 1) { stop(); clearAnn(); idx = -1; intro(); return; } go(idx + 1); schedule(); });
  prev.addEventListener("click", () => { go(idx - 1); schedule(); });
  play.addEventListener("click", () => {
    if (playing) return stop();
    playing = true; play.textContent = "Pause";
    if (idx < 0 || idx === steps.length - 1) go(0);
    schedule();
  });
  exit.addEventListener("click", () => { stop(); clearAnn(); idx = -1; intro(); });
  intro();
  return { go, dispose: () => { stop(); clearAnn(); } };
};
// Small helper for annotation labels in data coordinates.
ML.tag = (ann, text, pos, color, h = 0.24) => {
  const s = ML.label(text, { h, color: color || ML.colors().ink, bg: ML.colors().surface, depthTest: false, order: 20 });
  s.position.copy(pos);
  ann.add(s);
  return s;
};
ML.ring = (ann, pos, r, color, flat = true) => {
  const m = new THREE.Mesh(new THREE.TorusGeometry(r, r * 0.08, 8, 40), new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true }));
  if (flat) m.rotation.x = -Math.PI / 2;
  m.position.copy(pos);
  m.renderOrder = 15;
  ann.add(m);
  return m;
};
