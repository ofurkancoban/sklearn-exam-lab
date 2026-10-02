// Section 9: multilayer perceptron. A ReLU neuron as a hinged plane, and a real
// MLP (ReLU hidden layers, softmax output, Adam) trained on Iris in the browser.
ML.MLP = class {
  constructor(sizes, seed = 42) {
    const r = ML.rng(seed);
    this.sizes = sizes;
    this.W = []; this.b = [];
    for (let l = 1; l < sizes.length; l++) {
      const s = Math.sqrt(2 / sizes[l - 1]);
      this.W.push(Array.from({ length: sizes[l] }, () => Array.from({ length: sizes[l - 1] }, () => ML.randn(r) * s)));
      this.b.push(new Array(sizes[l]).fill(0));
    }
    this.m = this.zeros(); this.v = this.zeros(); this.t = 0;
  }
  zeros() { return { W: this.W.map((M) => M.map((row) => row.map(() => 0))), b: this.b.map((v) => v.map(() => 0)) }; }
  forward(x) {
    const acts = [x];
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const z = this.W[l].map((row, i) => row.reduce((s, w, j) => s + w * a[j], this.b[l][i]));
      if (l < this.W.length - 1) a = z.map((v) => Math.max(0, v));
      else { const mx = Math.max(...z), e = z.map((v) => Math.exp(v - mx)), s = e.reduce((p, q) => p + q, 0); a = e.map((v) => v / s); }
      acts.push(a);
    }
    return acts;
  }
  loss(X, y) { return ML.mean(X.map((x, i) => -Math.log(Math.max(this.forward(x).at(-1)[y[i]], 1e-12)))); }
  acc(X, y) { return X.filter((x, i) => { const p = this.forward(x).at(-1); return p.indexOf(Math.max(...p)) === y[i]; }).length / X.length; }
  step(X, y, lr = 0.01) {
    const g = this.zeros(), L = this.W.length;
    X.forEach((x, n) => {
      const acts = this.forward(x);
      let delta = acts[L].map((p, k) => p - (k === y[n] ? 1 : 0));
      for (let l = L - 1; l >= 0; l--) {
        const aPrev = acts[l];
        for (let i = 0; i < delta.length; i++) { g.b[l][i] += delta[i]; for (let j = 0; j < aPrev.length; j++) g.W[l][i][j] += delta[i] * aPrev[j]; }
        if (l > 0) delta = aPrev.map((av, j) => (av > 0 ? delta.reduce((s, d, i) => s + d * this.W[l][i][j], 0) : 0));
      }
    });
    this.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, N = X.length;
    const upd = (P, G, M, V) => {
      const gr = G / N;
      M = b1 * M + (1 - b1) * gr; V = b2 * V + (1 - b2) * gr * gr;
      const mh = M / (1 - Math.pow(b1, this.t)), vh = V / (1 - Math.pow(b2, this.t));
      return [P - (lr * mh) / (Math.sqrt(vh) + eps), M, V];
    };
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < this.W[l].length; i++) {
        for (let j = 0; j < this.W[l][i].length; j++) [this.W[l][i][j], this.m.W[l][i][j], this.v.W[l][i][j]] = upd(this.W[l][i][j], g.W[l][i][j], this.m.W[l][i][j], this.v.W[l][i][j]);
        [this.b[l][i], this.m.b[l][i], this.v.b[l][i]] = upd(this.b[l][i], g.b[l][i], this.m.b[l][i], this.v.b[l][i]);
      }
    }
  }
};

ML.register("mlp", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Pass features through <b>layers of neurons</b>. Hidden layers apply nonlinear transformations, allowing complex patterns. <b>Training adjusts weights to reduce prediction error.</b>", ["sup", "cls", "hidden_layer_sizes", "max_iter"])}
  ${ML.formula("One hidden neuron (ReLU)", String.raw`h = \max\Big(0,\; b + \sum_{j=1}^{p} w_j x_j\Big)`, "The neuron forms a weighted sum plus an offset b, then <b>replaces negative results by zero</b>. This ReLU activation lets the network learn nonlinear patterns. A network groups neurons into layers; each layer passes its outputs to the next.", "Eq. 13")}

  <div class="section-title"><h2>A ReLU neuron is a folded plane</h2><span class="hint">Height = neuron output h for inputs (x1, x2).</span></div>
  ${ML.bench("re", { tag: "h = max(0, b + w1·x1 + w2·x2)", stage: "" })}
  <p class="fig-cap">Where the weighted sum is negative, the output is flat at zero; elsewhere it is a tilted plane. The fold sits on the line b + w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> = 0 (dashed on the floor). One neuron alone is still very simple. Adding several folded planes (the next layer does exactly this, with its own weights) produces ridges, valleys and bumps: shapes no single linear model can make.</p>

  <div class="section-title"><h2>Train a network on Iris</h2><span class="hint">4 inputs → hidden layers → 3 outputs, trained live with backpropagation.</span></div>
  ${ML.bench("nn", { tag: "MLPClassifier architecture", stage: "tall", cls: "wide-panel" })}
  <p class="fig-cap">Lines are weights (teal positive, orange negative; stronger color = larger magnitude). Node brightness is the activation for the selected test flower. The output layer shows the predicted probabilities for the three species. Training uses 96 flowers, 24 are held out as validation for the loss curve, and the 30 test flowers stay untouched until the end.</p>
  <div id="nn-curve"></div>

  <div class="section-title"><h2>In scikit-learn</h2></div>
  <div class="two-col">
    ${ML.code(`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(20,),
        max_iter=2000,
        random_state=42
    )
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("MLP accuracy:", accuracy_score(y_test, predictions))`, `MLP accuracy: 0.9666666666666667`)}
    <div class="prose">
      <h3>Layer sizes and training time</h3>
      <ul>
        <li><code>hidden_layer_sizes=(20,)</code>: one hidden layer with 20 neurons. <b>The comma makes this a one-element Python tuple.</b></li>
        <li><code>hidden_layer_sizes=(50, 20)</code>: two hidden layers, with 50 and 20 neurons. A larger network can learn more complex patterns but needs more computation and may overfit.</li>
        <li><code>max_iter</code> limits training iterations. If training does not converge, check scaling and consider increasing it; more iterations do not guarantee better predictions.</li>
      </ul>
      <p><code>MLPClassifier</code> predicts categories. The related <code>MLPRegressor</code> predicts numbers.</p>
      ${ML.callout("tip", "Good practice", "Scale MLP inputs and compare models using cross-validation. For <b>early stopping</b>, use validation performance; keep the final test set untouched.")}
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>(20,) is one layer of 20 neurons; (20) without the comma is just the number 20.</li><li>(50, 20) means two hidden layers, not 50 inputs and 20 outputs. Input and output sizes come from the data.</li><li>ReLU outputs max(0, ·): negative sums become 0, positive sums pass unchanged.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="mlp"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // ----- ReLU surface -----
    const rh = root.querySelector("#re-stage"), rp = root.querySelector("#re-panel");
    const rs = new ML.Stage3D(rh, { cam: [5, 4.2, 5.6], target: [0, 0.7, 0] });
    disposers.push(() => rs.dispose());
    const RS = { mode: "one", w1: 1, w2: 0.5, b: 0 };
    const PRESETS = {
      ridge: { t: "Ridge", n: [[1, 0, 1, 1], [-1, 0, 1, 1], [0, 0, 0, 0]], note: "h1 + h2 with opposite slopes, minus a constant: a roof-shaped ridge." },
      bump: { t: "Pyramid", n: [[1, 0, 1, -1], [-1, 0, 1, -1], [0, 1, 1, -1], [0, -1, 1, -1]], note: "Four neurons, each subtracting a fold, carve a pyramid-shaped bump." },
      valley: { t: "V valley", n: [[1, 1, 0, 1], [-1, -1, 0, 1]], note: "Two folds pointing in opposite directions make a V-shaped valley: |x1 + x2|." },
    };
    let preset = "ridge";
    const G = 60, HS = 0.55;
    let surf, hinge = new THREE.Group();
    if (rs.ok) {
      ML.axes(rs, { x: [-2, 2], y: [-2, 2], labels: ["x1", "x2", null], div: 8 });
      const g = new THREE.PlaneGeometry(4, 4, G, G); g.rotateX(-Math.PI / 2);
      g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array((G + 1) ** 2 * 3), 3));
      surf = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
      rs.add(surf); rs.add(hinge);
    }
    const relu = (v) => Math.max(0, v);
    const fOne = (x1, x2) => relu(RS.b + RS.w1 * x1 + RS.w2 * x2);
    const fComb = (x1, x2) => {
      const P = PRESETS[preset].n;
      if (preset === "ridge") return relu(2 - relu(x1) - relu(-x1)) * 0.9;
      if (preset === "bump") return relu(2 - relu(x1) - relu(-x1) - relu(x2) - relu(-x2)) * 1.1;
      return P.reduce((s, [a, b, , c]) => s + c * relu(a * x1 + b * x2), 0) * 0.7;
    };
    ML.ui.heading(rp, "Neuron");
    ML.ui.seg(rp, { options: [{ v: "one", t: "One neuron" }, { v: "comb", t: "Several combined" }], value: "one", onChange: (v) => { RS.mode = v; oneBox.hidden = v !== "one"; combBox.hidden = v === "one"; drawR(); } });
    const oneBox = ML.el("div", { class: "panel", style: "padding:0" }), combBox = ML.el("div", { class: "panel", style: "padding:0" });
    combBox.hidden = true;
    rp.append(oneBox, combBox);
    ML.ui.slider(oneBox, { label: "w1", min: -2, max: 2, step: 0.05, value: RS.w1, onInput: (v) => { RS.w1 = v; drawR(); } });
    ML.ui.slider(oneBox, { label: "w2", min: -2, max: 2, step: 0.05, value: RS.w2, onInput: (v) => { RS.w2 = v; drawR(); } });
    ML.ui.slider(oneBox, { label: "b (offset)", min: -2, max: 2, step: 0.05, value: RS.b, onInput: (v) => { RS.b = v; drawR(); } });
    const probe = ML.el("div", { class: "verdict" }); oneBox.append(probe);
    ML.ui.seg(combBox, { options: Object.entries(PRESETS).map(([k, v]) => ({ v: k, t: v.t })), value: preset, onChange: (v) => { preset = v; drawR(); } });
    const combNote = ML.el("div", { class: "verdict" }); combBox.append(combNote);
    function drawR() {
      const f = RS.mode === "one" ? fOne : fComb;
      const x = [1, -0.5], z = RS.b + RS.w1 * x[0] + RS.w2 * x[1];
      probe.innerHTML = `Example input x = (1, −0.5):<br>b + w·x = ${RS.b.toFixed(2)} + ${RS.w1.toFixed(2)}·1 + ${RS.w2.toFixed(2)}·(−0.5) = ${z.toFixed(2)}<br>h = max(0, ${z.toFixed(2)}) = <b>${Math.max(0, z).toFixed(2)}</b>`;
      combNote.innerHTML = PRESETS[preset].note;
      if (!rs.ok) return;
      const pos = surf.geometry.attributes.position, cl = surf.geometry.attributes.color;
      const off = new THREE.Color(col.surface2), on1 = new THREE.Color(col.c[1]), on2 = new THREE.Color(col.c[2]), tmp = new THREE.Color();
      for (let v = 0; v < pos.count; v++) {
        const a = pos.getX(v), b = -pos.getZ(v), h = Math.min(f(a, b), 4.5);
        pos.setY(v, h * HS);
        if (h <= 1e-6) tmp.copy(off); else tmp.copy(on1).lerp(on2, Math.min(h / 4, 1));
        cl.setXYZ(v, tmp.r, tmp.g, tmp.b);
      }
      pos.needsUpdate = true; cl.needsUpdate = true; surf.geometry.computeVertexNormals();
      rs.clearGroup(hinge);
      if (RS.mode === "one" && (Math.abs(RS.w1) > 1e-6 || Math.abs(RS.w2) > 1e-6)) {
        // Points on b + w1 x1 + w2 x2 = 0 clipped to the square.
        const pts = [];
        for (let t = -2; t <= 2.0001; t += 0.05) {
          if (Math.abs(RS.w2) > Math.abs(RS.w1)) pts.push([t, -(RS.b + RS.w1 * t) / RS.w2]); else pts.push([-(RS.b + RS.w2 * t) / RS.w1, t]);
        }
        const inside = pts.filter(([a, b]) => Math.abs(a) <= 2 && Math.abs(b) <= 2).map(([a, b]) => ML.P(a, b, 0.01));
        if (inside.length > 1) hinge.add(ML.line(inside, new THREE.Color(col.ink), { dashed: true }));
      }
    }
    drawR();

    // ----- Network on Iris -----
    const D = ML.data.iris, sp = ML.data.extra.irisSplit;
    const Xtr0 = sp.train.map((i) => D.X[i]), ytr0 = sp.train.map((i) => D.y[i]);
    const sc = ML.scaler(Xtr0);
    const XtrAll = sc.transform(Xtr0);
    const Xte = sc.transform(sp.test.map((i) => D.X[i])), yte = sp.test.map((i) => D.y[i]);
    const vIdx = ML.shuffleIdx(120, ML.rng(3)).slice(0, 24), vSet = new Set(vIdx);
    const Xt = XtrAll.filter((_, i) => !vSet.has(i)), yt = ytr0.filter((_, i) => !vSet.has(i));
    const Xv = vIdx.map((i) => XtrAll[i]), yv = vIdx.map((i) => ytr0[i]);
    const nh = root.querySelector("#nn-stage"), np = root.querySelector("#nn-panel");
    const ns = new ML.Stage3D(nh, { cam: [0.5, 1.6, 8.6], target: [0, 0.2, 0], maxDist: 40 });
    disposers.push(() => ns.dispose());
    const N = { arch: [20], flower: 0, epoch: 0, running: false, hist: [] };
    let net, nodes = [], nodeMesh, edges, edgeList = [];
    const curve = new ML.Plot2D(root.querySelector("#nn-curve"), { height: 230, xlim: [0, 300], ylim: [0, 1.2], aria: "Training and validation loss" });
    disposers.push(() => curve.dispose());

    ML.ui.heading(np, "Architecture");
    const ARCH = { "(5,)": [5], "(20,)": [20], "(50, 20)": [50, 20], "(8, 8, 8)": [8, 8, 8] };
    ML.ui.seg(np, { label: "hidden_layer_sizes", options: Object.keys(ARCH).map((k) => ({ v: k, t: k })), value: "(20,)", onChange: (v) => { N.arch = ARCH[v]; reset(); } });
    const row = ML.ui.row(np);
    const trainBtn = ML.ui.button(row, "Train 300 iterations", () => { if (N.epoch >= 300) reset(); N.running = !N.running; trainBtn.textContent = N.running ? "Pause" : "Continue training"; });
    ML.ui.button(row, "Reset weights", () => reset(), true);
    ML.ui.slider(np, { label: "Test flower", min: 1, max: 30, value: 1, fmt: (v) => `#${v} (${D.names[yte[v - 1]]})`, onInput: (v) => { N.flower = v - 1; paint(); } });
    ML.ui.heading(np, "Output probabilities");
    const outBars = ML.ui.bars(np, D.names.map((nm, c) => ({ label: nm, color: col.c[c] })));
    const stats = ML.ui.stats(np, [{ key: "ep", label: "Iterations" }, { key: "va", label: "Validation accuracy" }, { key: "te", label: "Test accuracy (30)", big: true }]);

    function layout() {
      const sizes = [4, ...N.arch, 3], L = sizes.length;
      nodes = sizes.map((s, l) => {
        const x = (l - (L - 1) / 2) * 3.0;
        const cols = s > 12 ? Math.ceil(s / 10) : 1, rows = Math.ceil(s / cols);
        return Array.from({ length: s }, (_, i) => {
          const c = Math.floor(i / rows), r = i % rows;
          return new THREE.Vector3(x, ((rows - 1) / 2 - r) * (s > 12 ? 0.42 : 0.62), (c - (cols - 1) / 2) * 0.5);
        });
      });
    }
    function build() {
      if (!ns.ok) return;
      if (nodeMesh) { ns.scene.remove(nodeMesh.mesh); ML.disposeObj(nodeMesh.mesh); }
      if (edges) { ns.scene.remove(edges); ML.disposeObj(edges); }
      ns.scene.children.filter((o) => o.userData.tmp).forEach((o) => { ns.scene.remove(o); ML.disposeObj(o); });
      const total = nodes.reduce((s, l) => s + l.length, 0);
      nodeMesh = new ML.Spheres(ns, total, 0.13);
      edgeList = [];
      const pos = [];
      for (let l = 0; l < nodes.length - 1; l++) nodes[l].forEach((a, j) => nodes[l + 1].forEach((b, i) => { edgeList.push([l, i, j]); pos.push(a.x, a.y, a.z, b.x, b.y, b.z); }));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(pos.length), 3));
      edges = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
      ns.add(edges);
      const names = ["sepal len", "sepal wid", "petal len", "petal wid"];
      nodes[0].forEach((p, i) => { const s = ML.label(names[i], { h: 0.24, color: col.muted }); s.position.set(p.x - 1.1, p.y, p.z); s.userData.tmp = true; ns.add(s); });
      nodes.at(-1).forEach((p, i) => { const s = ML.label(D.names[i], { h: 0.26, color: col.c[i], center: "left" }); s.position.set(p.x + 0.3, p.y, p.z); s.userData.tmp = true; ns.add(s); });
      const tags = ["input (4 features)", ...N.arch.map((n, k) => `hidden ${k + 1}: ${n}`), "output (3 classes)"];
      nodes.forEach((ly, l) => { const top = Math.max(...ly.map((p) => p.y)); const s = ML.label(tags[l], { h: 0.24, color: col.ink }); s.position.set(ly[0].x, top + 0.6, 0); s.userData.tmp = true; ns.add(s); });
    }
    function paint() {
      const acts = net.forward(Xte[N.flower]);
      const p = acts.at(-1);
      outBars.set(p, (v) => v.toFixed(2));
      if (!ns.ok) return;
      const bg = new THREE.Color(col.stage), pc = new THREE.Color(col.c[1]), nc = new THREE.Color(col.warn), on = new THREE.Color(col.accent), tmp = new THREE.Color();
      let k = 0;
      acts.forEach((a, l) => {
        const mx = Math.max(1e-6, ...a.map(Math.abs));
        a.forEach((v, i) => {
          const isOut = l === acts.length - 1;
          tmp.copy(new THREE.Color(col.surface2)).lerp(isOut ? new THREE.Color(col.c[i]) : on, ML.clamp(Math.abs(v) / mx, 0.08, 1));
          nodeMesh.set(k++, nodes[l][i], tmp, isOut ? 0.8 + p[i] * 1.2 : 1);
        });
      });
      nodeMesh.commit();
      let wmax = 1e-6;
      net.W.forEach((M) => M.forEach((r) => r.forEach((w) => (wmax = Math.max(wmax, Math.abs(w))))));
      const ca = edges.geometry.attributes.color;
      edgeList.forEach(([l, i, j], e) => {
        const w = net.W[l][i][j], t = Math.pow(Math.abs(w) / wmax, 0.8) * (nodes[1].length > 30 ? 0.7 : 1);
        tmp.copy(bg).lerp(w > 0 ? pc : nc, t);
        ca.setXYZ(2 * e, tmp.r, tmp.g, tmp.b); ca.setXYZ(2 * e + 1, tmp.r, tmp.g, tmp.b);
      });
      ca.needsUpdate = true;
    }
    function drawCurve() {
      curve.clear();
      const ymax = Math.max(1.2, ...N.hist.map((h) => Math.max(h.tr, h.va))) * 1.05;
      curve.o.ylim = [0, ymax];
      curve.frame({ xticks: [0, 50, 100, 150, 200, 250, 300], yticks: ML.ticks(0, ymax, 4), xlabel: "training iteration", ylabel: "loss (cross-entropy)" });
      if (N.hist.length > 1) {
        curve.path(N.hist.map((h) => [h.ep, h.tr]), col.c[1], 2.2);
        curve.path(N.hist.map((h) => [h.ep, h.va]), col.warn, 2.2);
      }
      curve.text("training loss", curve.W - 24, 22, col.c[1], "right");
      curve.text("validation loss", curve.W - 24, 40, col.warn, "right");
    }
    curve.drawFn = drawCurve;
    function reset() {
      N.running = false; N.epoch = 0; N.hist = [];
      trainBtn.textContent = "Train 300 iterations";
      net = new ML.MLP([4, ...N.arch, 3], 42);
      layout(); build(); paint(); record(); drawCurve();
    }
    function record() {
      N.hist.push({ ep: N.epoch, tr: net.loss(Xt, yt), va: net.loss(Xv, yv) });
      stats.set("ep", `${N.epoch} / 300`);
      stats.set("va", net.acc(Xv, yv).toFixed(3));
      stats.set("te", N.epoch >= 300 ? net.acc(Xte, yte).toFixed(3) : "after training");
    }
    let raf;
    const loop = () => {
      raf = requestAnimationFrame(loop);
      if (!N.running) return;
      // Spend about 14 ms per frame on training steps.
      const t0 = performance.now();
      while (N.epoch < 300 && performance.now() - t0 < 14) { net.step(Xt, yt, 0.01); N.epoch++; }
      record(); paint(); drawCurve();
      if (N.epoch >= 300) { N.running = false; trainBtn.textContent = "Train again from scratch"; }
    };
    raf = requestAnimationFrame(loop);
    disposers.push(() => cancelAnimationFrame(raf));
    reset();
    return disposers;
  },
});
