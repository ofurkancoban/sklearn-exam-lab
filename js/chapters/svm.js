// Section 3: support vector machines. A real SMO solver trains in the browser;
// the 3D surface is the decision function f(x) over the feature plane.
ML.svmTrain = (X, y, { C = 1, kernel = "linear", gamma = 1, tol = 1e-3, maxPasses = 400 } = {}) => {
  const n = X.length;
  const kf = kernel === "linear"
    ? (a, b) => a[0] * b[0] + a[1] * b[1]
    : (a, b) => Math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2));
  const K = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => kf(X[i], X[j])));
  const alpha = new Float64Array(n);
  let b = 0;
  const E = y.map((yi) => -yi); // f(x) - y with alpha = 0, b = 0
  const eps = 1e-6;
  const takeStep = (i1, i2) => {
    if (i1 === i2) return false;
    const a1 = alpha[i1], a2 = alpha[i2], y1 = y[i1], y2 = y[i2], E1 = E[i1], E2 = E[i2], s = y1 * y2;
    let L, H;
    if (y1 !== y2) { L = Math.max(0, a2 - a1); H = Math.min(C, C + a2 - a1); }
    else { L = Math.max(0, a1 + a2 - C); H = Math.min(C, a1 + a2); }
    if (H - L < 1e-12) return false;
    const eta = K[i1][i1] + K[i2][i2] - 2 * K[i1][i2];
    if (eta <= 1e-12) return false;
    let a2n = ML.clamp(a2 + (y2 * (E1 - E2)) / eta, L, H);
    if (Math.abs(a2n - a2) < eps * (a2n + a2 + eps)) return false;
    const a1n = a1 + s * (a2 - a2n);
    const b1 = b - E1 - y1 * (a1n - a1) * K[i1][i1] - y2 * (a2n - a2) * K[i1][i2];
    const b2 = b - E2 - y1 * (a1n - a1) * K[i1][i2] - y2 * (a2n - a2) * K[i2][i2];
    let bn;
    if (a1n > 0 && a1n < C) bn = b1; else if (a2n > 0 && a2n < C) bn = b2; else bn = (b1 + b2) / 2;
    const d1 = y1 * (a1n - a1), d2 = y2 * (a2n - a2), db = bn - b;
    for (let i = 0; i < n; i++) E[i] += d1 * K[i][i1] + d2 * K[i][i2] + db;
    alpha[i1] = a1n; alpha[i2] = a2n; b = bn;
    return true;
  };
  const r = ML.rng(11);
  const examine = (i2) => {
    const y2 = y[i2], a2 = alpha[i2], r2 = E[i2] * y2;
    if ((r2 < -tol && a2 < C) || (r2 > tol && a2 > 0)) {
      let best = -1, bestGap = -1;
      for (let i = 0; i < n; i++) if (alpha[i] > 0 && alpha[i] < C) { const g = Math.abs(E[i] - E[i2]); if (g > bestGap) { bestGap = g; best = i; } }
      if (best >= 0 && takeStep(best, i2)) return 1;
      const st0 = Math.floor(r() * n);
      for (let k = 0; k < n; k++) { const i = (st0 + k) % n; if (alpha[i] > 0 && alpha[i] < C && takeStep(i, i2)) return 1; }
      const st1 = Math.floor(r() * n);
      for (let k = 0; k < n; k++) { const i = (st1 + k) % n; if (takeStep(i, i2)) return 1; }
    }
    return 0;
  };
  let changed = 0, all = true, passes = 0;
  while ((changed > 0 || all) && passes < maxPasses) {
    changed = 0;
    for (let i = 0; i < n; i++) if (all || (alpha[i] > 0 && alpha[i] < C)) changed += examine(i);
    if (all) all = false; else if (changed === 0) all = true;
    passes++;
  }
  const sv = [];
  for (let i = 0; i < n; i++) if (alpha[i] > 1e-7) sv.push(i);
  const f = (x) => { let s = b; for (const i of sv) s += alpha[i] * y[i] * kf(X[i], x); return s; };
  let w = null;
  if (kernel === "linear") { w = [0, 0]; for (const i of sv) { w[0] += alpha[i] * y[i] * X[i][0]; w[1] += alpha[i] * y[i] * X[i][1]; } }
  return { alpha, b, sv, f, w };
};

ML.register("svm", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Separate classes with a <b>wide margin</b>. A linear SVM for two classes, labeled −1 and +1, draws a boundary and keeps it as far as possible from the closest training examples: the <b>support vectors</b>.", ["sup", "cls", "C", "kernel", "gamma"])}

  <div class="two-col">
    ${ML.formula("Decision rule", String.raw`f(x) = w^\top x + b = \sum_{j=1}^{p} w_j x_j + b, \qquad \hat{y} = \operatorname{sign}\big(f(x)\big)`, "The weights w set the boundary's orientation and b shifts it. Positive score predicts +1, negative predicts −1, and <b>f(x) = 0 is the decision boundary</b>. The superscript ⊤ (transpose) just writes the weighted sum compactly.", "Eq. 2")}
    ${ML.formula("Margin width", String.raw`\text{margin width} = \frac{2}{\lVert w \rVert_2}, \qquad \lVert w \rVert_2 = \sqrt{\textstyle\sum_{j=1}^{p} w_j^2}`, "The margin lines are f(x) = −1 and f(x) = +1. A <b>smaller weight-vector length gives a wider margin</b>.", "Eq. 3")}
  </div>

  <div class="section-title"><h2>The decision function as a landscape</h2><span class="hint">Height = f(x). Where the surface crosses the zero level, the class changes.</span></div>
  ${ML.bench("svm", { tag: "SVC trained live in your browser (SMO solver)", stage: "tall", cls: "wide-panel" })}
  <p class="fig-cap">The floor shows the predicted classes. Solid line: boundary f(x) = 0. Dashed lines: margin lines f(x) = ±1, drawn both on the floor and on the surface. Ringed points are support vectors (α<sub>i</sub> &gt; 0). Use "Top view" to see the flat picture from the script's Figure 1 and 2.</p>

  <div class="two-col">
    ${ML.formula("Hard margin: perfectly separable data", String.raw`\min_{w,b}\ \tfrac12 \lVert w\rVert_2^2 \quad \text{subject to}\quad y_i\,(w^\top x_i + b) \ge 1,\ \ i = 1,\dots,n`, "Read “subject to” as “while satisfying”. Every example sits on the correct side, on or beyond its margin line. <b>Requires linearly separable training data.</b>", "Eq. 4")}
    ${ML.formula("Soft margin: allowing violations", String.raw`\min_{w,b,\xi}\ \tfrac12 \lVert w\rVert_2^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0`, "Slack variables ξ<sub>i</sub> let points violate the margin. <b>Larger C penalizes violations more strongly.</b> Choose C by validation; larger or smaller is not always better.", "Eq. 5")}
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Slack at the optimum</th><th>Where the point lies</th><th>Live count</th></tr></thead>
    <tbody>
      <tr><td>\(\xi_i = 0\)</td><td>on or beyond the correct margin line</td><td class="mono" id="svm-x0">-</td></tr>
      <tr><td>\(0 < \xi_i < 1\)</td><td>inside the margin but correctly classified</td><td class="mono" id="svm-x1">-</td></tr>
      <tr><td>\(\xi_i = 1\)</td><td>exactly on the boundary</td><td class="mono">rare</td></tr>
      <tr><td>\(\xi_i > 1\)</td><td>misclassified</td><td class="mono" id="svm-x2">-</td></tr>
    </tbody>
  </table></div>

  <div class="section-title"><h2>Kernels: curved boundaries</h2><span class="hint">Why a straight cut in a lifted space looks curved below.</span></div>
  <div class="two-col">
    ${ML.formula("RBF kernel", String.raw`K_{\mathrm{RBF}}(x, z) = \exp\!\big(-\gamma \lVert x - z\rVert_2^2\big)`, "A kernel measures <b>similarity</b> through an implicit feature mapping. Nearby points have higher similarity. <b>Larger γ makes similarity decrease faster with distance</b>, allowing more local detail.", "Eq. 6")}
    <div id="svm-gplot"></div>
  </div>
  ${ML.bench("lift", { tag: "Kernel idea: lift the circles dataset", stage: "short" })}
  <p class="fig-cap">Each point gets a third coordinate x<sub>1</sub>² + x<sub>2</sub>². In the lifted space, a flat plane separates the classes. Seen from above, the plane's cut is a circle: a linear separator in the mapped space becomes a curved boundary in the original coordinates. The RBF kernel does a similar mapping implicitly, without computing the new coordinates.</p>

  <div class="section-title"><h2>In scikit-learn</h2></div>
  <div class="two-col">
    ${ML.code(`
from sklearn.svm import SVC

# Reuse the imports and Iris split from page 2
model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0)
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("SVM accuracy:", accuracy_score(y_test, predictions))`, `SVM accuracy: 0.9666666666666667`)}
    <div class="prose">
      <p>Scikit-learn also handles class labels other than −1/+1 and more than two classes, so Iris works directly.</p>
      ${ML.callout("setting", "Main settings", "<b>C</b>: penalty for margin violations (small C = wide, tolerant margin; large C = tries hard to classify every training point). <b>kernel</b>: \"linear\" or \"rbf\". <b>gamma</b> (RBF only): reach of each training example. Tune C and gamma with cross-validation.")}
      ${ML.callout("trap", "Exam traps", "<ul><li>The margin width is <b>2/‖w‖</b>, the full distance between the two dashed lines.</li><li>Only support vectors determine the boundary. Moving any other point (without crossing the margin) changes nothing.</li><li>The figures show fitted data, not test accuracy.</li></ul>")}
    </div>
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="svm"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const NEG = col.c[0], POS = col.c[2];
    const DS = {
      separable: () => { const d = ML.makeBlobs([[-1, -0.8], [1, 0.9]], 30, 0.36, 21); return d; },
      overlap: () => ML.makeBlobs([[-0.55, -0.45], [0.55, 0.5]], 35, 0.62, 4),
      moons: () => { const d = ML.makeMoons(90, 0.14, 8); d.X = d.X.map(([a, b]) => [(a - 0.5) * 1.45, (b - 0.25) * 1.7]); return d; },
      circles: () => { const d = ML.makeCircles(90, 0.09, 0.45, 2); d.X = d.X.map(([a, b]) => [a * 1.9, b * 1.9]); return d; },
    };
    const state = { ds: "separable", kernel: "linear", C: 1000, gamma: 1 };
    let data, model;
    const host = root.querySelector("#svm-stage"), panel = root.querySelector("#svm-panel");
    const st = new ML.Stage3D(host, { cam: [0.4, 6.4, 8.2], target: [0, -0.4, 0] });
    disposers.push(() => st.dispose());
    const R = 2.4, N = 72, HS = 0.33, FLOOR = -1.25;
    let surf, floorTex, pts, lines = new THREE.Group(), rings = new THREE.Group();
    if (st.ok) {
      ML.axes(st, { x: [-R, R], y: [-R, R], z: [FLOOR / HS * 0 + FLOOR, FLOOR], labels: ["feature x1", "feature x2", null], div: 6 });
      const g = new THREE.PlaneGeometry(2 * R, 2 * R, N, N);
      g.rotateX(-Math.PI / 2);
      g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array((N + 1) * (N + 1) * 3), 3));
      surf = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.82 }));
      st.add(surf);
      floorTex = ML.texturePlane(2 * R, 2 * R, 220);
      floorTex.mesh.position.y = FLOOR + 0.001;
      st.add(floorTex.mesh);
      const zeroPlane = new THREE.Mesh(new THREE.PlaneGeometry(2 * R, 2 * R), new THREE.MeshBasicMaterial({ color: col.ink, transparent: true, opacity: 0.05, side: THREE.DoubleSide, depthWrite: false }));
      zeroPlane.rotation.x = -Math.PI / 2;
      st.add(zeroPlane);
      const zl = ML.label("f(x) = 0 level", { h: 0.2, color: col.muted });
      zl.position.set(R + 0.1, 0, -R);
      st.add(zl);
      st.add(lines); st.add(rings);
      pts = new ML.Spheres(st, 100, 0.07);
      st.views([
        { t: "3D", pos: [0.4, 6.4, 8.2], target: [0, -0.4, 0] },
        { t: "Top view", pos: [0, 8.6, 0.01], target: [0, FLOOR, 0] },
        { t: "Side", pos: [7.5, 0.4, 0.01], target: [0, -0.2, 0] },
      ]);
      ML.legend(host, [{ label: "class −1", color: NEG }, { label: "class +1", color: POS }, { label: "support vector", ring: col.ink, color: "transparent" }]);
    }

    // Panel.
    ML.ui.heading(panel, "Data and model");
    ML.ui.seg(panel, { label: "Dataset", options: [{ v: "separable", t: "Separable" }, { v: "overlap", t: "Overlap" }, { v: "moons", t: "Moons" }, { v: "circles", t: "Circles" }], value: state.ds, onChange: (v) => { state.ds = v; newData(); } });
    ML.ui.seg(panel, { label: "kernel", options: [{ v: "linear", t: "linear" }, { v: "rbf", t: "rbf" }], value: state.kernel, onChange: (v) => { state.kernel = v; gS.input.disabled = v === "linear"; train(); } });
    ML.ui.slider(panel, { label: "C (violation penalty)", min: 0.01, max: 1000, log: true, value: state.C, fmt: (v) => v >= 100 ? v.toFixed(0) : v.toString(), note: "C = 1000 on separable data behaves like a hard margin.", onInput: (v) => { state.C = v; train(); } });
    const gS = ML.ui.slider(panel, { label: "gamma (RBF only)", min: 0.05, max: 30, log: true, value: state.gamma, onInput: (v) => { state.gamma = v; train(); drawG(); } });
    gS.input.disabled = true;
    ML.ui.heading(panel, "Fitted model");
    const stats = ML.ui.stats(panel, [
      { key: "sv", label: "Support vectors", big: true },
      { key: "acc", label: "Training accuracy" },
      { key: "w", label: "‖w‖ (linear)" },
      { key: "mw", label: "Margin width 2/‖w‖" },
    ]);
    const verdict = ML.ui.verdict(panel);

    function newData() {
      data = DS[state.ds]();
      data.ys = data.y.map((v) => (v === 1 ? 1 : -1));
      train();
    }
    function train() {
      model = ML.svmTrain(data.X, data.ys, { C: state.C, kernel: state.kernel, gamma: state.gamma });
      const fx = data.X.map((x) => model.f(x));
      const xi = fx.map((f, i) => Math.max(0, 1 - data.ys[i] * f));
      const c0 = xi.filter((v) => v < 1e-3).length, c2 = xi.filter((v) => v > 1).length, c1 = xi.length - c0 - c2;
      root.querySelector("#svm-x0").textContent = c0;
      root.querySelector("#svm-x1").textContent = c1;
      root.querySelector("#svm-x2").textContent = c2;
      const acc = fx.filter((f, i) => Math.sign(f) === data.ys[i]).length / fx.length;
      stats.set("sv", `${model.sv.length} of ${data.X.length}`);
      stats.set("acc", acc.toFixed(3));
      if (model.w) {
        const nw = Math.hypot(...model.w);
        stats.set("w", nw.toFixed(3)); stats.set("mw", (2 / nw).toFixed(3));
      } else { stats.set("w", "n/a"); stats.set("mw", "curved"); }
      let msg;
      if (state.kernel === "linear" && (state.ds === "moons" || state.ds === "circles")) msg = "A straight line cannot follow this shape. Switch the kernel to <b>rbf</b>.";
      else if (state.C < 0.1) msg = "Small C: violations are cheap, so the margin is wide and many points become support vectors.";
      else if (state.C > 100 && c1 + c2 === 0) msg = "Large C and no violations: this is the <b>hard-margin</b> solution.";
      else if (state.kernel === "rbf" && state.gamma > 8) msg = "Large gamma: each point only influences its close neighborhood. The boundary wraps tightly around single points, a sign of overfitting.";
      else msg = `${c1 + c2} margin violations (ξ &gt; 0), ${c2} of them misclassified.`;
      verdict.innerHTML = msg;
      if (st.ok) drawScene(fx, xi);
    }
    function drawScene(fx, xi) {
      const g = surf.geometry, pos = g.attributes.position, colr = g.attributes.color;
      const cN = new THREE.Color(NEG), cP = new THREE.Color(POS), mid = new THREE.Color(col.surface), tmp = new THREE.Color();
      const grid = [];
      for (let j = 0; j <= N; j++) { grid.push([]); }
      for (let v = 0; v < pos.count; v++) {
        const X = pos.getX(v), Z = pos.getZ(v);
        const f = model.f([X, -Z]);
        const fc = ML.clamp(f, -2.2, 2.2);
        pos.setY(v, fc * HS);
        tmp.copy(mid).lerp(f > 0 ? cP : cN, ML.clamp(0.35 + Math.abs(f) * 0.45, 0, 1));
        colr.setXYZ(v, tmp.r, tmp.g, tmp.b);
        const ix = v % (N + 1), iz = Math.floor(v / (N + 1));
        grid[iz][ix] = f;
      }
      pos.needsUpdate = true; colr.needsUpdate = true; g.computeVertexNormals();
      // Floor texture: predicted class regions.
      const rn = ML.rgb(NEG), rp = ML.rgb(POS);
      floorTex.redraw((u, v) => {
        const f = model.f([-R + u * 2 * R, -R + v * 2 * R]);
        const c = f > 0 ? rp : rn, a = Math.abs(f) < 1 ? 120 : 70;
        return [c[0], c[1], c[2], a];
      });
      // Iso-lines with marching squares on the surface grid.
      st.clearGroup(lines);
      const step = (2 * R) / N;
      [-1, 0, 1].forEach((lv) => {
        const segsTop = [], segsFloor = [];
        for (let iz = 0; iz < N; iz++) for (let ix = 0; ix < N; ix++) {
          const c = [[ix, iz], [ix + 1, iz], [ix + 1, iz + 1], [ix, iz + 1]];
          const vals = c.map(([a, b]) => grid[b][a] - lv);
          const cr = [];
          for (let e = 0; e < 4; e++) {
            const a = vals[e], b = vals[(e + 1) % 4];
            if ((a > 0) !== (b > 0)) {
              const t = a / (a - b), p0 = c[e], p1 = c[(e + 1) % 4];
              cr.push([-R + (p0[0] + (p1[0] - p0[0]) * t) * step, -R + (p0[1] + (p1[1] - p0[1]) * t) * step]);
            }
          }
          for (let k = 0; k + 1 < cr.length; k += 2) {
            const [a, b] = [cr[k], cr[k + 1]];
            const h = ML.clamp(lv, -3, 3) * HS + 0.01;
            segsTop.push([new THREE.Vector3(a[0], h, a[1]), new THREE.Vector3(b[0], h, b[1])]);
            segsFloor.push([new THREE.Vector3(a[0], FLOOR + 0.01, a[1]), new THREE.Vector3(b[0], FLOOR + 0.01, b[1])]);
          }
        }
        const colL = new THREE.Color(col.ink);
        if (lv === 0) { lines.add(ML.segments(segsTop, colL, 1)); lines.add(ML.segments(segsFloor, colL, 1)); }
        else {
          // Dashed look: keep every other segment.
          lines.add(ML.segments(segsTop.filter((_, i) => i % 3 !== 2), colL, 0.55));
          lines.add(ML.segments(segsFloor.filter((_, i) => i % 3 !== 2), colL, 0.65));
        }
      });
      // Points on the floor + rings for support vectors.
      st.clearGroup(rings);
      pts.setCount(data.X.length);
      const svSet = new Set(model.sv);
      data.X.forEach((x, i) => {
        pts.set(i, ML.P(x[0], x[1], FLOOR + 0.07), data.ys[i] > 0 ? POS : NEG, xi[i] > 1 ? 1.2 : 1);
        if (svSet.has(i)) {
          const r = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.018, 8, 28), new THREE.MeshBasicMaterial({ color: col.ink }));
          r.rotation.x = -Math.PI / 2;
          r.position.copy(ML.P(x[0], x[1], FLOOR + 0.07));
          rings.add(r);
          // Stem from the point up to the surface shows f(x_i).
          const fz = ML.clamp(fx[i], -2.2, 2.2) * HS;
          rings.add(ML.line([ML.P(x[0], x[1], FLOOR + 0.07), ML.P(x[0], x[1], fz)], new THREE.Color(col.muted), { opacity: 0.5 }));
        }
      });
      pts.commit();
    }

    // Gamma similarity curve.
    const gp = new ML.Plot2D(root.querySelector("#svm-gplot"), { height: 230, xlim: [0, 3], ylim: [0, 1.05], aria: "RBF similarity versus distance" });
    function drawG() {
      gp.clear();
      gp.frame({ xticks: [0, 0.5, 1, 1.5, 2, 2.5, 3], yticks: [0, 0.25, 0.5, 0.75, 1], xlabel: "distance ‖x − z‖", ylabel: "K(x, z)" });
      const curve = (g) => Array.from({ length: 121 }, (_, i) => { const d = (i / 120) * 3; return [d, Math.exp(-g * d * d)]; });
      [0.1, 1, 10].forEach((g) => gp.path(curve(g), col.muted, 1.2, [4, 4]));
      gp.path(curve(state.gamma), col.accent, 2.6);
      gp.text(`γ = ${state.gamma}`, gp.W - 24, 24, col.accent, "right", '600 13px "IBM Plex Mono", monospace');
      gp.text("dashed: γ = 0.1, 1, 10", gp.W - 24, 42, col.muted, "right");
    }
    gp.drawFn = drawG;
    drawG();
    disposers.push(() => gp.dispose());
    newData();

    // Lift demo.
    const lhost = root.querySelector("#lift-stage"), lpanel = root.querySelector("#lift-panel");
    const ls = new ML.Stage3D(lhost, { cam: [4.6, 3.2, 5.2], target: [0, 0.8, 0] });
    disposers.push(() => ls.dispose());
    const cd = ML.makeCircles(160, 0.07, 0.45, 9);
    const LS = 1.6;
    let lift = 0, lsph, cutPlane, ring;
    if (ls.ok) {
      ML.axes(ls, { x: [-2, 2], y: [-2, 2], z: [0, 2.2], labels: ["x1", "x2", "x1² + x2²"], div: 4 });
      lsph = new ML.Spheres(ls, cd.X.length, 0.06);
      cutPlane = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), new THREE.MeshBasicMaterial({ color: col.accent, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
      cutPlane.rotation.x = -Math.PI / 2;
      ls.add(cutPlane);
      const rp = []; for (let i = 0; i <= 96; i++) { const t = (i / 96) * Math.PI * 2; rp.push(new THREE.Vector3(Math.cos(t), 0.005, Math.sin(t)).multiplyScalar(1)); }
      ring = ML.line(rp, new THREE.Color(col.ink));
      ls.add(ring);
    }
    const thr = 0.53; // squared radius between the two circles (0.45² and 1²), scaled
    const drawLift = () => {
      if (!ls.ok) return;
      cd.X.forEach((x, i) => {
        const p = x.map((v) => v * LS);
        const z = lift * (x[0] ** 2 + x[1] ** 2) * 1.6;
        lsph.set(i, ML.P(p[0], p[1], z), cd.y[i] ? POS : NEG);
      });
      lsph.commit();
      cutPlane.position.y = lift * thr * 1.6;
      cutPlane.material.opacity = 0.05 + 0.2 * lift;
      const rr = Math.sqrt(thr) * LS;
      ring.scale.set(rr, 1, rr);
      ring.material.opacity = lift;
    };
    ML.ui.heading(lpanel, "Feature mapping");
    ML.ui.slider(lpanel, { label: "Lift amount", min: 0, max: 1, step: 0.01, value: 0, fmt: (v) => `${Math.round(v * 100)}%`, onInput: (v) => { lift = v; drawLift(); } });
    const lr = ML.ui.row(lpanel);
    let animId = null;
    ML.ui.button(lr, "Animate lift", () => {
      cancelAnimationFrame(animId);
      const t0 = performance.now(), from = lift, to = lift > 0.5 ? 0 : 1;
      const go = (t) => { const k = ML.ease(Math.min(1, (t - t0) / 1600)); lift = ML.lerp(from, to, k); lpanel.querySelector("input").value = lift; lpanel.querySelector("output").textContent = `${Math.round(lift * 100)}%`; drawLift(); if (k < 1) animId = requestAnimationFrame(go); };
      animId = requestAnimationFrame(go);
    });
    ML.ui.button(lr, "Top view", () => ls.flyTo([0, 6.5, 0.01], [0, 0, 0]), true);
    ML.ui.button(lr, "Side view", () => ls.flyTo([6, 0.9, 0.01], [0, 0.8, 0]), true);
    lpanel.append(ML.el("div", { class: "verdict", html: "At 0% no straight line separates the rings. At 100% the inner ring sits low and the outer ring high, so the <b>flat blue plane</b> splits them. Its trace on the floor is the black circle." }));
    disposers.push(() => cancelAnimationFrame(animId));
    drawLift();
    return disposers;
  },
});
