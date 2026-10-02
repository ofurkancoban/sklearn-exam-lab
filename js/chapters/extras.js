// Extra section: topics that appear only in the long version of the script.
ML.register("extras", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Topics from the <b>long version</b> of the lecture notes that the short exam script leaves out. Use them to understand the methods more deeply; the exam questions are based on the short script.", ["long version", "optional"])}
  ${ML.callout("setting", "Not in the exam script", "Everything on this page comes from the 42-page long version. Questions about it are marked “Beyond the script” and are left out of the mock exam.")}

  <div class="section-title"><h2>Decision trees: entropy and information gain</h2><span class="hint">How a tree (ID3) picks its question.</span></div>
  <div class="two-col">
    ${ML.formula("Entropy and Gini impurity", String.raw`H = -\sum_{c} p_c \log_2 p_c, \qquad \text{Gini} = 1 - \sum_c p_c^2`, "Both measure how mixed the classes in a node are: 0 for a pure node, maximal when all classes are equally frequent. ID3 uses entropy; scikit-learn's trees use Gini by default.")}
    ${ML.formula("Information gain", String.raw`\text{Gain} = H(\text{parent}) - \sum_{k} \frac{n_k}{n} H(\text{child}_k)`, "ID3 builds the tree greedily: at each node it tries every feature and picks the split with the <b>largest information gain</b>, i.e. the one that reduces uncertainty about the class most.")}
  </div>
  ${ML.bench("en", { tag: "Impurity of a node with three classes", stage: "" })}
  <p class="fig-cap">Each point of the triangle floor is a class mix (p<sub>1</sub>, p<sub>2</sub>, p<sub>3</sub>); its height is the impurity. Corners are pure nodes (height 0); the peak in the middle is the 1/3, 1/3, 1/3 mix. A good split moves the children toward the corners.</p>
  <div class="bench wide-panel">
    <div class="stage short" style="background:var(--surface)"><span class="stage-tag">Split builder: parent node with 10 A and 10 B</span><div id="ig-tree" style="position:absolute;inset:34px 10px 10px 10px"></div></div>
    <div class="panel" id="ig-panel"></div>
  </div>

  <div class="section-title"><h2>Activation functions and softmax</h2></div>
  <div class="two-col">
    ${ML.formula("Neuron, sigmoid, ReLU", String.raw`z = w^\top x + b,\qquad \sigma(z) = \frac{1}{1 + e^{-z}},\qquad \text{ReLU}(z) = \max(0, z)`, "Layer by layer: h<sup>(ℓ)</sup> = σ(W<sup>(ℓ)</sup>h<sup>(ℓ−1)</sup> + b<sup>(ℓ)</sup>) with h<sup>(0)</sup> = x. Weights are learned by backpropagation and gradient methods such as SGD or Adam.")}
    ${ML.formula("Softmax output and cross-entropy loss", String.raw`\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}, \qquad L = -\sum_k y_k \log \hat{y}_k`, "Softmax turns the output scores into class probabilities that sum to 1. With a one-hot y, the loss is simply −log of the probability given to the true class.")}
  </div>
  <div class="two-col">
    <div id="act-plot"></div>
    <div class="formula" id="sm-box">
      <div class="f-label"><span>Try it: softmax of three scores</span></div>
    </div>
  </div>

  <div class="section-title"><h2>Min-max scaling vs. standardization</h2></div>
  <div class="two-col">
    ${ML.formula("Min-max scaling", String.raw`x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}`, "Maps the training values to [0, 1]. One extreme value sets x<sub>max</sub> and squeezes all other values together; standardization is less affected.")}
    <div id="mm-plot"></div>
  </div>

  <div class="section-title"><h2>More regression details</h2></div>
  <div class="two-col">
    ${ML.formula("Ordinary least squares, closed form", String.raw`\theta = (X^\top X)^{-1} X^\top y`, "θ contains the intercept and weights after a column of ones is added to X. Gradient descent reaches the same minimum iteratively, useful for large datasets.")}
    <div class="formula">
      <div class="f-label"><span>Worked example</span><span>x = (1, 2, 3), y = (2, 3, 5)</span></div>
      <div class="f-body" style="font-size:15px">\[X = \begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix},\ X^\top X = \begin{pmatrix}3&6\\6&14\end{pmatrix},\ X^\top y = \begin{pmatrix}10\\23\end{pmatrix}\]
      \[(X^\top X)^{-1} = \begin{pmatrix}2.333&-1\\-1&0.5\end{pmatrix} \;\Rightarrow\; \theta = \begin{pmatrix}b\\w\end{pmatrix} = \begin{pmatrix}0.333\\1.5\end{pmatrix}\]</div>
      <div class="f-note">The best line is ŷ = 0.333 + 1.5x.</div>
    </div>
  </div>
  ${ML.bench("gd", { tag: "Gradient descent on the MSE bowl of the worked example", stage: "" })}
  <p class="fig-cap">Height = MSE for every (b, w). The black ring marks the closed-form solution θ = (0.333, 1.5). Gradient descent starts at (−2, −1) and walks downhill; a too-large learning rate makes it zig-zag or diverge.</p>
  <div class="two-col">
    ${ML.formula("MAPE", String.raw`\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|`, "Error as a percentage of the true value. Problematic when targets are zero or close to zero: the division explodes.")}
    <div class="formula" id="mape-box"><div class="f-label"><span>Try it</span><span>ŷ is always 1 too high</span></div></div>
  </div>

  <div class="section-title"><h2>Fair experiments and the Wilcoxon test</h2></div>
  <div class="two-col">
    <div class="prose">
      <p>Experiments should be fair, reproducible and comparable: separate training and test data, fit preprocessing and hyperparameters on training data only, give all methods the same splits, and repeat with several random seeds, reporting mean and standard deviation.</p>
      <p>Differences may come from random variation. The <b>Wilcoxon signed-rank test</b> checks paired results (same seed, same split) from repeated runs. A p-value below 0.05 is commonly read as evidence of a significant difference; report it together with the observed difference.</p>
      <p>Random forests also report <code>feature_importances_</code>: how much each feature reduces the error across all trees.</p>
    </div>
    ${ML.code(`
from scipy.stats import wilcoxon

scores_A = [0.91, 0.90, 0.92, 0.89, 0.93,
            0.91, 0.90, 0.92, 0.94, 0.91]
scores_B = [0.88, 0.89, 0.90, 0.87, 0.91,
            0.89, 0.88, 0.90, 0.91, 0.89]
statistic, p_value = wilcoxon(scores_A, scores_B)
print(f"p-value: {p_value:.4f}")`, `p-value: 0.0020`)}
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="extras"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const H = (p) => -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
    const G = (p) => 1 - p.reduce((s, v) => s + v * v, 0);
    // ----- Impurity surface over the probability triangle -----
    const host = root.querySelector("#en-stage"), panel = root.querySelector("#en-panel");
    const st = new ML.Stage3D(host, { cam: [0.2, 4.2, 4.6], target: [0, 0.5, -0.2] });
    disposers.push(() => st.dispose());
    const S = { kind: "entropy", p: [0.6, 0.3, 0.1] };
    const corners = [[-1.6, -1.2], [1.6, -1.2], [0, 1.6]];
    const toXY = (p) => [0, 1].map((k) => p[0] * corners[0][k] + p[1] * corners[1][k] + p[2] * corners[2][k]);
    const HS = 1.1;
    let surf, marker;
    if (st.ok) {
      const N = 40, pos = [], colr = [], idx = [], ids = {};
      let k = 0;
      for (let i = 0; i <= N; i++) for (let j = 0; j <= N - i; j++) { ids[i + "," + j] = k++; const p = [i / N, j / N, (N - i - j) / N]; const [x, y] = toXY(p); pos.push(x, 0, -y); colr.push(0, 0, 0); }
      for (let i = 0; i < N; i++) for (let j = 0; j < N - i; j++) {
        idx.push(ids[i + "," + j], ids[i + 1 + "," + j], ids[i + "," + (j + 1)]);
        if (j < N - i - 1) idx.push(ids[i + 1 + "," + j], ids[i + 1 + "," + (j + 1)], ids[i + "," + (j + 1)]);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setAttribute("color", new THREE.Float32BufferAttribute(colr, 3));
      g.setIndex(idx);
      g.userData.N = N;
      surf = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
      st.add(surf);
      st.add(ML.line([...corners, corners[0]].map(([x, y]) => ML.P(x, y, 0)), new THREE.Color(col.muted)));
      ["class 1 pure", "class 2 pure", "class 3 pure"].forEach((t, i) => { const s = ML.label(t, { h: 0.18, color: col.c[i] }); const [x, y] = corners[i]; s.position.copy(ML.P(x * 1.15, y * 1.15, 0.05)); st.add(s); });
      marker = new THREE.Mesh(new THREE.SphereGeometry(0.08, 20, 14), new THREE.MeshStandardMaterial({ color: col.ink }));
      st.add(marker);
    }
    ML.ui.heading(panel, "Impurity measure");
    ML.ui.seg(panel, { options: [{ v: "entropy", t: "Entropy" }, { v: "gini", t: "Gini" }], value: "entropy", onChange: (v) => { S.kind = v; draw(); } });
    ML.ui.heading(panel, "Class mix of one node");
    const a = ML.ui.slider(panel, { label: "share of class 1", min: 0, max: 1, step: 0.01, value: 0.6, onInput: () => mix() });
    const b = ML.ui.slider(panel, { label: "share of class 2 (of the rest)", min: 0, max: 1, step: 0.01, value: 0.75, onInput: () => mix() });
    const stats = ML.ui.stats(panel, [{ key: "p", label: "p = (p1, p2, p3)" }, { key: "h", label: "Entropy H (bits)", big: true }, { key: "g", label: "Gini" }]);
    function mix() { const p1 = a.get(), p2 = (1 - p1) * b.get(); S.p = [p1, p2, 1 - p1 - p2]; draw(); }
    function draw() {
      const f = S.kind === "entropy" ? (p) => H(p) / Math.log2(3) * 1.58 : (p) => (G(p) / (2 / 3)) * 1.58;
      stats.set("p", S.p.map((v) => v.toFixed(2)).join(", "));
      stats.set("h", H(S.p).toFixed(3)); stats.set("g", G(S.p).toFixed(3));
      if (!st.ok) return;
      const g = surf.geometry, pos = g.attributes.position, cl = g.attributes.color, N = g.userData.N;
      const lo = new THREE.Color(col.c[1]), hi = new THREE.Color(col.c[2]), t = new THREE.Color();
      let k = 0;
      for (let i = 0; i <= N; i++) for (let j = 0; j <= N - i; j++) {
        const p = [i / N, j / N, (N - i - j) / N], h = f(p);
        pos.setY(k, h * HS * 0.6); t.copy(lo).lerp(hi, h / 1.6); cl.setXYZ(k, t.r, t.g, t.b); k++;
      }
      pos.needsUpdate = true; cl.needsUpdate = true; g.computeVertexNormals();
      const [x, y] = toXY(S.p);
      marker.position.copy(ML.P(x, y, f(S.p) * HS * 0.6 + 0.05));
    }
    mix();

    // ----- Split builder (information gain) -----
    const ip = root.querySelector("#ig-panel"), treeBox = root.querySelector("#ig-tree");
    const T = { aL: 8, bL: 2 };
    ML.ui.heading(ip, "Left child gets");
    ML.ui.slider(ip, { label: "class A examples", min: 0, max: 10, value: T.aL, onInput: (v) => { T.aL = v; drawSplit(); } });
    ML.ui.slider(ip, { label: "class B examples", min: 0, max: 10, value: T.bL, onInput: (v) => { T.bL = v; drawSplit(); } });
    const gst = ML.ui.stats(ip, [{ key: "hp", label: "H(parent)" }, { key: "hw", label: "Weighted H(children)" }, { key: "ig", label: "Information gain", big: true }]);
    const gv = ML.ui.verdict(ip);
    function drawSplit() {
      const L = [T.aL, T.bL], R = [10 - T.aL, 10 - T.bL];
      const ent = (c) => { const n = c[0] + c[1]; return n ? H(c.map((v) => v / n)) : 0; };
      const nL = L[0] + L[1], nR = R[0] + R[1];
      const hw = (nL / 20) * ent(L) + (nR / 20) * ent(R), ig = 1 - hw;
      gst.set("hp", "1.000"); gst.set("hw", hw.toFixed(3)); gst.set("ig", ig.toFixed(3));
      gv.innerHTML = ig > 0.99 ? "Perfect split: both children are pure." : ig < 0.01 ? "Useless split: the children have the same mix as the parent." : `ID3 prefers the candidate split with the largest gain. This one removes ${(ig * 100).toFixed(0)}% of the uncertainty.`;
      const node = (x, y, c, lab) => `<g><rect x="${x - 70}" y="${y - 26}" width="140" height="52" rx="8" fill="${col.surface2}" stroke="${col.line}"/><text x="${x}" y="${y - 6}" text-anchor="middle" font-size="13" font-family="IBM Plex Mono, monospace" fill="${col.ink}">${c[0]} A · ${c[1]} B</text><text x="${x}" y="${y + 14}" text-anchor="middle" font-size="11.5" fill="${col.muted}">${lab}</text></g>`;
      treeBox.innerHTML = `<svg viewBox="0 0 520 240" width="100%" height="100%" role="img" aria-label="Split diagram">
        <line x1="260" y1="66" x2="130" y2="164" stroke="${col.line}" stroke-width="2"/><line x1="260" y1="66" x2="390" y2="164" stroke="${col.line}" stroke-width="2"/>
        ${node(260, 40, [10, 10], "H = 1.000")}${node(130, 190, L, `H = ${ent(L).toFixed(3)}, n = ${nL}`)}${node(390, 190, R, `H = ${ent(R).toFixed(3)}, n = ${nR}`)}
        <text x="185" y="112" font-size="12" fill="${col.muted}">yes</text><text x="318" y="112" font-size="12" fill="${col.muted}">no</text></svg>`;
    }
    drawSplit();

    // ----- Activations -----
    const ap = new ML.Plot2D(root.querySelector("#act-plot"), { height: 280, xlim: [-5, 5], ylim: [-0.5, 3], aria: "Activation functions" });
    disposers.push(() => ap.dispose());
    ap.drawFn = () => {
      ap.clear();
      ap.frame({ xticks: [-4, -2, 0, 2, 4], yticks: [0, 1, 2, 3], xlabel: "z", ylabel: "activation" });
      const xs = Array.from({ length: 201 }, (_, i) => -5 + i * 0.05);
      ap.path(xs.map((z) => [z, 1 / (1 + Math.exp(-z))]), col.c[1], 2.5);
      ap.path(xs.map((z) => [z, Math.max(0, z)]), col.warn, 2.5);
      ap.text("sigmoid σ(z): squashes to (0, 1)", 56, 22, col.c[1]);
      ap.text("ReLU max(0, z)", 56, 40, col.warn);
    };
    ap.drawFn();
    const sb = root.querySelector("#sm-box");
    const Z = [2, 1, 0.1];
    let trueC = 0;
    const smBars = ML.ui.bars(sb, ["class 1", "class 2", "class 3"].map((t, i) => ({ label: t, color: col.c[i] })));
    const smOut = ML.el("div", { class: "f-note mono" });
    Z.forEach((z, i) => ML.ui.slider(sb, { label: `score z${i + 1}`, min: -3, max: 4, step: 0.1, value: z, onInput: (v) => { Z[i] = v; sm(); } }));
    ML.ui.seg(sb, { label: "True class", options: [0, 1, 2].map((i) => ({ v: i, t: `class ${i + 1}` })), value: 0, onChange: (v) => { trueC = v; sm(); } });
    sb.append(smOut);
    function sm() {
      const e = Z.map(Math.exp), s = e.reduce((p, q) => p + q, 0), p = e.map((v) => v / s);
      smBars.set(p, (v) => v.toFixed(3));
      smOut.innerHTML = `ŷ = (${p.map((v) => v.toFixed(3)).join(", ")}), sum = 1<br>loss L = −log ŷ<sub>${trueC + 1}</sub> = <b style="color:var(--accent)">${(-Math.log(p[trueC])).toFixed(3)}</b>`;
    }
    sm();

    // ----- Min-max vs standardization -----
    const base = [22, 25, 27, 30, 31, 33, 36, 38, 41, 45];
    const MM = { outlier: false };
    const mp = new ML.Plot2D(root.querySelector("#mm-plot"), { height: 250, xlim: [-2.5, 3.2], ylim: [0, 3], pad: [16, 16, 34, 120], aria: "Scaling comparison" });
    disposers.push(() => mp.dispose());
    mp.drawFn = () => {
      const v = MM.outlier ? [...base, 140] : base;
      const mn = Math.min(...v), mx = Math.max(...v), mu = ML.mean(v), sd = ML.std(v);
      mp.clear();
      mp.frame({ xticks: [-2, -1, 0, 1, 2, 3], yticks: [], xlabel: "scaled value", grid: true });
      const row = (y, vals, c, lab) => { mp.path([[-2.5, y], [3.2, y]], col.line, 1); vals.forEach((x, i) => mp.dot(x, y, 5, i === 10 ? col.bad : c, col.surface, 1)); mp.text(lab, 8, mp.Y(y), c); };
      row(2, v.map((x) => (x - mn) / (mx - mn)), col.c[0], "min-max [0, 1]");
      row(1, v.map((x) => (x - mu) / sd), col.c[1], "standardized");
      mp.text(MM.outlier ? "outlier 140 (red) squeezes min-max values near 0" : "click below to add an outlier", mp.W / 2 + 50, 16, col.muted, "center");
    };
    const mmBtn = ML.el("button", { type: "button", class: "btn ghost", text: "Add an outlier (140)", onclick: () => { MM.outlier = !MM.outlier; mmBtn.textContent = MM.outlier ? "Remove the outlier" : "Add an outlier (140)"; mp.drawFn(); } });
    root.querySelector("#mm-plot").append(ML.el("div", { class: "btn-row", style: "margin-top:8px" }, [mmBtn]));
    mp.drawFn();

    // ----- Gradient descent on the OLS example -----
    const gx = [1, 2, 3], gy = [2, 3, 5];
    const mse = (bb, ww) => ML.mean(gx.map((x, i) => (gy[i] - bb - ww * x) ** 2));
    const gh = root.querySelector("#gd-stage"), gpn = root.querySelector("#gd-panel");
    const gs = new ML.Stage3D(gh, { cam: [5, 4.6, 6], target: [0, 0.6, 0] });
    disposers.push(() => gs.dispose());
    const GB = [-3, 3.5], GW = [-1.5, 4], GH = 0.07, GMAX = 3;
    const toW = (bb, ww) => [bb - 0.25, ww - 1.25];
    let path = new THREE.Group();
    if (gs.ok) {
      ML.axes(gs, { x: [GB[0] - 0.25, GB[1] - 0.25], y: [GW[0] - 1.25, GW[1] - 1.25], labels: ["intercept b", "weight w", null], div: 8 });
      const N = 50, g = new THREE.PlaneGeometry(GB[1] - GB[0], GW[1] - GW[0], N, N);
      g.rotateX(-Math.PI / 2); g.translate(0, 0, 0);
      g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array((N + 1) ** 2 * 3), 3));
      const pos = g.attributes.position, cl = g.attributes.color, lo = new THREE.Color(col.c[1]), hi = new THREE.Color(col.c[2]), t = new THREE.Color();
      for (let v = 0; v < pos.count; v++) {
        const bb = pos.getX(v) + 0.25, ww = -pos.getZ(v) + 1.25, h = Math.min(mse(bb, ww) * GH, GMAX);
        pos.setY(v, h); t.copy(lo).lerp(hi, h / GMAX); cl.setXYZ(v, t.r, t.g, t.b);
      }
      g.computeVertexNormals();
      gs.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.8 })));
      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.14, 0.025, 8, 28), new THREE.MeshBasicMaterial({ color: col.ink }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(ML.P(...toW(1 / 3, 1.5), mse(1 / 3, 1.5) * GH + 0.02)); gs.add(ring);
      gs.add(path);
    }
    const GD = { lr: 0.05, steps: 40 };
    ML.ui.heading(gpn, "Gradient descent");
    ML.ui.slider(gpn, { label: "learning rate", min: 0.01, max: 0.2, step: 0.005, value: GD.lr, onInput: (v) => { GD.lr = v; runGD(); } });
    ML.ui.slider(gpn, { label: "steps", min: 1, max: 200, value: GD.steps, onInput: (v) => { GD.steps = v; runGD(); } });
    const gst2 = ML.ui.stats(gpn, [{ key: "b", label: "b" }, { key: "w", label: "w" }, { key: "m", label: "MSE", big: true }]);
    const gver = ML.ui.verdict(gpn);
    function runGD() {
      let bb = -2, ww = -1;
      const pts = [[bb, ww]];
      for (let k = 0; k < GD.steps; k++) {
        const db = ML.mean(gx.map((x, i) => -2 * (gy[i] - bb - ww * x))), dw = ML.mean(gx.map((x, i) => -2 * x * (gy[i] - bb - ww * x)));
        bb -= GD.lr * db; ww -= GD.lr * dw;
        if (!Number.isFinite(bb) || Math.abs(bb) > 1e6) break;
        pts.push([bb, ww]);
      }
      const m = mse(bb, ww);
      gst2.set("b", Number.isFinite(bb) ? bb.toFixed(3) : "∞"); gst2.set("w", Number.isFinite(ww) ? ww.toFixed(3) : "∞"); gst2.set("m", Number.isFinite(m) ? m.toFixed(4) : "diverged", Number.isFinite(m) && m < 1e3 ? null : col.bad);
      gver.innerHTML = !Number.isFinite(m) || m > 50 ? "The learning rate is too large: each step overshoots the valley and the error grows." : m < 0.112 ? "Reached the closed-form minimum (MSE ≈ 0.111)." : "Still walking downhill. Add steps or raise the learning rate a little.";
      if (!gs.ok) return;
      gs.clearGroup(path);
      const vis = pts.filter(([p, q]) => p >= GB[0] && p <= GB[1] && q >= GW[0] && q <= GW[1]).map(([p, q]) => ML.P(...toW(p, q), Math.min(mse(p, q) * GH, GMAX) + 0.03));
      if (vis.length > 1) path.add(ML.line(vis, new THREE.Color(col.ink)));
      const sp = new ML.Spheres(gs, vis.length, 0.04); vis.forEach((p, i) => sp.set(i, p, col.bad)); sp.commit();
      gs.scene.remove(sp.mesh); path.add(sp.mesh);
    }
    runGD();

    // ----- MAPE -----
    const mb = root.querySelector("#mape-box");
    const ys = [10, 8, 5];
    const mOut = ML.el("div", { class: "f-note mono" });
    ML.ui.slider(mb, { label: "smallest true value y₃", min: 0.05, max: 5, step: 0.05, value: 5, onInput: (v) => { ys[2] = v; mp2(); } });
    mb.append(mOut);
    function mp2() {
      const terms = ys.map((y) => Math.abs(1 / y));
      mOut.innerHTML = `y = (${ys.join(", ")}), ŷ = y + 1<br>MAPE = 100/3 · (${terms.map((v) => v.toFixed(3)).join(" + ")}) = <b style="color:${ys[2] < 0.5 ? "var(--bad)" : "var(--accent)"}">${((100 / 3) * terms.reduce((a2, b2) => a2 + b2, 0)).toFixed(1)}%</b>${ys[2] < 0.5 ? "<br>One target near zero dominates the whole metric." : ""}`;
    }
    mp2();
    return disposers;
  },
});
