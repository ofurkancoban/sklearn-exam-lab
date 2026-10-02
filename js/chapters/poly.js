// Section 8: polynomial regression, under- and overfitting, and the "it is still
// linear" view: with features (x, x²) the model is a flat plane in 3D.
ML.register("poly", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Give linear regression extra features such as x² and x³, so it can <b>fit a curve</b>. Degree 1 gives a line, degree 2 a quadratic curve, degree 3 a cubic curve.", ["sup", "reg", "degree"])}
  ${ML.formula("Degree 2", String.raw`\hat{y} = b + w_1 x + w_2 x^2`, "The curve can bend as x changes, but the prediction is still a <b>weighted sum of the features x and x²</b>. That is why we can use linear regression after creating polynomial features.", "Eq. 12")}

  <div class="section-title"><h2>Why it is still linear: look at it in 3D</h2><span class="hint">Treat x and x² as two separate features.</span></div>
  ${ML.bench("pl", { tag: "Features (x, x²) and target y", stage: "tall" })}
  <p class="fig-cap">Every training example is placed at (x, x², y). In this space the fitted model ŷ = b + w<sub>1</sub>x + w<sub>2</sub>x² is a <b>flat plane</b>, exactly like linear regression with two features. The examples lie on a curved ribbon because x² is tied to x. Click “Look along x²” to collapse the view to the familiar (x, y) plot: the line on the plane becomes the parabola.</p>

  <div class="section-title"><h2>Underfitting and overfitting</h2><span class="hint">Data follow y = x² + noise.</span></div>
  <div class="bench wide-panel">
    <div class="stage" id="pf-stage" style="background:var(--surface)"><span class="stage-tag">PolynomialFeatures(degree) + LinearRegression</span><div id="pf-plot" style="position:absolute;inset:34px 8px 8px 8px"></div></div>
    <div class="panel" id="pf-panel"></div>
  </div>
  <div id="pf-curve"></div>
  <div class="two-col">
    <div class="prose">
      <ul>
        <li><b>Underfitting</b>: the model is too simple to capture the pattern. A straight line cannot capture this U-shaped relationship. Training and test errors are both high.</li>
        <li><b>Overfitting</b>: the model fits training noise as well as the pattern. Its <b>training error looks good, but new-data error is worse</b>.</li>
      </ul>
    </div>
    ${ML.callout("setting", "Main setting: degree", "More flexibility is not automatically better. Select the degree using <b>training-only validation</b>, then check the chosen model on the final test set. A high degree alone does not prove overfitting (with 100 points, even degree 10 behaves reasonably; switch to 15 points to see it break).")}
  </div>

  <div class="section-title"><h2>A complete example</h2></div>
  <div class="two-col">
    ${ML.code(`
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + rng.normal(0, 1, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
model.fit(X_train, y_train)
print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))

plt.scatter(X_train[:, 0], y_train, label="Training examples")
plt.scatter(X_test[:, 0], y_test, label="Test examples")
plt.plot(X[:, 0], model.predict(X), color="black", label="Model")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()`, `MAE: 0.7146419232499187`)}
    <div class="prose">
      <ul>
        <li><code>reshape(-1, 1)</code> makes a table with <b>one column</b>, as scikit-learn expects.</li>
        <li><code>X[:, 0]</code> selects that column as a one-dimensional array.</li>
        <li><code>include_bias=False</code> avoids adding a constant feature, because <code>LinearRegression</code> already fits an intercept.</li>
        <li>For degree 2, <code>PolynomialFeatures</code> turns each x into the two columns (x, x²).</li>
      </ul>
      ${ML.callout("trap", "Exam traps", "<ul><li>Polynomial regression <b>is linear regression</b> on extra features; “linear” refers to the weights.</li><li>Low training error alone says nothing about new data.</li><li>Choose the degree with validation, not with the test set.</li></ul>")}
    </div>
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="poly"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const fitPoly = (xs, ys, d) => ML.lstsq(xs.map((x) => Array.from({ length: d + 1 }, (_, k) => Math.pow(x / 3, k))), ys, 1e-9);
    const predPoly = (w, x) => w.reduce((s, wk, k) => s + wk * Math.pow(x / 3, k), 0);

    // ----- 3D lift view -----
    const r = ML.rng(42), N3 = 60;
    const xs3 = Array.from({ length: N3 }, (_, i) => -3 + (6 * i) / (N3 - 1));
    const ys3 = xs3.map((x) => x * x + ML.randn(r));
    const w3 = ML.lstsq(xs3.map((x) => [1, x, x * x]), ys3);
    const W = (x, x2, yv) => ML.P(x * 0.7, x2 * 0.42 - 1.9, yv * 0.32);
    const host = root.querySelector("#pl-stage"), panel = root.querySelector("#pl-panel");
    const st = new ML.Stage3D(host, { cam: [6.5, 3.6, 6.2], target: [0, 1.4, 0] });
    disposers.push(() => st.dispose());
    if (st.ok) {
      ML.axes(st, { x: [-2.2, 2.2], y: [-1.9, 1.95], z: [-0.8, 3.8], labels: ["feature x", "feature x²", "target y"], div: 6 });
      const G = 24, pos = [], idx = [];
      for (let j = 0; j <= G; j++) for (let i = 0; i <= G; i++) {
        const x = -3 + (6 * i) / G, x2 = (9 * j) / G;
        const p = W(x, x2, w3[0] + w3[1] * x + w3[2] * x2);
        pos.push(p.x, p.y, p.z);
      }
      for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const a = j * (G + 1) + i; idx.push(a, a + 1, a + G + 1, a + 1, a + G + 2, a + G + 1); }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
      g.setIndex(idx); g.computeVertexNormals();
      const plane = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: col.accent, transparent: true, opacity: 0.26, side: THREE.DoubleSide, depthWrite: false }));
      st.add(plane);
      st.add(new THREE.LineSegments(new THREE.WireframeGeometry(g), new THREE.LineBasicMaterial({ color: col.accent, transparent: true, opacity: 0.12 })));
      // Fitted curve on the plane and its shadow on the floor (the ribbon x² = x·x).
      const curve = [], shadow = [];
      for (let i = 0; i <= 120; i++) { const x = -3 + (6 * i) / 120; curve.push(W(x, x * x, w3[0] + w3[1] * x + w3[2] * x * x)); shadow.push(W(x, x * x, -2.5)); }
      st.add(ML.line(curve, new THREE.Color(col.ink)));
      st.add(ML.line(shadow, new THREE.Color(col.muted), { dashed: true }));
      const sph = new ML.Spheres(st, N3, 0.07);
      xs3.forEach((x, i) => sph.set(i, W(x, x * x, ys3[i]), col.c[0]));
      sph.commit();
      st.views([{ t: "3D", pos: [6.5, 3.6, 6.2], target: [0, 1.4, 0] }]);
      ML.legend(host, [{ label: "example (x, x², y)", color: col.c[0] }, { label: "fitted plane", color: col.accent }]);
    }
    ML.ui.heading(panel, "Change the view");
    const row = ML.ui.row(panel);
    ML.ui.button(row, "Look along x²", () => st.ok && st.flyTo([0.01, 1.4, 9.5], [0, 1.4, 0]));
    ML.ui.button(row, "Look along x", () => st.ok && st.flyTo([9.5, 1.4, 0.01], [0, 1.4, 0]), true);
    ML.ui.button(row, "Top", () => st.ok && st.flyTo([0, 10, 0.01], [0, 0, 0]), true);
    panel.append(ML.el("div", { class: "verdict", html: `Least-squares weights: <b>b = ${w3[0].toFixed(2)}</b>, <b>w<sub>1</sub> = ${w3[1].toFixed(2)}</b>, <b>w<sub>2</sub> = ${w3[2].toFixed(2)}</b>. The true pattern is y = x², i.e. b = 0, w<sub>1</sub> = 0, w<sub>2</sub> = 1.` }));
    panel.append(ML.el("div", { class: "ctrl-note", html: "Looking along x² hides the x² axis, so you see y against x: a parabola. Looking along x you see y against x²: a straight line, because y depends linearly on the feature x²." }));

    // ----- Degree explorer -----
    const P = { deg: 2, n: 100 };
    const pp = root.querySelector("#pf-panel");
    const plot = new ML.Plot2D(root.querySelector("#pf-plot"), { height: 400, xlim: [-3.2, 3.2], ylim: [-3, 12], aria: "Polynomial fit" });
    const curvePlot = new ML.Plot2D(root.querySelector("#pf-curve"), { height: 220, xlim: [0.5, 15.5], ylim: [0, 3], aria: "Error versus degree" });
    disposers.push(() => { plot.dispose(); curvePlot.dispose(); });
    let data;
    const makeData = () => {
      const rr = ML.rng(7 + P.n);
      const xs = Array.from({ length: P.n }, (_, i) => -3 + (6 * i) / (P.n - 1));
      const ys = xs.map((x) => x * x + ML.randn(rr));
      const order = ML.shuffleIdx(P.n, ML.rng(42));
      const nTest = Math.round(P.n * 0.2);
      const te = order.slice(0, nTest), tr = order.slice(nTest);
      data = { xs, ys, tr, te };
      data.errs = Array.from({ length: 15 }, (_, k) => errs(k + 1));
    };
    const mae = (w, ids) => ML.mean(ids.map((i) => Math.abs(data.ys[i] - predPoly(w, data.xs[i]))));
    function errs(d) {
      const w = fitPoly(data.tr.map((i) => data.xs[i]), data.tr.map((i) => data.ys[i]), d);
      return { w, tr: mae(w, data.tr), te: mae(w, data.te) };
    }
    ML.ui.heading(pp, "Model");
    ML.ui.slider(pp, { label: "degree", min: 1, max: 15, value: P.deg, onInput: (v) => { P.deg = v; drawAll(); } });
    ML.ui.seg(pp, { label: "Number of examples", options: [{ v: 100, t: "100 (script)" }, { v: 15, t: "15 (few)" }], value: 100, onChange: (v) => { P.n = v; makeData(); drawAll(); } });
    const stats = ML.ui.stats(pp, [{ key: "tr", label: "Training MAE" }, { key: "te", label: "Test MAE", big: true }]);
    const verdict = ML.ui.verdict(pp);
    function drawAll() {
      const e = data.errs[P.deg - 1];
      stats.set("tr", e.tr.toFixed(3)); stats.set("te", e.te.toFixed(3));
      const best = data.errs.reduce((b, v, i) => (v.te < data.errs[b].te ? i : b), 0) + 1;
      verdict.innerHTML = P.deg === 1 ? "<b>Underfitting.</b> A straight line misses the U shape: both errors are high."
        : e.te > 1.6 * data.errs[1].te && P.deg > 3 ? "<b>Overfitting.</b> Training error keeps falling, but the curve wiggles between points and the test error grows."
        : P.deg === 2 ? "Degree 2 matches the true pattern y = x². Remaining error is the noise."
        : `Training error is small; test error ${e.te < data.errs[1].te * 1.15 ? "is still close to degree 2" : "is starting to grow"}. Lowest test error here: degree ${best}.`;
      // Main fit plot.
      plot.clear();
      plot.frame({ xticks: [-3, -2, -1, 0, 1, 2, 3], yticks: [0, 4, 8, 12], xlabel: "x", ylabel: "y" });
      plot.clip(() => {
        const pts = Array.from({ length: 400 }, (_, i) => { const x = -3.1 + (6.2 * i) / 399; return [x, predPoly(e.w, x)]; });
        plot.path(pts.map(([x, v]) => [x, ML.clamp(v, -50, 60)]), col.ink, 2.4);
        plot.path(pts.map(([x]) => [x, x * x]), col.muted, 1.2, [4, 5]);
      });
      data.tr.forEach((i) => plot.dot(data.xs[i], data.ys[i], 4, col.c[1], col.surface, 1));
      data.te.forEach((i) => plot.dot(data.xs[i], data.ys[i], 4.5, col.c[2], col.ink, 1));
      plot.text("● training", plot.W - 120, 22, col.c[1]);
      plot.text("● test", plot.W - 120, 40, col.c[2] === "#fde725" ? col.c[2] : col.warn);
      plot.text("- - true y = x²", plot.W - 120, 58, col.muted);
      plot.text(`degree ${P.deg}`, 60, 22, col.ink, "left", '600 14px "IBM Plex Mono", monospace');
      // Error vs degree.
      const ymax = Math.min(4, Math.max(...data.errs.map((v) => v.te)) * 1.1);
      curvePlot.o.ylim = [0, Math.max(1.5, ymax)];
      curvePlot.clear();
      curvePlot.frame({ xticks: [1, 3, 5, 7, 9, 11, 13, 15], yticks: ML.ticks(0, curvePlot.o.ylim[1], 4), xlabel: "degree", ylabel: "MAE" });
      curvePlot.path(data.errs.map((v, i) => [i + 1, v.tr]), col.c[1], 2.2);
      curvePlot.path(data.errs.map((v, i) => [i + 1, Math.min(v.te, curvePlot.o.ylim[1])]), col.warn, 2.2);
      data.errs.forEach((v, i) => { curvePlot.dot(i + 1, v.tr, 3, col.c[1]); curvePlot.dot(i + 1, Math.min(v.te, curvePlot.o.ylim[1]), 3, col.warn); });
      curvePlot.path([[P.deg, 0], [P.deg, curvePlot.o.ylim[1]]], col.ink, 1.4, [5, 4]);
      curvePlot.text("training MAE", curvePlot.W - 24, 22, col.c[1], "right");
      curvePlot.text("test MAE", curvePlot.W - 24, 40, col.warn, "right");
    }
    plot.drawFn = drawAll; curvePlot.drawFn = drawAll;
    makeData();
    drawAll();
    return disposers;
  },
});
