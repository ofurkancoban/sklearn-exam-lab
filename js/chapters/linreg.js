// Section 6: linear regression as a plane in 3D, with live error metrics.
ML.register("linreg", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Linear regression <b>adds up weighted features to predict a number</b>. With one feature it is a straight line; with two features it is a flat plane.", ["sup", "reg", "LinearRegression"])}
  <div class="two-col">
    ${ML.formula("Prediction", String.raw`\hat{y} = b + \sum_{j=1}^{p} w_j x_j`, "The intercept b is the starting value; each weight w<sub>j</sub> multiplies feature x<sub>j</sub>. With one feature, ŷ = b + w<sub>1</sub>x<sub>1</sub> is a straight line. <b>Training chooses weights and an intercept that minimize mean squared error.</b>", "Eq. 9")}
    ${ML.formula("Prediction errors", String.raw`e_i = y_i - \hat{y}_i,\quad \text{MAE} = \frac1n\sum_{i=1}^n |e_i|,\quad \text{MSE} = \frac1n\sum_{i=1}^n e_i^2,\quad \text{RMSE} = \sqrt{\text{MSE}}`, "MAE and RMSE use the target's units; <b>lower is better</b>. RMSE gives large errors more weight.", "Eq. 10")}
  </div>

  <div class="section-title"><h2>Fit the plane yourself</h2><span class="hint">Move the weights and watch the errors. Then let least squares do it.</span></div>
  ${ML.bench("lr", { tag: "ŷ = b + w1·x1 + w2·x2 over 40 houses", stage: "tall" })}
  <p class="fig-cap">Each vertical stick is one error e<sub>i</sub> = y<sub>i</sub> − ŷ<sub>i</sub> (blue above the plane, orange below). Switch on the squares to see what MSE literally averages: the area of a square with side |e<sub>i</sub>|. Least squares makes the total area as small as possible.</p>

  <div class="two-col">
    ${ML.formula("Coefficient of determination", String.raw`R^2 = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}, \qquad \bar{y} = \frac1n \sum_{i=1}^n y_i`, "Denominator: total variation around the mean. Numerator: unexplained variation (squared prediction errors). If these sums are 100 and 20, R² = 1 − 20/100 = 0.80: the model explains 80% of the variation. <b>Negative R²</b>: the squared errors exceed the total variation, worse than always predicting ȳ.")}
    <div class="formula">
      <div class="f-label"><span>Try it: the script's worked example</span></div>
      <div class="btn-row mono" style="align-items:center;font-size:14px">
        y = (<input id="lr-y1" type="number" value="3" style="width:60px">, <input id="lr-y2" type="number" value="5" style="width:60px">)
        ŷ = (<input id="lr-p1" type="number" value="4" style="width:60px">, <input id="lr-p2" type="number" value="7" style="width:60px">)
      </div>
      <div id="lr-calc" class="f-note mono"></div>
    </div>
  </div>

  <div class="section-title"><h2>A complete example: Diabetes</h2></div>
  <div class="two-col">
    ${ML.code(`
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions) ** 0.5)
print("R2:", r2_score(y_test, predictions))`, `
MAE: 42.79409467959994
RMSE: 53.85344583676592
R2: 0.4526027629719196`)}
    <div class="prose">
      <p>The Diabetes dataset has 442 patients, 10 features (age, sex, bmi, blood pressure, six blood measurements) and a numeric target: disease progression after one year.</p>
      <p>There is no <code>stratify=y</code> here: stratifying needs categories, and the target is a number.</p>
      <p>Reading the output: predictions are off by about 43 units on average (MAE). RMSE (54) is larger than MAE because a few large errors weigh more. R² ≈ 0.45: the model explains about 45% of the variation in the test targets.</p>
      ${ML.callout("setting", "Main setting", "Usually the defaults. Linear regression has no important hyperparameter to tune.")}
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>e<sub>i</sub> = y<sub>i</sub> − ŷ<sub>i</sub> (true minus predicted). MAE ignores the sign with |e<sub>i</sub>|.</li><li>RMSE ≥ MAE always; a big gap means some large errors.</li><li>R² can be negative on test data.</li><li>Linear regression cannot capture curved relationships (see polynomial regression).</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="linreg"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // Worked example calculator.
    const ins = ["y1", "y2", "p1", "p2"].map((k) => root.querySelector(`#lr-${k}`));
    const calc = () => {
      const [y1, y2, p1, p2] = ins.map((i) => parseFloat(i.value) || 0);
      const e1 = y1 - p1, e2 = y2 - p2;
      const mae = (Math.abs(e1) + Math.abs(e2)) / 2, mse = (e1 * e1 + e2 * e2) / 2;
      root.querySelector("#lr-calc").innerHTML = `errors e = (${e1}, ${e2})<br>MAE = (${Math.abs(e1)} + ${Math.abs(e2)}) / 2 = <b style="color:var(--accent)">${+mae.toFixed(3)}</b><br>MSE = (${e1 * e1} + ${e2 * e2}) / 2 = ${+mse.toFixed(3)}<br>RMSE = √${+mse.toFixed(3)} ≈ <b style="color:var(--accent)">${Math.sqrt(mse).toFixed(2)}</b>`;
    };
    ins.forEach((i) => i.addEventListener("input", calc));
    calc();

    // Data: energy use from house size and outside temperature (both scaled).
    const r = ML.rng(42), n = 40, X = [], y = [];
    for (let i = 0; i < n; i++) {
      const a = r() * 4 - 2, b = r() * 4 - 2;
      X.push([a, b]); y.push(2 + 0.9 * a - 0.6 * b + 0.45 * ML.randn(r));
    }
    const ols = ML.lstsq(X.map((x) => [1, x[0], x[1]]), y);
    const S = { b: 1, w1: 0, w2: 0, squares: false };
    const host = root.querySelector("#lr-stage"), panel = root.querySelector("#lr-panel");
    const st = new ML.Stage3D(host, { cam: [6.4, 4.2, 7], target: [0, 1.6, 0] });
    disposers.push(() => st.dispose());
    const Z0 = -1.5;
    let plane, sph, res = new THREE.Group();
    if (st.ok) {
      ML.axes(st, { x: [-2.2, 2.2], y: [-2.2, 2.2], z: [Z0, 5.5], labels: ["x1: house size (scaled)", "x2: outside temp. (scaled)", "y: energy used"], div: 6, ticks: { z: [0, 2, 4] } });
      const g = new THREE.PlaneGeometry(4.4, 4.4, 1, 1);
      g.rotateX(-Math.PI / 2);
      plane = new THREE.Mesh(g, new THREE.MeshStandardMaterial({ color: col.accent, transparent: true, opacity: 0.32, side: THREE.DoubleSide, depthWrite: false }));
      st.add(plane);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineBasicMaterial({ color: col.accent }));
      plane.add(edges);
      sph = new ML.Spheres(st, n, 0.075);
      st.add(res);
      st.views([{ t: "3D", pos: [6.4, 4.2, 7], target: [0, 1.6, 0] }, { t: "Along x2", pos: [0.01, 1.6, 9], target: [0, 1.6, 0] }, { t: "Along x1", pos: [9, 1.6, 0.01], target: [0, 1.6, 0] }]);
      ML.legend(host, [{ label: "house", color: col.c[0] }, { label: "e > 0", color: col.c[1] }, { label: "e < 0", color: col.warn }]);
    }
    ML.ui.heading(panel, "Your model");
    const sb = ML.ui.slider(panel, { label: "b (intercept)", min: -2, max: 5, step: 0.01, value: S.b, onInput: (v) => { S.b = v; upd(); } });
    const s1 = ML.ui.slider(panel, { label: "w1 (house size)", min: -2, max: 2, step: 0.01, value: S.w1, onInput: (v) => { S.w1 = v; upd(); } });
    const s2 = ML.ui.slider(panel, { label: "w2 (temperature)", min: -2, max: 2, step: 0.01, value: S.w2, onInput: (v) => { S.w2 = v; upd(); } });
    ML.ui.check(panel, { label: "Show squared errors as squares", onChange: (v) => { S.squares = v; upd(); } });
    const row = ML.ui.row(panel);
    let anim = null;
    const tween = (to) => {
      cancelAnimationFrame(anim);
      const from = { ...S }, t0 = performance.now();
      const go = (t) => {
        const k = ML.reducedMotion() ? 1 : ML.ease(Math.min(1, (t - t0) / 1100));
        S.b = ML.lerp(from.b, to[0], k); S.w1 = ML.lerp(from.w1, to[1], k); S.w2 = ML.lerp(from.w2, to[2], k);
        sb.set(+S.b.toFixed(2)); s1.set(+S.w1.toFixed(2)); s2.set(+S.w2.toFixed(2));
        upd();
        if (k < 1) anim = requestAnimationFrame(go);
      };
      anim = requestAnimationFrame(go);
    };
    disposers.push(() => cancelAnimationFrame(anim));
    ML.ui.button(row, "Fit (least squares)", () => tween(ols));
    ML.ui.button(row, "Predict the mean only", () => tween([ML.mean(y), 0, 0]), true);
    ML.ui.heading(panel, "Errors on these 40 houses");
    const stats = ML.ui.stats(panel, [{ key: "mae", label: "MAE" }, { key: "mse", label: "MSE" }, { key: "rmse", label: "RMSE" }, { key: "r2", label: "R²", big: true }]);
    const verdict = ML.ui.verdict(panel);
    function upd() {
      const yh = X.map((x) => S.b + S.w1 * x[0] + S.w2 * x[1]);
      const e = y.map((v, i) => v - yh[i]);
      const mae = ML.mean(e.map(Math.abs)), mse = ML.mean(e.map((v) => v * v));
      const ybar = ML.mean(y), sst = y.reduce((s, v) => s + (v - ybar) ** 2, 0), sse = e.reduce((s, v) => s + v * v, 0);
      const r2 = 1 - sse / sst;
      stats.set("mae", mae.toFixed(3)); stats.set("mse", mse.toFixed(3)); stats.set("rmse", Math.sqrt(mse).toFixed(3));
      stats.set("r2", r2.toFixed(3), r2 < 0 ? col.bad : null);
      const best = Math.abs(S.b - ols[0]) < 0.02 && Math.abs(S.w1 - ols[1]) < 0.02 && Math.abs(S.w2 - ols[2]) < 0.02;
      verdict.innerHTML = best
        ? `Least-squares solution: b = ${ols[0].toFixed(2)}, w1 = ${ols[1].toFixed(2)}, w2 = ${ols[2].toFixed(2)}. No other plane has a smaller MSE.`
        : r2 < 0
          ? `R² is negative: Σ(y − ŷ)² = ${sse.toFixed(1)} is larger than Σ(y − ȳ)² = ${sst.toFixed(1)}. Predicting the mean would be better.`
          : `Σ(y − ŷ)² = ${sse.toFixed(1)}, Σ(y − ȳ)² = ${sst.toFixed(1)}, so R² = 1 − ${sse.toFixed(1)}/${sst.toFixed(1)} = ${r2.toFixed(3)}.`;
      if (!st.ok) return;
      const pos = plane.geometry.attributes.position;
      for (let v = 0; v < pos.count; v++) { const a = pos.getX(v), b = -pos.getZ(v); pos.setY(v, S.b + S.w1 * a + S.w2 * b); }
      pos.needsUpdate = true; plane.geometry.computeVertexNormals();
      plane.children[0].geometry.dispose();
      plane.children[0].geometry = new THREE.EdgesGeometry(plane.geometry);
      st.clearGroup(res);
      X.forEach((x, i) => {
        sph.set(i, ML.P(x[0], x[1], y[i]), col.c[0]);
        const c = new THREE.Color(e[i] > 0 ? col.c[1] : col.warn);
        res.add(ML.line([ML.P(x[0], x[1], y[i]), ML.P(x[0], x[1], yh[i])], c));
        if (S.squares) {
          const s = Math.abs(e[i]);
          const sq = new THREE.Mesh(new THREE.PlaneGeometry(s, s), new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.18, side: THREE.DoubleSide, depthWrite: false }));
          sq.position.copy(ML.P(x[0] + s / 2, x[1], (y[i] + yh[i]) / 2));
          res.add(sq);
        }
      });
      sph.commit();
    }
    upd();
    return disposers;
  },
});
