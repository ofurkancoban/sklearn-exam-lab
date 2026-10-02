// Section 7: Lasso. The 3D view is the training objective over two weights;
// the path chart uses the real (standardized) Diabetes training split.
ML.lassoCD = (X, y, alpha, w0 = null, iters = 300, tol = 1e-7) => {
  const n = X.length, p = X[0].length;
  const w = w0 ? w0.slice() : new Array(p).fill(0);
  const ym = ML.mean(y);
  const yc = y.map((v) => v - ym);
  const z = Array.from({ length: p }, (_, j) => X.reduce((s, r) => s + r[j] * r[j], 0) / n);
  const resid = yc.map((v, i) => v - X[i].reduce((s, x, j) => s + x * w[j], 0));
  for (let it = 0; it < iters; it++) {
    let maxd = 0;
    for (let j = 0; j < p; j++) {
      let rho = 0;
      for (let i = 0; i < n; i++) rho += X[i][j] * (resid[i] + X[i][j] * w[j]);
      rho /= n;
      const nw = Math.sign(rho) * Math.max(Math.abs(rho) - alpha, 0) / z[j];
      const d = nw - w[j];
      if (d !== 0) { for (let i = 0; i < n; i++) resid[i] -= X[i][j] * d; w[j] = nw; }
      maxd = Math.max(maxd, Math.abs(d));
    }
    if (maxd < tol) break;
  }
  return w;
};

ML.register("lasso", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Lasso <b>penalizes large weights and can set some weights to exactly zero</b>. A zero weight removes that feature from the prediction. The penalty is called <b>regularization</b>.", ["sup", "reg", "alpha"])}
  ${ML.formula("Balance fit and simplicity", String.raw`\underbrace{\frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}_{\text{prediction error}} \;+\; \underbrace{\alpha \sum_{j=1}^{p} |w_j|}_{\text{penalty on weights}}`, "Same prediction ŷ<sub>i</sub> = b + Σ w<sub>j</sub>x<sub>ij</sub> as linear regression, different training objective. The first term rewards accurate predictions, the second rewards small weights, <b>α controls its strength</b>. The 1/(2n) convention matches scikit-learn (half the training MSE). <b>The intercept b is not penalized.</b>", "Eq. 11")}

  <div class="section-title"><h2>The objective as a landscape</h2><span class="hint">Height = objective value for every pair of weights (w1, w2).</span></div>
  ${ML.bench("la", { tag: "Lasso objective over two weights", stage: "tall" })}
  <p class="fig-cap">At α = 0 the surface is a smooth bowl and its lowest point (hollow ring) is the ordinary least-squares solution. Increasing α adds the term α(|w1| + |w2|), which has a sharp crease along the axes. The minimum (solid ball) slides toward the axes and then <b>sticks exactly on w2 = 0</b>: the weaker feature is dropped. The dotted trail is the full path of the minimum.</p>

  <div class="section-title"><h2>The real Diabetes weights</h2><span class="hint">10 standardized features, Lasso fitted for every α.</span></div>
  <div class="bench wide-panel">
    <div class="stage" id="lp-stage" style="background:var(--surface)"><span class="stage-tag">Coefficient path (scikit-learn objective, solved in the browser)</span><div id="lp-plot" style="position:absolute;inset:34px 8px 8px 8px"></div></div>
    <div class="panel" id="lp-panel"></div>
  </div>

  <div class="two-col">
    ${ML.code(`
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Reuse the Diabetes split and metric import from linear regression
model = make_pipeline(StandardScaler(), Lasso(alpha=1.0))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Lasso MAE:", mean_absolute_error(y_test, predictions))
print("Weights:", model.named_steps["lasso"].coef_)`, `
Lasso MAE: 42.80298373195777
Weights: [  0.687  -9.298  26.219  15.657  -8.228  -0.     -9.024   3.421  22.636   2.099]
(weights rounded to 3 decimals here)`)}
    <div class="prose">
      <p><b>Scale first.</b> Weights depend on feature units, so without scaling the penalty would treat features unfairly. That is why Lasso sits in a pipeline after <code>StandardScaler</code>.</p>
      <p><code>named_steps["lasso"]</code> reaches into the pipeline: <code>make_pipeline</code> names each step after its lowercase class name.</p>
      <p>At α = 1.0, one weight (s2) is exactly zero, so that feature contributes nothing to this model's prediction.</p>
      ${ML.callout("setting", "Main setting: alpha", "Larger values encourage smaller weights and often more zero weights. Too much regularization makes the model too simple. <b>Choose α with cross-validation.</b>")}
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>A zero weight means the feature contributes nothing <b>to this model</b>. It does not prove the feature is irrelevant in every setting.</li><li>The intercept is not penalized.</li><li>Linear regression and Lasso can both miss curved relationships.</li><li>α = 0 gives ordinary linear regression.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="lasso"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // ----- Two-weight landscape -----
    const r = ML.rng(5), n = 120, X = [], y = [];
    for (let i = 0; i < n; i++) {
      const a = ML.randn(r), b = 0.35 * a + 0.94 * ML.randn(r);
      X.push([a, b]); y.push(2.6 * a + 0.7 * b + 0.9 * ML.randn(r));
    }
    const sc = ML.scaler(X), Xs = sc.transform(X);
    const ym = ML.mean(y), yc = y.map((v) => v - ym);
    const J = (w, al) => { let s = 0; for (let i = 0; i < n; i++) { const e = yc[i] - Xs[i][0] * w[0] - Xs[i][1] * w[1]; s += e * e; } return s / (2 * n) + al * (Math.abs(w[0]) + Math.abs(w[1])); };
    const W1 = [-1, 4], W2 = [-1.5, 2.5], G = 70, HS = 0.4, HMAX = 2.2;
    const toW = (w1, w2) => [(w1 - 1.5) * 1.0, (w2 - 0.5) * 1.0];
    const host = root.querySelector("#la-stage"), panel = root.querySelector("#la-panel");
    const st = new ML.Stage3D(host, { cam: [6.2, 5.4, 7.2], target: [0, 0.5, 0] });
    disposers.push(() => st.dispose());
    const path = [];
    let w = [0, 0];
    for (let k = 0; k <= 140; k++) { const al = (k / 140) * 3.5; w = ML.lassoCD(Xs, y, al, w); path.push({ al, w: w.slice() }); }
    let surf, ball, ballFloor, ring, cont = new THREE.Group();
    const S = { al: 0 };
    if (st.ok) {
      const [x0, y0] = toW(W1[0], W2[0]), [x1, y1] = toW(W1[1], W2[1]);
      ML.axes(st, { x: [x0, x1], y: [y0, y1], labels: ["weight w1 (strong feature)", "weight w2 (weak feature)", null], div: 10 });
      // Highlight the w1 = 0 and w2 = 0 lines on the floor.
      const [ax, ay] = toW(0, 0);
      st.add(ML.line([ML.P(ax, y0, 0.01), ML.P(ax, y1, 0.01)], new THREE.Color(col.bad), { opacity: 0.8 }));
      st.add(ML.line([ML.P(x0, ay, 0.01), ML.P(x1, ay, 0.01)], new THREE.Color(col.bad), { opacity: 0.8 }));
      const l1 = ML.label("w1 = 0", { h: 0.2, color: col.bad }); l1.position.copy(ML.P(ax, y1 + 0.2, 0.05)); st.add(l1);
      const l2 = ML.label("w2 = 0", { h: 0.2, color: col.bad }); l2.position.copy(ML.P(x1 + 0.35, ay, 0.05)); st.add(l2);
      const g = new THREE.PlaneGeometry(x1 - x0, y1 - y0, G, G);
      g.rotateX(-Math.PI / 2);
      g.translate((x0 + x1) / 2, 0, -(y0 + y1) / 2);
      g.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array((G + 1) * (G + 1) * 3), 3));
      surf = new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.88 }));
      st.add(surf);
      st.add(cont);
      ball = new THREE.Mesh(new THREE.SphereGeometry(0.11, 24, 18), new THREE.MeshStandardMaterial({ color: col.accent }));
      ballFloor = new THREE.Mesh(new THREE.SphereGeometry(0.07, 16, 12), new THREE.MeshStandardMaterial({ color: col.accent }));
      st.add(ball); st.add(ballFloor);
      const [ox, oy] = toW(...path[0].w);
      ring = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.02, 8, 30), new THREE.MeshBasicMaterial({ color: col.ink }));
      ring.rotation.x = -Math.PI / 2; ring.position.copy(ML.P(ox, oy, 0.02)); st.add(ring);
      const trail = path.filter((_, i) => i % 2 === 0).map((p) => { const [a, b] = toW(...p.w); return ML.P(a, b, 0.015); });
      st.add(ML.line(trail, new THREE.Color(col.accent), { dashed: true, dash: 0.06, gap: 0.05 }));
    }
    ML.ui.heading(panel, "Regularization");
    ML.ui.slider(panel, { label: "alpha (α)", min: 0, max: 3.5, step: 0.01, value: 0, onInput: (v) => { S.al = v; upd(); } });
    const stats = ML.ui.stats(panel, [{ key: "w1", label: "w1 (strong feature)" }, { key: "w2", label: "w2 (weak feature)" }, { key: "z", label: "Zero weights", big: true }, { key: "J", label: "Objective at minimum" }]);
    const verdict = ML.ui.verdict(panel);
    const lowC = new THREE.Color(col.c[1]), highC = new THREE.Color(col.c[2]), tmp = new THREE.Color();
    function upd() {
      const k = Math.round((S.al / 3.5) * 140), wm = path[k].w;
      stats.set("w1", wm[0].toFixed(3), Math.abs(wm[0]) < 1e-9 ? col.bad : null);
      stats.set("w2", wm[1].toFixed(3), Math.abs(wm[1]) < 1e-9 ? col.bad : null);
      const zeros = wm.filter((v) => Math.abs(v) < 1e-9).length;
      stats.set("z", `${zeros} of 2`);
      const jmin = J(wm, S.al);
      stats.set("J", jmin.toFixed(3));
      verdict.innerHTML = S.al === 0 ? "α = 0: plain least squares. Both weights are non-zero."
        : zeros === 0 ? "Both weights shrink toward zero, but neither is zero yet."
        : zeros === 1 ? "<b>w2 is exactly zero.</b> The minimum sits in the crease on the w2 = 0 line. The weak feature is removed from the model."
        : "Both weights are zero: the model predicts the mean of y for everyone. Too much regularization.";
      if (!st.ok) return;
      const g = surf.geometry, pos = g.attributes.position, cl = g.attributes.color;
      const grid = [];
      for (let v = 0; v < pos.count; v++) {
        const wx = pos.getX(v) + 1.5, wy = -pos.getZ(v) + 0.5;
        const h = Math.min((J([wx, wy], S.al) - jmin) * HS, HMAX);
        pos.setY(v, h + 0.02);
        tmp.copy(lowC).lerp(highC, h / HMAX);
        cl.setXYZ(v, tmp.r, tmp.g, tmp.b);
        const ix = v % (G + 1), iz = Math.floor(v / (G + 1));
        (grid[iz] = grid[iz] || [])[ix] = h;
      }
      pos.needsUpdate = true; cl.needsUpdate = true; g.computeVertexNormals();
      // Floor contours.
      st.clearGroup(cont);
      const [x0, y0] = toW(W1[0], W2[0]), [x1, y1] = toW(W1[1], W2[1]);
      const dx = (x1 - x0) / G, dz = (y1 - y0) / G;
      [0.1, 0.3, 0.6, 1, 1.5, 2.1, 2.8].forEach((lv) => {
        const segs = [];
        for (let iz = 0; iz < G; iz++) for (let ix = 0; ix < G; ix++) {
          const c = [[ix, iz], [ix + 1, iz], [ix + 1, iz + 1], [ix, iz + 1]], vals = c.map(([a, b]) => grid[b][a] - lv), cr = [];
          for (let e = 0; e < 4; e++) {
            const a = vals[e], b = vals[(e + 1) % 4];
            if ((a > 0) !== (b > 0)) { const t = a / (a - b), p0 = c[e], p1 = c[(e + 1) % 4]; cr.push(new THREE.Vector3(x0 + (p0[0] + (p1[0] - p0[0]) * t) * dx, 0.012, -y1 + (p0[1] + (p1[1] - p0[1]) * t) * dz)); }
          }
          if (cr.length >= 2) segs.push([cr[0], cr[1]]);
        }
        cont.add(ML.segments(segs, new THREE.Color(col.muted), 0.55));
      });
      const [bx, by] = toW(...wm);
      ball.position.copy(ML.P(bx, by, 0.13));
      ballFloor.position.copy(ML.P(bx, by, 0.02));
    }
    upd();

    // ----- Step-by-step tour -----
    const drv = ML.drive(panel);
    const zeroAt = (j) => { const p = path.find((q) => Math.abs(q.w[j]) < 1e-9 && q.al > 0); return p ? p.al : 3.5; };
    const a2 = zeroAt(1), a1 = zeroAt(0);
    const wAt = (al) => path[Math.round((al / 3.5) * 140)].w;
    const ballCam = () => { const p = ball.position; return [[p.x + 2.6, p.y + 2.6, p.z + 2.8], [p.x, p.y, p.z]]; };
    const CAM3 = [[6.2, 5.4, 7.2], [0, 0.5, 0]];
    const tour = ML.tour(host.parentElement, st, [
      { t: "Every point on the floor is one model", cam: CAM3,
        run: () => { drv.slider("alpha", 0); },
        d: "The floor shows all possible pairs of weights (w1, w2) for two standardized features. The height above each pair is the training objective: how bad that model is. Training means finding the <b>lowest point</b>." },
      { t: "α = 0: a smooth bowl (linear regression)", cam: ballCam,
        run: () => { drv.slider("alpha", 0); },
        d: () => `Without a penalty the objective is half the training MSE, a smooth bowl. Its lowest point (ball inside the black ring) is the ordinary least-squares solution: w1 = ${path[0].w[0].toFixed(2)}, w2 = ${path[0].w[1].toFixed(2)}. Feature 1 is strong, feature 2 is weak.` },
      { t: "Adding the penalty α Σ|wⱼ|", cam: [[0.01, 9, 0.6], [0, 0, 0]],
        run: () => { drv.slider("alpha", 0.5); },
        d: () => `The L1 penalty \\(\\alpha(|w_1| + |w_2|)\\) grows linearly away from zero and has a sharp <b>crease along the red axis lines</b> w1 = 0 and w2 = 0. Seen from above at α = 0.5, the contour lines bend toward the axes, and the minimum moves toward zero: w = (${wAt(0.5).map((v) => v.toFixed(2)).join(", ")}).` },
      { t: "The weak weight becomes exactly zero", cam: ballCam,
        run: (ann) => { drv.slider("alpha", Math.min(3.5, +(a2 + 0.15).toFixed(2))); if (st.ok) ML.tag(ann, "w2 = 0 exactly", ball.position.clone().add(new THREE.Vector3(0, 0.5, 0)), col.bad, 0.22); },
        d: () => `From α ≈ ${a2.toFixed(2)} on, the minimum <b>sticks in the crease</b> on the w2 = 0 line: w2 is exactly zero, not just small. Feature 2 no longer contributes to the prediction. This is how Lasso selects features.` },
      { t: "Larger α shrinks the strong weight too", cam: CAM3,
        run: () => { drv.slider("alpha", +((a2 + a1) / 2).toFixed(2)); },
        d: () => `At α = ${((a2 + a1) / 2).toFixed(2)} the minimum slides along the w2 = 0 line toward the origin: w1 = ${wAt((a2 + a1) / 2)[0].toFixed(2)}. Larger α means a simpler model, but it also fits the data worse.` },
      { t: "Too much regularization: all weights zero", cam: ballCam,
        run: () => { drv.slider("alpha", Math.min(3.5, +(a1 + 0.2).toFixed(2))); },
        d: () => `From α ≈ ${a1.toFixed(2)} both weights are zero and the model predicts the mean of y for everyone (the intercept is not penalized). That is too simple. In practice <b>α is chosen with cross-validation</b>; the coefficient path of the real Diabetes data below shows the same story for 10 features.` },
    ], { title: "How Lasso sets weights exactly to zero" });
    disposers.push(() => tour.dispose());

    // ----- Diabetes coefficient path -----
    const DD = ML.data.extra.diabetes;
    const alphas = Array.from({ length: 90 }, (_, i) => Math.pow(10, 2 - (i / 89) * 4)); // 100 → 0.01
    let wp = null;
    const coefs = alphas.map((a) => (wp = ML.lassoCD(DD.X, DD.y, a, wp, 1000, 1e-6)));
    const pPanel = root.querySelector("#lp-panel");
    const plot = new ML.Plot2D(root.querySelector("#lp-plot"), { height: 400, xlim: [-2, 2], ylim: [-40, 40], pad: [12, 16, 38, 46], aria: "Lasso coefficient path" });
    disposers.push(() => plot.dispose());
    const PS = { la: 0 };
    const palette = (j) => ML.clusterColor(j);
    const ymin = Math.min(...coefs.flat()), ymax = Math.max(...coefs.flat());
    plot.o.ylim = [Math.floor(ymin / 10) * 10, Math.ceil(ymax / 10) * 10];
    ML.ui.heading(pPanel, "Pick α");
    ML.ui.slider(pPanel, { label: "alpha", min: 0.01, max: 100, log: true, value: 1, onInput: (v) => { PS.la = Math.log10(v); drawP(); } });
    const zeroOut = ML.ui.stats(pPanel, [{ key: "z", label: "Zero weights", big: true }]);
    const tbl = ML.el("div", { class: "bars" });
    pPanel.append(tbl);
    function drawP() {
      plot.clear();
      plot.frame({ xticks: [-2, -1, 0, 1, 2], yticks: ML.ticks(...plot.o.ylim, 6), xfmt: (v) => `${Math.pow(10, v)}`, xlabel: "alpha (log scale)", ylabel: "weight" });
      plot.path([[-2, 0], [2, 0]], col.muted, 1);
      DD.names.forEach((nm, j) => plot.path(alphas.map((a, i) => [Math.log10(a), coefs[i][j]]), palette(j), 2));
      plot.path([[PS.la, plot.o.ylim[0]], [PS.la, plot.o.ylim[1]]], col.ink, 1.5, [5, 4]);
      const w = ML.lassoCD(DD.X, DD.y, Math.pow(10, PS.la), null, 2000, 1e-8);
      const zc = w.filter((v) => Math.abs(v) < 1e-9).length;
      zeroOut.set("z", `${zc} of 10`);
      tbl.innerHTML = "";
      const mx = Math.max(...w.map(Math.abs), 1);
      DD.names.forEach((nm, j) => {
        const zero = Math.abs(w[j]) < 1e-9;
        tbl.append(ML.el("div", { class: "bar-row" }, [
          ML.el("span", { html: `<i style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${palette(j)};margin-right:6px"></i>${nm}` }),
          ML.el("div", { class: "bar-track" }, [ML.el("div", { class: "bar-fill", style: `width:${(Math.abs(w[j]) / mx) * 100}%;background:${palette(j)}` })]),
          ML.el("span", { class: "v", style: zero ? `color:${col.bad};font-weight:600` : "", text: zero ? "0" : w[j].toFixed(2) }),
        ]));
      });
      const ix = alphas.findIndex((a) => Math.log10(a) <= PS.la);
      if (ix >= 0) DD.names.forEach((nm, j) => { if (Math.abs(coefs[ix][j]) > 3) plot.text(nm, plot.X(Math.log10(alphas[ix])) + 6, plot.Y(coefs[ix][j]), palette(j), "left", '600 11px "IBM Plex Mono", monospace'); });
    }
    plot.drawFn = drawP;
    drawP();
    return disposers;
  },
});
