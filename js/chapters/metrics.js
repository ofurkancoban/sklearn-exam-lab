// Section 5: why scaling matters, data leakage, and the confusion matrix metrics.
ML.register("metrics", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Two habits that protect every result: <b>scale features</b> before distance-based models, and <b>look beyond accuracy</b> when checking predictions.", ["StandardScaler", "confusion_matrix", "classification_report"])}

  <div class="prose">
    <h2>Why scaling matters</h2>
    <p>Suppose one feature ranges from 0 to 1 and another from 0 to 10,000. The second can <b>dominate a distance calculation simply because its numbers are larger</b>. Below, three features of 80 customers are drawn at their true scale. The ringed customer's 5 nearest neighbors are highlighted.</p>
  </div>
  ${ML.bench("sc", { tag: "Same data, raw vs. standardized", stage: "tall" })}

  <div class="two-col">
    ${ML.formula("Standardization", String.raw`x' = \frac{x - \mu}{\sigma}`, "μ is the feature's <b>training</b> mean and σ its <b>training</b> standard deviation. Afterwards the training values have mean 0 and standard deviation 1. This does <b>not</b> make the data normally distributed.", "Eq. 7")}
    <div class="formula">
      <div class="f-label"><span>Try it</span><span>script example: μ = 10, σ = 2, x = 14</span></div>
      <div id="sc-calc" class="panel" style="padding:0"></div>
      <div id="sc-out" class="f-note mono"></div>
    </div>
  </div>
  <p class="prose">KNN, SVM, MLP and Lasso commonly benefit from scaling. Random forests usually do not need it.</p>

  <div class="section-title"><h2>Learn preprocessing from training data only</h2></div>
  <div class="two-col">
    <div class="formula">
      <div class="f-label"><span style="color:var(--good)">Correct: scaler inside the pipeline</span></div>
      <div class="pipeline">
        <div class="pipe-step"><small>split first</small><b>X_train | X_test</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step hl"><small>fit on X_train</small><b>StandardScaler</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>fit</small><b>model</b></div>
      </div>
      <div class="f-note">The pipeline learns μ and σ during <code>fit(X_train, y_train)</code> and applies <b>that same scaler</b> during prediction.</div>
    </div>
    <div class="formula">
      <div class="f-label"><span style="color:var(--bad)">Wrong: scale everything, then split</span></div>
      <div class="pipeline">
        <div class="pipe-step bad"><small>fit on ALL rows</small><b>StandardScaler</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>then</small><b>split</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>fit</small><b>model</b></div>
      </div>
      <div class="f-note">μ and σ now contain test rows. <b>Information from the test set leaks into training</b> and the test score becomes too optimistic.</div>
    </div>
  </div>

  <div class="prose">
    <h2>Accuracy is only one view</h2>
    <p>If 95 of 100 messages are ordinary mail, always predicting “ordinary” gives <b>95% accuracy while finding no spam</b>. When class sizes differ, inspect more than accuracy. Choose a positive class, such as spam. Rows show the true class; columns the predicted class; correct predictions sit on the diagonal.</p>
  </div>
  <div class="bench wide-panel">
    <div class="stage short" id="cm-stage" style="background:var(--surface)"><span class="stage-tag">Spam filter scores and the decision threshold</span><div id="cm-plot" style="position:absolute;inset:34px 10px 10px 10px"></div></div>
    <div class="panel" id="cm-panel"></div>
  </div>
  <div class="two-col">
    <div class="formula">
      <div class="f-label"><span>Confusion matrix</span><span>positive = spam</span></div>
      <div class="cm" id="cm-grid"></div>
    </div>
    <div class="formula">
      <div class="f-label"><span>Metrics with your numbers</span><span>Eq. 8</span></div>
      <div id="cm-formulas" class="f-body" style="font-size:15px"></div>
    </div>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Metric</th><th>Formula</th><th>Question it answers</th></tr></thead>
    <tbody>
      <tr><td>Accuracy</td><td>\(\frac{TP + TN}{TP + TN + FP + FN}\)</td><td>What fraction of all predictions is right?</td></tr>
      <tr><td>Precision</td><td>\(\frac{TP}{TP + FP}\)</td><td>How many <b>positive predictions</b> are correct?</td></tr>
      <tr><td>Recall</td><td>\(\frac{TP}{TP + FN}\)</td><td>How many <b>actual positives</b> did we find?</td></tr>
      <tr><td>F1</td><td>\(\frac{2TP}{2TP + FP + FN}\)</td><td>Combines precision and recall into one number.</td></tr>
    </tbody>
  </table></div>

  <div class="section-title"><h2>On Iris (three classes)</h2></div>
  <div class="two-col">
    ${ML.code(`
from sklearn.metrics import confusion_matrix, classification_report

# Run after any Iris classifier above (here: the random forest)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions, zero_division=0))`, `
[[10  0  0]
 [ 0  9  1]
 [ 0  2  8]]
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.82      0.90      0.86        10
           2       0.89      0.80      0.84        10

    accuracy                           0.90        30
   macro avg       0.90      0.90      0.90        30
weighted avg       0.90      0.90      0.90        30`)}
    <div class="prose">
      <p>Iris has three classes, so the matrix has <b>three rows and three columns</b>; the report scores each class as “positive” in turn.</p>
      <p>Class 1 (versicolor): 9 found out of 10 actual, so recall = 9/10 = 0.90. Of the 11 flowers predicted as class 1, 9 were right, so precision = 9/11 = 0.82.</p>
      <p><code>zero_division=0</code> reports zero for an undefined precision or recall, for example when a model never predicts a particular class (a zero denominator).</p>
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>Precision divides by <b>predicted</b> positives (TP + FP); recall divides by <b>actual</b> positives (TP + FN).</li><li>“How many malignant cases are detected?” is <b>recall</b>.</li><li>Scaler statistics come from the training rows only.</li><li>Standardized data is not automatically normally distributed.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="metrics"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // ----- Scaling scene -----
    const r = ML.rng(17), N = 80;
    const raw = [], cls = [];
    for (let i = 0; i < N; i++) {
      const c = i < N / 2 ? 0 : 1;
      const ratio = ML.clamp((c ? 0.72 : 0.3) + 0.13 * ML.randn(r), 0, 1);
      const income = ML.clamp(5000 + 2600 * ML.randn(r), 0, 10000);
      const age = ML.clamp((c ? 52 : 34) + 9 * ML.randn(r), 18, 80);
      raw.push([ratio, income, age]); cls.push(c);
    }
    const sc = ML.scaler(raw), Z = sc.transform(raw);
    const names = ["ratio (0-1)", "income (0-10,000)", "age (years)"];
    const host = root.querySelector("#sc-stage"), panel = root.querySelector("#sc-panel");
    const st = new ML.Stage3D(host, { cam: [6, 4.6, 6.8], target: [0, 1, 0] });
    disposers.push(() => st.dispose());
    const RAWS = 3 / 5000; // world units per raw unit (same factor for every axis)
    let t = 0, q = 7;
    const rawW = (v) => [(v[0] - sc.mu[0]) * RAWS, (v[1] - sc.mu[1]) * RAWS, (v[2] - sc.mu[2]) * RAWS + 1.4];
    const stdW = (z) => [z[0] * 0.75, z[1] * 0.75, z[2] * 0.75 + 1.4];
    let sph, links = new THREE.Group(), lbls = [];
    if (st.ok) {
      ML.axes(st, { x: [-3, 3], y: [-3, 3], labels: null, div: 6 });
      sph = new ML.Spheres(st, N, 0.075);
      st.add(links);
      const L = (txt, p) => { const s = ML.label(txt, { h: 0.22, color: col.muted }); s.position.copy(ML.P(...p)); st.add(s); lbls.push(s); return s; };
      L(names[0], [0, -3.4, 0]); L(names[1], [-3.5, 0, 0]); L(names[2], [-3, 3, 3.3]);
      ML.legend(host, [{ label: "group A", color: col.c[0] }, { label: "group B", color: col.c[1] }, { label: "query + 5 neighbors", ring: col.ink, color: "transparent" }]);
    }
    const neigh = (X) => X.map((v, i) => [ML.dist(v, X[q]), i]).filter(([, i]) => i !== q).sort((a, b) => a[0] - b[0]).slice(0, 5).map(([, i]) => i);
    const nbRaw = () => neigh(raw), nbStd = () => neigh(Z);
    ML.ui.heading(panel, "View");
    const seg = ML.ui.seg(panel, { options: [{ v: 0, t: "Raw units" }, { v: 1, t: "Standardized" }], value: 0, onChange: (v) => animate(v) });
    ML.ui.button(ML.ui.row(panel), "Pick another query", () => { q = (q + 13) % N; draw(); }, true);
    const stats = ML.ui.stats(panel, [{ key: "share", label: "Income's share of squared distance", big: true }, { key: "same", label: "Neighbors of the same group" }]);
    const verdict = ML.ui.verdict(panel);
    function draw() {
      const nb = t < 0.5 ? nbRaw() : nbStd();
      const X = t < 0.5 ? raw : Z;
      let inc = 0, tot = 0;
      nb.forEach((i) => { const d = X[i].map((v, j) => (v - X[q][j]) ** 2); inc += d[1]; tot += d[0] + d[1] + d[2]; });
      stats.set("share", `${((inc / tot) * 100).toFixed(t < 0.5 ? 3 : 1)}%`);
      const same = nb.filter((i) => cls[i] === cls[q]).length;
      stats.set("same", `${same} of 5`, same >= 4 ? col.good : col.bad);
      verdict.innerHTML = t < 0.5
        ? "In raw units the cloud collapses onto the income axis. Ratio and age barely count, so the “nearest” customers are simply those with similar income."
        : "After standardization each feature spans a similar range, so all three features shape the distance and the neighbors come from the right group.";
      if (!st.ok) return;
      const nbs = new Set(nb);
      st.clearGroup(links);
      const pos = raw.map((v, i) => { const a = rawW(v), b = stdW(Z[i]); return ML.P(ML.lerp(a[0], b[0], t), ML.lerp(a[1], b[1], t), ML.lerp(a[2], b[2], t)); });
      pos.forEach((p, i) => sph.set(i, p, col.c[cls[i]], i === q ? 2 : nbs.has(i) ? 1.5 : 0.85));
      sph.commit();
      nb.forEach((i) => links.add(ML.line([pos[q], pos[i]], new THREE.Color(col.ink), { opacity: 0.85 })));
    }
    let anim = null;
    function animate(to) {
      cancelAnimationFrame(anim);
      const from = t, t0 = performance.now();
      const go = (now) => { const k = ML.reducedMotion() ? 1 : ML.ease(Math.min(1, (now - t0) / 1300)); t = ML.lerp(from, to, k); draw(); if (k < 1) anim = requestAnimationFrame(go); };
      anim = requestAnimationFrame(go);
    }
    disposers.push(() => cancelAnimationFrame(anim));
    draw();

    // ----- Step-by-step tour: why scaling matters -----
    const sdrv = ML.drive(panel);
    const sameTxt = () => panel.querySelectorAll(".stat b")[1].textContent;
    const shareTxt = () => panel.querySelectorAll(".stat b")[0].textContent;
    const qpos = () => (st.ok && sph.pos[q] ? sph.pos[q].clone() : new THREE.Vector3());
    const scTour = ML.tour(host.parentElement, st, [
      { t: "Three features on their true scale", cam: [[6, 4.6, 6.8], [0, 1, 0]],
        run: (ann) => { sdrv.seg("Raw units"); if (st.ok) ML.tag(ann, "income: 0 to 10,000", ML.P(0.4, 2.6, 1.6), col.ink, 0.22); },
        d: "80 customers with a ratio (0 to 1), an income (0 to 10,000) and an age (18 to 80), drawn with the <b>same scale on every axis</b>, exactly how a distance formula sees them. The cloud collapses into a thin line along the income axis." },
      { t: "Ratio and age almost disappear", cam: [[0.01, 1.5, 6.5], [0, 1.4, 0]],
        run: (ann) => { sdrv.seg("Raw units"); if (st.ok) ML.tag(ann, "ratio and age differences: tiny", ML.P(0, 0, 1.9), col.bad, 0.18); },
        d: "Looking along the income axis, all customers sit almost on one point: a difference of 0.5 in ratio or 30 years in age is nothing next to a difference of 3,000 in income. In <b>squared</b> distances the gap is even larger." },
      { t: "Nearest neighbors in raw units", cam: () => { const p = qpos(); return [[p.x + 2.4, p.y + 1.8, p.z + 2.6], [p.x, p.y, p.z]]; },
        run: () => { sdrv.seg("Raw units"); },
        d: () => `The large ringed customer and its 5 nearest neighbors (lines). Income makes up <b>${shareTxt()}</b> of their squared distance, so the neighbors are simply people with similar income. Only ${sameTxt()} belong to the same group (color).` },
      { t: "Standardize: x′ = (x − μ) / σ", cam: [[6, 4.6, 6.8], [0, 1, 0]], ms: 8000,
        run: () => { sdrv.seg("Standardized"); },
        d: "Each feature is shifted by its training mean μ and divided by its training standard deviation σ. Watch the cloud open up: afterwards every feature has <b>mean 0 and standard deviation 1</b>, so all three span a similar range." },
      { t: "Nearest neighbors after scaling", cam: () => { const p = qpos(); return [[p.x + 2.4, p.y + 1.8, p.z + 2.6], [p.x, p.y, p.z]]; },
        run: () => { sdrv.seg("Standardized"); },
        d: () => `Now all features shape the distance (income's share drops to ${shareTxt()}) and ${sameTxt()} neighbors come from the same group. This is why KNN, SVM, MLP, Lasso and DBSCAN need scaling.` },
      { t: "Learn μ and σ from training data only", cam: [[6, 4.6, 6.8], [0, 1, 0]],
        run: () => { sdrv.seg("Standardized"); },
        d: "In a real model, μ and σ must come from the <b>training rows only</b>: put StandardScaler inside the pipeline. Fitting it on all data before splitting leaks test information into training. Standardized data is not automatically normally distributed." },
    ], { title: "Why scaling changes the neighbors" });
    disposers.push(() => scTour.dispose());

    // Standardization calculator.
    const cbox = root.querySelector("#sc-calc"), cout = root.querySelector("#sc-out");
    const cs = { mu: 10, sd: 2, x: 14 };
    const upd = () => { const z = (cs.x - cs.mu) / cs.sd; cout.innerHTML = `x' = (${cs.x} − ${cs.mu}) / ${cs.sd} = <b style="color:var(--accent)">${+z.toFixed(3)}</b>: ${Math.abs(z).toFixed(2)} standard deviations ${z >= 0 ? "above" : "below"} the mean.`; };
    ML.ui.slider(cbox, { label: "μ (training mean)", min: 0, max: 20, step: 0.5, value: 10, onInput: (v) => { cs.mu = v; upd(); } });
    ML.ui.slider(cbox, { label: "σ (training std)", min: 0.5, max: 6, step: 0.5, value: 2, onInput: (v) => { cs.sd = v; upd(); } });
    ML.ui.slider(cbox, { label: "x (raw value)", min: 0, max: 25, step: 0.5, value: 14, onInput: (v) => { cs.x = v; upd(); } });
    upd();

    // ----- Confusion matrix lab -----
    const cp = root.querySelector("#cm-panel");
    const S = { nPos: 50, n: 1000, thr: 0.5, sep: 1.6, always: false };
    const plot = new ML.Plot2D(root.querySelector("#cm-plot"), { height: 300, xlim: [0, 1], ylim: [0, 1], pad: [10, 12, 34, 40], aria: "Score distributions" });
    disposers.push(() => plot.dispose());
    // Scores: logistic of Gaussian, negatives centered low, positives high.
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const pdfN = (s, mu) => { const z = Math.log(s / (1 - s)); return Math.exp(-((z - mu) ** 2) / 2) / (s * (1 - s)); };
    const cdfZ = (z) => { const t = 1 / (1 + 0.2316419 * Math.abs(z)); const d = 0.3989423 * Math.exp(-z * z / 2); const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); return z > 0 ? 1 - p : p; };
    ML.ui.heading(cp, "Scenario");
    ML.ui.slider(cp, { label: "Spam among 1000 messages", min: 10, max: 500, step: 10, value: S.nPos, onInput: (v) => { S.nPos = v; S.always = false; updCM(); } });
    ML.ui.slider(cp, { label: "Decision threshold", min: 0.02, max: 0.98, step: 0.01, value: S.thr, note: "Score ≥ threshold → predict spam.", onInput: (v) => { S.thr = v; S.always = false; updCM(); } });
    ML.ui.slider(cp, { label: "Classifier quality", min: 0, max: 4, step: 0.1, value: S.sep, note: "Distance between the two score distributions.", onInput: (v) => { S.sep = v; updCM(); } });
    const br = ML.ui.row(cp);
    ML.ui.button(br, "Always predict “ordinary”", () => { S.always = true; S.nPos = 50; cp.querySelectorAll("input")[0].value = 50; cp.querySelectorAll("output")[0].textContent = "50"; updCM(); }, true);
    const mbars = ML.ui.bars(cp, [{ label: "Accuracy", color: col.accent }, { label: "Precision", color: col.c[1] }, { label: "Recall", color: col.c[0] }, { label: "F1", color: col.warn }]);
    function counts() {
      const nNeg = S.n - S.nPos;
      if (S.always) return { TN: nNeg, FP: 0, FN: S.nPos, TP: 0 };
      const z = Math.log(S.thr / (1 - S.thr));
      const muN = -S.sep / 2, muP = S.sep / 2;
      const TP = Math.round(S.nPos * (1 - cdfZ(z - muP))), FN = S.nPos - TP;
      const FP = Math.round(nNeg * (1 - cdfZ(z - muN))), TN = nNeg - FP;
      return { TN, FP, FN, TP };
    }
    function updCM() {
      const c = counts(), { TN, FP, FN, TP } = c;
      const acc = (TP + TN) / S.n, pre = TP + FP ? TP / (TP + FP) : 0, rec = TP + FN ? TP / (TP + FN) : 0, f1 = 2 * TP + FP + FN ? (2 * TP) / (2 * TP + FP + FN) : 0;
      mbars.set([acc, pre, rec, f1], (v) => v.toFixed(3));
      root.querySelector("#cm-grid").innerHTML = `
        <span></span><span class="h">Predicted negative</span><span class="h">Predicted positive</span>
        <span class="rh">Actually negative</span><div class="cell ok"><b>${TN}</b><span>True negative (TN)</span></div><div class="cell err"><b>${FP}</b><span>False positive (FP)</span></div>
        <span class="rh">Actually positive</span><div class="cell err"><b>${FN}</b><span>False negative (FN)</span></div><div class="cell ok"><b>${TP}</b><span>True positive (TP)</span></div>`;
      const fe = root.querySelector("#cm-formulas");
      fe.innerHTML = String.raw`\[\text{Accuracy} = \frac{${TP} + ${TN}}{${S.n}} = ${acc.toFixed(3)}\]
        \[\text{Precision} = \frac{${TP}}{${TP} + ${FP}} = ${TP + FP ? pre.toFixed(3) : "\\text{undefined} \\to 0"}\]
        \[\text{Recall} = \frac{${TP}}{${TP} + ${FN}} = ${rec.toFixed(3)}\]
        \[F_1 = \frac{2\cdot ${TP}}{2\cdot ${TP} + ${FP} + ${FN}} = ${f1.toFixed(3)}\]`;
      clearTimeout(fe._t);
      fe._t = setTimeout(() => ML.typesetLater(fe), 120);
      // Score distributions.
      plot.clear();
      const nNeg = S.n - S.nPos;
      const xs = Array.from({ length: 199 }, (_, i) => (i + 1) / 200);
      const dn = xs.map((s) => pdfN(s, -S.sep / 2) * nNeg), dp = xs.map((s) => pdfN(s, S.sep / 2) * S.nPos);
      const ymax = Math.max(...dn, ...dp) * 1.08;
      plot.o.ylim = [0, ymax];
      plot.frame({ xticks: [0, 0.25, 0.5, 0.75, 1], yticks: [], xlabel: "spam score from the classifier", grid: false });
      const fillArea = (d, color, from, to) => {
        const ctx = plot.ctx; ctx.save(); ctx.beginPath();
        let started = false;
        xs.forEach((x, i) => { if (x < from || x > to) return; if (!started) { ctx.moveTo(plot.X(x), plot.Y(0)); started = true; } ctx.lineTo(plot.X(x), plot.Y(d[i])); });
        ctx.lineTo(plot.X(Math.min(to, xs[xs.length - 1])), plot.Y(0)); ctx.closePath(); ctx.fillStyle = color; ctx.globalAlpha = 0.35; ctx.fill(); ctx.restore();
      };
      const thr = S.always ? 1 : S.thr;
      fillArea(dn, col.c[1], 0, thr); fillArea(dn, col.bad, thr, 1);
      fillArea(dp, col.bad, 0, thr); fillArea(dp, col.c[0], thr, 1);
      plot.path(xs.map((x, i) => [x, dn[i]]), col.c[1], 2);
      plot.path(xs.map((x, i) => [x, dp[i]]), col.c[0], 2);
      plot.path([[thr, 0], [thr, ymax]], col.ink, 2, [6, 4]);
      plot.text("ordinary mail", plot.X(0.06), plot.Y(ymax * 0.92), col.c[1], "left", '600 12.5px "IBM Plex Sans", sans-serif');
      plot.text("spam", plot.X(0.94), plot.Y(ymax * 0.92), col.c[0], "right", '600 12.5px "IBM Plex Sans", sans-serif');
      plot.text(S.always ? "threshold off: never spam" : `threshold ${S.thr.toFixed(2)}`, plot.X(thr) - 6, plot.Y(ymax * 0.78), col.ink, "right", '500 12px "IBM Plex Mono", monospace');
      plot.text("red = errors (FP right of the line, FN left of it)", plot.X(0.5), plot.Y(ymax * 0.62), col.muted, "center");
    }
    plot.drawFn = updCM;
    updCM();
    return disposers;
  },
});
