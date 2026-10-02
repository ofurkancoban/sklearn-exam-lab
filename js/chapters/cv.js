// Section 14: hyperparameters, K-fold cross-validation and GridSearchCV.
ML.register("cv", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Settings chosen before training, such as KNN's number of neighbors, are called <b>hyperparameters</b>. To compare settings <b>without using the test answers</b>, use cross-validation on the training set.", ["sel", "GridSearchCV", "cv"])}

  <div class="section-title"><h2>K-fold cross-validation</h2><span class="hint">Each fold is the validation set exactly once.</span></div>
  ${ML.bench("cv", { tag: "120 training rows split into K folds", stage: "short" })}
  <div class="formula">
    <div class="f-label"><span>The same schedule as a table</span><span id="cv-k-label"></span></div>
    <div class="kfold" id="cv-grid"></div>
  </div>
  <div class="two-col">
    ${ML.formula("Average validation score", String.raw`\bar{s} = \frac{1}{K} \sum_{r=1}^{K} s_r`, "In 3-fold cross-validation, the training set is divided into three parts. Each part becomes the validation set once while the other two are used for training. We average the three validation scores. <b>K is the number of folds, distinct from KNN's k neighbors.</b>", "Eq. 16")}
    ${ML.callout("tip", "Training learns; validation chooses; testing checks", "Do not keep changing the model to improve the same test score. That turns the test set into another validation set and makes the result <b>too optimistic</b>. A tiny score difference on Iris is not proof that one method is generally better.")}
  </div>

  <div class="section-title"><h2>Let a grid search choose k</h2><span class="hint">Real scikit-learn results: 3-fold CV on the Iris training set.</span></div>
  <div class="bench wide-panel">
    <div class="stage" style="background:var(--surface)"><span class="stage-tag">Mean validation accuracy for every n_neighbors (click to select)</span><div id="cv-plot" style="position:absolute;inset:32px 6px 6px 6px"></div></div>
    <div class="panel" id="gs-panel"></div>
  </div>

  <div class="two-col">
    ${ML.code(`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(StandardScaler(), KNeighborsClassifier())
settings = {"kneighborsclassifier__n_neighbors": [3, 5, 9]}

search = GridSearchCV(model, settings, cv=3, scoring="accuracy")
search.fit(X_train, y_train)

print("Best setting:", search.best_params_)
print("Mean validation accuracy:", search.best_score_)
print("Final test accuracy:", search.score(X_test, y_test))`, `
Best setting: {'kneighborsclassifier__n_neighbors': 3}
Mean validation accuracy: 0.9583333333333334
Final test accuracy: 0.9333333333333333`)}
    <div class="prose">
      <p><code>GridSearchCV</code> tries each listed setting using cross-validation. The scaler is inside the pipeline, so it is <b>fitted separately within every training fold</b>.</p>
      <p>k = 3 and k = 5 tie at 0.9583; scikit-learn keeps the first one in the list, so it reports k = 3.</p>
      <p>After choosing the best setting, GridSearchCV <b>automatically trains that pipeline again on the full training set</b>. The final line then checks it on the held-out test set.</p>
      <h3>The long setting name</h3>
      <p><code>pipeline_step_name__parameter_name</code>: <b>two underscores</b> between the step and the parameter. <code>make_pipeline</code> creates step names from the <b>lowercase class names</b>.</p>
      <div class="formula">
        <div class="f-label"><span>Build a setting name</span></div>
        <div class="btn-row"><select id="cv-step" aria-label="Pipeline step" style="flex:1"></select><select id="cv-param" aria-label="Parameter" style="flex:1"></select></div>
        <div class="f-note mono" id="cv-name"></div>
      </div>
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>K folds (cross-validation) ≠ k neighbors (KNN).</li><li>Random splits suit independent examples. For <b>time-ordered data</b> or <b>repeated observations from the same person</b>, use a split that respects time or keeps related observations together.</li><li>The final test set is used once, at the end.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="cv"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const S = { K: 3, it: 0, playing: true };
    const host = root.querySelector("#cv-stage");
    const st = new ML.Stage3D(host, { cam: [0, 4.2, 7.6], target: [0, 0.4, 0], noPan: true });
    disposers.push(() => st.dispose());
    let cubes, lifts = new Array(120).fill(0), labels = new THREE.Group();
    const cubeGeo = new THREE.BoxGeometry(0.16, 0.16, 0.16);
    if (st.ok) {
      cubes = new THREE.InstancedMesh(cubeGeo, new THREE.MeshStandardMaterial({ roughness: 0.6 }), 120);
      st.add(cubes); st.add(labels);
    }
    const sp = root.querySelector("#cv-panel");
    ML.ui.heading(sp, "Folds");
    ML.ui.slider(sp, { label: "K (number of folds)", min: 2, max: 10, value: 3, onInput: (v) => { S.K = v; S.it = 0; layout(); grid(); } });
    const row = ML.ui.row(sp);
    const playBtn = ML.ui.button(row, "Pause", () => { S.playing = !S.playing; playBtn.textContent = S.playing ? "Pause" : "Play"; });
    ML.ui.button(row, "Next iteration", () => { S.it = (S.it + 1) % S.K; grid(); }, true);
    const info = ML.ui.verdict(sp);
    const foldOf = (i) => Math.floor((i * S.K) / 120);
    const pos = [];
    function layout() {
      if (!st.ok) return;
      st.clearGroup(labels);
      const gap = 0.45, per = 120 / S.K, cols = Math.ceil(Math.sqrt(per * 1.6));
      const widths = [];
      for (let f = 0; f < S.K; f++) widths.push(Math.ceil(Math.ceil(per) / Math.ceil(Math.ceil(per) / cols)) * 0.2);
      const total = widths.reduce((a, b) => a + b, 0) + gap * (S.K - 1);
      let x0 = -total / 2;
      const counters = new Array(S.K).fill(0);
      const rowsPer = Math.ceil(Math.ceil(per) / cols);
      for (let i = 0; i < 120; i++) {
        const f = foldOf(i), k = counters[f]++;
        const fx = x0 + widths.slice(0, f).reduce((a, b) => a + b, 0) + gap * f;
        pos[i] = new THREE.Vector3(fx + (Math.floor(k / rowsPer)) * 0.2 + 0.1, 0.08, (k % rowsPer) * 0.2 - rowsPer * 0.1);
      }
      for (let f = 0; f < S.K; f++) {
        const fx = x0 + widths.slice(0, f).reduce((a, b) => a + b, 0) + gap * f + widths[f] / 2;
        const lb = ML.label(`Fold ${f + 1}`, { h: 0.22, color: col.muted }); lb.position.set(fx, -0.05, rowsPer * 0.1 + 0.35); labels.add(lb);
      }
    }
    function grid() {
      root.querySelector("#cv-k-label").textContent = `K = ${S.K}`;
      const g = root.querySelector("#cv-grid");
      g.style.setProperty("--k", S.K);
      g.innerHTML = Array.from({ length: S.K }, (_, it) => `<div class="kfold-row" style="opacity:${it === S.it ? 1 : 0.55}"><span class="lab">Iteration ${it + 1}</span>${Array.from({ length: S.K }, (_, f) => `<div class="blk ${f === it ? "va" : "tr"}">${S.K <= 6 ? (f === it ? "Validation" : "Training") : f === it ? "V" : "T"}</div>`).join("")}<span class="sc">${it === S.it ? "◀ now" : ""}</span></div>`).join("");
      info.innerHTML = `Iteration ${S.it + 1} of ${S.K}: train on ${S.K - 1} folds (${Math.round((120 * (S.K - 1)) / S.K)} rows), validate on fold ${S.it + 1} (${Math.round(120 / S.K)} rows). After all ${S.K} iterations, every row has been used for validation exactly once.`;
    }
    const cT = new THREE.Color(col.c[1]), cV = new THREE.Color(col.c[2]), tmp = new THREE.Color(), dm = new THREE.Object3D();
    let lastSwitch = performance.now();
    if (st.ok) {
      st.onFrame((t) => {
        if (S.playing && t - lastSwitch > 1800) { S.it = (S.it + 1) % S.K; lastSwitch = t; grid(); }
        for (let i = 0; i < 120; i++) {
          const val = foldOf(i) === S.it;
          lifts[i] += ((val ? 0.75 : 0) - lifts[i]) * 0.12;
          dm.position.copy(pos[i]); dm.position.y += lifts[i]; dm.updateMatrix();
          cubes.setMatrixAt(i, dm.matrix);
          tmp.copy(cT).lerp(cV, lifts[i] / 0.75); cubes.setColorAt(i, tmp);
        }
        cubes.instanceMatrix.needsUpdate = true; cubes.instanceColor.needsUpdate = true;
      });
    }
    layout(); grid();

    // Grid search plot over n_neighbors.
    const G = ML.data.extra.irisGrid;
    const panel2 = root.querySelector("#gs-panel");
    const plot = new ML.Plot2D(root.querySelector("#cv-plot"), { height: 400, xlim: [0, 31], ylim: [0.85, 1.0], aria: "Grid search results" });
    disposers.push(() => plot.dispose());
    const P = { k: 3 };
    ML.ui.heading(panel2, "Selected setting");
    const ks = ML.ui.slider(panel2, { label: "n_neighbors", min: 1, max: 30, value: 3, onInput: (v) => { P.k = v; draw(); } });
    const fb = ML.ui.bars(panel2, [0, 1, 2].map((f) => ({ label: `fold ${f + 1} s<sub>${f + 1}</sub>`, color: col.c[1] })).concat([{ label: "<b>mean s̄</b>", color: col.accent }]));
    const formula = ML.el("div", { class: "verdict mono", style: "font-size:13px" }); panel2.append(formula);
    const best = G.mean.indexOf(Math.max(...G.mean));
    panel2.append(ML.el("div", { class: "ctrl-note", html: `Best over k = 1…30: <b>k = ${G.k[best]}</b> (s̄ = ${G.mean[best].toFixed(4)}). The script's grid [3, 5, 9] is highlighted.` }));
    function draw() {
      const i = P.k - 1;
      const fs = G.folds.map((f) => f[i]);
      fb.set([...fs, G.mean[i]].map((v) => (v - 0.8) / 0.2), (v) => (v * 0.2 + 0.8).toFixed(3));
      formula.innerHTML = `s̄ = (${fs.map((v) => v.toFixed(3)).join(" + ")}) / 3<br>&nbsp;&nbsp;= ${G.mean[i].toFixed(4)}`;
      plot.clear();
      plot.frame({ xticks: [1, 5, 10, 15, 20, 25, 30], yticks: [0.85, 0.9, 0.95, 1], xlabel: "n_neighbors (k)", ylabel: "validation accuracy" });
      [3, 5, 9].forEach((k) => { const ctx = plot.ctx; ctx.fillStyle = col.accent; ctx.globalAlpha = 0.1; ctx.fillRect(plot.X(k - 0.45), plot.o.pad[0], plot.X(k + 0.45) - plot.X(k - 0.45), plot.ih); ctx.globalAlpha = 1; });
      G.folds.forEach((f) => f.forEach((v, j) => plot.dot(j + 1, v, 2.6, col.muted)));
      plot.path(G.mean.map((v, j) => [j + 1, v]), col.accent, 2.4);
      G.mean.forEach((v, j) => plot.dot(j + 1, v, j === i ? 6.5 : 3.6, j === i ? col.accent : col.surface, col.accent, 2));
      plot.text("grey dots: single folds · line: mean s̄", plot.W - 20, 20, col.muted, "right");
    }
    plot.drawFn = draw;
    plot.canvas.addEventListener("click", (ev) => {
      const r = plot.canvas.getBoundingClientRect();
      const k = ML.clamp(Math.round(plot.invX(ev.clientX - r.left)), 1, 30);
      P.k = k; ks.set(k); draw();
    });
    draw();

    // Setting-name builder.
    const steps = { kneighborsclassifier: ["n_neighbors", "weights"], svc: ["C", "gamma", "kernel"], lasso: ["alpha"], pca: ["n_components"], polynomialfeatures: ["degree"], mlpclassifier: ["hidden_layer_sizes", "max_iter"] };
    const ss = root.querySelector("#cv-step"), sp2 = root.querySelector("#cv-param"), out = root.querySelector("#cv-name");
    Object.keys(steps).forEach((s) => ss.append(ML.el("option", { value: s, text: s })));
    const fill = () => { sp2.innerHTML = ""; steps[ss.value].forEach((p) => sp2.append(ML.el("option", { value: p, text: p }))); name(); };
    const name = () => { out.innerHTML = `settings = {"<b style="color:var(--accent)">${ss.value}__${sp2.value}</b>": [...]}`; };
    ss.addEventListener("change", fill); sp2.addEventListener("change", name);
    fill();
    return disposers;
  },
});
