// Section 13: PCA inside a classification pipeline (Wine), and why MDS cannot be
// dropped into the same pipeline.
ML.register("pcaclf", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Use reduced coordinates as the inputs to a classifier: <b>original features → reduced coordinates → predicted class</b>. Fewer features may simplify classification, but can also remove useful information. Check validation accuracy.", ["sup", "cls", "GridSearchCV", "passthrough"])}

  <div class="section-title"><h2>The pipeline</h2></div>
  <div class="pipeline" id="pp-pipe"></div>
  <p class="fig-cap">PCA and MDS learn coordinates <b>without class labels</b>; only the classifier learns from the coordinates and the labels.</p>

  <div class="section-title"><h2>How many components? Let validation decide</h2><span class="hint">Wine: 178 wines, 13 features, 3 cultivars.</span></div>
  ${ML.bench("pp", { tag: "All 178 wines in their first three principal components", stage: "tall", cls: "wide-panel" })}
  <div id="pp-bars"></div>
  <p class="fig-cap">Bars: mean 3-fold cross-validation accuracy of StandardScaler → PCA(n) → KNN(k = 5) on the training set, as computed by scikit-learn's GridSearchCV. The script's grid only compares “none” (passthrough), 2 and 5 components (highlighted). The 3D view only explores the data (scaler and PCA fitted on all rows, as in the PCA section); the three cultivars already separate well in three coordinates.</p>

  <div class="two-col">
    ${ML.code(`
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model = make_pipeline(
    StandardScaler(), PCA(), KNeighborsClassifier(n_neighbors=5)
)
settings = {"pca": [
    "passthrough", PCA(n_components=2), PCA(n_components=5)
]}
search = GridSearchCV(model, settings, cv=3, scoring="accuracy")
search.fit(X_train, y_train)

print("Best preprocessing:", search.best_params_)
print("Validation accuracy:", search.best_score_)
print("Test accuracy:", search.score(X_test, y_test))`, `
Best preprocessing: {'pca': PCA(n_components=2)}
Validation accuracy: 0.9719267139479907
Test accuracy: 0.9166666666666666`)}
    <div class="prose">
      <ul>
        <li><code>"passthrough"</code> <b>skips PCA</b>: all 13 scaled features go to KNN.</li>
        <li>The grid replaces the whole <code>pca</code> step, so the setting name is just <code>"pca"</code>.</li>
        <li>Each fold learns its <b>own scaler and PCA directions</b> from its training rows.</li>
        <li>The best pipeline is refitted on the full training set; new samples then use that same scaler and projection.</li>
        <li>For classification, the number of components <b>need not be two</b>.</li>
      </ul>
      <p>Note how small the differences are (0.958 vs 0.972 vs 0.951). On 142 training wines, one or two wines change the ranking.</p>
    </div>
  </div>

  <div class="section-title"><h2>Why MDS needs an extra step</h2></div>
  <div class="two-col">
    <div class="prose">
      <p>MDS coordinates could also be classifier inputs: KNN would find neighbors in the MDS map. But scikit-learn's <code>MDS</code> has <code>fit_transform</code> and <b>no <code>transform</code> for new samples</b>, so it cannot directly replace PCA in this pipeline. A predictive MDS workflow needs an additional method that places new samples in the fixed training map.</p>
      <p>Try calling both on a new wine:</p>
      <div class="btn-row"><button class="btn" type="button" id="pp-try-pca">pca.transform(X_new)</button><button class="btn" type="button" id="pp-try-mds">mds.transform(X_new)</button></div>
      <div id="pp-try-out"></div>
    </div>
    ${ML.callout("trap", "Split before fitting preprocessing", "Fitting MDS on training and test rows together lets test data influence the map. Fitting separate maps gives <b>incompatible coordinates</b> for the classifier. PCA provides a <b>reusable projection</b> for new rows.")}
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="pcaclf"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const W = ML.data.wine, CV = ML.data.extra.wineCV;
    const S = { opt: 2 }; // index into CV.labels; 0 = passthrough
    const pipe = root.querySelector("#pp-pipe");
    const drawPipe = () => {
      const pcaTxt = S.opt === 0 ? "passthrough" : `PCA(n_components=${CV.labels[S.opt]})`;
      const nOut = S.opt === 0 ? 13 : CV.labels[S.opt];
      pipe.innerHTML = `
        <div class="pipe-step"><small>input</small><b>13 features</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>step 1</small><b>StandardScaler</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step hl"><small>step 2: "pca"</small><b>${pcaTxt}</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>${nOut} coordinates</small><b>KNN (k = 5)</b></div><span class="pipe-arrow">→</span>
        <div class="pipe-step"><small>output</small><b>cultivar 0, 1, 2</b></div>`;
    };
    // Exploration only: scaler and PCA fitted on all rows for the picture.
    const sc = ML.scaler(W.X), Xs = sc.transform(W.X), P = ML.pca(Xs), Z3 = P.project(Xs, 3);
    const host = root.querySelector("#pp-stage"), panel = root.querySelector("#pp-panel");
    const st = new ML.Stage3D(host, { cam: [6.4, 4.6, 7], target: [0, 0.8, 0] });
    disposers.push(() => st.dispose());
    if (st.ok) {
      ML.axes(st, { x: [-3.2, 3.2], y: [-3.2, 3.2], z: [-2.4, 3.2], labels: ["PC1", "PC2", "PC3"], div: 6 });
      const sph = new ML.Spheres(st, Z3.length, 0.07);
      Z3.forEach((z, i) => sph.set(i, ML.P(z[0] * 0.8, z[1] * 0.8, z[2] * 0.8 + 0.4), col.c[W.y[i]]));
      sph.commit();
      ML.legend(host, [0, 1, 2].map((k) => ({ label: `cultivar ${k}`, color: col.c[k] })));
      st.views([{ t: "3D", pos: [6.4, 4.6, 7], target: [0, 0.8, 0] }, { t: "PC1-PC2 only", pos: [0, 9, 0.01], target: [0, 0, 0] }]);
    }
    ML.ui.heading(panel, "Step \"pca\"");
    ML.ui.slider(panel, { label: "n_components", min: 0, max: 13, value: 2, fmt: (v) => (v === 0 ? "passthrough" : v), note: "0 = passthrough (no PCA, all 13 features).", onInput: (v) => { S.opt = v; upd(); } });
    const bar = ML.ui.bars(panel, [{ label: "PCA keeps", color: col.accent }]);
    const stats = ML.ui.stats(panel, [{ key: "cv", label: "Mean CV accuracy", big: true }]);
    const verdict = ML.ui.verdict(panel);
    const bp = new ML.Plot2D(root.querySelector("#pp-bars"), { height: 240, xlim: [-0.6, 13.6], ylim: [0.8, 1], aria: "CV accuracy per number of components" });
    disposers.push(() => bp.dispose());
    const cum = P.ratio.reduce((a, v) => (a.push((a.at(-1) || 0) + v), a), []);
    function upd() {
      drawPipe();
      const keep = S.opt === 0 ? 1 : cum[CV.labels[S.opt] - 1];
      bar.set([keep], (v) => `${(v * 100).toFixed(0)}% var.`);
      stats.set("cv", CV.cv[S.opt].toFixed(4));
      verdict.innerHTML = S.opt === 1 ? "One component throws away too much: validation accuracy drops to 0.83."
        : S.opt === 0 ? "No reduction: KNN sees all 13 scaled features."
        : [2, 5].includes(CV.labels[S.opt]) ? `This is one of the script's grid options. ${CV.labels[S.opt] === 2 ? "It wins the script's grid search." : ""}`
        : "Not in the script's grid, shown for comparison. The test set is not used to pick this.";
      bp.clear();
      bp.frame({ xticks: [], yticks: [0.8, 0.85, 0.9, 0.95, 1], ylabel: "mean CV accuracy" });
      CV.cv.forEach((v, i) => {
        const x = i, inGrid = i === 0 || CV.labels[i] === 2 || CV.labels[i] === 5;
        const ctx = bp.ctx;
        ctx.fillStyle = i === S.opt ? col.accent : inGrid ? col.c[1] : col.line;
        ctx.fillRect(bp.X(x - 0.35), bp.Y(v), bp.X(x + 0.35) - bp.X(x - 0.35), bp.Y(0.8) - bp.Y(v));
        bp.text(i === 0 ? "none" : String(CV.labels[i]), bp.X(x), bp.H - 22, col.muted, "center", '500 11.5px "IBM Plex Mono", monospace');
        bp.text(v.toFixed(3), bp.X(x), bp.Y(v) - 8, i === S.opt ? col.accent : col.muted, "center", '500 10px "IBM Plex Mono", monospace');
      });
      bp.text("n_components", bp.X(6.5), bp.H - 6, col.muted, "center");
    }
    bp.drawFn = upd;
    upd();
    const out = root.querySelector("#pp-try-out");
    root.querySelector("#pp-try-pca").addEventListener("click", () => { out.innerHTML = `<div class="ok-box">>>> pca.transform(X_new).shape\n(1, 2)\n# The new wine lands in the same 2D coordinate system.</div>`; });
    root.querySelector("#pp-try-mds").addEventListener("click", () => { out.innerHTML = `<div class="err-box">>>> mds.transform(X_new)\nAttributeError: 'MDS' object has no attribute 'transform'</div>`; });
    return disposers;
  },
});
