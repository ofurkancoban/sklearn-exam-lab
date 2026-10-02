// Section 2: K-nearest neighbors on the real Iris train/test split.
ML.register("knn", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Similar examples often have similar answers. <b>KNN finds the k closest training examples and lets them vote.</b> With k = 5, three votes for class A and two for class B produce a prediction of A.", ["sup", "cls", "n_neighbors"])}

  <div class="two-col">
    ${ML.formula("Euclidean distance", String.raw`\operatorname{dist}(x, z) = \sqrt{\sum_{j=1}^{p} (x_j - z_j)^2}`, "Subtract matching features, square the differences, add them, take the square root. KNN uses the k smallest distances and predicts the most frequent class among those neighbors.", "Eq. 1")}
    <div class="formula">
      <div class="f-label"><span>Try it: distance calculator</span><span>p = 2</span></div>
      <div class="btn-row" style="align-items:center;font-family:var(--f-mono);font-size:14px">
        x = (<input id="knn-x1" type="number" value="1" style="width:64px">, <input id="knn-x2" type="number" value="2" style="width:64px">)
        z = (<input id="knn-z1" type="number" value="4" style="width:64px">, <input id="knn-z2" type="number" value="6" style="width:64px">)
      </div>
      <div id="knn-calc" class="f-note mono"></div>
    </div>
  </div>

  <div class="section-title"><h2>Neighbors in 3D</h2><span class="hint">120 training flowers (spheres). The big diamond is the flower we want to classify.</span></div>
  ${ML.bench("knn", { tag: "KNN on Iris, after StandardScaler", stage: "tall" })}
  <p class="fig-cap">Lines connect the query to its k nearest training flowers; the glass sphere has the radius of the k-th distance, so every flower inside it votes. For the picture, distances use the three displayed features; the script's model uses all four. Accuracy below is computed on the 30 real test flowers from <code>train_test_split(..., random_state=42, stratify=y)</code>.</p>

  <div class="section-title"><h2>How k shapes the decision regions</h2><span class="hint">Same k as above, two petal features, every pixel classified by KNN.</span></div>
  <div id="knn-map"></div>
  ${ML.callout("setting", "Main setting: n_neighbors", "A <b>small value reacts more to individual examples</b> (islands, jagged borders, k = 1 memorizes the training set). A <b>larger value smooths the predictions</b>. KNN is easy to understand, but <b>predicting can become slow with many examples</b>: every prediction measures the distance to every training example.")}

  <div class="section-title"><h2>A complete example</h2></div>
  <div class="two-col">
    ${ML.code(`
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Predicted:", predictions[:5])
print("Correct: ", y_test[:5])`, `
Accuracy: 0.9333333333333333
Predicted: [0 2 1 1 0]
Correct:  [0 2 1 1 0]`)}
    <div class="prose">
      <h3>Read it line by line</h3>
      <ol class="code-notes">
        <li><code>train_test_split</code>: learn from 80%; reserve 20% for testing (150 flowers: 120 train, 30 test).</li>
        <li>The <b>pipeline</b> scales features, then applies KNN.</li>
        <li><code>fit</code> learns from training data; <code>predict</code> returns test labels.</li>
        <li><code>accuracy_score</code> gives the fraction correct: 0.90 means 90%. Here 28 of 30 are right.</li>
      </ol>
      <p><code>random_state=42</code> makes the split <b>repeatable</b>. The number 42 has no special meaning.</p>
      <p><code>stratify=y</code> keeps <b>class proportions</b> approximately the same in the training and test sets (here exactly 10 test flowers per species).</p>
      <p>Iris: 150 flowers, four measurements each, three species stored as labels 0, 1 and 2.</p>
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>KNN has almost no training work. The cost is at <b>prediction</b> time.</li><li>Distances are measured <b>after scaling</b> in the pipeline. Without scaling, a feature with large numbers dominates the distance.</li><li>Do not confuse KNN's <b>k neighbors</b> with the <b>K folds</b> of cross-validation.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="knn"></div>
  `,
  init(root) {
    const D = ML.data.iris, sp = ML.data.extra.irisSplit, col = ML.colors();
    // Distance calculator.
    const ids = ["x1", "x2", "z1", "z2"].map((k) => root.querySelector(`#knn-${k}`));
    const calc = () => {
      const [a, b, c, d] = ids.map((i) => parseFloat(i.value) || 0);
      const d1 = a - c, d2 = b - d, s = d1 * d1 + d2 * d2;
      root.querySelector("#knn-calc").innerHTML =
        `(${a} - ${c})² + (${b} - ${d})² = ${+(d1 * d1).toFixed(4)} + ${+(d2 * d2).toFixed(4)} = ${+s.toFixed(4)}<br>dist = √${+s.toFixed(4)} = <b style="color:var(--accent)">${+Math.sqrt(s).toFixed(4)}</b>`;
    };
    ids.forEach((i) => i.addEventListener("input", calc));
    calc();

    const Xtr0 = sp.train.map((i) => D.X[i]), ytr = sp.train.map((i) => D.y[i]);
    const Xte0 = sp.test.map((i) => D.X[i]), yte = sp.test.map((i) => D.y[i]);
    const sc = ML.scaler(Xtr0);
    const host = root.querySelector("#knn-stage"), panel = root.querySelector("#knn-panel");
    const st = new ML.Stage3D(host, { cam: [5.2, 3.8, 5.8], target: [0, 0.1, 0] });
    const state = { k: 5, scaled: true, q: 0, feats: [2, 3, 0] };
    const S = 0.85;
    let Xtr, Xte, center, frame = null;
    const spheres = st.ok ? new ML.Spheres(st, 120, 0.075) : null;
    const qMesh = st.ok ? new THREE.Mesh(new THREE.OctahedronGeometry(0.17), new THREE.MeshStandardMaterial({ color: col.ink, roughness: 0.3 })) : null;
    const glass = st.ok ? new THREE.Mesh(new THREE.SphereGeometry(1, 40, 28), new THREE.MeshStandardMaterial({ color: col.accent, transparent: true, opacity: 0.12, depthWrite: false })) : null;
    const glassWire = st.ok ? new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({ color: col.accent, wireframe: true, transparent: true, opacity: 0.18 })) : null;
    const linkGroup = st.ok ? new THREE.Group() : null;
    if (st.ok) { st.add(qMesh); st.add(glass); st.add(glassWire); st.add(linkGroup); }

    const prep = () => {
      const T = state.scaled ? sc.transform : (X) => X;
      const f = state.feats;
      Xtr = T(Xtr0).map((r) => f.map((j) => r[j]));
      Xte = T(Xte0).map((r) => f.map((j) => r[j]));
      center = [0, 1, 2].map((k) => ML.mean(Xtr.map((r) => r[k])));
    };
    const world = (v) => ML.P((v[0] - center[0]) * S, (v[1] - center[1]) * S, (v[2] - center[2]) * S);
    const knn = (q, k) => {
      const d = Xtr.map((r, i) => ({ i, d: ML.dist(r, q) })).sort((a, b) => a.d - b.d);
      const nb = d.slice(0, k), votes = [0, 0, 0];
      nb.forEach((o) => votes[ytr[o.i]]++);
      // Tie-break like scikit-learn: lowest class label among the most frequent.
      const pred = votes.indexOf(Math.max(...votes));
      return { nb, votes, pred };
    };

    // Panel controls.
    ML.ui.heading(panel, "Settings");
    const kS = ML.ui.slider(panel, { label: "n_neighbors (k)", min: 1, max: 30, value: 5, onInput: (v) => { state.k = v; update(); drawMap(); } });
    const qS = ML.ui.slider(panel, { label: "Test flower", min: 1, max: 30, value: 1, fmt: (v) => `#${v} of 30`, onInput: (v) => { state.q = v - 1; update(); } });
    ML.ui.check(panel, { label: "StandardScaler in the pipeline", checked: true, onChange: (v) => { state.scaled = v; rebuild(); drawMap(); } });
    const fsel = ML.el("select", { id: "knn-feats", "aria-label": "Displayed features" });
    [[[2, 3, 0], "petal length, petal width, sepal length"], [[0, 1, 2], "sepal length, sepal width, petal length"], [[0, 1, 3], "sepal length, sepal width, petal width"]].forEach(([f, t], i) => fsel.append(ML.el("option", { value: i, text: t })));
    const fopts = [[2, 3, 0], [0, 1, 2], [0, 1, 3]];
    fsel.addEventListener("change", () => { state.feats = fopts[+fsel.value]; rebuild(); });
    panel.append(ML.el("div", { class: "ctrl" }, [ML.el("div", { class: "ctrl-row" }, [ML.el("label", { for: "knn-feats", text: "Features used" })]), fsel]));
    ML.ui.heading(panel, "Votes of the k neighbors");
    const bars = ML.ui.bars(panel, D.names.map((n, c) => ({ label: n, color: col.c[c] })));
    const verdict = ML.ui.verdict(panel);
    const stats = ML.ui.stats(panel, [{ key: "acc", label: "Test accuracy (30 flowers)", big: true }, { key: "kd", label: "k-th distance" }]);

    function rebuild() {
      prep();
      if (st.ok) {
        if (frame) { st.scene.remove(frame); ML.disposeObj(frame); }
        const all = Xtr.concat(Xte);
        const lo = [0, 1, 2].map((k) => Math.min(...all.map((r) => r[k])) - center[k]);
        const hi = [0, 1, 2].map((k) => Math.max(...all.map((r) => r[k])) - center[k]);
        const fn = state.feats.map((j) => D.features[j] + (state.scaled ? " (scaled)" : " (cm)"));
        frame = ML.axes(st, { x: [lo[0] * S - 0.2, hi[0] * S + 0.2], y: [lo[1] * S - 0.2, hi[1] * S + 0.2], z: [lo[2] * S - 0.2, hi[2] * S + 0.2], labels: fn, div: 6 });
      }
      update();
    }
    function update() {
      const q = Xte[state.q], res = knn(q, state.k);
      if (st.ok) {
        const inNb = new Set(res.nb.map((o) => o.i));
        Xtr.forEach((r, i) => spheres.set(i, world(r), col.c[ytr[i]], inNb.has(i) ? 1.55 : 0.85));
        spheres.commit();
        const qp = world(q);
        qMesh.position.copy(qp);
        qMesh.material.color.set(col.c[res.pred]);
        const rad = res.nb[res.nb.length - 1].d * S;
        glass.position.copy(qp); glass.scale.setScalar(rad + 0.04);
        glassWire.position.copy(qp); glassWire.scale.setScalar(rad + 0.04);
        st.clearGroup(linkGroup);
        res.nb.forEach((o) => linkGroup.add(ML.line([qp, world(Xtr[o.i])], new THREE.Color(col.c[ytr[o.i]]), { opacity: 0.9 })));
      }
      bars.set(res.votes, (v) => String(v), state.k);
      const right = res.pred === yte[state.q];
      verdict.innerHTML = `Prediction: <b>${D.names[res.pred]}</b> (${Math.max(...res.votes)} of ${state.k} votes). True species: <b>${D.names[yte[state.q]]}</b>. <span style="color:var(${right ? "--good" : "--bad"})">${right ? "Correct." : "Wrong."}</span>`;
      let correct = 0;
      Xte.forEach((r, i) => { if (knn(r, state.k).pred === yte[i]) correct++; });
      stats.set("acc", `${correct}/30 = ${(correct / 30).toFixed(3)}`);
      stats.set("kd", res.nb[res.nb.length - 1].d.toFixed(3));
    }
    rebuild();
    ML.legend(host, D.names.map((n, c) => ({ label: n, color: col.c[c] })));

    // 2D decision map on petal length / width.
    const mapBox = root.querySelector("#knn-map");
    const plot = new ML.Plot2D(mapBox, { height: 340, xlim: [-2, 2.1], ylim: [-1.9, 2], aria: "KNN decision regions" });
    const P2 = sc.transform(Xtr0).map((r) => [r[2], r[3]]);
    const P2raw = Xtr0.map((r) => [r[2], r[3]]);
    function drawMap() {
      const pts = state.scaled ? P2 : P2raw;
      plot.o.xlim = state.scaled ? [-2, 2.1] : [0.5, 7.3];
      plot.o.ylim = state.scaled ? [-1.9, 2] : [-0.2, 2.8];
      plot.clear();
      const ctx = plot.ctx, step = 4, k = state.k;
      const rgb = col.c.map(ML.rgb);
      plot.clip(() => {
        for (let px = plot.o.pad[3]; px < plot.W - plot.o.pad[1]; px += step) {
          for (let py = plot.o.pad[0]; py < plot.H - plot.o.pad[2]; py += step) {
            const x = plot.invX(px + step / 2), y = plot.invY(py + step / 2);
            const d = pts.map((p, i) => [(p[0] - x) ** 2 + (p[1] - y) ** 2, i]).sort((a, b) => a[0] - b[0]).slice(0, k);
            const v = [0, 0, 0]; d.forEach(([, i]) => v[ytr[i]]++);
            const c = rgb[v.indexOf(Math.max(...v))];
            ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},0.28)`;
            ctx.fillRect(px, py, step, step);
          }
        }
      });
      plot.frame({ xticks: ML.ticks(...plot.o.xlim, 6), yticks: ML.ticks(...plot.o.ylim, 5), xlabel: state.scaled ? "petal length (scaled)" : "petal length (cm)", ylabel: state.scaled ? "petal width (scaled)" : "petal width (cm)", grid: false });
      pts.forEach((p, i) => plot.dot(p[0], p[1], 3.6, col.c[ytr[i]], col.surface, 1));
      plot.text(`k = ${k}`, plot.W - 20, 20, col.ink, "right", '600 14px "IBM Plex Mono", monospace');
    }
    plot.drawFn = drawMap;
    drawMap();
    // ----- Step-by-step tour -----
    const drv = ML.drive(panel);
    // Pick a test flower whose prediction depends on k (near the versicolor / virginica border).
    let qi = Xte.findIndex((r, i) => knn(r, 1).pred !== knn(r, 9).pred);
    if (qi < 0) qi = Xte.findIndex((r, i) => yte[i] === 2);
    const near = () => qMesh.position.clone();
    const closeCam = () => { const p = near(); return [[p.x + 2.3, p.y + 1.6, p.z + 2.6], [p.x, p.y, p.z]]; };
    const hide = (on) => { if (!st.ok) return; [qMesh, glass, glassWire, linkGroup].forEach((m) => (m.visible = on)); };
    const tour = ML.tour(host.parentElement, st, [
      { t: "The training data is the model", cam: [[5.2, 3.8, 5.8], [0, 0.1, 0]],
        run: (ann) => { drv.slider("n_neighbors", 5); drv.slider("Test flower", qi + 1); hide(false); ML.tag(ann, "each sphere = one training flower", ML.P(-1.4, 1.2, 1.8), col.muted, 0.2); return () => hide(true); },
        d: "Each sphere is one of the <b>120 training flowers</b>, placed by three of its measurements (after StandardScaler) and colored by its known species y. KNN's <code>fit</code> does almost nothing: it simply <b>stores</b> these examples. All the work happens at prediction time." },
      { t: "A new flower arrives", cam: closeCam,
        run: (ann) => { [glass, glassWire, linkGroup].forEach((m) => st.ok && (m.visible = false)); ML.tag(ann, "new flower: species = ?", near().add(new THREE.Vector3(0, 0.45, 0)), col.ink); return () => hide(true); },
        d: () => `The black diamond is test flower #${qi + 1}. We know its measurements but pretend we do not know its species. KNN will look at the training flowers <b>closest</b> to it.` },
      { t: "Measure the distance to every training flower", cam: closeCam,
        run: (ann) => {
          [glass, glassWire, linkGroup].forEach((m) => st.ok && (m.visible = false));
          if (!st.ok) return () => hide(true);
          const q = Xte[qi], pairs = Xtr.map((r) => [world(q), world(r)]);
          ann.add(ML.segments(pairs, new THREE.Color(col.muted), 0.18));
          const nn = knn(q, 1).nb[0];
          ML.tag(ann, `nearest: d = ${nn.d.toFixed(2)}`, world(Xtr[nn.i]).add(new THREE.Vector3(0, 0.3, 0)), col.accent, 0.2);
          return () => hide(true);
        },
        d: "KNN computes the Euclidean distance \\(\\sqrt{\\sum_j (x_j - z_j)^2}\\) from the new flower to <b>all 120</b> training flowers (grey lines). This is why predicting gets slow when there are many examples." },
      { t: "k = 1: the single closest flower decides", cam: closeCam,
        run: () => { drv.slider("n_neighbors", 1); },
        d: () => { const r = knn(Xte[qi], 1); return `With k = 1 only the nearest flower votes. The glass sphere's radius is the distance to it. Prediction: <b>${D.names[r.pred]}</b>. One single, possibly unusual flower decides everything: k = 1 reacts strongly to noise.`; } },
      { t: "k = 5: a small majority vote", cam: closeCam,
        run: () => { drv.slider("n_neighbors", 5); },
        d: () => { const r = knn(Xte[qi], 5); return `Now the 5 nearest flowers are inside the sphere and vote: ${r.votes.map((v, c) => `${v} × ${D.names[c]}`).join(", ")}. Prediction: <b>${D.names[r.pred]}</b> (true species: ${D.names[yte[qi]]}). The vote counts are also shown in the panel.`; } },
      { t: "k = 15: smoother, less sensitive", cam: () => { const p = near(); return [[p.x + 3.6, p.y + 2.6, p.z + 4], [p.x, p.y, p.z]]; },
        run: () => { drv.slider("n_neighbors", 15); },
        d: () => { const r = knn(Xte[qi], 15); return `With k = 15 the sphere grows and many more flowers vote: ${r.votes.map((v, c) => `${v} × ${D.names[c]}`).join(", ")} → <b>${D.names[r.pred]}</b>. Larger k <b>smooths</b> the predictions; too large k ignores local structure. The right k is chosen with cross-validation.`; } },
      { t: "The same rule everywhere: decision regions", cam: [[5.2, 3.8, 5.8], [0, 0.1, 0]],
        run: () => { drv.slider("n_neighbors", 5); },
        d: "Repeating this vote for every possible point gives the colored regions in the 2D map below this panel. Try the k slider there: k = 1 creates small islands around single flowers, larger k gives smooth borders. On the 30 test flowers this model scores <b>28 / 30 = 0.933</b>, the same as the script." },
    ], { title: "How KNN classifies a new flower" });
    return () => { tour.dispose(); st.dispose(); plot.dispose(); };
  },
});
