// Section 1: tasks, X and y, notation. Hero shows Iris in 3D.
ML.register("overview", {
  html: (m) => String.raw`
  <div class="hero-grid">
    <div class="ch-head">
      <div class="eyebrow"><span>Section ${m.n}</span><span class="pdfref">${m.pdf}</span></div>
      <h1 class="hero-title">Machine learning with scikit-learn, <em>in three dimensions</em></h1>
      <p class="idea">Machine learning <b>finds patterns in examples</b> to make predictions or explore data. Every method in the exam script has an interactive 3D model here: rotate it, change its main setting, and watch what happens.</p>
      <div class="chips"><span class="chip">10 methods</span><span class="chip">4 task types</span><span class="chip">fit / predict / transform</span></div>
    </div>
    ${ML.bench("ov", { tag: "Iris: 150 flowers, 4 features, 3 species", stage: "short", cls: "stack", help: "Drag to rotate, hover a flower" })}
  </div>
  <div id="ov-controls" class="btn-row" style="align-items:end"></div>

  <div class="section-title"><h2>Four kinds of questions</h2><span class="hint">Pick the task first, then the method.</span></div>
  <div class="task-grid">
    <div class="task-card"><span class="chip task-cls" style="justify-self:start">Supervised</span><h3>Classification</h3><div class="q">Which category?</div><p>Which flower species? Spam or ordinary mail?</p><div class="methods"><a href="#knn">KNN</a><a href="#svm">SVM</a><a href="#forest">Random forest</a><a href="#mlp">MLP</a></div></div>
    <div class="task-card"><span class="chip task-cls" style="justify-self:start">Supervised</span><h3>Regression</h3><div class="q">What number?</div><p>How much energy is used? How far does a disease progress?</p><div class="methods"><a href="#linreg">Linear</a><a href="#lasso">Lasso</a><a href="#poly">Polynomial</a></div></div>
    <div class="task-card"><span class="chip task-unsup" style="justify-self:start">Unsupervised</span><h3>Clustering</h3><div class="q">Which examples form groups?</div><p>Group similar observations without knowing any answers.</p><div class="methods"><a href="#dbscan">DBSCAN</a></div></div>
    <div class="task-card"><span class="chip task-unsup" style="justify-self:start">Unsupervised</span><h3>Dimensionality reduction</h3><div class="q">Can we use fewer features?</div><p>Draw many features in 2D.</p><div class="methods"><a href="#pca">PCA</a><a href="#mds">MDS</a></div></div>
  </div>

  <div class="two-col">
    <div class="prose">
      <h2>Two names you will see everywhere</h2>
      <ul>
        <li><b>X</b>: input data. <b>Rows are examples; columns are features.</b></li>
        <li><b>y</b>: correct answers. Classification uses categories; regression uses numbers.</li>
      </ul>
      <p>Learning from <b>X and y</b> is <b>supervised learning</b>. Clustering and dimensionality reduction work with <b>X alone</b>: <b>unsupervised learning</b>.</p>
      <p>Hover any cell of the table. It tells you the math name of that entry and the Python index you would use to read it.</p>
      <div id="ov-explain" class="verdict">Hover a cell in the table.</div>
    </div>
    <div class="matrix-demo">
      <div class="table-wrap"><table class="mtable" id="ov-matrix"></table></div>
      <div class="fig-cap">First six Iris rows. The four left columns form <b>X</b> (n = 150, p = 4); the right column is <b>y</b> (0 = setosa, 1 = versicolor, 2 = virginica).</div>
    </div>
  </div>

  <div class="prose">
    <h2>Reading the equations</h2>
  </div>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Symbol</th><th>Meaning</th><th>Watch out</th></tr></thead>
      <tbody>
        <tr><td>\(n\)</td><td>number of examples (rows)</td><td></td></tr>
        <tr><td>\(p\)</td><td>number of features (columns)</td><td></td></tr>
        <tr><td>\(x_{ij}\)</td><td>feature \(j\) of example \(i\)</td><td>A subscript identifies an entry; it is not a multiplication.</td></tr>
        <tr><td>\(x = (x_1, \dots, x_p)\)</td><td>one example as a feature vector</td><td></td></tr>
        <tr><td>\(y\)</td><td>the correct answer</td><td></td></tr>
        <tr><td>\(\hat{y}\)</td><td>a prediction ("y-hat")</td><td></td></tr>
        <tr><td>\(w_j\)</td><td>a weight for feature \(j\)</td><td></td></tr>
        <tr><td>indices</td><td>math starts at 1</td><td>Python starts at 0, so \(x_{11}\) is <code>X[0, 0]</code>.</td></tr>
      </tbody>
    </table>
  </div>

  <div class="two-col">
    <div class="prose">
      <h2>Getting ready</h2>
      <p>Install the packages in your terminal. In a notebook, use <code>%pip install scikit-learn numpy matplotlib</code> instead. In Python, <code>import</code> loads a tool and <code>print(...)</code> displays a result.</p>
      ${ML.code(`python -m pip install scikit-learn numpy matplotlib`, "", "terminal")}
    </div>
    <div class="prose">
      <h2>The main scikit-learn verbs</h2>
      <div class="table-wrap"><table>
        <tbody>
          <tr><td><code>fit</code></td><td>learn from data</td></tr>
          <tr><td><code>predict</code></td><td>produce answers</td></tr>
          <tr><td><code>transform</code></td><td>change the representation of data</td></tr>
          <tr><td><code>fit_transform</code></td><td>learn the transformation and apply it</td></tr>
          <tr><td><code>fit_predict</code></td><td>learn a grouping and return labels for the fitted data</td></tr>
        </tbody>
      </table></div>
    </div>
  </div>
  ${ML.callout("tip", "How to use this lab", "Each section follows the script: the idea, the key formula, a 3D model you can change, the scikit-learn code with its real output, the traps examiners like, and a short quiz. Mark a section as studied at the bottom to track your progress. The Exam practice page mixes all questions.")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="overview"></div>
  `,
  init(root) {
    const D = ML.data.iris, col = ML.colors();
    const host = root.querySelector("#ov-stage");
    const st = new ML.Stage3D(host, { cam: [5.2, 3.6, 5.6], target: [0, 0.9, 0], autoRotate: true });
    if (!st.ok) return () => st.dispose();
    const ctr = root.querySelector("#ov-controls");
    let axesIdx = [2, 3, 0], showY = true;
    const units = (j) => `${D.features[j]} (cm)`;
    const mins = [0, 1, 2, 3].map((j) => Math.min(...D.X.map((r) => r[j])));
    const maxs = [0, 1, 2, 3].map((j) => Math.max(...D.X.map((r) => r[j])));
    const S = 3.2;
    const map = (r) => axesIdx.map((j, k) => ((r[j] - mins[j]) / (maxs[j] - mins[j])) * S - (k < 2 ? S / 2 : 0));
    const sph = new ML.Spheres(st, 150, 0.075);
    let frame = null;
    const draw = () => {
      if (frame) { st.scene.remove(frame); ML.disposeObj(frame); }
      frame = ML.axes(st, { x: [-S / 2, S / 2], y: [-S / 2, S / 2], z: [0, S], labels: axesIdx.map(units), div: 6 });
      D.X.forEach((r, i) => {
        const p = map(r);
        sph.set(i, ML.P(p[0], p[1], p[2]), showY ? col.c[D.y[i]] : col.muted);
      });
      sph.commit();
    };
    draw();
    const legend = ML.legend(host, D.names.map((n, k) => ({ label: `${k} = ${n}`, color: col.c[k] })));
    // Controls: axis selectors + labels toggle.
    const sel = (k, lab) => {
      const s = ML.el("select", { id: `ov-ax${k}`, "aria-label": lab });
      D.features.forEach((f, j) => s.append(ML.el("option", { value: j, text: f })));
      s.value = axesIdx[k];
      s.addEventListener("change", () => { axesIdx[k] = +s.value; draw(); });
      ctr.append(ML.el("div", { class: "ctrl", style: "min-width:150px;flex:1" }, [ML.el("div", { class: "ctrl-row" }, [ML.el("label", { for: `ov-ax${k}`, text: lab })]), s]));
    };
    sel(0, "Horizontal axis 1"); sel(1, "Horizontal axis 2"); sel(2, "Vertical axis");
    const tg = ML.el("div", { class: "ctrl", style: "min-width:220px;flex:1.4" });
    ML.ui.seg(tg, {
      label: "What does the algorithm see?",
      options: [{ v: true, t: "X and y (supervised)" }, { v: false, t: "X only (unsupervised)" }],
      value: true,
      onChange: (v) => { showY = v; legend.hidden = !v; draw(); },
    });
    ctr.append(tg);
    // Hover tooltip.
    const tip = ML.tooltip(host);
    const onMove = (ev) => {
      const hit = st.raycast(ev, [sph.mesh])[0];
      if (!hit) return tip.hide();
      const i = hit.instanceId, r = D.X[i];
      const rect = host.getBoundingClientRect();
      tip.show(ev.clientX - rect.left, ev.clientY - rect.top,
        `<b>row i = ${i + 1}</b> (Python ${i})<br>${D.features.map((f, j) => `${f}: ${r[j]}`).join("<br>")}<br>y = ${showY ? `${D.y[i]} (${D.names[D.y[i]]})` : "<i>hidden</i>"}`);
    };
    st.renderer.domElement.addEventListener("pointermove", onMove);
    st.renderer.domElement.addEventListener("pointerleave", () => tip.hide());

    // X / y matrix with index explanation.
    const tbl = root.querySelector("#ov-matrix"), ex = root.querySelector("#ov-explain");
    const rows = [0, 1, 50, 51, 100, 101];
    let h = "<thead><tr><th>i</th>" + D.features.map((f, j) => `<th>j = ${j + 1}<br><span style="text-transform:none">${f}</span></th>`).join("") + `<th class="ycol">y</th></tr></thead><tbody>`;
    rows.forEach((ri, k) => {
      h += `<tr><th>${ri + 1}</th>` + D.X[ri].map((v, j) => `<td data-i="${ri}" data-j="${j}" data-k="${k}">${v}</td>`).join("") + `<td class="ycol" data-i="${ri}" data-y="1">${D.y[ri]}</td></tr>`;
    });
    tbl.innerHTML = h + "</tbody>";
    tbl.addEventListener("mouseover", (ev) => {
      const td = ev.target.closest("td"); if (!td) return;
      tbl.querySelectorAll("td").forEach((c) => c.classList.remove("hl-row", "hl-col", "hl-cell"));
      const i = +td.dataset.i;
      if (td.dataset.y) {
        td.classList.add("hl-cell");
        ex.innerHTML = `This is <b>y<sub>${i + 1}</sub> = ${D.y[i]}</b>, the correct answer (species <b>${D.names[D.y[i]]}</b>) for example ${i + 1}. In Python: <code>y[${i}]</code>.`;
        return;
      }
      const j = +td.dataset.j;
      tbl.querySelectorAll(`td[data-i="${i}"]:not([data-y])`).forEach((c) => c.classList.add("hl-row"));
      tbl.querySelectorAll(`td[data-j="${j}"]`).forEach((c) => c.classList.add("hl-col"));
      td.classList.remove("hl-row", "hl-col"); td.classList.add("hl-cell");
      ex.innerHTML = `This is <b>x<sub>${i + 1},${j + 1}</sub> = ${D.X[i][j]}</b>: feature ${j + 1} (${D.features[j]}) of example ${i + 1}. The highlighted row is one example x; the highlighted column is one feature. In Python: <code>X[${i}, ${j}]</code>, because Python counts from 0.`;
    });
    return () => st.dispose();
  },
});
