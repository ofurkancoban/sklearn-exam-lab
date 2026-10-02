// Section 4: random forests. CART trees with Gini splits, bootstrap samples and
// one random feature per split (max_features="sqrt" of 2 features = 1).
ML.tree = {
  build(X, y, idx, nC, maxDepth, rng, depth = 0) {
    const counts = new Array(nC).fill(0);
    idx.forEach((i) => counts[y[i]]++);
    const leaf = { leaf: true, p: counts.map((c) => c / idx.length), n: idx.length };
    if (depth >= maxDepth || idx.length < 2 || counts.filter((c) => c > 0).length < 2) return leaf;
    const p = X[0].length;
    const order = ML.shuffleIdx(p, rng);
    let best = null;
    // Try one random feature first; fall back to the others only if it cannot split.
    for (let fi = 0; fi < p; fi++) {
      const f = order[fi];
      const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
      const left = new Array(nC).fill(0), right = counts.slice();
      for (let k = 0; k < s.length - 1; k++) {
        left[y[s[k]]]++; right[y[s[k]]]--;
        const a = X[s[k]][f], b = X[s[k + 1]][f];
        if (a === b) continue;
        const nl = k + 1, nr = s.length - nl;
        const gl = 1 - left.reduce((t, c) => t + (c / nl) ** 2, 0), gr = 1 - right.reduce((t, c) => t + (c / nr) ** 2, 0);
        const g = (nl * gl + nr * gr) / s.length;
        if (!best || g < best.g) best = { g, f, t: (a + b) / 2 };
      }
      if (best) break;
    }
    if (!best) return leaf;
    const L = idx.filter((i) => X[i][best.f] <= best.t), Rr = idx.filter((i) => X[i][best.f] > best.t);
    return { f: best.f, t: best.t, n: idx.length, p: leaf.p, l: ML.tree.build(X, y, L, nC, maxDepth, rng, depth + 1), r: ML.tree.build(X, y, Rr, nC, maxDepth, rng, depth + 1) };
  },
  proba(node, x) { while (!node.leaf) node = x[node.f] <= node.t ? node.l : node.r; return node.p; },
};

ML.register("forest", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Combine many decision trees. A tree asks a sequence of questions, such as <i>“Is the petal shorter than 2 cm?”</i> A forest trains many <b>different</b> trees and <b>averages their opinions</b>.", ["sup", "cls", "n_estimators", "max_depth"])}

  <div class="two-col">
    <div class="prose">
      <h2>Bootstrapping and bagging</h2>
      <p><b>Bootstrapping</b> creates a new training sample by drawing rows <b>with replacement</b>: after selecting a row, we can select it again. For each tree we draw <b>as many rows as the original training set contains</b>. Some rows appear several times, others are missing. Rows A, B, C, D might give the sample A, C, C, D.</p>
      <p><b>Bagging</b> = <b>b</b>ootstrap <b>agg</b>regat<b>ing</b>: train several models on different bootstrap samples, then combine their predictions. This often reduces the variability of predictions compared with one tree.</p>
      <p>A random forest also considers a <b>random subset of features at each split</b>. This makes the trees less similar, so their errors can partly cancel.</p>
    </div>
    <div class="formula bootstrap">
      <div class="f-label"><span>Try it: draw a bootstrap sample</span><span>n = 10 rows</span></div>
      <div class="boot-rows" id="boot-orig"></div>
      <div class="btn-row"><button class="btn" id="boot-go" type="button">Draw 10 rows with replacement</button></div>
      <div class="f-note mono" id="boot-seq">Sample: (press the button)</div>
      <div class="f-note" id="boot-stat"></div>
    </div>
  </div>

  ${ML.formula("Average the trees' opinions", String.raw`\bar{q}(c \mid x) = \frac{1}{T} \sum_{t=1}^{T} q_t(c \mid x)`, "q<sub>t</sub>(c | x) is the probability that tree t assigns to class c for input x (read “class c, given input x”). The forest predicts the class with the <b>largest average probability</b>. This is an average of class probabilities, not necessarily a vote among the trees' final labels. For regression, a forest averages the trees' numerical predictions.")}

  <div class="section-title"><h2>A forest, layer by layer</h2><span class="hint">Each floating layer is one tree's map. The bottom layer is their average.</span></div>
  ${ML.bench("rf", { tag: "RandomForestClassifier, trained live", stage: "tall", help: "Hover the bottom layer to probe a point" })}
  <p class="fig-cap">Every tree saw a different bootstrap sample, so the layers disagree: blocky, axis-aligned regions that overfit their own sample. The average (bottom) is smoother and more stable. The vertical line is the probe; the panel lists q<sub>t</sub>(c | x) for the shown trees and the forest's q̄(c | x).</p>

  <div class="two-col">
    <div class="prose">
      <h3>The first tree, as questions</h3>
      <p>Top three levels of tree 1 from the forest above. Each internal node asks one yes/no question about one feature; each leaf holds class probabilities.</p>
      <div id="rf-tree" class="table-wrap" style="padding:8px"></div>
    </div>
    <div class="prose">
      <h3>In scikit-learn</h3>
      ${ML.code(`
from sklearn.ensemble import RandomForestClassifier

# Reuse the Iris split and accuracy_score from page 2
model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Forest accuracy:", accuracy_score(y_test, predictions))`, `Forest accuracy: 0.9`)}
      ${ML.callout("setting", "Main settings", "<b>n_estimators</b>: number of trees. <b>max_depth</b>: how many successive splits a tree can make. More trees take more time and <b>do not guarantee a better result</b>. Feature scaling is usually unnecessary: a split like x ≤ 2.5 works the same in any unit.")}
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>Bootstrapping draws <b>with</b> replacement and the <b>same number</b> of rows as the training set.</li><li>Two sources of randomness: bootstrap rows and random features per split.</li><li>The forest averages probabilities q<sub>t</sub>(c | x); it does not have to count label votes.</li><li>No pipeline with StandardScaler is needed for the forest.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="forest"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // Bootstrap widget.
    const rows = "ABCDEFGHIJ".split("");
    const orig = root.querySelector("#boot-orig");
    const cells = rows.map((r) => { const c = ML.el("div", { class: "boot-cell", text: r }); orig.append(c); return c; });
    let bootTimer = null, bseed = 1;
    root.querySelector("#boot-go").addEventListener("click", () => {
      clearInterval(bootTimer);
      const r = ML.rng(bseed++ * 7919), counts = new Array(10).fill(0), seq = [];
      cells.forEach((c) => { c.className = "boot-cell"; c.querySelector("sup")?.remove(); });
      let k = 0;
      const step = () => {
        const i = Math.floor(r() * 10);
        counts[i]++; seq.push(rows[i]);
        const c = cells[i];
        c.classList.add("drawn", "pulse");
        setTimeout(() => c.classList.remove("pulse"), 180);
        let s = c.querySelector("sup"); if (!s) { s = ML.el("sup"); c.append(s); }
        s.textContent = `×${counts[i]}`;
        root.querySelector("#boot-seq").textContent = "Sample: " + seq.join(", ");
        if (++k === 10) {
          clearInterval(bootTimer);
          cells.forEach((c, j) => { if (!counts[j]) c.classList.add("oob"); });
          const miss = counts.filter((c) => !c).length;
          root.querySelector("#boot-stat").innerHTML = `${miss} of 10 rows were never drawn (faded). ${counts.filter((c) => c > 1).length} rows appear more than once. On average about 37% of rows are missing from each bootstrap sample.`;
        }
      };
      if (ML.reducedMotion()) { for (let j = 0; j < 10; j++) step(); } else bootTimer = setInterval(step, 160);
    });
    disposers.push(() => clearInterval(bootTimer));

    // Datasets.
    const DS = {
      blobs: (s) => ML.makeBlobs([[-1.1, -0.7], [1.1, -0.6], [0, 1.2]], 40, 0.62, s),
      moons: (s) => { const d = ML.makeMoons(120, 0.22, s); d.X = d.X.map(([a, b]) => [(a - 0.5) * 1.5, (b - 0.25) * 1.8]); return d; },
    };
    const state = { ds: "blobs", T: 10, depth: 6, seed: 1 };
    const R = 2.4, LAYERS = 5, GAP = 0.85, RES = 120;
    const host = root.querySelector("#rf-stage"), panel = root.querySelector("#rf-panel");
    const st = new ML.Stage3D(host, { cam: [7.2, 5.6, 8.4], target: [0.4, 1.9, 0] });
    disposers.push(() => st.dispose());
    let train, test, trees = [], probe = [0.3, 0.2];
    const layerPlanes = [], layerLabels = [];
    let basePlane, pts, probeLine, probeDots;
    if (st.ok) {
      basePlane = ML.texturePlane(2 * R, 2 * R, RES);
      st.add(basePlane.mesh);
      const bl = ML.label("average q̄", { h: 0.24, color: col.ink, center: "left" });
      bl.position.set(R + 0.15, 0.05, -R + 0.3); st.add(bl);
      for (let k = 0; k < LAYERS; k++) {
        const lp = ML.texturePlane(2 * R, 2 * R, RES);
        lp.mesh.position.y = (k + 1) * GAP + 0.4;
        lp.mesh.material.opacity = 0.85;
        st.add(lp.mesh);
        const frame = ML.line([[-R, -R], [R, -R], [R, R], [-R, R], [-R, -R]].map(([a, b]) => new THREE.Vector3(a, lp.mesh.position.y, b)), new THREE.Color(col.line));
        st.add(frame);
        const lab = ML.label(`tree ${k + 1}`, { h: 0.22, color: col.muted, center: "left" });
        lab.position.set(R + 0.15, lp.mesh.position.y, -R + 0.3); st.add(lab);
        layerPlanes.push(lp); layerLabels.push({ lab, frame });
      }
      ML.axes(st, { x: [-R, R], y: [-R, R], labels: ["x1", "x2", null], div: 6 });
      pts = new ML.Spheres(st, 130, 0.055);
      probeLine = ML.line([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)], new THREE.Color(col.ink), { opacity: 0.8 });
      st.add(probeLine);
      probeDots = new ML.Spheres(st, LAYERS + 1, 0.07);
      ML.legend(host, ["class 0", "class 1", "class 2"].map((t, i) => ({ label: t, color: col.c[i] })));
    }
    ML.ui.heading(panel, "Forest");
    ML.ui.seg(panel, { label: "Dataset", options: [{ v: "blobs", t: "3 blobs" }, { v: "moons", t: "Moons" }], value: "blobs", onChange: (v) => { state.ds = v; fit(); } });
    ML.ui.slider(panel, { label: "n_estimators", min: 1, max: 60, value: state.T, onInput: (v) => { state.T = v; fit(); } });
    ML.ui.slider(panel, { label: "max_depth", min: 1, max: 12, value: state.depth, fmt: (v) => (v === 12 ? "12 (deep)" : v), onInput: (v) => { state.depth = v; fit(); } });
    ML.ui.button(ML.ui.row(panel), "New bootstrap samples", () => { state.seed++; fit(); }, true);
    ML.ui.heading(panel, "Accuracy on fresh test data");
    const stats = ML.ui.stats(panel, [{ key: "tree", label: "Tree 1 alone" }, { key: "forest", label: "Whole forest", big: true }, { key: "train", label: "Forest on training data" }]);
    ML.ui.heading(panel, "Probe: q(c | x) per tree");
    const probeBox = ML.el("div", { class: "bars" });
    panel.append(probeBox);

    const mix = (p) => {
      const rgb = col.c.slice(0, 3).map(ML.rgb);
      const c = [0, 1, 2].map((k) => p.reduce((s, pc, ci) => s + pc * rgb[ci][k], 0));
      return c;
    };
    function fit() {
      const nC = state.ds === "blobs" ? 3 : 2;
      train = DS[state.ds](3); test = DS[state.ds](99);
      const rng = ML.rng(state.seed * 101);
      trees = [];
      for (let t = 0; t < state.T; t++) {
        const idx = Array.from({ length: train.X.length }, () => Math.floor(rng() * train.X.length));
        trees.push(ML.tree.build(train.X, train.y, idx, nC, state.depth, rng));
      }
      const fp = (x) => { const s = new Array(nC).fill(0); trees.forEach((t) => ML.tree.proba(t, x).forEach((v, c) => (s[c] += v / trees.length))); return s; };
      const am = (p) => p.indexOf(Math.max(...p));
      const acc = (pred, d) => d.X.filter((x, i) => am(pred(x)) === d.y[i]).length / d.X.length;
      stats.set("tree", acc((x) => ML.tree.proba(trees[0], x), test).toFixed(3));
      stats.set("forest", acc(fp, test).toFixed(3));
      stats.set("train", acc(fp, train).toFixed(3));
      if (st.ok) {
        const pad = (p) => (p.length === 3 ? p : [p[0], p[1], 0]);
        basePlane.redraw((u, v) => { const p = pad(fp([-R + u * 2 * R, -R + v * 2 * R])); const c = mix(p); return [c[0], c[1], c[2], 60 + 150 * (Math.max(...p) - 0.33)]; });
        layerPlanes.forEach((lp, k) => {
          const show = k < trees.length;
          lp.mesh.visible = show; layerLabels[k].lab.visible = show; layerLabels[k].frame.visible = show;
          if (show) lp.redraw((u, v) => { const p = pad(ML.tree.proba(trees[k], [-R + u * 2 * R, -R + v * 2 * R])); const c = mix(p); return [c[0], c[1], c[2], 150]; });
        });
        pts.setCount(train.X.length);
        train.X.forEach((x, i) => pts.set(i, ML.P(x[0], x[1], 0.06), col.c[train.y[i]]));
        pts.commit();
      }
      drawTree();
      updateProbe();
    }
    function updateProbe() {
      const nC = state.ds === "blobs" ? 3 : 2;
      const shown = Math.min(LAYERS, trees.length);
      const ps = trees.slice(0, shown).map((t) => ML.tree.proba(t, probe));
      const avg = new Array(nC).fill(0); trees.forEach((t) => ML.tree.proba(t, probe).forEach((v, c) => (avg[c] += v / trees.length)));
      probeBox.innerHTML = "";
      const row = (label, p, strong) => {
        const track = ML.el("div", { class: "bar-track", style: "display:flex" });
        p.forEach((v, c) => track.append(ML.el("div", { style: `width:${v * 100}%;background:${col.c[c]};height:100%` })));
        probeBox.append(ML.el("div", { class: "bar-row" }, [ML.el("span", { html: strong ? `<b>${label}</b>` : label }), track, ML.el("span", { class: "v", text: p.map((v) => v.toFixed(1)).join("/") })]));
      };
      ps.forEach((p, k) => row(`tree ${k + 1}`, p));
      if (trees.length > shown) probeBox.append(ML.el("div", { class: "ctrl-note", text: `+ ${trees.length - shown} more trees in the average` }));
      row("q̄ forest", avg, true);
      const win = avg.indexOf(Math.max(...avg));
      probeBox.append(ML.el("div", { class: "verdict", html: `Prediction at x = (${probe[0].toFixed(2)}, ${probe[1].toFixed(2)}): <b>class ${win}</b> with q̄ = ${avg[win].toFixed(2)}.` }));
      if (st.ok) {
        const top = shown * GAP + 0.4;
        probeLine.geometry.setFromPoints([ML.P(probe[0], probe[1], 0), ML.P(probe[0], probe[1], top + 0.3)]);
        probeDots.setCount(shown + 1);
        probeDots.set(0, ML.P(probe[0], probe[1], 0.02), col.c[win], 1.3);
        ps.forEach((p, k) => probeDots.set(k + 1, ML.P(probe[0], probe[1], (k + 1) * GAP + 0.4), col.c[p.indexOf(Math.max(...p))]));
        probeDots.commit();
      }
    }
    if (st.ok) {
      st.renderer.domElement.addEventListener("pointermove", (ev) => {
        const p = st.planeHit(ev, 0);
        if (!p || Math.abs(p.x) > R || Math.abs(p.z) > R) return;
        probe = [p.x, -p.z];
        updateProbe();
      });
    }
    // Tree diagram (SVG) for tree 1, three levels.
    function drawTree() {
      const t = trees[0], box = root.querySelector("#rf-tree");
      const W = 640, H = 250, levels = 4;
      let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="min-width:520px;display:block" role="img" aria-label="Decision tree diagram">`;
      const nodeText = (n, d) => n.leaf || d === levels - 1 ? `class ${n.p.indexOf(Math.max(...n.p))}` : `x${n.f + 1} ≤ ${n.t.toFixed(2)}?`;
      const walk = (n, d, x0, x1, px, py) => {
        const x = (x0 + x1) / 2, y = 26 + d * 64;
        if (px !== null) svg += `<line x1="${px}" y1="${py + 13}" x2="${x}" y2="${y - 13}" stroke="${col.line}" stroke-width="1.5"/>`;
        const leafish = n.leaf || d === levels - 1;
        const fill = leafish ? col.c[n.p.indexOf(Math.max(...n.p))] : col.surface2;
        const txt = nodeText(n, d);
        const w = Math.max(64, txt.length * 7.4 + 14);
        svg += `<rect x="${x - w / 2}" y="${y - 13}" width="${w}" height="26" rx="6" fill="${fill}" fill-opacity="${leafish ? 0.85 : 1}" stroke="${col.line}"/>`;
        svg += `<text x="${x}" y="${y + 4}" text-anchor="middle" font-family="IBM Plex Mono, monospace" font-size="11.5" fill="${leafish ? (n.p.indexOf(Math.max(...n.p)) === 2 ? "#1a1a1a" : "#ffffff") : col.ink}">${txt}</text>`;
        if (!leafish) {
          walk(n.l, d + 1, x0, x, x, y); walk(n.r, d + 1, x, x1, x, y);
          if (d === 0) svg += `<text x="${(x0 + x) / 2 + 30}" y="${y + 34}" font-size="11" fill="${col.muted}">yes</text><text x="${(x + x1) / 2 - 40}" y="${y + 34}" font-size="11" fill="${col.muted}">no</text>`;
        }
      };
      walk(t, 0, 0, W, null, null);
      box.innerHTML = svg + "</svg>";
    }
    fit();
    return disposers;
  },
});
