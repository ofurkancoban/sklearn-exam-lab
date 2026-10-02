// Section 12: metric MDS solved with SMACOF iterations, animated step by step.
ML.mdsStress = (D, Z) => {
  let s = 0;
  for (let i = 0; i < Z.length; i++) for (let j = i + 1; j < Z.length; j++) s += (D[i][j] - ML.dist(Z[i], Z[j])) ** 2;
  return s;
};
ML.smacofStep = (D, Z) => {
  const n = Z.length, out = Z.map(() => [0, 0]);
  for (let i = 0; i < n; i++) {
    let bii = 0;
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const dh = ML.dist(Z[i], Z[j]);
      const b = dh > 1e-12 ? -D[i][j] / dh : 0;
      bii -= b;
      out[i][0] += b * Z[j][0]; out[i][1] += b * Z[j][1];
    }
    out[i][0] += bii * Z[i][0]; out[i][1] += bii * Z[i][1];
  }
  return out.map(([a, b]) => [a / n, b / n]);
};

ML.register("mds", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Place examples on a map so that the <b>distances between them resemble their original distances</b>.", ["unsup", "n_components", "fit_transform"])}
  ${ML.formula("Raw stress: how much the map distorts distances", String.raw`\text{Stress} = \sum_{i \lt j} \big(d_{ij} - \hat{d}_{ij}\big)^2`, "d<sub>ij</sub>: distance between examples i and j before mapping. d̂<sub>ij</sub>: their distance on the 2D map. The sum covers <b>every distinct pair once</b>. A perfect distance-preserving map would have zero stress, but <b>this may be impossible in only two dimensions</b>. (This is raw stress; normalized variants also exist.)", "Eq. 15")}

  <div class="section-title"><h2>Watch MDS build a map</h2><span class="hint">Original points on the left, the 2D map on the right.</span></div>
  <div class="two-col">
    ${ML.bench("md", { tag: "Original space", stage: "", cls: "stack" })}
    <div class="bench stack">
      <div class="stage" id="md2-stage" style="background:var(--surface)"><span class="stage-tag">2D map (MDS dimension 1 and 2)</span><div id="md-map" style="position:absolute;inset:30px 6px 6px 6px"></div></div>
    </div>
  </div>
  <div class="bench wide-panel">
    <div class="stage short" style="background:var(--surface)"><span class="stage-tag">Shepard plot: original distance vs. map distance for every pair</span><div id="md-shep" style="position:absolute;inset:30px 6px 6px 6px"></div></div>
    <div class="panel" id="mds-ctrl"></div>
  </div>
  <p class="fig-cap">MDS starts from random map positions and repeatedly moves points to reduce stress (SMACOF algorithm, the one scikit-learn uses). In the Shepard plot each dot is one pair: on the diagonal, the map keeps that distance exactly. Hover a point on the map to highlight it in 3D. A dome cannot be flattened without distortion, so its stress never reaches zero.</p>

  <div class="table-wrap"><table>
    <thead><tr><th></th><th>PCA</th><th>Metric MDS</th></tr></thead>
    <tbody>
      <tr><td><b>Main idea</b></td><td>Retain variance in a few linear directions.</td><td>Match pairwise distances.</td></tr>
      <tr><td><b>What is learned?</b></td><td>Axes for projecting examples.</td><td>Positions of the fitted examples.</td></tr>
      <tr><td><b>New examples?</b></td><td><code>transform</code> projects them.</td><td>No <code>transform</code>; only <code>fit_transform</code>.</td></tr>
      <tr><td><b>Speed</b></td><td>Fast.</td><td>Often slower (all pairs, many iterations).</td></tr>
    </tbody>
  </table></div>

  <div class="two-col">
    ${ML.code(`
from sklearn.manifold import MDS

# Reuse X_scaled, y and the plotting import from the PCA example
mds = MDS(n_components=2, n_init=4, random_state=42)
Z = mds.fit_transform(X_scaled)
plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis")
plt.xlabel("MDS dimension 1")
plt.ylabel("MDS dimension 2")
plt.title("Iris: MDS")
plt.show()`, "")}
    <div class="prose">
      <p><code>n_init=4</code> runs MDS from four different random starts and keeps the map with the lowest stress (the optimization can end in different local solutions).</p>
      <p>Stress measures distance distortion; <b>lower stress means a closer match for the same data and settings</b>. MDS focuses on distances between points rather than finding principal directions.</p>
      ${ML.callout("trap", "Exam traps", "<ul><li>Stress sums over pairs i &lt; j, each pair once.</li><li>Zero stress is not always reachable in 2D.</li><li>The map can be rotated or mirrored without changing any distance, so the axes have no fixed meaning.</li><li>The plots explore the whole dataset on purpose; visual separation does not establish test accuracy.</li></ul>")}
    </div>
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="mds"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const dome = (() => {
      const r = ML.rng(4), X = [], y = [];
      for (let i = 0; i < 90; i++) {
        const lat = (Math.asin(r()) * 180) / Math.PI * 0.85, lon = r() * 360;
        const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180;
        X.push([Math.cos(la) * Math.cos(lo) * 1.6, Math.cos(la) * Math.sin(lo) * 1.6, Math.sin(la) * 1.6]);
        y.push(lat < 22 ? 0 : lat < 45 ? 1 : 2);
      }
      return { X, y, note: "Points on a dome (like cities on a globe). Colors: latitude bands." };
    })();
    const blobs = (() => {
      const d = ML.makeBlobs([[-1.3, 0, 0], [1.3, 0.2, 0.3], [0, 1.2, 1.6]], 30, 0.35, 8);
      return { X: d.X, y: d.y, note: "Three groups in 3D." };
    })();
    const iris = (() => {
      const D = ML.data.iris, Xs = ML.scaler(D.X).transform(D.X);
      return { X: Xs, y: D.y, note: "Iris, all four standardized features (the 3D view shows only the first three)." };
    })();
    const DS = { dome, blobs, iris };
    const S = { ds: "dome", it: 0, running: false };
    let D, Z, stressHist = [], hov = -1;
    const host = root.querySelector("#md-stage");
    const st = new ML.Stage3D(host, { cam: [4.4, 3.6, 4.8], target: [0, 0.6, 0] });
    disposers.push(() => st.dispose());
    let sph, hl = new THREE.Group(), frame;
    if (st.ok) { sph = new ML.Spheres(st, 150, 0.06); st.add(hl); }
    const map = new ML.Plot2D(root.querySelector("#md-map"), { height: 400, xlim: [-3, 3], ylim: [-3, 3], pad: [10, 10, 30, 36], aria: "MDS map" });
    const shep = new ML.Plot2D(root.querySelector("#md-shep"), { height: 300, xlim: [0, 4], ylim: [0, 4], pad: [10, 14, 34, 44], aria: "Shepard plot" });
    disposers.push(() => { map.dispose(); shep.dispose(); });
    const panel = root.querySelector("#mds-ctrl");
    ML.ui.heading(panel, "Data");
    ML.ui.seg(panel, { options: [{ v: "dome", t: "Dome" }, { v: "blobs", t: "3 groups" }, { v: "iris", t: "Iris (4D)" }], value: "dome", onChange: (v) => { S.ds = v; load(); } });
    const note = ML.el("div", { class: "ctrl-note" }); panel.append(note);
    const row = ML.ui.row(panel);
    const runBtn = ML.ui.button(row, "Run MDS", () => { S.running = !S.running; runBtn.textContent = S.running ? "Pause" : "Continue"; });
    ML.ui.button(row, "Random restart", () => { initZ(S.it + 7); }, true);
    const stats = ML.ui.stats(panel, [{ key: "it", label: "Iteration" }, { key: "s", label: "Raw stress", big: true }, { key: "pca", label: "Stress of the PCA map" }]);
    const verdict = ML.ui.verdict(panel);

    function load() {
      const X = DS[S.ds].X;
      note.textContent = DS[S.ds].note;
      D = X.map((a) => X.map((b) => ML.dist(a, b)));
      const P = ML.pca(X), Zp = P.project(X, 2);
      stats.set("pca", ML.mdsStress(D, Zp).toFixed(1));
      if (st.ok) {
        if (frame) { st.scene.remove(frame); ML.disposeObj(frame); }
        frame = ML.axes(st, { x: [-2.5, 2.5], y: [-2.5, 2.5], labels: S.ds === "iris" ? ["sepal length", "sepal width", "petal length"] : ["x", "y", "z"], div: 6 });
        sph.setCount(X.length);
        X.forEach((x, i) => sph.set(i, ML.P(x[0], x[1], (x[2] || 0) + (S.ds === "dome" ? 0 : 1.2)), col.c[DS[S.ds].y[i]]));
        sph.commit();
      }
      const maxD = Math.max(...D.flat());
      shep.o.xlim = [0, maxD * 1.05]; shep.o.ylim = [0, maxD * 1.05];
      initZ(1);
    }
    function initZ(seed) {
      const r = ML.rng(seed * 31 + 5), X = DS[S.ds].X;
      Z = X.map(() => [ML.randn(r) * 0.5, ML.randn(r) * 0.5]);
      S.it = 0; stressHist = [];
      S.running = true; runBtn.textContent = "Pause";
      drawAll();
    }
    function drawAll() {
      const s = ML.mdsStress(D, Z);
      stats.set("it", S.it); stats.set("s", s.toFixed(1));
      stressHist.push(s);
      const conv = stressHist.length > 3 && Math.abs(stressHist.at(-2) - s) < 1e-4 * s;
      verdict.innerHTML = S.it === 0 ? "Random start: map distances have nothing to do with the original ones yet."
        : conv ? `Converged. ${S.ds === "dome" ? "Stress stays above zero: a curved dome cannot be flattened exactly." : "The groups keep their distances well."}`
        : "Each iteration moves every point to reduce the stress.";
      if (conv && S.running) { S.running = false; runBtn.textContent = "Continue"; }
      // Map.
      const ext = Math.max(2.4, ...Z.flat().map(Math.abs)) * 1.1;
      map.o.xlim = [-ext, ext]; map.o.ylim = [-ext, ext];
      map.clear();
      map.frame({ xticks: ML.ticks(-ext, ext, 5), yticks: ML.ticks(-ext, ext, 5), xlabel: "MDS dimension 1", ylabel: "MDS dimension 2" });
      const yl = DS[S.ds].y;
      if (hov >= 0) Z.forEach((z, j) => { if (j !== hov) { const ratio = ML.dist(Z[hov], z) / Math.max(D[hov][j], 1e-9); map.path([Z[hov], z], ratio > 1 ? col.bad : col.c[1], 0.7); } });
      Z.forEach((z, i) => map.dot(z[0], z[1], i === hov ? 7 : 4.2, col.c[yl[i]], col.surface, 1));
      if (hov >= 0) map.text("red: stretched on the map · teal: squeezed", 44, 18, col.muted);
      // Shepard plot.
      shep.clear();
      shep.frame({ xticks: ML.ticks(...shep.o.xlim, 5), yticks: ML.ticks(...shep.o.ylim, 5), xlabel: "original distance d_ij", ylabel: "map distance d̂_ij" });
      const ctx = shep.ctx;
      ctx.fillStyle = col.accent; ctx.globalAlpha = 0.25;
      const n = Z.length, step = n > 100 ? 3 : 1;
      for (let i = 0; i < n; i += 1) for (let j = i + 1; j < n; j += step) {
        const dh = ML.dist(Z[i], Z[j]);
        ctx.fillRect(shep.X(D[i][j]) - 1, shep.Y(Math.min(dh, shep.o.ylim[1])) - 1, 2, 2);
      }
      ctx.globalAlpha = 1;
      shep.path([[0, 0], [shep.o.xlim[1], shep.o.xlim[1]]], col.ink, 1.5, [6, 4]);
      // 3D highlight.
      if (st.ok) {
        st.clearGroup(hl);
        if (hov >= 0) {
          const X = DS[S.ds].X, P3 = (x) => ML.P(x[0], x[1], (x[2] || 0) + (S.ds === "dome" ? 0 : 1.2));
          const ring = new THREE.Mesh(new THREE.SphereGeometry(0.13, 16, 12), new THREE.MeshBasicMaterial({ color: col.ink, wireframe: true }));
          ring.position.copy(P3(X[hov])); hl.add(ring);
        }
      }
    }
    map.drawFn = drawAll; shep.drawFn = () => {};
    map.canvas.addEventListener("pointermove", (ev) => {
      const r = map.canvas.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
      let best = -1, bd = 12;
      Z.forEach((z, i) => { const d = Math.hypot(map.X(z[0]) - px, map.Y(z[1]) - py); if (d < bd) { bd = d; best = i; } });
      if (best !== hov) { hov = best; stressHist.pop(); drawAll(); }
    });
    map.canvas.addEventListener("pointerleave", () => { hov = -1; stressHist.pop(); drawAll(); });
    let raf, last = 0;
    const loop = (t) => {
      raf = requestAnimationFrame(loop);
      if (!S.running || t - last < 60) return;
      last = t;
      Z = ML.smacofStep(D, Z); S.it++;
      drawAll();
    };
    raf = requestAnimationFrame(loop);
    disposers.push(() => cancelAnimationFrame(raf));
    load();
    return disposers;
  },
});
