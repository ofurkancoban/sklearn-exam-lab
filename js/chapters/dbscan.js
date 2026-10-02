// Section 10: DBSCAN with the exact make_moons sample from the script.
ML.dbscan = (X, eps, minS) => {
  const n = X.length, nb = [];
  for (let i = 0; i < n; i++) { const a = []; for (let j = 0; j < n; j++) if (ML.dist(X[i], X[j]) <= eps) a.push(j); nb.push(a); }
  const core = nb.map((a) => a.length >= minS);
  const labels = new Array(n).fill(-1), order = [];
  let c = 0;
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1 || !core[i]) continue;
    const queue = [i]; labels[i] = c; order.push(i);
    while (queue.length) {
      const p = queue.shift();
      if (!core[p]) continue;
      for (const q of nb[p]) if (labels[q] === -1) { labels[q] = c; order.push(q); queue.push(q); }
    }
    c++;
  }
  return { labels, core, nb, nClusters: c, order };
};

ML.register("dbscan", {
  html: (m) => String.raw`
  ${ML.chHead(m, "<b>Connect examples in dense neighborhoods</b> and leave isolated examples unassigned. DBSCAN does not need the number of clusters in advance and can find curved groups, such as two moons.", ["unsup", "eps", "min_samples", "fit_predict"])}
  <div class="two-col">
    <div class="prose">
      <h2>Two settings define a dense neighborhood</h2>
      <ul>
        <li><code>eps</code> (ε): the distance used to decide which examples are nearby.</li>
        <li><code>min_samples</code>: how many examples a neighborhood needs for its center to be a <b>core point</b>. <b>This count includes the point itself.</b></li>
      </ul>
      <p>Nearby core points connect to form clusters. A <b>border point</b> is close to a core point but is not itself dense enough to be one. All other points are <b>noise</b> (label −1).</p>
    </div>
    ${ML.formula("Core point", String.raw`x_i \text{ is a core point if } |N_\varepsilon(x_i)| \ge \texttt{min\_samples}`, "N<sub>ε</sub>(x<sub>i</sub>) contains all examples whose distance from x<sub>i</sub> is at most ε. Around a set, the vertical bars <b>count</b> its elements (including x<sub>i</sub> itself); they do not mean absolute value.", "Eq. 14")}
  </div>

  <div class="section-title"><h2>Grow the clusters</h2><span class="hint">Hover a point to see its ε-neighborhood.</span></div>
  ${ML.bench("db", { tag: "DBSCAN(eps, min_samples).fit_predict(X)", stage: "tall", cls: "wide-panel", help: "Hover a point, drag to rotate" })}
  <p class="fig-cap">Large spheres are core points, small colored spheres are border points, small grey spheres are noise. The two-moons data is the exact sample from <code>make_moons(n_samples=300, noise=0.08, random_state=42)</code>, so the presets reproduce Exercise 4. The 3D chain shows that DBSCAN separates interlocked curved shapes that no straight cut could split.</p>

  <div class="two-col">
    ${ML.code(`
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
labels = DBSCAN(eps=0.25, min_samples=5).fit_predict(X)

n_clusters = len(set(labels) - {-1})
print("Clusters:", n_clusters)
print("Noise points:", (labels == -1).sum())

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("DBSCAN: each color is a cluster or noise label")
plt.show()`, `
Clusters: 2
Noise points: 0`)}
    <div class="prose">
      <ul>
        <li><code>fit_predict(X)</code> learns the grouping and returns a label for each input row.</li>
        <li>Labels such as 0 and 1 are <b>arbitrary cluster names</b>. Label <b>−1 means noise</b>, so <code>set(labels) - {-1}</code> excludes it from the cluster count.</li>
        <li>The underscore in <code>X, _</code> means we do not use the generator's true labels.</li>
      </ul>
      <h3>What happens when the settings change?</h3>
      <ul>
        <li>A <b>smaller eps</b> usually gives smaller groups and more noise.</li>
        <li>A <b>larger eps</b> can connect groups that should stay separate.</li>
        <li>A <b>larger min_samples</b> demands a denser neighborhood.</li>
      </ul>
    </div>
  </div>
  ${ML.callout("trap", "Exam traps", "<ul><li>DBSCAN uses distances, so feature units matter. Scale first for other data and choose eps in the scaled space.</li><li>It can struggle when clusters have <b>very different densities</b> (try the third dataset).</li><li>It is exploratory grouping, not a classifier trained on known answers. Scikit-learn's DBSCAN has <b>no general predict method</b> for new examples.</li><li>min_samples counts the point itself.</li></ul>")}
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="dbscan"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    const moons = ML.data.extra.moons;
    const chain = (() => {
      const r = ML.rng(12), X = [];
      for (let i = 0; i < 150; i++) { const t = r() * Math.PI * 2; X.push([Math.cos(t) + 0.06 * ML.randn(r), Math.sin(t) + 0.06 * ML.randn(r), 0.06 * ML.randn(r)]); }
      for (let i = 0; i < 150; i++) { const t = r() * Math.PI * 2; X.push([1 + Math.cos(t) + 0.06 * ML.randn(r), 0.06 * ML.randn(r), Math.sin(t) + 0.06 * ML.randn(r)]); }
      return X;
    })();
    const dens = (() => {
      const a = ML.makeBlobs([[-0.6, 0.2]], 140, 0.12, 31).X, b = ML.makeBlobs([[0.9, 0.3]], 60, 0.5, 32).X;
      return a.concat(b);
    })();
    const DS = {
      moons: { X: moons, c: [0.5, 0.25, 0], s: 1.7, dim: 2 },
      chain: { X: chain, c: [0.5, 0, 0], s: 1.35, dim: 3 },
      dens: { X: dens, c: [0.2, 0.25, 0], s: 1.7, dim: 2 },
    };
    const S = { ds: "moons", eps: 0.25, minS: 5, showEps: false, grow: 1 };
    const host = root.querySelector("#db-stage"), panel = root.querySelector("#db-panel");
    const st = new ML.Stage3D(host, { cam: [0.4, 6.4, 6.6], target: [0, 0.2, 0] });
    disposers.push(() => st.dispose());
    let res, sph, disks, hover = new THREE.Group(), frame;
    const W = (x) => { const d = DS[S.ds]; return ML.P((x[0] - d.c[0]) * d.s, (x[1] - d.c[1]) * d.s, ((x[2] || 0) - d.c[2]) * d.s + (d.dim === 2 ? 0.05 : 1.4)); };
    if (st.ok) {
      sph = new ML.Spheres(st, 300, 0.055);
      st.add(hover);
      ML.legend(host, [{ label: "core", color: col.c[1] }, { label: "border (small)", color: col.c[1] }, { label: "noise (−1)", color: col.noise }]);
    }
    ML.ui.heading(panel, "Data");
    ML.ui.seg(panel, { options: [{ v: "moons", t: "Two moons" }, { v: "chain", t: "3D chain" }, { v: "dens", t: "Mixed density" }], value: "moons", onChange: (v) => { S.ds = v; setFrame(); run(); } });
    ML.ui.heading(panel, "Settings");
    const eS = ML.ui.slider(panel, { label: "eps (ε)", min: 0.03, max: 0.8, step: 0.01, value: S.eps, onInput: (v) => { S.eps = v; run(); } });
    ML.ui.slider(panel, { label: "min_samples", min: 1, max: 20, value: S.minS, onInput: (v) => { S.minS = v; run(); } });
    const pr = ML.ui.row(panel);
    [0.1, 0.25, 0.5].forEach((e) => ML.ui.button(pr, `eps = ${e}`, () => { eS.set(e); S.eps = e; run(); }, true));
    ML.ui.check(panel, { label: "Show ε-neighborhoods of core points", onChange: (v) => { S.showEps = v; paint(); } });
    let growAnim = null;
    ML.ui.button(ML.ui.row(panel), "Animate cluster growth", () => {
      cancelAnimationFrame(growAnim);
      const t0 = performance.now();
      const go = (t) => { S.grow = Math.min(1, (t - t0) / 3500); paint(); if (S.grow < 1) growAnim = requestAnimationFrame(go); };
      growAnim = requestAnimationFrame(go);
    });
    disposers.push(() => cancelAnimationFrame(growAnim));
    const stats = ML.ui.stats(panel, [{ key: "c", label: "Clusters", big: true }, { key: "n", label: "Noise points (label −1)" }, { key: "core", label: "Core points" }, { key: "b", label: "Border points" }]);
    const verdict = ML.ui.verdict(panel);
    function setFrame() {
      if (!st.ok) return;
      if (frame) { st.scene.remove(frame); ML.disposeObj(frame); }
      frame = ML.axes(st, { x: [-3, 3], y: [-2, 2], labels: S.ds === "chain" ? null : ["Feature 1", "Feature 2", null], div: 8 });
      if (S.ds === "chain") st.flyTo([4.2, 4.2, 5.4], [0, 1.2, 0]); else st.flyTo([0.4, 6.4, 6.6], [0, 0.2, 0]);
    }
    function run() {
      const X = DS[S.ds].X;
      res = ML.dbscan(X, S.eps, S.minS);
      const noise = res.labels.filter((l) => l === -1).length, core = res.core.filter(Boolean).length;
      stats.set("c", res.nClusters); stats.set("n", noise); stats.set("core", core); stats.set("b", X.length - noise - core);
      if (S.ds === "moons") {
        const exp = { 0.1: "19 clusters, 60 noise", 0.25: "2 clusters, 0 noise", 0.5: "1 cluster, 0 noise" }[S.eps];
        verdict.innerHTML = (exp && S.minS === 5 ? `Exercise 4 with eps = ${S.eps}: scikit-learn reports <b>${exp}</b>. ` : "")
          + (res.nClusters === 2 && noise < 5 ? "Both moons are found as separate curved clusters." : res.nClusters > 2 ? "eps is too small: the moons break into many small pieces and many points become noise." : res.nClusters === 1 ? "eps is too large: the bridge between the moons connects them into one cluster." : "No dense region: everything is noise.");
      } else if (S.ds === "chain") verdict.innerHTML = res.nClusters === 2 ? "The two interlocked rings are separated: density follows each ring around its curve." : "Change eps until exactly two rings appear.";
      else verdict.innerHTML = "One eps cannot fit both densities: a small eps turns the sparse group into noise; a large eps lets the dense group swallow its surroundings.";
      S.grow = 1;
      paint();
    }
    function paint() {
      if (!st.ok) return;
      const X = DS[S.ds].X, d = DS[S.ds];
      const shown = new Set(res.order.slice(0, Math.floor(res.order.length * S.grow)));
      sph.setCount(X.length);
      X.forEach((x, i) => {
        const l = res.labels[i];
        const visible = S.grow >= 1 || shown.has(i);
        const c = l === -1 || !visible ? col.noise : ML.clusterColor(l);
        sph.set(i, W(x), c, l === -1 ? 0.7 : res.core[i] ? 1.25 : 0.8);
      });
      sph.commit();
      if (disks) { st.scene.remove(disks); ML.disposeObj(disks); disks = null; }
      if (S.showEps) {
        const ids = X.map((_, i) => i).filter((i) => res.core[i]);
        const geo = d.dim === 2 ? new THREE.CircleGeometry(1, 32).rotateX(-Math.PI / 2) : new THREE.SphereGeometry(1, 16, 12);
        disks = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial({ color: col.c[1], transparent: true, opacity: d.dim === 2 ? 0.05 : 0.025, depthWrite: false }), Math.max(1, ids.length));
        const dm = new THREE.Object3D();
        ids.forEach((i, k) => { dm.position.copy(W(X[i])); if (d.dim === 2) dm.position.y = 0.01; dm.scale.setScalar(S.eps * d.s); dm.updateMatrix(); disks.setMatrixAt(k, dm.matrix); });
        disks.count = ids.length;
        st.add(disks);
      }
    }
    // Hover: show the ε-neighborhood of one point.
    const tip = ML.tooltip(host);
    if (st.ok) {
      st.renderer.domElement.addEventListener("pointermove", (ev) => {
        const hit = st.raycast(ev, [sph.mesh])[0];
        st.clearGroup(hover);
        if (!hit) return tip.hide();
        const i = hit.instanceId, X = DS[S.ds].X, d = DS[S.ds];
        const geo = d.dim === 2 ? new THREE.CircleGeometry(S.eps * d.s, 48).rotateX(-Math.PI / 2) : new THREE.SphereGeometry(S.eps * d.s, 24, 16);
        const m = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col.accent, transparent: true, opacity: d.dim === 2 ? 0.2 : 0.12, depthWrite: false, side: THREE.DoubleSide }));
        m.position.copy(W(X[i])); if (d.dim === 2) m.position.y = 0.02;
        hover.add(m);
        res.nb[i].forEach((j) => { if (j !== i) hover.add(ML.line([W(X[i]), W(X[j])], new THREE.Color(col.accent), { opacity: 0.7 })); });
        const kind = res.core[i] ? "core point" : res.labels[i] === -1 ? "noise" : "border point";
        const rect = host.getBoundingClientRect();
        tip.show(ev.clientX - rect.left, ev.clientY - rect.top, `<b>${kind}</b> · label ${res.labels[i]}<br>|N<sub>ε</sub>| = ${res.nb[i].length} (incl. itself)<br>min_samples = ${S.minS}`);
      });
      st.renderer.domElement.addEventListener("pointerleave", () => { st.clearGroup(hover); tip.hide(); });
    }
    setFrame();
    run();

    // ----- Step-by-step tour -----
    const drv = ML.drive(panel);
    const moonsOf = (eps, m) => { drv.seg("Two moons"); drv.slider("eps", eps); drv.slider("min_samples", m); };
    const disc = (ann, i, color) => {
      if (!st.ok) return;
      const X = DS[S.ds].X, d = DS[S.ds];
      const m = new THREE.Mesh(new THREE.CircleGeometry(S.eps * d.s, 48).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide }));
      m.position.copy(W(X[i])); m.position.y = 0.02; ann.add(m);
      ann.add(ML.ring(ann, W(X[i]).setY(0.03), S.eps * d.s, color));
      res.nb[i].forEach((j) => { if (j !== i) ann.add(ML.line([W(X[i]), W(X[j])], new THREE.Color(color), { opacity: 0.8 })); });
    };
    const pick = (fn) => DS[S.ds].X.findIndex((_, i) => fn(i));
    const camAt = (i, dist = 2.2) => { const p = W(DS[S.ds].X[i]); return [[p.x + 0.2, p.y + dist * 1.3, p.z + dist], [p.x, p.y, p.z]]; };
    let coreI = -1, borderI = -1;
    const getCore = () => { if (coreI < 0) coreI = pick((i) => res.core[i] && res.nb[i].length >= 9); return coreI; };
    const TOP = [[0.4, 6.4, 6.6], [0, 0.2, 0]];
    const tour = ML.tour(host.parentElement, st, [
      { t: "Data without labels", cam: TOP,
        run: () => { moonsOf(0.25, 5); S.grow = 0; paint(); return () => { S.grow = 1; paint(); }; },
        d: "300 points from <code>make_moons</code>. Clustering is unsupervised: DBSCAN sees only the coordinates, no labels, and it is <b>not told how many groups</b> exist." },
      { t: "Draw the ε-neighborhood of one point", cam: () => camAt(coreI),
        run: (ann) => { moonsOf(0.25, 5); S.grow = 0; paint(); getCore(); disc(ann, coreI, col.accent); if (st.ok) ML.tag(ann, `|Nε| = ${res.nb[coreI].length}`, W(DS[S.ds].X[coreI]).add(new THREE.Vector3(0, 0.55, 0)), col.ink, 0.22); return () => { S.grow = 1; paint(); }; },
        d: () => `The circle has radius eps = ${S.eps}. Every point inside it is a neighbor. This point has <b>${res.nb[coreI].length}</b> points in its neighborhood, counting itself.` },
      { t: "Core point: at least min_samples neighbors", cam: () => camAt(coreI),
        run: (ann) => { moonsOf(0.25, 5); S.grow = 0; paint(); getCore(); disc(ann, coreI, col.good); if (st.ok) ML.tag(ann, "core point", W(DS[S.ds].X[coreI]).add(new THREE.Vector3(0, 0.55, 0)), col.good, 0.22); return () => { S.grow = 1; paint(); }; },
        d: () => `\\(|N_\\varepsilon(x_i)| = ${res.nb[coreI].length} \\ge \\texttt{min\\_samples} = ${S.minS}\\), so it is a <b>core point</b>: it sits in a dense region and may start or extend a cluster.` },
      { t: "Clusters grow from core to core", cam: TOP, ms: 9000,
        run: () => { moonsOf(0.25, 5); drv.btn("Animate cluster growth"); },
        d: "Starting at a core point, DBSCAN adds all its neighbors to the cluster. Every neighbor that is itself a core point adds <i>its</i> neighbors too, and so on, until no new points are reachable. Then it starts the next cluster. Result: <b>2 clusters</b>, following the curved moons." },
      { t: "Border points join but do not extend", cam: () => (borderI >= 0 ? camAt(borderI) : TOP),
        run: (ann) => { moonsOf(0.25, 12); borderI = pick((i) => !res.core[i] && res.labels[i] !== -1); if (borderI >= 0) { disc(ann, borderI, col.warn); if (st.ok) ML.tag(ann, "border point", W(DS[S.ds].X[borderI]).add(new THREE.Vector3(0, 0.55, 0)), col.warn, 0.22); } },
        d: () => borderI >= 0 ? `With min_samples = 12, this point has only ${res.nb[borderI].length} neighbors: not a core point. But it lies within eps of a core point, so it is a <b>border point</b> of that cluster (small sphere). Border points never pull further points in.` : "Border points are close to a core point but not dense enough themselves." },
      { t: "Too small eps: fragments and noise", cam: TOP,
        run: () => { moonsOf(0.1, 5); },
        d: () => `eps = 0.1: most neighborhoods are too small to reach 5 points. The moons shatter into <b>${res.nClusters} clusters</b> and <b>${res.labels.filter((l) => l === -1).length}</b> points are noise (grey, label −1), exactly what scikit-learn reports in Exercise 4.` },
      { t: "Too large eps: the moons merge", cam: TOP,
        run: () => { moonsOf(0.5, 5); },
        d: () => `eps = 0.5 reaches across the gap between the moons, so core points connect them into <b>${res.nClusters} cluster</b>. A good eps (0.25 here) lies between these extremes. Because eps is a distance, feature scaling matters.` },
      { t: "Curved shapes in 3D", cam: [[4.2, 4.2, 5.4], [0, 1.2, 0]],
        run: () => { drv.seg("3D chain"); drv.slider("eps", 0.25); drv.slider("min_samples", 5); },
        d: () => `Two interlocked rings. No straight cut can separate them, but density follows each ring around its curve: DBSCAN finds <b>${res.nClusters}</b> clusters. Remember: it labels only the fitted data and has no predict method for new points.` },
    ], { title: "How DBSCAN finds clusters" });
    disposers.push(() => tour.dispose());
    return disposers;
  },
});
