// Section 11: PCA in three animated steps (center, find directions, project),
// plus the real Iris PCA of all four standardized features.
ML.register("pca", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Replace many features with a few new coordinates. Imagine a cloud of points: <b>PCA finds the directions in which the cloud spreads out most</b>. Keeping two directions lets us draw the data in 2D.", ["unsup", "n_components", "fit_transform"])}

  <div class="section-title"><h2>PCA in three steps</h2><span class="hint">Press the steps in order and rotate the cloud between them.</span></div>
  ${ML.bench("pc", { tag: "A 3D cloud reduced to 2D", stage: "tall" })}
  <p class="fig-cap">Arrow lengths are proportional to the spread (standard deviation) along each principal direction. In step 3 every point drops straight onto the PC1-PC2 plane along PC3; the short grey sticks are the information that is lost.</p>

  <div class="two-col">
    ${ML.formula("1. Center the data", String.raw`x_j^{\text{centered}} = x_j - \mu_j`, "μ<sub>j</sub> is the mean of feature j. This moves the center of the cloud to zero. If features have different scales, <b>standardize them first</b>.")}
    ${ML.formula("3. Calculate the new coordinates", String.raw`z = \sum_{j=1}^{p} v_j (x_j - \mu_j)`, "v = (v<sub>1</sub>, …, v<sub>p</sub>) is a unit-length direction. The new coordinate is a weighted sum of the centered features. Two directions give two coordinates per example: the horizontal and vertical positions in the plot.")}
  </div>
  <div class="prose">
    <h3>2. Find new directions</h3>
    <p>The <b>first principal direction</b> follows the greatest spread of the data. The <b>second is perpendicular</b> to the first and follows the greatest remaining spread. PCA learns these directions <b>from the features alone, without using the class labels</b>.</p>
  </div>

  <div class="section-title"><h2>Iris: four features to two</h2><span class="hint">Computed here from the real data; hover a flower to see its z-values.</span></div>
  <div class="two-col">
    <div id="pc-iris"></div>
    <div class="prose">
      <div id="pc-evr"></div>
      <p class="fig-cap">Explained variance ratio: PC1 keeps 73.0% and PC2 22.9% of the total variance of the four standardized features (scikit-learn's <code>pca.explained_variance_ratio_</code> gives the same numbers). Together about 96%: the 2D picture loses little. Colors are the species, used <b>only for coloring</b>.</p>
      <div id="pc-load" class="table-wrap"></div>
    </div>
  </div>

  <div class="two-col">
    ${ML.code(`
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
Z = pca.fit_transform(X_scaled)

plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("Iris: PCA")
plt.show()`, "")}
    <div class="prose">
      ${ML.callout("setting", "Main setting: n_components", "Here it is 2, so <code>Z</code> has two columns. Reducing the number of coordinates <b>can lose information</b>.")}
      ${ML.callout("trap", "Exam traps", "<ul><li><b>PCA centers its input but does not automatically standardize</b> feature scales. That is why StandardScaler comes first.</li><li>This plot explores all rows on purpose; y only chooses colors. For predictive evaluation, put scaling and PCA <b>inside a pipeline fitted on training data only</b>.</li><li>PC2 is perpendicular (orthogonal) to PC1.</li><li>PCA is unsupervised: it never looks at y.</li></ul>")}
    </div>
  </div>
  <div class="section-title"><h2>Check yourself</h2></div>
  <div data-quiz="pca"></div>
  `,
  init(root) {
    const col = ML.colors(), disposers = [];
    // A stretched, rotated, shifted 3D Gaussian cloud.
    const r = ML.rng(9), n = 220, raw = [];
    const rot = (v, ax, ay) => {
      let [x, y, z] = v;
      [y, z] = [y * Math.cos(ax) - z * Math.sin(ax), y * Math.sin(ax) + z * Math.cos(ax)];
      [x, z] = [x * Math.cos(ay) + z * Math.sin(ay), -x * Math.sin(ay) + z * Math.cos(ay)];
      return [x, y, z];
    };
    for (let i = 0; i < n; i++) {
      const v = rot([ML.randn(r) * 1.25, ML.randn(r) * 0.55, ML.randn(r) * 0.2], 0.5, 0.65);
      raw.push([v[0] + 1.0, v[1] + 0.7, v[2] + 1.3]);
    }
    const P = ML.pca(raw);
    const host = root.querySelector("#pc-stage"), panel = root.querySelector("#pc-panel");
    const st = new ML.Stage3D(host, { cam: [6.2, 4.6, 6.8], target: [0.4, 1, 0] });
    disposers.push(() => st.dispose());
    const S = { center: 0, dirs: 0, proj: 0 };
    let sph, arrows = new THREE.Group(), sticks = new THREE.Group(), planeM, muMark;
    const cur = (i) => {
      const x = raw[i];
      const c = x.map((v, j) => v - S.center * P.mu[j]);
      if (S.proj === 0) return c;
      // Remove the PC3 component progressively (projection onto PC1-PC2 plane).
      const z3 = P.vectors[2].reduce((s, vj, j) => s + vj * (x[j] - P.mu[j]), 0);
      return c.map((v, j) => v - S.proj * z3 * P.vectors[2][j]);
    };
    if (st.ok) {
      ML.axes(st, { x: [-3, 3], y: [-3, 3], z: [-1.5, 3.5], labels: ["feature 1", "feature 2", "feature 3"], div: 6 });
      sph = new ML.Spheres(st, n, 0.05);
      st.add(arrows); st.add(sticks);
      muMark = new THREE.Mesh(new THREE.OctahedronGeometry(0.12), new THREE.MeshStandardMaterial({ color: col.bad }));
      st.add(muMark);
      planeM = new THREE.Mesh(new THREE.PlaneGeometry(7, 4), new THREE.MeshBasicMaterial({ color: col.accent, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
      const v1 = ML.P(...P.vectors[0]), v2 = ML.P(...P.vectors[1]), v3 = ML.P(...P.vectors[2]);
      planeM.matrixAutoUpdate = false;
      planeM.matrix.makeBasis(v1, v2, v3);
      st.add(planeM);
      ML.legend(host, [{ label: "example", color: col.c[0] }, { label: "mean μ", color: col.bad }, { label: "PC1", color: col.c[1] }, { label: "PC2", color: col.warn }, { label: "PC3", color: col.muted }]);
    }
    ML.ui.heading(panel, "Steps");
    const b1 = ML.ui.button(panel, "1. Center the data", () => tween("center", S.center ? 0 : 1));
    const b2 = ML.ui.button(panel, "2. Find the directions", () => { S.dirs = S.dirs ? 0 : 1; draw(); });
    const b3 = ML.ui.button(panel, "3. Project onto PC1 and PC2", () => { if (!S.center) tween("center", 1); S.dirs = 1; tween("proj", S.proj ? 0 : 1); });
    ML.ui.button(ML.ui.row(panel), "Look at the PC1-PC2 plane", () => {
      if (!st.ok) return;
      const v3 = ML.P(...P.vectors[2]).normalize().multiplyScalar(9);
      st.flyTo([v3.x, v3.y, v3.z], [0, 0, 0]);
    }, true);
    ML.ui.heading(panel, "Variance along each direction");
    const bars = ML.ui.bars(panel, [{ label: "PC1", color: col.c[1] }, { label: "PC2", color: col.warn }, { label: "PC3", color: col.muted }]);
    bars.set(P.ratio, (v) => `${(v * 100).toFixed(1)}%`);
    const verdict = ML.ui.verdict(panel);
    [b1, b2, b3].forEach((b) => (b.style.textAlign = "left"));
    let anim = null;
    function tween(key, to) {
      cancelAnimationFrame(anim);
      const from = S[key], t0 = performance.now();
      const go = (t) => { const k = ML.reducedMotion() ? 1 : ML.ease(Math.min(1, (t - t0) / 1300)); S[key] = ML.lerp(from, to, k); draw(); if (k < 1) anim = requestAnimationFrame(go); };
      anim = requestAnimationFrame(go);
    }
    disposers.push(() => cancelAnimationFrame(anim));
    function draw() {
      verdict.innerHTML = S.proj > 0.99 ? `Every point now has only two coordinates (z1, z2). Kept: ${((P.ratio[0] + P.ratio[1]) * 100).toFixed(1)}% of the variance; lost: ${(P.ratio[2] * 100).toFixed(1)}% (the grey sticks).`
        : S.dirs ? "PC1 follows the greatest spread. PC2 is perpendicular to PC1 and follows the greatest remaining spread. PC3 is what is left."
        : S.center > 0.99 ? "The mean (red diamond) now sits at the origin. Next: find the directions."
        : "The cloud is shifted away from the origin and stretched in one direction. Start with step 1.";
      if (!st.ok) return;
      raw.forEach((_, i) => sph.set(i, ML.P(...cur(i)), col.c[0]));
      sph.commit();
      const mu = P.mu.map((v) => v * (1 - S.center));
      muMark.position.copy(ML.P(...mu));
      st.clearGroup(arrows);
      if (S.dirs) {
        const cols = [col.c[1], col.warn, col.muted];
        P.vectors.forEach((v, k) => {
          const len = Math.sqrt(P.values[k]) * 2.2 + 0.15;
          const dir = ML.P(...v).normalize();
          const a = new THREE.ArrowHelper(dir, ML.P(...mu), len, new THREE.Color(cols[k]).getHex(), 0.22, 0.12);
          arrows.add(a);
          const lb = ML.label(`PC${k + 1}`, { h: 0.24, color: cols[k] });
          lb.position.copy(ML.P(...mu).add(dir.clone().multiplyScalar(len + 0.25)));
          arrows.add(lb);
        });
      }
      st.clearGroup(sticks);
      if (S.proj > 0.01) {
        const segs = raw.map((_, i) => { const a = cur(i); const save = S.proj; S.proj = 0; const b = cur(i); S.proj = save; return [ML.P(...a), ML.P(...b)]; });
        sticks.add(ML.segments(segs, new THREE.Color(col.muted), 0.5));
      }
      planeM.material.opacity = 0.16 * S.proj;
      planeM.matrix.setPosition(ML.P(...mu));
    }
    draw();

    // Iris PCA on all four standardized features.
    const D = ML.data.iris, sc = ML.scaler(D.X), Xs = sc.transform(D.X), IP = ML.pca(Xs);
    const Zi = IP.project(Xs, 2);
    const sgn = [1, 1];
    const Z = Zi.map(([a, b]) => [a * sgn[0], b * sgn[1]]);
    const ip = new ML.Plot2D(root.querySelector("#pc-iris"), { height: 380, xlim: [-3.2, 4], ylim: [-3, 3], aria: "Iris PCA" });
    disposers.push(() => ip.dispose());
    let hov = -1;
    const drawIris = () => {
      ip.clear();
      ip.frame({ xticks: [-3, -2, -1, 0, 1, 2, 3, 4], yticks: [-3, -2, -1, 0, 1, 2, 3], xlabel: "Principal component 1", ylabel: "Principal component 2" });
      Z.forEach((z, i) => ip.dot(z[0], z[1], i === hov ? 7 : 4.5, col.c[D.y[i]], col.surface, 1));
      D.names.forEach((nm, k) => ip.text(`● ${nm}`, ip.W - 110, 20 + k * 17, col.c[k]));
      if (hov >= 0) {
        const x = Xs[hov], v1 = IP.vectors[0].map((v) => v * sgn[0]);
        ip.text(`flower ${hov + 1}: z1 = Σ vⱼ(xⱼ − μⱼ) = ${v1.map((v, j) => `${v.toFixed(2)}·${x[j].toFixed(2)}`).join(" + ")} = ${Z[hov][0].toFixed(2)}`, 52, ip.H - 50, col.ink, "left", '500 11px "IBM Plex Mono", monospace');
      }
    };
    ip.drawFn = drawIris;
    ip.canvas.addEventListener("pointermove", (ev) => {
      const rct = ip.canvas.getBoundingClientRect(), px = ev.clientX - rct.left, py = ev.clientY - rct.top;
      let best = -1, bd = 12;
      Z.forEach((z, i) => { const d = Math.hypot(ip.X(z[0]) - px, ip.Y(z[1]) - py); if (d < bd) { bd = d; best = i; } });
      if (best !== hov) { hov = best; drawIris(); }
    });
    drawIris();
    const evr = root.querySelector("#pc-evr");
    const eb = ML.ui.bars(evr, IP.ratio.map((_, k) => ({ label: `PC${k + 1}`, color: k < 2 ? col.accent : col.muted })));
    eb.set(IP.ratio, (v) => `${(v * 100).toFixed(1)}%`);
    root.querySelector("#pc-load").innerHTML = `<table><thead><tr><th>Direction</th>${D.features.map((f) => `<th>${f}</th>`).join("")}</tr></thead><tbody>${[0, 1].map((k) => `<tr><td>v (PC${k + 1})</td>${IP.vectors[k].map((v) => `<td class="mono">${(v * sgn[k]).toFixed(3)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    return disposers;
  },
});
