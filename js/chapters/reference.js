// Section 15: quick reference, formula sheet, exercises with solutions, challenge appendix.
ML.register("reference", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Everything on one page: which tool does what, the main verbs, every formula from the script, the five exercises with worked solutions, and the course challenge rules.", ["cheat sheet", "exercises"])}

  <div class="section-title"><h2>Which tool does what?</h2></div>
  <div class="table-wrap print-area"><table>
    <thead><tr><th>Method</th><th>Task</th><th>Remember the idea</th><th>First setting</th><th>Scale?</th></tr></thead>
    <tbody>
      <tr><td><a href="#knn">KNN</a></td><td>classification</td><td>Ask nearby training examples.</td><td><code>n_neighbors</code></td><td>yes</td></tr>
      <tr><td><a href="#svm">SVM</a></td><td>classification</td><td>Separate classes with a margin.</td><td><code>C</code></td><td>yes</td></tr>
      <tr><td><a href="#forest">Random Forest</a></td><td>classification</td><td>Combine many decision trees.</td><td><code>n_estimators</code></td><td>usually not</td></tr>
      <tr><td><a href="#linreg">Linear regression</a></td><td>regression</td><td>Add weighted features.</td><td>usually defaults</td><td>not required</td></tr>
      <tr><td><a href="#lasso">Lasso</a></td><td>regression</td><td>Shrink linear-model weights.</td><td><code>alpha</code></td><td>yes</td></tr>
      <tr><td><a href="#poly">Polynomial regression</a></td><td>regression</td><td>Add powers to fit curves.</td><td><code>degree</code></td><td>-</td></tr>
      <tr><td><a href="#mlp">MLP</a></td><td>classification (MLPRegressor: numbers)</td><td>Learn weights across layers.</td><td><code>hidden_layer_sizes</code></td><td>yes</td></tr>
      <tr><td><a href="#dbscan">DBSCAN</a></td><td>clustering</td><td>Connect dense neighborhoods.</td><td><code>eps</code></td><td>yes (distances)</td></tr>
      <tr><td><a href="#pca">PCA</a></td><td>dimensionality reduction</td><td>Keep directions of large variance.</td><td><code>n_components</code></td><td>standardize first</td></tr>
      <tr><td><a href="#mds">MDS</a></td><td>dimensionality reduction</td><td>Preserve distances on a map.</td><td><code>n_components</code></td><td>yes (distances)</td></tr>
    </tbody>
  </table></div>

  <div class="two-col">
    <div class="prose">
      <h2>The main scikit-learn verbs</h2>
      <div class="table-wrap"><table><tbody>
        <tr><td><code>fit</code></td><td>learn from data</td><td>all estimators</td></tr>
        <tr><td><code>predict</code></td><td>produce answers</td><td>KNN, SVC, forest, regressions, MLP</td></tr>
        <tr><td><code>transform</code></td><td>change the representation of data</td><td>StandardScaler, PCA, PolynomialFeatures (not MDS!)</td></tr>
        <tr><td><code>fit_transform</code></td><td>learn the transformation and apply it</td><td>scaler, PCA, MDS</td></tr>
        <tr><td><code>fit_predict</code></td><td>learn a grouping, return labels for the fitted data</td><td>DBSCAN</td></tr>
      </tbody></table></div>
    </div>
    <div class="prose">
      <h2>Settings in one sentence</h2>
      <ul>
        <li><code>random_state=42</code>: repeatable randomness; 42 has no special meaning.</li>
        <li><code>stratify=y</code>: keep class proportions in train and test.</li>
        <li><code>test_size=0.2</code>: 20% held out for the final check.</li>
        <li><code>cv=3</code>: 3-fold cross-validation inside the training set.</li>
        <li><code>step__param</code>: two underscores address a pipeline step's setting.</li>
        <li><code>"passthrough"</code>: skip a pipeline step.</li>
        <li><code>zero_division=0</code>: report 0 for undefined precision or recall.</li>
      </ul>
    </div>
  </div>

  <div class="section-title"><h2>Formula sheet</h2><span class="hint">Equation numbers follow the script.</span><button type="button" class="btn ghost" id="ref-print" style="margin-left:auto">Print formula sheet</button></div>
  <div class="two-col print-area" id="ref-formulas"></div>

  <div class="section-title"><h2>Five exercises, solved</h2><span class="hint">Try first. Then open the solution; the output is from a real run.</span></div>
  <div id="ref-ex" style="display:grid;gap:14px"></div>

  <div class="section-title"><h2>Optional course challenge</h2></div>
  <div class="two-col">
    <div class="prose">
      <p>The Machine Learning Challenge 2026 uses separate files: <code>train.csv</code> (features, an <code>id</code>, and a target), <code>test.csv</code> (features and <code>id</code>, no target) and <code>submission_template.csv</code>. Treat <code>id</code> as the row identifier: keep it for the submission, use the feature columns as model inputs.</p>
      <ol>
        <li>Use only the supplied data. Split labeled training data for validation.</li>
        <li>Build a reproducible workflow that includes preprocessing and fitting.</li>
        <li>Compare models using the challenge's metric. Record the validation strategy and estimated score.</li>
        <li>Once the choice is fixed, refit on all labeled training rows and predict the supplied test rows.</li>
      </ol>
      <p>Submit one CSV per challenge with exactly two columns <code>id,prediction</code>. Classification: labels 0 or 1. Regression: finite numbers. Every test id exactly once, in the supplied order. F1 is computed for class 1.</p>
    </div>
    <div class="table-wrap"><table>
      <thead><tr><th>Challenge folder</th><th>Task</th><th>Metric</th></tr></thead>
      <tbody>
        <tr><td class="mono">challenge_01_the_final_boss</td><td>Classification</td><td>Accuracy ↑</td></tr>
        <tr><td class="mono">challenge_02_the_scaling_trap</td><td>Regression</td><td>RMSE ↓</td></tr>
        <tr><td class="mono">challenge_03_three_worlds</td><td>Classification</td><td>Accuracy ↑</td></tr>
        <tr><td class="mono">challenge_04_needle_in_a_haystack</td><td>Classification</td><td>F1 ↑</td></tr>
        <tr><td class="mono">challenge_05_hidden_xor</td><td>Classification</td><td>F1 ↑</td></tr>
      </tbody>
    </table></div>
  </div>
  `,
  init(root) {
    const F = [
      ["Eq. 1 · KNN distance", String.raw`\operatorname{dist}(x,z)=\sqrt{\textstyle\sum_{j=1}^{p}(x_j-z_j)^2}`, "#knn"],
      ["Eq. 2 · SVM decision rule", String.raw`f(x)=w^\top x+b,\quad \hat y=\operatorname{sign}(f(x))`, "#svm"],
      ["Eq. 3 · Margin width", String.raw`\frac{2}{\lVert w\rVert_2},\quad \lVert w\rVert_2=\sqrt{\textstyle\sum_j w_j^2}`, "#svm"],
      ["Eq. 4 · Hard margin", String.raw`\min_{w,b}\tfrac12\lVert w\rVert_2^2\ \text{s.t.}\ y_i(w^\top x_i+b)\ge1`, "#svm"],
      ["Eq. 5 · Soft margin", String.raw`\min\tfrac12\lVert w\rVert_2^2+C\textstyle\sum_i\xi_i\ \text{s.t.}\ y_i(w^\top x_i+b)\ge1-\xi_i,\ \xi_i\ge0`, "#svm"],
      ["Eq. 6 · RBF kernel", String.raw`K(x,z)=\exp(-\gamma\lVert x-z\rVert_2^2)`, "#svm"],
      ["Forest average", String.raw`\bar q(c\mid x)=\frac1T\textstyle\sum_{t=1}^{T}q_t(c\mid x)`, "#forest"],
      ["Eq. 7 · Standardization", String.raw`x'=\frac{x-\mu}{\sigma}`, "#metrics"],
      ["Eq. 8 · Classification metrics", String.raw`\text{Acc}=\tfrac{TP+TN}{TP+TN+FP+FN},\ \text{Prec}=\tfrac{TP}{TP+FP},\ \text{Rec}=\tfrac{TP}{TP+FN},\ F_1=\tfrac{2TP}{2TP+FP+FN}`, "#metrics"],
      ["Eq. 9 · Linear regression", String.raw`\hat y=b+\textstyle\sum_{j=1}^{p}w_jx_j`, "#linreg"],
      ["Eq. 10 · Regression errors", String.raw`\text{MAE}=\tfrac1n\textstyle\sum|e_i|,\ \text{MSE}=\tfrac1n\sum e_i^2,\ \text{RMSE}=\sqrt{\text{MSE}}`, "#linreg"],
      ["R²", String.raw`R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}{\sum_i(y_i-\bar y)^2}`, "#linreg"],
      ["Eq. 11 · Lasso objective", String.raw`\tfrac{1}{2n}\textstyle\sum_i(y_i-\hat y_i)^2+\alpha\sum_j|w_j|`, "#lasso"],
      ["Eq. 12 · Polynomial (degree 2)", String.raw`\hat y=b+w_1x+w_2x^2`, "#poly"],
      ["Eq. 13 · ReLU neuron", String.raw`h=\max\!\big(0,\ b+\textstyle\sum_j w_jx_j\big)`, "#mlp"],
      ["Eq. 14 · DBSCAN core point", String.raw`|N_\varepsilon(x_i)|\ge\texttt{min\_samples}`, "#dbscan"],
      ["PCA coordinate", String.raw`z=\textstyle\sum_{j=1}^{p}v_j(x_j-\mu_j)`, "#pca"],
      ["Eq. 15 · MDS raw stress", String.raw`\text{Stress}=\textstyle\sum_{i \lt j}(d_{ij}-\hat d_{ij})^2`, "#mds"],
      ["Eq. 16 · CV mean score", String.raw`\bar s=\frac1K\textstyle\sum_{r=1}^{K}s_r`, "#cv"],
    ];
    const box = root.querySelector("#ref-formulas");
    box.innerHTML = F.map(([l, t, h]) => `<div class="formula"><div class="f-label"><span>${l}</span><a href="${h}">open section</a></div><div class="f-body">\\[${t}\\]</div></div>`).join("");
    ML.typesetLater(box);
    // Printing is unavailable inside an embedded viewer, so the button only shows in a normal tab.
    const pb = root.querySelector("#ref-print");
    if (window.self !== window.top) pb.hidden = true;
    pb.addEventListener("click", () => {
      document.body.classList.add("print-sheet");
      const off = () => { document.body.classList.remove("print-sheet"); window.removeEventListener("afterprint", off); };
      window.addEventListener("afterprint", off);
      window.print();
      setTimeout(off, 1000);
    });
    const ex = root.querySelector("#ref-ex");
    ML.exercises.forEach((e, i) => {
      const d = ML.el("div", { class: "q-card" });
      d.innerHTML = `<div class="q-meta"><span>Exercise ${i + 1}</span><span>${e.title}</span></div><div class="q-text">${e.task}</div>
        <details><summary class="btn ghost" style="display:inline-block;list-style:none">Show solution</summary>
        <div style="display:grid;gap:12px;margin-top:12px">${ML.code(e.code, e.out)}<div class="q-explain">${e.why}</div></div></details>`;
      ex.append(d);
    });
    ML.decorate(ex);
  },
});
