// Router, navigation, progress tracking and theme switching.
(() => {
  const ORDER = [
    { id: "overview", n: "01", title: "The basic idea", group: "Start", pdf: "Script p. 1" },
    { id: "knn", n: "02", title: "K-nearest neighbors", group: "Classification", pdf: "Script p. 2" },
    { id: "svm", n: "03", title: "Support vector machines", group: "Classification", pdf: "Script pp. 3-4" },
    { id: "forest", n: "04", title: "Random forests", group: "Classification", pdf: "Script p. 5" },
    { id: "metrics", n: "05", title: "Scaling and checking predictions", group: "Classification", pdf: "Script p. 6" },
    { id: "linreg", n: "06", title: "Linear regression", group: "Regression", pdf: "Script p. 7" },
    { id: "lasso", n: "07", title: "Lasso", group: "Regression", pdf: "Script p. 8" },
    { id: "poly", n: "08", title: "Polynomial regression", group: "Regression", pdf: "Script p. 9" },
    { id: "mlp", n: "09", title: "Multilayer perceptron", group: "Neural network", pdf: "Script p. 10" },
    { id: "dbscan", n: "10", title: "DBSCAN", group: "Unsupervised", pdf: "Script pp. 11-12" },
    { id: "pca", n: "11", title: "PCA", group: "Unsupervised", pdf: "Script pp. 12-13" },
    { id: "mds", n: "12", title: "MDS", group: "Unsupervised", pdf: "Script pp. 14-15" },
    { id: "pcaclf", n: "13", title: "PCA before classification", group: "Model selection", pdf: "Script pp. 15-16" },
    { id: "cv", n: "14", title: "Choosing a model setting", group: "Model selection", pdf: "Script pp. 16-17" },
    { id: "reference", n: "15", title: "Quick reference", group: "Review", pdf: "Script p. 18" },
    { id: "extras", n: "+", title: "Beyond the script", group: "Review", pdf: "Long version only" },
    { id: "exam", n: "Q", title: "Exam practice", group: "Review", pdf: "All sections" },
  ];
  ML.order = ORDER;
  const main = document.getElementById("main");
  const nav = document.getElementById("nav");
  const rail = document.getElementById("rail");
  let current = null, cleanup = null, scrim = null;

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
  };
  ML.store = store;
  const done = new Set(store.get("mlsite-done", []));

  function buildNav() {
    nav.innerHTML = "";
    let group = null, box = null;
    ORDER.forEach((c) => {
      if (c.group !== group) {
        group = c.group;
        box = ML.el("div", { class: "nav-group" }, [ML.el("div", { class: "nav-group-title", text: group })]);
        nav.append(box);
      }
      box.append(ML.el("a", { class: "nav-link", href: `#${c.id}`, "data-id": c.id }, [
        ML.el("span", { class: "nav-num", text: c.n }), ML.el("span", { text: c.title }), ML.el("span", { class: "nav-check", "aria-hidden": "true" }),
      ]));
    });
  }
  function syncNav() {
    nav.querySelectorAll(".nav-link").forEach((a) => {
      a.classList.toggle("active", a.dataset.id === current);
      a.classList.toggle("done", done.has(a.dataset.id));
      if (a.dataset.id === current) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    const core = ORDER.filter((c) => c.id !== "exam" && c.id !== "extras");
    const total = core.length;
    const n = core.filter((c) => done.has(c.id)).length;
    document.getElementById("progress-text").textContent = `${n} / ${total}`;
    document.getElementById("progress-fill").style.width = `${(n / total) * 100}%`;
  }

  ML.pending = [];
  ML.onMathReady = () => { ML.pending.splice(0).forEach((r) => ML.typeset(r)); };
  const typeset = (root) => {
    if (window.MathJax && MathJax.typesetPromise && MathJax.startup && MathJax.startup.document) ML.typeset(root);
    else ML.pending.push(root);
  };
  ML.typesetLater = typeset;

  function render(id) {
    const ch = ML.chapters[id] ? id : "overview";
    if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
    current = ch;
    const meta = ORDER.find((c) => c.id === ch);
    const idx = ORDER.indexOf(meta);
    const def = ML.chapters[ch];
    main.innerHTML = "";
    const root = ML.el("article", { class: "chapter", id: `ch-${ch}` });
    root.innerHTML = def.html(meta);
    main.append(root);
    ML.decorate(root);
    root.querySelectorAll("[data-quiz]").forEach((el) => ML.quiz.mini(el, el.dataset.quiz));
    // Footer: mark as studied + prev/next.
    if (ch !== "exam") {
      const btn = ML.el("button", { type: "button", class: "btn done-btn" + (done.has(ch) ? " ghost" : "") });
      const sync = () => { btn.textContent = done.has(ch) ? "Studied. Click to unmark" : "Mark this section as studied"; btn.classList.toggle("ghost", done.has(ch)); };
      btn.addEventListener("click", () => { done.has(ch) ? done.delete(ch) : done.add(ch); store.set("mlsite-done", [...done]); sync(); syncNav(); });
      sync();
      root.append(btn);
    }
    const prev = ORDER[idx - 1], next = ORDER[idx + 1];
    root.append(ML.el("div", { class: "pager" }, [
      prev ? ML.el("a", { href: `#${prev.id}` }, [ML.el("small", { text: "Previous" }), ML.el("span", { text: prev.title })]) : ML.el("span"),
      next ? ML.el("a", { class: "next", href: `#${next.id}` }, [ML.el("small", { text: "Next" }), ML.el("span", { text: next.title })]) : null,
    ]));
    typeset(root);
    let disposers = [];
    try {
      const r = def.init ? def.init(root) : null;
      if (typeof r === "function") disposers.push(r);
      else if (Array.isArray(r)) disposers = r;
    } catch (e) { console.error(e); }
    cleanup = () => disposers.forEach((d) => d && d());
    if (ML.pendingFind) { const term = ML.pendingFind; ML.pendingFind = null; setTimeout(() => findInPage(root, term), 60); }
    syncNav();
    store.set("mlsite-last", ch);
    if (!ML.pendingFindScroll) window.scrollTo(0, 0);
    ML.pendingFindScroll = false;
    document.title = `${meta.title} | Scikit-Learn Exam Lab`;
    closeRail();
  }

  function route() {
    const id = (location.hash || "").replace("#", "") || store.get("mlsite-last", "overview");
    render(id);
  }

  // Mobile drawer.
  const menuBtn = document.getElementById("menu-btn");
  function openRail() {
    rail.classList.add("open"); menuBtn.setAttribute("aria-expanded", "true");
    scrim = ML.el("div", { class: "scrim", onclick: closeRail }); document.body.append(scrim);
  }
  function closeRail() {
    rail.classList.remove("open"); menuBtn.setAttribute("aria-expanded", "false");
    if (scrim) { scrim.remove(); scrim = null; }
  }
  menuBtn.addEventListener("click", () => (rail.classList.contains("open") ? closeRail() : openRail()));

  // Theme switching. 3D scenes read colors at build time, so the section re-renders.
  const themeBtns = document.querySelectorAll("[data-theme-set]");
  const syncThemeBtns = () => {
    const t = document.documentElement.getAttribute("data-theme") || "system";
    themeBtns.forEach((b) => b.setAttribute("aria-pressed", b.dataset.themeSet === t ? "true" : "false"));
  };
  themeBtns.forEach((b) => b.addEventListener("click", () => {
    const t = b.dataset.themeSet;
    if (t === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("mlsite-theme", t); } catch (e) {}
  }));
  let lastDark = ML.isDark();
  const onTheme = () => {
    syncThemeBtns();
    const d = ML.isDark();
    if (d !== lastDark) { lastDark = d; const y = window.scrollY; render(current); window.scrollTo(0, y); }
  };
  new MutationObserver(onTheme).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", onTheme);
  syncThemeBtns();

  // ---------- Search ----------
  let index = null;
  const strip = (h) => h.replace(/<style[\s\S]*?<\/style>/g, " ").replace(/<[^>]+>/g, " ").replace(/\\[()[\]]/g, " ").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&").replace(/\s+/g, " ");
  const buildIndex = () => {
    index = ORDER.map((c) => ({ c, text: strip(ML.chapters[c.id].html(c)) }));
    ML.questions.forEach((q) => { const e = index.find((x) => x.c.id === q.ch); if (e) e.text += " " + strip(q.q + " " + q.why); });
  };
  const sInput = document.getElementById("search");
  const sRes = document.getElementById("search-results");
  let hits = [], hitSel = 0;
  function search() {
    const term = sInput.value.trim().toLowerCase();
    if (term.length < 2) { sRes.hidden = true; hits = []; return; }
    if (!index) buildIndex();
    hits = [];
    index.forEach(({ c, text }) => {
      const low = text.toLowerCase();
      let pos = low.indexOf(term), count = 0;
      const first = pos;
      while (pos >= 0) { count++; pos = low.indexOf(term, pos + term.length); }
      if (count) hits.push({ c, count, snip: text.slice(Math.max(0, first - 40), first + term.length + 60).trim(), title: c.title.toLowerCase().includes(term) });
    });
    hits.sort((a, b) => (b.title - a.title) || (b.count - a.count));
    hitSel = 0;
    renderHits(term);
  }
  function renderHits(term) {
    sRes.innerHTML = "";
    if (!hits.length) { sRes.append(ML.el("div", { class: "s-empty", text: "No matches." })); sRes.hidden = false; return; }
    const esc = (t) => ML.esc(t);
    hits.slice(0, 8).forEach((h, i) => {
      const k = h.snip.toLowerCase().indexOf(term);
      const snip = k >= 0 ? `…${esc(h.snip.slice(0, k))}<mark>${esc(h.snip.slice(k, k + term.length))}</mark>${esc(h.snip.slice(k + term.length))}…` : esc(h.snip);
      const a = ML.el("a", { href: `#${h.c.id}`, class: "s-hit" + (i === hitSel ? " sel" : ""), html: `<b>${h.c.n} ${esc(h.c.title)}</b><span class="s-count">${h.count}×</span><small>${snip}</small>` });
      a.addEventListener("click", () => openHit(i, term));
      sRes.append(a);
    });
    sRes.hidden = false;
  }
  function openHit(i, term) {
    const h = hits[i]; if (!h) return;
    ML.pendingFind = term || sInput.value.trim();
    sRes.hidden = true;
    sInput.blur();
    if (location.hash === `#${h.c.id}`) render(h.c.id); else location.hash = h.c.id;
  }
  function findInPage(root, term) {
    const low = term.toLowerCase();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (n.parentElement.closest("script,style,mjx-container,.code-head,button") ? NodeFilter.FILTER_REJECT : n.nodeValue.toLowerCase().includes(low) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP) });
    const node = walker.nextNode();
    if (!node) return;
    const k = node.nodeValue.toLowerCase().indexOf(low);
    const range = document.createRange();
    range.setStart(node, k); range.setEnd(node, k + term.length);
    const mark = document.createElement("mark");
    mark.className = "find-hit";
    range.surroundContents(mark);
    mark.scrollIntoView({ block: "center" });
    setTimeout(() => mark.classList.add("fade"), 2500);
  }
  sInput.addEventListener("input", search);
  sInput.addEventListener("focus", search);
  sInput.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { hitSel = Math.min(hitSel + 1, Math.min(hits.length, 8) - 1); renderHits(sInput.value.trim().toLowerCase()); e.preventDefault(); }
    else if (e.key === "ArrowUp") { hitSel = Math.max(hitSel - 1, 0); renderHits(sInput.value.trim().toLowerCase()); e.preventDefault(); }
    else if (e.key === "Enter") { openHit(hitSel); e.preventDefault(); }
    else if (e.key === "Escape") { sRes.hidden = true; sInput.blur(); }
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search-wrap")) sRes.hidden = true; });

  // ---------- Keyboard shortcuts ----------
  const help = document.getElementById("kb-help");
  const toggleHelp = (show) => { help.hidden = show === undefined ? !help.hidden : !show; if (!help.hidden) help.querySelector("button").focus(); };
  help.querySelector("button").addEventListener("click", () => toggleHelp(false));
  help.addEventListener("click", (e) => { if (e.target === help) toggleHelp(false); });
  document.getElementById("kb-btn").addEventListener("click", () => toggleHelp(true));
  document.addEventListener("keydown", (e) => {
    const t = e.target, typing = t.closest && t.closest("input, textarea, select, [contenteditable]");
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (window.innerWidth <= 960) openRail(); sInput.focus(); sInput.select(); return; }
    if (e.key === "Escape") { toggleHelp(false); closeRail(); return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
    const idx = ORDER.findIndex((c) => c.id === current);
    if (e.key === "/") { e.preventDefault(); if (window.innerWidth <= 960) openRail(); sInput.focus(); }
    else if (e.key === "?") toggleHelp();
    else if ((e.key === "ArrowRight" || e.key === "]") && ORDER[idx + 1]) location.hash = ORDER[idx + 1].id;
    else if ((e.key === "ArrowLeft" || e.key === "[") && ORDER[idx - 1]) location.hash = ORDER[idx - 1].id;
    else if (e.key === "m" && current !== "exam") { const b = document.querySelector(".done-btn"); if (b) b.click(); }
    else if (e.key === "g") location.hash = "exam";
  });

  // ---------- Unique visitor counter ----------
  // Each browser increments the shared counter once; later visits only read it.
  (() => {
    const API = "https://abacus.jasoncameron.dev", NS = "ofurkancoban-sklearn-exam-lab", KEY = "unique-visitors";
    let counted = false;
    try { counted = localStorage.getItem("mlsite-counted") === "1"; } catch (e) {}
    const url = `${API}/${counted ? "get" : "hit"}/${NS}/${KEY}`;
    fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(r.status))).then((d) => {
      if (typeof d.value !== "number") return;
      if (!counted) { try { localStorage.setItem("mlsite-counted", "1"); } catch (e) {} }
      document.getElementById("visitors-n").textContent = d.value.toLocaleString("en-US");
      document.getElementById("visitors-label").textContent = d.value === 1 ? "unique visitor" : "unique visitors";
      document.getElementById("visitors").hidden = false;
    }).catch(() => {});
  })();

  buildNav();
  window.addEventListener("hashchange", route);
  const start = () => route();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(start); else start();
})();
