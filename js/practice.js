// Quiz engine: answer tracking with spaced repetition (Leitner boxes), the per-section
// mini quizzes, and the Exam practice page (review, quiz, mock exam, drills, cards, progress).
ML.qid = (q) => {
  let h = 0;
  for (const c of q.q + (q.code || "")) h = (Math.imul(h, 31) + c.charCodeAt(0)) | 0;
  return `${q.ch}:${(h >>> 0).toString(36)}`;
};
const BOX_WAIT = [0, 10 * 60e3, 60 * 60e3, 24 * 60 * 60e3, 3 * 24 * 60 * 60e3];
ML.qstats = {
  all() { return ML.store.get("mlsite-qstats", {}); },
  get(q) { return this.all()[ML.qid(q)] || null; },
  record(q, ok) {
    const all = this.all(), id = ML.qid(q);
    const s = all[id] || { r: 0, w: 0, box: 0, last: 0, ok: false };
    if (ok) { s.r++; s.box = Math.min(s.box + 1, 4); } else { s.w++; s.box = 0; }
    s.last = Date.now(); s.ok = ok;
    all[id] = s;
    ML.store.set("mlsite-qstats", all);
  },
  due(q, now = Date.now()) { const s = this.get(q); return !s || now - s.last >= BOX_WAIT[s.box]; },
  reset() { ML.store.set("mlsite-qstats", {}); },
};
const TYPE_LABEL = { scenario: "Scenario", code: "Code reading", extra: "Beyond the script" };
const chTitle = (ch) => (ML.order.find((o) => o.id === ch) || {}).title || ch;

ML.quiz = {
  // mode "instant": feedback right away. mode "exam": only records the choice.
  card(q, idx, total, onAnswer, mode = "instant") {
    const c = ML.el("div", { class: "q-card" });
    const tag = TYPE_LABEL[q.type] ? `${TYPE_LABEL[q.type]} · ` : "";
    c.append(ML.el("div", { class: "q-meta" }, [ML.el("span", { text: `Question ${idx + 1}${total ? ` of ${total}` : ""}` }), ML.el("span", { text: tag + chTitle(q.ch) })]));
    c.append(ML.el("div", { class: "q-text", html: q.q }));
    if (q.code) {
      const pre = ML.el("pre", { class: "py" });
      pre.innerHTML = ML.highlight(q.code);
      c.append(ML.el("div", { class: "code" }, [pre]));
    }
    const opts = ML.el("div", { class: "q-opts" });
    const btns = q.o.map((t, i) => ML.el("button", { type: "button", class: "q-opt" }, [ML.el("span", { class: "key", text: "ABCD"[i] }), ML.el("span", { html: t })]));
    btns.forEach((b, i) => b.addEventListener("click", () => {
      if (mode === "exam") {
        btns.forEach((x) => x.classList.remove("picked"));
        b.classList.add("picked");
        onAnswer && onAnswer(i);
        return;
      }
      btns.forEach((x) => (x.disabled = true));
      btns[q.a].classList.add("right");
      if (i !== q.a) b.classList.add("wrong");
      c.append(ML.el("div", { class: "q-explain", html: `<b class="${i === q.a ? "ok" : "no"}">${i === q.a ? "Correct." : `Not quite. Answer: ${"ABCD"[q.a]}.`}</b> ${q.why}` }));
      if (ML.ai) c.append(ML.ai.explainButton(q, i));
      ML.qstats.record(q, i === q.a);
      onAnswer && onAnswer(i === q.a);
    }));
    btns.forEach((b) => opts.append(b));
    c.append(opts);
    c._btns = btns;
    return c;
  },
  shuffled(q, r) {
    const ord = ML.shuffleIdx(q.o.length, r);
    return { ...q, o: ord.map((k) => q.o[k]), a: ord.indexOf(q.a) };
  },
  mini(el, ch) {
    const qs = ML.questions.filter((q) => q.ch === ch);
    el.classList.add("quiz");
    const base = qs.filter((q) => !q.type), extra = qs.filter((q) => q.type);
    base.concat(extra).forEach((q, i) => el.append(ML.quiz.card(q, i, qs.length)));
  },
};

ML.register("exam", {
  html: (m) => String.raw`
  ${ML.chHead(m, "Practice like the exam. <b>Review</b> brings back the questions you got wrong; <b>fill-in-the-blank code</b> drills the exact question type Oliver named in the exam announcement; the <b>mock exam</b> is timed and scored per section; <b>drills</b> generate fresh numbers every time.", ["all sections"])}
  <div class="tabs" id="ex-modes" role="tablist" aria-label="Practice mode"></div>
  <div id="ex-body" style="display:grid;gap:16px"></div>
  `,
  init(root) {
    const modesEl = root.querySelector("#ex-modes"), body = root.querySelector("#ex-body");
    const MODES = [["review", "Review (spaced)"], ["quiz", "Practice quiz"], ["fillcode", "Fill-in-the-blank code"], ["mock", "Mock exam"], ["drill", "Calculations"], ["cards", "Flashcards"], ["progress", "My progress"]];
    let mode = ML.store.get("mlsite-exmode", "review"), seed = Date.now() % 100000, timer = null, preset = null;
    const tabs = MODES.map(([v, t]) => {
      const b = ML.el("button", { type: "button", class: "tab", role: "tab", text: t, onclick: () => go(v) });
      modesEl.append(b);
      return [v, b];
    });
    function go(v, opts) {
      mode = v; preset = opts || null;
      ML.store.set("mlsite-exmode", v);
      tabs.forEach(([k, b]) => { b.classList.toggle("on", k === v); b.setAttribute("aria-selected", k === v); });
      clearInterval(timer);
      body.innerHTML = "";
      ({ review, quiz, fillcode, mock, drill, cards, progress })[v]();
      window.scrollTo(0, 0);
    }
    const scriptQs = () => ML.questions.filter((q) => q.ch !== "extras");

    // ---------- Review (spaced repetition) ----------
    function review() {
      const now = Date.now(), all = ML.qstats.all();
      const qs = ML.questions;
      const missed = qs.filter((q) => { const s = all[ML.qid(q)]; return s && s.box === 0 && s.w > 0; });
      const due = qs.filter((q) => { const s = all[ML.qid(q)]; return s && s.box > 0 && now - s.last >= BOX_WAIT[s.box]; }).sort((a, b) => all[ML.qid(a)].box - all[ML.qid(b)].box);
      const fresh = qs.filter((q) => !all[ML.qid(q)] && q.ch !== "extras");
      const mastered = qs.filter((q) => { const s = all[ML.qid(q)]; return s && s.box >= 3; }).length;
      body.append(ML.el("div", { class: "task-grid stat-grid" }, [
        statCard("To repeat (last answer wrong)", missed.length, "var(--bad)"),
        statCard("Due again", due.length, "var(--warn)"),
        statCard("Not seen yet", fresh.length, "var(--muted)"),
        statCard("Mastered (3+ correct in a row)", `${mastered} / ${qs.length}`, "var(--good)"),
      ]));
      body.append(ML.el("p", { class: "fig-cap", html: "A correct answer moves a question one box up; it comes back after 10 minutes, 1 hour, 1 day, then 3 days. A wrong answer sends it back to box 0, so it returns right away. Progress is stored in this browser." }));
      const r = ML.rng(seed++);
      const shuffle = (a) => ML.shuffleIdx(a.length, r).map((i) => a[i]);
      const batch = missed.concat(due, shuffle(fresh)).slice(0, 10);
      if (!batch.length) {
        body.append(ML.el("div", { class: "verdict", html: "<b>Nothing is due right now.</b> Every question you have answered is scheduled for later. Try the mock exam or the calculation drills." }));
        return;
      }
      const list = ML.el("div", { class: "quiz" });
      body.append(list);
      let done = 0;
      batch.forEach((q, i) => list.append(ML.quiz.card(ML.quiz.shuffled(q, r), i, batch.length, () => { if (++done === batch.length) list.append(nextBtn); })));
      const nextBtn = ML.el("button", { type: "button", class: "btn", text: "Next review batch", onclick: () => go("review") });
    }
    function statCard(label, value, color) {
      return ML.el("div", { class: "task-card" }, [ML.el("div", { class: "q", text: label }), ML.el("div", { style: `font:700 30px var(--f-mono);color:${color}`, text: String(value) })]);
    }

    // ---------- Practice quiz with filters ----------
    function quiz() {
      const r = ML.rng(seed++);
      const chs = [...new Set(ML.questions.map((q) => q.ch))];
      const sel = ML.el("select", { id: "ex-filter", "aria-label": "Section" });
      sel.append(ML.el("option", { value: "all", text: "All sections" }));
      chs.forEach((c) => sel.append(ML.el("option", { value: c, text: chTitle(c) })));
      const typ = ML.el("select", { id: "ex-type", "aria-label": "Question type" });
      [["all", "All question types"], ["concept", "Concept questions"], ["scenario", "Scenarios: which method or metric?"], ["code", "Code reading"], ["extra", "Beyond the script"]].forEach(([v, t]) => typ.append(ML.el("option", { value: v, text: t })));
      if (preset && preset.ch) sel.value = preset.ch;
      if (preset && preset.type) typ.value = preset.type;
      const score = ML.el("div", { class: "verdict num" });
      body.append(ML.el("div", { class: "btn-row" }, [ML.el("div", { style: "flex:1;min-width:200px" }, [sel]), ML.el("div", { style: "flex:1;min-width:200px" }, [typ])]), score);
      const list = ML.el("div", { class: "quiz" });
      body.append(list);
      const start = () => {
        list.innerHTML = "";
        let pool = ML.questions.filter((q) => (sel.value === "all" ? true : q.ch === sel.value))
          .filter((q) => (typ.value === "all" ? true : typ.value === "concept" ? !q.type : q.type === typ.value));
        if (sel.value === "all" && typ.value !== "extra" && typ.value !== "all") pool = pool.filter((q) => q.ch !== "extras");
        pool = ML.shuffleIdx(pool.length, r).map((i) => pool[i]).slice(0, 20);
        let answered = 0, right = 0;
        const upd = () => {
          score.innerHTML = pool.length ? `Score: <b>${right} / ${answered}</b> answered of ${pool.length}.${answered === pool.length ? ` <b>Finished: ${Math.round((right / pool.length) * 100)}%.</b>` : ""}` : "No questions match this filter.";
        };
        pool.forEach((q, i) => list.append(ML.quiz.card(ML.quiz.shuffled(q, r), i, pool.length, (ok) => { answered++; if (ok) right++; upd(); })));
        upd();
        if (pool.length) list.append(ML.el("button", { type: "button", class: "btn", text: "New set of questions", onclick: start }));
      };
      sel.addEventListener("change", start); typ.addEventListener("change", start);
      start();
    }

    // ---------- Fill-in-the-blank code ----------
    function fillcode() {
      body.append(ML.el("p", { class: "fig-cap", html: "Every blank (<code>____</code>) has exactly one intended answer: a scikit-learn class, method or argument, taken from the script's own code. Type into the blanks, then <b>Check</b> or <b>Reveal</b>." }));
      const scoreEl = ML.el("div", { class: "verdict num" });
      body.append(scoreEl);
      const list = ML.el("div", { class: "quiz" });
      body.append(list);
      const results = new Map();
      const updTotal = () => {
        if (!results.size) { scoreEl.innerHTML = "Answer any exercise below to see your running score."; return; }
        let r = 0, n = 0;
        results.forEach((v) => { r += v.right; n += v.total; });
        scoreEl.innerHTML = `Running score: <b>${r} / ${n}</b> blanks correct across ${results.size} / ${ML.fillcode.length} exercises checked.`;
      };
      ML.fillcode.forEach((ex, idx) => {
        const keys = Object.keys(ex.ans);
        const inputs = {};
        const codeHtml = ML.highlight(ex.code).replace(/__(b\d+)__/g, (_, k) => {
          const w = Math.max(ex.ans[k].length + 2.5, 8);
          inputs[k] = null;
          return `<input class="blank" data-k="${k}" style="width:${w}ch" autocomplete="off" spellcheck="false" aria-label="blank ${k}">`;
        });
        const pre = ML.el("pre", { class: "py", html: codeHtml });
        const res = ML.el("div");
        const card = ML.el("div", { class: "q-card" }, [
          ML.el("div", { class: "q-meta" }, [ML.el("span", { text: `${idx + 1}. ${ex.title}` }), ML.el("span", { text: `${ex.pts} pts · ${chTitle(ex.ch)}` })]),
          ML.el("div", { class: "code" }, [pre]),
          ML.el("div", { class: "btn-row" }, [
            ML.el("button", { type: "button", class: "btn", text: "Check" }),
            ML.el("button", { type: "button", class: "btn ghost", text: "Reveal" }),
            ML.el("button", { type: "button", class: "btn ghost", text: "Clear" }),
          ]),
          res,
        ]);
        pre.querySelectorAll(".blank").forEach((inp) => { inputs[inp.dataset.k] = inp; });
        const [checkBtn, revealBtn, clearBtn] = card.querySelectorAll(".btn-row button");
        const check = () => {
          let right = 0;
          keys.forEach((k) => {
            const inp = inputs[k];
            const ok = inp.value.trim().toLowerCase() === ex.ans[k].toLowerCase();
            inp.classList.remove("ok", "no");
            inp.classList.add(ok ? "ok" : "no");
            if (ok) right++;
          });
          results.set(idx, { right, total: keys.length });
          res.innerHTML = `<div class="q-explain"><b class="${right === keys.length ? "ok" : "no"}">${right} / ${keys.length} correct.</b></div>`;
          updTotal();
        };
        const reveal = () => {
          keys.forEach((k) => { inputs[k].value = ex.ans[k]; inputs[k].classList.remove("no"); inputs[k].classList.add("ok"); });
          results.set(idx, { right: keys.length, total: keys.length });
          res.innerHTML = `<div class="q-explain">Answers revealed.</div>`;
          updTotal();
        };
        const clear = () => {
          keys.forEach((k) => { inputs[k].value = ""; inputs[k].classList.remove("ok", "no"); });
          results.delete(idx);
          res.innerHTML = "";
          updTotal();
        };
        checkBtn.addEventListener("click", check);
        revealBtn.addEventListener("click", reveal);
        clearBtn.addEventListener("click", clear);
        pre.querySelectorAll(".blank").forEach((inp) => inp.addEventListener("keydown", (e) => { if (e.key === "Enter") check(); }));
        list.append(card);
      });
    }

    // ---------- Mock exam ----------
    function mock() {
      const hist = ML.store.get("mlsite-mocks", []);
      const intro = ML.el("div", { class: "q-card" });
      intro.innerHTML = `<div class="q-meta"><span>Mock exam</span><span>30 minutes</span></div>
        <div class="q-text">20 multiple-choice questions from all script sections (concepts, scenarios and code reading) plus 5 calculations. No feedback until you submit; then you get your score per section and a full review.</div>
        ${hist.length ? `<div class="table-wrap"><table><thead><tr><th>Date</th><th>Score</th><th>Result</th></tr></thead><tbody>${hist.slice(-5).reverse().map((h) => `<tr><td>${new Date(h.date).toLocaleString()}</td><td class="mono">${h.score} / ${h.total}</td><td class="mono">${Math.round((h.score / h.total) * 100)}%</td></tr>`).join("")}</tbody></table></div>` : ""}`;
      const startBtn = ML.el("button", { type: "button", class: "btn", text: "Start the mock exam", style: "justify-self:start" });
      intro.append(startBtn);
      body.append(intro);
      startBtn.addEventListener("click", runMock);
    }
    function runMock() {
      body.innerHTML = "";
      const r = ML.rng(seed++);
      // Round-robin over sections so every topic appears.
      const bySec = {};
      scriptQs().forEach((q) => (bySec[q.ch] = bySec[q.ch] || []).push(q));
      Object.keys(bySec).forEach((k) => (bySec[k] = ML.shuffleIdx(bySec[k].length, r).map((i) => bySec[k][i])));
      const secs = ML.shuffleIdx(Object.keys(bySec).length, r).map((i) => Object.keys(bySec)[i]);
      const picked = [];
      for (let round = 0; picked.length < 20; round++) for (const s of secs) { if (picked.length < 20 && bySec[s][round]) picked.push(bySec[s][round]); }
      const mcq = picked.map((q) => ({ q: ML.quiz.shuffled(q, r), orig: q, choice: null }));
      const drills = Array.from({ length: 5 }, () => ({ d: ML.drills[Math.floor(r() * ML.drills.length)](r), val: "" }));
      let left = 30 * 60;
      const bar = ML.el("div", { class: "timer-bar" });
      const tEl = ML.el("b", { class: "mono" }), cEl = ML.el("span");
      const submit = ML.el("button", { type: "button", class: "btn", text: "Submit", onclick: () => finish() });
      bar.append(ML.el("span", { text: "Time left" }), tEl, cEl, submit);
      body.append(bar);
      const answeredCount = () => mcq.filter((x) => x.choice !== null).length + drills.filter((x) => x.val.trim() !== "").length;
      const tick = () => {
        tEl.textContent = `${String(Math.floor(left / 60)).padStart(2, "0")}:${String(left % 60).padStart(2, "0")}`;
        tEl.style.color = left < 300 ? "var(--bad)" : "";
        cEl.textContent = `${answeredCount()} / 25 answered`;
      };
      tick();
      timer = setInterval(() => { left--; tick(); if (left <= 0) finish(); }, 1000);
      const list = ML.el("div", { class: "quiz" });
      body.append(list);
      mcq.forEach((x, i) => list.append(ML.quiz.card(x.q, i, 25, (choice) => { x.choice = choice; tick(); }, "exam")));
      drills.forEach((x, i) => {
        const id = ML.ui.id("mk");
        const inp = ML.el("input", { type: "text", id, inputmode: "decimal", autocomplete: "off", style: "max-width:200px" });
        inp.addEventListener("input", () => { x.val = inp.value; tick(); });
        list.append(ML.el("div", { class: "q-card" }, [
          ML.el("div", { class: "q-meta" }, [ML.el("span", { text: `Question ${21 + i} of 25` }), ML.el("span", { text: "Calculation" })]),
          ML.el("label", { class: "q-text", for: id, html: x.d.q }), inp,
        ]));
      });
      list.append(ML.el("button", { type: "button", class: "btn", text: "Submit the exam", onclick: () => finish() }));
      let finished = false;
      function finish() {
        if (finished) return;
        finished = true;
        clearInterval(timer);
        const sec = {};
        const add = (name, ok) => { sec[name] = sec[name] || { r: 0, n: 0 }; sec[name].n++; if (ok) sec[name].r++; };
        let score = 0;
        mcq.forEach((x) => { const ok = x.choice === x.q.a; if (ok) score++; add(chTitle(x.q.ch), ok); if (x.choice !== null) ML.qstats.record(x.orig, ok); });
        drills.forEach((x) => { const v = parseFloat(x.val.replace(",", ".")); x.ok = Number.isFinite(v) && Math.abs(v - x.d.a) < 0.006 + Math.abs(x.d.a) * 0.002; if (x.ok) score++; add("Calculations", x.ok); });
        const hist = ML.store.get("mlsite-mocks", []);
        hist.push({ date: Date.now(), score, total: 25 });
        ML.store.set("mlsite-mocks", hist.slice(-20));
        body.innerHTML = "";
        const pct = Math.round((score / 25) * 100);
        body.append(ML.el("div", { class: "q-card" }, [
          ML.el("div", { class: "q-meta" }, [ML.el("span", { text: "Result" }), ML.el("span", { text: `${30 * 60 - Math.max(left, 0) >= 60 ? Math.round((30 * 60 - Math.max(left, 0)) / 60) + " min" : "under a minute"}` })]),
          ML.el("div", { style: `font:700 40px var(--f-mono);color:${pct >= 70 ? "var(--good)" : pct >= 50 ? "var(--warn)" : "var(--bad)"}`, text: `${score} / 25 (${pct}%)` }),
        ]));
        const rows = Object.entries(sec).sort((a, b) => a[1].r / a[1].n - b[1].r / b[1].n);
        body.append(ML.el("div", { class: "table-wrap", html: `<table><thead><tr><th>Section</th><th>Correct</th><th style="width:40%">Share</th></tr></thead><tbody>${rows.map(([k, v]) => `<tr><td>${k}</td><td class="mono">${v.r} / ${v.n}</td><td><div class="bar-track"><div class="bar-fill" style="width:${(v.r / v.n) * 100}%;background:${v.r / v.n >= 0.7 ? "var(--good)" : v.r / v.n >= 0.5 ? "var(--warn)" : "var(--bad)"}"></div></div></td></tr>`).join("")}</tbody></table>` }));
        body.append(ML.el("div", { class: "section-title" }, [ML.el("h2", { text: "Review your answers" })]));
        const rl = ML.el("div", { class: "quiz" });
        mcq.forEach((x, i) => {
          const card = ML.quiz.card(x.q, i, 25, null, "exam");
          card._btns.forEach((b) => (b.disabled = true));
          card._btns[x.q.a].classList.add("right");
          if (x.choice !== null && x.choice !== x.q.a) card._btns[x.choice].classList.add("wrong");
          card.append(ML.el("div", { class: "q-explain", html: `<b class="${x.choice === x.q.a ? "ok" : "no"}">${x.choice === x.q.a ? "Correct." : x.choice === null ? "Not answered." : "Wrong."}</b> ${x.q.why}` }));
          if (ML.ai) card.append(ML.ai.explainButton(x.q, x.choice));
          rl.append(card);
        });
        drills.forEach((x, i) => rl.append(ML.el("div", { class: "q-card" }, [
          ML.el("div", { class: "q-meta" }, [ML.el("span", { text: `Question ${21 + i}` }), ML.el("span", { text: "Calculation" })]),
          ML.el("div", { class: "q-text", html: x.d.q }),
          ML.el("div", { class: "q-explain", html: `<b class="${x.ok ? "ok" : "no"}">${x.ok ? "Correct." : `Your answer: ${ML.esc(x.val || "none")}.`}</b> ${x.d.s} = <b>${+x.d.a.toFixed(4)}</b>` }),
        ])));
        body.append(rl, ML.el("div", { class: "btn-row" }, [ML.el("button", { type: "button", class: "btn", text: "Another mock exam", onclick: () => go("mock") }), ML.el("button", { type: "button", class: "btn ghost", text: "Review my mistakes", onclick: () => go("review") })]));
        window.scrollTo(0, 0);
      }
    }

    // ---------- Calculation drills ----------
    function drill() {
      const r = ML.rng(seed++);
      const wrap = ML.el("div", { class: "quiz" });
      body.append(ML.el("p", { class: "fig-cap", text: "Type your answer (3 decimals are enough) and press Enter. Each check reveals the worked solution." }), wrap);
      const make = () => {
        const d = ML.drills[Math.floor(r() * ML.drills.length)](r);
        const id = ML.ui.id("dr");
        const inp = ML.el("input", { type: "text", id, inputmode: "decimal", autocomplete: "off", style: "max-width:180px" });
        const res = ML.el("div");
        const card = ML.el("div", { class: "q-card" }, [
          ML.el("div", { class: "q-meta" }, [ML.el("span", { text: "Calculation" })]),
          ML.el("label", { class: "q-text", for: id, html: d.q }),
          ML.el("div", { class: "btn-row", style: "align-items:center" }, [inp, ML.el("button", { type: "button", class: "btn", text: "Check", onclick: () => check() }), ML.el("button", { type: "button", class: "btn ghost", text: "New one", onclick: () => { wrap.prepend(make()); } })]),
          res,
        ]);
        const check = () => {
          const v = parseFloat(inp.value.replace(",", "."));
          const ok = Number.isFinite(v) && Math.abs(v - d.a) < 0.006 + Math.abs(d.a) * 0.002;
          res.innerHTML = `<div class="q-explain"><b class="${ok ? "ok" : "no"}">${ok ? "Correct." : "Not quite."}</b> ${d.s} = <b>${+d.a.toFixed(4)}</b></div>`;
        };
        inp.addEventListener("keydown", (e) => { if (e.key === "Enter") check(); });
        return card;
      };
      for (let i = 0; i < 6; i++) wrap.append(make());
    }

    // ---------- Flashcards ----------
    function cards() {
      const deck = [
        ["KNN", "Ask the k nearest training examples to vote. Main setting: n_neighbors. Needs scaling. Slow predictions with many examples."],
        ["SVM", "Separate classes with a wide margin, width 2/‖w‖. C: violation penalty. RBF kernel with γ for curved boundaries. Support vectors define the boundary."],
        ["Random forest", "Many trees on bootstrap samples with random feature subsets; average q_t(c | x). n_estimators, max_depth. No scaling needed."],
        ["Standardization", "x′ = (x − μ)/σ with training μ and σ. Mean 0, std 1. Fit inside the pipeline to avoid leakage."],
        ["Precision vs. recall", "Precision = TP/(TP + FP): how many positive predictions are right. Recall = TP/(TP + FN): how many actual positives were found."],
        ["Linear regression", "ŷ = b + Σ w_j x_j, minimizes MSE. Metrics: MAE, RMSE (same units), R² = 1 − SS_res/SS_tot."],
        ["Lasso", "(1/2n) Σ (y − ŷ)² + α Σ |w_j|. Shrinks weights, some exactly to zero. Intercept not penalized. Scale first, choose α by CV."],
        ["Polynomial regression", "Add x², x³ … then use linear regression. degree controls flexibility: too low underfits, too high can overfit."],
        ["MLP", "Layers of neurons h = max(0, b + Σ w_j x_j). hidden_layer_sizes=(20,) is one layer of 20. Scale inputs; early stopping on validation."],
        ["DBSCAN", "Core point: |N_ε(x)| ≥ min_samples (incl. itself). Clusters of connected core points, border points, noise = −1. No number of clusters needed, no predict."],
        ["PCA", "Center, find directions of largest variance (PC2 ⟂ PC1), project: z = Σ v_j (x_j − μ_j). Unsupervised. Standardize first. Has transform."],
        ["MDS", "Map examples so distances match: Stress = Σ over pairs i &lt; j of (d_ij − d̂_ij)². Often slower than PCA. Only fit_transform, no transform."],
        ["Cross-validation", "K folds, each validates once; s̄ = mean of fold scores. GridSearchCV refits the best setting on the full training set."],
        ["Train / validation / test", "Training learns; validation chooses; testing checks. Do not tune on the test set."],
      ];
      let i = 0, flipped = false;
      const r = ML.rng(seed++), order = ML.shuffleIdx(deck.length, r);
      const card = ML.el("button", { type: "button", class: "flashcard", "aria-live": "polite" });
      const nav = ML.el("div", { class: "btn-row" });
      const show = () => {
        const [t, d] = deck[order[i]];
        card.innerHTML = flipped ? `<div class="side"><small>${t}</small><div style="font-size:16px;line-height:1.55;max-width:60ch">${d}</div></div>` : `<div class="side"><small>Card ${i + 1} of ${deck.length} · click or press Space to flip</small><div style="font-family:var(--f-display);font-size:30px;font-weight:700">${t}</div><div style="font-size:14px;color:var(--muted)">Explain the idea, the formula and the main setting.</div></div>`;
      };
      card.addEventListener("click", () => { flipped = !flipped; show(); });
      ML.ui.button(nav, "Previous", () => { i = (i - 1 + deck.length) % deck.length; flipped = false; show(); }, true);
      ML.ui.button(nav, "Next card", () => { i = (i + 1) % deck.length; flipped = false; show(); });
      body.append(ML.el("div", { class: "flash-wrap" }, [card, nav]));
      show();
    }

    // ---------- Progress dashboard ----------
    function progress() {
      const all = ML.qstats.all();
      const secs = [...new Set(ML.questions.map((q) => q.ch))];
      const rows = secs.map((ch) => {
        const qs = ML.questions.filter((q) => q.ch === ch);
        let seen = 0, r = 0, w = 0, mastered = 0;
        qs.forEach((q) => { const s = all[ML.qid(q)]; if (s) { seen++; r += s.r; w += s.w; if (s.box >= 3) mastered++; } });
        return { ch, total: qs.length, seen, r, w, acc: r + w ? r / (r + w) : null, mastered };
      });
      const tot = rows.reduce((a, x) => ({ seen: a.seen + x.seen, total: a.total + x.total, r: a.r + x.r, w: a.w + x.w }), { seen: 0, total: 0, r: 0, w: 0 });
      body.append(ML.el("div", { class: "task-grid stat-grid" }, [
        statCard("Questions seen", `${tot.seen} / ${tot.total}`, "var(--accent)"),
        statCard("All-time accuracy", tot.r + tot.w ? `${Math.round((tot.r / (tot.r + tot.w)) * 100)}%` : "–", "var(--ink)"),
        statCard("Mock exams taken", ML.store.get("mlsite-mocks", []).length, "var(--ink)"),
      ]));
      const weak = rows.filter((x) => x.r + x.w >= 3).sort((a, b) => a.acc - b.acc).slice(0, 3);
      body.append(ML.el("div", { class: "verdict", html: weak.length ? `<b>Weakest sections:</b> ${weak.map((x) => `<a href="#${x.ch}">${chTitle(x.ch)}</a> (${Math.round(x.acc * 100)}%)`).join(", ")}. Reread them, then practice with the buttons below.` : "Answer at least three questions in a section to see your weakest topics here." }));
      const tbl = ML.el("div", { class: "table-wrap" });
      tbl.innerHTML = `<table><thead><tr><th>Section</th><th>Seen</th><th>Mastered</th><th style="width:30%">Accuracy</th><th></th></tr></thead><tbody></tbody></table>`;
      const tb = tbl.querySelector("tbody");
      rows.forEach((x) => {
        const tr = ML.el("tr");
        const color = x.acc === null ? "var(--line)" : x.acc >= 0.75 ? "var(--good)" : x.acc >= 0.5 ? "var(--warn)" : "var(--bad)";
        tr.innerHTML = `<td><a href="#${x.ch}">${chTitle(x.ch)}</a></td><td class="mono">${x.seen} / ${x.total}</td><td class="mono">${x.mastered}</td><td><div style="display:flex;gap:8px;align-items:center"><div class="bar-track" style="flex:1"><div class="bar-fill" style="width:${(x.acc || 0) * 100}%;background:${color}"></div></div><span class="mono" style="width:40px;text-align:right">${x.acc === null ? "–" : Math.round(x.acc * 100) + "%"}</span></div></td>`;
        const td = ML.el("td");
        td.append(ML.el("button", { type: "button", class: "btn ghost", style: "padding:4px 10px;font-size:12.5px", text: "Practice", onclick: () => go("quiz", { ch: x.ch }) }));
        tr.append(td);
        tb.append(tr);
      });
      body.append(tbl);
      const reset = ML.el("button", { type: "button", class: "btn ghost", text: "Reset my answer history", style: "justify-self:start" });
      let armed = false;
      reset.addEventListener("click", () => {
        if (!armed) { armed = true; reset.textContent = "Click again to delete all answer history"; reset.style.color = "var(--bad)"; reset.style.borderColor = "var(--bad)"; return; }
        ML.qstats.reset(); ML.store.set("mlsite-mocks", []); go("progress");
      });
      body.append(reset);
    }
    go(MODES.some(([k]) => k === mode) ? mode : "review");
    return () => clearInterval(timer);
  },
});
