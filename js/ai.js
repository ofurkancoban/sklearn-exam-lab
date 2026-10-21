/* AI tutor. Default: a free open-weight model (GPT-OSS 20B, Apache 2.0) served without a key by Pollinations.
 * Alternatives: Anthropic, OpenAI, Google Gemini or any OpenAI-compatible endpoint (OpenRouter, Ollama, LM Studio).
 * Keys stay in this browser and go only to the chosen provider. */
(() => {
  const FREE_URL = "https://text.pollinations.ai/openai";
  const PROVIDERS = {
    free: { name: "Free open model: GPT-OSS 20B (no key, no cost)", keyLabel: "No key needed", needsKey: false, models: ["openai-fast"], help: "GPT-OSS 20B is an open-weight model (Apache 2.0 licence), served free of charge by Pollinations (pollinations.ai). No account and no key needed. The free tier answers about once every 30 seconds per internet connection; your question and the current section's text are sent to Pollinations." },
    anthropic: { name: "Anthropic (Claude)", keyLabel: "Anthropic API key", needsKey: true, models: ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"], help: "Create a key at console.anthropic.com." },
    openai: { name: "OpenAI", keyLabel: "OpenAI API key", needsKey: true, models: [], help: "Create a key at platform.openai.com and enter a model ID from your account." },
    gemini: { name: "Google Gemini", keyLabel: "Gemini API key", needsKey: true, models: [], help: "Create a key in Google AI Studio and enter a Gemini model ID." },
    compatible: { name: "OpenAI-compatible (OpenRouter, Ollama, LM Studio, ...)", keyLabel: "API key (leave empty for local servers)", needsKey: false, models: [], help: "Enter the server's base URL ending in /v1, e.g. http://localhost:11434/v1 for Ollama or http://localhost:1234/v1 for LM Studio. Local servers must allow requests from this page (for Ollama set OLLAMA_ORIGINS=*)." },
  };
  const FALLBACK_MODELS = ["claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"];
  const DEFAULT = { provider: "free", model: "openai-fast", key: "", base: "" };
  const STORE_KEY = "mlsite-ai";

  const load = () => { try { const c = JSON.parse(localStorage.getItem(STORE_KEY)); return c && PROVIDERS[c.provider] ? c : DEFAULT; } catch (e) { return DEFAULT; } };
  const save = (c) => { try { localStorage.setItem(STORE_KEY, JSON.stringify(c)); } catch (e) {} };
  const cfg = load;
  const configured = () => { const c = cfg(); return !!(c.provider && c.model && (c.key || !PROVIDERS[c.provider].needsKey) && (c.provider !== "compatible" || c.base)); };
  const wait = (ms, signal) => new Promise((ok, no) => {
    const t = setTimeout(ok, ms);
    if (signal) signal.addEventListener("abort", () => { clearTimeout(t); no(Object.assign(new Error("Stopped."), { name: "AbortError" })); });
  });
  const err = (message, code) => Object.assign(new Error(message), { code: code || "error" });

  // Fetch with an idle timeout: aborts if the server sends nothing for IDLE ms (reset on every chunk).
  const IDLE = 60000;
  function guard(signal) {
    const ctl = new AbortController(); let timer = null, timedOut = false;
    const kick = () => { clearTimeout(timer); timer = setTimeout(() => { timedOut = true; ctl.abort(); }, IDLE); };
    if (signal) { if (signal.aborted) ctl.abort(); else signal.addEventListener("abort", () => ctl.abort()); }
    kick();
    return { signal: ctl.signal, kick, done: () => clearTimeout(timer), timedOut: () => timedOut };
  }
  // Read a server-sent-events body; calls onJSON for every "data: {...}" line.
  async function readSSE(res, g, onJSON) {
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      g.kick();
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
        if (!line.startsWith("data:")) continue;
        const d = line.slice(5).trim();
        if (d === "[DONE]") return;
        try { onJSON(JSON.parse(d)); } catch (e) { if (e && e.code) throw e; }
      }
    }
  }
  const errJSON = async (res) => { try { return await res.json(); } catch (e) { return {}; } };

  /* One answer, streamed. system: instructions; turns: [{role, content}] ending on a user turn.
   * onText(fullTextSoFar) is called while the answer arrives; the promise resolves with the whole answer. */
  async function chat(system, turns, signal, onText) {
    const c = cfg(); if (!configured()) throw err("No AI connected.", "not_configured");
    const emit = (t) => { if (onText) onText(t); };
    let g = null;
    try {
      if (c.provider === "free" || c.provider === "openai" || c.provider === "compatible") {
        const url = c.provider === "free" ? FREE_URL : (c.provider === "openai" ? "https://api.openai.com/v1" : c.base.replace(/\/+$/, "")) + "/chat/completions";
        const headers = { "content-type": "application/json" };
        if (c.key && c.provider !== "free") headers.authorization = "Bearer " + c.key;
        const body = JSON.stringify({ model: c.model || "openai-fast", stream: true, messages: [{ role: "system", content: system }, ...turns] });
        for (let attempt = 0; attempt < 2; attempt++) {
          g = guard(signal);
          const res = await fetch(url, { method: "POST", headers, body, signal: g.signal });
          if (!res.ok) {
            const data = await errJSON(res); g.done();
            const limited = res.status === 402 || res.status === 429;
            if (limited && c.provider === "free" && attempt === 0) {
              // Free tier: about one answer per 30 s. Wait once and retry instead of failing.
              const m = /(\d+)\s*s/.exec((data.error && data.error.message) || "");
              await wait(Math.min(35, m ? +m[1] + 1 : 30) * 1000, signal);
              continue;
            }
            const msg = data.error && (data.error.message || data.error);
            // Pollinations asks for a bot check on pages served from "localhost"; 127.0.0.1 and the live site work.
            if (c.provider === "free" && /turnstile/i.test(String(msg))) throw err("The free model does not answer pages opened on localhost. Open the site via http://127.0.0.1 instead, or use the live site.", "http");
            throw err(limited && c.provider === "free" ? "The free model allows about one answer every 30 seconds. Please wait a moment and ask again." : (msg || "HTTP " + res.status), res.status === 401 ? "auth" : limited ? "rate" : "http");
          }
          let text = "";
          if (/event-stream/.test(res.headers.get("content-type") || "")) {
            await readSSE(res, g, (d) => { const ch = d.choices && d.choices[0]; const piece = ch && ch.delta && ch.delta.content; if (piece) { text += piece; emit(text); } });
          } else {
            const data = await errJSON(res); const m = data.choices && data.choices[0] && data.choices[0].message;
            text = m ? String(m.content || "") : ""; emit(text);
          }
          g.done();
          text = text.trim();
          if (!text) throw err("The model returned no answer. Try again in a moment.", "empty");
          return text;
        }
      }
      if (c.provider === "anthropic") {
        const body = { model: c.model, max_tokens: 16000, system, messages: turns, stream: true };
        const headers = { "content-type": "application/json", "x-api-key": c.key, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" };
        if (FALLBACK_MODELS.includes(c.model)) { body.fallbacks = "default"; headers["anthropic-beta"] = "server-side-fallback-2026-07-01"; }
        g = guard(signal);
        const res = await fetch("https://api.anthropic.com/v1/messages", { method: "POST", headers, body: JSON.stringify(body), signal: g.signal });
        if (!res.ok) { const data = await errJSON(res); g.done(); throw err((data.error && data.error.message) || "HTTP " + res.status, res.status === 401 ? "auth" : res.status === 429 ? "rate" : "http"); }
        let text = "", refused = false;
        await readSSE(res, g, (d) => {
          if (d.type === "content_block_delta" && d.delta && d.delta.type === "text_delta") { text += d.delta.text; emit(text); }
          else if (d.type === "message_delta" && d.delta && d.delta.stop_reason === "refusal") refused = true;
          else if (d.type === "error") throw err((d.error && d.error.message) || "The answer failed.", "http");
        });
        g.done();
        if (refused) throw err("The model declined to answer this question.", "refused");
        return text.trim();
      }
      if (c.provider === "gemini") {
        const url = "https://generativelanguage.googleapis.com/v1beta/models/" + encodeURIComponent(c.model) + ":streamGenerateContent?alt=sse&key=" + encodeURIComponent(c.key);
        const body = { systemInstruction: { parts: [{ text: system }] }, contents: turns.map((t) => ({ role: t.role === "assistant" ? "model" : "user", parts: [{ text: t.content }] })) };
        g = guard(signal);
        const res = await fetch(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body), signal: g.signal });
        if (!res.ok) { const data = await errJSON(res); g.done(); throw err((data.error && data.error.message) || "HTTP " + res.status, res.status === 400 || res.status === 403 ? "auth" : res.status === 429 ? "rate" : "http"); }
        let text = "", reason = "";
        await readSSE(res, g, (d) => {
          const cand = d.candidates && d.candidates[0]; if (!cand) return;
          if (cand.finishReason) reason = cand.finishReason;
          const p = cand.content && cand.content.parts ? cand.content.parts.map((x) => x.text || "").join("") : "";
          if (p) { text += p; emit(text); }
        });
        g.done();
        text = text.trim();
        if (!text) throw err("The model returned no answer" + (reason ? " (" + reason + ")" : "") + ".", "empty");
        return text;
      }
      throw err("Unknown provider.", "not_configured");
    } catch (e) {
      if (g) g.done();
      if (g && g.timedOut()) throw err("The AI did not answer within 60 seconds (the service is busy). Please ask again in a moment.", "timeout");
      if (e.name === "AbortError") throw err("Stopped.", "cancelled");
      if (e.code) throw e;
      throw err("Could not reach the provider. Check your internet connection, the base URL, and (for local servers) that requests from this page are allowed.", "network");
    }
  }

  const errorText = (e) => {
    const code = e && e.code;
    return code === "cancelled" ? "" : code === "not_configured" ? "No AI connected. Open the settings (gear icon) to choose one." : code === "auth" ? "The provider rejected the key. Check it in the settings." : code === "rate" ? (/free model/.test((e && e.message) || "") ? e.message : "Too many requests right now. Try again in a little while.") : code === "timeout" ? e.message : code === "refused" ? "The model declined this question. Try rephrasing it." : e && e.message ? "The answer failed: " + e.message : "The answer was interrupted. Ask again.";
  };

  /* Safe Markdown for AI answers: escapes HTML, keeps TeX (\( \), \[ \], $ $, $$ $$) intact for MathJax,
   * and supports headings, bold, italics, inline code, code blocks, bullet and numbered lists, tables, paragraphs. */
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function renderMD(src) {
    const math = [];
    const keep = (s) => "\u0000" + (math.push(s) - 1) + "\u0000";
    let t = String(src || "").replace(/\r/g, "");
    t = t.replace(/```[\s\S]*?(```|$)/g, (m) => keep('<pre class="md-code">' + esc(m.replace(/^```[^\n]*\n?/, "").replace(/```$/, "")) + "</pre>"));
    t = t.replace(/\\\[[\s\S]*?\\\]/g, (m) => keep(esc(m)))
      .replace(/\$\$([\s\S]+?)\$\$/g, (m, x) => keep("\\[" + esc(x) + "\\]"))
      .replace(/\\\([\s\S]*?\\\)/g, (m) => keep(esc(m)))
      .replace(/(^|[^\\$])\$([^\s$](?:[^$\n]*?[^\s$])?)\$(?!\d)/g, (m, pre, x) => pre + keep("\\(" + esc(x) + "\\)"));
    const inline = (s) => esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
    const out = []; let list = null, para = [], table = [];
    const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const flushTable = () => {
      if (!table.length) return;
      const rows = table.filter((r) => !/^\s*\|?\s*:?-{2,}/.test(r)), hasHead = table.length > 1 && /^\s*\|?\s*:?-{2,}/.test(table[1]);
      const head = hasHead ? rows.shift() : null;
      out.push('<div class="md-table"><table>' + (head ? "<thead><tr>" + cells(head).map((c) => "<th>" + inline(c) + "</th>").join("") + "</tr></thead>" : "") +
        "<tbody>" + rows.map((r) => "<tr>" + cells(r).map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("") + "</tbody></table></div>");
      table = [];
    };
    const flushPara = () => { if (para.length) { out.push("<p>" + para.map(inline).join("<br>") + "</p>"); para = []; } };
    const flushList = () => { if (list) { out.push("<" + list.tag + ">" + list.items.map((i) => "<li>" + i.map(inline).join("<br>") + "</li>").join("") + "</" + list.tag + ">"); list = null; } };
    t.split("\n").forEach((line) => {
      const h = /^\s{0,3}(#{1,6})\s+(.*)$/.exec(line), ul = /^\s*[-*•]\s+(.*)$/.exec(line), ol = /^\s*\d+[.)]\s+(.*)$/.exec(line);
      if (/^\s*\|.*\|\s*$/.test(line)) { flushPara(); flushList(); table.push(line); return; }
      flushTable();
      if (!line.trim()) { flushPara(); flushList(); return; }
      if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) { flushPara(); flushList(); out.push("<hr>"); return; }
      if (h) { flushPara(); flushList(); out.push('<p class="md-h">' + inline(h[2]) + "</p>"); return; }
      if (ul || ol) {
        flushPara(); const tag = ul ? "ul" : "ol";
        if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
        list.items.push([(ul || ol)[1]]); return;
      }
      if (list && /^\s{2,}\S/.test(line)) { list.items[list.items.length - 1].push(line.trim()); return; }
      flushList(); para.push(line);
    });
    flushTable(); flushPara(); flushList();
    return out.join("").replace(/\u0000(\d+)\u0000/g, (m, i) => math[+i]);
  }

  /* ---------------- Chat drawer ---------------- */
  const sectionTitle = () => {
    const id = (location.hash || "").replace("#", "");
    const o = (ML.order || []).find((c) => c.id === id);
    return o ? o.title : document.title.split(" | ")[0];
  };
  // Text of the section on screen, without controls, used as grounding for the answer.
  const sectionText = () => {
    const art = document.querySelector("#main .chapter"); if (!art) return "";
    // Keep the TeX source of typeset formulas so the model sees them.
    try { MathJax.startup.document.getMathItemsWithin(art).forEach((it) => it.typesetRoot && it.typesetRoot.setAttribute("data-tex", it.math)); } catch (e) {}
    const tmp = art.cloneNode(true);
    tmp.querySelectorAll("canvas, button, .stage, .pager, mjx-assistive-mml, script, style").forEach((n) => n.remove());
    tmp.querySelectorAll("mjx-container").forEach((m) => { const tex = m.getAttribute("data-tex"); m.replaceWith(tex ? " " + tex + " " : " "); });
    return tmp.textContent.replace(/\s+/g, " ").trim().slice(0, 12000);
  };
  const rules = () => 'You are a patient tutor for the university course "Machine Learning with Scikit-Learn: A Little Mathematics" (Oliver Kramer, University of Oldenburg). The student is studying the section "' + sectionTitle() + '" of an exam-prep site. Stay consistent with the section text below and its notation, explain step by step in plain English, keep it short (under 250 words) unless asked for more, write math in LaTeX between \\( and \\), and show Python code with scikit-learn in fenced code blocks when useful. If a question is outside the course, say so briefly.\n\nSECTION TEXT:\n' + sectionText();

  const ICON_CHAT = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true"><path fill="currentColor" d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-5 4v-4H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zm3 6.5a1.5 1.5 0 1 0 0 .01zm5 0a1.5 1.5 0 1 0 0 .01zm5 0a1.5 1.5 0 1 0 0 .01z"/></svg>';
  const ICON_GEAR = '<svg viewBox="0 0 24 24" width="17" height="17" aria-hidden="true"><path fill="currentColor" d="M19.4 13a7.5 7.5 0 0 0 0-2l2.1-1.6-2-3.5-2.5 1a7.6 7.6 0 0 0-1.7-1L15 3h-4l-.4 2.9a7.6 7.6 0 0 0-1.7 1l-2.5-1-2 3.5L6.6 11a7.5 7.5 0 0 0 0 2l-2.1 1.6 2 3.5 2.5-1a7.6 7.6 0 0 0 1.7 1L11 21h4l.4-2.9a7.6 7.6 0 0 0 1.7-1l2.5 1 2-3.5zM13 15.5a3.5 3.5 0 1 1 0-7 3.5 3.5 0 0 1 0 7z" transform="translate(-1 0)"/></svg>';

  const fab = ML.el("button", { type: "button", class: "ai-fab", "aria-label": "Ask the AI tutor", title: "Ask the AI tutor (a)", "aria-expanded": "false", "aria-controls": "ai-panel" });
  fab.innerHTML = ICON_CHAT + "<span>Ask AI</span>";
  const panel = ML.el("aside", { class: "ai-panel", id: "ai-panel", "aria-label": "AI tutor", hidden: "" });
  panel.innerHTML =
    '<div class="ai-head"><div><b>AI tutor</b><span class="ai-sub" id="ai-sub"></span></div>' +
    '<button type="button" class="ai-icon" id="ai-gear" aria-label="AI settings" aria-expanded="false">' + ICON_GEAR + "</button>" +
    '<button type="button" class="ai-icon" id="ai-close" aria-label="Close the AI tutor">&times;</button></div>' +
    '<div class="ai-settings" id="ai-settings" hidden></div>' +
    '<div class="ai-log" id="ai-log" aria-live="polite"></div>' +
    '<div class="ai-foot"><div class="ai-compose"><textarea id="ai-in" rows="2" placeholder="Ask about this section, e.g. Why does k = 1 overfit?" aria-label="Your question"></textarea>' +
    '<button type="button" class="btn" id="ai-go">Ask</button><button type="button" class="btn ghost" id="ai-stop" hidden>Stop</button></div>' +
    '<p class="ai-note" id="ai-note"></p><p class="ai-hint">Enter to send, Shift + Enter for a new line. <button type="button" class="ai-link" id="ai-clear-log">Clear chat</button></p></div>';
  document.body.append(fab, panel);

  const $ = (s) => panel.querySelector(s);
  const log = $("#ai-log"), inp = $("#ai-in"), go = $("#ai-go"), stop = $("#ai-stop"), note = $("#ai-note"), settings = $("#ai-settings"), gear = $("#ai-gear");
  let turns = [], turnSection = null, ctl = null, busy = false;

  const sourceLabel = () => {
    const c = cfg();
    return c.provider === "free" ? "Free open model GPT-OSS 20B" : PROVIDERS[c.provider].name.split(" (")[0] + " · " + c.model;
  };
  const syncSub = () => { $("#ai-sub").textContent = sectionTitle() + " · " + sourceLabel(); };
  const emptyHint = () => {
    if (log.childElementCount) return;
    log.innerHTML = '<div class="ai-empty"><p>Ask anything about <b>' + esc(sectionTitle()) + "</b>. The tutor reads the section on screen, so answers use the same notation.</p>" +
      '<div class="ai-chips">' + ["Explain this section in simple words", "What are typical exam traps here?", "Give me one practice question with a solution"].map((s) => '<button type="button" class="ai-chip">' + s + "</button>").join("") + "</div></div>";
    log.querySelectorAll(".ai-chip").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
  };

  function open(focus = true) {
    panel.hidden = false; fab.setAttribute("aria-expanded", "true"); document.body.classList.add("ai-open");
    syncSub(); emptyHint();
    if (focus) inp.focus();
  }
  function close() {
    panel.hidden = true; fab.setAttribute("aria-expanded", "false"); document.body.classList.remove("ai-open");
    fab.focus({ preventScroll: true });
  }
  fab.addEventListener("click", () => (panel.hidden ? open() : close()));
  $("#ai-close").addEventListener("click", close);
  panel.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } });
  $("#ai-clear-log").addEventListener("click", () => { if (ctl) ctl.abort(); turns = []; log.innerHTML = ""; note.textContent = ""; emptyHint(); });

  // A new section starts a fresh conversation, so the grounding text always matches the history.
  window.addEventListener("hashchange", () => {
    if (panel.hidden) return;
    syncSub();
    if (turns.length && turnSection !== sectionTitle()) {
      turns = [];
      log.append(ML.el("div", { class: "ai-divider", text: "Now on: " + sectionTitle() }));
    }
  });

  const scrollEnd = () => { log.scrollTop = log.scrollHeight; };
  async function ask(q, shown) {
    q = String(q || "").trim(); if (!q || busy) return;
    if (!configured()) { note.textContent = errorText({ code: "not_configured" }); return; }
    const empty = log.querySelector(".ai-empty"); if (empty) empty.remove();
    busy = true; go.hidden = true; stop.hidden = false; note.textContent = "";
    if (turnSection !== sectionTitle()) { turns = []; turnSection = sectionTitle(); }
    log.append(ML.el("div", { class: "ai-q", text: shown || q }));
    const ad = ML.el("div", { class: "ai-a streaming", html: '<p class="muted">Thinking...</p>' });
    log.append(ad); scrollEnd();
    let last = "", timer = null, stick = true;
    const onScroll = () => { stick = log.scrollHeight - log.scrollTop - log.clientHeight < 40; };
    log.addEventListener("scroll", onScroll);
    const paint = () => { timer = null; ad.innerHTML = renderMD(last) || '<p class="muted">Thinking...</p>'; if (stick) scrollEnd(); };
    turns.push({ role: "user", content: q });
    ctl = new AbortController();
    try {
      const text = await chat(rules(), turns.slice(-8), ctl.signal, (t) => { last = t; if (!timer) timer = setTimeout(paint, 60); });
      clearTimeout(timer);
      turns.push({ role: "assistant", content: text });
      ad.classList.remove("streaming");
      ad.innerHTML = renderMD(text);
      ML.typeset(ad).then(() => { if (stick) scrollEnd(); });
    } catch (e) {
      clearTimeout(timer);
      turns.pop();
      ad.classList.remove("streaming");
      if (last) { ad.innerHTML = renderMD(last); ML.typeset(ad); } else ad.remove();
      note.textContent = errorText(e);
    } finally {
      log.removeEventListener("scroll", onScroll);
      busy = false; ctl = null; go.hidden = false; stop.hidden = true;
      if (!panel.hidden) inp.focus({ preventScroll: true });
    }
  }
  go.addEventListener("click", () => { const q = inp.value; inp.value = ""; ask(q); });
  stop.addEventListener("click", () => ctl && ctl.abort());
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); go.click(); } });

  /* ---------------- Settings ---------------- */
  function renderSettings() {
    const c = cfg();
    settings.innerHTML =
      "<p>By default the tutor uses a <b>free, open-weight model</b> that needs no account and no key. You can switch to another provider; API keys are stored only in this browser and sent only to the provider you pick.</p>" +
      '<label for="ai-p">Provider</label><select id="ai-p">' + Object.keys(PROVIDERS).map((k) => '<option value="' + k + '"' + (k === c.provider ? " selected" : "") + ">" + PROVIDERS[k].name + "</option>").join("") + "</select>" +
      '<p class="ai-help" id="ai-help"></p>' +
      '<div id="ai-base-row"><label for="ai-base">Base URL</label><input type="text" id="ai-base" value="' + esc(c.base || "") + '" placeholder="http://localhost:11434/v1"></div>' +
      '<label for="ai-key" id="ai-key-l">API key</label><input type="password" id="ai-key" value="' + esc(c.key || "") + '" autocomplete="off">' +
      '<label for="ai-m">Model ID</label><input type="text" id="ai-m" list="ai-models" value="' + esc(c.model || "") + '" placeholder="the model ID from your provider"><datalist id="ai-models"></datalist>' +
      '<div class="btn-row"><button class="btn" id="ai-save" type="button">Save and test</button><button class="btn ghost" id="ai-reset" type="button">Use the free model</button></div>' +
      '<p class="ai-help" id="ai-msg" role="status"></p>';
    const p = settings.querySelector("#ai-p"), msg = settings.querySelector("#ai-msg"), m = settings.querySelector("#ai-m");
    const sync = () => {
      const P = PROVIDERS[p.value], free = p.value === "free";
      settings.querySelector("#ai-help").textContent = P.help;
      settings.querySelector("#ai-key-l").textContent = P.keyLabel;
      settings.querySelector("#ai-base-row").hidden = p.value !== "compatible";
      settings.querySelector("#ai-key").hidden = free; settings.querySelector("#ai-key-l").hidden = free;
      m.readOnly = free; if (free) m.value = "openai-fast";
      settings.querySelector("#ai-models").innerHTML = P.models.map((x) => '<option value="' + x + '">').join("");
    };
    p.addEventListener("change", () => { m.value = PROVIDERS[p.value].models[0] || ""; sync(); });
    sync();
    settings.querySelector("#ai-save").addEventListener("click", async () => {
      const v = { provider: p.value, key: settings.querySelector("#ai-key").value.trim(), model: m.value.trim(), base: settings.querySelector("#ai-base").value.trim() };
      save(v); syncSub();
      if (!configured()) { msg.textContent = "Saved, but incomplete: " + (!v.model ? "enter a model ID." : PROVIDERS[v.provider].needsKey && !v.key ? "enter an API key." : "enter the base URL."); return; }
      msg.textContent = "Saved. Testing...";
      try { const t = await chat("Reply with exactly: OK", [{ role: "user", content: "Connection test." }], null, null); msg.textContent = "Connected. The model replied: " + t.slice(0, 60); }
      catch (e) { msg.textContent = "Not connected: " + e.message; }
    });
    settings.querySelector("#ai-reset").addEventListener("click", () => { save(DEFAULT); renderSettings(); syncSub(); settings.querySelector("#ai-msg").textContent = "Switched to the free model."; });
  }
  gear.addEventListener("click", () => {
    const show = settings.hidden;
    if (show) renderSettings();
    settings.hidden = !show; gear.setAttribute("aria-expanded", String(show));
  });

  // "a" opens the tutor (ignored while typing).
  document.addEventListener("keydown", (e) => {
    const t = e.target;
    if (e.key !== "a" || e.metaKey || e.ctrlKey || e.altKey || (t.closest && t.closest("input, textarea, select, [contenteditable]"))) return;
    if (document.getElementById("kb-help") && !document.getElementById("kb-help").hidden) return;
    e.preventDefault(); panel.hidden ? open() : close();
  });

  // Plain text of an HTML snippet (question texts contain markup and TeX).
  const plain = (html) => { const d = document.createElement("div"); d.innerHTML = html || ""; return d.textContent.replace(/\s+/g, " ").trim(); };
  /* Button for a quiz card: asks the tutor why the right answer is right (and why the picked one is not). */
  function explainButton(q, picked) {
    const b = ML.el("button", { type: "button", class: "btn ghost ai-explain" });
    b.innerHTML = ICON_CHAT + " Explain with AI";
    b.addEventListener("click", () => {
      const opts = q.o.map((o, i) => "ABCD"[i] + ") " + plain(o)).join("\n");
      const wrong = picked != null && picked !== q.a ? "\nI picked " + "ABCD"[picked] + ". Explain why that is wrong." : "";
      const prompt = "Explain this exam question step by step.\n\nQuestion: " + plain(q.q) + (q.code ? "\n\nCode:\n```python\n" + q.code + "\n```" : "") + "\n\nOptions:\n" + opts + "\n\nCorrect answer: " + "ABCD"[q.a] + ". Script explanation: " + plain(q.why) + wrong;
      open(false);
      ask(prompt, "Explain question: " + plain(q.q).slice(0, 140) + (picked != null && picked !== q.a ? " (I picked " + "ABCD"[picked] + ")" : ""));
    });
    return b;
  }

  ML.ai = { PROVIDERS, cfg, configured, chat, renderMD, open, ask, explainButton };
})();
