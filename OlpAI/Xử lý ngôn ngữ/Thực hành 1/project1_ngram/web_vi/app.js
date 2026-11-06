/**
 * Demo tương tác dự đoán từ tiếp theo bằng N-gram.
 *
 * Huấn luyện: số đếm MLE từ toàn bộ tiểu thuyết Emma (corpus-data.js).
 * Hiển thị: ba đoạn mẫu chỉ dùng để trực quan hóa nhấp–kiểm tra.
 */

const ORDER_NAMES = {
  1: "unigram · không dùng ngữ cảnh trước",
  2: "bigram · dùng 1 từ trước",
  3: "trigram · dùng 2 từ trước",
  4: "4-gram · dùng 3 từ trước",
};

const TITLE_VI = {
  "Paragraph 1 — Emma Woodhouse": "Đoạn 1 — Emma Woodhouse",
  "Paragraph 2 — Miss Taylor": "Đoạn 2 — Miss Taylor",
  "Paragraph 3 — Mr. Knightley": "Đoạn 3 — Mr. Knightley",
};

const state = {
  order: 2,
  sep: "\u0001",
  paragraphs: [],
  /** @type {{display: string, key: string, globalIndex: number, paraIndex: number, localIndex: number}[]} */
  tokens: [],
  /** models[order].cont[histKey] = {__c, word: count, ...} */
  models: {},
  displayForm: {},
  totalTokens: 0,
  numSentences: 0,
  /** bảng tiếp nối đang dùng cho N đã chọn */
  cont: null,
  selectedIndex: null,
  sequence: [],
};

const els = {
  paragraphs: document.getElementById("paragraphs"),
  equation: document.getElementById("equation"),
  equationHint: document.getElementById("equation-hint"),
  equationNote: document.getElementById("equation-note"),
  orderCaption: document.getElementById("order-caption"),
  floatLayer: document.getElementById("float-layer"),
  sequence: document.getElementById("sequence"),
  suggestions: document.getElementById("suggestions"),
  historyLine: document.getElementById("history-line"),
  histLen: document.getElementById("hist-len"),
  clearSeq: document.getElementById("clear-seq"),
  corpusStats: document.getElementById("corpus-stats"),
};

function keyOf(parts) {
  return parts.join(state.sep);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function formatProb(p) {
  if (!Number.isFinite(p)) return "0";
  if (p >= 0.01) return p.toFixed(3);
  if (p <= 0) return "0";
  return p.toExponential(2);
}

function formatToken(t) {
  return t === " " ? "␣" : t;
}

function prettyJoin(tokens) {
  if (!tokens.length) return "∅";
  return tokens.map(formatToken).join(" ");
}

function contForOrder(order) {
  return state.models[String(order)]?.cont || {};
}

function bucketFor(historyKeys, order = historyKeys.length + 1) {
  const cont = contForOrder(order);
  const histKey = historyKeys.length ? keyOf(historyKeys) : "";
  return cont[histKey] || null;
}

/** MLE toàn corpus: P(w|h) = C(h,w) / C(h). */
function getProbability(historyKeys, wordKey) {
  const order = historyKeys.length + 1;
  const bucket = bucketFor(historyKeys, order);
  if (!bucket) return { p: 0, num: 0, den: 0, order };
  const den = bucket.__c || 0;
  const num = bucket[wordKey] || 0;
  return { p: den ? num / den : 0, num, den, order };
}

function listContinuations(historyKeys, order = historyKeys.length + 1) {
  const bucket = bucketFor(historyKeys, order);
  if (!bucket) return [];
  const den = bucket.__c || 0;
  if (!den) return [];
  return Object.entries(bucket)
    .filter(([w]) => w !== "__c")
    .map(([w, c]) => [w, c, c / den])
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function clearHighlights() {
  document.querySelectorAll(".token").forEach((el) => {
    el.classList.remove("is-history", "is-target", "has-prob-label");
  });
  els.floatLayer.innerHTML = "";
}

function placeLabel(tokenIndex, text, className = "") {
  const span = document.querySelector(`.token[data-index="${tokenIndex}"]`);
  if (!span) return;
  const corpus = document.querySelector(".corpus");
  const corpusRect = corpus.getBoundingClientRect();
  const rect = span.getBoundingClientRect();
  const label = document.createElement("div");
  label.className = `float-label ${className}`.trim();
  label.textContent = text;
  label.style.left = `${rect.left + rect.width / 2 - corpusRect.left + corpus.scrollLeft}px`;
  label.style.top = `${rect.top - corpusRect.top + corpus.scrollTop}px`;
  els.floatLayer.appendChild(label);
}

function highlightSelection(index) {
  clearHighlights();
  const order = state.order;
  const histLen = order - 1;
  const tok = state.tokens[index];
  const local = tok.localIndex;

  if (local < histLen) {
    document.querySelector(`.token[data-index="${index}"]`)?.classList.add("is-target");
    showEquationUnavailable(tok, histLen);
    return;
  }

  const historyIdx = [];
  for (let i = index - histLen; i < index; i++) historyIdx.push(i);
  const targetIdx = index;

  historyIdx.forEach((i) => {
    document.querySelector(`.token[data-index="${i}"]`)?.classList.add("is-history");
  });
  document.querySelector(`.token[data-index="${targetIdx}"]`)?.classList.add("is-target");

  const historyKeys = historyIdx.map((i) => state.tokens[i].key);
  const wordKey = state.tokens[targetIdx].key;
  const { p, num, den } = getProbability(historyKeys, wordKey);

  placeLabel(targetIdx, `P = ${formatProb(p)}`);
  document.querySelector(`.token[data-index="${targetIdx}"]`)?.classList.add("has-prob-label");

  renderEquation({
    historyDisplay: historyIdx.map((i) => state.tokens[i].display),
    historyKeys,
    wordDisplay: state.tokens[targetIdx].display,
    num,
    den,
    p,
  });
}

function showEquationUnavailable(tok, histLen) {
  els.equationHint.hidden = true;
  els.equation.hidden = false;
  els.equationNote.hidden = false;
  els.equation.innerHTML = `
    <span class="eq-part">Không đủ token đứng trước cho ${state.order}-gram tại
    “${escapeHtml(tok.display)}”.</span>
  `;
  els.equationNote.textContent = `Một ${state.order}-gram cần ${histLen} từ ngữ cảnh bên trái. Hãy thử từ sau hơn, hoặc chọn N nhỏ hơn.`;
}

function renderEquation({ historyDisplay, historyKeys, wordDisplay, num, den, p }) {
  els.equationHint.hidden = true;
  els.equation.hidden = false;
  els.equationNote.hidden = false;

  const histPretty = prettyJoin(historyDisplay);
  const wordPretty = formatToken(wordDisplay);
  const lhs =
    state.order === 1
      ? `P(${wordPretty})`
      : `P(${wordPretty} | ${histPretty})`;
  const numLabel =
    state.order === 1 ? `C(${wordPretty})` : `C(${histPretty}, ${wordPretty})`;
  const denLabel = state.order === 1 ? `C(·)` : `C(${histPretty})`;

  els.equation.innerHTML = `
    <span class="eq-part">${escapeHtml(lhs)}</span>
    <span class="eq-part">=</span>
    <span class="eq-part">${escapeHtml(numLabel)}</span>
    <span class="eq-part">/</span>
    <span class="eq-part">${escapeHtml(denLabel)}</span>
    <span class="eq-part">=</span>
    <span class="eq-part">${num}</span>
    <span class="eq-part">/</span>
    <span class="eq-part">${den}</span>
    <span class="eq-part">=</span>
    <span class="eq-part eq-result">${formatProb(p)}</span>
  `;

  const cont = listContinuations(historyKeys, state.order);
  const topCont = cont
    .slice(0, 6)
    .map(([w, c, pw]) => `${formatToken(displayForm(w))} (${c}, ${formatProb(pw)})`)
    .join(" · ");

  els.equationNote.innerHTML = `
    <strong>MLE toàn corpus</strong> từ toàn bộ <em>Emma</em>
    (${state.totalTokens.toLocaleString("vi-VN")} token, ${state.numSentences.toLocaleString("vi-VN")} câu).
    Ba đoạn bên dưới chỉ là mẫu để kiểm tra.
    ${
      state.order === 1
        ? ` Mẫu số unigram là tổng số token (${den.toLocaleString("vi-VN")}).`
        : den
          ? ` Các tiếp nối hàng đầu của “${escapeHtml(histPretty)}” trong toàn tiểu thuyết: ${escapeHtml(topCont || "không có")}.`
          : ` Lịch sử này không bao giờ xuất hiện trong corpus huấn luyện đầy đủ, nên P = 0.`
    }
  `;
}

function shouldGlue(prev, next) {
  if (!next) return false;
  if (/^[.,;:!?'”’)\]]$/.test(next)) return true;
  if (next === "n't" || next === "'s" || next === "'re" || next === "'ve" || next === "'ll" || next === "'d") {
    return true;
  }
  if (next.startsWith("'") || next.startsWith("’")) return true;
  if (prev === "(" || prev === "[" || prev === "“" || prev === '"') return true;
  return false;
}

function renderParagraphs() {
  els.paragraphs.innerHTML = "";
  let globalIndex = 0;

  state.paragraphs.forEach((para) => {
    const card = document.createElement("article");
    card.className = "paragraph-card";
    const h = document.createElement("h3");
    h.textContent = TITLE_VI[para.title] || para.title;
    const body = document.createElement("div");
    body.className = "paragraph-body";

    para.tokens.forEach((display, localIndex) => {
      const index = globalIndex;
      const span = document.createElement("span");
      span.className = "token";
      span.textContent = display;
      span.dataset.index = String(index);
      span.title = "Nhấp để xem xác suất MLE toàn corpus";
      span.tabIndex = 0;
      span.addEventListener("click", () => selectToken(index));
      span.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectToken(index);
        }
      });
      body.appendChild(span);

      const next = para.tokens[localIndex + 1];
      if (next && !shouldGlue(display, next)) {
        body.appendChild(document.createTextNode(" "));
      }
      globalIndex += 1;
    });

    card.appendChild(h);
    card.appendChild(body);
    els.paragraphs.appendChild(card);
  });
}

function selectToken(index) {
  state.selectedIndex = index;
  highlightSelection(index);
  document.getElementById("equation-panel")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function setOrder(n) {
  state.order = n;
  state.cont = contForOrder(n);
  document.querySelectorAll(".n-btn").forEach((btn) => {
    btn.classList.toggle("is-active", Number(btn.dataset.n) === n);
  });
  els.orderCaption.textContent = `N = ${n} · ${ORDER_NAMES[n]}`;
  els.histLen.textContent = String(Math.max(0, n - 1));
  if (state.selectedIndex != null) highlightSelection(state.selectedIndex);
  else {
    clearHighlights();
    els.equation.hidden = true;
    els.equationNote.hidden = true;
    els.equationHint.hidden = false;
  }
  renderSuggestions();
  renderSequence();
}

function currentHistoryKeys() {
  const need = state.order - 1;
  if (need <= 0) return [];
  const seq = state.sequence.map((t) => t.toLowerCase());
  if (seq.length >= need) return seq.slice(-need);
  return seq.slice();
}

function isWordLike(tokenKey) {
  return /[a-z]/i.test(tokenKey);
}

function displayForm(key) {
  if (state.displayForm[key]) return state.displayForm[key];
  const hit = state.tokens.find((t) => t.key === key);
  return hit ? hit.display : key;
}

function getTopCandidates(limit = 5) {
  const historyKeys = currentHistoryKeys();
  const order = historyKeys.length + 1;

  if (historyKeys.length === 0) {
    return getTopCandidatesUnigram(limit);
  }

  const ranked = listContinuations(historyKeys, order)
    .filter(([w]) => isWordLike(w))
    .map(([w, c, p]) => ({
      word: displayForm(w),
      key: w,
      prob: p,
      count: c,
    }));

  if (!ranked.length) {
    // Lịch sử chưa thấy trong toàn corpus → lùi về tần suất unigram
    return getTopCandidatesUnigram(limit);
  }
  return ranked.slice(0, limit);
}

function getTopCandidatesUnigram(limit) {
  const cont = listContinuations([], 1).filter(([w]) => isWordLike(w));
  // Chuẩn hóa lại trên token dạng từ để palette khởi đầu rõ hơn
  const total = cont.reduce((s, [, c]) => s + c, 0);
  return cont
    .map(([w, c]) => ({
      word: displayForm(w),
      key: w,
      prob: total ? c / total : 0,
      count: c,
    }))
    .slice(0, limit);
}

function renderSequence() {
  const need = state.order - 1;
  if (!state.sequence.length) {
    els.sequence.innerHTML = `<span class="sequence-placeholder">Chọn một từ bắt đầu bên dưới…</span>`;
    els.historyLine.innerHTML = `Lịch sử để dự đoán: <em>(trống — đang hiện 5 từ xuất hiện nhiều nhất trong toàn bộ Emma)</em>`;
    return;
  }

  els.sequence.innerHTML = "";
  state.sequence.forEach((word, i) => {
    const span = document.createElement("span");
    span.className = "seq-token";
    const inHistory = need > 0 && i >= state.sequence.length - need;
    if (inHistory) span.classList.add("is-context");
    span.textContent = word;
    els.sequence.appendChild(span);
  });

  const hist = currentHistoryKeys();
  if (!hist.length) {
    els.historyLine.innerHTML = `Lịch sử để dự đoán: <em>(unigram · toàn corpus)</em>`;
  } else {
    const shown = state.sequence.slice(-hist.length);
    els.historyLine.innerHTML = `Lịch sử (${hist.length} từ cuối): <strong>${escapeHtml(
      shown.join(" ")
    )}</strong> · MLE ${hist.length + 1}-gram toàn corpus`;
  }
}

function renderSuggestions() {
  const top = getTopCandidates(5);
  els.suggestions.innerHTML = "";
  if (!top.length) {
    els.suggestions.innerHTML = `<p class="equation-hint">Không có tiếp nối cho lịch sử này.</p>`;
    return;
  }
  top.forEach((item, rank) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion";
    btn.setAttribute("role", "listitem");
    btn.innerHTML = `
      <span class="rank">#${rank + 1}</span>
      <span class="word">${escapeHtml(item.word)}</span>
      <span class="prob">P = ${formatProb(item.prob)} · C = ${item.count.toLocaleString("vi-VN")}</span>
    `;
    btn.addEventListener("click", () => {
      state.sequence.push(item.word);
      renderSequence();
      renderSuggestions();
    });
    els.suggestions.appendChild(btn);
  });
}

function wireControls() {
  document.querySelectorAll(".n-btn").forEach((btn) => {
    btn.addEventListener("click", () => setOrder(Number(btn.dataset.n)));
  });
  els.clearSeq.addEventListener("click", () => {
    state.sequence = [];
    renderSequence();
    renderSuggestions();
  });
  window.addEventListener("resize", () => {
    if (state.selectedIndex != null) highlightSelection(state.selectedIndex);
  });
}

function main() {
  const data =
    (typeof CORPUS_DATA !== "undefined" && CORPUS_DATA) ||
    (typeof window !== "undefined" && window.CORPUS_DATA);
  if (!data?.paragraphs || !data?.models) {
    throw new Error("Không tải được corpus-data.js. Hãy giữ index.html, app.js và corpus-data.js cùng thư mục.");
  }

  state.sep = data.sep || "\u0001";
  state.models = data.models;
  state.displayForm = data.display_form || {};
  state.totalTokens = data.total_tokens || 0;
  state.numSentences = data.num_sentences || 0;
  state.paragraphs = data.paragraphs;

  state.tokens = [];
  state.paragraphs.forEach((para, paraIndex) => {
    para.tokens.forEach((display, localIndex) => {
      state.tokens.push({
        display,
        key: String(display).toLowerCase(),
        globalIndex: state.tokens.length,
        paraIndex,
        localIndex,
      });
    });
  });

  if (els.corpusStats) {
    els.corpusStats.textContent =
      `Mô hình huấn luyện trên toàn bộ Emma · ${state.totalTokens.toLocaleString("vi-VN")} token · ` +
      `${state.numSentences.toLocaleString("vi-VN")} câu · các đoạn mẫu bên dưới chỉ để kiểm tra`;
  }

  wireControls();
  renderParagraphs();
  setOrder(2);
}

try {
  main();
} catch (err) {
  document.body.innerHTML = `
    <div style="max-width:40rem;margin:3rem auto;font-family:sans-serif;padding:1rem">
      <h1>Không khởi động được demo</h1>
      <p>${escapeHtml(err.message)}</p>
    </div>
  `;
}
