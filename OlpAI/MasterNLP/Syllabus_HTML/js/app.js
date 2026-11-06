/* ==========================================================================
   MasterNLP — First-Principles Natural Language Processing & LLM Engine
   Progress Sync, Theme Switcher, Code Copy, Scratch Drawers & NLP Composer
   ========================================================================== */

const STORAGE_KEY_PROGRESS = 'masternlp_progress_v1';
const STORAGE_KEY_THEME = 'masternlp_theme_v1';
const TOTAL_SYLLABUS_CHECKPOINTS = 54;

/* 1. Theme Switcher */
function initTheme() {
  const saved = localStorage.getItem(STORAGE_KEY_THEME) || 'dark';
  document.documentElement.setAttribute('data-theme', saved);
  updateThemeButtonLabel(saved);
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem(STORAGE_KEY_THEME, next);
  updateThemeButtonLabel(next);
}

function updateThemeButtonLabel(theme) {
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) {
    btn.textContent = theme === 'dark' ? 'Chế độ Sáng' : 'Chế độ Tối';
  }
}

/* 2. Persistent Checklists */
function getCheckedMap() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY_PROGRESS) || '{}');
  } catch (e) {
    return {};
  }
}

function saveCheckedMap(map) {
  localStorage.setItem(STORAGE_KEY_PROGRESS, JSON.stringify(map));
}

function initChecklists() {
  const checkedMap = getCheckedMap();
  const checkboxes = document.querySelectorAll('input[type="checkbox"][data-check-id]');

  checkboxes.forEach((cb) => {
    const id = cb.getAttribute('data-check-id');
    const wrapper = cb.closest('.check-item');
    if (checkedMap[id]) {
      cb.checked = true;
      if (wrapper) wrapper.classList.add('done');
    }

    cb.addEventListener('change', () => {
      const latestMap = getCheckedMap();
      if (cb.checked) {
        latestMap[id] = true;
        if (wrapper) wrapper.classList.add('done');
      } else {
        delete latestMap[id];
        if (wrapper) wrapper.classList.remove('done');
      }
      saveCheckedMap(latestMap);
      updateGlobalProgressUI();
    });
  });

  updateGlobalProgressUI();
}

function updateGlobalProgressUI() {
  const checkedMap = getCheckedMap();
  const completedCount = Object.keys(checkedMap).length;
  const pct = Math.min(100, Math.round((completedCount / TOTAL_SYLLABUS_CHECKPOINTS) * 100));

  const fillEl = document.getElementById('global-progress-fill');
  const textEl = document.getElementById('global-progress-text');
  if (fillEl) fillEl.style.width = pct + '%';
  if (textEl) textEl.textContent = `${completedCount}/${TOTAL_SYLLABUS_CHECKPOINTS} Mục (${pct}%)`;

  const localBoxes = document.querySelectorAll('input[type="checkbox"][data-check-id]');
  const localSummary = document.getElementById('page-progress-summary');
  if (localSummary && localBoxes.length > 0) {
    let localDone = 0;
    localBoxes.forEach((b) => { if (b.checked) localDone++; });
    localSummary.textContent = `Tiến độ giai đoạn này: ${localDone}/${localBoxes.length} mục`;
  }
}

function resetAllProgress() {
  if (confirm('Đặt lại toàn bộ tiến độ hoàn thành NLP về 0%?')) {
    localStorage.removeItem(STORAGE_KEY_PROGRESS);
    document.querySelectorAll('input[type="checkbox"][data-check-id]').forEach((cb) => {
      cb.checked = false;
      const w = cb.closest('.check-item');
      if (w) w.classList.remove('done');
    });
    updateGlobalProgressUI();
  }
}

/* 3. Interactive Tabs & Toggleable From-Scratch Drawers */
function initTabs() {
  document.querySelectorAll('.tabs').forEach((tabGroup) => {
    const buttons = tabGroup.querySelectorAll('.tab-btn');
    const panels = tabGroup.querySelectorAll('.tab-panel');

    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const target = btn.getAttribute('data-tab');
        buttons.forEach((b) => b.classList.remove('active'));
        panels.forEach((p) => p.classList.remove('active'));
        btn.classList.add('active');
        const activePanel = tabGroup.querySelector(`.tab-panel[data-panel="${target}"]`);
        if (activePanel) activePanel.classList.add('active');
      });
    });
  });
}

function initScratchToggles() {
  document.querySelectorAll('.scratch-drawer').forEach((drawer) => {
    const btn = drawer.querySelector('.scratch-toggle-btn');
    const actionSpan = drawer.querySelector('.scratch-btn-action');
    if (!btn) return;

    btn.addEventListener('click', () => {
      const isOpen = drawer.classList.toggle('open');
      btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
      if (actionSpan) {
        actionSpan.textContent = isOpen ? 'Ẩn Code Tự Build ▴' : 'Hiện Code Tự Build ▾';
      }
    });
  });
}

/* 4. Enhance Code Headers & Clean Clipboard Copy */
function initCodeBlocks() {
  document.querySelectorAll('.code-header').forEach((header) => {
    const firstSpan = header.querySelector('span');
    if (firstSpan && !header.querySelector('.window-dots')) {
      const leftWrap = document.createElement('div');
      leftWrap.className = 'code-header-left';
      leftWrap.innerHTML = '<span class="window-dots"><i></i><i></i><i></i></span>';
      leftWrap.appendChild(firstSpan);
      header.insertBefore(leftWrap, header.firstChild);
    }
  });

  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const container = btn.closest('.code-container');
      const pre = container ? container.querySelector('pre') : null;
      if (!pre) return;
      navigator.clipboard.writeText(pre.dataset.rawCode || pre.textContent).then(() => {
        const orig = btn.textContent;
        btn.textContent = 'Đã chép ✓';
        setTimeout(() => { btn.textContent = orig; }, 1600);
      });
    });
  });
}

/* 5. Interactive NLP Architecture Composer (index.html) */
const PROBLEM_ARCHETYPES = {
  multitask_affect: {
    title: 'Dạng 1: Phân loại đa nhãn, Cảm xúc liên tục (Valence-Arousal) & Đa nhiệm xung đột (VD: Feedback Prize, Jigsaw, MXH)',
    diagnosis: 'Chẩn đoán nguyên lý: Văn bản chứa teencode/ẩn dụ đa chiều; chỉ lấy vector [CLS] làm mất tín hiệu từ khóa rải rác, còn cộng tuyến tính nhiều hàm mất mát gây triệt tiêu đạo hàm giữa các nhánh.',
    kaggle: 'Bài luyện chuẩn: Kaggle Feedback Prize ELL & Jigsaw Unintended Bias',
    m1: { tag: 'Token & Split', name: 'Multilabel/Group Split + Dynamic Padding Collator', link: 'stage-0.html#baseline-core' },
    m2: { tag: 'Encoder', name: 'DeBERTa-v3 / PhoBERT + FGM/AWP Adversarial', link: 'stage-1.html#week-2' },
    m3: { tag: 'Pooler & Head', name: 'Attention / MeanMax + Weighted 4-Layer Pooler', link: 'stage-1.html#week-1' },
    m4: { tag: 'Loss & Grad', name: 'CCC / Asymmetric Focal Loss + PCGrad Surgery', link: 'stage-1.html#week-4' },
    m5: { tag: 'Inference', name: 'OOF Threshold Tuner + Cross-Tokenizer Blend', link: 'stage-6.html#week-20' }
  },
  nested_span_ner: {
    title: 'Dạng 2: Trích xuất Thực thể Lồng nhau, Biên Dài >512 Token & Bộ Tứ Khía Cạnh ACOS (VD: NBME Clinical, PII Detection)',
    diagnosis: 'Chẩn đoán nguyên lý: Gán nhãn BIO từng token độc lập sinh chuỗi bất hợp lệ ($O \\to I\\text{-PER}$) và bất lực hoàn toàn khi thực thể lồng nhau. Văn bản dài bị cắt cụt ở 512 subwords.',
    kaggle: 'Bài luyện chuẩn: Kaggle NBME Clinical Notes & Learning Agency PII',
    m1: { tag: 'Token & Split', name: 'Offset Mapping Alignment + Stride Sliding Window', link: 'stage-2.html#week-6' },
    m2: { tag: 'Encoder', name: 'DeBERTa-v3 Long-Context / PhoBERT Word-Segmented', link: 'stage-2.html#week-5' },
    m3: { tag: 'Span / Tag Head', name: 'Linear-Chain CRF hoặc Biaffine Span Matrix $(i, j)$', link: 'stage-2.html#week-6' },
    m4: { tag: 'Loss & Grad', name: 'Negative Log-Likelihood CRF / Span Focal BCE', link: 'stage-2.html#week-5' },
    m5: { tag: 'Inference', name: 'Viterbi Decoding + Character-Level Span Voting', link: 'stage-6.html#week-20' }
  },
  cipher_lowresource: {
    title: 'Dạng 3: Ngôn ngữ Mật mã Chưa Biết / Xuyên Ngôn Ngữ Ít Nhãn (Đề thi thật IOAI 2024 & Cross-Lingual Transfer)',
    diagnosis: 'Chẩn đoán nguyên lý: Bộ tách từ tiếng Anh có sẵn băm nát ngôn ngữ lạ/mật mã thành ký tự vô nghĩa ($\\text{UNK}$). Cần tự huấn luyện từ vựng BPE mới và tiền huấn luyện MLM thích nghi miền (TAPT).',
    kaggle: 'Bài luyện chuẩn: IOAI 2024 Ciphered NLP & Contradictory My Dear Watson',
    m1: { tag: 'Tokenizer', name: 'Train Custom BPE / Unigram Vocabulary From Scratch', link: 'stage-0.html#baseline-core' },
    m2: { tag: 'Pretraining', name: 'Task-Adaptive Masked Language Modeling (15% MLM)', link: 'stage-1.html#week-3' },
    m3: { tag: 'Cross-Lingual', name: 'XLM-RoBERTa Shared Space + Mean Pooling', link: 'stage-1.html#week-3' },
    m4: { tag: 'Loss & Grad', name: 'Contrastive Translation Alignment + R-Drop KL', link: 'stage-4.html#week-13' },
    m5: { tag: 'Inference', name: 'Confidence Pseudo-Labeling trên tập Unlabeled', link: 'stage-6.html#week-20' }
  },
  hybrid_rag_search: {
    title: 'Dạng 4: Truy xuất Triệu Đoạn Văn Bản Pháp Luật / Khoa Học & Hỏi Đáp RAG Chống Ảo Giác (VD: Kaggle LLM Science Exam)',
    diagnosis: 'Chẩn đoán nguyên lý: Vector Dense bỏ sót mã điều khoản/thuật ngữ hiếm, còn BM25 mù từ đồng nghĩa. Bi-Encoder nhanh nhưng kém chính xác hơn Cross-Encoder chú ý chéo từng cặp token.',
    kaggle: 'Bài luyện chuẩn: Kaggle LLM Science Exam & Learning Equality',
    m1: { tag: 'Chunking', name: 'Structure-Aware Chunking + Hard Negative Mining', link: 'stage-3.html#week-10' },
    m2: { tag: 'Dual Retriever', name: 'Okapi BM25 + Contrastive Bi-Encoder (InfoNCE)', link: 'stage-3.html#week-9' },
    m3: { tag: 'Reranker', name: 'Reciprocal Rank Fusion (RRF) → Cross-Encoder', link: 'stage-3.html#week-10' },
    m4: { tag: 'Generator', name: 'Citation-Grounded Reader / Multiple-Choice Head', link: 'stage-3.html#week-11' },
    m5: { tag: 'Verification', name: 'NLI Entailment Hallucination Guard $P(\\text{entail})$', link: 'stage-3.html#week-11' }
  },
  seq2seq_summarization: {
    title: 'Dạng 5: Tóm tắt & Sinh Văn Bản Chất Lượng Cao — Vượt Bẫy Exposure Bias (VD: SimCLS / BRIO, CommonLit Summaries)',
    diagnosis: 'Chẩn đoán nguyên lý: Huấn luyện Teacher Forcing tối ưu xác suất từng từ tiếp theo nhưng lúc suy luận mô hình tự ăn lỗi của chính nó, khiến bản tóm tắt điểm Likelihood cao lại có ROUGE thấp.',
    kaggle: 'Bài luyện chuẩn: Kaggle CommonLit Summaries & CNN/DailyMail',
    m1: { tag: 'Data & Prompt', name: 'Salient Sentence Filtering + Prefix Formatting', link: 'stage-4.html#week-12' },
    m2: { tag: 'Seq2Seq Core', name: 'Encoder-Decoder (T5 / BART / ViT5) + KV-Cache', link: 'stage-4.html#week-12' },
    m3: { tag: 'Candidates', name: 'Diverse Beam Search sinh $K=16$ ứng viên đa dạng', link: 'stage-4.html#week-12' },
    m4: { tag: 'Contrastive', name: 'SimCLS Candidate Scorer (Margin Ranking Loss)', link: 'stage-4.html#week-13' },
    m5: { tag: 'Inference', name: 'Length-Penalized Reranking + Minimum Bayes Risk', link: 'stage-4.html#week-13' }
  },
  llm_alignment_lora: {
    title: 'Dạng 6: Tinh chỉnh LLM Tham số Nhỏ (LoRA) & Căn chỉnh Sở thích Con người DPO (VD: LMSYS Chatbot Arena)',
    diagnosis: 'Chẩn đoán nguyên lý: Full fine-tuning LLM hàng tỷ tham số gây quá khớp tức thì trên tập nhỏ. Cần đóng băng mạng gốc, chèn nhánh hạng thấp $\\Delta W = \\frac{\\alpha}{r}BA$ và tối ưu tỷ số xác suất DPO.',
    kaggle: 'Bài luyện chuẩn: Kaggle LMSYS Chatbot Arena & WSDM Cup',
    m1: { tag: 'Formatting', name: 'Chat Template + Assistant-Only Loss Mask (-100)', link: 'stage-5.html#week-15' },
    m2: { tag: 'LLM Backbone', name: 'Qwen2.5 / Llama-3 (RoPE + GQA) Mixed Precision', link: 'stage-5.html#week-15' },
    m3: { tag: 'Adapter', name: 'Inject LoRA $\\Delta W = \\frac{\\alpha}{r}BA$ bằng PyTorch thuần', link: 'stage-5.html#week-15' },
    m4: { tag: 'Alignment', name: 'Bradley-Terry Reward hoặc Closed-Form DPO Loss', link: 'stage-5.html#week-16' },
    m5: { tag: 'Inference', name: 'Swap-Order Debiasing (chống thiên kiến vị trí A/B)', link: 'stage-5.html#week-16' }
  },
  model_surgery_forensics: {
    title: 'Dạng 7: Phẫu thuật Trọng số Mô hình Hỏng, Activation Steering & Phát hiện Văn bản AI (IOAI 2025 & Kaggle DAIGT)',
    diagnosis: 'Chẩn đoán nguyên lý: Đề thi khóa dữ liệu huấn luyện lớn hoặc làm hỏng ma trận nhúng (Broken BERT), đòi hỏi căn chỉnh trực giao Procrustes $\\Omega = UV^\\top$ và kết hợp thống kê n-gram + độ hỗn loạn token.',
    kaggle: 'Bài luyện chuẩn: IOAI 2025 Broken BERT & Kaggle Detect AI Text',
    m1: { tag: 'Anchor Pairs', name: 'Trích xuất cặp từ neo / cặp câu đối lập hành vi', link: 'stage-5.html#week-17' },
    m2: { tag: 'White-Box', name: 'Can thiệp Residual Stream & Embedding Matrix', link: 'stage-5.html#week-17' },
    m3: { tag: 'Closed-Form', name: 'Orthogonal Procrustes SVD + CAA Steering Vector', link: 'stage-5.html#week-17' },
    m4: { tag: 'Forensics', name: 'Subword TF-IDF (3–5 ngram) + Perplexity Curvature', link: 'stage-5.html#week-18' },
    m5: { tag: 'Calibration', name: 'Rank Averaging chống lệch phân phối miền chủ đề', link: 'stage-6.html#week-22-24' }
  },
  audio_sequence_order: {
    title: 'Dạng 8: Chuỗi Tiếng Nói - Ngôn Ngữ (ASR/CTC) & Sắp xếp Lại Thứ tự Hội thoại / Đoạn Code (IOAI 2026 & AI4Code)',
    diagnosis: 'Chẩn đoán nguyên lý: Độ dài khung âm thanh $T$ lớn gấp nhiều lần chuỗi ký tự $U$ mà không có căn chỉnh sẵn; còn bài toán sắp xếp thứ tự cần kết hợp điểm vị trí toàn cục và xác suất kề nhau từng cặp.',
    kaggle: 'Bài luyện chuẩn: Kaggle Bengali.AI Speech & Google AI4Code',
    m1: { tag: 'Signal / Pair', name: 'Log-Mel Spectrogram + SpecAugment / Context Window', link: 'stage-4.html#week-14' },
    m2: { tag: 'Encoder', name: 'Wav2Vec2 / Whisper hoặc Code-Text DeBERTa', link: 'stage-4.html#week-14' },
    m3: { tag: 'Alignment', name: 'CTC Blank Collapse $\\mathcal{B}$ / Pairwise Rank Head', link: 'stage-2.html#week-7' },
    m4: { tag: 'Loss', name: 'Forward-Backward CTC Loss / L1 Percentile Loss', link: 'stage-4.html#week-14' },
    m5: { tag: 'Decoding', name: 'LM-Guided CTC Beam Search / Kendall-$\\tau$ Sort', link: 'stage-2.html#week-7' }
  }
};

function initComposer() {
  const container = document.getElementById('arch-composer');
  if (!container) return;

  const btns = container.querySelectorAll('.scenario-btn');
  btns.forEach((btn) => {
    btn.addEventListener('click', () => {
      btns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      renderArchetype(btn.getAttribute('data-scenario'));
    });
  });

  renderArchetype('multitask_affect');
}

function renderArchetype(key) {
  const d = PROBLEM_ARCHETYPES[key];
  if (!d) return;

  document.getElementById('comp-title').textContent = d.title;
  document.getElementById('comp-diagnosis').textContent = d.diagnosis;
  document.getElementById('comp-kaggle').textContent = d.kaggle;

  const stages = [
    { num: '01 · Tokenizer & Split', cls: 'mod-1', item: d.m1 },
    { num: '02 · Encoder & Context', cls: 'mod-2', item: d.m2 },
    { num: '03 · Pooler & Span Head', cls: 'mod-3', item: d.m3 },
    { num: '04 · Loss & Alignment', cls: 'mod-4', item: d.m4 },
    { num: '05 · Decoding & Rerank', cls: 'mod-5', item: d.m5 }
  ];

  document.getElementById('comp-pipeline').innerHTML = stages.map((s) => `
    <div class="pipeline-stage">
      <div class="pipeline-stage-label">${s.num}</div>
      <div><span class="mod-badge ${s.cls}">${s.item.tag}</span></div>
      <div style="font-size:0.87rem; font-weight:600; color:var(--text-primary); line-height:1.45;">${s.item.name}</div>
      <a href="./${s.item.link}" style="font-size:0.78rem; margin-top:auto;">Đi tới bài học →</a>
    </div>
  `).join('');

  initMathRendering(document.getElementById('arch-composer'));
}

function initMathRendering(rootEl) {
  if (typeof renderMathInElement === 'function') {
    renderMathInElement(rootEl || document.body, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false }
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      throwOnError: false
    });
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initChecklists();
  initTabs();
  initScratchToggles();
  if (typeof window.initExternalCodeLoader === 'function') {
    window.initExternalCodeLoader();
  }
  initCodeBlocks();
  initComposer();
  initMathRendering();
});
