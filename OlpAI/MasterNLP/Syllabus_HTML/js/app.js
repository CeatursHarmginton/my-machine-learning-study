/* ==========================================================================
   MasterNLP — First-Principles Natural Language Processing & LLM Engine
   Progress Sync, Theme Switcher, Top Context Bar, ScrollSpy & NLP Composer
   ========================================================================== */

const STORAGE_KEY_PROGRESS = 'masternlp_progress_v1';
const STORAGE_KEY_THEME = 'masternlp_theme_v1';
const TOTAL_SYLLABUS_CHECKPOINTS = 54;
const TRACK_SHORT_NAME = 'MasterNLP';

const STAGE_CHECK_SPECS = {
  'foundations.html': { prefix: 'nb0-', total: 6 },
  'stage-0.html': { prefix: 'n0-', total: 3 },
  'stage-1.html': { prefix: 'n1-', total: 8 },
  'stage-2.html': { prefix: 'n2-', total: 8 },
  'stage-3.html': { prefix: 'n3-', total: 6 },
  'stage-4.html': { prefix: 'n4-', total: 6 },
  'stage-5.html': { prefix: 'n5-', total: 8 },
  'stage-6.html': { prefix: 'n6-', total: 9 }
};

const ICONS = {
  sun: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
  moon: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/></svg>',
  reset: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>',
  menu: '<svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/></svg>',
  code: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>',
  copy: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg>',
  check: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  chevronDown: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>',
  chevronUp: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>',
  arrowUp: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>'
};

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
  const isDark = theme === 'dark';
  const btn = document.getElementById('theme-toggle-btn');
  if (btn) {
    btn.innerHTML = `${isDark ? ICONS.sun : ICONS.moon}<span>${isDark ? 'Giao diện Sáng' : 'Giao diện Tối'}</span>`;
  }
  const topBtn = document.getElementById('topbar-theme-btn');
  if (topBtn) {
    topBtn.innerHTML = isDark ? ICONS.sun : ICONS.moon;
    topBtn.setAttribute('title', isDark ? 'Chuyển sang Giao diện Sáng' : 'Chuyển sang Giao diện Tối');
  }
  const resetBtn = document.querySelector('.sidebar-footer button[onclick*="resetAllProgress"]');
  if (resetBtn && !resetBtn.dataset.iconified) {
    resetBtn.dataset.iconified = '1';
    resetBtn.innerHTML = `${ICONS.reset}<span>Đặt lại</span>`;
  }
}

/* 2. Persistent Checklists & Sidebar Stage Stats */
function getCheckedMap() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY_PROGRESS) || '{}');
  } catch (_e) {
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

function updateSidebarStageStats(checkedMap) {
  const keys = Object.keys(checkedMap);
  document.querySelectorAll('.sidebar .nav-link').forEach((link) => {
    const href = (link.getAttribute('href') || '').replace(/^\.\//, '').split('#')[0];
    const spec = STAGE_CHECK_SPECS[href];
    if (!spec) return;

    const doneCount = keys.filter((k) => k.startsWith(spec.prefix)).length;
    let statEl = link.querySelector('.nav-stage-stat');
    if (!statEl) {
      statEl = document.createElement('span');
      statEl.className = 'nav-stage-stat';
      link.appendChild(statEl);
    }
    statEl.textContent = `${doneCount}/${spec.total}`;
    statEl.classList.remove('has-progress', 'is-complete');
    if (doneCount >= spec.total) {
      statEl.classList.add('is-complete');
    } else if (doneCount > 0) {
      statEl.classList.add('has-progress');
    }
  });
}

function updateGlobalProgressUI() {
  const checkedMap = getCheckedMap();
  const completedCount = Object.keys(checkedMap).length;
  const pct = Math.min(100, Math.round((completedCount / TOTAL_SYLLABUS_CHECKPOINTS) * 100));

  const fillEl = document.getElementById('global-progress-fill');
  const textEl = document.getElementById('global-progress-text');
  if (fillEl) fillEl.style.width = pct + '%';
  if (textEl) textEl.textContent = `${completedCount}/${TOTAL_SYLLABUS_CHECKPOINTS} (${pct}%)`;

  const localBoxes = document.querySelectorAll('input[type="checkbox"][data-check-id]');
  const localSummary = document.getElementById('page-progress-summary');
  if (localSummary && localBoxes.length > 0) {
    let localDone = 0;
    localBoxes.forEach((b) => { if (b.checked) localDone++; });
    localSummary.textContent = `Hoàn thành trang này: ${localDone}/${localBoxes.length} mục`;
  }

  updateSidebarStageStats(checkedMap);
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

function setScratchDrawerState(drawer, isOpen) {
  const btn = drawer.querySelector('.scratch-toggle-btn');
  const actionSpan = drawer.querySelector('.scratch-btn-action');
  drawer.classList.toggle('open', isOpen);
  if (btn) btn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  if (actionSpan) {
    actionSpan.innerHTML = isOpen
      ? `<span>Ẩn Code Tự Build</span>${ICONS.chevronUp}`
      : `<span>Hiện Code Tự Build</span>${ICONS.chevronDown}`;
  }
}

function initScratchToggles() {
  document.querySelectorAll('.scratch-drawer').forEach((drawer) => {
    const btn = drawer.querySelector('.scratch-toggle-btn');
    if (!btn) return;
    setScratchDrawerState(drawer, drawer.classList.contains('open'));

    btn.addEventListener('click', () => {
      const nextOpen = !drawer.classList.contains('open');
      setScratchDrawerState(drawer, nextOpen);
      syncExpandAllButtonLabel();
    });
  });
}

function syncExpandAllButtonLabel() {
  const expandBtn = document.getElementById('expand-all-scratch-btn');
  if (!expandBtn) return;
  const drawers = Array.from(document.querySelectorAll('.scratch-drawer'));
  const allOpen = drawers.length > 0 && drawers.every((d) => d.classList.contains('open'));
  expandBtn.innerHTML = `${ICONS.code}<span>${allOpen ? 'Thu gọn Tự Build' : 'Mở tất cả Tự Build'}</span>`;
}

/* 4. Enhance Code Headers, Copy Buttons & Table Scroll Wrappers */
function initCodeBlocks() {
  document.querySelectorAll('.code-header').forEach((header) => {
    const firstSpan = header.querySelector('span');
    if (firstSpan && !firstSpan.classList.contains('code-header-left') && !header.querySelector('.window-dots')) {
      const leftWrap = document.createElement('div');
      leftWrap.className = 'code-header-left';
      leftWrap.innerHTML = '<span class="window-dots"><i></i><i></i><i></i></span>';
      leftWrap.appendChild(firstSpan);
      header.insertBefore(leftWrap, header.firstChild);
    }
  });

  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.innerHTML = `${ICONS.copy}<span>Sao chép</span>`;
    btn.addEventListener('click', () => {
      const container = btn.closest('.code-container');
      const pre = container ? container.querySelector('pre') : null;
      if (!pre) return;
      navigator.clipboard.writeText(pre.dataset.rawCode || pre.textContent).then(() => {
        btn.classList.add('copied');
        btn.innerHTML = `${ICONS.check}<span>Đã chép</span>`;
        setTimeout(() => {
          btn.classList.remove('copied');
          btn.innerHTML = `${ICONS.copy}<span>Sao chép</span>`;
        }, 1600);
      });
    });
  });

  document.querySelectorAll('.res-table').forEach((tbl) => {
    if (!tbl.parentElement || tbl.parentElement.classList.contains('table-scroll-wrap')) return;
    const wrap = document.createElement('div');
    wrap.className = 'table-scroll-wrap';
    tbl.parentNode.insertBefore(wrap, tbl);
    wrap.appendChild(tbl);
  });
}

/* 5. Top Context Bar, Quick-Jump ScrollSpy, Mobile Drawer & Back-to-Top */
function extractShortJumpLabel(el, idx) {
  if (el.matches('article.week-block')) {
    const numEl = el.querySelector('.week-num');
    const raw = numEl ? numEl.textContent.trim() : '';
    const mWeek = raw.match(/^(Tuần\s*\d+(?:\s*[–-]\s*\d+)?)/i);
    if (mWeek) return mWeek[1];
    const mBase = raw.match(/^Bài\s+Cơ\s+Bản\s+(\d+)/i);
    if (mBase) return `Bài ${mBase[1]}`;
    const firstSeg = raw.split('·')[0].trim();
    if (firstSeg && firstSeg.length <= 22) return firstSeg;
    return `Phần ${idx + 1}`;
  }
  const text = el.textContent.trim();
  const mSec = text.match(/^(\d+)\.\s*(.+)$/);
  if (mSec) {
    const shortTitle = mSec[2].split(/[—(&]/)[0].trim();
    return shortTitle.length <= 24 ? `${mSec[1]}. ${shortTitle}` : `Mục ${mSec[1]}`;
  }
  return `Mục ${idx + 1}`;
}

function initReadingShell() {
  const main = document.querySelector('.main-content');
  const sidebar = document.querySelector('.sidebar');
  if (!main || !sidebar) return;

  const progBar = document.createElement('div');
  progBar.className = 'reading-progress-bar';
  progBar.innerHTML = '<div class="reading-progress-fill" id="reading-progress-fill"></div>';
  document.body.appendChild(progBar);

  const scrim = document.createElement('div');
  scrim.className = 'sidebar-scrim';
  document.body.appendChild(scrim);

  const closeMobileMenu = () => {
    sidebar.classList.remove('mobile-open');
    scrim.classList.remove('open');
  };
  scrim.addEventListener('click', closeMobileMenu);
  sidebar.querySelectorAll('.nav-link').forEach((l) => l.addEventListener('click', closeMobileMenu));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMobileMenu();
  });

  let targets = Array.from(main.querySelectorAll('article.week-block[id]'));
  if (targets.length < 2) {
    const secs = Array.from(main.querySelectorAll('h2.section-title'));
    secs.forEach((h2, idx) => {
      if (!h2.id) h2.id = `section-${idx + 1}`;
    });
    targets = secs;
  }

  const activeNav = sidebar.querySelector('.nav-link.active span:nth-child(2)');
  const currentTitle = activeNav ? activeNav.textContent.trim() : 'Tổng quan';

  const topbar = document.createElement('header');
  topbar.className = 'top-context-bar';

  const hasScratch = document.querySelectorAll('.scratch-drawer').length > 0;
  const jumpsHtml = targets
    .map((el, i) => `<a class="jump-pill${i === 0 ? ' active' : ''}" href="#${el.id}" data-jump-target="${el.id}">${extractShortJumpLabel(el, i)}</a>`)
    .join('');

  topbar.innerHTML = `
    <div class="topbar-left">
      <button type="button" class="mobile-menu-btn" id="mobile-menu-btn" aria-label="Mở danh mục">${ICONS.menu}</button>
      <div class="topbar-crumb">
        <strong>${TRACK_SHORT_NAME}</strong>
        <span class="crumb-sep">/</span>
        <span class="crumb-sub">${currentTitle}</span>
      </div>
    </div>
    <nav class="topbar-jumps" aria-label="Mục lục trang">${jumpsHtml}</nav>
    <div class="topbar-actions">
      ${hasScratch ? `<button type="button" class="topbar-icon-btn" id="expand-all-scratch-btn">${ICONS.code}<span>Mở tất cả Tự Build</span></button>` : ''}
      <button type="button" class="topbar-icon-btn" id="topbar-theme-btn" aria-label="Đổi giao diện"></button>
    </div>
  `;

  main.insertBefore(topbar, main.firstChild);

  const menuBtn = document.getElementById('mobile-menu-btn');
  if (menuBtn) {
    menuBtn.addEventListener('click', () => {
      const open = sidebar.classList.toggle('mobile-open');
      scrim.classList.toggle('open', open);
    });
  }

  const topThemeBtn = document.getElementById('topbar-theme-btn');
  if (topThemeBtn) {
    topThemeBtn.addEventListener('click', toggleTheme);
  }

  const expandAllBtn = document.getElementById('expand-all-scratch-btn');
  if (expandAllBtn) {
    expandAllBtn.addEventListener('click', () => {
      const drawers = Array.from(document.querySelectorAll('.scratch-drawer'));
      const shouldOpen = !drawers.every((d) => d.classList.contains('open'));
      drawers.forEach((d) => setScratchDrawerState(d, shouldOpen));
      syncExpandAllButtonLabel();
    });
  }

  const backBtn = document.createElement('button');
  backBtn.type = 'button';
  backBtn.className = 'back-to-top-btn';
  backBtn.setAttribute('aria-label', 'Lên đầu trang');
  backBtn.setAttribute('title', 'Lên đầu trang');
  backBtn.innerHTML = ICONS.arrowUp;
  backBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  document.body.appendChild(backBtn);

  const jumpPills = Array.from(topbar.querySelectorAll('.jump-pill'));
  const progFill = document.getElementById('reading-progress-fill');

  const onScroll = () => {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = docHeight > 20 ? Math.min(100, Math.max(0, (scrollTop / docHeight) * 100)) : 0;
    if (progFill) progFill.style.width = `${ratio}%`;
    backBtn.classList.toggle('visible', scrollTop > 420);

    if (targets.length > 0 && jumpPills.length > 0) {
      let activeId = targets[0].id;
      const offset = scrollTop + 130;
      for (const t of targets) {
        if (t.offsetTop <= offset) {
          activeId = t.id;
        }
      }
      jumpPills.forEach((p) => {
        p.classList.toggle('active', p.getAttribute('data-jump-target') === activeId);
      });
    }
  };

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  updateThemeButtonLabel(document.documentElement.getAttribute('data-theme') || 'dark');
}

/* 6. Interactive NLP Architecture Composer (index.html) */
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
      <div class="pipeline-stage-name">${s.item.name}</div>
      <a href="./${s.item.link}" class="pipeline-stage-link">Đi tới bài học →</a>
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

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initReadingShell();
  initChecklists();
  initTabs();
  initScratchToggles();
  if (typeof window.initExternalCodeLoader === 'function') {
    await window.initExternalCodeLoader();
  }
  initCodeBlocks();
  initComposer();
  initMathRendering();
});
