/* ==========================================================================
   MasterCV — First-Principles Computer Vision Interactive Engine
   Progress Sync, Theme Switcher, Top Context Bar, ScrollSpy & Composer
   ========================================================================== */

const STORAGE_KEY_PROGRESS = 'mastercv_progress_v2';
const STORAGE_KEY_THEME = 'mastercv_theme_v2';
const TOTAL_SYLLABUS_CHECKPOINTS = 54;
const TRACK_SHORT_NAME = 'MasterCV';

const STAGE_CHECK_SPECS = {
  'foundations.html': { prefix: 'b0-', total: 6 },
  'stage-0.html': { prefix: 's0-', total: 3 },
  'stage-1.html': { prefix: 's1-', total: 8 },
  'stage-2.html': { prefix: 's2-', total: 6 },
  'stage-3.html': { prefix: 's3-', total: 8 },
  'stage-4.html': { prefix: 's4-', total: 6 },
  'stage-5.html': { prefix: 's5-', total: 8 },
  'stage-6.html': { prefix: 's6-', total: 9 }
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
  if (confirm('Đặt lại toàn bộ tiến độ hoàn thành về 0%?')) {
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

  // Reading progress hairline at top of viewport
  const progBar = document.createElement('div');
  progBar.className = 'reading-progress-bar';
  progBar.innerHTML = '<div class="reading-progress-fill" id="reading-progress-fill"></div>';
  document.body.appendChild(progBar);

  // Mobile sidebar backdrop scrim
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

  // Discover jump targets
  let targets = Array.from(main.querySelectorAll('article.week-block[id]'));
  if (targets.length < 2) {
    const secs = Array.from(main.querySelectorAll('h2.section-title'));
    secs.forEach((h2, idx) => {
      if (!h2.id) h2.id = `section-${idx + 1}`;
    });
    targets = secs;
  }

  // Active page title from sidebar
  const activeNav = sidebar.querySelector('.nav-link.active span:nth-child(2)');
  const currentTitle = activeNav ? activeNav.textContent.trim() : 'Tổng quan';

  // Build Sticky Top Context Bar
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

  // Floating Back-to-Top button
  const backBtn = document.createElement('button');
  backBtn.type = 'button';
  backBtn.className = 'back-to-top-btn';
  backBtn.setAttribute('aria-label', 'Lên đầu trang');
  backBtn.setAttribute('title', 'Lên đầu trang');
  backBtn.innerHTML = ICONS.arrowUp;
  backBtn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  document.body.appendChild(backBtn);

  // Scroll listener for reading progress, back-to-top, and ScrollSpy
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

/* 6. Interactive Architecture Composer (index.html) */
const PROBLEM_ARCHETYPES = {
  fgvc_hierarchy: {
    title: 'Dạng 1: Phân loại siêu chi tiết & đa nhãn phân cấp (VD: Plant Pathology, Bengali.AI, Thực vật / Tranh cổ)',
    diagnosis: 'Chẩn đoán nguyên lý: Khác biệt giữa các lớp nằm ở chi tiết tần số cao (gân lá, nét bút), mô hình rất dễ học vẹt cục bộ vào 1 góc ảnh hoặc nền, đồng thời xảy ra xung đột đạo hàm giữa các nhánh phân loại cha - con.',
    kaggle: 'Bài luyện chuẩn: Kaggle Bengali.AI / Plant Pathology FGVC8',
    m1: { tag: 'Data & Split', name: 'Progressive High-Res + Mixup/CutMix Collator', link: 'stage-1.html#week-3' },
    m2: { tag: 'Backbone', name: 'ConvNeXt / Res2Net Multi-scale Feature Extractor', link: 'stage-1.html#week-1' },
    m3: { tag: 'Head', name: 'GeM Pooling + Hierarchical Multi-Task Head', link: 'stage-1.html#week-4' },
    m4: { tag: 'Loss', name: 'Soft-Target CE + PCGrad / Uncertainty MTL', link: 'stage-1.html#week-4' },
    m5: { tag: 'Inference', name: 'Horizontal Flip TTA + Bayes Consistency', link: 'stage-1.html#week-4' }
  },
  blurry_segmentation: {
    title: 'Dạng 2: Phân đoạn đối tượng biên mờ, hòa lẫn nền (VD: UW-Madison GI Tract, Severstal Steel, Camouflage)',
    diagnosis: 'Chẩn đoán nguyên lý: Vùng đối tượng chiếm $\\lt 3\\%$ diện tích ảnh và không có biên sắc nét. Cross-Entropy thường bị pixel nền lấn át, còn chú ý xuôi (Forward Attention) chỉ bắt phần lõi mà bỏ sót viền mảnh.',
    kaggle: 'Bài luyện chuẩn: Kaggle UW-Madison GI Tract / Severstal Steel',
    m1: { tag: 'Data & Split', name: 'GroupKFold + transforms.v2 Sync Image-Mask', link: 'stage-3.html#week-8' },
    m2: { tag: 'Backbone & Neck', name: 'FPN Pyramid + Reverse Attention $1 - \\sigma(P)$', link: 'stage-3.html#week-9' },
    m3: { tag: 'Head', name: 'Coarse-to-Fine Partial Decoder + Deep Sup', link: 'stage-3.html#week-9' },
    m4: { tag: 'Loss', name: 'Pixel-Adaptive Edge-Weighted BCE + IoU Loss', link: 'stage-3.html#week-8' },
    m5: { tag: 'Inference', name: 'Connected-Component Clean + Vectorized RLE', link: 'stage-3.html#week-9' }
  },
  imbalance_detection: {
    title: 'Dạng 3: Phát hiện vật thể siêu hiếm & mất cân bằng 1:10.000 (VD: VinBigData Chest X-ray, Lỗi vi mạch)',
    diagnosis: 'Chẩn đoán nguyên lý: Hàng trăm nghìn vùng nền dễ cộng dồn đạo hàm đè bẹp vài vùng tổn thương nhỏ. Nếu khởi tạo bias mặc định bằng $0$, mạng sụp đổ ngay epoch đầu tiên.',
    kaggle: 'Bài luyện chuẩn: Kaggle VinBigData Chest X-ray Detection',
    m1: { tag: 'Data & Split', name: 'Multi-scale Resize + Stratified Group Split', link: 'stage-3.html#week-10' },
    m2: { tag: 'Backbone & Neck', name: 'ConvNeXt / Swin + Feature Pyramid ($P_3\\text{–}P_7$)', link: 'stage-3.html#week-8' },
    m3: { tag: 'Head', name: 'Dense Head (Prior Bias Init) hoặc DETR Queries', link: 'stage-3.html#week-10' },
    m4: { tag: 'Loss', name: 'Sigmoid Focal Loss + GIoU / Hungarian Loss', link: 'stage-3.html#week-10' },
    m5: { tag: 'Inference', name: 'Weighted Boxes Fusion (WBF) + OOF Threshold', link: 'stage-3.html#week-10' }
  },
  dense_counting: {
    title: 'Dạng 4: Đếm vật thể dày đặc, chồng lấn lên nhau (VD: IOAI 2025 Chicken Counting, Global Wheat)',
    diagnosis: 'Chẩn đoán nguyên lý: Khi các vật thể chụm sát nhau, hộp bao chồng lấn mạnh khiến NMS xóa nhầm hàng loạt vật thể thật. Cần chuyển từ phát hiện rời rạc sang hồi quy tích phân bản đồ mật độ.',
    kaggle: 'Bài luyện chuẩn: IOAI 2025 Chicken Counting & Global Wheat',
    m1: { tag: 'Data & Split', name: 'Boundary-Preserving 2D Gaussian Density Map', link: 'stage-3.html#week-11' },
    m2: { tag: 'Backbone & Neck', name: 'Dilated Front-end + FPN giữ độ phân giải $1/4$', link: 'stage-3.html#week-11' },
    m3: { tag: 'Head', name: 'Non-Negative $1\\times 1$ Density Regressor', link: 'stage-3.html#week-11' },
    m4: { tag: 'Loss', name: 'Pixel MSE + Multi-scale Local Count Loss', link: 'stage-3.html#week-11' },
    m5: { tag: 'Inference', name: 'Spatial Integral Sum $\\sum \\hat{D}(x,y)$ + Tiled TTA', link: 'stage-3.html#week-11' }
  },
  open_retrieval: {
    title: 'Dạng 5: Truy xuất & Định danh Tập mở hàng vạn lớp (VD: Kaggle Happywhale, Shopee, Landmark Retrieval)',
    diagnosis: 'Chẩn đoán nguyên lý: Tập kiểm tra chứa các cá thể/sản phẩm/địa điểm mới chưa từng xuất hiện lúc train. Phải ánh xạ toàn bộ lên mặt cầu đơn vị $\\mathbb{S}^{d-1}$, gom cụm bất biến nền và tái xếp hạng đồ thị láng giềng.',
    kaggle: 'Bài luyện chuẩn: Kaggle Happywhale & Google Landmark Retrieval',
    m1: { tag: 'Data & Split', name: 'Entity/Geo GroupSplit + PK Instance Sampler', link: 'stage-2.html#week-5' },
    m2: { tag: 'Backbone', name: 'DINOv2 / ConvNeXt Multi-Layer Feature Hook', link: 'stage-5.html#week-15' },
    m3: { tag: 'Pooling & Head', name: 'GeM / NetVLAD + Sub-center ArcFace / CosFace', link: 'stage-2.html#week-6' },
    m4: { tag: 'Loss', name: 'Angular Margin + Batch-Hard Triplet / SupCon', link: 'stage-2.html#week-5' },
    m5: { tag: 'Inference', name: 'Fast Cosine Search + $\\alpha$-QE + $k$-Reciprocal Rerank', link: 'stage-2.html#week-7' }
  },
  domain_ood: {
    title: 'Dạng 6: Lệch phân phối miền (Domain Shift) kèm Vật thể lạ ngoài tập nhãn (Open-Set / OOD)',
    diagnosis: 'Chẩn đoán nguyên lý: Mạng học vẹt màu sắc thiết bị chụp ở tập Train, còn Softmax luôn quá tự tin ($p \\gt 0.98$) trên vật thể lạ do tính bất biến tịnh tiến của hàm mũ.',
    kaggle: 'Bài luyện chuẩn: WILDS Camelyon17 & OpenOOD Benchmark',
    m1: { tag: 'Data & Split', name: 'Leave-One-Domain-Out + Multi-Domain Sampler', link: 'stage-4.html#week-12' },
    m2: { tag: 'Backbone', name: 'Backbone gắn MixStyle (Layer 1–2) / GRL', link: 'stage-4.html#week-12' },
    m3: { tag: 'Head', name: 'Linear / Cosine Classifier bảo toàn biên độ Logit', link: 'stage-4.html#week-13' },
    m4: { tag: 'Loss', name: 'Deep CORAL Alignment + Energy Margin Loss', link: 'stage-4.html#week-13' },
    m5: { tag: 'Inference', name: 'Helmholtz Free Energy + Mahalanobis OOD Score', link: 'stage-4.html#week-13' }
  },
  diffusion_steering: {
    title: 'Dạng 7: Điều khiển mô hình sinh Diffusion không đổi Text Prompt (Đề thi thật IOAI 2024 Scientific)',
    diagnosis: 'Chẩn đoán nguyên lý: Prompt đầu vào bị khóa cứng và không có dữ liệu lớn để fine-tune dài. Ngữ nghĩa văn bản đi vào U-Net qua các ma trận chiếu Cross-Attention $W_K, W_V$ — giải trực tiếp bằng đại số tuyến tính nghiệm đóng.',
    kaggle: 'Bài luyện chuẩn: IOAI 2024 Official SDXL Steering Task',
    m1: { tag: 'Concept Pair', name: 'Cặp biểu diễn CLIP $\\mathbf{c}_{\\text{src}}$ & $\\mathbf{c}_{\\text{tgt}}$', link: 'stage-5.html#week-17' },
    m2: { tag: 'Target Layers', name: 'Trích xuất Cross-Attention $W_K, W_V$ trong U-Net', link: 'stage-5.html#week-17' },
    m3: { tag: 'Projection', name: 'Closed-Form Rank-1 / Ridge Matrix Update', link: 'stage-5.html#week-17' },
    m4: { tag: 'Objective', name: '$\\|W\\mathbf{c}_{\\text{src}} - W_{\\text{old}}\\mathbf{c}_{\\text{tgt}}\\|_2^2 + \\lambda\\|W - W_{\\text{old}}\\|_F^2$', link: 'stage-5.html#week-17' },
    m5: { tag: 'Verification', name: 'DDIM Deterministic Sampling + CLIP Score', link: 'stage-5.html#week-17' }
  },
  video_spatiotemporal: {
    title: 'Dạng 8: Nhận diện thao tác nhanh / sự cố Video & Đa đối tượng (VD: Kaggle DFL Bundesliga, NFL Tracking)',
    diagnosis: 'Chẩn đoán nguyên lý: Hai khung hình kề nhau dư thừa $\\gt 95\\%$ thông tin dễ gây rò rỉ thời gian khi học tự giám sát, còn hành vi ngắn đòi hỏi tần số lấy mẫu thời gian cao kết hợp hậu xử lý quỹ đạo.',
    kaggle: 'Bài luyện chuẩn: Kaggle DFL Bundesliga & NFL Contact Detection',
    m1: { tag: 'Data & Split', name: 'Video-level Split + Dual-Pathway / 90% Tube Mask', link: 'stage-6.html#week-20' },
    m2: { tag: 'Backbone', name: '2D+TSM / SlowFast Lateral / Asymmetric VideoMAE', link: 'stage-6.html#week-19' },
    m3: { tag: 'Head', name: 'Spatiotemporal Pool + Multi-Task Event Head', link: 'stage-6.html#week-19' },
    m4: { tag: 'Loss', name: 'Normalized Tube MSE → Fused LabelSmooth + Focal', link: 'stage-6.html#week-20' },
    m5: { tag: 'Inference', name: 'ByteTrack 2-Stage Association + Logic Check', link: 'stage-6.html#week-21' }
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

  renderArchetype('fgvc_hierarchy');
}

function renderArchetype(key) {
  const d = PROBLEM_ARCHETYPES[key];
  if (!d) return;

  document.getElementById('comp-title').textContent = d.title;
  document.getElementById('comp-diagnosis').textContent = d.diagnosis;
  document.getElementById('comp-kaggle').textContent = d.kaggle;

  const stages = [
    { num: '01 · Data & Validation', cls: 'mod-1', item: d.m1 },
    { num: '02 · Backbone & Neck', cls: 'mod-2', item: d.m2 },
    { num: '03 · Pooling & Head', cls: 'mod-3', item: d.m3 },
    { num: '04 · Loss & Gradient', cls: 'mod-4', item: d.m4 },
    { num: '05 · Inference & Metric', cls: 'mod-5', item: d.m5 }
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
