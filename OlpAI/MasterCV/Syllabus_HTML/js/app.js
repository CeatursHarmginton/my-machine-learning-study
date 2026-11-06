/* ==========================================================================
   MasterCV — First-Principles Computer Vision Interactive Engine
   Progress Sync, Theme Switcher, Code Copy, Tabs & Architecture Composer
   ========================================================================== */

const STORAGE_KEY_PROGRESS = 'mastercv_progress_v2';
const STORAGE_KEY_THEME = 'mastercv_theme_v2';
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

/* 3. Interactive Tabs */
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

/* 4. Enhance Code Headers & Copy Buttons */
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

/* 5. Interactive Architecture Composer (index.html) */
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

/* 3b. Toggleable From-Scratch Code Drawers (Default Hidden) */
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
