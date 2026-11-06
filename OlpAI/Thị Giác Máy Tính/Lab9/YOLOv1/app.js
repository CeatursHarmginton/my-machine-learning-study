(() => {
  "use strict";

  const S = 7;
  const B = 2;
  const CLASSES = ["dog", "bicycle", "car"];
  const C = CLASSES.length;
  const CELL_SIZE = 13; // B*5 + C
  const LAMBDA_COORD = 5;
  const LAMBDA_NOOBJ = 0.5;

  // Ground-truth boxes in image-normalized [xmin, ymin, xmax, ymax]
  // Centers fall in the classic YOLOv1 responsible cells.
  const GT_OBJECTS = [
    {
      name: "dog",
      classIndex: 0,
      color: "#c2410c",
      box: { xmin: 0.16, ymin: 0.36, xmax: 0.42, ymax: 0.94 },
    },
    {
      name: "bicycle",
      classIndex: 1,
      color: "#1d4ed8",
      box: { xmin: 0.15, ymin: 0.21, xmax: 0.75, ymax: 0.74 },
    },
    {
      name: "car",
      classIndex: 2,
      color: "#15803d",
      box: { xmin: 0.60, ymin: 0.12, xmax: 0.91, ymax: 0.31 },
    },
  ];

  const BOX_LABELS = ["x", "y", "w", "h", "conf"];
  const CLS_LABELS = CLASSES;

  const CLASS_COLORS = ["#c2410c", "#1d4ed8", "#15803d"];

  const els = {
    stage: document.getElementById("stage"),
    img: document.getElementById("sample-img"),
    overlay: document.getElementById("overlay"),
    grid: document.getElementById("grid"),
    drawRect: document.getElementById("draw-rect"),
    cellBadge: document.getElementById("cell-badge"),
    tensorNote: document.getElementById("tensor-note"),
    neuronsB1: document.getElementById("neurons-b1"),
    neuronsB2: document.getElementById("neurons-b2"),
    neuronsCls: document.getElementById("neurons-cls"),
    neuronGroups: document.getElementById("neuron-groups"),
    modeHint: document.getElementById("mode-hint"),
    btnPred: document.getElementById("btn-pred"),
    btnGt: document.getElementById("btn-gt"),
    btnAuto: document.getElementById("btn-auto"),
    btnClear: document.getElementById("btn-clear"),
    editTools: document.getElementById("edit-tools"),
    boxPicker: document.getElementById("box-picker"),
    classPicker: document.getElementById("class-picker"),
    tensorMap: document.getElementById("tensor-map"),
    channelBadge: document.getElementById("channel-badge"),
    lossTotal: document.getElementById("loss-total"),
    lossDetail: document.getElementById("loss-detail"),
    lossDetailTitle: document.getElementById("loss-detail-title"),
    lossDetailFormula: document.getElementById("loss-detail-formula"),
    lossDetailCalc: document.getElementById("loss-detail-calc"),
  };

  const ctx = els.overlay.getContext("2d");

  const state = {
    mode: "prediction", // prediction | gt
    editTool: "box", // box | class
    activeBox: 0,
    hoverCell: null, // {row, col}
    selectedCell: null,
    selectedBox: null, // {row, col, boxIndex}
    drawing: null, // {startX, startY, cell}
    predictions: createEmptyGrid(),
    groundTruth: buildGroundTruth(),
    activeLoss: null,
    lossParts: null,
  };

  function createEmptyCell() {
    return {
      boxes: [
        { x: 0.5, y: 0.5, w: 0, h: 0, conf: 0 },
        { x: 0.5, y: 0.5, w: 0, h: 0, conf: 0 },
      ],
      classes: [0, 0, 0],
    };
  }

  function createEmptyGrid() {
    return Array.from({ length: S }, () =>
      Array.from({ length: S }, () => createEmptyCell())
    );
  }

  function cloneCell(cell) {
    return {
      boxes: cell.boxes.map((b) => ({ ...b })),
      classes: [...cell.classes],
    };
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function round4(v) {
    return Math.round(v * 10000) / 10000;
  }

  function boxCenter(box) {
    return {
      cx: (box.xmin + box.xmax) / 2,
      cy: (box.ymin + box.ymax) / 2,
    };
  }

  function boxToYolo(box, row, col) {
    const { cx, cy } = boxCenter(box);
    const w = box.xmax - box.xmin;
    const h = box.ymax - box.ymin;
    return {
      x: clamp(cx * S - col, 0, 1),
      y: clamp(cy * S - row, 0, 1),
      w: clamp(w, 0, 1),
      h: clamp(h, 0, 1),
      conf: 1,
    };
  }

  function yoloToAbs(box, row, col) {
    const cx = (col + box.x) / S;
    const cy = (row + box.y) / S;
    const halfW = box.w / 2;
    const halfH = box.h / 2;
    return {
      xmin: clamp(cx - halfW, 0, 1),
      ymin: clamp(cy - halfH, 0, 1),
      xmax: clamp(cx + halfW, 0, 1),
      ymax: clamp(cy + halfH, 0, 1),
    };
  }

  function cellOfPoint(nx, ny) {
    return {
      col: clamp(Math.floor(nx * S), 0, S - 1),
      row: clamp(Math.floor(ny * S), 0, S - 1),
    };
  }

  function buildGroundTruth() {
    const grid = createEmptyGrid();
    for (const obj of GT_OBJECTS) {
      const { cx, cy } = boxCenter(obj.box);
      const { row, col } = cellOfPoint(cx, cy);
      const cell = grid[row][col];
      cell.boxes[0] = boxToYolo(obj.box, row, col);
      cell.boxes[1] = { x: 0.5, y: 0.5, w: 0, h: 0, conf: 0 };
      cell.classes = [0, 0, 0];
      cell.classes[obj.classIndex] = 1;
      cell._gtMeta = { name: obj.name, color: obj.color, classIndex: obj.classIndex };
    }
    return grid;
  }

  function currentGrid() {
    return state.mode === "gt" ? state.groundTruth : state.predictions;
  }

  function buildGridCells() {
    els.grid.innerHTML = "";
    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.row = String(row);
        cell.dataset.col = String(col);
        cell.title = `Cell (row ${row}, col ${col})`;
        els.grid.appendChild(cell);
      }
    }
  }

  function makeNeuron(label) {
    const el = document.createElement("div");
    el.className = "neuron";
    el.innerHTML = `<span class="label">${label}</span><span class="value">—</span>`;
    return el;
  }

  function buildNeuronUI() {
    els.neuronsB1.innerHTML = "";
    els.neuronsB2.innerHTML = "";
    els.neuronsCls.innerHTML = "";
    BOX_LABELS.forEach((l) => els.neuronsB1.appendChild(makeNeuron(l)));
    BOX_LABELS.forEach((l) => els.neuronsB2.appendChild(makeNeuron(l)));
    CLS_LABELS.forEach((l) => els.neuronsCls.appendChild(makeNeuron(l)));
  }

  function buildTensorMap() {
    els.tensorMap.innerHTML = "";
    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "tensor-cell";
        btn.dataset.row = String(row);
        btn.dataset.col = String(col);
        btn.title = `Tensor cell [${row}, ${col}] · 13 channels`;
        btn.innerHTML = `<span class="dot"></span>`;
        btn.addEventListener("click", () => {
          selectBox(row, col, state.activeBox);
        });
        btn.addEventListener("pointerenter", () => {
          state.hoverCell = { row, col };
          syncCellClasses();
          updateNeuronPanel();
          drawOverlay();
        });
        els.tensorMap.appendChild(btn);
      }
    }
  }

  function setNeuronValues(container, values, winnerIndex = -1) {
    const nodes = container.querySelectorAll(".neuron");
    nodes.forEach((node, i) => {
      const v = values[i];
      node.querySelector(".value").textContent =
        typeof v === "number" ? round4(v).toFixed(3) : "—";
      node.classList.toggle("is-winner", i === winnerIndex);
    });
  }

  function argmax(arr) {
    let best = 0;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > arr[best]) best = i;
    }
    return best;
  }

  function syncClassPicker() {
    const chips = els.classPicker.querySelectorAll(".chip-cls");
    const focus =
      state.selectedBox ||
      (state.selectedCell
        ? { row: state.selectedCell.row, col: state.selectedCell.col, boxIndex: state.activeBox }
        : null);
    let activeClass = -1;
    if (focus) {
      const cell = currentGrid()[focus.row][focus.col];
      if (cell.classes.some((v) => v > 0)) activeClass = argmax(cell.classes);
    }
    chips.forEach((chip) => {
      const idx = Number(chip.dataset.class);
      chip.classList.toggle("is-active", idx === activeClass);
      chip.disabled = state.mode === "gt" || !focus;
    });
  }

  function syncBoxGroupFocus() {
    els.neuronGroups.querySelectorAll(".neuron-group").forEach((g) => {
      g.classList.remove("is-box-focus");
    });
    els.neuronsB1.querySelectorAll(".neuron").forEach((n) => n.classList.remove("is-selected-box"));
    els.neuronsB2.querySelectorAll(".neuron").forEach((n) => n.classList.remove("is-selected-box"));

    const sel = state.selectedBox;
    if (!sel) return;
    const group = els.neuronGroups.querySelector(
      `.neuron-group[data-group="${sel.boxIndex === 0 ? "b1" : "b2"}"]`
    );
    if (group) group.classList.add("is-box-focus");
    const boxNeurons = sel.boxIndex === 0 ? els.neuronsB1 : els.neuronsB2;
    boxNeurons.querySelectorAll(".neuron").forEach((n) => n.classList.add("is-selected-box"));
  }

  function updateTensorMap() {
    const grid = currentGrid();
    const focus = state.hoverCell || state.selectedCell || (state.selectedBox
      ? { row: state.selectedBox.row, col: state.selectedBox.col }
      : null);

    els.tensorMap.querySelectorAll(".tensor-cell").forEach((el) => {
      const row = Number(el.dataset.row);
      const col = Number(el.dataset.col);
      const cell = grid[row][col];
      const maxConf = Math.max(cell.boxes[0].conf, cell.boxes[1].conf);
      const hasClass = cell.classes.some((v) => v > 0);
      const signal = Math.max(maxConf, hasClass ? Math.max(...cell.classes) : 0);

      el.classList.toggle("has-signal", signal > 0.05);
      el.classList.toggle(
        "is-hover",
        !!(focus && focus.row === row && focus.col === col)
      );
      el.classList.toggle(
        "is-selected",
        !!(state.selectedCell && state.selectedCell.row === row && state.selectedCell.col === col) ||
          !!(state.selectedBox && state.selectedBox.row === row && state.selectedBox.col === col)
      );

      const intensity = clamp(signal, 0, 1);
      el.style.background =
        signal > 0.05
          ? `rgba(45, 212, 191, ${0.15 + intensity * 0.7})`
          : "rgba(255, 255, 255, 0.08)";

      const dot = el.querySelector(".dot");
      if (hasClass) {
        dot.style.background = CLASS_COLORS[argmax(cell.classes)];
      } else {
        dot.style.background = "transparent";
      }
    });
  }

  function updateNeuronPanel() {
    const focus =
      state.hoverCell ||
      state.selectedCell ||
      (state.selectedBox
        ? { row: state.selectedBox.row, col: state.selectedBox.col }
        : null);
    const groups = els.neuronGroups.querySelectorAll(".neuron-group");

    els.cellBadge.textContent = `7 × 7 × ${CELL_SIZE}`;
    updateTensorMap();

    if (!focus) {
      setNeuronValues(els.neuronsB1, [null, null, null, null, null]);
      setNeuronValues(els.neuronsB2, [null, null, null, null, null]);
      setNeuronValues(els.neuronsCls, [null, null, null]);
      groups.forEach((g) => g.classList.remove("is-lit"));
      els.channelBadge.textContent = "—";
      els.tensorNote.textContent =
        state.mode === "gt"
          ? "Ground-truth tensor 7×7×13. Click a spatial cell to inspect its 13 target channels."
          : "Prediction tensor 7×7×13 (637 values). Click a spatial cell to inspect its 13 channels.";
      syncBoxGroupFocus();
      syncClassPicker();
      return;
    }

    const { row, col } = focus;
    const cell = currentGrid()[row][col];
    setNeuronValues(els.neuronsB1, [
      cell.boxes[0].x,
      cell.boxes[0].y,
      cell.boxes[0].w,
      cell.boxes[0].h,
      cell.boxes[0].conf,
    ]);
    setNeuronValues(els.neuronsB2, [
      cell.boxes[1].x,
      cell.boxes[1].y,
      cell.boxes[1].w,
      cell.boxes[1].h,
      cell.boxes[1].conf,
    ]);
    const clsWinner = cell.classes.every((p) => p === 0) ? -1 : argmax(cell.classes);
    setNeuronValues(els.neuronsCls, cell.classes, clsWinner);

    groups.forEach((g) => g.classList.add("is-lit"));
    els.channelBadge.textContent = `[${row}, ${col}, :] · 13`;

    const gtMeta = state.groundTruth[row][col]._gtMeta;
    if (state.mode === "gt" && gtMeta) {
      els.tensorNote.textContent = `GT slice [${row}, ${col}, :]: B1 encodes ${gtMeta.name}; class one-hot = ${gtMeta.name}.`;
    } else if (state.mode === "gt") {
      els.tensorNote.textContent = `GT slice [${row}, ${col}, :]: background — all 13 channels are 0.`;
    } else if (state.editTool === "class") {
      els.tensorNote.textContent = `Pred slice [${row}, ${col}, :]. Select B1/B2 in the tensor groups, then set class.`;
    } else {
      els.tensorNote.textContent = `Pred slice [${row}, ${col}, :] of the 7×7×13 tensor. Drawing/editing B${state.activeBox + 1}.`;
    }

    syncBoxGroupFocus();
    syncClassPicker();
  }

  function syncCellClasses() {
    const cells = els.grid.querySelectorAll(".cell");
    cells.forEach((el) => {
      const row = Number(el.dataset.row);
      const col = Number(el.dataset.col);
      el.classList.toggle(
        "is-hover",
        !!(state.hoverCell && state.hoverCell.row === row && state.hoverCell.col === col)
      );
      el.classList.toggle(
        "is-selected",
        !!(state.selectedCell && state.selectedCell.row === row && state.selectedCell.col === col)
      );
    });
  }

  function stageSize() {
    const rect = els.stage.getBoundingClientRect();
    return { width: rect.width, height: rect.height, left: rect.left, top: rect.top };
  }

  function pointerToNorm(clientX, clientY) {
    const { width, height, left, top } = stageSize();
    return {
      nx: clamp((clientX - left) / width, 0, 1),
      ny: clamp((clientY - top) / height, 0, 1),
    };
  }

  function resizeCanvas() {
    const { width, height } = stageSize();
    const dpr = window.devicePixelRatio || 1;
    els.overlay.width = Math.round(width * dpr);
    els.overlay.height = Math.round(height * dpr);
    els.overlay.style.width = `${width}px`;
    els.overlay.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawOverlay();
  }

  function drawBox(abs, color, label, dashed) {
    const { width, height } = stageSize();
    const x = abs.xmin * width;
    const y = abs.ymin * height;
    const w = (abs.xmax - abs.xmin) * width;
    const h = (abs.ymax - abs.ymin) * height;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.strokeRect(x, y, w, h);
    if (label) {
      ctx.setLineDash([]);
      ctx.font = "600 12px IBM Plex Sans, sans-serif";
      const padX = 6;
      const padY = 3;
      const tw = ctx.measureText(label).width;
      const bx = x;
      const by = Math.max(0, y - 18);
      ctx.fillStyle = color;
      ctx.fillRect(bx, by, tw + padX * 2, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, bx + padX, by + 13);
    }
    ctx.restore();
  }

  function pointInAbsBox(nx, ny, abs) {
    return nx >= abs.xmin && nx <= abs.xmax && ny >= abs.ymin && ny <= abs.ymax;
  }

  function hitTestBox(nx, ny) {
    let best = null;
    let bestArea = Infinity;
    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const cell = state.predictions[row][col];
        cell.boxes.forEach((box, bi) => {
          if (box.w <= 0 || box.h <= 0) return;
          const abs = yoloToAbs(box, row, col);
          if (!pointInAbsBox(nx, ny, abs)) return;
          const area = Math.max(1e-6, (abs.xmax - abs.xmin) * (abs.ymax - abs.ymin));
          if (area < bestArea) {
            bestArea = area;
            best = { row, col, boxIndex: bi };
          }
        });
      }
    }
    return best;
  }

  function selectBox(row, col, boxIndex) {
    state.selectedCell = { row, col };
    state.selectedBox = { row, col, boxIndex };
    state.activeBox = boxIndex;
    els.boxPicker.querySelectorAll("[data-box]").forEach((c) => {
      c.classList.toggle("is-active", Number(c.dataset.box) === boxIndex);
    });
    syncCellClasses();
    updateNeuronPanel();
    drawOverlay();
  }

  function assignClass(classIndex) {
    if (state.mode === "gt") return;
    const focus =
      state.selectedBox ||
      (state.selectedCell
        ? { row: state.selectedCell.row, col: state.selectedCell.col, boxIndex: state.activeBox }
        : null);
    if (!focus) return;

    const { row, col, boxIndex } = focus;
    if (!state.selectedBox) selectBox(row, col, boxIndex);

    const cell = state.predictions[row][col];
    cell.classes = [0, 0, 0];
    cell.classes[classIndex] = 1;
    if (cell.boxes[boxIndex].conf < 0.35) cell.boxes[boxIndex].conf = 0.75;
    if (cell.boxes[boxIndex].w <= 0 || cell.boxes[boxIndex].h <= 0) {
      cell.boxes[boxIndex] = {
        x: 0.5,
        y: 0.5,
        w: 0.2,
        h: 0.2,
        conf: 0.75,
      };
    }
    updateNeuronPanel();
    drawOverlay();
    updateLossUI();
  }

  function classLabelForCell(cell) {
    if (!cell.classes.some((v) => v > 0)) return null;
    return CLASSES[argmax(cell.classes)];
  }

  function drawOverlay() {
    const { width, height } = stageSize();
    ctx.clearRect(0, 0, width, height);

    if (state.mode === "gt") {
      for (const obj of GT_OBJECTS) {
        drawBox(obj.box, obj.color, obj.name, false);
      }
      return;
    }

    const focus = state.hoverCell || state.selectedCell;
    const sel = state.selectedBox;

    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const cell = state.predictions[row][col];
        const isFocus = focus && focus.row === row && focus.col === col;
        const clsName = classLabelForCell(cell);
        cell.boxes.forEach((box, bi) => {
          if (box.w <= 0 || box.h <= 0 || box.conf <= 0) return;
          const isSelected =
            sel && sel.row === row && sel.col === col && sel.boxIndex === bi;
          if (!isFocus && !isSelected && box.conf < 0.15) return;
          const abs = yoloToAbs(box, row, col);
          let color = bi === 0 ? "#0f766e" : "#a16207";
          if (clsName) {
            color = CLASS_COLORS[argmax(cell.classes)];
          }
          if (isSelected) color = "#b45309";
          const score = box.conf * (clsName ? cell.classes[argmax(cell.classes)] : 0);
          let label = `B${bi + 1}`;
          if (isFocus || isSelected) {
            label = clsName
              ? `B${bi + 1} ${clsName} ${score.toFixed(2)}`
              : `B${bi + 1} conf ${box.conf.toFixed(2)}`;
          }
          drawBox(abs, color, isFocus || isSelected ? label : null, bi === 1 && !isSelected);

          if (isSelected) {
            ctx.save();
            ctx.strokeStyle = "#fbbf24";
            ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            const { width: W, height: H } = stageSize();
            ctx.strokeRect(
              abs.xmin * W - 3,
              abs.ymin * H - 3,
              (abs.xmax - abs.xmin) * W + 6,
              (abs.ymax - abs.ymin) * H + 6
            );
            ctx.restore();
          }
        });
      }
    }
  }

  function randomPredictions() {
    const grid = createEmptyGrid();
    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const cell = grid[row][col];
        for (let bi = 0; bi < B; bi++) {
          const w = 0.08 + Math.random() * 0.35;
          const h = 0.08 + Math.random() * 0.35;
          cell.boxes[bi] = {
            x: 0.15 + Math.random() * 0.7,
            y: 0.15 + Math.random() * 0.7,
            w,
            h,
            conf: 0.2 + Math.random() * 0.8,
          };
        }
        // Random class distribution (softmax-ish)
        const logits = CLASSES.map(() => Math.random());
        const sum = logits.reduce((a, b) => a + b, 0);
        cell.classes = logits.map((v) => v / sum);

        // Nudge cells that own GT objects toward sensible values for demo clarity
        const gt = state.groundTruth[row][col];
        if (gt._gtMeta) {
          const target = gt.boxes[0];
          cell.boxes[0] = {
            x: clamp(target.x + (Math.random() - 0.5) * 0.15, 0, 1),
            y: clamp(target.y + (Math.random() - 0.5) * 0.15, 0, 1),
            w: clamp(target.w * (0.85 + Math.random() * 0.3), 0.05, 1),
            h: clamp(target.h * (0.85 + Math.random() * 0.3), 0.05, 1),
            conf: 0.55 + Math.random() * 0.4,
          };
          cell.classes = [0.1, 0.1, 0.1];
          cell.classes[gt._gtMeta.classIndex] = 0.7;
          const rem = 0.3;
          let others = CLASSES.map((_, i) => (i === gt._gtMeta.classIndex ? 0 : Math.random()));
          const oSum = others.reduce((a, b) => a + b, 0) || 1;
          others = others.map((v) => (v / oSum) * rem);
          cell.classes = cell.classes.map((v, i) =>
            i === gt._gtMeta.classIndex ? 0.7 : others[i]
          );
        }
      }
    }
    state.predictions = grid;
  }

  function clearPredictions() {
    state.predictions = createEmptyGrid();
  }

  function iou(a, b) {
    const x1 = Math.max(a.xmin, b.xmin);
    const y1 = Math.max(a.ymin, b.ymin);
    const x2 = Math.min(a.xmax, b.xmax);
    const y2 = Math.min(a.ymax, b.ymax);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = Math.max(0, a.xmax - a.xmin) * Math.max(0, a.ymax - a.ymin);
    const areaB = Math.max(0, b.xmax - b.xmin) * Math.max(0, b.ymax - b.ymin);
    const uni = areaA + areaB - inter;
    return uni <= 0 ? 0 : inter / uni;
  }

  function responsibleBoxIndex(predCell, gtBoxAbs, row, col) {
    let best = 0;
    let bestIou = -1;
    for (let bi = 0; bi < B; bi++) {
      const abs = yoloToAbs(predCell.boxes[bi], row, col);
      const v = iou(abs, gtBoxAbs);
      if (v > bestIou) {
        bestIou = v;
        best = bi;
      }
    }
    // If both empty, prefer B1
    if (bestIou <= 0) {
      const b0 = predCell.boxes[0];
      const b1 = predCell.boxes[1];
      if (b0.w * b0.h === 0 && b1.w * b1.h > 0) return 1;
      return 0;
    }
    return best;
  }

  function computeLoss() {
    const details = {
      xy: { value: 0, rows: [] },
      wh: { value: 0, rows: [] },
      obj: { value: 0, rows: [] },
      noobj: { value: 0, rows: [] },
      cls: { value: 0, rows: [] },
    };

    const objectCells = new Map();
    for (const obj of GT_OBJECTS) {
      const { cx, cy } = boxCenter(obj.box);
      const { row, col } = cellOfPoint(cx, cy);
      objectCells.set(`${row},${col}`, obj);
    }

    for (let row = 0; row < S; row++) {
      for (let col = 0; col < S; col++) {
        const pred = state.predictions[row][col];
        const key = `${row},${col}`;
        const obj = objectCells.get(key);

        if (obj) {
          const gtYolo = boxToYolo(obj.box, row, col);
          const resp = responsibleBoxIndex(pred, obj.box, row, col);

          for (let bi = 0; bi < B; bi++) {
            const p = pred.boxes[bi];
            if (bi === resp) {
              const dx = p.x - gtYolo.x;
              const dy = p.y - gtYolo.y;
              const termXY = LAMBDA_COORD * (dx * dx + dy * dy);
              details.xy.value += termXY;
              details.xy.rows.push({
                cell: `[${row},${col}]`,
                box: `B${bi + 1}`,
                text: `λ((${round4(p.x)}−${round4(gtYolo.x)})² + (${round4(p.y)}−${round4(gtYolo.y)})²) = ${round4(termXY)}`,
              });

              const sw = Math.sqrt(Math.max(p.w, 1e-6)) - Math.sqrt(Math.max(gtYolo.w, 1e-6));
              const sh = Math.sqrt(Math.max(p.h, 1e-6)) - Math.sqrt(Math.max(gtYolo.h, 1e-6));
              const termWH = LAMBDA_COORD * (sw * sw + sh * sh);
              details.wh.value += termWH;
              details.wh.rows.push({
                cell: `[${row},${col}]`,
                box: `B${bi + 1}`,
                text: `λ((√${round4(p.w)}−√${round4(gtYolo.w)})² + (√${round4(p.h)}−√${round4(gtYolo.h)})²) = ${round4(termWH)}`,
              });

              const targetConf = iou(yoloToAbs(p, row, col), obj.box);
              const dObj = p.conf - targetConf;
              const termObj = dObj * dObj;
              details.obj.value += termObj;
              details.obj.rows.push({
                cell: `[${row},${col}]`,
                box: `B${bi + 1}`,
                text: `(${round4(p.conf)} − IoU=${round4(targetConf)})² = ${round4(termObj)}`,
              });
            } else {
              const termNo = LAMBDA_NOOBJ * (p.conf - 0) ** 2;
              details.noobj.value += termNo;
              details.noobj.rows.push({
                cell: `[${row},${col}]`,
                box: `B${bi + 1}`,
                text: `λ_noobj · (${round4(p.conf)} − 0)² = ${round4(termNo)}`,
              });
            }
          }

          let clsTerm = 0;
          const bits = [];
          for (let c = 0; c < C; c++) {
            const target = c === obj.classIndex ? 1 : 0;
            const d = pred.classes[c] - target;
            clsTerm += d * d;
            bits.push(`(${round4(pred.classes[c])}−${target})²`);
          }
          details.cls.value += clsTerm;
          details.cls.rows.push({
            cell: `[${row},${col}]`,
            box: obj.name,
            text: `${bits.join(" + ")} = ${round4(clsTerm)}`,
          });
        } else {
          for (let bi = 0; bi < B; bi++) {
            const p = pred.boxes[bi];
            const termNo = LAMBDA_NOOBJ * (p.conf - 0) ** 2;
            if (termNo === 0) continue;
            details.noobj.value += termNo;
            details.noobj.rows.push({
              cell: `[${row},${col}]`,
              box: `B${bi + 1}`,
              text: `λ_noobj · (${round4(p.conf)} − 0)² = ${round4(termNo)}`,
            });
          }
        }
      }
    }

    // Keep noobj table readable: show top contributors + count
    if (details.noobj.rows.length > 12) {
      details.noobj.rows.sort((a, b) => {
        const av = Number(a.text.split("=").pop());
        const bv = Number(b.text.split("=").pop());
        return bv - av;
      });
      const rest = details.noobj.rows.length - 10;
      details.noobj.rows = details.noobj.rows.slice(0, 10);
      details.noobj.rows.push({
        cell: "…",
        box: "",
        text: `+ ${rest} more background terms`,
      });
    }

    const total =
      details.xy.value +
      details.wh.value +
      details.obj.value +
      details.noobj.value +
      details.cls.value;

    state.lossParts = { details, total };
    return state.lossParts;
  }

  function updateLossUI() {
    const { details, total } = computeLoss();
    document.getElementById("loss-xy-val").textContent = round4(details.xy.value).toFixed(3);
    document.getElementById("loss-wh-val").textContent = round4(details.wh.value).toFixed(3);
    document.getElementById("loss-obj-val").textContent = round4(details.obj.value).toFixed(3);
    document.getElementById("loss-noobj-val").textContent = round4(details.noobj.value).toFixed(3);
    document.getElementById("loss-cls-val").textContent = round4(details.cls.value).toFixed(3);
    els.lossTotal.textContent = `Total L = ${round4(total).toFixed(3)}`;
    if (state.activeLoss) renderLossDetail(state.activeLoss);
  }

  const LOSS_META = {
    xy: {
      title: "1 · Coordinate loss (x, y)",
      formula:
        "λ_coord Σ 1_ij^obj [ (x_i − x̂_i)² + (y_i − ŷ_i)² ]\n\nOnly the responsible predictor (highest IoU with GT) in object cells contributes.\nλ_coord = 5",
    },
    wh: {
      title: "2 · Size loss (√w, √h)",
      formula:
        "λ_coord Σ 1_ij^obj [ (√w_i − √ŵ_i)² + (√h_i − √ĥ_i)² ]\n\nSquare roots reduce the penalty on large boxes relative to small ones.\nλ_coord = 5",
    },
    obj: {
      title: "3 · Object confidence loss",
      formula:
        "Σ 1_ij^obj (C_i − Ĉ_i)²\n\nĈ_i is the IoU between the predicted box and the ground-truth box.\nApplied to the responsible predictor in cells that contain an object.",
    },
    noobj: {
      title: "4 · No-object confidence loss",
      formula:
        "λ_noobj Σ 1_ij^noobj (C_i − Ĉ_i)²\n\nPushes confidence toward 0 for predictors that are not responsible\n(and for all predictors in empty cells). λ_noobj = 0.5",
    },
    cls: {
      title: "5 · Classification loss",
      formula:
        "Σ 1_i^obj Σ_c (p_i(c) − p̂_i(c))²\n\nOnly cells that contain an object center. Class targets are one-hot\n(dog / bicycle / car).",
    },
  };

  function renderLossDetail(key) {
    const meta = LOSS_META[key];
    const part = state.lossParts.details[key];
    els.lossDetail.hidden = false;
    els.lossDetailTitle.textContent = meta.title;
    els.lossDetailFormula.textContent = meta.formula;

    if (!part.rows.length) {
      els.lossDetailCalc.innerHTML =
        `<div class="sum">Contribution = ${round4(part.value).toFixed(4)} · no active terms yet. Auto-fill or draw boxes to see terms.</div>`;
      return;
    }

    const rows = part.rows
      .map(
        (r) =>
          `<tr><td>${r.cell}</td><td>${r.box}</td><td>${r.text}</td></tr>`
      )
      .join("");

    els.lossDetailCalc.innerHTML = `
      <table>
        <thead><tr><th>Cell</th><th>Box / class</th><th>Term</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="sum">Component sum = ${round4(part.value).toFixed(4)}</div>
    `;
  }

  function syncToolUI() {
    els.editTools.querySelectorAll("[data-tool]").forEach((chip) => {
      chip.classList.toggle("is-active", chip.dataset.tool === state.editTool);
    });
    els.stage.classList.toggle("is-box-tool", state.editTool === "box" && state.mode !== "gt");
    els.stage.classList.toggle("is-class-tool", state.editTool === "class" && state.mode !== "gt");
    els.editTools.style.opacity = state.mode === "gt" ? "0.45" : "1";
    els.editTools.style.pointerEvents = state.mode === "gt" ? "none" : "auto";
  }

  function setEditTool(tool) {
    state.editTool = tool;
    syncToolUI();
    if (state.mode === "prediction") {
      if (tool === "box") {
        els.modeHint.innerHTML =
          "Draw box: choose <strong>B1</strong> or <strong>B2</strong>, then click-drag on the image.";
      } else {
        els.modeHint.innerHTML =
          "Set class: click a box to select it, then choose <strong>Dog / Bicycle / Car</strong>. Classifier outputs update on the right.";
      }
    }
    updateNeuronPanel();
    drawOverlay();
  }

  function setMode(mode) {
    state.mode = mode;
    els.btnPred.classList.toggle("is-active", mode === "prediction");
    els.btnGt.classList.toggle("is-active", mode === "gt");
    els.stage.classList.toggle("is-gt", mode === "gt");
    els.btnAuto.disabled = mode === "gt";
    els.btnClear.disabled = mode === "gt";
    syncToolUI();

    if (mode === "gt") {
      state.selectedBox = null;
      els.modeHint.textContent =
        "Ground-truth view: true boxes for dog, bicycle, and car. Hover cells to inspect target neurons.";
    } else {
      setEditTool(state.editTool);
      return;
    }
    updateNeuronPanel();
    drawOverlay();
  }

  function applyDrawnBox(row, col, absBox) {
    const cell = state.predictions[row][col];
    const yolo = boxToYolo(absBox, row, col);
    yolo.conf = Math.max(0.6, cell.boxes[state.activeBox].conf || 0.6);
    cell.boxes[state.activeBox] = yolo;

    const gt = state.groundTruth[row][col];
    if (cell.classes.every((v) => v === 0)) {
      if (gt._gtMeta) {
        cell.classes = [0, 0, 0];
        cell.classes[gt._gtMeta.classIndex] = 1;
      } else {
        cell.classes = [1 / C, 1 / C, 1 / C];
      }
    }

    selectBox(row, col, state.activeBox);
    updateLossUI();
  }

  function onPointerDown(e) {
    if (state.mode === "gt") return;
    if (e.button !== undefined && e.button !== 0) return;
    const { nx, ny } = pointerToNorm(e.clientX, e.clientY);
    const cell = cellOfPoint(nx, ny);

    if (state.editTool === "class") {
      const hit = hitTestBox(nx, ny);
      if (hit) {
        selectBox(hit.row, hit.col, hit.boxIndex);
      } else {
        // Fall back: select active box slot in this cell
        selectBox(cell.row, cell.col, state.activeBox);
      }
      e.preventDefault();
      return;
    }

    state.selectedCell = cell;
    state.drawing = { startX: nx, startY: ny, cell };
    els.drawRect.hidden = false;
    updateDrawRect(nx, ny, nx, ny);
    syncCellClasses();
    updateNeuronPanel();
    e.preventDefault();
  }

  function updateDrawRect(x0, y0, x1, y1) {
    const { width, height } = stageSize();
    const left = Math.min(x0, x1) * width;
    const top = Math.min(y0, y1) * height;
    const w = Math.abs(x1 - x0) * width;
    const h = Math.abs(y1 - y0) * height;
    els.drawRect.style.left = `${left}px`;
    els.drawRect.style.top = `${top}px`;
    els.drawRect.style.width = `${w}px`;
    els.drawRect.style.height = `${h}px`;
  }

  function onPointerMove(e) {
    const { nx, ny } = pointerToNorm(e.clientX, e.clientY);
    const cell = cellOfPoint(nx, ny);
    const prev = state.hoverCell;
    if (!prev || prev.row !== cell.row || prev.col !== cell.col) {
      state.hoverCell = cell;
      syncCellClasses();
      updateNeuronPanel();
      drawOverlay();
    }
    if (state.drawing) {
      updateDrawRect(state.drawing.startX, state.drawing.startY, nx, ny);
    }
  }

  function onPointerUp(e) {
    if (!state.drawing || state.mode === "gt" || state.editTool !== "box") return;
    const { nx, ny } = pointerToNorm(e.clientX, e.clientY);
    const x0 = state.drawing.startX;
    const y0 = state.drawing.startY;
    const xmin = Math.min(x0, nx);
    const xmax = Math.max(x0, nx);
    const ymin = Math.min(y0, ny);
    const ymax = Math.max(y0, ny);
    els.drawRect.hidden = true;
    state.drawing = null;

    // Tiny click: select that cell's active box instead of drawing
    if (xmax - xmin < 0.015 || ymax - ymin < 0.015) {
      const cell = cellOfPoint(x0, y0);
      selectBox(cell.row, cell.col, state.activeBox);
      return;
    }

    const abs = { xmin, ymin, xmax, ymax };
    const { cx, cy } = boxCenter(abs);
    const owner = cellOfPoint(cx, cy);
    applyDrawnBox(owner.row, owner.col, abs);
  }

  function onPointerLeave() {
    state.hoverCell = null;
    syncCellClasses();
    updateNeuronPanel();
    drawOverlay();
    if (state.drawing) {
      els.drawRect.hidden = true;
      state.drawing = null;
    }
  }

  function bindUI() {
    els.btnPred.addEventListener("click", () => setMode("prediction"));
    els.btnGt.addEventListener("click", () => setMode("gt"));
    els.btnAuto.addEventListener("click", () => {
      if (state.mode === "gt") return;
      randomPredictions();
      state.selectedBox = null;
      updateNeuronPanel();
      drawOverlay();
      updateLossUI();
    });
    els.btnClear.addEventListener("click", () => {
      if (state.mode === "gt") return;
      clearPredictions();
      state.selectedBox = null;
      updateNeuronPanel();
      drawOverlay();
      updateLossUI();
    });

    els.editTools.querySelectorAll("[data-tool]").forEach((chip) => {
      chip.addEventListener("click", () => setEditTool(chip.dataset.tool));
    });

    els.boxPicker.querySelectorAll("[data-box]").forEach((chip) => {
      chip.addEventListener("click", () => {
        state.activeBox = Number(chip.dataset.box);
        els.boxPicker.querySelectorAll("[data-box]").forEach((c) => c.classList.remove("is-active"));
        chip.classList.add("is-active");
        const focus = state.selectedCell || state.hoverCell;
        if (focus) {
          selectBox(focus.row, focus.col, state.activeBox);
        } else {
          updateNeuronPanel();
          drawOverlay();
        }
      });
    });

    els.classPicker.querySelectorAll(".chip-cls").forEach((chip) => {
      chip.addEventListener("click", () => {
        if (chip.disabled) return;
        assignClass(Number(chip.dataset.class));
      });
    });

    // Select B1 / B2 by clicking their channel groups in the tensor panel
    els.neuronGroups.querySelectorAll(".neuron-group[data-group='b1'], .neuron-group[data-group='b2']").forEach((group) => {
      group.style.cursor = "pointer";
      group.addEventListener("click", () => {
        const focus = state.selectedCell || state.hoverCell || (state.selectedBox
          ? { row: state.selectedBox.row, col: state.selectedBox.col }
          : null);
        if (!focus || state.mode === "gt") return;
        const boxIndex = group.dataset.group === "b1" ? 0 : 1;
        selectBox(focus.row, focus.col, boxIndex);
      });
    });

    // Assign class by clicking class channel neurons in the output tensor
    els.neuronsCls.addEventListener("click", (e) => {
      const neuron = e.target.closest(".neuron");
      if (!neuron || state.mode === "gt") return;
      e.stopPropagation();
      const focus =
        state.selectedBox ||
        state.selectedCell ||
        state.hoverCell;
      if (!focus) return;
      if (!state.selectedBox) {
        selectBox(focus.row, focus.col, state.activeBox);
      }
      const idx = [...els.neuronsCls.children].indexOf(neuron);
      if (idx >= 0) assignClass(idx);
    });

    document.querySelectorAll(".loss-card").forEach((card) => {
      card.addEventListener("click", () => {
        const key = card.dataset.loss;
        const was = state.activeLoss === key;
        document.querySelectorAll(".loss-card").forEach((c) => c.classList.remove("is-active"));
        if (was) {
          state.activeLoss = null;
          els.lossDetail.hidden = true;
          return;
        }
        state.activeLoss = key;
        card.classList.add("is-active");
        renderLossDetail(key);
      });
    });

    els.stage.addEventListener("pointerdown", onPointerDown);
    els.stage.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    els.stage.addEventListener("pointerleave", onPointerLeave);

    els.grid.addEventListener("pointerover", (e) => {
      const t = e.target.closest(".cell");
      if (!t) return;
      state.hoverCell = { row: Number(t.dataset.row), col: Number(t.dataset.col) };
      syncCellClasses();
      updateNeuronPanel();
      drawOverlay();
    });

    window.addEventListener("resize", resizeCanvas);
    els.img.addEventListener("load", resizeCanvas);
  }

  function init() {
    buildGridCells();
    buildNeuronUI();
    buildTensorMap();
    bindUI();
    setMode("prediction");
    updateLossUI();
    if (els.img.complete) resizeCanvas();
  }

  init();
})();
