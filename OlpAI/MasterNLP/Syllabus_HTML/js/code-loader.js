/* ==========================================================================
   Dynamic Code Loader & VS Code Dark+ Client-Side Syntax Highlighter
   - Loads standalone Python (.py) & tree (.txt) files into <pre data-code-src="...">
   - Primary: live fetch() directly from ./code/*.py (HTTP / Live Server / GitHub Pages)
   - Fallback: window.SYLLABUS_CODE_FILES registry when opened offline via file://
   ========================================================================== */

(function () {
  const KW_CTRL = new Set([
    'if', 'elif', 'else', 'for', 'while', 'return', 'yield',
    'import', 'from', 'as', 'with', 'try', 'except', 'finally',
    'raise', 'assert', 'pass', 'break', 'continue', 'await', 'async'
  ]);

  const KW_DECL = new Set([
    'def', 'class', 'lambda', 'and', 'or', 'not', 'in', 'is',
    'True', 'False', 'None', 'global', 'nonlocal', 'del'
  ]);

  const BUILTIN_TYPES = new Set([
    'int', 'float', 'str', 'bool', 'list', 'dict', 'tuple', 'set',
    'frozenset', 'bytes', 'bytearray', 'object', 'slice', 'complex',
    'Callable', 'Optional', 'Any', 'Union', 'Iterable', 'Sequence', 'Type',
    'Path', 'Tensor', 'Module', 'DataFrame', 'Series', 'DataLoader', 'Dataset', 'Sampler'
  ]);

  const BUILTIN_FNS = new Set([
    'len', 'range', 'enumerate', 'zip', 'print', 'sum', 'min', 'max',
    'abs', 'round', 'sorted', 'reversed', 'isinstance', 'issubclass',
    'getattr', 'setattr', 'hasattr', 'super', 'iter', 'next', 'map',
    'filter', 'any', 'all', 'open', 'type', 'id', 'repr', 'vars'
  ]);

  const DEFAULT_MODULES = new Set([
    'torch', 'nn', 'F', 'np', 'pd', 'cv2', 'scipy', 'sklearn', 'skimage',
    'models', 'v2', 'tv_tensors', 'transformers', 'gensim', 'xgboost',
    'catboost', 'datasets', 'evaluate', 'nltk', 'spacy', 'PIL', 'math',
    'os', 'sys', 're', 'io', 'time', 'copy', 'random', 'json', 'pickle',
    'unicodedata', 'ast', 'itertools', 'collections', 'joblib', 'tqdm'
  ]);

  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function wrapSpan(cls, text) {
    if (!text) return '';
    return text
      .split('\n')
      .map((part) => (part ? `<span class="${cls}">${escapeHtml(part)}</span>` : ''))
      .join('\n');
  }

  function highlightFString(rawTok) {
    let out = '';
    let i = 0;
    const n = rawTok.length;
    while (i < n) {
      if (rawTok[i] === '{' && i + 1 < n && rawTok[i + 1] !== '{') {
        const j = rawTok.indexOf('}', i + 1);
        if (j !== -1 && !rawTok.slice(i + 1, j).includes('\n')) {
          const expr = rawTok.slice(i + 1, j);
          out += wrapSpan('vs-br-2', '{');
          const colonIdx = expr.lastIndexOf(':');
          if (colonIdx !== -1 && !expr.trim().startsWith('lambda')) {
            out += wrapSpan('vs-var', expr.slice(0, colonIdx));
            out += wrapSpan('vs-fspec', expr.slice(colonIdx));
          } else {
            out += wrapSpan('vs-var', expr);
          }
          out += wrapSpan('vs-br-2', '}');
          i = j + 1;
          continue;
        }
      }
      const start = i;
      while (i < n) {
        if (rawTok[i] === '{') {
          if (i + 1 < n && rawTok[i + 1] === '{') {
            i += 2;
            continue;
          }
          if (i > start) break;
        }
        i += 1;
      }
      out += wrapSpan('vs-str', rawTok.slice(start, i));
    }
    return out;
  }

  function highlightTree(rawCode) {
    return rawCode
      .split('\n')
      .map((line) => {
        const hashIdx = line.indexOf('#');
        const mainPart = hashIdx !== -1 ? line.slice(0, hashIdx) : line;
        const cmtPart = hashIdx !== -1 ? line.slice(hashIdx) : '';
        const re = /([├└│─]+|[A-Za-z0-9_.-]+\/|[A-Za-z0-9_.-]+\.(?:py|txt))/g;
        let pos = 0;
        let buf = '';
        let m;
        while ((m = re.exec(mainPart)) !== null) {
          if (m.index > pos) {
            buf += escapeHtml(mainPart.slice(pos, m.index));
          }
          const tok = m[0];
          if ('├└│─'.includes(tok[0])) {
            buf += wrapSpan('vs-tree', tok);
          } else if (tok.endsWith('/')) {
            buf += wrapSpan('vs-dir', tok);
          } else {
            buf += wrapSpan('vs-var', tok);
          }
          pos = m.index + tok.length;
        }
        if (pos < mainPart.length) {
          buf += escapeHtml(mainPart.slice(pos));
        }
        if (cmtPart) {
          buf += wrapSpan('vs-cmt', cmtPart);
        }
        return `<span class="code-line">${buf}</span>`;
      })
      .join('\n');
  }

  function collectLightSemantics(rawCode) {
    const importedNames = new Set(DEFAULT_MODULES);
    const paramNames = new Set();

    // Collect imports
    const importRe = /^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import\s+([^\n#]+)|import\s+([^\n#]+))/gm;
    let m;
    while ((m = importRe.exec(rawCode)) !== null) {
      if (m[1]) {
        m[1].split('.').forEach((p) => importedNames.add(p.trim()));
        m[2].split(',').forEach((item) => {
          const clean = item.replace(/[()]/g, '').trim();
          const parts = clean.split(/\s+as\s+/);
          const bound = (parts[1] || parts[0]).trim();
          if (bound && (/^[A-Z]/.test(bound) || DEFAULT_MODULES.has(bound))) {
            importedNames.add(bound);
          }
        });
      } else if (m[3]) {
        m[3].split(',').forEach((item) => {
          const parts = item.trim().split(/\s+as\s+/);
          const bound = (parts[1] || parts[0].split('.')[0]).trim();
          if (bound) importedNames.add(bound);
        });
      }
    }

    // Collect def parameter names
    const defRe = /\bdef\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([\s\S]*?)\)\s*(?:->[^:]+)?:/g;
    while ((m = defRe.exec(rawCode)) !== null) {
      const argsBlock = m[1];
      argsBlock.split(',').forEach((seg) => {
        const lhs = seg.split('=')[0].split(':')[0].replace(/\*/g, '').trim();
        if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(lhs) && lhs !== 'self' && lhs !== 'cls') {
          paramNames.add(lhs);
        }
      });
    }

    return { importedNames, paramNames };
  }

  function isClassLike(name, importedNames) {
    if (BUILTIN_TYPES.has(name) || importedNames.has(name)) return true;
    return /^[A-Z]/.test(name) && /[a-z]/.test(name);
  }

  function isConstantName(name) {
    if (name === 'F' || name === 'A') return false;
    return /^[A-Z][A-Z0-9_]*$/.test(name);
  }

  function highlightPython(rawCode) {
    const { importedNames, paramNames } = collectLightSemantics(rawCode);

    // Master Python token regex (strings, comments, numbers, identifiers, operators/brackets, whitespace)
    const tokRe = /([fFrRbBuU]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(#[^\n]*)|(\b(?:0[xX][0-9a-fA-F_]+|0[bB][01_]+|0[oO][0-7_]+|(?:\d[\d_]*\.?[\d_]*|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?j?)\b)|([A-Za-z_][A-Za-z0-9_]*)|(->|:=|==|!=|<=|>=|\*\*|\/\/|<<|>>|[+\-*/%&|^~<>:=@.,;()[\]{}])/g;

    const tokens = [];
    let lastIdx = 0;
    let match;
    let lineNo = 1;

    const countNewlines = (s) => {
      let c = 0;
      for (let i = 0; i < s.length; i++) if (s[i] === '\n') c++;
      return c;
    };

    while ((match = tokRe.exec(rawCode)) !== null) {
      if (match.index > lastIdx) {
        const ws = rawCode.slice(lastIdx, match.index);
        tokens.push({ type: 'ws', val: ws, line: lineNo });
        lineNo += countNewlines(ws);
      }
      const val = match[0];
      let type = 'op';
      if (match[1] !== undefined) type = 'str';
      else if (match[2] !== undefined) type = 'cmt';
      else if (match[3] !== undefined) type = 'num';
      else if (match[4] !== undefined) type = 'name';

      tokens.push({ type, val, line: lineNo });
      lineNo += countNewlines(val);
      lastIdx = match.index + val.length;
    }
    if (lastIdx < rawCode.length) {
      tokens.push({ type: 'ws', val: rawCode.slice(lastIdx), line: lineNo });
    }

    // Filter non-ws index lookup helpers
    const prevNonWs = (idx) => {
      for (let k = idx - 1; k >= 0; k--) {
        if (tokens[k].type !== 'ws' && tokens[k].type !== 'cmt') return tokens[k];
      }
      return null;
    };
    const nextNonWs = (idx) => {
      for (let k = idx + 1; k < tokens.length; k++) {
        if (tokens[k].type !== 'ws' && tokens[k].type !== 'cmt') return tokens[k];
      }
      return null;
    };

    let bracketDepth = 0;
    let decLine = -1;
    let out = '';

    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (t.type === 'ws') {
        out += escapeHtml(t.val);
        continue;
      }

      const prev = prevNonWs(i);
      const next = nextNonWs(i);

      if (decLine !== -1 && t.line !== decLine) decLine = -1;
      if (t.type === 'op' && t.val === '(') decLine = -1;

      if (t.type === 'cmt') {
        out += wrapSpan('vs-cmt', t.val);
      } else if (t.type === 'num') {
        out += wrapSpan('vs-num', t.val);
      } else if (t.type === 'str') {
        const prefixMatch = t.val.match(/^[fFrRbBuU]*/);
        const prefix = prefixMatch ? prefixMatch[0].toLowerCase() : '';
        if (prefix.includes('f') && t.val.includes('{')) {
          out += highlightFString(t.val);
        } else {
          out += wrapSpan('vs-str', t.val);
        }
      } else if (t.type === 'op') {
        if (t.val === '@' && (!prev || prev.line !== t.line)) {
          decLine = t.line;
          out += wrapSpan('vs-dec', t.val);
        } else if ('([{'.includes(t.val)) {
          const cls = `vs-br-${(bracketDepth % 3) + 1}`;
          bracketDepth += 1;
          out += wrapSpan(cls, t.val);
        } else if (')]}'.includes(t.val)) {
          bracketDepth = Math.max(0, bracketDepth - 1);
          const cls = `vs-br-${(bracketDepth % 3) + 1}`;
          out += wrapSpan(cls, t.val);
        } else {
          out += wrapSpan('vs-op', t.val);
        }
      } else if (t.type === 'name') {
        const name = t.val;
        const isAttr = Boolean(prev && prev.type === 'op' && prev.val === '.');
        const isCall = Boolean(next && next.type === 'op' && next.val === '(');
        const isKwarg = Boolean(
          bracketDepth > 0 &&
          !isAttr &&
          next && next.type === 'op' && next.val === '=' &&
          (!prev || (prev.type === 'op' && (prev.val === '(' || prev.val === ',')))
        );

        if (decLine === t.line) {
          out += wrapSpan('vs-dec', name);
        } else if (KW_CTRL.has(name)) {
          out += wrapSpan('vs-kw-ctrl', name);
        } else if (KW_DECL.has(name)) {
          out += wrapSpan('vs-kw', name);
        } else if ((name === 'self' || name === 'cls') && !isAttr) {
          out += wrapSpan('vs-self', name);
        } else if (prev && prev.type === 'name' && prev.val === 'def') {
          out += wrapSpan('vs-fn', name);
        } else if (prev && prev.type === 'name' && prev.val === 'class') {
          out += wrapSpan('vs-cls', name);
        } else if (isKwarg) {
          out += wrapSpan('vs-param', name);
        } else if (isAttr && isCall && /^[a-z_]/.test(name)) {
          out += wrapSpan('vs-fn', name);
        } else if (isClassLike(name, importedNames) && !(isAttr && /^[a-z_]/.test(name))) {
          out += wrapSpan('vs-cls', name);
        } else if (isCall || (BUILTIN_FNS.has(name) && !isAttr)) {
          out += wrapSpan('vs-fn', name);
        } else if (isConstantName(name)) {
          out += wrapSpan('vs-const', name);
        } else if (paramNames.has(name) && !isAttr) {
          out += wrapSpan('vs-param', name);
        } else {
          out += wrapSpan('vs-var', name);
        }
      }
    }

    return out
      .split('\n')
      .map((ln) => `<span class="code-line">${ln}</span>`)
      .join('\n');
  }

  function renderCodeString(rawCode) {
    const normalized = rawCode.replace(/\r\n/g, '\n').replace(/^\n+|\n+$/g, '');
    if (/[├└│]/.test(normalized)) {
      return { html: highlightTree(normalized), raw: normalized };
    }
    return { html: highlightPython(normalized), raw: normalized };
  }

  async function fetchCodeSource(src) {
    // 1. Try live fetch() first (works on Live Server, python -m http.server, GitHub Pages, etc.)
    if (typeof fetch === 'function' && window.location && window.location.protocol !== 'file:') {
      try {
        const resp = await fetch(src, { cache: 'no-cache' });
        if (resp.ok) {
          return await resp.text();
        }
      } catch (_err) {
        // Fall through to offline registry
      }
    }
    // 2. Offline file:// fallback via auto-synced window.SYLLABUS_CODE_FILES
    const reg = window.SYLLABUS_CODE_FILES || {};
    if (Object.prototype.hasOwnProperty.call(reg, src)) {
      return reg[src];
    }
    const cleanKey = src.replace(/^\.\//, '');
    if (Object.prototype.hasOwnProperty.call(reg, cleanKey)) {
      return reg[cleanKey];
    }
    throw new Error(`Không tìm thấy file mã nguồn: ${src}`);
  }

  async function initExternalCodeLoader() {
    const blocks = document.querySelectorAll('pre[data-code-src]');
    await Promise.all(
      Array.from(blocks).map(async (pre) => {
        const src = pre.getAttribute('data-code-src');
        if (!src) return;
        try {
          const rawText = await fetchCodeSource(src);
          const rendered = renderCodeString(rawText);
          pre.dataset.rawCode = rendered.raw;
          pre.innerHTML = rendered.html;
        } catch (err) {
          pre.textContent = `# Lỗi nạp file ${src}: ${err.message}`;
        }
      })
    );
  }

  // Export for Node verification & Browser runtime
  if (typeof window !== 'undefined') {
    window.renderSyllabusCodeString = renderCodeString;
    window.initExternalCodeLoader = initExternalCodeLoader;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { renderCodeString, initExternalCodeLoader };
  }
})();
