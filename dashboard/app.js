/* ═══════════════════════════════════════════════
   app.js — UI Controller for Tech Stack Recommender
   Handles tabs, presets, autocomplete, skill tags, 
   recommendations, and responsive light-theme updates.
   ═══════════════════════════════════════════════ */

// ── SKILL PRESETS ──────────────────────────────
const PRESETS = {
  data:     ["python", "machine_learning", "statistics", "pandas", "sql"],
  devops:   ["aws", "docker", "kubernetes", "linux", "terraform"],
  frontend: ["javascript", "react", "css", "typescript", "ui_ux"],
  security: ["networking", "security", "linux", "cryptography", "firewalls"],
  cold:     ["photography", "painting", "sculpture"],
};

// ── APP STATE ──────────────────────────────────
let currentSkills = [];

// ── DOM ELEMENTS ────────────────────────────────
const skillInput    = document.getElementById('skill-input');
const addBtn        = document.getElementById('add-skill-btn');
const skillTagsEl   = document.getElementById('skill-tags');
const skillHint     = document.getElementById('skill-hint');
const skillBadge    = document.getElementById('skill-count-badge');
const runBtn        = document.getElementById('run-btn');
const resultsEl     = document.getElementById('results-container');
const debugPanel    = document.getElementById('debug-panel');
const debugBody     = document.getElementById('debug-body');
const topNSlider    = document.getElementById('top-n-slider');
const topNValue     = document.getElementById('top-n-value');
const verboseToggle = document.getElementById('verbose-toggle');
const resultTime    = document.getElementById('result-time');
const toast         = document.getElementById('toast');
const statVocab     = document.getElementById('stat-vocab');
const statRoles     = document.getElementById('stat-roles');
const autocompleteDropdown = document.getElementById('autocomplete-dropdown');

// ── INITIALIZATION ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // 1. Setup layout, datasets & tests
  buildDatasetGrid();
  buildTestMatrix();
  drawCosineCanvas();
  setupCopyButtons();
  setupTabNavigation();
  setupDatasetSearch();
  setupAutocomplete();

  // 2. Compute and display vocab size in sidebar
  if (statVocab) statVocab.textContent = VOCAB.length;
  if (statRoles) statRoles.textContent = DATASET.length;

  // 3. Slider controls
  if (topNSlider && topNValue) {
    topNSlider.addEventListener('input', () => {
      topNValue.textContent = topNSlider.value;
    });
  }

  // 4. Skill input controls
  if (skillInput) {
    skillInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        addSkillFromInput();
        hideAutocomplete();
      }
    });
  }
  if (addBtn) {
    addBtn.addEventListener('click', () => {
      addSkillFromInput();
      hideAutocomplete();
    });
  }

  // 5. Presets selection
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.preset;
      currentSkills = [];
      PRESETS[key].forEach(s => addSkill(s));
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // 6. Recommendation trigger
  if (runBtn) {
    runBtn.addEventListener('click', runRecommender);
  }
});

// ── TAB NAVIGATION LOGIC ───────────────────────
function setupTabNavigation() {
  const tabs = document.querySelectorAll('.nav-link[data-tab]');
  const pages = document.querySelectorAll('.tab-page');

  tabs.forEach(tab => {
    tab.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = tab.dataset.tab;
      
      // Update links and tab pages
      tabs.forEach(t => t.classList.remove('active'));
      pages.forEach(p => p.classList.remove('active'));
      
      tab.classList.add('active');
      const targetPage = document.getElementById(`page-${targetTab}`);
      if (targetPage) targetPage.classList.add('active');

      // Special redraw trigger for canvas if viewing math tab
      if (targetTab === 'algorithm') {
        setTimeout(drawCosineCanvas, 60);
      }
    });
  });
}

// ── AUTOCOMPLETE SUGGEST ENGINE ────────────────
function setupAutocomplete() {
  if (!skillInput || !autocompleteDropdown) return;

  skillInput.addEventListener('input', () => {
    const query = skillInput.value.toLowerCase().trim();
    if (!query) {
      hideAutocomplete();
      return;
    }

    // Find vocab words starting with or containing the query, not already in selected skills
    const matches = VOCAB
      .filter(term => term.includes(query) && !currentSkills.includes(term))
      .slice(0, 6);

    if (matches.length === 0) {
      hideAutocomplete();
      return;
    }

    autocompleteDropdown.innerHTML = '';
    matches.forEach(match => {
      const item = document.createElement('div');
      item.className = 'autocomplete-item';
      item.textContent = match.replace(/_/g, ' '); // human friendly text
      item.addEventListener('click', () => {
        addSkill(match);
        skillInput.value = '';
        hideAutocomplete();
        skillInput.focus();
      });
      autocompleteDropdown.appendChild(item);
    });

    autocompleteDropdown.classList.remove('hidden');
  });

  // Close suggestions click-outside
  document.addEventListener('click', (e) => {
    if (!e.target.closest('.skill-input-wrapper')) {
      hideAutocomplete();
    }
  });
}

function hideAutocomplete() {
  if (autocompleteDropdown) {
    autocompleteDropdown.classList.add('hidden');
    autocompleteDropdown.innerHTML = '';
  }
}

// ── DATASET SEARCH FILTER ──────────────────────
function setupDatasetSearch() {
  const searchInput = document.getElementById('dataset-search');
  const visibleCounter = document.getElementById('dataset-visible-count');
  
  if (!searchInput) return;
  
  searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase().trim();
    const cards = document.querySelectorAll('.dataset-card');
    let visibleCount = 0;

    cards.forEach(card => {
      const roleName = card.querySelector('.dataset-card-name').textContent.toLowerCase();
      const tags = Array.from(card.querySelectorAll('.dataset-skill-tag')).map(t => t.textContent.toLowerCase());
      
      const matches = roleName.includes(query) || tags.some(t => t.includes(query));
      if (matches) {
        card.classList.remove('card-hidden');
        visibleCount++;
      } else {
        card.classList.add('card-hidden');
      }
    });

    if (visibleCounter) {
      visibleCounter.textContent = visibleCount;
    }
  });
}

// ── SKILL MANAGEMENT ──────────────────────────
function addSkillFromInput() {
  if (!skillInput) return;
  const val = skillInput.value.trim();
  if (!val) return;
  addSkill(val);
  skillInput.value = '';
  skillInput.focus();
}

function addSkill(raw) {
  const norm = normalizeText(raw);
  if (!norm || currentSkills.includes(norm)) return;
  currentSkills.push(norm);
  renderSkillTags();
  updateRunState();
}

function removeSkill(norm) {
  currentSkills = currentSkills.filter(s => s !== norm);
  renderSkillTags();
  updateRunState();
}

function renderSkillTags() {
  if (!skillTagsEl) return;
  skillTagsEl.innerHTML = '';
  currentSkills.forEach(s => {
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    tag.innerHTML = `${s.replace(/_/g, ' ')} <span class="skill-tag-x" data-skill="${s}" title="Remove">×</span>`;
    tag.querySelector('.skill-tag-x').addEventListener('click', () => removeSkill(s));
    skillTagsEl.appendChild(tag);
  });
  if (skillBadge) {
    skillBadge.textContent = `${currentSkills.length} skill${currentSkills.length !== 1 ? 's' : ''}`;
  }
}

function updateRunState() {
  if (!runBtn || !skillHint) return;
  const ok = currentSkills.length >= 3;
  runBtn.disabled = !ok;
  if (ok) {
    skillHint.textContent = `Skills loaded and ready.`;
    skillHint.className = 'skill-hint ok';
  } else {
    skillHint.innerHTML = `Add at least <strong>3 skills</strong> to compute recommendations.`;
    skillHint.className = 'skill-hint';
  }
}

// ── RUN RECOMMENDER ───────────────────────────
function runRecommender() {
  if (currentSkills.length < 3) return;

  if (runBtn) {
    runBtn.innerHTML = 'Computing Similarity...';
    runBtn.disabled = true;
  }

  const t0 = performance.now();
  let output;
  try {
    output = recommend(currentSkills, parseInt(topNSlider.value));
  } catch (err) {
    showError(err.message);
    resetRunBtn();
    return;
  }
  const elapsed = (performance.now() - t0).toFixed(1);

  renderResults(output.results, output.userVec, output.vocab, output.oov, elapsed);

  if (verboseToggle && verboseToggle.checked) {
    renderDebug(output.userVec, output.vocab, output.oov, output.results[0]);
    if (debugPanel) debugPanel.classList.remove('hidden');
  } else {
    if (debugPanel) debugPanel.classList.add('hidden');
  }

  resetRunBtn();
}

function resetRunBtn() {
  if (runBtn) {
    runBtn.innerHTML = `
      <svg class="run-icon" viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
      Run Recommender`;
    runBtn.disabled = false;
  }
}

// ── RENDER RECOMMENDATIONS ───────────────────
function renderResults(results, userVec, vocab, oov, elapsed) {
  if (!resultsEl) return;
  resultsEl.innerHTML = '';
  resultsEl.classList.remove('results-empty');

  if (resultTime) {
    resultTime.textContent = `${elapsed}ms`;
  }

  const hasFallback = results.some(r => r.isFallback);
  if (hasFallback) {
    const banner = document.createElement('div');
    banner.className = 'fallback-banner';
    banner.innerHTML = '<strong>Cold Start Fallback:</strong> All provided skills are out-of-vocabulary. Showing trending popular paths.';
    resultsEl.appendChild(banner);
  }

  const maxScore = Math.max(...results.filter(r => r.score !== null).map(r => r.score), 0.0001);

  results.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = `result-card${i === 0 && !r.isFallback ? ' result-card--top' : ''}${r.isFallback ? ' result-card--fallback' : ''}`;
    card.style.animationDelay = `${i * 30}ms`;

    const scorePercent = r.score !== null ? (r.score / maxScore) * 100 : 0;
    const matchedStr = r.matched.length
      ? `Matched: <span>${r.matched.join(', ')}</span>`
      : `<span style="color:var(--text-muted)">No overlapping vocabulary</span>`;

    card.innerHTML = `
      <div class="result-rank">${i + 1}</div>
      <div class="result-body">
        <div class="result-name">${r.role}</div>
        <div class="result-skills">${r.isFallback ? 'Popularity-ranked fallback' : matchedStr}</div>
      </div>
      <div class="result-score-wrap">
        ${r.isFallback
          ? `<span class="result-score result-score--fallback">fallback</span>`
          : `<span class="result-score">${r.score.toFixed(4)}</span>
             <div class="result-score-bar-wrap">
               <div class="result-score-bar" style="width:${scorePercent}%"></div>
             </div>`}
      </div>`;
    resultsEl.appendChild(card);
  });
}

function showError(msg) {
  if (!resultsEl) return;
  resultsEl.innerHTML = `<div class="empty-state" style="color:var(--color-error)">
    <p>${msg}</p></div>`;
  resultsEl.classList.add('results-empty');
}

// ── RENDER DEBUG / EXPLAINER VIEW ──────────────
function renderDebug(userVec, vocab, oov, topResult) {
  if (!debugBody) return;
  debugBody.innerHTML = '';

  const maxW = Math.max(...userVec, 0.0001);
  let hasNonzero = false;

  vocab.forEach((term, i) => {
    const w = userVec[i];
    if (w === 0) return;
    hasNonzero = true;
    const dim = document.createElement('div');
    dim.className = 'debug-dim';
    dim.innerHTML = `
      <span class="debug-dim-term">${term}</span>
      <span class="debug-dim-weight">${w.toFixed(4)}</span>
      <div class="debug-dim-bar"><div class="debug-dim-fill" style="width:${(w/maxW)*100}%"></div></div>`;
    debugBody.appendChild(dim);
  });

  if (!hasNonzero) {
    const zero = document.createElement('span');
    zero.className = 'debug-oov';
    zero.textContent = 'Zero Vector — Cold Start detected (all input skills OOV)';
    debugBody.appendChild(zero);
  }

  if (oov.length) {
    const oovEl = document.createElement('span');
    oovEl.className = 'debug-oov';
    oovEl.textContent = `OOV (zero weight): ${oov.join(', ')}`;
    debugBody.appendChild(oovEl);
  }
}

// ── DATASET CORPUS GRID ─────────────────────────
function buildDatasetGrid() {
  const grid = document.getElementById('dataset-grid');
  if (!grid) return;
  
  grid.innerHTML = '';
  DATASET.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'dataset-card';
    card.style.animationDelay = `${i * 15}ms`;
    card.innerHTML = `
      <div class="dataset-card-rank">POPULARITY RANK ${item.rank}</div>
      <div class="dataset-card-name">${item.role}</div>
      <div class="dataset-card-skills">
        ${item.skills.map(s => `<span class="dataset-skill-tag">${s.replace(/_/g, ' ')}</span>`).join('')}
      </div>`;
    grid.appendChild(card);
  });
}

// ── TEST RESULTS MATRIX ─────────────────────────
const TEST_MODULES = [
  { file: 'test_data_loader.py',       cases: 29, desc: 'TC-DL-*' },
  { file: 'test_vectorizer.py',        cases: 29, desc: 'TC-VEC-*' },
  { file: 'test_similarity_engine.py', cases: 20, desc: 'TC-SIM-*' },
  { file: 'test_ranker.py',            cases: 22, desc: 'TC-RANK-*' },
  { file: 'test_presenter.py',         cases: 15, desc: 'TC-PRES-*' },
  { file: 'test_integration.py',       cases: 28, desc: 'TC-INT-* · TC-EDGE-*' },
];

function buildTestMatrix() {
  const matrix = document.getElementById('test-matrix');
  if (!matrix) return;
  
  matrix.innerHTML = '';
  TEST_MODULES.forEach(mod => {
    const row = document.createElement('div');
    row.className = 'test-row';
    row.innerHTML = `
      <span class="test-row-file">${mod.file}</span>
      <span class="test-row-cases">${mod.desc}</span>
      <span class="test-row-badge">pass: ${mod.cases} tests</span>`;
    matrix.appendChild(row);
  });
}

let cosineAnimFrameId = null;

// ── COSINE ANGLE VISUALIZATION ─────────────────
function drawCosineCanvas() {
  const canvas = document.getElementById('cosine-canvas');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const ox = W / 2, oy = H * 0.88;

  if (cosineAnimFrameId) {
    cancelAnimationFrame(cosineAnimFrameId);
  }

  let progress = 0;
  const duration = 40; // frames (~650ms)
  const angle1 = Math.PI * 0.82; // User vector A
  const angle2 = Math.PI * 0.65; // Role vector B

  function step() {
    progress += 1 / duration;
    if (progress > 1) progress = 1;

    ctx.clearRect(0, 0, W, H);

    // Draw concentric helper grids (semicircles radiating outwards)
    ctx.strokeStyle = 'rgba(15,23,42,0.03)';
    ctx.lineWidth = 1;
    for (let r = 40; r <= 160; r += 40) {
      ctx.beginPath();
      ctx.arc(ox, oy, r * progress, -Math.PI, 0);
      ctx.stroke();
    }

    const currentLen = 120 * progress;
    const currentAngle1 = angle1 * progress;
    const currentAngle2 = angle2 * progress;

    // Draw angle arc representing theta
    if (progress > 0.5) {
      const arcProgress = (progress - 0.5) * 2;
      ctx.beginPath();
      ctx.arc(ox, oy, 38, -angle1, -angle1 + (angle1 - angle2) * arcProgress);
      ctx.strokeStyle = 'rgba(22,163,74,0.3)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }

    // Label θ
    if (progress === 1) {
      ctx.fillStyle = 'rgba(22,163,74,0.8)';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillText('θ', ox - 26, oy - 44);
    }

    // User vector A (Blue)
    drawArrow(ctx, ox, oy, ox + Math.cos(Math.PI - currentAngle1) * currentLen, oy - Math.sin(currentAngle1) * currentLen, '#2563eb', 2, progress === 1 ? 'A' : '');
    
    // Item vector B (Cyan)
    drawArrow(ctx, ox, oy, ox + Math.cos(Math.PI - currentAngle2) * currentLen, oy - Math.sin(currentAngle2) * currentLen, '#0ea5e9', 2, progress === 1 ? 'B' : '');

    // Origin point
    ctx.beginPath();
    ctx.arc(ox, oy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(15,23,42,0.2)';
    ctx.fill();

    if (progress < 1) {
      cosineAnimFrameId = requestAnimationFrame(step);
    }
  }

  step();
}

function drawArrow(ctx, x1, y1, x2, y2, color, width, label) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  
  // Draw arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 8 * Math.cos(angle - 0.4), y2 - 8 * Math.sin(angle - 0.4));
  ctx.lineTo(x2 - 8 * Math.cos(angle + 0.4), y2 - 8 * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  
  // Label name
  if (label) {
    ctx.fillStyle = color;
    ctx.font = 'bold 12px Plus Jakarta Sans, sans-serif';
    ctx.fillText(label, x2 + 8, y2 - 4);
  }
}

// ── COPY TO CLIPBOARD BUTTONS ──────────────────
function setupCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.dataset.copy;
      navigator.clipboard.writeText(text).then(() => {
        // Change SVG icon inside button to checkmark
        const originalIconHtml = btn.innerHTML;
        btn.innerHTML = `
          <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;
        btn.classList.add('copied');
        showToast();
        setTimeout(() => { 
          btn.innerHTML = originalIconHtml; 
          btn.classList.remove('copied'); 
        }, 1800);
      });
    });
  });
}

function showToast() {
  if (!toast) return;
  toast.classList.remove('hidden');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.add('hidden'), 1800);
}
