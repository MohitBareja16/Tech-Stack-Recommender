/* ═══════════════════════════════════════════════
   app.js — UI controller
═══════════════════════════════════════════════ */

// ── PRESETS ──────────────────────────────────
const PRESETS = {
  data:     ["python", "machine_learning", "statistics", "pandas", "sql"],
  devops:   ["aws", "docker", "kubernetes", "linux", "terraform"],
  frontend: ["javascript", "react", "css", "typescript", "ui_ux"],
  security: ["networking", "security", "linux", "cryptography", "firewalls"],
  cold:     ["photography", "painting", "sculpture"],
};

// ── STATE ─────────────────────────────────────
let currentSkills = [];

// ── ELEMENTS ──────────────────────────────────
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

// ── INIT ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  buildDatasetGrid();
  buildTestMatrix();
  drawCosineCanvas();
  setupCopyButtons();

  // Compute and show vocab size in hero stats
  const v = buildVocabulary(DATASET);
  statVocab.textContent = v.length;

  // Controls
  topNSlider.addEventListener('input', () => {
    topNValue.textContent = topNSlider.value;
  });

  // Skill input
  skillInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') addSkillFromInput();
  });
  addBtn.addEventListener('click', addSkillFromInput);

  // Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const key = btn.dataset.preset;
      currentSkills = [];
      PRESETS[key].forEach(s => addSkill(s));
      document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Run
  runBtn.addEventListener('click', runRecommender);

  // Hero CTA smooth scroll
  document.getElementById('hero-demo-btn').addEventListener('click', e => {
    e.preventDefault();
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
  });
});

// ── SKILL MANAGEMENT ──────────────────────────
function addSkillFromInput() {
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
  skillTagsEl.innerHTML = '';
  currentSkills.forEach(s => {
    const tag = document.createElement('span');
    tag.className = 'skill-tag';
    tag.innerHTML = `${s}<span class="skill-tag-x" data-skill="${s}" title="Remove">×</span>`;
    tag.querySelector('.skill-tag-x').addEventListener('click', () => removeSkill(s));
    skillTagsEl.appendChild(tag);
  });
  skillBadge.textContent = `${currentSkills.length} skill${currentSkills.length !== 1 ? 's' : ''}`;
}

function updateRunState() {
  const ok = currentSkills.length >= 3;
  runBtn.disabled = !ok;
  if (ok) {
    skillHint.textContent = `✔  ${currentSkills.length} skills ready.`;
    skillHint.className = 'skill-hint ok';
  } else {
    skillHint.innerHTML = `Add at least <strong>3 skills</strong> to unlock recommendations.`;
    skillHint.className = 'skill-hint';
  }
}

// ── RUN RECOMMENDER ───────────────────────────
function runRecommender() {
  if (currentSkills.length < 3) return;

  runBtn.innerHTML = '<span class="run-btn-icon">⟳</span> Computing…';
  runBtn.disabled = true;

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

  if (verboseToggle.checked) {
    renderDebug(output.userVec, output.vocab, output.oov, output.results[0]);
    debugPanel.classList.remove('hidden');
  } else {
    debugPanel.classList.add('hidden');
  }

  resetRunBtn();
}

function resetRunBtn() {
  runBtn.innerHTML = '<span class="run-btn-icon">▶</span> Run Recommender';
  runBtn.disabled = false;
}

// ── RENDER RESULTS ────────────────────────────
function renderResults(results, userVec, vocab, oov, elapsed) {
  resultTime.textContent = `${elapsed}ms`;
  resultsEl.innerHTML = '';
  resultsEl.classList.remove('results-empty');

  const hasFallback = results.some(r => r.isFallback);
  if (hasFallback) {
    const banner = document.createElement('div');
    banner.className = 'fallback-banner';
    banner.innerHTML = '⚠️ <strong>Cold Start</strong> — all skills are out-of-vocabulary. Showing trending fallback.';
    resultsEl.appendChild(banner);
  }

  const rankEmojis = ['🥇', '🥈', '🥉'];
  const maxScore = Math.max(...results.filter(r => r.score !== null).map(r => r.score), 0.001);

  results.forEach((r, i) => {
    const card = document.createElement('div');
    card.className = `result-card${i === 0 && !r.isFallback ? ' result-card--top' : ''}${r.isFallback ? ' result-card--fallback' : ''}`;
    card.style.animationDelay = `${i * 60}ms`;

    const scorePercent = r.score !== null ? (r.score / maxScore) * 100 : 0;
    const matchedStr = r.matched.length
      ? `Matched: <span>${r.matched.join(', ')}</span>`
      : `<span style="color:var(--c-text-3)">No overlapping vocabulary</span>`;

    card.innerHTML = `
      <div class="result-rank result-rank--${i + 1}">${rankEmojis[i] || `#${i + 1}`}</div>
      <div class="result-body">
        <div class="result-name">${r.role}</div>
        <div class="result-skills">${r.isFallback ? '⭐ Popularity-ranked default' : matchedStr}</div>
      </div>
      <div class="result-score-wrap">
        ${r.isFallback
          ? `<span class="result-score result-score--fallback">⚠ fallback</span>`
          : `<span class="result-score">${r.score.toFixed(4)}</span>
             <div class="result-score-bar-wrap">
               <div class="result-score-bar" style="width:${scorePercent}%"></div>
             </div>`}
      </div>`;
    resultsEl.appendChild(card);
  });
}

function showError(msg) {
  resultsEl.innerHTML = `<div class="empty-state" style="color:var(--c-red)">
    <div class="empty-icon">✗</div><p>${msg}</p></div>`;
  resultsEl.classList.add('results-empty');
}

// ── RENDER DEBUG VIEW ─────────────────────────
function renderDebug(userVec, vocab, oov, topResult) {
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
    zero.textContent = '⃝ Zero vector — Cold Start condition (all skills OOV)';
    debugBody.appendChild(zero);
  }

  if (oov.length) {
    const oovEl = document.createElement('span');
    oovEl.className = 'debug-oov';
    oovEl.textContent = `OOV (zero weight): ${oov.join(', ')}`;
    debugBody.appendChild(oovEl);
  }
}

// ── DATASET GRID ──────────────────────────────
function buildDatasetGrid() {
  const grid = document.getElementById('dataset-grid');
  DATASET.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'dataset-card';
    card.style.animationDelay = `${i * 40}ms`;
    card.innerHTML = `
      <div class="dataset-card-rank">RANK #${item.rank}</div>
      <div class="dataset-card-name">${item.role}</div>
      <div class="dataset-card-skills">
        ${item.skills.map(s => `<span class="dataset-skill-tag">${s}</span>`).join('')}
      </div>`;
    grid.appendChild(card);
  });
}

// ── TEST MATRIX ───────────────────────────────
const TEST_MODULES = [
  { file: 'test_data_loader.py',       cases: 18, desc: 'TC-DL-*' },
  { file: 'test_vectorizer.py',        cases: 17, desc: 'TC-VEC-*' },
  { file: 'test_similarity_engine.py', cases: 12, desc: 'TC-SIM-*' },
  { file: 'test_ranker.py',            cases: 13, desc: 'TC-RANK-*' },
  { file: 'test_presenter.py',         cases: 8,  desc: 'TC-PRES-*' },
  { file: 'test_integration.py',       cases: 20, desc: 'TC-INT-* · TC-EDGE-*' },
];

function buildTestMatrix() {
  const matrix = document.getElementById('test-matrix');
  TEST_MODULES.forEach(mod => {
    const row = document.createElement('div');
    row.className = 'test-row';
    row.innerHTML = `
      <span class="test-row-file">${mod.file}</span>
      <span class="test-row-cases">${mod.desc}</span>
      <span class="test-row-badge">✔ ${mod.cases} tests</span>`;
    matrix.appendChild(row);
  });
}

// ── COSINE CANVAS ─────────────────────────────
function drawCosineCanvas() {
  const canvas = document.getElementById('cosine-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const ox = W / 2, oy = H * 0.88;

  ctx.clearRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  for (let r = 40; r < 160; r += 40) {
    ctx.beginPath();
    ctx.arc(ox, oy, r, -Math.PI, 0);
    ctx.stroke();
  }

  const len = 120;
  const angle1 = Math.PI * 0.82; // ~147° from positive x → upper-left
  const angle2 = Math.PI * 0.65; // ~117° from positive x → upper-left (closer)

  // Angle arc
  ctx.beginPath();
  ctx.arc(ox, oy, 38, -angle1, -angle2);
  ctx.strokeStyle = 'rgba(52,211,153,0.5)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Label θ
  ctx.fillStyle = 'rgba(52,211,153,0.9)';
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText('θ', ox - 26, oy - 44);

  // User vector A (indigo)
  drawArrow(ctx, ox, oy, ox + Math.cos(Math.PI - angle1) * len, oy - Math.sin(angle1) * len, '#6366f1', 2.5, 'A');
  // Item vector B (cyan)
  drawArrow(ctx, ox, oy, ox + Math.cos(Math.PI - angle2) * len, oy - Math.sin(angle2) * len, '#22d3ee', 2.5, 'B');

  // Origin dot
  ctx.beginPath();
  ctx.arc(ox, oy, 4, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.4)';
  ctx.fill();
}

function drawArrow(ctx, x1, y1, x2, y2, color, width, label) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
  // Arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 10 * Math.cos(angle - 0.4), y2 - 10 * Math.sin(angle - 0.4));
  ctx.lineTo(x2 - 10 * Math.cos(angle + 0.4), y2 - 10 * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
  // Label
  if (label) {
    ctx.fillStyle = color;
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.fillText(label, x2 + 6, y2 - 6);
  }
}

// ── COPY BUTTONS ──────────────────────────────
function setupCopyButtons() {
  document.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const text = btn.dataset.copy;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = '✔';
        btn.classList.add('copied');
        showToast();
        setTimeout(() => { btn.textContent = '⧉'; btn.classList.remove('copied'); }, 2000);
      });
    });
  });
}

function showToast() {
  toast.classList.remove('hidden');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => toast.classList.add('hidden'), 2000);
}
