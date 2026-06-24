/* ═══════════════════════════════════════════════
   data.js — Dataset + TF-IDF engine (browser-side)
   Mirrors src/vectorizer.py + src/similarity_engine.py
═══════════════════════════════════════════════ */

const DATASET = [
  { role: "Data Scientist",       skills: ["python","sql","machine_learning","statistics","pandas","numpy","data_visualization"], rank: 1 },
  { role: "Backend Developer",    skills: ["java","python","sql","rest_api","databases","git","spring_boot"],                    rank: 2 },
  { role: "DevOps Engineer",      skills: ["aws","docker","kubernetes","linux","ci_cd","terraform","git"],                      rank: 3 },
  { role: "Frontend Developer",   skills: ["javascript","html","css","react","typescript","ui_ux","git"],                       rank: 4 },
  { role: "Full Stack Developer", skills: ["javascript","python","react","node.js","sql","rest_api","git"],                     rank: 5 },
  { role: "ML Engineer",          skills: ["python","machine_learning","deep_learning","tensorflow","pytorch","numpy","mlops"], rank: 6 },
  { role: "Cloud Architect",      skills: ["aws","azure","gcp","kubernetes","terraform","networking","security"],               rank: 7 },
  { role: "Cybersecurity Analyst",skills: ["networking","security","linux","firewalls","penetration_testing","cryptography"],   rank: 8 },
  { role: "Data Analyst",         skills: ["sql","python","data_visualization","excel","tableau","statistics","pandas"],        rank: 9 },
  { role: "Mobile Developer",     skills: ["kotlin","swift","java","react_native","ui_ux","git","rest_api"],                   rank: 10 },
];

// ── Normalize text (mirrors data_loader.normalize_text) ──
function normalizeText(s) {
  return String(s).trim().toLowerCase().replace(/\s+/g, ' ');
}

// ── Build vocabulary ──
function buildVocabulary(items) {
  const set = new Set();
  items.forEach(item => item.skills.forEach(s => set.add(s)));
  return [...set].sort();
}

// ── Compute TF ──
function computeTF(tokens, vocab) {
  const tf = new Float64Array(vocab.length);
  if (!tokens.length) return tf;
  const counts = {};
  tokens.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
  vocab.forEach((term, i) => { tf[i] = (counts[term] || 0) / tokens.length; });
  return tf;
}

// ── Compute IDF ──
function computeIDF(items, vocab) {
  const N = items.length;
  return new Float64Array(vocab.map(term => {
    const df = items.filter(it => it.skills.includes(term)).length;
    return df > 0 ? Math.log(N / df) : 0;
  }));
}

// ── Build item matrix (N × V) ──
function buildItemMatrix(items, vocab, idf) {
  return items.map(item => {
    const tf = computeTF(item.skills, vocab);
    return tf.map((v, i) => v * idf[i]);
  });
}

// ── Build user vector ──
function buildUserVector(userSkills, vocab, idf) {
  const vocabSet = new Set(vocab);
  const oov = userSkills.filter(s => !vocabSet.has(s));
  const tf = computeTF(userSkills, vocab);
  const vec = tf.map((v, i) => v * idf[i]);
  return { vec, oov };
}

// ── Cosine similarity ──
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Score all items ──
function scoreAllItems(userVec, itemMatrix, vocab) {
  const userNonzero = userVec.map((v, i) => ({ i, v })).filter(x => x.v > 0);
  return DATASET.map((item, idx) => {
    const row = itemMatrix[idx];
    const score = cosineSim(userVec, row);
    const matched = userNonzero
      .filter(x => row[x.i] > 0)
      .map(x => vocab[x.i]);
    return { role: item.role, score, matched, rank: item.rank, isFallback: false };
  });
}

// ── Cold start detection ──
function isColdStart(scored) {
  return scored.every(s => s.score === 0);
}

// ── Rank ──
function rankResults(scored, topN) {
  const sorted = [...scored].sort((a, b) =>
    b.score !== a.score ? b.score - a.score : a.role.localeCompare(b.role)
  );
  return sorted.slice(0, topN);
}

// ── Cold start fallback ──
function coldStartFallback(topN) {
  return [...DATASET]
    .sort((a, b) => a.rank - b.rank)
    .slice(0, topN)
    .map(item => ({ role: item.role, score: null, matched: [], rank: item.rank, isFallback: true }));
}

// ── Main recommend function ──
function recommend(rawSkills, topN = 3) {
  const normalized = rawSkills.map(normalizeText).filter(Boolean);
  if (normalized.length < 3) throw new Error(`Need ≥3 skills. Got ${normalized.length}.`);

  const vocab = buildVocabulary(DATASET);
  const idf   = computeIDF(DATASET, vocab);
  const matrix = buildItemMatrix(DATASET, vocab, idf);
  const { vec: userVec, oov } = buildUserVector(normalized, vocab, idf);

  const scored = scoreAllItems(userVec, matrix, vocab);
  const results = isColdStart(scored)
    ? coldStartFallback(topN)
    : rankResults(scored, topN);

  return { results, userVec, vocab, idf, oov };
}
