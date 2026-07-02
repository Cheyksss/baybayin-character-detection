/**
 * ================================================================
 * BAYBAYIN AI — Frontend Application Script
 * ================================================================
 * Handles:
 *  - Navbar scroll behaviour
 *  - Image upload via drag-and-drop or file picker
 *  - Mock "Process Image" pipeline (replace fetch() stub with
 *    your real Flask/FastAPI endpoint)
 *  - Results dashboard rendering (stats, annotation canvas,
 *    detection log table)
 *  - Baybayin Reference Guide tab switching + grid generation
 *  - CSV export
 *  - Toast notifications
 * ================================================================
 */

'use strict';

/* ================================================================
   0. SECURITY UTILITY
   ================================================================
   escapeHTML() prevents XSS when backend data is injected into
   innerHTML. Call it on every string field that comes from the
   API response before inserting it into the DOM.
   Currently all data is from local constants, but this is already
   wired in so it's safe the moment you connect the real backend.
================================================================= */

/**
 * Escapes a value to be safely inserted into HTML.
 * Converts the five characters that could break out of an HTML
 * context: & < > " '
 * @param {*} value - Any value; non-strings are coerced first.
 * @returns {string} HTML-safe string
 */
function escapeHTML(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
}

/* ================================================================
   1. DATA — Baybayin syllabary
   ================================================================
   59 classes = 17 base chars × 3 vowel states + 2 standalone
   vowels + virama forms.  Unicode glyphs use the Tagalog Unicode
   block (U+1700–U+171F).
================================================================= */

const BAYBAYIN_DATA = {
  vowels: [
    { glyph: 'ᜀ', label: 'A',  romanized: 'a',  classId: 'CLS-01' },
    { glyph: 'ᜁ', label: 'I',  romanized: 'i/e', classId: 'CLS-02' },
    { glyph: 'ᜂ', label: 'U',  romanized: 'u/o', classId: 'CLS-03' },
  ],
  consonants: [
    { glyph: 'ᜊ', label: 'BA', romanized: 'ba',  classId: 'CLS-04' },
    { glyph: 'ᜃ', label: 'KA', romanized: 'ka',  classId: 'CLS-07' },
    { glyph: 'ᜇ', label: 'DA', romanized: 'da',  classId: 'CLS-10' },
    { glyph: 'ᜄ', label: 'GA', romanized: 'ga',  classId: 'CLS-13' },
    { glyph: 'ᜑ', label: 'HA', romanized: 'ha',  classId: 'CLS-16' },
    { glyph: 'ᜎ', label: 'LA', romanized: 'la',  classId: 'CLS-19' },
    { glyph: 'ᜋ', label: 'MA', romanized: 'ma',  classId: 'CLS-22' },
    { glyph: 'ᜈ', label: 'NA', romanized: 'na',  classId: 'CLS-25' },
    { glyph: 'ᜅ', label: 'NGA', romanized: 'nga', classId: 'CLS-28' },
    { glyph: 'ᜉ', label: 'PA', romanized: 'pa',  classId: 'CLS-31' },
    { glyph: 'ᜍ', label: 'RA', romanized: 'ra',  classId: 'CLS-34' },
    { glyph: 'ᜐ', label: 'SA', romanized: 'sa',  classId: 'CLS-37' },
    { glyph: 'ᜆ', label: 'TA', romanized: 'ta',  classId: 'CLS-40' },
    { glyph: 'ᜏ', label: 'WA', romanized: 'wa',  classId: 'CLS-43' },
    { glyph: 'ᜌ', label: 'YA', romanized: 'ya',  classId: 'CLS-46' },
  ],
  // Base consonants used in the kudlit table
  kudlitBase: [
    { glyph: 'ᜊ', base: 'B', a: 'BA', i: 'BI', u: 'BU' },
    { glyph: 'ᜃ', base: 'K', a: 'KA', i: 'KI', u: 'KU' },
    { glyph: 'ᜇ', base: 'D', a: 'DA', i: 'DI', u: 'DU' },
    { glyph: 'ᜄ', base: 'G', a: 'GA', i: 'GI', u: 'GU' },
    { glyph: 'ᜑ', base: 'H', a: 'HA', i: 'HI', u: 'HU' },
    { glyph: 'ᜎ', base: 'L', a: 'LA', i: 'LI', u: 'LU' },
    { glyph: 'ᜋ', base: 'M', a: 'MA', i: 'MI', u: 'MU' },
    { glyph: 'ᜈ', base: 'N', a: 'NA', i: 'NI', u: 'NU' },
    { glyph: 'ᜅ', base: 'NG', a: 'NGA', i: 'NGI', u: 'NGU' },
    { glyph: 'ᜉ', base: 'P', a: 'PA', i: 'PI', u: 'PU' },
    { glyph: 'ᜍ', base: 'R', a: 'RA', i: 'RI', u: 'RU' },
    { glyph: 'ᜐ', base: 'S', a: 'SA', i: 'SI', u: 'SU' },
    { glyph: 'ᜆ', base: 'T', a: 'TA', i: 'TI', u: 'TU' },
    { glyph: 'ᜏ', base: 'W', a: 'WA', i: 'WI', u: 'WU' },
    { glyph: 'ᜌ', base: 'Y', a: 'YA', i: 'YI', u: 'YU' },
  ],
};

/* ================================================================
   2. API CONFIGURATION
   The backend runs at http://127.0.0.1:8000 by default.
   Set SULATAI_API to a different URL if deployed remotely.
================================================================= */

const SULATAI_API = (
  typeof window !== 'undefined' &&
  window.SULATAI_API_URL
) || 'http://127.0.0.1:8000';

/**
 * Maps a raw model class label (e.g. "ba", "be_bi") to a
 * Baybayin Unicode glyph for display in the results table.
 * Falls back gracefully when no exact match is found.
 */
const LABEL_TO_GLYPH = {
  a: 'ᜀ', e_i: 'ᜁ', o_u: 'ᜂ',
  ba: 'ᜊ', be_bi: 'ᜊᜒ', bo_bu: 'ᜊᜓ', b: 'ᜊ᜔',
  ka: 'ᜃ', ke_ki: 'ᜃᜒ', ko_ku: 'ᜃᜓ', k: 'ᜃ᜔',
  da: 'ᜇ', de_di: 'ᜇᜒ', do_du: 'ᜇᜓ', d: 'ᜇ᜔',
  ga: 'ᜄ', ge_gi: 'ᜄᜒ', go_gu: 'ᜄᜓ', g: 'ᜄ᜔',
  ha: 'ᜑ', he_hi: 'ᜑᜒ', ho_hu: 'ᜑᜓ', h: 'ᜑ᜔',
  la: 'ᜎ', le_li: 'ᜎᜒ', lo_lu: 'ᜎᜓ', l: 'ᜎ᜔',
  ma: 'ᜋ', me_mi: 'ᜋᜒ', mo_mu: 'ᜋᜓ', m: 'ᜋ᜔',
  na: 'ᜈ', ne_ni: 'ᜈᜒ', no_nu: 'ᜈᜓ', n: 'ᜈ᜔',
  nga: 'ᜅ', nge_ngi: 'ᜅᜒ', ngo_ngu: 'ᜅᜓ', ng: 'ᜅ᜔',
  pa: 'ᜉ', pe_pi: 'ᜉᜒ', po_pu: 'ᜉᜓ', p: 'ᜉ᜔',
  sa: 'ᜐ', se_si: 'ᜐᜒ', so_su: 'ᜐᜓ', s: 'ᜐ᜔',
  ta: 'ᜆ', te_ti: 'ᜆᜒ', to_tu: 'ᜆᜓ', t: 'ᜆ᜔',
  wa: 'ᜏ', we_wi: 'ᜏᜒ', wo_wu: 'ᜏᜓ', w: 'ᜏ᜔',
  ya: 'ᜌ', ye_yi: 'ᜌᜒ', yo_yu: 'ᜌᜓ', y: 'ᜌ᜔',
};

/** Fallback detections used ONLY when the backend is unreachable */
const FALLBACK_DETECTIONS = [
  { id: 1, label: 'ba',  transliteration: 'BA',    bbox: [42,  30, 140, 125], confidence: 0.97 },
  { id: 2, label: 'ya',  transliteration: 'YA',    bbox: [155, 28, 251, 124], confidence: 0.91 },
  { id: 3, label: 'be_bi', transliteration: 'BE/BI', bbox: [268, 32, 364, 127], confidence: 0.88 },
  { id: 4, label: 'na',  transliteration: 'NA',    bbox: [42,  145, 140, 240], confidence: 0.95 },
  { id: 5, label: 'ka',  transliteration: 'KA',    bbox: [155, 148, 252, 243], confidence: 0.73 },
  { id: 6, label: 'ha',  transliteration: 'HA',    bbox: [270, 146, 366, 241], confidence: 0.82 },
];

/* ================================================================
   3. DOM ELEMENT REFERENCES
================================================================= */
// Holds detections from the most recent successful API call
let lastDetections = [];

const navbar        = document.getElementById('navbar');
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileMenu    = document.getElementById('mobileMenu');

const dropZone      = document.getElementById('dropZone');
const fileInput     = document.getElementById('fileInput');
const dropDefault   = document.getElementById('dropDefault');
const dropPreview   = document.getElementById('dropPreview');
const previewImg    = document.getElementById('previewImg');
const previewFilename = document.getElementById('previewFilename');
const removeImageBtn  = document.getElementById('removeImageBtn');

const processBtn      = document.getElementById('processBtn');
const processBtnText  = document.getElementById('processBtnText');
const processBtnIcon  = document.getElementById('processBtnIcon');
const processBtnSpinner = document.getElementById('processBtnSpinner');

const resultsEmpty   = document.getElementById('resultsEmpty');
const resultsContent = document.getElementById('resultsContent');
const resultOriginal = document.getElementById('resultOriginal');
const resultAnnotated  = document.getElementById('resultAnnotated');
const annotationCanvas = document.getElementById('annotationCanvas');
const resultsTableBody = document.getElementById('resultsTableBody');

const statTotal      = document.getElementById('statTotal');
const statConfidence = document.getElementById('statConfidence');
const statClasses    = document.getElementById('statClasses');
const statTime       = document.getElementById('statTime');

const exportBtn = document.getElementById('exportBtn');
const resetBtn  = document.getElementById('resetBtn');
const toast     = document.getElementById('toast');
const toastMsg  = document.getElementById('toastMsg');
const toastIcon = document.getElementById('toastIcon');

/* ================================================================
   4. NAVBAR — scroll effect
================================================================= */
window.addEventListener('scroll', () => {
  if (window.scrollY > 20) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
}, { passive: true });

/* Mobile menu toggle */
mobileMenuBtn.addEventListener('click', () => {
  mobileMenu.classList.toggle('hidden');
});

/* Close mobile menu when a link is clicked */
document.querySelectorAll('.mobile-nav-link').forEach(link => {
  link.addEventListener('click', () => mobileMenu.classList.add('hidden'));
});

/* ================================================================
   5. IMAGE UPLOAD — drag-and-drop + file picker
================================================================= */
let currentFile = null;    // Holds the File object currently staged
let currentImageURL = null; // Object URL for the staged image

/** Show the preview pane with the given File */
function showPreview(file) {
  if (!file || !file.type.startsWith('image/')) {
    showToast('Please upload a valid image file (PNG, JPG, WEBP).', 'error');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showToast('File exceeds 10MB limit. Please choose a smaller image.', 'error');
    return;
  }

  // Revoke previous object URL if any
  if (currentImageURL) URL.revokeObjectURL(currentImageURL);

  currentFile     = file;
  currentImageURL = URL.createObjectURL(file);

  previewImg.src            = currentImageURL;
  previewFilename.textContent = file.name;

  dropDefault.classList.add('hidden');
  dropPreview.classList.remove('hidden');
  fileInput.classList.remove('hidden'); // ensure it's accessible

  // Enable process button
  processBtn.disabled = false;
  showToast('Image ready — click "Process Image" to run detection.', 'success');
}

/** Reset back to empty upload state */
function clearUpload() {
  if (currentImageURL) {
    URL.revokeObjectURL(currentImageURL);
    currentImageURL = null;
  }
  currentFile = null;
  previewImg.src = '';
  fileInput.value = '';

  dropPreview.classList.add('hidden');
  dropDefault.classList.remove('hidden');
  processBtn.disabled = true;
}

/* File input change */
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) showPreview(file);
});

/* Click on drop zone (not on buttons) opens file picker */
dropZone.addEventListener('click', (e) => {
  // Don't re-trigger if remove button was clicked
  if (e.target.closest('#removeImageBtn')) return;
  if (dropPreview.classList.contains('hidden')) {
    fileInput.click();
  }
});

/* Keyboard accessibility for drop zone */
dropZone.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && dropPreview.classList.contains('hidden')) {
    e.preventDefault();
    fileInput.click();
  }
});

/* Remove image */
removeImageBtn.addEventListener('click', (e) => {
  e.stopPropagation();
  clearUpload();
  resetResults();
});

/* Drag-and-drop events */
['dragenter', 'dragover'].forEach(evt => {
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
});

['dragleave', 'dragend'].forEach(evt => {
  dropZone.addEventListener(evt, () => {
    dropZone.classList.remove('drag-over');
  });
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) showPreview(file);
});

/* ================================================================
   6. PROCESS IMAGE
   ================================================================
   Currently runs a mock pipeline with simulated delay.
   To connect your Flask/FastAPI backend, replace the
   simulateBackendCall() function with a real fetch() call.
================================================================= */

processBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  setProcessingState(true);

  try {
    const data = await callPredictAPI(currentFile);
    renderResults(data.detections, data.processingTime, data.annotatedImage);

    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
    showToast(`Detection complete — ${data.detections.length} characters found.`, 'success');

  } catch (err) {
    console.error('Processing error:', err);
    showToast(err.message || 'Processing failed. Please try again.', 'error');
  } finally {
    setProcessingState(false);
  }
});

/**
 * POST the image to the FastAPI /predict endpoint.
 * Falls back to demo data if the backend is unreachable so the
 * UI remains functional for offline demonstrations.
 *
 * @param {File} file
 * @returns {Promise<{detections: Array, processingTime: string, annotatedImage: string|null}>}
 */
async function callPredictAPI(file) {
  const formData = new FormData();
  formData.append('image', file);

  let response;
  try {
    response = await fetch(`${SULATAI_API}/predict`, {
      method: 'POST',
      body:   formData,
    });
  } catch (networkErr) {
    // Backend unreachable — fall back to demo detections
    console.warn('Backend unreachable, using fallback demo data.', networkErr);
    return buildFallbackResponse();
  }

  if (!response.ok) {
    let detail = `Server error ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_) { /* ignore */ }
    throw new Error(detail);
  }

  const json = await response.json();

  // Normalise backend response to the shape renderResults() expects
  return {
    detections:     normaliseDetections(json.detections ?? []),
    processingTime: `${(json.processing_time_ms / 1000).toFixed(2)}s`,
    annotatedImage: json.annotated_image ?? null,
  };
}

/**
 * Convert backend Detection objects into the shape the renderer uses.
 * Backend: { id, label, transliteration, bbox:[x1,y1,x2,y2], confidence }
 * Renderer needs: { id, glyph, label, romanized, bbox:[x,y,w,h], confidence }
 */
function normaliseDetections(raw) {
  return raw.map(d => {
    const [x1, y1, x2, y2] = d.bbox;
    return {
      id:         d.id,
      glyph:      LABEL_TO_GLYPH[d.label] ?? d.label,
      label:      d.transliteration ?? d.label.toUpperCase(),
      romanized:  d.label,
      bbox:       [x1, y1, x2 - x1, y2 - y1],   // convert to [x, y, w, h]
      confidence: d.confidence,
    };
  });
}

/** Build a response object from the static fallback data for offline demos. */
function buildFallbackResponse() {
  const delay = 1800 + Math.random() * 600;
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        detections:     FALLBACK_DETECTIONS.map(d => ({
          ...d,
          glyph:    LABEL_TO_GLYPH[d.label] ?? d.label,
          romanized: d.label,
        })),
        processingTime: `${(delay / 1000).toFixed(2)}s (demo)`,
        annotatedImage: null,
      });
    }, delay);
  });
}

/** Toggle button into loading / idle state */
function setProcessingState(isProcessing) {
  processBtn.disabled = isProcessing;
  processBtnIcon.classList.toggle('hidden', isProcessing);
  processBtnSpinner.classList.toggle('hidden', !isProcessing);
  processBtnText.textContent = isProcessing ? 'Processing…' : 'Process Image';
}

/* ================================================================
   7. RENDER RESULTS
================================================================= */

/**
 * Populate the entire results dashboard.
 * @param {Array}       detections      - Array of detection objects
 * @param {string}      processingTime  - Human-readable time string
 * @param {string|null} annotatedImage  - Base64 data-URI from backend (optional)
 */
function renderResults(detections, processingTime, annotatedImage = null) {
  // Show results panel
  resultsEmpty.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // ── Images ──────────────────────────────────────────────────
  resultOriginal.src = currentImageURL;

  if (annotatedImage) {
    // Backend returned a fully-annotated image — use it directly
    resultAnnotated.src = annotatedImage;
    clearCanvas();          // no canvas overlay needed
  } else {
    // Fall back to canvas-drawn bounding boxes (demo / offline mode)
    resultAnnotated.src = currentImageURL;
    resultAnnotated.onload = () => drawAnnotations(detections);
    if (resultAnnotated.complete) drawAnnotations(detections);
  }

  // ── Stats bar ────────────────────────────────────────────────
  lastDetections = detections;   // persist for CSV export + canvas resize
  const avgConf    = detections.reduce((s, d) => s + d.confidence, 0) / detections.length;
  const uniqueCls  = new Set(detections.map(d => d.label)).size;

  animateCounter(statTotal,      detections.length, '');
  animateCounter(statConfidence, Math.round(avgConf * 100), '%');
  animateCounter(statClasses,    uniqueCls, '');
  // textContent is inherently safe — no HTML parsing, just plain text
  statTime.textContent = processingTime;

  // ── Table ────────────────────────────────────────────────────
  resultsTableBody.innerHTML = '';
  detections.forEach(det => {
    const [x, y, w, h] = det.bbox;
    const confPct = Math.round(det.confidence * 100);
    const badgeCls = confPct >= 90 ? 'badge-high' : confPct >= 75 ? 'badge-medium' : 'badge-low';
    const badgeLbl = confPct >= 90 ? 'High' : confPct >= 75 ? 'Medium' : 'Low';

    // Escape all API-sourced string fields before DOM insertion (XSS prevention)
    const safeId        = escapeHTML(String(det.id).padStart(2, '0'));
    const safeGlyph     = escapeHTML(det.glyph);
    const safeLabel     = escapeHTML(det.label);
    const safeRomanized = escapeHTML(det.romanized);
    // Numeric values are coerced to numbers — not injectable
    const safeX  = Number(x)   || 0;
    const safeY  = Number(y)   || 0;
    const safeW  = Number(w)   || 0;
    const safeH  = Number(h)   || 0;

    const row = document.createElement('tr');
    row.className = 'table-row';
    row.innerHTML = `
      <td class="px-6 py-3.5 text-stone-500 text-xs font-mono">${safeId}</td>
      <td class="px-6 py-3.5">
        <span class="text-2xl leading-none" title="${safeLabel}">${safeGlyph}</span>
      </td>
      <td class="px-6 py-3.5">
        <span class="text-amber-400 font-bold text-base">${safeLabel}</span>
        <span class="text-stone-500 text-xs ml-1.5">(${safeRomanized})</span>
      </td>
      <td class="px-6 py-3.5 text-stone-400 text-xs font-mono">
        [${safeX}, ${safeY}, ${safeX + safeW}, ${safeY + safeH}]
      </td>
      <td class="px-6 py-3.5 min-w-[120px]">
        <div class="flex items-center gap-2">
          <span class="text-stone-300 text-sm font-medium">${confPct}%</span>
        </div>
        <div class="confidence-bar w-20">
          <div class="confidence-fill" style="width: ${confPct}%"></div>
        </div>
      </td>
      <td class="px-6 py-3.5">
        <span class="text-xs px-2.5 py-1 rounded-full font-medium ${badgeCls}">${badgeLbl}</span>
      </td>
    `;
    resultsTableBody.appendChild(row);
  });
}

/** Reset results panel to empty state */
function resetResults() {
  lastDetections = [];
  resultsEmpty.classList.remove('hidden');
  resultsContent.classList.add('hidden');
  resultsTableBody.innerHTML = '';
  clearCanvas();
}

/* ================================================================
   8. ANNOTATION CANVAS
   Draws colour-coded bounding boxes over the annotated image.
================================================================= */

/**
 * Draw detection bounding boxes onto the overlay canvas.
 * @param {Array} detections
 */
function drawAnnotations(detections) {
  const img    = resultAnnotated;
  const canvas = annotationCanvas;
  const ctx    = canvas.getContext('2d');

  // Match canvas to rendered image size
  canvas.width  = img.offsetWidth;
  canvas.height = img.offsetHeight;

  // Scale factor: model bbox is relative to natural image size
  const scaleX = img.offsetWidth  / img.naturalWidth;
  const scaleY = img.offsetHeight / img.naturalHeight;

  // Colour palette for boxes (cycles through)
  const COLORS = ['#f59e0b','#34d399','#60a5fa','#f472b6','#a78bfa','#fb923c'];

  detections.forEach((det, i) => {
    const [bx, by, bw, bh] = det.bbox;
    const x = bx * scaleX;
    const y = by * scaleY;
    const w = bw * scaleX;
    const h = bh * scaleY;
    const color = COLORS[i % COLORS.length];
    const confPct = Math.round(det.confidence * 100);

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(x, y, w, h);

    // Semi-transparent fill
    ctx.fillStyle = color + '18';
    ctx.fillRect(x, y, w, h);

    // Label background
    const labelText = `${det.label} ${confPct}%`;
    ctx.font = 'bold 11px Inter, sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    const labelH    = 18;
    const labelY    = y > labelH + 2 ? y - labelH - 2 : y + 2;

    ctx.fillStyle = color;
    ctx.fillRect(x, labelY, textWidth + 10, labelH);

    // Label text
    ctx.fillStyle = '#1c1917';
    ctx.fillText(labelText, x + 5, labelY + 13);
  });
}

function clearCanvas() {
  const ctx = annotationCanvas.getContext('2d');
  ctx.clearRect(0, 0, annotationCanvas.width, annotationCanvas.height);
}

/* Redraw on window resize to keep canvas aligned */
window.addEventListener('resize', () => {
  if (!resultsContent.classList.contains('hidden') && lastDetections.length) {
    drawAnnotations(lastDetections);
  }
}, { passive: true });

/* ================================================================
   9. ANIMATED COUNTER UTILITY
================================================================= */
/**
 * Animates a number counting up from 0 to target.
 * @param {HTMLElement} el     - Target element
 * @param {number}      target - Final value
 * @param {string}      suffix - Unit string appended (e.g. '%')
 */
function animateCounter(el, target, suffix) {
  const duration = 800;
  const start    = performance.now();
  function step(now) {
    const elapsed  = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased    = 1 - Math.pow(1 - progress, 3); // ease-out cubic
    el.textContent = Math.round(eased * target) + suffix;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ================================================================
   10. RESET BUTTON
================================================================= */
resetBtn.addEventListener('click', () => {
  clearUpload();
  resetResults();
  document.getElementById('workspace').scrollIntoView({ behavior: 'smooth' });
});

/* ================================================================
   11. EXPORT CSV
================================================================= */
exportBtn.addEventListener('click', () => {
  if (!lastDetections.length) return;

  const rows = [['#', 'Glyph', 'Label', 'Romanized', 'x_min', 'y_min', 'x_max', 'y_max', 'Confidence']];
  lastDetections.forEach(d => {
    const [x, y, w, h] = d.bbox;
    rows.push([d.id, d.glyph, d.label, d.romanized, x, y, x+w, y+h, Math.round(d.confidence * 100) + '%']);
  });

  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = 'baybayin_detections.csv';
  a.click();
  URL.revokeObjectURL(url);
  showToast('CSV exported successfully.', 'success');
});

/* ================================================================
   12. REFERENCE GUIDE — build grids + tab switching
================================================================= */

/** Build a character card element */
function buildCharCard(item) {
  const div = document.createElement('div');
  div.className = 'char-card';
  div.setAttribute('title', `${item.label} (${item.romanized})`);
  div.innerHTML = `
    <span class="char-glyph">${item.glyph}</span>
    <div class="char-label">${item.label}</div>
    <div class="char-romanized">${item.romanized}</div>
    <div class="char-class-id">${item.classId}</div>
  `;
  return div;
}

/** Populate the vowels grid */
function buildVowelsGrid() {
  const container = document.getElementById('vowelsGrid');
  BAYBAYIN_DATA.vowels.forEach(v => container.appendChild(buildCharCard(v)));
}

/** Populate the consonants grid */
function buildConsonantsGrid() {
  const container = document.getElementById('consonantsGrid');
  BAYBAYIN_DATA.consonants.forEach(c => container.appendChild(buildCharCard(c)));
}

/** Populate the kudlit table */
function buildKudlitTable() {
  const tbody = document.getElementById('kudlitTableBody');
  BAYBAYIN_DATA.kudlitBase.forEach(row => {
    const tr = document.createElement('tr');
    tr.className = 'table-row';
    tr.innerHTML = `
      <td class="px-4 py-3 text-amber-400 font-bold">${row.base}</td>
      <td class="px-4 py-3 text-2xl text-stone-200">${row.glyph}</td>
      <td class="px-4 py-3 text-center">
        <span class="text-stone-300 font-medium">${row.a}</span>
      </td>
      <td class="px-4 py-3 text-center">
        <span class="text-amber-400 font-medium">${row.i}</span>
        <span class="text-stone-600 text-xs block">kudlit ↑</span>
      </td>
      <td class="px-4 py-3 text-center">
        <span class="text-sky-400 font-medium">${row.u}</span>
        <span class="text-stone-600 text-xs block">kudlit ↓</span>
      </td>
      <td class="px-4 py-3 text-center">
        <span class="text-stone-500 font-medium">${row.base}—</span>
        <span class="text-stone-600 text-xs block">virama ✕</span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

/** Tab switching logic */
function initRefTabs() {
  const tabs   = document.querySelectorAll('.ref-tab');
  const panels = document.querySelectorAll('.ref-panel');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;

      // Update tab styles
      tabs.forEach(t => {
        t.classList.remove('active-tab');
        t.setAttribute('aria-selected', 'false');
      });
      tab.classList.add('active-tab');
      tab.setAttribute('aria-selected', 'true');

      // Show/hide panels
      panels.forEach(p => p.classList.add('hidden'));
      document.getElementById('tab-' + target).classList.remove('hidden');
    });
  });
}

/* ================================================================
   13. TOAST NOTIFICATIONS
================================================================= */
let toastTimer = null;

/**
 * Display a brief toast message.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 */
function showToast(message, type = 'info') {
  const icons = { success: '✓', error: '✕', info: 'ℹ' };
  toastMsg.textContent  = message;
  toastIcon.textContent = icons[type] || icons.info;

  // Colour coding
  toast.firstElementChild.className = [
    'flex items-center gap-3 text-sm px-5 py-3.5 rounded-xl shadow-2xl',
    type === 'success' ? 'bg-stone-800 border border-emerald-700/50 text-stone-200' :
    type === 'error'   ? 'bg-stone-800 border border-red-700/50 text-stone-200'     :
                         'bg-stone-800 border border-stone-700 text-stone-200'
  ].join(' ');

  toast.classList.remove('hidden', 'hide');

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.add('hide');
    setTimeout(() => toast.classList.add('hidden'), 300);
  }, 3200);
}

/* ================================================================
   14. INIT — run on DOM ready
================================================================= */
document.addEventListener('DOMContentLoaded', () => {
  buildVowelsGrid();
  buildConsonantsGrid();
  buildKudlitTable();
  initRefTabs();

  // Make the file input visible inside dropZone for click-through
  fileInput.classList.remove('hidden');
  fileInput.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer;z-index:1;';
  dropZone.style.position = 'relative';
  dropZone.appendChild(fileInput);
});
