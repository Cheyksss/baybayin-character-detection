/**
 * ================================================================
 * SulatAI - Frontend Application Script
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
  1. DATA - Baybayin syllabary
   ================================================================
  59 classes = 17 base chars × 3 vowel states + 2 standalone
  vowels + virama forms.  Unicode glyphs use the Tagalog Unicode
  block (U+1700-U+171F).
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
  4. NAVBAR - scroll effect
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
  5. IMAGE UPLOAD - drag-and-drop + file picker
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
  showToast('Image ready: click "Process Image" to run detection.', 'success');
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
    showToast(`Detection complete: ${data.detections.length} characters found.`, 'success');

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
    // Backend unreachable - fall back to demo detections
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
    // Backend returned a fully-annotated image - use it directly
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
  // textContent is inherently safe - no HTML parsing, just plain text
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
    // Numeric values are coerced to numbers - not injectable
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
  12. REFERENCE GUIDE - build grids + tab switching
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
        <span class="text-stone-600 text-xs block">kudlit above</span>
      </td>
      <td class="px-4 py-3 text-center">
        <span class="text-sky-400 font-medium">${row.u}</span>
        <span class="text-stone-600 text-xs block">kudlit below</span>
      </td>
      <td class="px-4 py-3 text-center">
        <span class="text-stone-500 font-medium">${row.base}-</span>
        <span class="text-stone-600 text-xs block">virama</span>
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
  const icons = { success: 'OK', error: 'X', info: 'i' };
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
   14. QUIZZES FEATURE
   ================================================================ */

// Curated dictionary of common Filipino words mapped to Baybayin equivalents
const WORD_BANK = [
  { filipino: 'MATA', baybayin: 'ᜋᜆ', difficulty: 'easy' },
  { filipino: 'BATA', baybayin: 'ᜊᜆ', difficulty: 'easy' },
  { filipino: 'TALA', baybayin: 'ᜆᜎ', difficulty: 'easy' },
  { filipino: 'SAYA', baybayin: 'ᜐᜌ', difficulty: 'easy' },
  { filipino: 'DAMA', baybayin: 'ᜇᜋ', difficulty: 'easy' },
  { filipino: 'ULO', baybayin: 'ᜂᜎᜓ', difficulty: 'medium' },
  { filipino: 'BASA', baybayin: 'ᜊᜐ', difficulty: 'easy' },
  { filipino: 'TAPA', baybayin: 'ᜆᜉ', difficulty: 'easy' },
  { filipino: 'PANA', baybayin: 'ᜉᜈ', difficulty: 'easy' },
  { filipino: 'GABI', baybayin: 'ᜄᜊᜒ', difficulty: 'medium' },
  { filipino: 'LAHI', baybayin: 'ᜎᜑᜒ', difficulty: 'medium' },
  { filipino: 'MALI', baybayin: 'ᜋᜎᜒ', difficulty: 'medium' },
  { filipino: 'LOBO', baybayin: 'ᜎᜓᜊᜓ', difficulty: 'medium' },
  { filipino: 'PUSO', baybayin: 'ᜉᜓᜐᜓ', difficulty: 'medium' },
  { filipino: 'KAPE', baybayin: 'ᜃᜉᜒ', difficulty: 'medium' },
  { filipino: 'MURA', baybayin: 'ᜋᜓᜍ', difficulty: 'medium' },
  { filipino: 'KUTA', baybayin: 'ᜃᜓᜆ', difficulty: 'medium' },
  { filipino: 'HALO', baybayin: 'ᜑᜎᜓ', difficulty: 'medium' },
  { filipino: 'WIKA', baybayin: 'ᜏᜒᜃ', difficulty: 'medium' },
  { filipino: 'TAO', baybayin: 'ᜆᜂ', difficulty: 'easy' },
  { filipino: 'AMBO', baybayin: 'ᜀᜋ᜔ᜊᜓ', difficulty: 'hard' },
  { filipino: 'DILAW', baybayin: 'ᜇᜒᜎᜏ᜔', difficulty: 'hard' },
  { filipino: 'BAYANI', baybayin: 'ᜊᜌᜈᜒ', difficulty: 'medium' },
  { filipino: 'ILAW', baybayin: 'ᜁᜎᜏ᜔', difficulty: 'hard' },
  { filipino: 'GATAS', baybayin: 'ᜄᜆᜐ᜔', difficulty: 'hard' },
  { filipino: 'YAMAN', baybayin: 'ᜌᜋᜈ᜔', difficulty: 'hard' },
  { filipino: 'PAGASA', baybayin: 'ᜉᜄᜀᜐ', difficulty: 'easy' },
  { filipino: 'KASAMA', baybayin: 'ᜃᜐᜋ', difficulty: 'easy' }
];

const QUIZ_CHARACTERS = [];

// Initialize all available character items from display mapping
function initQuizCharacters() {
  if (QUIZ_CHARACTERS.length > 0) return;

  for (const [label, glyph] of Object.entries(LABEL_TO_GLYPH)) {
    let difficulty = 'hard';
    const isVowel = ['a', 'e_i', 'o_u'].includes(label);
    const isBaseConsonant = ['ba', 'ka', 'da', 'ga', 'ha', 'la', 'ma', 'na', 'nga', 'pa', 'ra', 'sa', 'ta', 'wa', 'ya'].includes(label);
    
    if (isVowel || isBaseConsonant) {
      difficulty = 'easy';
    } else if (
      label === 'e_i' || 
      label.endsWith('_bi') || 
      label.endsWith('_ki') || 
      label.endsWith('_di') || 
      label.endsWith('_gi') || 
      label.endsWith('_hi') || 
      label.endsWith('_li') || 
      label.endsWith('_mi') || 
      label.endsWith('_ni') || 
      label.endsWith('_ngi') || 
      label.endsWith('_pi') || 
      label.endsWith('_ri') || 
      label.endsWith('_si') || 
      label.endsWith('_ti') || 
      label.endsWith('_wi') || 
      label.endsWith('_yi')
    ) {
      difficulty = 'medium';
    }

    let displayLabel = label.toUpperCase().replace('_', '/');
    if (label.length === 1 && !isVowel) {
      displayLabel = label.toUpperCase() + ' (silent)';
    }

    QUIZ_CHARACTERS.push({
      label,
      glyph,
      displayLabel,
      difficulty,
      isVowel
    });
  }
}

// Fisher-Yates array shuffling
function shuffleArray(array) {
  const arr = [...array];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// Quiz Game Session State
let quizState = {
  active: false,
  mode: null,
  difficulty: null,
  questions: [],
  currentIndex: 0,
  score: 0,
  startTime: null,
  timerInterval: null,
  composition: [],
  hasAnswered: false
};

let activeConsonantIndex = -1;

// DOM Element References for Quiz Section
let quizModeSelectView, quizDifficultySelectView, quizActiveView, quizResultsView;
let quizBackToModeBtn, quizProgressNum, quizLiveScore, quizProgressBar;
let quizActiveTitle, quizActiveDiffLabel, quizQuestionPrompt, quizQuestionMain;
let quizAnswerOptionsContainer, quizAnswerTextContainer, quizTextForm, quizTextInput;
let quizAnswerKeyboardContainer, quizComposedOutput, quizComposedRomanized;
let quizComposedClearBtn, quizComposedBackspaceBtn, kbdVowels, kbdConsonants;
let quizKeyboardSubmitBtn, quizFeedbackAlert, quizQuitBtn, quizNextBtn;
let quizResultsScorePercent, quizResultsStats, quizResultsTimeText, quizResultsAccuracy;
let quizRetryBtn, quizHomeBtn;

function initQuizzes() {
  initQuizCharacters();

  // Wire up DOM elements
  quizModeSelectView = document.getElementById('quizModeSelectView');
  quizDifficultySelectView = document.getElementById('quizDifficultySelectView');
  quizActiveView = document.getElementById('quizActiveView');
  quizResultsView = document.getElementById('quizResultsView');

  quizBackToModeBtn = document.getElementById('quizBackToModeBtn');
  quizProgressNum = document.getElementById('quizProgressNum');
  quizLiveScore = document.getElementById('quizLiveScore');
  quizProgressBar = document.getElementById('quizProgressBar');

  quizActiveTitle = document.getElementById('quizActiveTitle');
  quizActiveDiffLabel = document.getElementById('quizActiveDiffLabel');
  quizQuestionPrompt = document.getElementById('quizQuestionPrompt');
  quizQuestionMain = document.getElementById('quizQuestionMain');

  quizAnswerOptionsContainer = document.getElementById('quizAnswerOptionsContainer');
  quizAnswerTextContainer = document.getElementById('quizAnswerTextContainer');
  quizTextForm = document.getElementById('quizTextForm');
  quizTextInput = document.getElementById('quizTextInput');

  quizAnswerKeyboardContainer = document.getElementById('quizAnswerKeyboardContainer');
  quizComposedOutput = document.getElementById('quizComposedOutput');
  quizComposedRomanized = document.getElementById('quizComposedRomanized');
  quizComposedClearBtn = document.getElementById('quizComposedClearBtn');
  quizComposedBackspaceBtn = document.getElementById('quizComposedBackspaceBtn');
  kbdVowels = document.getElementById('kbdVowels');
  kbdConsonants = document.getElementById('kbdConsonants');
  quizKeyboardSubmitBtn = document.getElementById('quizKeyboardSubmitBtn');

  quizFeedbackAlert = document.getElementById('quizFeedbackAlert');
  quizQuitBtn = document.getElementById('quizQuitBtn');
  quizNextBtn = document.getElementById('quizNextBtn');

  quizResultsScorePercent = document.getElementById('quizResultsScorePercent');
  quizResultsStats = document.getElementById('quizResultsStats');
  quizResultsTimeText = document.getElementById('quizResultsTimeText');
  quizResultsAccuracy = document.getElementById('quizResultsAccuracy');
  quizRetryBtn = document.getElementById('quizRetryBtn');
  quizHomeBtn = document.getElementById('quizHomeBtn');

  // Event Listeners
  document.querySelectorAll('.quiz-mode-card').forEach(card => {
    card.addEventListener('click', () => selectQuizMode(card.dataset.mode));
  });

  document.querySelectorAll('.quiz-diff-btn').forEach(btn => {
    btn.addEventListener('click', () => selectQuizDifficulty(btn.dataset.diff));
  });

  quizBackToModeBtn.addEventListener('click', () => {
    quizDifficultySelectView.classList.add('hidden');
    quizModeSelectView.classList.remove('hidden');
  });

  quizTextForm.addEventListener('submit', handleTextSubmit);
  quizQuitBtn.addEventListener('click', quitQuiz);
  quizNextBtn.addEventListener('click', nextQuestion);
  quizRetryBtn.addEventListener('click', () => selectQuizDifficulty(quizState.difficulty));
  quizHomeBtn.addEventListener('click', goHomeQuiz);

  // Virtual Keyboard controls
  quizComposedClearBtn.addEventListener('click', () => {
    if (quizState.hasAnswered) return;
    quizState.composition = [];
    activeConsonantIndex = -1;
    renderComposition();
    updateModifierButtonHighlight();
  });

  quizComposedBackspaceBtn.addEventListener('click', backspaceComposition);

  document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => {
    btn.addEventListener('click', () => applyModifier(btn.dataset.mod));
  });

  quizKeyboardSubmitBtn.addEventListener('click', handleKeyboardSubmit);

  populateKeyboard();
}

function selectQuizMode(mode) {
  quizState.mode = mode;
  
  // Custom title formatting for difficulty selection header
  const modeTitles = {
    recognition: 'Character Recognition',
    transliteration: 'Transliteration',
    identification: 'Character Identification',
    'filipino-to-baybayin': 'Filipino to Baybayin',
    'baybayin-to-filipino': 'Baybayin to Filipino'
  };
  document.getElementById('selectedModeTitle').textContent = modeTitles[mode] || 'Select Difficulty';

  quizModeSelectView.classList.add('hidden');
  quizDifficultySelectView.classList.remove('hidden');
}

function selectQuizDifficulty(difficulty) {
  quizState.difficulty = difficulty;
  quizDifficultySelectView.classList.add('hidden');
  startQuizSession();
}

function startQuizSession() {
  quizState.active = true;
  quizState.currentIndex = 0;
  quizState.score = 0;
  quizState.startTime = performance.now();
  quizState.hasAnswered = false;
  quizState.questions = generateQuestions(quizState.mode, quizState.difficulty);

  const modeLabels = {
    recognition: 'Character Recognition',
    transliteration: 'Transliteration',
    identification: 'Character Identification',
    'filipino-to-baybayin': 'Filipino to Baybayin',
    'baybayin-to-filipino': 'Baybayin to Filipino'
  };
  quizActiveTitle.textContent = modeLabels[quizState.mode];
  quizActiveDiffLabel.textContent = quizState.difficulty;

  // Visual difficulty badge color-coding
  quizActiveDiffLabel.className = 'text-[10px] px-2 py-0.5 ml-2 rounded-full border font-semibold uppercase ';
  if (quizState.difficulty === 'easy') {
    quizActiveDiffLabel.className += 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  } else if (quizState.difficulty === 'medium') {
    quizActiveDiffLabel.className += 'bg-violet-500/10 text-violet-400 border-violet-500/30';
  } else {
    quizActiveDiffLabel.className += 'bg-rose-500/10 text-rose-400 border-rose-500/30';
  }

  quizResultsView.classList.add('hidden');
  quizActiveView.classList.remove('hidden');
  loadQuestion();
}

function generateQuestions(mode, difficulty) {
  const questions = [];
  
  let charPool = [];
  if (difficulty === 'easy') {
    charPool = QUIZ_CHARACTERS.filter(c => c.difficulty === 'easy');
  } else if (difficulty === 'medium') {
    charPool = QUIZ_CHARACTERS.filter(c => c.difficulty === 'easy' || c.difficulty === 'medium');
  } else {
    charPool = [...QUIZ_CHARACTERS];
  }

  let wordPool = [];
  if (difficulty === 'easy') {
    wordPool = WORD_BANK.filter(w => w.difficulty === 'easy');
  } else if (difficulty === 'medium') {
    wordPool = WORD_BANK.filter(w => w.difficulty === 'easy' || w.difficulty === 'medium');
  } else {
    wordPool = [...WORD_BANK];
  }

  if (charPool.length < 5) charPool = [...QUIZ_CHARACTERS];
  if (wordPool.length < 5) wordPool = [...WORD_BANK];

  if (mode === 'recognition') {
    const pool = shuffleArray(charPool).slice(0, 10);
    pool.forEach(item => {
      const distractors = shuffleArray(charPool.filter(c => c.displayLabel !== item.displayLabel))
        .slice(0, 3)
        .map(c => c.displayLabel);
      const choices = shuffleArray([item.displayLabel, ...distractors]);
      questions.push({
        prompt: 'What is the transliteration of this Baybayin glyph?',
        question: item.glyph,
        choices,
        answer: item.displayLabel
      });
    });
  } else if (mode === 'transliteration') {
    const pool = shuffleArray(charPool).slice(0, 10);
    pool.forEach(item => {
      const distractors = shuffleArray(charPool.filter(c => c.glyph !== item.glyph))
        .slice(0, 3)
        .map(c => c.glyph);
      const choices = shuffleArray([item.glyph, ...distractors]);
      questions.push({
        prompt: 'Which Baybayin glyph matches this syllable?',
        question: item.displayLabel,
        choices,
        answer: item.glyph
      });
    });
  } else if (mode === 'identification') {
    const pool = shuffleArray(charPool).slice(0, 10);
    pool.forEach(item => {
      questions.push({
        prompt: 'Type the romanized transliteration of this glyph (e.g. BA, BI, KA):',
        question: item.glyph,
        answer: item.displayLabel
      });
    });
  } else if (mode === 'filipino-to-baybayin') {
    const pool = shuffleArray(wordPool).slice(0, 10);
    pool.forEach(item => {
      questions.push({
        prompt: 'Translate this Filipino word to Baybayin:',
        question: item.filipino,
        answer: item.baybayin
      });
    });
  } else if (mode === 'baybayin-to-filipino') {
    const pool = shuffleArray(wordPool).slice(0, 10);
    pool.forEach(item => {
      const distractors = shuffleArray(WORD_BANK.filter(w => w.filipino !== item.filipino))
        .slice(0, 3)
        .map(w => w.filipino);
      const choices = shuffleArray([item.filipino, ...distractors]);
      questions.push({
        prompt: 'Translate this Baybayin word to its Filipino meaning:',
        question: item.baybayin,
        choices,
        answer: item.filipino
      });
    });
  }

  return questions;
}

function loadQuestion() {
  quizState.hasAnswered = false;
  quizNextBtn.classList.add('hidden');
  quizFeedbackAlert.classList.add('hidden');
  
  // Clear any inputs
  quizTextInput.value = '';
  quizState.composition = [];
  activeConsonantIndex = -1;
  renderComposition();
  updateModifierButtonHighlight();

  // Disable modifier keys initially since no consonant is selected
  document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => btn.disabled = true);

  const q = quizState.questions[quizState.currentIndex];
  
  // Progress Bar
  const progPct = ((quizState.currentIndex) / quizState.questions.length) * 100;
  quizProgressBar.style.width = `${progPct}%`;
  quizProgressNum.textContent = `Question ${quizState.currentIndex + 1} of ${quizState.questions.length}`;
  quizLiveScore.textContent = `Score: ${quizState.score}`;

  // Prompts & Main Question Content
  quizQuestionPrompt.textContent = q.prompt;
  quizQuestionMain.textContent = q.question;

  // Toggle Visibility by Answer Mode
  quizAnswerOptionsContainer.classList.add('hidden');
  quizAnswerTextContainer.classList.add('hidden');
  quizAnswerKeyboardContainer.classList.add('hidden');

  if (quizState.mode === 'recognition' || quizState.mode === 'transliteration' || quizState.mode === 'baybayin-to-filipino') {
    quizAnswerOptionsContainer.classList.remove('hidden');
    renderOptions(q.choices);
  } else if (quizState.mode === 'identification') {
    quizAnswerTextContainer.classList.remove('hidden');
    setTimeout(() => quizTextInput.focus(), 100);
  } else if (quizState.mode === 'filipino-to-baybayin') {
    quizAnswerKeyboardContainer.classList.remove('hidden');
  }
}

function renderOptions(choices) {
  quizAnswerOptionsContainer.innerHTML = '';
  choices.forEach(opt => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quiz-option-btn w-full px-5 py-4 rounded-xl font-medium flex items-center justify-between group';
    
    // Large text for Baybayin glyphs
    const isGlyph = opt.match(/[\u1700-\u171F]/);
    btn.innerHTML = `
      <span class="${isGlyph ? 'text-2xl leading-none' : 'text-stone-300 font-semibold'}">${escapeHTML(opt)}</span>
      <span class="opacity-0 group-hover:opacity-100 transition-opacity text-xs text-violet-400">Select</span>
    `;

    btn.addEventListener('click', () => handleOptionSelection(opt, btn));
    quizAnswerOptionsContainer.appendChild(btn);
  });
}

function handleOptionSelection(selected, element) {
  if (quizState.hasAnswered) return;
  quizState.hasAnswered = true;

  const q = quizState.questions[quizState.currentIndex];
  const isCorrect = selected === q.answer;

    if (isCorrect) {
    quizState.score++;
    element.classList.add('correct');
    element.querySelector('span:last-child').innerHTML = 'Correct';
    element.querySelector('span:last-child').className = 'text-xs text-emerald-400 opacity-100';
    showQuestionFeedback(true, `Correct! ${q.answer} is indeed the right answer.`);
  } else {
    element.classList.add('incorrect');
    element.querySelector('span:last-child').innerHTML = 'Incorrect';
    element.querySelector('span:last-child').className = 'text-xs text-red-400 opacity-100';

    // Highlight the correct one
    Array.from(quizAnswerOptionsContainer.children).forEach(btn => {
      const text = btn.querySelector('span:first-child').textContent;
        if (text === q.answer) {
        btn.classList.add('correct');
        btn.querySelector('span:last-child').innerHTML = 'Correct Answer';
        btn.querySelector('span:last-child').className = 'text-xs text-emerald-400 opacity-100';
      }
    });

    showQuestionFeedback(false, `Incorrect. The correct answer was ${q.answer}.`);
  }

  // Lock choices
  Array.from(quizAnswerOptionsContainer.children).forEach(btn => btn.disabled = true);
  quizNextBtn.classList.remove('hidden');
}

function handleTextSubmit(e) {
  e.preventDefault();
  if (quizState.hasAnswered) return;

  const inputVal = quizTextInput.value.trim();
  if (!inputVal) return;

  quizState.hasAnswered = true;
  const q = quizState.questions[quizState.currentIndex];
  
  // Smart loose matching check
  const isCorrect = checkIdentificationAnswer(inputVal, q.answer);

  if (isCorrect) {
    quizState.score++;
    quizTextInput.className = 'flex-1 bg-stone-950 border border-emerald-500 rounded-xl px-4 py-3 text-emerald-400 text-center uppercase font-semibold outline-none transition-all';
    showQuestionFeedback(true, `Correct! ${q.answer} is the right transliteration.`);
  } else {
    quizTextInput.className = 'flex-1 bg-stone-950 border border-red-500 rounded-xl px-4 py-3 text-red-400 text-center uppercase font-semibold outline-none transition-all';
    showQuestionFeedback(false, `Incorrect. The correct transliteration was ${q.answer}.`);
  }

  quizNextBtn.classList.remove('hidden');
}

function checkIdentificationAnswer(userAnswer, correctAnswer) {
  const cleanUser = userAnswer.trim().toUpperCase().replace(/[^A-Z]/g, '');
  const cleanCorrect = correctAnswer.trim().toUpperCase();
  
  if (cleanCorrect.includes('/')) {
    const parts = cleanCorrect.split('/').map(p => p.trim().replace(/[^A-Z]/g, ''));
    return parts.includes(cleanUser) || cleanUser === cleanCorrect.replace(/[^A-Z]/g, '');
  }
  return cleanUser === cleanCorrect.replace(/[^A-Z]/g, '');
}

function handleKeyboardSubmit() {
  if (quizState.hasAnswered) return;
  
  // Build glyph string from composition array
  const composedStr = quizState.composition.map(symbol => {
    let glyph = symbol.baseGlyph;
    if (symbol.modifier === 'above') glyph += 'ᜒ';
    else if (symbol.modifier === 'below') glyph += 'ᜓ';
    else if (symbol.modifier === 'virama') glyph += '᜔';
    return glyph;
  }).join('');

  if (composedStr.length === 0) {
    showToast('Please type at least one character before submitting.', 'info');
    return;
  }

  quizState.hasAnswered = true;
  const q = quizState.questions[quizState.currentIndex];
  
  // Exact match required for composed baybayin text
  const isCorrect = composedStr === q.answer;

  if (isCorrect) {
    quizState.score++;
    showQuestionFeedback(true, `Correct! "${composedStr}" is the right Baybayin writing for ${q.question}.`);
  } else {
    showQuestionFeedback(false, `Incorrect. For "${q.question}", correct Baybayin was "${q.answer}". You entered "${composedStr}".`);
  }

  quizNextBtn.classList.remove('hidden');
}

function showQuestionFeedback(isCorrect, feedbackText) {
  quizFeedbackAlert.classList.remove('hidden', 'feedback-success', 'feedback-error');
  if (isCorrect) {
    quizFeedbackAlert.classList.add('feedback-success');
    quizFeedbackAlert.innerHTML = `<span>${escapeHTML(feedbackText)}</span>`;
  } else {
    quizFeedbackAlert.classList.add('feedback-error');
    quizFeedbackAlert.innerHTML = `<span>${escapeHTML(feedbackText)}</span>`;
  }
}

function nextQuestion() {
  quizState.currentIndex++;
  
  if (quizState.currentIndex >= quizState.questions.length) {
    // End session, update progress bar to 100%
    quizProgressBar.style.width = '100%';
    quizLiveScore.textContent = `Score: ${quizState.score}`;
    setTimeout(showQuizResults, 400);
  } else {
    loadQuestion();
  }
}

function quitQuiz() {
  if (confirm('Are you sure you want to quit the current quiz session? Your progress will be lost.')) {
    goHomeQuiz();
  }
}

function showQuizResults() {
  quizState.active = false;
  quizActiveView.classList.add('hidden');
  quizResultsView.classList.remove('hidden');

  const elapsedSecs = Math.round((performance.now() - quizState.startTime) / 1000);
  const mins = Math.floor(elapsedSecs / 60);
  const secs = elapsedSecs % 60;
  const timeStr = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;

  const accuracyPct = Math.round((quizState.score / quizState.questions.length) * 100);

  // Score circular indicator animation
  animateCounter(quizResultsScorePercent, accuracyPct, '%');
  
  quizResultsStats.textContent = `You correctly answered ${quizState.score} out of ${quizState.questions.length} questions.`;
  quizResultsTimeText.textContent = timeStr;
  quizResultsAccuracy.textContent = `${accuracyPct}%`;

  // Color-coded accuracy percentage text color
  quizResultsAccuracy.className = 'text-base font-bold ';
  if (accuracyPct >= 80) quizResultsAccuracy.className += 'text-emerald-400';
  else if (accuracyPct >= 50) quizResultsAccuracy.className += 'text-amber-400';
  else quizResultsAccuracy.className += 'text-red-400';
}

function goHomeQuiz() {
  quizState.active = false;
  quizActiveView.classList.add('hidden');
  quizResultsView.classList.add('hidden');
  quizDifficultySelectView.classList.add('hidden');
  quizModeSelectView.classList.remove('hidden');
}

/* ================================================================
   15. VISUAL KEYBOARD SYSTEM
   ================================================================ */

function populateKeyboard() {
  if (!kbdVowels || !kbdConsonants) return;

  kbdVowels.innerHTML = '';
  BAYBAYIN_DATA.vowels.forEach(v => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'kbd-key';
    btn.innerHTML = `<span class="kbd-glyph">${v.glyph}</span><span class="kbd-label">${v.label}</span>`;
    btn.addEventListener('click', () => addKeyToComposition(v.glyph, v.label, true));
    kbdVowels.appendChild(btn);
  });

  kbdConsonants.innerHTML = '';
  BAYBAYIN_DATA.consonants.forEach(c => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'kbd-key';
    btn.innerHTML = `<span class="kbd-glyph">${c.glyph}</span><span class="kbd-label">${c.label}</span>`;
    btn.addEventListener('click', () => addKeyToComposition(c.glyph, c.label, false));
    kbdConsonants.appendChild(btn);
  });
}

function addKeyToComposition(glyph, label, isVowel) {
  if (quizState.hasAnswered) return;

  quizState.composition.push({
    baseGlyph: glyph,
    baseLabel: label,
    modifier: null,
    isVowel
  });

  // Automatically select the newly typed consonant for modifier operations
  if (!isVowel) {
    activeConsonantIndex = quizState.composition.length - 1;
    // Enable modifier keys
    document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => btn.disabled = false);
  } else {
    activeConsonantIndex = -1;
    // Disable modifier keys as vowels cannot take kudlits
    document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => btn.disabled = true);
  }

  renderComposition();
  updateModifierButtonHighlight();
}

function applyModifier(modifierType) {
  if (quizState.hasAnswered || activeConsonantIndex === -1) return;

  const symbol = quizState.composition[activeConsonantIndex];
  if (symbol.isVowel) return;

  // Toggle modifier state if clicked again
  if (symbol.modifier === modifierType) {
    symbol.modifier = null;
  } else {
    symbol.modifier = modifierType;
  }

  renderComposition();
  updateModifierButtonHighlight();
}

function updateModifierButtonHighlight() {
  document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => {
    btn.classList.remove('active-mod');
  });

  if (activeConsonantIndex !== -1) {
    const symbol = quizState.composition[activeConsonantIndex];
    if (symbol && symbol.modifier) {
      const activeBtn = document.querySelector(`.quiz-kbd-modifier[data-mod="${symbol.modifier}"]`);
      if (activeBtn) {
        activeBtn.classList.add('active-mod');
      }
    }
  }
}

function renderComposition() {
  if (!quizComposedOutput || !quizComposedRomanized) return;
  quizComposedOutput.innerHTML = '';
  
  if (quizState.composition.length === 0) {
    quizComposedOutput.innerHTML = '<span class="text-stone-700 text-lg font-normal">Start composing...</span>';
    quizComposedRomanized.textContent = 'Composed: ';
    return;
  }

  let romanizedParts = [];
  quizState.composition.forEach((symbol, index) => {
    let glyph = symbol.baseGlyph;
    if (symbol.modifier === 'above') glyph += 'ᜒ';
    else if (symbol.modifier === 'below') glyph += 'ᜓ';
    else if (symbol.modifier === 'virama') glyph += '᜔';

    const span = document.createElement('span');
    span.textContent = glyph;
    span.className = 'cursor-pointer hover:text-violet-400 transition-colors mx-0.5 ';
    
    if (index === activeConsonantIndex) {
      span.className += 'text-violet-400 border-b-2 border-violet-500 pb-0.5';
    }

    // Clicking composed glyph changes active selection focus
    span.addEventListener('click', () => {
      if (!symbol.isVowel && !quizState.hasAnswered) {
        activeConsonantIndex = index;
        renderComposition();
        updateModifierButtonHighlight();
      }
    });

    quizComposedOutput.appendChild(span);

    // Compute active transliteration display
    let rom = symbol.baseLabel;
    if (symbol.modifier === 'above') {
      rom = rom.replace(/A$/, 'I');
    } else if (symbol.modifier === 'below') {
      rom = rom.replace(/A$/, 'U');
    } else if (symbol.modifier === 'virama') {
      rom = rom.replace(/A$/, '');
    }
    romanizedParts.push(rom);
  });

  quizComposedRomanized.textContent = 'Composed: ' + romanizedParts.join('-');

  // Reset highlight keyboard keys matching selections
  document.querySelectorAll('.kbd-key').forEach(btn => btn.classList.remove('active-consonant'));
  if (activeConsonantIndex !== -1) {
    const activeSymbol = quizState.composition[activeConsonantIndex];
    document.querySelectorAll('.kbd-key').forEach(btn => {
      const label = btn.querySelector('.kbd-label').textContent;
      if (label === activeSymbol.baseLabel) {
        btn.classList.add('active-consonant');
      }
    });
  }
}

function backspaceComposition() {
  if (quizState.hasAnswered || quizState.composition.length === 0) return;
  
  if (activeConsonantIndex === quizState.composition.length - 1) {
    quizState.composition.pop();
    activeConsonantIndex = -1;
    for (let i = quizState.composition.length - 1; i >= 0; i--) {
      if (!quizState.composition[i].isVowel) {
        activeConsonantIndex = i;
        break;
      }
    }
  } else {
    quizState.composition.pop();
    if (activeConsonantIndex >= quizState.composition.length) {
      activeConsonantIndex = -1;
      for (let i = quizState.composition.length - 1; i >= 0; i--) {
        if (!quizState.composition[i].isVowel) {
          activeConsonantIndex = i;
          break;
        }
      }
    }
  }

  // Update modifiers state after pop
  if (activeConsonantIndex !== -1) {
    document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => btn.disabled = false);
  } else {
    document.querySelectorAll('.quiz-kbd-modifier').forEach(btn => btn.disabled = true);
  }
  
  renderComposition();
  updateModifierButtonHighlight();
}

/* ================================================================
  16. INIT - run on DOM ready
   ================================================================ */
document.addEventListener('DOMContentLoaded', () => {
  buildVowelsGrid();
  buildConsonantsGrid();
  buildKudlitTable();
  initRefTabs();
  initQuizzes();

  // Make the file input visible inside dropZone for click-through
  fileInput.classList.remove('hidden');
  fileInput.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:pointer;z-index:1;';
  dropZone.style.position = 'relative';
  dropZone.appendChild(fileInput);
});
