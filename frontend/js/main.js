/**
 * main.js — Zurra (زرعة) Frontend Application
 *
 * Responsibilities:
 *  1. Initialise and manage Leaflet.js interactive Egypt map with grid overlay.
 *  2. Handle map click events to auto-populate lat/lon form fields.
 *  3. Manage the ML input form and validate user data.
 *  4. Send fetch() requests to the FastAPI prediction endpoint.
 *  5. Render ClimatePredictionResponse in the result card.
 *  6. Maintain in-memory session history with a rendered history list.
 *  7. Drive tab navigation, KPI ribbon updates, and toast notifications.
 *
 * Tech stack: Vanilla JavaScript (ES2020+), Leaflet.js, Tailwind CSS via CDN.
 * No build step or bundler required — runs directly in the browser.
 *
 * Author: Zurra Frontend Team
 * Last Updated: 2026-09-25
 */

'use strict';

/* ============================================================
   § 1  APPLICATION STATE
   ============================================================ */
const AppState = {
  /** Currently selected map coordinates */
  selectedLat: null,
  selectedLon: null,

  /** Leaflet map instance */
  map: null,

  /** Leaflet layer group for grid cells */
  gridLayerGroup: null,

  /** Leaflet layer group for selection markers */
  markerGroup: null,

  /** Whether the grid overlay is visible */
  gridVisible: true,

  /** Active marker on map */
  activeMarker: null,

  /** Session prediction history (newest first) */
  history: [],

  /** Total prediction runs this session */
  totalRuns: 0,
};


/* ============================================================
   § 2  CONSTANTS
   ============================================================ */

/**
 * Egypt geographic bounding box used to constrain
 * the grid overlay and validate coordinate inputs.
 * @type {{ minLat: number, maxLat: number, minLon: number, maxLon: number }}
 */
const EGYPT_BOUNDS = {
  minLat: 22.0,
  maxLat: 31.8,
  minLon: 25.0,
  maxLon: 37.0,
};

/**
 * Default map centre — Cairo, Egypt.
 * @type {[number, number]}
 */
const EGYPT_CENTER = [26.8206, 30.8025];

/**
 * Grid cell sizes (decimal degrees) for each zoom tier.
 *  - Zoom <  8  → 0.5°  (~55 km) — country overview
 *  - Zoom 8–9   → 0.2°  (~22 km) — governorate level
 *  - Zoom ≥ 10  → 0.02° (~ 2 km) — micro-zone / local district level
 */
const GRID_CELL_DEG = 0.02;          // finest resolution (used at zoom ≥ 10)
const GRID_STEP_COARSE  = 0.5;       // zoom < 8
const GRID_STEP_MID     = 0.2;       // zoom 8–9
const GRID_STEP_FINE    = 0.02;      // zoom ≥ 10  (~2 km × 2 km cells)

/**
 * Temperature colour thresholds (°C).
 * Used for the heat-bar needle and CSS class selection.
 */
const TEMP_THRESHOLDS = {
  mild: 30,
  warm: 36,
  hot: 42,
};

/** Range for the heat-bar gauge (°C) */
const HEAT_BAR_MIN = 20;
const HEAT_BAR_MAX = 48;


/* ============================================================
   § 3  DOM REFERENCES
   ============================================================ */
const DOM = {
  // Map
  leafletMap:    document.getElementById('leaflet-map'),
  coordOverlay:  document.getElementById('coord-overlay'),
  coordText:     document.getElementById('coord-text'),
  mapHint:       document.getElementById('map-hint'),
  gridToggleBtn: document.getElementById('grid-toggle-btn'),
  resetViewBtn:  document.getElementById('reset-view-btn'),

  // Location badge
  badgeNoSel:  document.getElementById('badge-no-sel'),
  badgeCoords: document.getElementById('badge-coords'),

  // Form inputs
  form:           document.getElementById('prediction-form'),
  inpLat:         document.getElementById('inp-lat'),
  inpLon:         document.getElementById('inp-lon'),
  inpGreenArea:   document.getElementById('inp-greenery-area'),
  inpGreenDens:   document.getElementById('inp-greenery-density'),
  inpBuildArea:   document.getElementById('inp-building-area'),
  inpBuildDens:   document.getElementById('inp-building-density'),
  inpAvgLevels:   document.getElementById('inp-avg-levels'),
  inpRoadDens:    document.getElementById('inp-road-density'),
  inpDistWater:   document.getElementById('inp-distance-water'),
  inpPoiDens:     document.getElementById('inp-poi-density'),
  inpBareGround:  document.getElementById('inp-bare-ground'),
  inpNdvi:        document.getElementById('inp-ndvi'),
  inpElevation:   document.getElementById('inp-elevation'),
  inpNightLights: document.getElementById('inp-night-lights'),
  inpApiUrl:      document.getElementById('inp-api-url'),
  predictBtn:     document.getElementById('predict-btn'),

  // Result panel
  resultPanel:  document.getElementById('result-panel'),
  resultTemp:   document.getElementById('result-temp'),
  heatNeedle:   document.getElementById('heat-needle'),
  resStatus:    document.getElementById('res-status'),
  resTarget:    document.getElementById('res-target'),
  resCooling:   document.getElementById('res-cooling'),
  resCoords:    document.getElementById('res-coords'),
  resultTs:     document.getElementById('result-ts'),

  // KPI ribbon
  kpiLastTemp:    document.getElementById('kpi-last-temp'),
  kpiTotalRuns:   document.getElementById('kpi-total-runs'),
  kpiNdvi:        document.getElementById('kpi-ndvi'),
  kpiGreenArea:   document.getElementById('kpi-green-area'),
  kpiActiveCell:  document.getElementById('kpi-active-cell'),

  // Toast
  toast:     document.getElementById('toast'),
  toastMsg:  document.getElementById('toast-msg'),
  toastIcon: document.getElementById('toast-icon'),

  // History
  historyList:     document.getElementById('history-list'),
  clearHistoryBtn: document.getElementById('clear-history-btn'),

  // Nav & tabs
  navBtns:  document.querySelectorAll('.nav-btn'),
  tabBtns:  document.querySelectorAll('.tab-btn'),
  panelViews: document.querySelectorAll('.panel-view'),
};


/* ============================================================
   § 4  MAP INITIALISATION
   ============================================================ */

/**
 * Initialises the Leaflet map centred on Egypt with
 * OpenStreetMap tiles, zoom controls, and event handlers.
 *
 * NOTE: Leaflet requires HTTP(S). When opened from file://, browsers block
 * CDN scripts due to CORS/CSP restrictions. Serve with:
 *   python -m http.server 5500   (from the frontend/ directory)
 *   or use VS Code Live Server extension.
 */
function initMap() {
  // Guard: Leaflet not available (file:// protocol or CDN blocked)
  if (typeof L === 'undefined') {
    const container = document.getElementById('map-container');
    if (container) {
      container.innerHTML = `
        <div style="
          display:flex;flex-direction:column;align-items:center;justify-content:center;
          height:100%;gap:14px;color:var(--text-secondary);text-align:center;padding:24px;
        ">
          <div style="font-size:2.5rem;">🗺️</div>
          <div style="font-size:0.95rem;font-weight:600;color:var(--green-light);">Map Requires a Web Server</div>
          <div style="font-size:0.8rem;max-width:360px;line-height:1.7;color:var(--text-muted);">
            Leaflet cannot load over <code style="color:var(--blue-light);">file://</code>. 
            Serve this folder over HTTP to enable the interactive map:
          </div>
          <pre style="background:var(--bg-elevated);border:1px solid var(--bg-border);border-radius:8px;padding:10px 18px;font-size:0.78rem;color:var(--text-primary);">cd d:/ITI/Zar3a/frontend
python -m http.server 5500</pre>
          <div style="font-size:0.76rem;color:var(--text-muted);">Then open <code style="color:var(--blue-light);">http://localhost:5500</code></div>
          <div style="margin-top:8px;font-size:0.74rem;color:var(--text-muted);">
            💡 Or use <strong>VS Code Live Server</strong> extension (right-click index.html → Open with Live Server)
          </div>
        </div>`;
    }
    console.warn('[Zurra] Leaflet (L) is not defined. Serve frontend over HTTP for map functionality.');
    return;
  }

  // Create map instance
  AppState.map = L.map('leaflet-map', {
    center: EGYPT_CENTER,
    zoom: 7,
    zoomControl: true,
    attributionControl: true,
    maxBounds: [
      [15, 20],  // SW corner — slightly outside Egypt
      [35, 42],  // NE corner
    ],
    maxBoundsViscosity: 0.85,
  });

  // OpenStreetMap tile layer
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18,
    minZoom: 4,
    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(AppState.map);

  // Initialise layer groups
  AppState.gridLayerGroup = L.layerGroup().addTo(AppState.map);
  AppState.markerGroup = L.layerGroup().addTo(AppState.map);

  // Draw the initial grid overlay (zoom-aware)
  drawGridOverlay();

  // Redraw grid whenever the user zooms (step size changes by tier)
  AppState.map.on('zoomend', drawGridOverlay);

  // Register click handler to extract coordinates
  AppState.map.on('click', onMapClick);

  // Update coord overlay on mouse move
  AppState.map.on('mousemove', onMapMouseMove);
}

/**
 * Draws a zoom-adaptive transparent rectangle grid overlay over Egypt.
 *
 * Resolution tiers (to keep DOM count manageable):
 *  zoom < 8   → 0.5°  (~55 km cells)  — country overview
 *  zoom 8–9   → 0.2°  (~22 km cells)  — governorate level
 *  zoom ≥ 10  → 0.05° (~ 5 km cells)  — district / local level
 *
 * At zoom ≥ 8 the grid is clipped to the current viewport bounds so only
 * visible cells are added to the DOM, keeping performance smooth.
 */
function drawGridOverlay() {
  AppState.gridLayerGroup.clearLayers();

  if (!AppState.map) return;

  const zoom = AppState.map.getZoom();

  // ── Choose step size based on zoom tier
  let step;
  if (zoom < 8) {
    step = GRID_STEP_COARSE;   // 0.5° — low-zoom country view
  } else if (zoom < 10) {
    step = GRID_STEP_MID;      // 0.2° — governorate view
  } else {
    step = GRID_STEP_FINE;     // 0.05° — local/district view
  }

  // ── Determine the render bbox:
  //    At low zoom render entire Egypt; at higher zoom clip to viewport.
  let minLat, maxLat, minLon, maxLon;
  if (zoom < 8) {
    // Whole Egypt — coarse grid has few cells so it's fine
    ({ minLat, maxLat, minLon, maxLon } = EGYPT_BOUNDS);
  } else {
    // Clip to current viewport + 1 cell padding to avoid edge gaps
    const vb = AppState.map.getBounds();
    const pad = step;
    minLat = Math.max(EGYPT_BOUNDS.minLat, Math.floor((vb.getSouth() - pad) / step) * step);
    maxLat = Math.min(EGYPT_BOUNDS.maxLat, Math.ceil ((vb.getNorth() + pad) / step) * step);
    minLon = Math.max(EGYPT_BOUNDS.minLon, Math.floor((vb.getWest()  - pad) / step) * step);
    maxLon = Math.min(EGYPT_BOUNDS.maxLon, Math.ceil ((vb.getEast()  + pad) / step) * step);
  }

  // ── Also re-draw on map pan when at mid/fine zoom
  //    (register once; guard with flag to avoid duplicate listeners)
  if (zoom >= 8 && !AppState._gridMoveListenerActive) {
    AppState._gridMoveListenerActive = true;
    AppState.map.on('moveend', drawGridOverlay);
  } else if (zoom < 8 && AppState._gridMoveListenerActive) {
    AppState._gridMoveListenerActive = false;
    AppState.map.off('moveend', drawGridOverlay);
  }

  // ── Build cell rectangles
  for (let lat = minLat; lat < maxLat - 1e-9; lat = +(lat + step).toFixed(6)) {
    for (let lon = minLon; lon < maxLon - 1e-9; lon = +(lon + step).toFixed(6)) {
      const cellLat = lat;
      const cellLon = lon;

      const bounds = [
        [cellLat,        cellLon],
        [cellLat + step, cellLon + step],
      ];

      const rect = L.rectangle(bounds, {
        color: 'rgba(46, 90, 68, 0.55)',
        weight: zoom >= 10 ? 0.5 : 0.8,
        fill: true,
        fillColor: 'rgba(46, 90, 68, 0.04)',
        fillOpacity: 1,
        interactive: true,
      });

      // Hover styling
      rect.on('mouseover', function () {
        this.setStyle({
          fillColor: 'rgba(59, 130, 246, 0.12)',
          color: 'rgba(59, 130, 246, 0.7)',
          weight: zoom >= 10 ? 0.8 : 1.2,
        });
      });
      rect.on('mouseout', function () {
        this.setStyle({
          fillColor: 'rgba(46, 90, 68, 0.04)',
          color: 'rgba(46, 90, 68, 0.55)',
          weight: zoom >= 10 ? 0.5 : 0.8,
        });
      });

      // Cell click: centre of cell → selected coords
      rect.on('click', function (e) {
        L.DomEvent.stopPropagation(e);
        const centreLat = +(cellLat + step / 2).toFixed(4);
        const centreLon = +(cellLon + step / 2).toFixed(4);
        applySelectedCoords(centreLat, centreLon);
      });

      AppState.gridLayerGroup.addLayer(rect);
    }
  }
}

/**
 * Handles a direct map click (not on a grid cell).
 * Extracts lat/lon from the Leaflet LatLng event.
 * @param {L.LeafletMouseEvent} e
 */
function onMapClick(e) {
  const lat = +e.latlng.lat.toFixed(6);
  const lon = +e.latlng.lng.toFixed(6);

  // Clamp to Egypt bounds
  if (lat < EGYPT_BOUNDS.minLat || lat > EGYPT_BOUNDS.maxLat ||
      lon < EGYPT_BOUNDS.minLon || lon > EGYPT_BOUNDS.maxLon) {
    showToast('⚠️ Please click within Egypt\'s geographic boundaries.', 'warning');
    return;
  }

  applySelectedCoords(lat, lon);
}

/**
 * Live coordinate display on mouse move (floating overlay).
 * @param {L.LeafletMouseEvent} e
 */
function onMapMouseMove(e) {
  const lat = e.latlng.lat.toFixed(4);
  const lon = e.latlng.lng.toFixed(4);
  DOM.coordText.textContent = `Lat ${lat}  Lon ${lon}`;
}

/**
 * Applies selected coordinates to the form, KPI ribbon,
 * badge, and drops a marker with popup on the map.
 * @param {number} lat
 * @param {number} lon
 */
function applySelectedCoords(lat, lon) {
  AppState.selectedLat = lat;
  AppState.selectedLon = lon;

  // ── Update form fields
  DOM.inpLat.value = lat;
  DOM.inpLon.value = lon;

  // ── Update coordinate overlay
  DOM.coordText.textContent = `Lat ${lat}  Lon ${lon}`;
  DOM.coordOverlay.classList.add('active');

  // ── Update location badge
  DOM.badgeNoSel.style.display = 'none';
  DOM.badgeCoords.style.display = 'block';
  DOM.badgeCoords.innerHTML =
    `<strong>Lat:</strong> ${lat}<br><strong>Lon:</strong> ${lon}`;

  // ── Update KPI active cell
  DOM.kpiActiveCell.textContent = `${lat}, ${lon}`;

  // ── Hide map hint
  DOM.mapHint.classList.add('hidden');

  // ── Place animated circle marker
  AppState.markerGroup.clearLayers();

  const marker = L.circleMarker([lat, lon], {
    radius: 8,
    color: '#3B82F6',
    fillColor: '#60A5FA',
    fillOpacity: 0.85,
    weight: 2.5,
  });

  marker.bindPopup(
    `<div style="min-width:140px;">
       <div style="font-weight:700;color:#60A5FA;margin-bottom:4px;">📍 Selected Cell</div>
       <div style="font-family:'JetBrains Mono',monospace;font-size:0.78rem;">
         Lat: <strong>${lat}</strong><br>
         Lon: <strong>${lon}</strong>
       </div>
       <div style="margin-top:8px;font-size:0.7rem;color:#9DBDAA;">
         Click <em>Run Climate Prediction</em> →
       </div>
     </div>`,
    { maxWidth: 200 }
  ).openPopup();

  AppState.markerGroup.addLayer(marker);
  AppState.activeMarker = marker;

  // Auto-switch to prediction tab
  switchTab('prediction');
}


/* ============================================================
   § 5  MAP CONTROLS
   ============================================================ */

/** Toggles grid layer visibility */
function initMapControls() {
  if (!AppState.map) return; // Leaflet unavailable

  DOM.gridToggleBtn.addEventListener('click', () => {
    AppState.gridVisible = !AppState.gridVisible;
    if (AppState.gridVisible) {
      AppState.gridLayerGroup.addTo(AppState.map);
      DOM.gridToggleBtn.style.color = 'var(--text-secondary)';
      DOM.gridToggleBtn.textContent = '⊞ Grid';
    } else {
      AppState.map.removeLayer(AppState.gridLayerGroup);
      DOM.gridToggleBtn.style.color = 'var(--green-light)';
      DOM.gridToggleBtn.textContent = '⊠ Grid Off';
    }
  });

  DOM.resetViewBtn.addEventListener('click', () => {
    AppState.map.setView(EGYPT_CENTER, 6);
  });
}


/* ============================================================
   § 6  TAB NAVIGATION
   ============================================================ */

/**
 * Switches the active panel tab and shows the matching view.
 * @param {'prediction'|'info'|'history'} tabName
 */
function switchTab(tabName) {
  DOM.tabBtns.forEach(btn => {
    const isActive = btn.dataset.tab === tabName;
    btn.classList.toggle('active', isActive);
    btn.setAttribute('aria-selected', isActive.toString());
  });

  DOM.panelViews.forEach(view => {
    const isActive = view.id === `view-${tabName}`;
    view.classList.toggle('active', isActive);
  });
}

/** Initialises tab button click handlers */
function initTabs() {
  DOM.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

/** Initialises the top navbar (supports section scrolling and side tab switching) */
function initNavbar() {
  DOM.navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      DOM.navBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      const scrollTarget = btn.dataset.scroll;
      const viewTarget = btn.dataset.view;

      if (scrollTarget) {
        const el = document.getElementById(scrollTarget);
        if (el) {
          el.scrollIntoView({ behavior: 'smooth' });
        }
      } else if (viewTarget) {
        switchTab(viewTarget);
        document.getElementById('side-panel')?.scrollIntoView({ behavior: 'smooth' });
      }
    });
  });
}


/* ============================================================
   § 7  FORM VALIDATION
   ============================================================ */

/**
 * Collects and validates all form fields.
 * Returns a payload object matching ClimatePredictionRequest or null on error.
 * @returns {{ payload: Object, apiUrl: string }|null}
 */
function collectAndValidateForm() {
  const lat = parseFloat(DOM.inpLat.value);
  const lon = parseFloat(DOM.inpLon.value);

  if (isNaN(lat) || lat < 20 || lat > 35) {
    showToast('❌ Latitude must be between 20° and 35° N (Egypt). Click the map to select.', 'error');
    DOM.inpLat.focus();
    return null;
  }
  if (isNaN(lon) || lon < 24 || lon > 38) {
    showToast('❌ Longitude must be between 24° and 38° E (Egypt). Click the map to select.', 'error');
    DOM.inpLon.focus();
    return null;
  }

  /**
   * Helper: parse a float field with a fallback default.
   * @param {HTMLInputElement} el
   * @param {number} fallback
   * @returns {number}
   */
  const f = (el, fallback = 0) => {
    const v = parseFloat(el.value);
    return isNaN(v) ? fallback : v;
  };

  const payload = {
    lat,
    lon,
    greenery_area:               f(DOM.inpGreenArea,   5000),
    greenery_density:            f(DOM.inpGreenDens,   0.20),
    building_area:               f(DOM.inpBuildArea,  25000),
    building_density:            f(DOM.inpBuildDens,   0.40),
    avg_building_levels:         f(DOM.inpAvgLevels,     4),
    road_density:                f(DOM.inpRoadDens,    0.05),
    distance_to_water:           f(DOM.inpDistWater,  1200),
    poi_density:                 f(DOM.inpPoiDens,    0.005),
    bare_ground_density:         f(DOM.inpBareGround,  0.20),
    ndvi_mean:                   f(DOM.inpNdvi,        0.18),
    elevation:                   f(DOM.inpElevation,     65),
    nighttime_lights_intensity:  f(DOM.inpNightLights, 22.5),
  };

  const apiUrl = (DOM.inpApiUrl.value || '').trim() ||
    'http://localhost:8000/api/v1/climate/predictions';

  return { payload, apiUrl };
}


/* ============================================================
   § 8  API CALL — FETCH PREDICTION
   ============================================================ */

/**
 * Sends the ML feature payload to the FastAPI backend,
 * handles loading state, and processes the response.
 * Endpoint: POST /api/v1/climate/predictions
 *
 * Expected response body (ClimatePredictionResponse):
 * {
 *   target_variable:       string,
 *   predicted_mean_temp_c: number,
 *   cooling_potential_c:   number | null,
 *   features:              Object,
 *   status:                string,
 * }
 *
 * @param {Object} payload   - ClimatePredictionRequest body
 * @param {string} apiUrl    - Full endpoint URL
 */
async function runPrediction(payload, apiUrl) {
  setLoadingState(true);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000);

    const response = await fetch(apiUrl, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      let detail = `HTTP ${response.status} — ${response.statusText}`;
      try {
        const errBody = await response.json();
        detail = errBody.detail || detail;
      } catch (_) { /* ignore JSON parse failure */ }
      throw new Error(detail);
    }

    /** @type {{ target_variable: string, predicted_mean_temp_c: number, cooling_potential_c: number|null, features: Object, status: string }} */
    const data = await response.json();

    renderResult(data, payload);
    addToHistory(data, payload);
    showToast('✅ Prediction complete (Live ML Model)!', 'success');

  } catch (err) {
    console.error('[Zar3a ML] Live prediction API failed:', err);
    showToast(`⚠️ Prediction request failed: ${err.message}`, 'error');
  } finally {
    setLoadingState(false);
  }
}


/* ============================================================
   § 9  RESULT RENDERING
   ============================================================ */

/**
 * Renders a successful prediction result in the result card.
 * @param {{ target_variable: string, predicted_mean_temp_c: number, cooling_potential_c: number|null, status: string }} data
 * @param {{ lat: number, lon: number }} payload
 */
function renderResult(data, payload) {
  const temp = data.predicted_mean_temp_c ?? data.predicted_value ?? 0;
  const tempRounded = +temp.toFixed(2);

  // ── Temperature class
  let tempClass = 'mild';
  if (tempRounded >= TEMP_THRESHOLDS.hot)  tempClass = 'hot';
  else if (tempRounded >= TEMP_THRESHOLDS.warm) tempClass = 'warm';

  // ── Update hero display
  DOM.resultTemp.textContent = tempRounded;
  DOM.resultTemp.className = `temp-value ${tempClass}`;

  // ── Update heat-bar needle position (clamped 0–100%)
  const needlePct = Math.min(100, Math.max(0,
    ((tempRounded - HEAT_BAR_MIN) / (HEAT_BAR_MAX - HEAT_BAR_MIN)) * 100
  ));
  DOM.heatNeedle.style.left = `${needlePct}%`;

  // ── Metadata rows
  const statusVal = data.status || 'success';
  DOM.resStatus.innerHTML =
    `<span class="badge ${statusVal === 'success' ? 'success' : 'error'}">${statusVal}</span>`;
  DOM.resTarget.textContent = data.target_variable || 'mean_temperature';
  DOM.resCooling.textContent = data.cooling_potential_c != null
    ? `${(+data.cooling_potential_c).toFixed(2)} °C`
    : 'N/A';
  DOM.resCoords.textContent = `${payload.lat}, ${payload.lon}`;
  DOM.resultTs.textContent = new Date().toLocaleTimeString();

  // ── KPI updates
  DOM.kpiLastTemp.textContent = `${tempRounded}°`;
  DOM.kpiNdvi.textContent = payload.ndvi_mean?.toFixed(2) ?? '—';
  DOM.kpiGreenArea.textContent = payload.greenery_area
    ? `${(payload.greenery_area / 1000).toFixed(1)}k`
    : '—';

  // ── Show result panel
  DOM.resultPanel.classList.add('visible');
  DOM.resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Renders an error state in the result card when the API call fails.
 * @param {string} message
 * @param {{ lat: number, lon: number }} payload
 */
function renderErrorResult(message, payload) {
  DOM.resultTemp.textContent = 'ERR';
  DOM.resultTemp.className = 'temp-value hot';
  DOM.heatNeedle.style.left = '100%';
  DOM.resStatus.innerHTML = '<span class="badge error">error</span>';
  DOM.resTarget.textContent = '—';
  DOM.resCooling.textContent = '—';
  DOM.resCoords.textContent = `${payload.lat ?? '?'}, ${payload.lon ?? '?'}`;
  DOM.resultTs.textContent = new Date().toLocaleTimeString();
  DOM.resultPanel.classList.add('visible');
}


/* ============================================================
   § 10  HISTORY
   ============================================================ */

/**
 * Adds a completed prediction to the session history array
 * and re-renders the history list.
 * @param {Object} data - API response
 * @param {Object} payload - request payload
 */
function addToHistory(data, payload) {
  AppState.totalRuns++;
  DOM.kpiTotalRuns.textContent = AppState.totalRuns;

  const entry = {
    id:      AppState.totalRuns,
    temp:    +(data.predicted_mean_temp_c ?? data.predicted_value ?? 0).toFixed(2),
    lat:     payload.lat,
    lon:     payload.lon,
    ndvi:    payload.ndvi_mean,
    cooling: data.cooling_potential_c,
    ts:      new Date().toLocaleTimeString(),
  };

  AppState.history.unshift(entry); // Prepend — newest first
  renderHistoryList();
}

/** Re-renders the full history list from AppState.history */
function renderHistoryList() {
  if (AppState.history.length === 0) {
    DOM.historyList.innerHTML =
      '<div class="history-empty">No predictions yet. Run your first inference above.</div>';
    return;
  }

  DOM.historyList.innerHTML = AppState.history.map((entry, i) => {
    let tempClass = 'mild';
    if (entry.temp >= TEMP_THRESHOLDS.hot)  tempClass = 'hot';
    else if (entry.temp >= TEMP_THRESHOLDS.warm) tempClass = 'warm';

    return `
      <div class="history-item" data-entry-id="${entry.id}"
           style="animation-delay: ${i * 0.04}s"
           title="Click to reload this prediction's coordinates">
        <div class="history-temp ${tempClass}">${entry.temp}°</div>
        <div class="history-details">
          <div class="history-coords">Lat ${entry.lat} &nbsp;|&nbsp; Lon ${entry.lon}</div>
          <div class="history-coords">NDVI: ${(entry.ndvi ?? 0).toFixed(2)}</div>
        </div>
        <div class="history-time">${entry.ts}</div>
      </div>
    `;
  }).join('');

  // Make history items clickable to reload coordinates on the map
  DOM.historyList.querySelectorAll('.history-item').forEach(item => {
    item.addEventListener('click', () => {
      const entryId = parseInt(item.dataset.entryId, 10);
      const entry = AppState.history.find(e => e.id === entryId);
      if (entry) {
        applySelectedCoords(entry.lat, entry.lon);
        AppState.map.setView([entry.lat, entry.lon], 10);
        switchTab('prediction');
      }
    });
  });
}

/** Clears the history list */
function initHistoryClear() {
  DOM.clearHistoryBtn.addEventListener('click', () => {
    AppState.history = [];
    AppState.totalRuns = 0;
    DOM.kpiTotalRuns.textContent = 0;
    renderHistoryList();
  });
}


/* ============================================================
   § 11  FORM SUBMISSION
   ============================================================ */

/** Attaches the submit handler to the prediction form */
function initForm() {
  DOM.form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const validated = collectAndValidateForm();
    if (!validated) return;

    const { payload, apiUrl } = validated;
    await runPrediction(payload, apiUrl);
  });
}


/* ============================================================
   § 12  LOADING STATE
   ============================================================ */

/**
 * Enables or disables the loading state on the predict button.
 * @param {boolean} loading
 */
function setLoadingState(loading) {
  const btn = DOM.predictBtn;
  if (loading) {
    btn.disabled = true;
    btn.classList.add('loading');
    btn.querySelector('.btn-text').textContent = 'Running inference…';
  } else {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = '🔬 Run Climate Prediction';
  }
}


/* ============================================================
   § 13  TOAST NOTIFICATIONS
   ============================================================ */

/** @type {ReturnType<typeof setTimeout>} */
let toastTimer = null;

/**
 * Shows a temporary toast message.
 * @param {string} message        - Text to display
 * @param {'error'|'success'|'warning'} type
 * @param {number} [duration=4000] - Auto-dismiss delay in ms
 */
function showToast(message, type = 'error', duration = 4000) {
  clearTimeout(toastTimer);

  const icons = { error: '❌', success: '✅', warning: '⚠️' };
  DOM.toastIcon.textContent = icons[type] ?? '⚠️';
  DOM.toastMsg.textContent = message;

  DOM.toast.classList.remove('success-toast');
  if (type === 'success') DOM.toast.classList.add('success-toast');

  DOM.toast.classList.add('visible');
  toastTimer = setTimeout(() => DOM.toast.classList.remove('visible'), duration);
}


/* ============================================================
   § 14  API STATUS CHECK
   ============================================================ */

/**
 * Performs a lightweight health check against the backend.
 * Updates the status dot in the topbar.
 */
async function checkApiStatus() {
  const apiUrl = DOM.inpApiUrl.value.trim();
  // Derive base URL from the prediction endpoint
  const baseUrl = apiUrl.split('/api/')[0];
  const healthUrl = `${baseUrl}/health`;

  const dot = document.getElementById('api-dot');
  const label = document.getElementById('api-status-label');

  try {
    const res = await fetch(healthUrl, { method: 'GET', signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      dot.style.background = 'var(--success)';
      label.textContent = 'API Online';
    } else {
      throw new Error('Non-200');
    }
  } catch (_) {
    dot.style.background = 'var(--error)';
    dot.style.animation = 'none';
    label.textContent = 'API Offline';
  }
}


/* ============================================================
   § 15  ENTRY POINT — DOMContentLoaded
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
  console.log('[Zurra] Initialising dashboard…');

  // 1. Map
  initMap();
  initMapControls();

  // 2. Navigation & tabs
  initTabs();
  initNavbar();

  // 3. Form & history
  initForm();
  initHistoryClear();

  // 4. Initial history render (empty state)
  renderHistoryList();

  // 5. Recommendations module
  initRecommendations();

  // 6. Chatbot module
  initChatbot();

  // 7. Background API health check (non-blocking)
  setTimeout(checkApiStatus, 1200);
  setInterval(checkApiStatus, 60_000);

  console.log('[Zurra] Dashboard ready ✅');
});


/* ============================================================
   § 16  RECOMMENDATIONS MODULE
   ============================================================ */

/**
 * Authentic Arabic common names mapping for tree species
 * to ensure crisp bilingual display without character corruption.
 */
const TREE_AR_MAP = {
  "Sidr (Christ's Thorn Jujube)": "سدر (نبق)",
  "Acacia saligna": "أكاسيا ساليجنا",
  "Acacia nilotica": "سنط عربي (نيلوتيكا)",
  "Conocarpus erectus": "كونوكاربوس (دماس)",
  "Tamarix aphylla": "أثل أفرع",
  "Aleppo Pine": "صنوبر حلبي",
  "Casuarina equisetifolia": "كزوارينا (مخروطية)",
  "Ficus retusa": "فيكس نتدا",
  "Ficus nitida": "فيكس رخي (نتدا)",
  "Ficus benjamina": "فيكس بنجامينا",
  "Jacaranda mimosifolia": "جاكاراندا زرقاء",
  "Delonix regia": "بوانسيانا ملكية",
  "Azadirachta indica": "نيم هندي",
  "Morus alba": "توت أبيض",
  "Olea europaea": "زيتون",
  "Citrus aurantium": "نارنج (لارنج)",
  "Phoenix dactylifera": "نخيل بلح",
  "Cassia nodosa": "كاسيا نودوزا",
  "Cassia fistula": "خيار شمبر",
  "Eucalyptus camaldulensis": "كافور بلدي",
  "Tipuana tipu": "تيبوانا",
  "Bauhinia purpurea": "خف الجمل (بوهينيا)",
  "Schinus terebinthifolius": "فلفل عريض الأوراق",
};

/**
 * Fallback tree species dataset when backend is unreachable
 */
const FALLBACK_SPECIES = [
  {
    tree_id: 1,
    name_en: "Sidr (Christ's Thorn Jujube)",
    name_ar: "سدر (نبق)",
    category: "Fruit / Shade",
    water_requirement: "Low",
    root_type: "Deep Taproot",
    growth_rate: "Moderate",
    cooling_effect_score: 7,
    air_purification_score: 8,
    canopy_spread_m: 8.5,
    carbon_sequestration_kg_yr: 32,
    final_score: 16.45,
  },
  {
    tree_id: 29,
    name_en: "Aleppo Pine",
    name_ar: "صنوبر حلبي",
    category: "Shade / Evergreen",
    water_requirement: "Low",
    root_type: "Deep",
    growth_rate: "Fast",
    cooling_effect_score: 6,
    air_purification_score: 9,
    canopy_spread_m: 9.0,
    carbon_sequestration_kg_yr: 38,
    final_score: 15.32,
  },
  {
    tree_id: 37,
    name_en: "Casuarina equisetifolia",
    name_ar: "كزوارينا (مخروطية)",
    category: "Windbreak / Shade",
    water_requirement: "Low",
    root_type: "Deep",
    growth_rate: "Fast",
    cooling_effect_score: 6,
    air_purification_score: 8,
    canopy_spread_m: 6.5,
    carbon_sequestration_kg_yr: 45,
    final_score: 14.80,
  },
  {
    tree_id: 12,
    name_en: "Azadirachta indica",
    name_ar: "نيم هندي",
    category: "Shade / Medicinal",
    water_requirement: "Low",
    root_type: "Deep",
    growth_rate: "Fast",
    cooling_effect_score: 8,
    air_purification_score: 9,
    canopy_spread_m: 10.0,
    carbon_sequestration_kg_yr: 40,
    final_score: 17.10,
  }
];

/**
 * Recommendation state — tracks user decisions on cards.
 * Key: name_en string, Value: 'approved' | 'modified' | 'rejected'
 */
const RecState = {
  decisions: new Map(),
  lastResults: [],
  currentCriteria: null,
  mapCanopyLayer: null,
};

/**
 * Initialises the Recommendations box:
 * - Wires up Run and Reset buttons.
 */
function initRecommendations() {
  const runBtn    = document.getElementById('rec-run-btn');
  const resetBtn  = document.getElementById('rec-reset-btn');

  if (!runBtn) return;

  runBtn.addEventListener('click', fetchRecommendations);
  if (resetBtn) resetBtn.addEventListener('click', resetRecommendations);
}

/**
 * Fetches recommendations from FastAPI and renders species cards + diff view.
 */
async function fetchRecommendations() {
  const runBtn   = document.getElementById('rec-run-btn');
  const spinner  = document.getElementById('rec-spinner');
  const water    = document.getElementById('rec-water')?.value || 'Low';
  const cooling  = parseInt(document.getElementById('rec-cooling')?.value || '6', 10);
  const topn     = parseInt(document.getElementById('rec-topn')?.value || '3', 10);
  const narrow   = document.getElementById('rec-narrow')?.checked || false;

  // Loading state
  runBtn.disabled = true;
  if (spinner) spinner.style.display = 'inline-block';
  const btnText = runBtn.querySelector('.btn-text');
  if (btnText) btnText.textContent = 'Analyzing…';

  const baseUrl = (document.getElementById('inp-api-url')?.value || 'http://localhost:8000/api/v1/climate/predictions')
    .split('/api/')[0];
  const endpointUrl = `${baseUrl}/api/v1/recommendations`;

  const payload = {
    narrow_street:     narrow,
    water_requirement: water,
    min_cooling_score: cooling,
    top_n:             topn,
  };
  RecState.currentCriteria = payload;

  try {
    const res = await fetch(endpointUrl, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    /** @type {{ total_results: number, criteria: Object, recommendations: Array }} */
    const data = await res.json();
    RecState.lastResults = (data.recommendations && data.recommendations.length > 0)
      ? data.recommendations
      : FALLBACK_SPECIES.slice(0, topn);

    // Sanitize Arabic names
    RecState.lastResults.forEach(sp => {
      if (!sp.name_ar || sp.name_ar.includes('\uFFFD') || sp.name_ar.includes('?')) {
        sp.name_ar = TREE_AR_MAP[sp.name_en] || sp.name_ar || '';
      }
    });

    RecState.decisions.clear();
    renderSpeciesCards(RecState.lastResults);
    updateDiffAfterCol(RecState.lastResults, payload);
    document.getElementById('rec-count-badge').textContent = `${RecState.lastResults.length} species`;
    showToast(`🌳 Found ${RecState.lastResults.length} matching species!`, 'success');

  } catch (err) {
    console.warn('[Zurra] Recommendations using local catalog:', err);
    // Graceful fallback with rich catalog
    let filtered = FALLBACK_SPECIES.filter(s => {
      if (water && s.water_requirement !== water) return false;
      if (cooling && (s.cooling_effect_score || 0) < cooling) return false;
      return true;
    });
    if (filtered.length === 0) filtered = FALLBACK_SPECIES.slice(0, topn);
    RecState.lastResults = filtered.slice(0, topn);
    RecState.decisions.clear();

    renderSpeciesCards(RecState.lastResults);
    updateDiffAfterCol(RecState.lastResults, payload);
    document.getElementById('rec-count-badge').textContent = `${RecState.lastResults.length} species`;
    showToast(`🌳 Loaded ${RecState.lastResults.length} Egyptian species`, 'info');
  } finally {
    runBtn.disabled = false;
    if (spinner) spinner.style.display = 'none';
    if (btnText) btnText.textContent = '🔍 Get Recs';
  }
}

/**
 * Renders species cards in #species-grid with action buttons.
 * @param {Array|null} species
 * @param {string} [errorMsg]
 */
function renderSpeciesCards(species, errorMsg) {
  const grid = document.getElementById('species-grid');
  if (!grid) return;

  if (!species || species.length === 0) {
    grid.innerHTML = `<div class="rec-empty">
      <div class="rec-empty-icon">${errorMsg ? '⚠️' : '🌱'}</div>
      <div>${errorMsg ? `Error: ${errorMsg}` : 'No species matched the selected criteria.'}</div>
    </div>`;
    return;
  }

  grid.innerHTML = species.map((sp, i) => {
    const waterClass =
      sp.water_requirement === 'Low'  ? 'water-low' :
      sp.water_requirement === 'High' ? 'water-high' : 'water-med';
    const delay = i * 0.08;
    const score = sp.final_score != null ? sp.final_score.toFixed(2) : '15.20';
    const nameEn = sp.name_en || 'Unknown Species';
    const nameAr = (sp.name_ar && !sp.name_ar.includes('\uFFFD') && !sp.name_ar.includes('?'))
      ? sp.name_ar
      : (TREE_AR_MAP[nameEn] || sp.name_ar || '');
    const cardId = `species-card-${i}`;

    const coolingVal = sp.cooling_effect_score ?? 7;
    const airVal = sp.air_purification_score ?? 8;
    const canopyVal = sp.canopy_spread_m != null ? `${sp.canopy_spread_m}m` : (sp.category === 'Shade' ? '8.5m' : '6.0m');
    const rootVal = sp.root_type || (sp.suitable_for_narrow_streets ? 'Deep' : 'Deep Taproot');
    const growthVal = sp.growth_rate || 'Moderate';

    const currentDecision = RecState.decisions.get(nameEn);

    return `
      <div class="species-card" id="${cardId}" style="animation-delay:${delay}s;">
        <div class="species-card-header">
          <div>
            <div class="species-name-en">🌳 ${nameEn}</div>
            ${nameAr ? `<div class="species-name-ar" style="font-family:'Cairo',sans-serif;font-weight:600;color:var(--text-secondary);">${nameAr}</div>` : ''}
          </div>
          <span class="species-score-badge" title="AI Composite Match Score">★ ${score}</span>
        </div>

        <div class="species-chips">
          <span class="species-chip ${waterClass}">💧 ${sp.water_requirement || 'Low'} Need</span>
          ${sp.category ? `<span class="species-chip">📂 ${sp.category}</span>` : ''}
          <span class="species-chip">🌱 ${rootVal}</span>
          <span class="species-chip">⬆ ${growthVal}</span>
        </div>

        <div class="species-metrics">
          <div class="species-metric">
            <div class="species-metric-label">🌡️ Cooling</div>
            <div class="species-metric-val">${coolingVal}/10</div>
          </div>
          <div class="species-metric">
            <div class="species-metric-label">🌬️ Air Purif.</div>
            <div class="species-metric-val">${airVal}/10</div>
          </div>
          <div class="species-metric">
            <div class="species-metric-label">🌲 Canopy</div>
            <div class="species-metric-val">${canopyVal}</div>
          </div>
        </div>

        <!-- Approve / Modify / Reject action buttons -->
        <div class="species-action-row" id="action-row-${i}">
          <button class="action-btn approve ${currentDecision === 'approved' ? 'done' : ''}" data-index="${i}" data-action="approved"
                  title="Approve this species for planting">
            ✅ Approve
            <span class="btn-status-badge">✓</span>
          </button>
          <button class="action-btn modify ${currentDecision === 'modified' ? 'done' : ''}" data-index="${i}" data-action="modified"
                  title="Accept with density modification">
            ✏️ Modify
            <span class="btn-status-badge">✎</span>
          </button>
          <button class="action-btn reject ${currentDecision === 'rejected' ? 'done' : ''}" data-index="${i}" data-action="rejected"
                  title="Reject this species from plan">
            ❌ Reject
            <span class="btn-status-badge">✗</span>
          </button>
        </div>
      </div>
    `;
  }).join('');

  // Wire up action buttons
  grid.querySelectorAll('.action-btn').forEach(btn => {
    btn.addEventListener('click', () => onSpeciesAction(btn));
  });
}

/**
 * Handles Approve / Modify / Reject button clicks.
 * @param {HTMLButtonElement} btn
 */
function onSpeciesAction(btn) {
  const idx    = parseInt(btn.dataset.index, 10);
  const action = btn.dataset.action; // 'approved' | 'modified' | 'rejected'
  const row    = document.getElementById(`action-row-${idx}`);
  const sp     = RecState.lastResults[idx];
  if (!sp || !row) return;

  // Toggle off if same action clicked again
  if (RecState.decisions.get(sp.name_en) === action) {
    RecState.decisions.delete(sp.name_en);
    row.querySelectorAll('.action-btn').forEach(b => b.classList.remove('done'));
    showToast(`↩️ Decision cleared for ${sp.name_en}`, 'warning');
  } else {
    RecState.decisions.set(sp.name_en, action);
    row.querySelectorAll('.action-btn').forEach(b => {
      b.classList.toggle('done', b.dataset.action === action);
    });

    const icons   = { approved: '✅', modified: '✏️', rejected: '❌' };
    const colours = { approved: 'success', modified: 'warning', rejected: 'error' };
    showToast(`${icons[action]} ${sp.name_en} marked as ${action.toUpperCase()}`, colours[action]);
  }

  // Update Diff View and stats in real time
  updateDiffAfterCol(RecState.lastResults, RecState.currentCriteria);
}

/**
 * Updates the After column in the Diff View with live scenario data,
 * decision badges, and impact statistics.
 * @param {Array} species
 * @param {{ water_requirement: string, min_cooling_score: number }} criteria
 */
function updateDiffAfterCol(species, criteria) {
  const col = document.getElementById('diff-after-col');
  if (!col || !species || !species.length) return;

  const approvedList = species.filter(s => RecState.decisions.get(s.name_en) === 'approved');
  const modifiedList = species.filter(s => RecState.decisions.get(s.name_en) === 'modified');
  const rejectedList = species.filter(s => RecState.decisions.get(s.name_en) === 'rejected');

  // Update decision summary pills
  const appPill = document.getElementById('dec-pill-approved');
  const modPill = document.getElementById('dec-pill-modified');
  const rejPill = document.getElementById('dec-pill-rejected');
  if (appPill) appPill.textContent = `${approvedList.length} Approved`;
  if (modPill) modPill.textContent = `${modifiedList.length} Modified`;
  if (rejPill) rejPill.textContent = `${rejectedList.length} Rejected`;

  // Compute live microclimate diff metrics
  const activeTrees = approvedList.length > 0 ? approvedList : species;
  const avgCooling = activeTrees.reduce((acc, t) => acc + (t.cooling_effect_score || 6), 0) / (activeTrees.length || 1);
  const coolingDelta = Math.min(6.5, +(avgCooling * 0.48).toFixed(1));
  const canopyTotal = activeTrees.reduce((acc, t) => acc + (parseFloat(t.canopy_spread_m) || 7.5) * 48, 0);
  const co2Total = activeTrees.reduce((acc, t) => acc + (t.carbon_sequestration_kg_yr || 32) * 6, 0);

  // Update Diff Stats Bar
  const statTemp = document.getElementById('diff-stat-temp');
  const statCanopy = document.getElementById('diff-stat-canopy');
  const statCo2 = document.getElementById('diff-stat-co2');
  const statWater = document.getElementById('diff-stat-water');

  if (statTemp) statTemp.textContent = `−${coolingDelta}°C`;
  if (statCanopy) statCanopy.textContent = `+${Math.round(canopyTotal).toLocaleString()} m²`;
  if (statCo2) statCo2.textContent = `+${Math.round(co2Total)} kg/yr`;
  if (statWater) statWater.textContent = criteria?.water_requirement ? `${criteria.water_requirement} Need` : 'Low Need';

  // Build diff lines
  const lines = [];

  if (approvedList.length > 0) {
    approvedList.forEach(sp => {
      const nameAr = TREE_AR_MAP[sp.name_en] || sp.name_ar || '';
      lines.push({ cls: 'added', icon: '✓', text: `Plant: <strong>${escapeHtml(sp.name_en)}</strong> (${nameAr}) — [APPROVED]` });
    });
  } else {
    const top = species[0];
    const nameAr = TREE_AR_MAP[top.name_en] || top.name_ar || '';
    lines.push({ cls: 'added', icon: '+', text: `Proposed: <strong>${escapeHtml(top.name_en)}</strong> (${nameAr})` });
  }

  if (modifiedList.length > 0) {
    modifiedList.forEach(sp => {
      lines.push({ cls: 'neutral', icon: '✎', text: `Adjusted spacing: <strong>${escapeHtml(sp.name_en)}</strong> — [MODIFIED]` });
    });
  }

  lines.push({ cls: 'added', icon: '🌡️', text: `Est. surface cooling: <strong>−${coolingDelta}°C</strong> heat island reduction` });
  lines.push({ cls: 'added', icon: '🌲', text: `Added canopy shade: <strong>+${Math.round(canopyTotal).toLocaleString()} m²</strong> active cover` });
  lines.push({ cls: 'added', icon: '💨', text: `Carbon sequestration: <strong>+${Math.round(co2Total)} kg/yr</strong>` });
  lines.push({ cls: 'neutral', icon: '💧', text: `Irrigation profile: Subsurface drip (15 L/tree/wk)` });

  if (rejectedList.length > 0) {
    lines.push({ cls: 'removed', icon: '✗', text: `Excluded: ${rejectedList.map(r => r.name_en).join(', ')}` });
  }

  col.innerHTML = `
    <div class="diff-col-label">✅ After — AI Proposed Action (${approvedList.length > 0 ? approvedList.length + ' approved' : '1 recommended'})</div>
    ${lines.map(l => `
      <div class="diff-line ${l.cls}">
        <span class="diff-marker ${l.icon === '+' || l.icon === '✓' ? 'plus' : (l.icon === '✗' ? 'minus' : 'eq')}">${l.icon}</span>
        <div>${l.text}</div>
      </div>`).join('')}
    
    <button id="apply-to-map-btn"
            style="margin-top:10px;width:100%;padding:6px 12px;font-size:0.75rem;font-weight:600;background:linear-gradient(135deg,var(--green-primary),var(--green-light));border:none;border-radius:var(--radius-sm);color:#fff;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:6px;transition:all var(--t-fast);"
            title="Visualize green canopy coverage on active grid cell">
      🗺️ Project Greening on Active Map Cell
    </button>
  `;

  // Wire up map projection button
  const applyBtn = document.getElementById('apply-to-map-btn');
  if (applyBtn) {
    applyBtn.addEventListener('click', () => {
      projectCanopyOnMap(coolingDelta, canopyTotal);
    });
  }
}

/**
 * Projects a shaded green canopy overlay on the Leaflet map at selected coordinates.
 * @param {number} coolingDelta
 * @param {number} canopyTotal
 */
function projectCanopyOnMap(coolingDelta, canopyTotal) {
  if (!AppState.map) {
    showToast('🗺️ Map unavailable', 'warning');
    return;
  }

  const lat = AppState.selectedLat || 30.0444;
  const lon = AppState.selectedLon || 31.2357;

  if (RecState.mapCanopyLayer) {
    AppState.map.removeLayer(RecState.mapCanopyLayer);
  }

  // Draw a lush green semi-transparent circle representing newly shaded canopy
  RecState.mapCanopyLayer = L.circle([lat, lon], {
    color: '#2E5A44',
    fillColor: '#22C55E',
    fillOpacity: 0.38,
    radius: 450,
  }).addTo(AppState.map);

  RecState.mapCanopyLayer.bindPopup(`
    <div style="font-family:'Inter',sans-serif;padding:4px;">
      <strong style="color:var(--green-light);font-size:0.9rem;">🌳 Projected Urban Forest Cell</strong>
      <div style="font-size:0.75rem;margin-top:4px;">
        • Surface Cooling: <strong>−${coolingDelta}°C</strong><br>
        • Shaded Canopy: <strong>+${Math.round(canopyTotal).toLocaleString()} m²</strong><br>
        • Status: <strong>Approved Plan Active</strong>
      </div>
    </div>
  `).openPopup();

  AppState.map.setView([lat, lon], 12);
  showToast(`🗺️ Green canopy projected on map! (−${coolingDelta}°C)`, 'success');
}

/**
 * Resets the recommendations panel to its initial state.
 */
function resetRecommendations() {
  RecState.decisions.clear();
  RecState.lastResults = [];
  document.getElementById('species-grid').innerHTML = `
    <div class="rec-empty">
      <div class="rec-empty-icon">🌱</div>
      <div>Configure filters and click <strong>Get Recs</strong> to generate AI-ranked tree species.</div>
    </div>`;
  document.getElementById('diff-after-col').innerHTML = `
    <div class="diff-col-label">✅ After — AI Proposed Action</div>
    <div class="diff-line neutral" style="color:var(--text-muted);font-style:italic;">
      <span class="diff-marker eq">•</span> Run recommendations to populate
    </div>`;
  document.getElementById('rec-count-badge').textContent = '0 species';

  const appPill = document.getElementById('dec-pill-approved');
  const modPill = document.getElementById('dec-pill-modified');
  const rejPill = document.getElementById('dec-pill-rejected');
  if (appPill) appPill.textContent = '0 Approved';
  if (modPill) modPill.textContent = '0 Modified';
  if (rejPill) rejPill.textContent = '0 Rejected';

  const statTemp = document.getElementById('diff-stat-temp');
  const statCanopy = document.getElementById('diff-stat-canopy');
  const statCo2 = document.getElementById('diff-stat-co2');
  const statWater = document.getElementById('diff-stat-water');
  if (statTemp) statTemp.textContent = '—';
  if (statCanopy) statCanopy.textContent = '—';
  if (statCo2) statCo2.textContent = '—';
  if (statWater) statWater.textContent = '—';

  if (RecState.mapCanopyLayer && AppState.map) {
    AppState.map.removeLayer(RecState.mapCanopyLayer);
    RecState.mapCanopyLayer = null;
  }
}


/* ============================================================
   § 17  CHATBOT MODULE
   ============================================================ */

/** Chat session state */
const ChatState = {
  /** Active language: 'ar' | 'en' */
  language: 'ar',
  /** Thread ID for multi-turn LangGraph persistence */
  threadId: null,
  /** Whether the assistant is currently typing */
  isTyping: false,
  /** Message counter for unique IDs */
  msgCount: 0,
};

/**
 * Checks if a string contains Arabic characters.
 * @param {string} text
 * @returns {boolean}
 */
function isArabic(text) {
  return /[\u0600-\u06FF]/.test(text);
}

/**
 * Initialises the chatbot widget:
 * - Renders the greeting message.
 * - Wires send button, Enter key, quick-chips, language toggle, clear button.
 */
function initChatbot() {
  const sendBtn   = document.getElementById('chat-send-btn');
  const clearBtn  = document.getElementById('chat-clear-btn');
  const input     = document.getElementById('chat-input');
  const chips     = document.querySelectorAll('#chat-chips .chip');
  const langBtns  = document.querySelectorAll('.lang-btn');

  if (!sendBtn || !input) return;

  // Greeting from Groot
  appendBotMessage(
    ChatState.language === 'ar'
      ? '🌳 أهلاً بك! أنا مساعد زرعة الذكي للمناخ والتشجير الحضري في مصر. يمكنني مساعدتك في تحليل الجزر الحرارية، اختيار الأشجار الملائمة، والاستشارات القانونية البيئية (قانون 4/1994). كيف يمكنني مساعدتك اليوم؟'
      : '🌳 Welcome! I am Zurra\'s Smart Assistant for urban climate and forestry in Egypt. I can help analyze urban heat islands, recommend low-water tree species, and provide environmental legal guidance. How can I help you today?',
    ['supervisor_agent', 'climate_agent']
  );

  // Send on button click
  sendBtn.addEventListener('click', sendChatMessage);

  // Send on Enter key
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage();
    }
  });

  // Quick-reply chips
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      input.value = chip.dataset.prompt || chip.textContent.trim();
      sendChatMessage();
    });
  });

  // Language toggle
  langBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      langBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      ChatState.language = btn.dataset.lang;
      showToast(`🌐 Language set to ${btn.dataset.lang.toUpperCase()}`, 'info');
    });
  });

  // Clear chat
  clearBtn.addEventListener('click', () => {
    const msgs = document.getElementById('chat-messages');
    if (msgs) msgs.innerHTML = '';
    ChatState.threadId = null;
    appendBotMessage(
      ChatState.language === 'ar'
        ? '🌳 تم مسح سجل المحادثة. اسألني عن المناخ الحضري أو الأشجار المناسبة!'
        : '🌳 Chat history cleared. Ask me about urban climate, trees, or Egyptian environmental laws!',
      []
    );
  });
}

/**
 * Reads the chat input and sends the message to the FastAPI chat endpoint.
 * Dispatches query directly to the LangGraph RAG pipeline.
 */
async function sendChatMessage() {
  const input   = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send-btn');
  const text    = (input?.value || '').trim();
  if (!text || ChatState.isTyping) return;

  input.value = '';
  appendUserMessage(text);

  ChatState.isTyping = true;
  if (sendBtn) sendBtn.disabled = true;
  showTypingIndicator();

  const baseUrl = (document.getElementById('inp-api-url')?.value || 'http://localhost:8000/api/v1/climate/predictions')
    .split('/api/')[0];
  const chatUrl = `${baseUrl}/api/v1/chat/messages`;

  const payload = {
    message:   text,
    thread_id: ChatState.threadId || undefined,
    language:  ChatState.language,
  };

  try {
    // 120-second timeout controller for local Ollama LLM execution
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 120000);

    const res = await fetch(chatUrl, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
      signal:  controller.signal,
    });
    clearTimeout(timeoutId);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    /** @type {{ response: string, thread_id: string, routes: string[], candidate_species: Array|null }} */
    const data = await res.json();
    ChatState.threadId = data.thread_id;

    removeTypingIndicator();
    appendBotMessage(data.response, data.routes || ['langgraph_agent']);

    if (data.candidate_species && data.candidate_species.length > 0) {
      const speciesList = data.candidate_species
        .slice(0, 3)
        .map(s => `• ${s.name_en || s.name_ar}`)
        .join('\n');
      appendBotMessage(`🌳 Top candidate trees:\n${speciesList}`, ['recommendation_engine']);
    }

  } catch (err) {
    console.error('[Zar3a Chat] Backend error:', err);
    removeTypingIndicator();
    const isAr = ChatState.language === 'ar' || isArabic(text);
    const errMsg = isAr
      ? `⚠️ **تعذر الاتصال بخادم الذكاء الاصطناعي:** ${err.message || 'انتهت مهلة الطلب'}\nيرجى التأكد من تشغيل خادم الواجهة الخلفية وإعادة المحاولة.`
      : `⚠️ **Could not connect to the Multi-Agent API server:** ${err.message || 'Request timed out'}\nPlease ensure the FastAPI backend is running and try again.`;
    appendBotMessage(errMsg, ['system_error']);
  } finally {
    ChatState.isTyping = false;
    if (sendBtn) sendBtn.disabled = false;
  }
}

/**
 * Appends a user message bubble to the chat with RTL/LTR awareness.
 * @param {string} text
 */
function appendUserMessage(text) {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;
  const id = ++ChatState.msgCount;
  const rtl = isArabic(text);

  const wrapper = document.createElement('div');
  wrapper.className = 'msg user';
  wrapper.id = `msg-${id}`;
  wrapper.innerHTML = `
    <div>
      <div class="msg-bubble ${rtl ? 'rtl' : ''}" ${rtl ? 'dir="rtl"' : ''}>${escapeHtml(text)}</div>
      <div class="msg-time">${now()}</div>
    </div>
    <div class="msg-avatar user-avatar" aria-hidden="true">U</div>
  `;
  msgs.appendChild(wrapper);
  scrollChatToBottom();
}

/**
 * Appends a bot message bubble with Groot avatar and route tags.
 * @param {string} text
 * @param {string[]} routes
 */
function appendBotMessage(text, routes) {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;
  const id = ++ChatState.msgCount;
  const rtl = isArabic(text);

  // Convert markdown bold to <strong> and newlines to <br>
  let htmlText = escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');

  const routeTagsHtml = (routes && routes.length > 0)
    ? `<div class="msg-routes">${routes.map(r => `<span class="route-tag">⚡ ${escapeHtml(r)}</span>`).join('')}</div>`
    : '';

  const wrapper = document.createElement('div');
  wrapper.className = 'msg bot';
  wrapper.id = `msg-${id}`;
  wrapper.style.animationDelay = '0.05s';
  wrapper.innerHTML = `
    <img src="assets/I am Groot.png"
         alt="Zurra Assistant"
         class="msg-avatar"
         onerror="this.style.display='none'" />
    <div>
      <div class="msg-bubble ${rtl ? 'rtl' : ''}" ${rtl ? 'dir="rtl"' : ''}>${htmlText}${routeTagsHtml}</div>
      <div class="msg-time">${now()}</div>
    </div>
  `;
  msgs.appendChild(wrapper);
  scrollChatToBottom();
}

/**
 * Shows the animated typing indicator as a bot message.
 */
function showTypingIndicator() {
  const msgs = document.getElementById('chat-messages');
  if (!msgs) return;

  const indicator = document.createElement('div');
  indicator.className = 'msg bot';
  indicator.id = 'typing-indicator-msg';
  indicator.innerHTML = `
    <img src="assets/I am Groot.png"
         alt="Zurra typing"
         class="msg-avatar"
         onerror="this.style.display='none'" />
    <div class="msg-bubble" style="padding:10px 14px;">
      <div class="typing-indicator">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>
  `;
  msgs.appendChild(indicator);
  scrollChatToBottom();
}

/** Removes the typing indicator bubble. */
function removeTypingIndicator() {
  document.getElementById('typing-indicator-msg')?.remove();
}

/** Scrolls the chat container to the very bottom. */
function scrollChatToBottom() {
  const msgs = document.getElementById('chat-messages');
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

/**
 * Escapes HTML special characters to prevent XSS.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

/**
 * Returns the current time as HH:MM string.
 * @returns {string}
 */
function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}


