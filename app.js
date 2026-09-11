/**
 * AquaG — Operational Urban Flood GIS Command Center
 * Phase 3A Frontend Controller (Vanilla JavaScript + Leaflet.js)
 * 
 * Centralized API_BASE_URL connecting strictly to Stage 7 FastAPI backend.
 * Integrates:
 * - Street-Level Waterlogging GeoJSON Layer (POST /waterlogging)
 * - Critical Infrastructure GeoJSON Layer (GET /infrastructure)
 * - MPD-1976 Drainage Network Reference GeoJSON Layer (GET /drainage)
 * - Permanent Pumping Stations Metadata Registry (GET /pumps)
 * - Model V2 Scenario Simulation (POST /predict)
 * - AquaGraph Risk Routing (POST /route)
 */

// Configurable API_BASE_URL: supports window.AQUAG_API_URL, automatically defaults to https://aquag.onrender.com in production (e.g. GitHub Pages) and http://127.0.0.1:8000 on localhost for local development.
function getApiBaseUrl() {
  if (typeof window !== "undefined" && window.AQUAG_API_URL) {
    return window.AQUAG_API_URL;
  }
  if (typeof window !== "undefined" && window.location) {
    const hostname = window.location.hostname;
    if (hostname === "localhost" || hostname === "127.0.0.1") {
      return "http://127.0.0.1:8000";
    }
  }
  return "https://aquag.onrender.com";
}

const API_BASE_URL = getApiBaseUrl();

// Global State Variables
let map = null;
let waterloggingLayer = null;
let infraLayer = null;
let drainageNetworkLayer = null;
let routeLayer = null;
let zonesLayer = null;
let drainsLayer = null;
let markersLayer = null;
let populationPriorityLayer = null;

let pickingMode = null; // 'start' or 'end'
let activeScenario = "NORMAL";
let activeTimestep = "T+0";
let activePresetName = "NORMAL";

let spatialDebounceTimer = null;
const scenarioHistory = [];

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  initSideTabs();
  initPresetButtons();
  initFormHandlers();
  initLayerToggles();
  initTimelineBar();
  initInspectorCard();

  // Initial Backend API Calls
  checkSystemHealth();
  loadZoneLayer();
  loadWaterloggingLayer(); // Auto-load street-level waterlogging for current viewport (NORMAL T+0)
  loadDrainageNetworkLayer(); // Auto-load MPD-1976 drainage network layer
  loadAlertsPanel(); // Auto-load model-derived forecast incidents for current viewport
});

// --------------------------------------------------------------------------
// Leaflet Map Initialization
// --------------------------------------------------------------------------
function initMap() {
  // Center on Delhi coordinates [28.6139, 77.2090]
  map = L.map("map", {
    center: [28.6139, 77.2090],
    zoom: 12,
    zoomControl: true,
  });

  // OpenStreetMap Open Source Tile Layer
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);

  // GeoJSON & Vector Layer Groups
  waterloggingLayer = L.geoJSON(null, {
    style: getRoadStyle,
    onEachFeature: bindRoadPopup,
  }).addTo(map);

  infraLayer = L.geoJSON(null, {
    pointToLayer: createInfraMarker,
    onEachFeature: bindInfraPopup,
  });

  drainageNetworkLayer = L.geoJSON(null, {
    pointToLayer: createDrainMarker,
    onEachFeature: bindDrainPopup,
  }).addTo(map);

  zonesLayer = L.layerGroup().addTo(map);
  drainsLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  markersLayer = L.layerGroup().addTo(map);

  populationPriorityLayer = L.geoJSON(null, {
    style: getPriorityRoadStyle,
    onEachFeature: bindPriorityRoadPopup,
  });

  // Map Click Event (Point Inspector & Coord Picker)
  map.on("click", handleMapClick);

  // Viewport Movement & Zoom Debounce (300ms) for All Active Spatial Layers
  map.on("moveend", onMapMoveOrZoom);
  map.on("zoomend", onMapMoveOrZoom);
}

// --------------------------------------------------------------------------
// Viewport & Spatial Layer Utilities
// --------------------------------------------------------------------------
function getMapViewportBbox() {
  if (!map) return [76.8, 28.3, 77.4, 28.9];
  const bounds = map.getBounds();
  return [
    bounds.getWest(),
    bounds.getSouth(),
    bounds.getEast(),
    bounds.getNorth()
  ];
}

function onMapMoveOrZoom() {
  if (spatialDebounceTimer) clearTimeout(spatialDebounceTimer);
  spatialDebounceTimer = setTimeout(() => {
    loadWaterloggingLayer();
    loadInfraLayer();
    loadDrainageNetworkLayer();
    loadPopulationPriorityLayer();
    loadAlertsPanel();
  }, 300);
}

// --------------------------------------------------------------------------
// 1. Waterlogging Layer Engine (POST /waterlogging)
// --------------------------------------------------------------------------
function getRoadColor(depth) {
  if (depth > 100) return "#9333ea"; // >100 cm: High / deep red-purple
  if (depth > 50) return "#1d4ed8";  // 50-100 cm: High / dark blue
  if (depth > 25) return "#2563eb";  // 25-50 cm: Medium / stronger blue
  if (depth > 10) return "#3b82f6";  // 10-25 cm: Medium / blue
  return "#60a5fa";                   // 0-10 cm: Low / light blue
}

function getRoadStyle(feature) {
  const depth = feature.properties ? (feature.properties.water_depth_cm || 0) : 0;
  const color = getRoadColor(depth);
  const weight = depth > 50 ? 5 : depth > 25 ? 4 : 3;
  return {
    color: color,
    weight: weight,
    opacity: 0.85,
    lineCap: "round",
    lineJoin: "round"
  };
}

function bindRoadPopup(feature, layer) {
  const p = feature.properties || {};
  const roadId = p.road_id || "Unknown Segment";
  const depthVal = typeof p.water_depth_cm === "number" ? p.water_depth_cm.toFixed(1) : (p.water_depth_cm || "0.0");
  const severity = p.severity || "Low";
  const depthBand = p.depth_band || "0-10 cm";
  const elevation = typeof p.elevation_m === "number" ? `${p.elevation_m.toFixed(1)} m` : "N/A";
  const drainDist = typeof p.distance_to_drain_m === "number" ? `${p.distance_to_drain_m.toFixed(1)} m` : "N/A";
  const popExp = typeof p.population_exposure === "number" ? p.population_exposure.toLocaleString() : "N/A";
  const critInfra = p.critical_infra_flag ? "Yes" : "No";
  const forecastTimestep = p.timestep || activeTimestep;

  const content = `
    <div class="waterlogging-popup">
      <div class="wl-popup-title">${roadId}</div>
      <div class="wl-popup-row"><span>Water depth:</span> <strong>${depthVal} cm</strong></div>
      <div class="wl-popup-row"><span>Severity:</span> <span class="sev-tag ${severity.toLowerCase()}">${severity}</span></div>
      <div class="wl-popup-row"><span>Band:</span> <strong>${depthBand}</strong></div>
      <div class="wl-popup-row"><span>Elevation:</span> <strong>${elevation}</strong></div>
      <div class="wl-popup-row"><span>Drain distance:</span> <strong>${drainDist}</strong></div>
      <div class="wl-popup-row"><span>Population exposure:</span> <strong>${popExp}</strong></div>
      <div class="wl-popup-row"><span>Critical infrastructure:</span> <strong>${critInfra}</strong></div>
      <div class="wl-popup-row"><span>Forecast:</span> <strong>${forecastTimestep}</strong></div>
      <div class="wl-popup-footer">Model-derived waterlogging depth proxy</div>
    </div>
  `;
  layer.bindPopup(content, { className: "dark-leaflet-popup" });
}

async function loadWaterloggingLayer() {
  const wCheck = document.getElementById("layer-waterlogging-check");
  if (wCheck && !wCheck.checked) return;

  const bbox = getMapViewportBbox();
  const r1hEl = document.getElementById("rainfall_1h");
  const r3hEl = document.getElementById("rainfall_3h");
  const r6hEl = document.getElementById("rainfall_6h");
  const intEl = document.getElementById("recent_rainfall_intensity");

  const payload = {
    scenario: activeScenario,
    timestep: activeTimestep,
    rainfall_1h: r1hEl ? parseFloat(r1hEl.value) || 10.0 : 10.0,
    rainfall_3h: r3hEl ? parseFloat(r3hEl.value) || 20.0 : 20.0,
    rainfall_6h: r6hEl ? parseFloat(r6hEl.value) || 30.0 : 30.0,
    recent_rainfall_intensity: intEl ? parseFloat(intEl.value) || 5.0 : 5.0,
    bbox: bbox
  };

  try {
    const res = await fetch(`${API_BASE_URL}/waterlogging`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (waterloggingLayer) {
      waterloggingLayer.clearLayers();
      waterloggingLayer.addData(geojson);
    }
  } catch (err) {
    console.warn("Waterlogging layer fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 2. Critical Infrastructure Layer Engine (GET /infrastructure)
// --------------------------------------------------------------------------
function createInfraMarker(feature, latlng) {
  const p = feature.properties || {};
  const cat = p.category || "other";
  const critical = p.critical || false;

  let color = "#94a3b8"; // default gray
  if (cat === "hospital") color = "#ef4444";     // red
  else if (cat === "police") color = "#3b82f6";  // blue
  else if (cat === "emergency") color = "#f59e0b"; // amber
  else if (cat === "metro" || cat === "railway") color = "#a855f7"; // purple
  else if (cat === "school" || cat === "college") color = "#10b981"; // emerald

  const radius = critical ? 7 : 5;
  return L.circleMarker(latlng, {
    radius: radius,
    color: color,
    fillColor: color,
    fillOpacity: 0.8,
    weight: 2
  });
}

function bindInfraPopup(feature, layer) {
  const p = feature.properties || {};
  const name = p.name || "Infrastructure Facility";
  const category = p.category || "General";
  const criticalText = p.critical ? "Yes (Priority Asset)" : "No";

  const content = `
    <div class="infra-popup">
      <div class="infra-popup-title">${name}</div>
      <div class="infra-popup-row"><span>Category:</span> <strong>${category}</strong></div>
      <div class="infra-popup-row"><span>Critical:</span> <span class="infra-badge ${p.critical ? 'critical' : 'normal'}">${criticalText}</span></div>
      <div class="infra-popup-row"><span>Source:</span> <strong>OSM</strong></div>
    </div>
  `;
  layer.bindPopup(content, { className: "dark-leaflet-popup" });
}

async function loadInfraLayer() {
  const iCheck = document.getElementById("layer-infra-check");
  if (iCheck && !iCheck.checked) return;

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");

  try {
    const res = await fetch(`${API_BASE_URL}/infrastructure?bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (infraLayer) {
      infraLayer.clearLayers();
      infraLayer.addData(geojson);
    }
  } catch (err) {
    console.warn("Infrastructure layer fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 3. Drainage Network Layer Engine (GET /drainage)
// --------------------------------------------------------------------------
function createDrainMarker(feature, latlng) {
  return L.circleMarker(latlng, {
    radius: 5,
    color: "#06b6d4",
    fillColor: "#06b6d4",
    fillOpacity: 0.75,
    weight: 1.5
  });
}

function bindDrainPopup(feature, layer) {
  const p = feature.properties || {};
  const drainName = p.drain_name || "MPD-1976 Drain";
  const basin = p.basin || "Delhi Basin";
  const seqNo = p.seq_no || "N/A";
  const status = p.status || "Existing / Remodeling";
  const source = p.source || "MPD-1976";

  const content = `
    <div class="drain-popup">
      <div class="drain-popup-title">${drainName}</div>
      <div class="drain-popup-row"><span>Basin:</span> <strong>${basin}</strong></div>
      <div class="drain-popup-row"><span>Seq No:</span> <strong>${seqNo}</strong></div>
      <div class="drain-popup-row"><span>Status:</span> <strong>${status}</strong></div>
      <div class="drain-popup-row"><span>Source:</span> <strong>${source}</strong></div>
    </div>
  `;
  layer.bindPopup(content, { className: "dark-leaflet-popup" });
}

async function loadDrainageNetworkLayer() {
  const dCheck = document.getElementById("layer-drains-check");
  if (dCheck && !dCheck.checked) return;

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");

  try {
    const res = await fetch(`${API_BASE_URL}/drainage?bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (drainageNetworkLayer) {
      drainageNetworkLayer.clearLayers();
      drainageNetworkLayer.addData(geojson);
    }
  } catch (err) {
    console.warn("Drainage network layer fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 4. Pump Stations Metadata Registry (GET /pumps)
// --------------------------------------------------------------------------
async function handlePumpsToggle(enabled) {
  if (!enabled) return;

  try {
    const res = await fetch(`${API_BASE_URL}/pumps`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const count = data.total_stations || 10;
    alert(
      `PUMP STATIONS METADATA (Delhi Flood Control Order 2025):\n\n` +
      `• Registered Pumping Stations: ${count}\n` +
      `• Geographic Coordinates: Unavailable in official raw CSV source.\n` +
      `• SCADA Telemetry: Unavailable in official raw source.\n\n` +
      `Per AquaG Data Integrity Rules, no fake coordinates or fake markers are rendered on the map.`
    );
  } catch (err) {
    console.warn("Pumps metadata fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 5. Population Priority Layer Engine (GET /population-priority)
// --------------------------------------------------------------------------
function getPriorityRoadColor(level) {
  if (level === "CRITICAL") return "#9333ea"; // Red-purple
  if (level === "HIGH") return "#ef4444";     // Orange-red
  if (level === "MEDIUM") return "#f59e0b";   // Yellow-orange
  return "#3b82f6";                           // Subtle blue (LOW)
}

function getPriorityRoadStyle(feature) {
  const p = feature.properties || {};
  const level = p.priority_level || "LOW";
  const color = getPriorityRoadColor(level);
  const weight = level === "CRITICAL" ? 6 : level === "HIGH" ? 5 : level === "MEDIUM" ? 4 : 3;
  return {
    color: color,
    weight: weight,
    opacity: level === "LOW" ? 0.45 : 0.9,
    lineCap: "round",
    lineJoin: "round"
  };
}

function bindPriorityRoadPopup(feature, layer) {
  const p = feature.properties || {};
  const roadId = p.road_id || "Unknown Segment";
  const level = p.priority_level || "LOW";
  const score = typeof p.priority_score === "number" ? p.priority_score.toFixed(1) : p.priority_score;
  const depth = typeof p.water_depth_cm === "number" ? p.water_depth_cm.toFixed(1) : p.water_depth_cm;
  const popExp = typeof p.population_exposure === "number" ? p.population_exposure.toLocaleString() : p.population_exposure;
  const critInfra = p.critical_infra_flag === 1 ? "Yes" : "No";
  const district = p.district || "Delhi District";
  const basis = p.basis || "district_population_exposure_proxy";

  const content = `
    <div class="pop-priority-popup">
      <div class="wl-popup-title">${roadId} (${district})</div>
      <div class="wl-popup-row"><span>Priority Level:</span> <span class="sev-tag ${level.toLowerCase()}">${level}</span></div>
      <div class="wl-popup-row"><span>Priority Score:</span> <strong>${score} / 100</strong></div>
      <div class="wl-popup-row"><span>Water Depth:</span> <strong>${depth} cm</strong></div>
      <div class="wl-popup-row"><span>Population Exposure:</span> <strong>${popExp}</strong></div>
      <div class="wl-popup-row"><span>Critical Infrastructure:</span> <strong>${critInfra}</strong></div>
      <div class="wl-popup-footer">${basis}</div>
    </div>
  `;
  layer.bindPopup(content, { className: "dark-leaflet-popup" });
}

async function loadPopulationPriorityLayer() {
  const popCheck = document.getElementById("layer-pop-priority-check");
  if (popCheck && !popCheck.checked) return;

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");
  const encodedTimestep = encodeURIComponent(activeTimestep);

  try {
    const res = await fetch(`${API_BASE_URL}/population-priority?scenario=${activeScenario}&timestep=${encodedTimestep}&bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (populationPriorityLayer) {
      populationPriorityLayer.clearLayers();
      populationPriorityLayer.addData(geojson);
    }
  } catch (err) {
    console.warn("Population priority layer fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 6. Alerts & Triage Engine Panel (GET /alerts)
// --------------------------------------------------------------------------
async function loadAlertsPanel(force = false) {
  const alertsTab = document.getElementById("tab-alerts");
  if (!force && alertsTab && !alertsTab.classList.contains("active")) {
    return;
  }

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");
  const encodedTimestep = encodeURIComponent(activeTimestep);
  const container = document.getElementById("alerts-list-container");
  const countBadge = document.getElementById("alerts-count-badge");

  try {
    const res = await fetch(`${API_BASE_URL}/alerts?scenario=${activeScenario}&timestep=${encodedTimestep}&bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const incidents = data.incidents || [];
    if (countBadge) countBadge.textContent = `${incidents.length} Incidents`;

    if (!container) return;
    if (incidents.length === 0) {
      container.innerHTML = '<div class="alert-empty-msg">No high priority incidents for current viewport.</div>';
      return;
    }

    container.innerHTML = incidents.map((inc) => {
      const coordsJson = JSON.stringify(inc.coordinates || []);
      const pLevel = (inc.priority_level || "LOW").toUpperCase();
      return `
        <div class="alert-card priority-${pLevel.toLowerCase()}" data-coords='${coordsJson}'>
          <div class="alert-card-header">
            <span class="inc-id">${inc.incident_id} • ${inc.road_id}</span>
            <span class="inc-badge ${pLevel.toLowerCase()}">${pLevel}</span>
          </div>
          <div class="alert-card-body">
            <div class="inc-row"><span>District:</span> <strong>${inc.district || "Delhi"}</strong></div>
            <div class="inc-row"><span>Water Depth:</span> <strong>${inc.water_depth_cm.toFixed(1)} cm</strong></div>
            <div class="inc-row"><span>Population Exposure:</span> <strong>${(inc.population_exposure || 0).toLocaleString()}</strong></div>
            <div class="inc-row"><span>Critical Infra:</span> <strong>${inc.critical_infra_flag === 1 ? 'YES' : 'NO'}</strong></div>
            <div class="inc-row"><span>Forecast:</span> <strong>${inc.forecast_timestep}</strong></div>
            <div class="inc-action-box">
              <strong>Recommended Action:</strong> ${inc.recommended_action}
            </div>
          </div>
          <div class="alert-card-footer">
            <span>Basis: ${inc.basis}</span>
            <button class="locate-inc-btn">Locate on Map &rarr;</button>
          </div>
        </div>
      `;
    }).join("");

    // Add click listeners to locate incidents on Leaflet map
    container.querySelectorAll(".alert-card").forEach((card) => {
      card.addEventListener("click", () => {
        try {
          const coords = JSON.parse(card.getAttribute("data-coords"));
          if (coords && coords.length > 0) {
            // Convert GeoJSON [lon, lat] pairs to Leaflet [lat, lon]
            const leafletLatLngs = coords.map((c) => [c[1], c[0]]);
            const bounds = L.latLngBounds(leafletLatLngs);
            map.fitBounds(bounds, { maxZoom: 16, padding: [50, 50] });

            // Temporary highlight line
            const highlightPolyline = L.polyline(leafletLatLngs, {
              color: "#f43f5e",
              weight: 8,
              opacity: 0.9,
            }).addTo(map);

            setTimeout(() => {
              if (map.hasLayer(highlightPolyline)) map.removeLayer(highlightPolyline);
            }, 3000);
          }
        } catch (e) {
          console.warn("Failed to locate incident feature on map:", e);
        }
      });
    });
  } catch (err) {
    if (container) container.innerHTML = `<div class="alert-empty-msg">Alerts load notice: ${err.message}</div>`;
  }
}

// --------------------------------------------------------------------------
// UI Tabs & Controls Initialization
// --------------------------------------------------------------------------
function initSideTabs() {
  const sideTabBtns = document.querySelectorAll(".side-tab-btn");
  sideTabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      sideTabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".sidebar-tab-content").forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const targetEl = document.getElementById(targetTab);
      if (targetEl) targetEl.classList.add("active");

      if (targetTab === "tab-alerts") {
        loadAlertsPanel(true);
      }
    });
  });

  const navAlertsBtn = document.getElementById("nav-alerts-btn");
  if (navAlertsBtn) {
    navAlertsBtn.addEventListener("click", () => {
      const alertTabBtn = document.querySelector('.side-tab-btn[data-tab="tab-alerts"]');
      if (alertTabBtn) alertTabBtn.click();
    });
  }
}

function initPresetButtons() {
  const presets = {
    normal: {
      rainfall_1h: 10.0,
      rainfall_3h: 20.0,
      rainfall_6h: 30.0,
      recent_rainfall_intensity: 5.0,
      elevation: 218.0,
      slope: 1.2,
      distance_to_drain: 300.0,
      distance_to_road: 15.0,
      distance_to_infra: 400.0,
      population_total: 50000,
      critical_infra_flag: 0,
    },
    moderate: {
      rainfall_1h: 45.0,
      rainfall_3h: 85.0,
      rainfall_6h: 130.0,
      recent_rainfall_intensity: 22.5,
      elevation: 208.5,
      slope: 1.8,
      distance_to_drain: 45.0,
      distance_to_road: 20.0,
      distance_to_infra: 150.0,
      population_total: 120000,
      critical_infra_flag: 1,
    },
    heavy: {
      rainfall_1h: 75.0,
      rainfall_3h: 130.0,
      rainfall_6h: 180.0,
      recent_rainfall_intensity: 37.5,
      elevation: 204.0,
      slope: 0.5,
      distance_to_drain: 15.0,
      distance_to_road: 10.0,
      distance_to_infra: 50.0,
      population_total: 250000,
      critical_infra_flag: 1,
    },
    extreme: {
      rainfall_1h: 110.0,
      rainfall_3h: 180.0,
      rainfall_6h: 250.0,
      recent_rainfall_intensity: 55.0,
      elevation: 202.0,
      slope: 0.2,
      distance_to_drain: 10.0,
      distance_to_road: 5.0,
      distance_to_infra: 25.0,
      population_total: 350000,
      critical_infra_flag: 1,
    },
  };

  // Backwards compatibility aliases
  presets.low = presets.normal;
  presets.medium = presets.moderate;
  presets.high = presets.heavy;

  const presetBtns = document.querySelectorAll(".preset-btn, .preset-btn-4");
  presetBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const type = btn.getAttribute("data-preset");
      const data = presets[type];
      if (!data) return;

      presetBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      activeScenario = type.toUpperCase();
      activePresetName = activeScenario;

      Object.keys(data).forEach((key) => {
        const el = document.getElementById(key);
        if (el) el.value = data[key];
      });

      // Update Summary Card Preview
      updateScenarioSummaryCard(data.rainfall_1h, data.rainfall_3h, data.rainfall_6h);

      // Trigger Waterlogging & Population Priority & Alerts Refresh
      loadWaterloggingLayer();
      loadPopulationPriorityLayer();
      loadAlertsPanel();
    });
  });
}

function updateScenarioSummaryCard(r1h, r3h, r6h) {
  const sum1h = document.getElementById("sim-sum-1h");
  const sum3h = document.getElementById("sim-sum-3h");
  const sum6h = document.getElementById("sim-sum-6h");

  if (sum1h) sum1h.textContent = typeof r1h === "number" ? r1h.toFixed(1) : r1h;
  if (sum3h) sum3h.textContent = typeof r3h === "number" ? r3h.toFixed(1) : r3h;
  if (sum6h) sum6h.textContent = typeof r6h === "number" ? r6h.toFixed(1) : r6h;
}

function renderScenarioLog() {
  const listEl = document.getElementById("scenario-log-list");
  if (!listEl) return;

  if (scenarioHistory.length === 0) {
    listEl.innerHTML = '<div class="log-empty-msg">Run simulations to compare scenario outcomes...</div>';
    return;
  }

  listEl.innerHTML = scenarioHistory
    .slice()
    .reverse()
    .map(
      (item) => `
    <div class="log-item">
      <div class="log-top">
        <span class="log-name">${item.preset} SCENARIO</span>
        <span class="log-time">${item.timestamp}</span>
      </div>
      <div class="log-details">
        <span>Rain: ${item.r1h}/${item.r3h}/${item.r6h} mm</span>
        <span class="log-sev ${item.severity.toLowerCase()}">${item.severity}</span>
        <span class="log-score">Score: ${item.score}/100</span>
      </div>
    </div>
  `
    )
    .join("");
}

function initLayerToggles() {
  const wCheck = document.getElementById("layer-waterlogging-check");
  if (wCheck) {
    wCheck.addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!map.hasLayer(waterloggingLayer)) map.addLayer(waterloggingLayer);
        loadWaterloggingLayer();
      } else {
        if (map.hasLayer(waterloggingLayer)) map.removeLayer(waterloggingLayer);
      }
    });
  }

  const iCheck = document.getElementById("layer-infra-check");
  if (iCheck) {
    iCheck.addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!map.hasLayer(infraLayer)) map.addLayer(infraLayer);
        loadInfraLayer();
      } else {
        if (map.hasLayer(infraLayer)) map.removeLayer(infraLayer);
      }
    });
  }

  const pCheck = document.getElementById("layer-pumps-check");
  if (pCheck) {
    pCheck.addEventListener("change", (e) => {
      handlePumpsToggle(e.target.checked);
    });
  }

  const dCheck = document.getElementById("layer-drains-check");
  if (dCheck) {
    dCheck.addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!map.hasLayer(drainageNetworkLayer)) map.addLayer(drainageNetworkLayer);
        loadDrainageNetworkLayer();
      } else {
        if (map.hasLayer(drainageNetworkLayer)) map.removeLayer(drainageNetworkLayer);
      }
    });
  }

  const zCheck = document.getElementById("layer-zones-check");
  if (zCheck) {
    zCheck.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(zonesLayer);
      else map.removeLayer(zonesLayer);
    });
  }

  const rCheck = document.getElementById("layer-route-check");
  if (rCheck) {
    rCheck.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(routeLayer);
      else map.removeLayer(routeLayer);
    });
  }

  const popCheck = document.getElementById("layer-pop-priority-check");
  if (popCheck) {
    popCheck.addEventListener("change", (e) => {
      if (e.target.checked) {
        if (!map.hasLayer(populationPriorityLayer)) map.addLayer(populationPriorityLayer);
        loadPopulationPriorityLayer();
      } else {
        if (map.hasLayer(populationPriorityLayer)) map.removeLayer(populationPriorityLayer);
      }
    });
  }
}

function initTimelineBar() {
  const stepBtns = document.querySelectorAll(".time-step-btn");
  const stepMap = {
    "now": "T+0",
    "+30m": "T+1",
    "+1h": "T+1",
    "+2h": "T+2",
    "+3h": "T+3"
  };

  stepBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const rawStep = btn.getAttribute("data-step");
      const mappedTimestep = stepMap[rawStep] || "T+0";

      stepBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      activeTimestep = mappedTimestep;
      loadWaterloggingLayer();
      loadPopulationPriorityLayer();
      loadAlertsPanel();
    });
  });
}

function initInspectorCard() {
  const closeBtn = document.getElementById("btn-close-inspector");
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      document.getElementById("inspector-card").classList.add("hidden");
    });
  }
}

// --------------------------------------------------------------------------
// GET /health — Backend System Health Telemetry
// --------------------------------------------------------------------------
async function checkSystemHealth() {
  const healthBadge = document.getElementById("status-health-badge");
  const healthText = document.getElementById("health-status-text");

  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("Health check failed");
    const data = await res.json();

    if (data.status === "ok") {
      if (healthBadge) healthBadge.className = "telemetry-pill online";
      if (healthText) healthText.textContent = "System Online";

      const tModel = document.getElementById("t-model");
      const tRouter = document.getElementById("t-router");
      if (tModel) tModel.textContent = data.model_version || "AquaG Model V2";
      if (tRouter) tRouter.textContent = data.router_loaded ? "AquaGraph A*" : "Loaded";
    } else {
      throw new Error("System unhealthy");
    }
  } catch (err) {
    if (healthBadge) healthBadge.className = "telemetry-pill offline";
    if (healthText) healthText.textContent = "Backend Offline";
  }
}

// --------------------------------------------------------------------------
// POST /flood_info — Point GIS Inspector
// --------------------------------------------------------------------------
async function handleMapClick(e) {
  const lat = e.latlng.lat;
  const lon = e.latlng.lng;

  // Handle Pick Mode for Route Panel
  if (pickingMode === "start") {
    const sLat = document.getElementById("start_lat");
    const sLon = document.getElementById("start_lon");
    if (sLat) sLat.value = lat.toFixed(6);
    if (sLon) sLon.value = lon.toFixed(6);
    pickingMode = null;
    map.getContainer().style.cursor = "";
    return;
  }
  if (pickingMode === "end") {
    const eLat = document.getElementById("end_lat");
    const eLon = document.getElementById("end_lon");
    if (eLat) eLat.value = lat.toFixed(6);
    if (eLon) eLon.value = lon.toFixed(6);
    pickingMode = null;
    map.getContainer().style.cursor = "";
    return;
  }

  // Show Inspector Card Loading
  const card = document.getElementById("inspector-card");
  document.getElementById("insp-coords").textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)}`;
  document.getElementById("insp-elevation").textContent = "Querying...";
  document.getElementById("insp-drain-name").textContent = "Querying...";
  document.getElementById("insp-drain-dist").textContent = "Querying...";
  if (card) card.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE_URL}/flood_info`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon }),
    });

    if (!res.ok) throw new Error("Flood info query failed");
    const data = await res.json();

    const elevText = data.elevation_m !== null ? `${data.elevation_m} m` : "Out of DEM bounds";
    const drainDist = data.nearest_drain_distance_m !== null ? `${data.nearest_drain_distance_m} m` : "N/A";
    const drainName = data.nearest_drain ? (data.nearest_drain.drain_name || `Drain #${data.nearest_drain.drain_id}`) : "MPD-1976 Drain";

    document.getElementById("insp-elevation").textContent = elevText;
    document.getElementById("insp-drain-name").textContent = drainName;
    document.getElementById("insp-drain-dist").textContent = drainDist;
    document.getElementById("insp-basis").textContent = data.risk_basis || "spatial_proxy";
  } catch (err) {
    document.getElementById("insp-elevation").textContent = "Query Failed";
    document.getElementById("insp-drain-name").textContent = err.message;
  }
}

// --------------------------------------------------------------------------
// POST /predict — Model V2 & Action Priority Engine
// --------------------------------------------------------------------------
function initFormHandlers() {
  const predictForm = document.getElementById("predict-form");
  const routeForm = document.getElementById("route-form");

  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-predict");
      const spinner = btn ? btn.querySelector(".btn-spinner") : null;
      const btnText = btn ? btn.querySelector("span:not(.btn-spinner)") : null;
      const resultsPanel = document.getElementById("predict-results-panel");

      if (spinner) spinner.classList.remove("hidden");
      if (btnText) btnText.textContent = "Running scenario...";
      if (btn) btn.disabled = true;

      const payload = {
        rainfall_1h: parseFloat(document.getElementById("rainfall_1h").value) || 0,
        rainfall_3h: parseFloat(document.getElementById("rainfall_3h").value) || 0,
        rainfall_6h: parseFloat(document.getElementById("rainfall_6h").value) || 0,
        recent_rainfall_intensity: parseFloat(document.getElementById("recent_rainfall_intensity").value) || 0,
        elevation: parseFloat(document.getElementById("elevation").value) || 0,
        slope: parseFloat(document.getElementById("slope").value) || 0,
        distance_to_drain: parseFloat(document.getElementById("distance_to_drain").value) || 0,
        distance_to_road: parseFloat(document.getElementById("distance_to_road").value) || 0,
        distance_to_infra: parseFloat(document.getElementById("distance_to_infra").value) || 0,
        population_total: parseFloat(document.getElementById("population_total").value) || 0,
        critical_infra_flag: parseInt(document.getElementById("critical_infra_flag").value, 10) || 0,
      };

      // Update Summary Card
      updateScenarioSummaryCard(payload.rainfall_1h, payload.rainfall_3h, payload.rainfall_6h);

      try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error("Scenario simulation unavailable. Please retry.");
        const data = await res.json();

        // Update Severity Badge
        const sevBadge = document.getElementById("res-severity-badge");
        if (sevBadge) {
          sevBadge.textContent = data.flood_severity;
          sevBadge.className = `sev-badge ${data.flood_severity.toLowerCase()}`;
        }

        // Update Probabilities Distribution
        if (data.probabilities) {
          const probs = data.probabilities;
          const lowP = Math.round((probs.Low || 0) * 100);
          const medP = Math.round((probs.Medium || 0) * 100);
          const highP = Math.round((probs.High || 0) * 100);

          const bLow = document.getElementById("p-bar-low");
          const vLow = document.getElementById("p-val-low");
          if (bLow) bLow.style.width = `${lowP}%`;
          if (vLow) vLow.textContent = `${lowP}%`;

          const bMed = document.getElementById("p-bar-medium");
          const vMed = document.getElementById("p-val-medium");
          if (bMed) bMed.style.width = `${medP}%`;
          if (vMed) vMed.textContent = `${medP}%`;

          const bHigh = document.getElementById("p-bar-high");
          const vHigh = document.getElementById("p-val-high");
          if (bHigh) bHigh.style.width = `${highP}%`;
          if (vHigh) vHigh.textContent = `${highP}%`;
        }

        // Update Action Priority Engine Outcome
        if (data.action_priority) {
          const act = data.action_priority;
          const pScore = document.getElementById("res-priority-score");
          if (pScore) pScore.textContent = `Score: ${act.priority_score}`;
          
          const actLvl = document.getElementById("res-action-level");
          if (actLvl) {
            actLvl.textContent = act.action_level;
            actLvl.className = `act-level ${act.action_level.toLowerCase().replace(" ", "-")}`;
          }

          const actDesc = document.getElementById("res-action-desc");
          if (actDesc) actDesc.textContent = act.recommended_action;
        }

        // Add to Scenario Comparison Log
        scenarioHistory.push({
          preset: activePresetName,
          r1h: payload.rainfall_1h,
          r3h: payload.rainfall_3h,
          r6h: payload.rainfall_6h,
          severity: data.flood_severity,
          score: data.action_priority ? data.action_priority.priority_score : "--",
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        });
        if (scenarioHistory.length > 5) scenarioHistory.shift();
        renderScenarioLog();

        activeScenario = activePresetName || "NORMAL";
        loadWaterloggingLayer();

        if (resultsPanel) resultsPanel.classList.remove("hidden");
      } catch (err) {
        alert(`Scenario Simulation Failed: ${err.message}`);
      } finally {
        if (spinner) spinner.classList.add("hidden");
        if (btnText) btnText.textContent = "RUN SCENARIO SIMULATION";
        if (btn) btn.disabled = false;
      }
    });
  }

  // --------------------------------------------------------------------------
  // POST /route — AquaGraph A* Risk Router
  // --------------------------------------------------------------------------
  if (routeForm) {
    routeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-route");
      const spinner = btn ? btn.querySelector(".btn-spinner") : null;
      const resultsPanel = document.getElementById("route-results-panel");

      if (spinner) spinner.classList.remove("hidden");
      if (btn) btn.disabled = true;

      const r1hEl = document.getElementById("rainfall_1h");
      const r3hEl = document.getElementById("rainfall_3h");
      const r6hEl = document.getElementById("rainfall_6h");
      const intEl = document.getElementById("recent_rainfall_intensity");

      const payload = {
        start_lat: parseFloat(document.getElementById("start_lat").value),
        start_lon: parseFloat(document.getElementById("start_lon").value),
        end_lat: parseFloat(document.getElementById("end_lat").value),
        end_lon: parseFloat(document.getElementById("end_lon").value),
        risk: document.getElementById("route_risk").value,
        scenario: activeScenario,
        timestep: activeTimestep,
        rainfall_1h: r1hEl ? parseFloat(r1hEl.value) || 10.0 : 10.0,
        rainfall_3h: r3hEl ? parseFloat(r3hEl.value) || 20.0 : 20.0,
        rainfall_6h: r6hEl ? parseFloat(r6hEl.value) || 30.0 : 30.0,
        recent_rainfall_intensity: intEl ? parseFloat(intEl.value) || 5.0 : 5.0,
        flood_aware: true,
      };

      try {
        const res = await fetch(`${API_BASE_URL}/route`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error("Route API call failed");
        const data = await res.json();

        if (data.status === "error") {
          throw new Error(data.message || "Route calculation failed");
        }

        // Update Route Metrics
        const tag = document.getElementById("route-status-tag");
        if (tag) {
          tag.textContent = "OK";
          tag.className = "ver-tag ok";
        }

        const distVal = data.physical_distance_m || data.distance_m || 0;
        const distKm = distVal > 1000 ? `${(distVal / 1000).toFixed(2)} km` : `${distVal.toFixed(1)} m`;

        document.getElementById("route-dist").textContent = distKm;
        document.getElementById("route-cost").textContent = `${data.routing_cost || data.estimated_cost}`;
        document.getElementById("route-nodes").textContent = `${data.nodes_in_route}`;
        document.getElementById("route-sensitivity").textContent = `${data.risk_mode || data.risk_level}`;
        document.getElementById("route-snap-orig").textContent = `${data.origin_snap_distance_m} m`;
        document.getElementById("route-snap-dest").textContent = `${data.destination_snap_distance_m} m`;
        
        const elMaxDepth = document.getElementById("route-max-depth");
        const elFloodedCnt = document.getElementById("route-flooded-count");
        const elRiskLvl = document.getElementById("route-risk-level");
        const elForecastTag = document.getElementById("route-forecast-tag");
        const elAvoidedBanner = document.getElementById("route-avoided-banner");
        const elRiskBasis = document.getElementById("route-risk-basis");

        if (elMaxDepth) elMaxDepth.textContent = `${data.maximum_water_depth_cm || 0} cm`;
        if (elFloodedCnt) elFloodedCnt.textContent = `${data.flooded_segments_on_route || 0}`;
        if (elRiskLvl) elRiskLvl.textContent = `${data.route_risk_level || 'Low'}`;
        if (elForecastTag) elForecastTag.textContent = `${data.scenario || activeScenario} (${data.timestep || activeTimestep})`;
        if (elRiskBasis) elRiskBasis.textContent = data.basis || data.risk_basis || "model_derived_flood_aware_routing";

        if (elAvoidedBanner) {
          if (data.avoided_high_risk_segments && data.avoided_high_risk_segments > 0) {
            elAvoidedBanner.textContent = `AquaGraph avoided ${data.avoided_high_risk_segments} high-risk road segments.`;
            elAvoidedBanner.classList.remove("hidden");
          } else {
            elAvoidedBanner.classList.add("hidden");
          }
        }

        // Clear previous route overlays
        routeLayer.clearLayers();
        markersLayer.clearLayers();

        // Render Route Polyline
        const coords = data.coordinates || [];
        if (coords.length > 0) {
          const polyline = L.polyline(coords, {
            color: "#06b6d4",
            weight: 5,
            opacity: 0.85,
            lineJoin: "round",
          }).addTo(routeLayer);

          const popupContent = `
            <div class="waterlogging-popup">
              <div class="wl-popup-title">Model-Derived Flood-Aware Route</div>
              <div class="wl-popup-row"><span>Distance:</span> <strong>${distKm}</strong></div>
              <div class="wl-popup-row"><span>Routing Cost:</span> <strong>${data.routing_cost}</strong></div>
              <div class="wl-popup-row"><span>Max Water Depth:</span> <strong>${data.maximum_water_depth_cm || 0} cm</strong></div>
              <div class="wl-popup-row"><span>Flooded Segments:</span> <strong>${data.flooded_segments_on_route || 0}</strong></div>
              <div class="wl-popup-row"><span>Route Risk:</span> <strong>${data.route_risk_level || 'Low'}</strong></div>
              <div class="wl-popup-row"><span>Forecast:</span> <strong>${data.scenario} ${data.timestep}</strong></div>
              <div class="wl-popup-footer">Risk-aware flood routing (Model-derived proxy)</div>
            </div>
          `;
          polyline.bindPopup(popupContent, { className: "dark-leaflet-popup" });

          const startPt = coords[0];
          const endPt = coords[coords.length - 1];

          L.circleMarker(startPt, { radius: 7, color: "#10b981", fillColor: "#10b981", fillOpacity: 0.9 })
            .bindPopup("Route Origin")
            .addTo(markersLayer);

          L.circleMarker(endPt, { radius: 7, color: "#ef4444", fillColor: "#ef4444", fillOpacity: 0.9 })
            .bindPopup("Route Destination")
            .addTo(markersLayer);

          map.fitBounds(polyline.getBounds(), { padding: [40, 40] });
        }

        if (resultsPanel) resultsPanel.classList.remove("hidden");
      } catch (err) {
        alert(`Routing Failed: ${err.message}`);
      } finally {
        if (spinner) spinner.classList.add("hidden");
        if (btn) btn.disabled = false;
      }
    });
  }

  const clearBtn = document.getElementById("btn-clear-route");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      routeLayer.clearLayers();
      markersLayer.clearLayers();
      const panel = document.getElementById("route-results-panel");
      if (panel) panel.classList.add("hidden");
    });
  }

  const pickStartBtn = document.getElementById("btn-pick-start");
  if (pickStartBtn) {
    pickStartBtn.addEventListener("click", () => {
      pickingMode = "start";
      map.getContainer().style.cursor = "crosshair";
      alert("Click anywhere on map to set Origin coordinates.");
    });
  }

  const pickEndBtn = document.getElementById("btn-pick-end");
  if (pickEndBtn) {
    pickEndBtn.addEventListener("click", () => {
      pickingMode = "end";
      map.getContainer().style.cursor = "crosshair";
      alert("Click anywhere on map to set Destination coordinates.");
    });
  }
}

// --------------------------------------------------------------------------
// GET /zones — Sample Zone Severity GIS Layer
// --------------------------------------------------------------------------
async function loadZoneLayer() {
  try {
    const res = await fetch(`${API_BASE_URL}/zones`);
    if (!res.ok) return;
    const zones = await res.json();

    zonesLayer.clearLayers();

    zones.forEach((zone) => {
      const color =
        zone.severity === "High" ? "#ef4444" :
        zone.severity === "Medium" ? "#f59e0b" : "#10b981";

      if (zone.latitude !== 0.0 && zone.longitude !== 0.0) {
        L.circle([zone.latitude, zone.longitude], {
          radius: 300,
          color: color,
          fillColor: color,
          fillOpacity: 0.35,
        })
          .bindPopup(`<b>Zone:</b> ${zone.zone_id}<br/><b>Severity:</b> ${zone.severity}`)
          .addTo(zonesLayer);
      }
    });
  } catch (err) {
    console.log("Zone layer skipped:", err.message);
  }
}
