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

function getApiBaseUrl() {
  if (typeof window !== "undefined" && window.AQUAG_API_URL) {
    return window.AQUAG_API_URL;
  }
  if (typeof window !== "undefined" && window.location) {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;
    if (
      hostname === "localhost" ||
      hostname === "127.0.0.1" ||
      hostname === "0.0.0.0" ||
      hostname === "::1" ||
      hostname === "" ||
      protocol === "file:" ||
      hostname.startsWith("192.168.") ||
      hostname.startsWith("10.") ||
      hostname.startsWith("172.")
    ) {
      return "http://127.0.0.1:8001";
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
let isSmartRouterActive = false;

let pickingMode = null; // 'start' or 'end'
let activeScenario = "NORMAL";
let activeTimestep = "T+0";
let activePresetName = "NORMAL";

let spatialDebounceTimer = null;
const scenarioHistory = [];

let lastLoadedWaterloggingGeojson = null;
let lastLoadedInfraGeojson = null;
let lastLoadedAlertsData = null;

function updateScadaTelemetry() {
  const tsPill = document.getElementById("panel-active-timestep");
  if (tsPill) {
    let displayTs = activeTimestep;
    if (activeTimestep === "T+0") displayTs = "T+0 (NOW)";
    else if (activeTimestep === "T+1") displayTs = "+30m / +1h";
    else if (activeTimestep === "T+2") displayTs = "+2h";
    else if (activeTimestep === "T+3") displayTs = "+3h";
    tsPill.textContent = displayTs;
  }

  let maxDepth = 0.0;
  let floodedCount = 0;

  if (lastLoadedWaterloggingGeojson && lastLoadedWaterloggingGeojson.features) {
    lastLoadedWaterloggingGeojson.features.forEach((feat) => {
      const depth = feat.properties ? (feat.properties.water_depth_cm || 0) : 0;
      if (depth > maxDepth) maxDepth = depth;
      if (depth > 10.0) floodedCount++;
    });
  }

  const maxDepthEl = document.getElementById("scada-max-depth");
  const floodedCountEl = document.getElementById("scada-flooded-roads");
  if (maxDepthEl) maxDepthEl.textContent = `${maxDepth.toFixed(1)} cm`;
  if (floodedCountEl) floodedCountEl.textContent = floodedCount.toLocaleString();

  const statusBadge = document.getElementById("scada-overall-status");
  if (statusBadge) {
    let statusText = "NORMAL";
    let statusClass = "normal";

    if (maxDepth > 100.0) {
      statusText = "CRITICAL";
      statusClass = "critical";
    } else if (maxDepth > 25.0) {
      statusText = "HIGH";
      statusClass = "high";
    } else if (maxDepth > 10.0) {
      statusText = "WATCH";
      statusClass = "watch";
    }

    statusBadge.textContent = statusText;
    statusBadge.className = `scada-status-badge ${statusClass}`;
  }

  let infraCount = 0;
  if (lastLoadedInfraGeojson && lastLoadedInfraGeojson.features) {
    infraCount = lastLoadedInfraGeojson.features.filter(
      (f) => f.properties && f.properties.critical_asset !== false && f.properties.category !== "other"
    ).length;
  } else if (infraLayer && typeof infraLayer.getLayers === "function") {
    infraCount = infraLayer.getLayers().length;
  }
  const infraEl = document.getElementById("scada-infra-count");
  if (infraEl) infraEl.textContent = infraCount.toLocaleString();

  let alertsCount = 0;
  const alertsListEl = document.getElementById("scada-alerts-list");

  if (lastLoadedAlertsData && lastLoadedAlertsData.incidents) {
    const incs = lastLoadedAlertsData.incidents;
    alertsCount = incs.length;

    if (alertsListEl) {
      if (incs.length === 0) {
        alertsListEl.innerHTML = '<div class="alert-item-mini ok">● Standard operational monitoring active</div>';
      } else {
        const top3 = incs.slice(0, 3);
        alertsListEl.innerHTML = top3.map((inc) => {
          const pLvl = (inc.priority_level || "LOW").toLowerCase();
          const cls = pLvl === "critical" || pLvl === "high" ? "crit" : pLvl === "medium" ? "warn" : "ok";
          return `<div class="alert-item-mini ${cls}" title="${inc.road_id}: ${inc.recommended_action}">● ${inc.priority_level}: ${inc.road_id} (${inc.water_depth_cm.toFixed(0)}cm)</div>`;
        }).join("");
      }
    }
  } else if (alertsListEl) {
    alertsListEl.innerHTML = '<div class="alert-item-mini ok">● Standard operational monitoring active</div>';
  }

  const alertsCountEl = document.getElementById("scada-alerts-count");
  if (alertsCountEl) alertsCountEl.textContent = alertsCount.toLocaleString();

  const kpiTotal = document.getElementById("alerts-kpi-total");
  const kpiInfra = document.getElementById("alerts-kpi-infra");
  const kpiP1 = document.getElementById("alerts-kpi-p1");
  const kpiStatus = document.getElementById("alerts-kpi-status");

  if (kpiTotal) kpiTotal.textContent = alertsCount.toLocaleString();
  if (kpiInfra) kpiInfra.textContent = infraCount.toLocaleString();

  const cntP1El = document.getElementById("cnt-p1");
  if (kpiP1 && cntP1El) kpiP1.textContent = cntP1El.textContent;

  if (kpiStatus && statusBadge) {
    kpiStatus.textContent = statusBadge.textContent;
  }
}

// --------------------------------------------------------------------------
// Dual Theme Controller (Bright Mode & Dark Command Center)
// --------------------------------------------------------------------------
const STORAGE_KEY_THEME = "aquag-theme";

function applyTheme(themeName) {
  const theme = themeName === "dark" ? "dark" : "bright";
  
  if (theme === "dark") {
    document.body.classList.remove("theme-bright");
    document.body.classList.add("theme-dark");
  } else {
    document.body.classList.remove("theme-dark");
    document.body.classList.add("theme-bright");
  }

  const btnBright = document.getElementById("btn-theme-bright");
  const btnDark = document.getElementById("btn-theme-dark");

  if (btnBright && btnDark) {
    if (theme === "dark") {
      btnBright.classList.remove("active");
      btnDark.classList.add("active");
    } else {
      btnDark.classList.remove("active");
      btnBright.classList.add("active");
    }
  }

  try {
    localStorage.setItem(STORAGE_KEY_THEME, theme);
  } catch (e) {
    console.warn("Theme preference storage notice:", e);
  }
}

function initThemeSystem() {
  let savedTheme = "bright";
  try {
    savedTheme = localStorage.getItem(STORAGE_KEY_THEME) || "bright";
  } catch (e) {
    savedTheme = "bright";
  }

  if (savedTheme !== "dark" && savedTheme !== "bright") {
    savedTheme = "bright";
  }

  applyTheme(savedTheme);

  const btnBright = document.getElementById("btn-theme-bright");
  const btnDark = document.getElementById("btn-theme-dark");

  if (btnBright) {
    btnBright.onclick = () => applyTheme("bright");
  }
  if (btnDark) {
    btnDark.onclick = () => applyTheme("dark");
  }
}

// --------------------------------------------------------------------------
// Initialization
// --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  initThemeSystem();
  initTopNavModuleButtons();
  initMap();
  initSideTabs();
  initPresetButtons();
  initFormHandlers();
  initLayerToggles();
  initWaterloggingCategoryFilters();
  initTimelineBar();
  initInspectorCard();
  initTopPriorityFocusHandler();
  initSmartRouterState();

  // Initial Backend API Calls
  checkSystemHealth();
  loadZoneLayer();
  loadWaterloggingLayer(); // Auto-load street-level waterlogging for current viewport (NORMAL T+0)
  loadDrainageNetworkLayer(); // Auto-load MPD-1976 drainage network layer
  loadInfraLayer(); // Auto-load critical infrastructure layer
  loadPopulationPriorityLayer(); // Auto-load population priority layer
  loadAlertsPanel(); // Auto-load model-derived forecast incidents for current viewport
  updateScadaTelemetry();
});

// --------------------------------------------------------------------------
// Leaflet Map Initialization
// --------------------------------------------------------------------------
function initMap() {
  // Center on Delhi coordinates [28.6139, 77.2090] bounded by Delhi/NCR study domain
  const delhiBounds = L.latLngBounds(
    L.latLng(28.35, 76.75), // SW margin
    L.latLng(28.95, 77.45)  // NE margin
  );

  map = L.map("map", {
    center: [28.6139, 77.2090],
    zoom: 12,
    minZoom: 10,
    maxZoom: 19,
    maxBounds: delhiBounds,
    maxBoundsViscosity: 0.8,
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
  });
  if (document.getElementById("layer-waterlogging-check")?.checked) waterloggingLayer.addTo(map);

  // MarkerCluster Group for Critical Infrastructure to eliminate low-zoom clutter
  if (typeof L.markerClusterGroup === "function") {
    infraLayer = L.markerClusterGroup({
      maxClusterRadius: 45,
      showCoverageOnHover: false,
      disableClusteringAtZoom: 15,
      spiderfyOnMaxZoom: true,
    });
  } else {
    infraLayer = L.geoJSON(null, {
      pointToLayer: createInfraMarker,
      onEachFeature: bindInfraPopup,
    });
  }
  if (document.getElementById("layer-infra-check")?.checked) infraLayer.addTo(map);

  drainageNetworkLayer = L.geoJSON(null, {
    pointToLayer: createDrainMarker,
    onEachFeature: function (feature, layer) {
        const popupContent = bindDrainPopupContent(feature.properties || {});
        layer.bindPopup(popupContent, {
            className: "dark-leaflet-popup"
        });
    },
});
  if (document.getElementById("layer-drains-check")?.checked) drainageNetworkLayer.addTo(map);

  zonesLayer = L.layerGroup();
  if (document.getElementById("layer-zones-check")?.checked) zonesLayer.addTo(map);

  drainsLayer = L.layerGroup().addTo(map);

  routeLayer = L.layerGroup();
  if (document.getElementById("layer-route-check")?.checked) routeLayer.addTo(map);

  markersLayer = L.layerGroup().addTo(map);

  populationPriorityLayer = L.layerGroup();
  if (document.getElementById("layer-pop-priority-check")?.checked) populationPriorityLayer.addTo(map);

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
    if (document.getElementById("layer-waterlogging-check")?.checked) loadWaterloggingLayer();
    if (document.getElementById("layer-infra-check")?.checked) loadInfraLayer();
    if (document.getElementById("layer-drains-check")?.checked) loadDrainageNetworkLayer();
    if (document.getElementById("layer-pop-priority-check")?.checked) loadPopulationPriorityLayer();
    loadAlertsPanel();
  }, 300);
}

// --------------------------------------------------------------------------
// Waterlogging Category Filter State
// --------------------------------------------------------------------------
let filterMedium = true;
let filterHigh = true;
let filterCritical = true;

function initWaterloggingCategoryFilters() {
  const mCheck = document.getElementById("filter-wl-medium");
  const hCheck = document.getElementById("filter-wl-high");
  const cCheck = document.getElementById("filter-wl-critical");

  const btnAll = document.getElementById("filter-btn-all");
  const btnHigh = document.getElementById("filter-btn-high");
  const btnCrit = document.getElementById("filter-btn-crit");

  const updateFilters = () => {
    filterMedium = mCheck ? mCheck.checked : true;
    filterHigh = hCheck ? hCheck.checked : true;
    filterCritical = cCheck ? cCheck.checked : true;

    if (btnAll && btnHigh && btnCrit) {
      btnAll.classList.toggle("active", filterMedium && filterHigh && filterCritical);
      btnHigh.classList.toggle("active", !filterMedium && filterHigh && filterCritical);
      btnCrit.classList.toggle("active", !filterMedium && !filterHigh && filterCritical);
    }

    applyWaterloggingFilter();
  };

  mCheck?.addEventListener("change", updateFilters);
  hCheck?.addEventListener("change", updateFilters);
  cCheck?.addEventListener("change", updateFilters);

  btnAll?.addEventListener("click", () => {
    if (mCheck) mCheck.checked = true;
    if (hCheck) hCheck.checked = true;
    if (cCheck) cCheck.checked = true;
    updateFilters();
  });

  btnHigh?.addEventListener("click", () => {
    if (mCheck) mCheck.checked = false;
    if (hCheck) hCheck.checked = true;
    if (cCheck) cCheck.checked = true;
    updateFilters();
  });

  btnCrit?.addEventListener("click", () => {
    if (mCheck) mCheck.checked = false;
    if (hCheck) hCheck.checked = false;
    if (cCheck) cCheck.checked = true;
    updateFilters();
  });
}

function applyWaterloggingFilter() {
  if (waterloggingLayer) {
    waterloggingLayer.setStyle(getRoadStyle);
  }
}

// --------------------------------------------------------------------------
// 1. Waterlogging Layer Engine (POST /waterlogging)
// --------------------------------------------------------------------------
function getRoadColor(depth) {
  if (depth > 100) return "#ef4444"; // >100 cm: Critical / Red
  if (depth > 25) return "#f97316";  // >25-100 cm: High / Orange
  if (depth > 10) return "#eab308";  // >10-25 cm: Medium / Yellow
  return "transparent";              // 0-10 cm: Normal / Base-map road (uncoloured)
}

function getRoadStyle(feature) {
  const depth = feature.properties ? (feature.properties.water_depth_cm || 0) : 0;

  // 1. NORMAL (0-10 cm): Normal base-map road, NO flood color at all
  if (depth <= 10.0) {
    return {
      color: "transparent",
      weight: 0,
      opacity: 0,
      fillOpacity: 0
    };
  }

  // 2. MEDIUM / LOW FLOODING (>10-25 cm): Yellow highlight
  if (depth <= 25.0) {
    if (!filterMedium) {
      return { color: "transparent", weight: 0, opacity: 0, fillOpacity: 0 };
    }
    return {
      color: "#eab308",
      weight: 3.5,
      opacity: 0.85,
      lineCap: "round",
      lineJoin: "round"
    };
  }

  // 3. HIGH FLOODING (>25-100 cm): Orange highlight
  if (depth <= 100.0) {
    if (!filterHigh) {
      return { color: "transparent", weight: 0, opacity: 0, fillOpacity: 0 };
    }
    return {
      color: "#f97316",
      weight: 5.5,
      opacity: 0.95,
      lineCap: "round",
      lineJoin: "round"
    };
  }

  // 4. CRITICAL FLOODING (>100 cm): Red highlight
  if (!filterCritical) {
    return { color: "transparent", weight: 0, opacity: 0, fillOpacity: 0 };
  }
  return {
    color: "#ef4444",
    weight: 7.5,
    opacity: 1.0,
    lineCap: "round",
    lineJoin: "round"
  };
}

function bindRoadPopup(feature, layer) {
  const p = feature.properties || {};
  const roadId = p.road_id || "Unknown Segment";
  const depthVal = typeof p.water_depth_cm === "number" ? p.water_depth_cm.toFixed(1) : (p.water_depth_cm || "0.0");
  const depthNum = typeof p.water_depth_cm === "number" ? p.water_depth_cm : parseFloat(p.water_depth_cm || 0);

  let displaySeverity = "Normal";
  let displayBand = "0-10 cm";
  
  if (depthNum <= 10.0) {
    displaySeverity = "Normal";
    displayBand = "0-10 cm";
  } else if (depthNum <= 25.0) {
    displaySeverity = "Medium";
    displayBand = ">10-25 cm";
  } else if (depthNum <= 100.0) {
    displaySeverity = "High";
    displayBand = ">25-100 cm";
  } else {
    displaySeverity = "Critical";
    displayBand = ">100 cm";
  }

  const elevation = typeof p.elevation_m === "number" ? `${p.elevation_m.toFixed(1)} m` : "N/A";
  const drainDist = typeof p.distance_to_drain_m === "number" ? `${p.distance_to_drain_m.toFixed(1)} m` : "N/A";
  const popExp = typeof p.population_exposure === "number" ? p.population_exposure.toLocaleString() : "N/A";
  const critInfra = p.critical_infra_flag ? "Yes" : "No";
  const forecastTimestep = p.timestep || activeTimestep;

  const content = `
    <div class="waterlogging-popup">
      <div class="wl-popup-title">${roadId}</div>
      <div class="wl-popup-row"><span>Water depth:</span> <strong>${depthVal} cm</strong></div>
      <div class="wl-popup-row"><span>Severity:</span> <span class="sev-tag ${displaySeverity.toLowerCase()}">${displaySeverity}</span></div>
      <div class="wl-popup-row"><span>Band:</span> <strong>${displayBand}</strong></div>
      <div class="wl-popup-row"><span>Elevation:</span> <strong>${elevation}</strong></div>
      <div class="wl-popup-row"><span>Drain distance:</span> <strong>${drainDist}</strong></div>
      <div class="wl-popup-row"><span>Population exposure:</span> <strong>${popExp}</strong></div>
      <div class="wl-popup-row"><span>Critical infrastructure:</span> <strong>${critInfra}</strong></div>
      <div class="wl-popup-row"><span>Forecast:</span> <strong>${forecastTimestep}</strong></div>
      <div class="wl-popup-action-row" style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.1);">
        <button class="${isSmartRouterActive ? 'btn-open-smart-router font-mono active-on' : 'btn-open-smart-router font-mono'}" style="width:100%; padding:5px 8px; background:${isSmartRouterActive ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #0284c7, #06b6d4)'}; border:none; border-radius:4px; color:#fff; font-size:0.72rem; font-weight:700; cursor:pointer;">${isSmartRouterActive ? '⚡ Smart Router ON' : '⚡ Launch Smart Router'}</button>
      </div>
      <div class="wl-popup-footer" style="margin-top:4px;">Model-derived waterlogging depth proxy</div>
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
      lastLoadedWaterloggingGeojson = geojson;
      updateScadaTelemetry();
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
  const cat = (p.category || "").toLowerCase();

  let iconChar = "⚡";
  let bgClass = "power";

  if (cat === "medical" || cat === "hospital") { iconChar = "🏥"; bgClass = "medical"; }
  else if (cat === "police") { iconChar = "🛡️"; bgClass = "police"; }
  else if (cat === "fire" || cat === "emergency") { iconChar = "🚨"; bgClass = "fire"; }
  else if (cat === "transport" || cat === "metro" || cat === "railway") { iconChar = "🚇"; bgClass = "metro"; }
  else if (cat === "water/utility" || cat === "water" || cat === "water_drainage") { iconChar = "💧"; bgClass = "water"; }

  const html = `<div class="infra-map-badge ${bgClass} is-critical">${iconChar}</div>`;
  const customIcon = L.divIcon({
    html: html,
    className: "infra-div-icon",
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });

  return L.marker(latlng, { icon: customIcon });
}

function bindInfraPopup(feature, layer) {
  const p = feature.properties || {};
  const name = p.name || "Critical Infrastructure Facility";
  
  const catRaw = (p.category || "General").toUpperCase();
  let catDisplay = "CRITICAL UTILITY";
  if (catRaw === "HOSPITAL" || catRaw === "MEDICAL") catDisplay = "MEDICAL";
  else if (catRaw === "POLICE") catDisplay = "POLICE";
  else if (catRaw === "FIRE" || catRaw === "EMERGENCY") catDisplay = "FIRE";
  else if (catRaw === "TRANSPORT" || catRaw === "METRO" || catRaw === "RAILWAY") catDisplay = "TRANSPORT";
  else if (catRaw === "POWER") catDisplay = "POWER";
  else if (catRaw === "WATER/UTILITY" || catRaw === "WATER") catDisplay = "WATER / UTILITY";

  const forecastStatus = p.forecast_status || "Forecast Threatened";
  const nearestDepth = typeof p.nearest_water_depth_cm === "number" ? `${p.nearest_water_depth_cm.toFixed(1)} cm` : "N/A";
  const proximityDist = typeof p.proximity_distance_m === "number" ? `${p.proximity_distance_m.toFixed(1)} m` : "N/A";
  const forecastTimestep = p.timestep || activeTimestep;
  const exposureBasis = p.flood_exposure_basis === "high_critical_waterlogging" ? "High flood proximity"
    : p.flood_exposure_basis === "critical_waterlogging_proximity" ? "Critical flood proximity"
    : p.flood_exposure_basis || "Waterlogging proximity";

  const content = `
    <div class="infra-popup">
      <div class="infra-popup-title">${name}</div>
      <div class="infra-popup-row"><span>Category:</span> <strong>${catDisplay}</strong></div>
      <div class="infra-popup-row"><span>Forecast Status:</span> <span class="sev-tag high">${forecastStatus}</span></div>
      <div class="infra-popup-row"><span>Nearby Flood Depth:</span> <strong>${nearestDepth}</strong></div>
      <div class="infra-popup-row"><span>Distance to Flooded Street:</span> <strong>${proximityDist}</strong></div>
      <div class="infra-popup-row"><span>Forecast Timestep:</span> <strong>${forecastTimestep}</strong></div>
      <div class="infra-popup-row"><span>Exposure Basis:</span> <strong>${exposureBasis}</strong></div>
      <div class="infra-popup-action-row" style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.1);">
        <button class="${isSmartRouterActive ? 'btn-open-smart-router font-mono active-on' : 'btn-open-smart-router font-mono'}" style="width:100%; padding:5px 8px; background:${isSmartRouterActive ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #0284c7, #06b6d4)'}; border:none; border-radius:4px; color:#fff; font-size:0.72rem; font-weight:700; cursor:pointer;">${isSmartRouterActive ? '⚡ Smart Router ON' : '⚡ Launch Smart Router'}</button>
      </div>
      <div class="infra-popup-footer" style="margin-top:4px;">Model-derived flood exposure proxy</div>
    </div>
  `;
  layer.bindPopup(content, { className: "dark-leaflet-popup" });
}

async function loadInfraLayer() {
  const iCheck = document.getElementById("layer-infra-check");
  if (iCheck && !iCheck.checked) return;

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");

  const r1hEl = document.getElementById("rainfall_1h");
  const r3hEl = document.getElementById("rainfall_3h");
  const r6hEl = document.getElementById("rainfall_6h");
  const intEl = document.getElementById("recent_rainfall_intensity");

  const r1h = r1hEl ? parseFloat(r1hEl.value) || 10.0 : 10.0;
  const r3h = r3hEl ? parseFloat(r3hEl.value) || 20.0 : 20.0;
  const r6h = r6hEl ? parseFloat(r6hEl.value) || 30.0 : 30.0;
  const intVal = intEl ? parseFloat(intEl.value) || 5.0 : 5.0;

  const params = new URLSearchParams({
    scenario: activeScenario,
    timestep: activeTimestep,
    rainfall_1h: r1h,
    rainfall_3h: r3h,
    rainfall_6h: r6h,
    recent_rainfall_intensity: intVal,
    bbox: bboxStr
  });

  try {
    const res = await fetch(`${API_BASE_URL}/infrastructure?${params.toString()}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (infraLayer) {
      infraLayer.clearLayers();
      if (typeof infraLayer.addLayer === "function" && typeof L.markerClusterGroup === "function" && infraLayer instanceof L.MarkerClusterGroup) {
        (geojson.features || []).forEach((feat) => {
          const props = feat.properties || {};
          // Hard Critical-Asset Rule: Discard non-critical, category OTHER, or missing flood proximity evidence
          if (!props.critical || props.critical_asset === false || props.category === "other" || props.category === "OTHER") {
            return;
          }
          if (typeof props.nearest_water_depth_cm !== "number" || typeof props.proximity_distance_m !== "number") {
            return;
          }

          if (feat.geometry && feat.geometry.coordinates) {
            const coords = feat.geometry.coordinates;
            const latlng = [coords[1], coords[0]];
            const marker = createInfraMarker(feat, latlng);
            bindInfraPopup(feat, marker);
            infraLayer.addLayer(marker);
          }
        });
      } else if (typeof infraLayer.addData === "function") {
        infraLayer.addData(geojson);
      }
      lastLoadedInfraGeojson = geojson;
      updateScadaTelemetry();
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
    radius: 3,
    color: "#06b6d4",
    fillColor: "#06b6d4",
    fillOpacity: 0.45,
    weight: 1
  });
}

function bindDrainPopupContent(p) {
  const drainName = p.drain_name || "MPD-1976 Drain";
  const basin = p.basin || "Delhi Basin";
  const seqNo = p.seq_no || "N/A";
  const status = p.status || "Existing / Remodeling";
  const source = p.source || "MPD-1976";

  return `
    <div class="drain-popup">
      <div class="drain-popup-title font-mono" style="color:#00f3ff; font-weight:700; font-size:0.85rem; margin-bottom:6px;">${drainName}</div>
      <div class="drain-popup-row"><span>Basin:</span> <strong>${basin}</strong></div>
      <div class="drain-popup-row"><span>Sequence No:</span> <strong>${seqNo}</strong></div>
      <div class="drain-popup-row"><span>Status:</span> <strong>${status}</strong></div>
      <div class="drain-popup-row"><span>Source:</span> <strong>${source}</strong></div>
      <div class="drain-popup-footer" style="margin-top:6px; font-size:0.65rem; color:#94a3b8; border-top:1px solid rgba(255,255,255,0.1); padding-top:4px;">
        MPD-1976 Stormwater Drainage System
      </div>
    </div>
  `;
}

async function loadDrainageNetworkLayer() {
  const dCheck = document.getElementById("layer-drains-check");
  const legendDrainage = document.getElementById("legend-drainage-section");

  if (dCheck && !dCheck.checked) {
    if (drainageNetworkLayer) drainageNetworkLayer.clearLayers();
    if (legendDrainage) legendDrainage.classList.add("hidden");
    return;
  }

  if (legendDrainage) legendDrainage.classList.remove("hidden");

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");

  try {
    const res = await fetch(`${API_BASE_URL}/drainage?bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (!drainageNetworkLayer) return;
    drainageNetworkLayer.clearLayers();

    const features = geojson.features || [];

    // Group features by drain_name to draw electric cyan polyline channels
    const drainGroups = {};
    features.forEach((feat) => {
      const p = feat.properties || {};
      const name = p.drain_name || "MPD-1976 Drain";
      if (!drainGroups[name]) drainGroups[name] = [];
      if (feat.geometry && feat.geometry.coordinates) {
        drainGroups[name].push({
          seq: p.seq_no || 0,
          coords: feat.geometry.coordinates,
          feature: feat
        });
      }
    });

    let renderedFeatureCount = 0;

    // Draw electric cyan polyline channels for grouped drain points
    Object.keys(drainGroups).forEach((name) => {
      const pts = drainGroups[name];
      renderedFeatureCount += pts.length;

      pts.sort((a, b) => a.seq - b.seq);

      if (pts.length >= 2) {
        // Convert GeoJSON [lon, lat] pairs to Leaflet [lat, lon]
        const latlngs = pts.map((pt) => [pt.coords[1], pt.coords[0]]);
        const sampleFeat = pts[0].feature;
        const popupContent = bindDrainPopupContent(sampleFeat.properties || {});

        // Outer glow line (electric cyan / blue halo)
        const outerGlowLine = L.polyline(latlngs, {
          color: "#00f3ff",
          weight: 7,
          opacity: 0.75,
          lineCap: "round",
          lineJoin: "round"
        });
        outerGlowLine.bindPopup(popupContent, { className: "dark-leaflet-popup" });

        // Inner core line (bright cyan)
        const innerCoreLine = L.polyline(latlngs, {
          color: "#00f3ff",
          weight: 4,
          opacity: 1.0,
          lineCap: "round",
          lineJoin: "round"
        });
        innerCoreLine.bindPopup(popupContent, { className: "dark-leaflet-popup" });

        drainageNetworkLayer.addLayer(outerGlowLine);
        drainageNetworkLayer.addLayer(innerCoreLine);
      }

      // Draw outfall / node circle markers
      pts.forEach((pt) => {
        const latlng = [pt.coords[1], pt.coords[0]];
        const marker = L.circleMarker(latlng, {
          radius: 4,
          color: "#00f3ff",
          fillColor: "#06b6d4",
          fillOpacity: 0.9,
          weight: 2
        });
        marker.bindPopup(bindDrainPopupContent(pt.feature.properties || {}), { className: "dark-leaflet-popup" });
        drainageNetworkLayer.addLayer(marker);
      });
    });

    window.AQUAG_LAST_DRAINAGE_COUNT = renderedFeatureCount;

  } catch (err) {
    console.warn("Drainage network layer fetch non-blocking warning:", err.message);
  }
}

// --------------------------------------------------------------------------
// 4. Pump Stations Metadata Registry (GET /pumps)
// --------------------------------------------------------------------------
async function handlePumpsToggle(enabled) {
  const modal = document.getElementById("pump-modal");
  if (!modal) return;

  if (!enabled) {
    modal.classList.add("hidden");
    return;
  }

  modal.classList.remove("hidden");

  const listContainer = document.getElementById("pump-stations-list");
  const countVal = document.getElementById("pump-count-val");

  try {
    const res = await fetch(`${API_BASE_URL}/pumps`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const count = data.total_stations || 10;
    if (countVal) countVal.textContent = `${count} Registered Stations`;

    const stations = data.stations || [];
    if (listContainer) {
      if (stations.length === 0) {
        listContainer.innerHTML = '<div class="pump-list-loading">No pump station entries returned.</div>';
      } else {
        listContainer.innerHTML = stations.map((st, i) => `
          <div class="pump-item-row">
            <span class="pump-num">#${st.station_id || (i + 1)}</span>
            <span class="pump-name">${st.name || st.station_name || st.raw_name || 'Station ' + (i + 1)}</span>
            <span class="pump-cap">${st.status || (st.capacity_cusec ? st.capacity_cusec + ' cusec' : 'Delhi')}</span>
          </div>
        `).join("");
      }
    }
  } catch (err) {
    if (listContainer) {
      listContainer.innerHTML = `<div class="pump-list-loading">Pump metadata offline: ${err.message}</div>`;
    }
  }
}

// --------------------------------------------------------------------------
// 5. Population Priority Layer Engine (GET /population-priority)
// --------------------------------------------------------------------------
let topPriorityFeatureGlobal = null;

function createPriorityPopupContent(p) {
  const roadId = p.road_id || "Priority Corridor";
  const rawLevel = (p.priority_level || "MEDIUM").toUpperCase();
  const pCode = rawLevel === "CRITICAL" ? "P1" : rawLevel === "HIGH" ? "P2" : "P3";
  const tagClass = rawLevel === "CRITICAL" ? "critical" : rawLevel === "HIGH" ? "high" : "medium";

  const depthNum = typeof p.water_depth_cm === "number" ? p.water_depth_cm : parseFloat(p.water_depth_cm || 0);
  const depthVal = depthNum.toFixed(1);
  let floodSev = depthNum > 100 ? "CRITICAL (>100 cm)" : depthNum > 25 ? "HIGH (>25-100 cm)" : "MEDIUM (>10-25 cm)";

  const popExp = typeof p.population_exposure === "number" ? p.population_exposure.toLocaleString() : (p.population_exposure || "N/A");
  const critInfra = p.critical_infra_flag === 1 ? "1 Critical Asset" : "0 Assets";
  const timestepVal = p.timestep || activeTimestep;

  let recAction = "Deploy high-capacity dewatering pumps & clear key transit arteries";
  if (rawLevel === "CRITICAL") {
    recAction = "Immediate emergency response: deploy heavy pumps, reroute traffic & protect critical assets";
  } else if (rawLevel === "HIGH") {
    recAction = "High response priority: dispatch mobile pump units & issue localized traffic warnings";
  } else {
    recAction = "Monitor corridor: maintain SCADA telemetry & prepare drainage clearing";
  }

  return `
    <div class="pop-priority-popup">
      <div class="wl-popup-title font-mono" style="font-size:0.85rem; font-weight:700; color:#f8fafc; margin-bottom:6px;">${roadId}</div>
      <div class="wl-popup-row"><span>RESPONSE PRIORITY:</span> <span class="sev-tag ${tagClass}" style="font-weight:700;">${pCode} · ${rawLevel}</span></div>
      <div class="wl-popup-row"><span>Flood severity:</span> <strong>${floodSev}</strong></div>
      <div class="wl-popup-row"><span>Water depth:</span> <strong>${depthVal} cm</strong></div>
      <div class="wl-popup-row"><span>Population exposed:</span> <strong>${popExp}</strong></div>
      <div class="wl-popup-row"><span>Critical assets:</span> <strong>${critInfra}</strong></div>
      <div class="wl-popup-row"><span>Recommended action:</span> <strong style="color:#c084fc;">${recAction}</strong></div>
      <div class="wl-popup-row"><span>Forecast Timestep:</span> <strong>${timestepVal}</strong></div>
      <div class="wl-popup-action-row" style="margin-top:8px; padding-top:6px; border-top:1px solid rgba(255,255,255,0.1);">
        <button class="${isSmartRouterActive ? 'btn-open-smart-router font-mono active-on' : 'btn-open-smart-router font-mono'}" style="width:100%; padding:5px 8px; background:${isSmartRouterActive ? 'linear-gradient(135deg, #10b981, #059669)' : 'linear-gradient(135deg, #0284c7, #06b6d4)'}; border:none; border-radius:4px; color:#fff; font-size:0.72rem; font-weight:700; cursor:pointer;">${isSmartRouterActive ? '⚡ Smart Router ON' : '⚡ Launch Smart Router'}</button>
      </div>
      <div class="wl-popup-footer" style="margin-top:4px; font-size:0.65rem; color:#94a3b8;">
        Population used as response exposure factor (not simple density)
      </div>
    </div>
  `;
}

function initTopPriorityFocusHandler() {
  const focusBtn = document.getElementById("btn-focus-top-p");
  if (focusBtn) {
    focusBtn.addEventListener("click", () => {
      if (!topPriorityFeatureGlobal || !map) return;
      const tempLayer = L.geoJSON(topPriorityFeatureGlobal);
      const bounds = tempLayer.getBounds();
      map.fitBounds(bounds, { maxZoom: 16, padding: [60, 60] });

      const popupHtml = createPriorityPopupContent(topPriorityFeatureGlobal.properties || {});
      const center = bounds.getCenter();
      L.popup({ className: "dark-leaflet-popup" })
        .setLatLng(center)
        .setContent(popupHtml)
        .openOn(map);
    });
  }
}

async function loadPopulationPriorityLayer() {
  const popCheck = document.getElementById("layer-pop-priority-check");
  if (popCheck && !popCheck.checked) {
    if (populationPriorityLayer) populationPriorityLayer.clearLayers();
    return;
  }

  const bbox = getMapViewportBbox();
  const bboxStr = bbox.join(",");
  const encodedTimestep = encodeURIComponent(activeTimestep);

  try {
    const res = await fetch(`${API_BASE_URL}/population-priority?scenario=${activeScenario}&timestep=${encodedTimestep}&bbox=${bboxStr}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (!populationPriorityLayer) return;
    populationPriorityLayer.clearLayers();

    const rawFeatures = geojson.features || [];

    // ONLY SHOW OPERATIONAL PRIORITY: Filter out unflooded roads (water_depth_cm <= 10.0) or priority LOW
    const priorityFeatures = rawFeatures.filter((f) => {
      const p = f.properties || {};
      const depth = typeof p.water_depth_cm === "number" ? p.water_depth_cm : parseFloat(p.water_depth_cm || 0);
      const level = (p.priority_level || "LOW").toUpperCase();
      return depth > 10.0 && level !== "LOW";
    });

    let cntP1 = 0;
    let cntP2 = 0;
    let cntP3 = 0;

    priorityFeatures.forEach((f) => {
      const p = f.properties || {};
      const level = (p.priority_level || "MEDIUM").toUpperCase();
      if (level === "CRITICAL") cntP1++;
      else if (level === "HIGH") cntP2++;
      else if (level === "MEDIUM") cntP3++;
    });

    // Update SCADA Panel response priority counts
    const elP1 = document.getElementById("cnt-p1");
    const elP2 = document.getElementById("cnt-p2");
    const elP3 = document.getElementById("cnt-p3");
    if (elP1) elP1.textContent = cntP1;
    if (elP2) elP2.textContent = cntP2;
    if (elP3) elP3.textContent = cntP3;

    // Sort priority features by priority_score descending
    priorityFeatures.sort((a, b) => {
      const scoreA = (a.properties && a.properties.priority_score) || 0;
      const scoreB = (b.properties && b.properties.priority_score) || 0;
      return scoreB - scoreA;
    });

    // Handle TOP PRIORITY CARD update
    const topCardEl = document.getElementById("top-priority-card");
    if (priorityFeatures.length > 0 && topCardEl) {
      topCardEl.classList.remove("hidden");
      const topF = priorityFeatures[0];
      topPriorityFeatureGlobal = topF;

      const topP = topF.properties || {};
      const rawLevel = (topP.priority_level || "MEDIUM").toUpperCase();
      const pCode = rawLevel === "CRITICAL" ? "P1" : rawLevel === "HIGH" ? "P2" : "P3";
      const tagClass = rawLevel === "CRITICAL" ? "critical" : rawLevel === "HIGH" ? "high" : "medium";

      const badgeEl = document.getElementById("top-p-badge");
      const titleEl = document.getElementById("top-p-title");
      const popEl = document.getElementById("top-p-pop");
      const depthEl = document.getElementById("top-p-depth");
      const infraEl = document.getElementById("top-p-infra");

      if (badgeEl) {
        badgeEl.textContent = pCode;
        badgeEl.className = `sev-tag ${tagClass}`;
      }
      if (titleEl) titleEl.textContent = topP.road_id || "Top Priority Segment";
      if (popEl) popEl.textContent = typeof topP.population_exposure === "number" ? topP.population_exposure.toLocaleString() : (topP.population_exposure || "--");
      if (depthEl) depthEl.textContent = typeof topP.water_depth_cm === "number" ? `${topP.water_depth_cm.toFixed(1)} cm` : "--";
      if (infraEl) infraEl.textContent = topP.critical_infra_flag === 1 ? "1 Critical Asset" : "0 Assets";
    } else if (topCardEl) {
      topCardEl.classList.add("hidden");
      topPriorityFeatureGlobal = null;
    }

    // Render Dual Polylines for each priority feature
    priorityFeatures.forEach((feature) => {
      const p = feature.properties || {};
      const level = (p.priority_level || "MEDIUM").toUpperCase();
      const depth = typeof p.water_depth_cm === "number" ? p.water_depth_cm : parseFloat(p.water_depth_cm || 0);

      // Outer glow style (purple/magenta priority highlight)
      let outerColor = "#818cf8"; // P3 subtle indigo-purple
      let outerWeight = 5.5;
      let outerOpacity = 0.65;

      if (level === "CRITICAL") {
        outerColor = "#a855f7"; // P1 prominent purple/magenta outer glow
        outerWeight = 9.5;
        outerOpacity = 0.85;
      } else if (level === "HIGH") {
        outerColor = "#c084fc"; // P2 strong purple glow
        outerWeight = 7.5;
        outerOpacity = 0.75;
      }

      // Inner water depth line color (must NOT replace underlying water depth colors)
      let innerColor = "#eab308"; // >10-25 cm yellow
      let innerWeight = 3.0;
      if (depth > 100.0) {
        innerColor = "#ef4444"; // >100 cm red
        innerWeight = 4.0;
      } else if (depth > 25.0) {
        innerColor = "#f97316"; // >25-100 cm orange
        innerWeight = 3.5;
      }

      const popupHtml = createPriorityPopupContent(p);

      // Create outer halo polyline
      const outerLayer = L.geoJSON(feature, {
        style: {
          color: outerColor,
          weight: outerWeight,
          opacity: outerOpacity,
          lineCap: "round",
          lineJoin: "round"
        }
      });
      outerLayer.bindPopup(popupHtml, { className: "dark-leaflet-popup" });

      // Create inner core polyline
      const innerLayer = L.geoJSON(feature, {
        style: {
          color: innerColor,
          weight: innerWeight,
          opacity: 0.95,
          lineCap: "round",
          lineJoin: "round"
        }
      });
      innerLayer.bindPopup(popupHtml, { className: "dark-leaflet-popup" });

      populationPriorityLayer.addLayer(outerLayer);
      populationPriorityLayer.addLayer(innerLayer);
    });

    // Add compact priority badges ONLY for top 15 highest-priority locations
    const topNBadges = priorityFeatures.slice(0, 15);
    topNBadges.forEach((feature) => {
      const p = feature.properties || {};
      const level = (p.priority_level || "MEDIUM").toUpperCase();
      const pCode = level === "CRITICAL" ? "P1" : level === "HIGH" ? "P2" : "P3";
      const pClass = level === "CRITICAL" ? "p1" : level === "HIGH" ? "p2" : "p3";

      const tempGeo = L.geoJSON(feature);
      const bounds = tempGeo.getBounds();
      const center = bounds.getCenter();

      const customIcon = L.divIcon({
        className: "priority-badge-div",
        html: `<div class="priority-badge-marker ${pClass}"><span>⚡</span> ${pCode}</div>`,
        iconSize: [36, 18],
        iconAnchor: [18, 9]
      });

      const badgeMarker = L.marker(center, { icon: customIcon });
      const popupHtml = createPriorityPopupContent(p);
      badgeMarker.bindPopup(popupHtml, { className: "dark-leaflet-popup" });

      populationPriorityLayer.addLayer(badgeMarker);
    });

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
    lastLoadedAlertsData = data;
    updateScadaTelemetry();

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
// --------------------------------------------------------------------------
// Smart Router State Controller (Single Source of Truth)
// --------------------------------------------------------------------------
function setSmartRouterState(enabled, options = {}) {
  isSmartRouterActive = !!enabled;

  // 1. Synchronize the Active Layers checkbox (#layer-route-check)
  const rCheck = document.getElementById("layer-route-check");
  if (rCheck && rCheck.checked !== isSmartRouterActive) {
    rCheck.checked = isSmartRouterActive;
  }

  // 2. Synchronize Leaflet map layer (routeLayer)
  if (map && routeLayer) {
    if (isSmartRouterActive) {
      if (!map.hasLayer(routeLayer)) map.addLayer(routeLayer);
    } else {
      if (map.hasLayer(routeLayer)) map.removeLayer(routeLayer);
    }
  }

  // 3. Synchronize right sidebar tab (#tab-routing) & automatically expand Smart Router Finder accordion
  if (options.openTab !== false && isSmartRouterActive) {
    const sideTabBtns = document.querySelectorAll(".side-tab-btn");
    const routerTabBtn = document.querySelector('.side-tab-btn[data-tab="tab-routing"]');
    if (routerTabBtn) {
      sideTabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".sidebar-tab-content").forEach((c) => c.classList.remove("active"));
      routerTabBtn.classList.add("active");
      const targetEl = document.getElementById("tab-routing");
      if (targetEl) {
        targetEl.classList.add("active");

        // Action 2: Automatically open/expand the Smart Router Finder accordion (<details>)
        const accordion = targetEl.querySelector("details, #route-finder-accordion");
        if (accordion) {
          accordion.open = true;
        }

        // Action 3: Focus origin input so finder is immediately ready for user input
        setTimeout(() => {
          const startLatInput = document.getElementById("start_lat");
          const routeForm = document.getElementById("route-form");
          if (startLatInput) {
            startLatInput.focus();
            if (typeof startLatInput.select === "function") {
              startLatInput.select();
            }
          } else if (routeForm) {
            routeForm.scrollIntoView({ behavior: "smooth", block: "nearest" });
          }
        }, 50);
      }
    }
  }

  // 4. Synchronize all popup buttons (.btn-open-smart-router) across DOM
  const routerBtns = document.querySelectorAll(".btn-open-smart-router, [data-action='open-smart-router'], .popup-smart-router-btn, [data-action='route']");
  routerBtns.forEach((btn) => {
    if (isSmartRouterActive) {
      btn.classList.add("active-on");
      btn.textContent = "⚡ Smart Router ON";
      btn.style.background = "linear-gradient(135deg, #10b981, #059669)";
    } else {
      btn.classList.remove("active-on");
      btn.textContent = "⚡ Launch Smart Router";
      btn.style.background = "linear-gradient(135deg, #0284c7, #06b6d4)";
    }
  });
}

function toggleSmartRouterState(options = {}) {
  setSmartRouterState(!isSmartRouterActive, options);
}

function initSmartRouterState() {
  const rCheck = document.getElementById("layer-route-check");
  const initialState = rCheck ? rCheck.checked : false;
  setSmartRouterState(initialState, { openTab: false });
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

  // Global delegated click handler connecting initial/popup Smart Router buttons to Smart Router state controller
  document.addEventListener("click", (e) => {
    const routerBtn = e.target.closest(".btn-open-smart-router, [data-action='open-smart-router'], .popup-smart-router-btn, [data-action='route']");
    if (routerBtn) {
      e.preventDefault();
      toggleSmartRouterState({ openTab: true });
    }
  });
}

function initTopNavModuleButtons() {
  const btnGis = document.getElementById("nav-btn-gis");
  const btnScada = document.getElementById("nav-btn-scada");
  const btnAlerts = document.getElementById("nav-btn-alerts");
  const btnAnalytics = document.getElementById("nav-btn-analytics");

  const navBtns = document.querySelectorAll(".nav-module-btn");

  const setActiveNav = (activeBtn) => {
    navBtns.forEach((b) => b.classList.remove("active"));
    if (activeBtn) activeBtn.classList.add("active");
  };

  if (btnGis) {
    btnGis.addEventListener("click", () => {
      setActiveNav(btnGis);
      const tabBtn = document.querySelector('.side-tab-btn[data-tab="tab-situation"]');
      if (tabBtn) tabBtn.click();
    });
  }

  if (btnScada) {
    btnScada.addEventListener("click", () => {
      setActiveNav(btnScada);
      const dCheck = document.getElementById("layer-drains-check");
      if (dCheck && !dCheck.checked) {
        dCheck.checked = true;
        loadDrainageNetworkLayer();
      }
      const pModal = document.getElementById("pump-modal");
      if (pModal) {
        pModal.classList.remove("hidden");
        loadPumpStationsMetadata();
      }
    });
  }

  if (btnAlerts) {
    btnAlerts.addEventListener("click", () => {
      setActiveNav(btnAlerts);
      const tabBtn = document.querySelector('.side-tab-btn[data-tab="tab-alerts"]');
      if (tabBtn) tabBtn.click();
    });
  }

  if (btnAnalytics) {
    btnAnalytics.addEventListener("click", () => {
      setActiveNav(btnAnalytics);
      const tabBtn = document.querySelector('.side-tab-btn[data-tab="tab-situation"]');
      if (tabBtn) tabBtn.click();
      const scadaGrid = document.querySelector(".scada-metrics-grid");
      if (scadaGrid) {
        scadaGrid.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
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

  const closePumpsBtn = document.getElementById("btn-close-pumps-modal");
  if (closePumpsBtn) {
    closePumpsBtn.addEventListener("click", () => {
      handlePumpsToggle(false);
      if (pCheck) pCheck.checked = false;
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
      if (e.target.checked) {
        if (!map.hasLayer(zonesLayer)) map.addLayer(zonesLayer);
        loadZoneLayer();
      } else {
        if (map.hasLayer(zonesLayer)) map.removeLayer(zonesLayer);
      }
    });
  }

  const rCheck = document.getElementById("layer-route-check");
  if (rCheck) {
    rCheck.addEventListener("change", (e) => {
      setSmartRouterState(e.target.checked, { openTab: e.target.checked });
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

let isPlaying = false;
let playIntervalTimer = null;
let currentTimestepRequestId = 0;

function initTimelineBar() {
  const stepBtns = document.querySelectorAll(".time-step-btn");
  const playBtn = document.getElementById("btn-play-pause");

  stepBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (isPlaying) {
        stopPlayAnimation();
      }
      selectTimestepButton(btn);
    });
  });

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (isPlaying) {
        stopPlayAnimation();
      } else {
        startPlayAnimation();
      }
    });
  }
}

function selectTimestepButton(btn) {
  const stepBtns = document.querySelectorAll(".time-step-btn");
  stepBtns.forEach((b) => {
    b.classList.remove("active");
    const dot = b.querySelector(".step-dot");
    if (dot) dot.remove();
  });

  btn.classList.add("active");
  if (!btn.querySelector(".step-dot")) {
    const dot = document.createElement("span");
    dot.className = "step-dot";
    btn.insertBefore(dot, btn.firstChild);
  }

  const backendStep = btn.getAttribute("data-backend-step") || "T+0";
  activeTimestep = backendStep;
  updateScadaTelemetry();

  triggerTimestepUpdate();
}

function startPlayAnimation() {
  isPlaying = true;
  const playIcon = document.getElementById("play-icon");
  const pauseIcon = document.getElementById("pause-icon");
  const playBtnText = document.getElementById("play-btn-text");
  const playBtn = document.getElementById("btn-play-pause");

  if (playIcon) playIcon.classList.add("hidden");
  if (pauseIcon) pauseIcon.classList.remove("hidden");
  if (playBtnText) playBtnText.textContent = "PAUSE";
  if (playBtn) playBtn.classList.add("playing");

  const stepBtns = Array.from(document.querySelectorAll(".time-step-btn"));

  if (playIntervalTimer) clearInterval(playIntervalTimer);
  playIntervalTimer = setInterval(() => {
    const currentIndex = stepBtns.findIndex((b) => b.classList.contains("active"));
    const nextIndex = (currentIndex + 1) % stepBtns.length;
    selectTimestepButton(stepBtns[nextIndex]);
  }, 2500);
}

function stopPlayAnimation() {
  isPlaying = false;
  if (playIntervalTimer) {
    clearInterval(playIntervalTimer);
    playIntervalTimer = null;
  }

  const playIcon = document.getElementById("play-icon");
  const pauseIcon = document.getElementById("pause-icon");
  const playBtnText = document.getElementById("play-btn-text");
  const playBtn = document.getElementById("btn-play-pause");

  if (playIcon) playIcon.classList.remove("hidden");
  if (pauseIcon) pauseIcon.classList.add("hidden");
  if (playBtnText) playBtnText.textContent = "PLAY";
  if (playBtn) playBtn.classList.remove("playing");
}

async function triggerTimestepUpdate() {
  const reqId = ++currentTimestepRequestId;
  const spinner = document.getElementById("timeline-spinner");
  if (spinner) spinner.classList.remove("hidden");

  try {
    const promises = [];
    const wCheck = document.getElementById("layer-waterlogging-check");
    const iCheck = document.getElementById("layer-infra-check");
    const popCheck = document.getElementById("layer-pop-priority-check");

    if (wCheck && wCheck.checked) promises.push(loadWaterloggingLayer());
    if (iCheck && iCheck.checked) promises.push(loadInfraLayer());
    if (popCheck && popCheck.checked) promises.push(loadPopulationPriorityLayer());
    promises.push(loadAlertsPanel());

    await Promise.all(promises);
  } catch (err) {
    console.warn("Timestep update non-blocking notice:", err);
  } finally {
    if (reqId === currentTimestepRequestId) {
      if (spinner) spinner.classList.add("hidden");
    }
  }
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
async function checkSystemHealth(retryOnFail = true) {
  const healthBadge = document.getElementById("status-health-badge");
  const healthText = document.getElementById("health-status-text");

  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("Health check HTTP error");
    const data = await res.json();

    if (data && data.status === "ok") {
      if (healthBadge) healthBadge.className = "telemetry-pill online";
      if (healthText) healthText.textContent = "System Online";

      const tModel = document.getElementById("t-model");
      const tRouter = document.getElementById("t-router");
      if (tModel) tModel.textContent = data.model_version || "AquaG Model V2";
      if (tRouter) tRouter.textContent = data.router_loaded ? "AquaGraph A*" : "Loaded";
    } else {
      throw new Error("System status not ok");
    }
  } catch (err) {
    console.warn("Backend health status check notice:", err.message);
    if (healthBadge) healthBadge.className = "telemetry-pill offline";
    if (healthText) healthText.textContent = "Backend Offline";

    if (retryOnFail) {
      setTimeout(() => checkSystemHealth(false), 2000);
    }
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
const DELHI_ZONE_CENTROIDS = [
  { name: "Rohini (North West)", lat: 28.715, lon: 77.115 },
  { name: "Karol Bagh (Central)", lat: 28.652, lon: 77.190 },
  { name: "Connaught Place (Central)", lat: 28.632, lon: 77.219 },
  { name: "Dwarka (South West)", lat: 28.582, lon: 77.060 },
  { name: "Saket (South)", lat: 28.524, lon: 77.210 },
  { name: "Okhla (South East)", lat: 28.560, lon: 77.285 },
  { name: "Shahdara (East)", lat: 28.673, lon: 77.285 },
  { name: "Narela (North)", lat: 28.850, lon: 77.095 }
];

async function loadZoneLayer() {
  const zCheck = document.getElementById("layer-zones-check");
  if (zCheck && !zCheck.checked) return;

  try {
    const res = await fetch(`${API_BASE_URL}/zones`);
    if (!res.ok) return;
    const zones = await res.json();

    if (zonesLayer) zonesLayer.clearLayers();

    zones.forEach((zone, idx) => {
      const color =
        zone.severity === "High" ? "#ef4444" :
        zone.severity === "Medium" ? "#f59e0b" : "#10b981";

      let lat = zone.latitude;
      let lon = zone.longitude;
      let zoneName = `Zone ${zone.zone_id}`;

      if (!lat || !lon || (lat === 0.0 && lon === 0.0)) {
        const centroid = DELHI_ZONE_CENTROIDS[idx % DELHI_ZONE_CENTROIDS.length];
        lat = centroid.lat;
        lon = centroid.lon;
        zoneName = `${zone.zone_id} (${centroid.name})`;
      }

      L.circle([lat, lon], {
        radius: 1200,
        color: color,
        fillColor: color,
        fillOpacity: 0.28,
        weight: 2,
        dashArray: "6, 6"
      })
        .bindPopup(`
          <div class="waterlogging-popup">
            <div class="wl-popup-title">Flood Risk Zone ${zoneName}</div>
            <div class="wl-popup-row"><span>Severity:</span> <span class="sev-tag ${zone.severity.toLowerCase()}">${zone.severity}</span></div>
            <div class="wl-popup-row"><span>Model Version:</span> <strong>AquaG Model V2</strong></div>
            <div class="wl-popup-footer">Model-derived zone severity (GET /zones)</div>
          </div>
        `, { className: "dark-leaflet-popup" })
        .addTo(zonesLayer);
    });
  } catch (err) {
    console.log("Zone layer skipped:", err.message);
  }
}
