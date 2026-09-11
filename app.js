/**
 * AquaG — Operational Urban Flood GIS Command Center
 * Stage 10 Frontend Controller (Vanilla JavaScript + Leaflet.js)
 * 
 * Centralized API_BASE_URL connecting strictly to Stage 7 FastAPI backend.
 * Integrates Stage 10 Rainfall Scenario Simulator & Model V2 Inference.
 */

// Configurable API_BASE_URL: supports window.AQUAG_API_URL for production deployment, defaulting to local http://127.0.0.1:8000
const API_BASE_URL = (typeof window !== "undefined" && window.AQUAG_API_URL)
  ? window.AQUAG_API_URL
  : "https://aquag.onrender.com";

// Global State Variables
let map = null;
let routeLayer = null;
let zonesLayer = null;
let drainsLayer = null;
let markersLayer = null;

let pickingMode = null; // 'start' or 'end'
let activePresetName = "MODERATE";
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

  // Overlay Layer Groups
  zonesLayer = L.layerGroup().addTo(map);
  drainsLayer = L.layerGroup().addTo(map);
  routeLayer = L.layerGroup().addTo(map);
  markersLayer = L.layerGroup().addTo(map);

  // Map Click Event (Point Inspector & Coord Picker)
  map.on("click", handleMapClick);
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
    });
  });
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
      activePresetName = type.toUpperCase();

      Object.keys(data).forEach((key) => {
        const el = document.getElementById(key);
        if (el) el.value = data[key];
      });

      // Update Summary Card Preview
      updateScenarioSummaryCard(data.rainfall_1h, data.rainfall_3h, data.rainfall_6h);
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
  const zCheck = document.getElementById("layer-zones-check");
  if (zCheck) {
    zCheck.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(zonesLayer);
      else map.removeLayer(zonesLayer);
    });
  }

  const dCheck = document.getElementById("layer-drains-check");
  if (dCheck) {
    dCheck.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(drainsLayer);
      else map.removeLayer(drainsLayer);
    });
  }

  const rCheck = document.getElementById("layer-route-check");
  if (rCheck) {
    rCheck.addEventListener("change", (e) => {
      if (e.target.checked) map.addLayer(routeLayer);
      else map.removeLayer(routeLayer);
    });
  }
}

function initTimelineBar() {
  const stepBtns = document.querySelectorAll(".time-step-btn");
  stepBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const step = btn.getAttribute("data-step");
      if (step === "now") {
        stepBtns.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
      } else {
        alert(`Scenario UI Prototype (${step}): User-defined scenario timeline prototype.`);
      }
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
// 1. GET /health — Backend System Health Telemetry
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
// 2. POST /flood_info — Point GIS Inspector
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
// 3. POST /predict — Model V2 & Action Priority Engine
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
  // 4. POST /route — AquaGraph A* Risk Router
  // --------------------------------------------------------------------------
  if (routeForm) {
    routeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const btn = document.getElementById("btn-route");
      const spinner = btn ? btn.querySelector(".btn-spinner") : null;
      const resultsPanel = document.getElementById("route-results-panel");

      if (spinner) spinner.classList.remove("hidden");
      if (btn) btn.disabled = true;

      const payload = {
        start_lat: parseFloat(document.getElementById("start_lat").value),
        start_lon: parseFloat(document.getElementById("start_lon").value),
        end_lat: parseFloat(document.getElementById("end_lat").value),
        end_lon: parseFloat(document.getElementById("end_lon").value),
        risk: document.getElementById("route_risk").value,
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

        document.getElementById("route-dist").textContent = `${data.physical_distance_m || data.distance_m} m`;
        document.getElementById("route-cost").textContent = `${data.estimated_cost || data.routing_cost}`;
        document.getElementById("route-nodes").textContent = `${data.nodes_in_route}`;
        document.getElementById("route-sensitivity").textContent = `${data.risk_mode || data.risk_level}`;
        document.getElementById("route-snap-orig").textContent = `${data.origin_snap_distance_m} m`;
        document.getElementById("route-snap-dest").textContent = `${data.destination_snap_distance_m} m`;
        document.getElementById("route-risk-basis").textContent = data.risk_basis || "spatial_proxy";

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
// 5. GET /zones — Sample Zone Severity GIS Layer
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
