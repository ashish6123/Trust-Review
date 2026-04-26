/* ═══════════════════════════════════════════════════════════
   Trust Review — Client-side JavaScript
   ═══════════════════════════════════════════════════════════ */

const API = "";  // same origin

// ── State ───────────────────────────────────────────────
let selectedFile = null;
let lastDownloadId = null;
let bulkChartInstance = null;
let urlChartInstance = null;

// ── Tab switching ───────────────────────────────────────
document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`panel-${tab.dataset.tab}`).classList.add("active");
    });
});

// ── Dropzone ────────────────────────────────────────────
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx"].includes(ext)) {
        showToast("Please upload a CSV or XLSX file.");
        return;
    }
    selectedFile = file;
    document.getElementById("file-name").textContent = file.name;
    document.getElementById("file-info").classList.remove("hidden");
    dropzone.classList.add("hidden");
}

function clearFile() {
    selectedFile = null;
    fileInput.value = "";
    document.getElementById("file-info").classList.add("hidden");
    dropzone.classList.remove("hidden");
}

// ── Single review analysis ──────────────────────────────
async function analyzeText() {
    const text = document.getElementById("review-input").value.trim();
    if (!text) { showToast("Please enter a review."); return; }

    const modelType = document.getElementById("model-type-text").value;
    showLoading("Analyzing review…");

    try {
        const res = await fetch(`${API}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, model_type: modelType }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const data = await res.json();
        displaySingleResult(data.result);
    } catch (err) {
        showToast(err.message);
    } finally {
        hideLoading();
    }
}

function displaySingleResult(result) {
    const card = document.getElementById("result-text");
    card.classList.remove("hidden");

    const isReal = result.label === "Real";
    const badge = document.getElementById("result-badge");
    badge.textContent = result.label;
    badge.className = `badge ${isReal ? "real" : "fake"}`;

    const fill = document.getElementById("confidence-fill");
    const pct = Math.round(result.confidence * 100);
    fill.className = `confidence-fill ${isReal ? "real" : "fake"}`;
    // Trigger animation
    fill.style.width = "0%";
    requestAnimationFrame(() => {
        requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
    });

    document.getElementById("confidence-value").textContent = `${pct}%`;
    document.getElementById("result-meta").textContent = `Model: ${result.model_used}`;

    card.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ── Bulk upload ─────────────────────────────────────────
async function analyzeBulk() {
    if (!selectedFile) { showToast("Please upload a file first."); return; }

    const modelSelect = document.getElementById("model-type-bulk");
    const selectedOption = modelSelect.options[modelSelect.selectedIndex];
    const modelType = selectedOption.value;
    const modelName = selectedOption.getAttribute("data-model");
    const formData = new FormData();
    formData.append("file", selectedFile);

    showLoading("Processing file…");

    try {
        const res = await fetch(`${API}/api/predict/bulk?model_type=${modelType}`, {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const data = await res.json();
        lastDownloadId = data.download_id;
        displayBulkResult(data, "bulk");
    } catch (err) {
        showToast(err.message);
    } finally {
        hideLoading();
    }
}

// ── URL scraping ────────────────────────────────────────
async function analyzeURL() {
    const url = document.getElementById("url-input").value.trim();
    if (!url) { showToast("Please enter a URL."); return; }

    const modelType = document.getElementById("model-type-url").value;
    const modelName = document.getElementById("model-type-url").options[document.getElementById("model-type-url").selectedIndex].getAttribute("data-model");
    showLoading("Scraping & analyzing…");

    try {
        const res = await fetch(`${API}/api/predict/url`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({text: text, model_type: modelType, model_name: modelName})
        });
        if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
        const data = await res.json();
        document.getElementById("reviews-found").textContent = `${data.reviews_found} reviews found`;
        displayBulkResult(data, "url");
    } catch (err) {
        showToast(err.message);
    } finally {
        hideLoading();
    }
}

// ── Shared bulk result renderer ─────────────────────────
function displayBulkResult(data, type) {
    const card = document.getElementById(`result-${type}`);
    card.classList.remove("hidden");

    const statsEl = document.getElementById(`${type}-stats`);
    const s = data.summary;
    statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-value">${s.total}</div><div class="stat-label">Total</div></div>
        <div class="stat-card real"><div class="stat-value">${s.real_count}</div><div class="stat-label">Real</div></div>
        <div class="stat-card fake"><div class="stat-value">${s.fake_count}</div><div class="stat-label">Fake</div></div>
        <div class="stat-card"><div class="stat-value">${Math.round(s.avg_confidence * 100)}%</div><div class="stat-label">Avg Conf.</div></div>
    `;

    // Chart
    const chartId = `${type}-chart`;
    const ctx = document.getElementById(chartId).getContext("2d");
    const existingChart = type === "bulk" ? bulkChartInstance : urlChartInstance;
    if (existingChart) existingChart.destroy();

    const chart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Real", "Fake"],
            datasets: [{
                data: [s.real_count, s.fake_count],
                backgroundColor: ["#00d68f", "#ff6b6b"],
                borderColor: ["rgba(0,214,143,0.3)", "rgba(255,107,107,0.3)"],
                borderWidth: 2,
                hoverOffset: 6,
            }],
        },
        options: {
            responsive: true,
            cutout: "65%",
            plugins: {
                legend: { position: "bottom", labels: { color: "#8892b0", font: { family: "'Inter'" } } },
            },
        },
    });
    if (type === "bulk") bulkChartInstance = chart;
    else urlChartInstance = chart;

    // Table
    const tbody = document.getElementById(`${type}-tbody`);
    tbody.innerHTML = data.results.map((r, i) => `
        <tr>
            <td>${i + 1}</td>
            <td title="${escapeHtml(r.text)}">${escapeHtml(r.text.substring(0, 80))}${r.text.length > 80 ? "…" : ""}</td>
            <td class="label-${r.label.toLowerCase()}">${r.label}</td>
            <td>${Math.round(r.confidence * 100)}%</td>
        </tr>
    `).join("");

    card.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Download ────────────────────────────────────────────
function downloadLabelled() {
    if (!lastDownloadId) { showToast("No file to download."); return; }
    window.open(`${API}/api/download/${lastDownloadId}`, "_blank");
}

// ── Helpers ─────────────────────────────────────────────
function showLoading(msg) {
    document.getElementById("loading-text").textContent = msg;
    document.getElementById("loading").classList.remove("hidden");
}
function hideLoading() {
    document.getElementById("loading").classList.add("hidden");
}

function showToast(msg) {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.classList.remove("hidden");
    toast.classList.add("show");
    setTimeout(() => {
        toast.classList.remove("show");
        setTimeout(() => toast.classList.add("hidden"), 300);
    }, 3500);
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}
