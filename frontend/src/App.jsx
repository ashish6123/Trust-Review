import { useEffect, useMemo, useRef, useState } from "react";
import { HashRouter as Router, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Doughnut } from "react-chartjs-2";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
// import homeScreenOne from "./assets/home-screen-1.png";
import homeScreenTwo from "./assets/home-screen-2.png";
import homeScreenThree from "./assets/home-screen-3.png";

ChartJS.register(ArcElement, Tooltip, Legend);

const API = "";
const MODEL_CHOICES = [
  { key: "ml:svm", label: "SVM (Best)" },
  { key: "ml:logistic_regression", label: "Logistic Regression" },
  { key: "ml:random_forest", label: "Random Forest" },
  { key: "dl", label: "DistilBERT (Coming Soon)" },
];

const TEAM_MEMBERS = [
  {
    name: "Ashish Ranjan",
    role: "ML Lead & Project Architect",
    bio: "Built the complete ML pipeline, EDA, model training and evaluation, charts, workflow design, and overall project architecture.",
  },
  {
    name: "Gyan Ranjan",
    role: "Backend Developer",
    bio: "Flask API backend, prediction endpoint, model serving, and CSV handling.",
  },
  {
    name: "Disha Ranjan",
    role: "React Frontend",
    bio: "Frontend UI, input forms, result display, Chart.js visuals, and URL scraping flow.",
  },
  {
    name: "Miheer Ranjan",
    role: "Literature & Dataset Research",
    bio: "Literature review (3 papers), dataset comparison, and rationale for choosing OPUS.",
  },
  {
    name: "Amrit Ranjan",
    role: "System Design & Documentation",
    bio: "System architecture, project scope, timeline, and use cases documentation.",
  },
];

const parseModelKey = (key) => {
  const [type, name] = key.split(":");
  return { type, name: name || null };
};

const chartOptions = {
  responsive: true,
  cutout: "65%",
  plugins: {
    legend: {
      position: "bottom",
      labels: { color: "#9fb0d1", font: { family: "'Inter'" } },
    },
  },
};

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, [pathname]);
  return null;
}

function HomePage() {
  return (
    <div className="home-page">
      <section className="section hero-section">
        <div className="hero-content">
          <p className="eyebrow">Trust Review Platform</p>
          <h1>AI-powered defense against fake reviews.</h1>
          <p className="lead">
            Trust Review protects platforms and shoppers by classifying review authenticity in seconds. We
            combine traditional ML with transformer models for high signal, low latency, and transparent
            confidence scores.
          </p>
          <div className="hero-actions">
            <NavLink className="btn btn-primary" to="/project">Launch Demo</NavLink>
            <NavLink className="btn btn-ghost" to="/about">Meet the Team</NavLink>
          </div>
          <div className="hero-metrics">
            <div>
              <span className="metric-value">ML + DL</span>
              <span className="metric-label">Hybrid Inference</span>
            </div>
            <div>
              <span className="metric-value">CSV · URL</span>
              <span className="metric-label">Multiple Inputs</span>
            </div>
            <div>
              <span className="metric-value">Realtime</span>
              <span className="metric-label">Instant Output</span>
            </div>
          </div>
        </div>
        <div className="hero-panels">
          <div className="info-card">
            <h3>Problem Statement</h3>
            <p>
              Fake reviews distort buyer trust and platform integrity. Manual moderation cannot keep up with
              scale, speed, and adversarial tactics.
            </p>
          </div>
          <div className="info-card accent-card">
            <h3>Our Solution</h3>
            <p>
              A unified pipeline that cleans, analyzes, and labels reviews using proven ML baselines and
              deep learning for stronger generalization.
            </p>
          </div>
          <div className="info-card">
            <h3>Impact</h3>
            <p>
              Accurate labels, confidence insights, and bulk automation help teams maintain trust and
              compliance at scale.
            </p>
          </div>
        </div>
      </section>

      <section className="section home-gallery-section">
        <div className="page-header">
          <div>
            <p className="eyebrow">Interface</p>
            <h2>Product Snapshots</h2>
            <p className="section-subtitle">
              A quick look at the interface and analysis flow inside Trust Review.
            </p>
          </div>
        </div>
        <div className="home-gallery">
          {/* <figure className="gallery-card">
                        <img src={homeScreenOne} alt="Trust Review interface overview" loading="lazy" />
                    </figure> */}
          <figure className="gallery-card">
            <img src={homeScreenTwo} alt="Single review analysis screen" loading="lazy" />
          </figure>
          <figure className="gallery-card">
            <img src={homeScreenThree} alt="Prediction result example" loading="lazy" />
          </figure>
        </div>
      </section>
    </div>
  );
}

function ProjectPage({
  activeTab,
  setActiveTab,
  singleText,
  setSingleText,
  singleModelKey,
  setSingleModelKey,
  analyzeText,
  singleResult,
  singleConfidencePct,
  singleResultRef,
  selectedFile,
  dragOver,
  setDragOver,
  handleFile,
  fileName,
  clearFile,
  fileInputRef,
  bulkModelKey,
  setBulkModelKey,
  analyzeBulk,
  bulkResult,
  bulkResultRef,
  bulkChartData,
  downloadLabelled,
  urlInput,
  setUrlInput,
  urlModelType,
  setUrlModelType,
  analyzeURL,
  urlResult,
  urlResultRef,
  urlChartData,
}) {
  return (
    <section className="section">
      <div className="page-header">
        <div>
          <p className="eyebrow">Live Demo</p>
          <h2>Our Project</h2>
          <p className="section-subtitle">
            Analyze a single review, process bulk uploads, or scrape reviews directly from URLs.
          </p>
        </div>
      </div>

      <div className="project-shell">
        <div className="tab-row">
          <button
            className={`tab ${activeTab === "text" ? "active" : ""}`}
            onClick={() => setActiveTab("text")}
          >
            Single Review
          </button>
          <button
            className={`tab ${activeTab === "bulk" ? "active" : ""}`}
            onClick={() => setActiveTab("bulk")}
          >
            Bulk Upload
          </button>
          <button
            className={`tab ${activeTab === "url" ? "active" : ""}`}
            onClick={() => setActiveTab("url")}
          >
            URL Scrape
          </button>
        </div>

        <section className={`panel ${activeTab === "text" ? "active" : ""}`}>
          <div className="card input-card">
            <h2>Analyze a Review</h2>
            <p className="subtitle">Paste any review text to check if it&apos;s genuine or fabricated</p>
            <textarea
              id="review-input"
              placeholder="Enter or paste a review here…"
              rows="5"
              value={singleText}
              onChange={(e) => setSingleText(e.target.value)}
            ></textarea>
            <div className="controls-row">
              <div className="model-select">
                <label htmlFor="model-type-text">Model</label>
                <select
                  id="model-type-text"
                  value={singleModelKey}
                  onChange={(e) => setSingleModelKey(e.target.value)}
                >
                  {MODEL_CHOICES.map((choice) => (
                    <option key={choice.key} value={choice.key}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </div>
              <button className="btn btn-primary" onClick={analyzeText}>
                Analyze
              </button>
            </div>
          </div>
          {singleResult && (
            <div className="card result-card" ref={singleResultRef}>
              <div className="result-header">
                <h3>Analysis Result</h3>
                <span className={`badge ${singleResult.label === "Real" ? "real" : "fake"}`}>
                  {singleResult.label}
                </span>
              </div>
              <div className="confidence-bar-wrapper">
                <div className="confidence-label">Confidence</div>
                <div className="confidence-track">
                  <div
                    className={`confidence-fill ${singleResult.label === "Real" ? "real" : "fake"}`}
                    style={{ width: `${singleConfidencePct}%` }}
                  ></div>
                </div>
                <div className="confidence-value">{singleConfidencePct}%</div>
              </div>
              <div className="result-meta">Model: {singleResult.model_used}</div>
            </div>
          )}
        </section>

        <section className={`panel ${activeTab === "bulk" ? "active" : ""}`}>
          <div className="card input-card">
            <h2>Bulk Analysis</h2>
            <p className="subtitle">Upload a CSV or XLSX file with a &quot;text&quot; column</p>

            {!selectedFile && (
              <div
                className={`dropzone ${dragOver ? "dragover" : ""}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragOver(true);
                }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragOver(false);
                  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
                }}
              >
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <p>Drag &amp; drop your file here, or <span className="link">browse</span></p>
                <p className="hint">Supports CSV, XLSX</p>
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  hidden
                  ref={fileInputRef}
                  onChange={(e) => {
                    if (e.target.files.length) handleFile(e.target.files[0]);
                  }}
                />
              </div>
            )}

            {selectedFile && (
              <div className="file-info">
                <span>{fileName}</span>
                <button className="btn-icon" onClick={clearFile}>✕</button>
              </div>
            )}

            <div className="controls-row">
              <div className="model-select">
                <label htmlFor="model-type-bulk">Model</label>
                <select
                  id="model-type-bulk"
                  value={bulkModelKey}
                  onChange={(e) => setBulkModelKey(e.target.value)}
                >
                  {MODEL_CHOICES.map((choice) => (
                    <option key={choice.key} value={choice.key}>
                      {choice.label}
                    </option>
                  ))}
                </select>
              </div>
              <button className="btn btn-primary" onClick={analyzeBulk}>
                Process File
              </button>
            </div>
          </div>

          {bulkResult && (
            <div className="card result-card" ref={bulkResultRef}>
              <div className="result-header">
                <h3>Bulk Results</h3>
                <button className="btn btn-outline btn-sm" onClick={downloadLabelled}>
                  Download CSV
                </button>
              </div>
              <div className="bulk-stats">
                <div className="stat-card">
                  <div className="stat-value">{bulkResult.summary.total}</div>
                  <div className="stat-label">Total</div>
                </div>
                <div className="stat-card real">
                  <div className="stat-value">{bulkResult.summary.real_count}</div>
                  <div className="stat-label">Real</div>
                </div>
                <div className="stat-card fake">
                  <div className="stat-value">{bulkResult.summary.fake_count}</div>
                  <div className="stat-label">Fake</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{Math.round(bulkResult.summary.avg_confidence * 100)}%</div>
                  <div className="stat-label">Avg Conf.</div>
                </div>
              </div>

              <div className="chart-container">
                {bulkChartData && <Doughnut data={bulkChartData} options={chartOptions} />}
              </div>

              <div className="bulk-table-wrapper">
                <table id="bulk-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Review (preview)</th>
                      <th>Label</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bulkResult.results.map((r, i) => (
                      <tr key={`${i}-${r.text.slice(0, 12)}`}>
                        <td>{i + 1}</td>
                        <td title={r.text}>
                          {r.text.substring(0, 80)}{r.text.length > 80 ? "…" : ""}
                        </td>
                        <td className={`label-${r.label.toLowerCase()}`}>{r.label}</td>
                        <td>{Math.round(r.confidence * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>

        <section className={`panel ${activeTab === "url" ? "active" : ""}`}>
          <div className="card input-card">
            <h2>Scrape &amp; Analyze</h2>
            <p className="subtitle">Enter a URL to scrape reviews and analyze them automatically</p>
            <div className="url-input-row">
              <input
                type="url"
                placeholder="https://example.com/product/reviews"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
              />
            </div>
            <div className="controls-row">
              <div className="model-select">
                <label htmlFor="model-type-url">Model</label>
                <select
                  id="model-type-url"
                  value={urlModelType}
                  onChange={(e) => setUrlModelType(e.target.value)}
                >
                  <option value="ml">ML (Fast)</option>
                  <option value="dl">Deep Learning</option>
                </select>
              </div>
              <button className="btn btn-primary" onClick={analyzeURL}>
                Scrape &amp; Analyze
              </button>
            </div>
          </div>

          {urlResult && (
            <div className="card result-card" ref={urlResultRef}>
              <div className="result-header">
                <h3>Scraped Reviews</h3>
                <span className="reviews-found">{urlResult.reviews_found} reviews found</span>
              </div>
              <div className="bulk-stats">
                <div className="stat-card">
                  <div className="stat-value">{urlResult.summary.total}</div>
                  <div className="stat-label">Total</div>
                </div>
                <div className="stat-card real">
                  <div className="stat-value">{urlResult.summary.real_count}</div>
                  <div className="stat-label">Real</div>
                </div>
                <div className="stat-card fake">
                  <div className="stat-value">{urlResult.summary.fake_count}</div>
                  <div className="stat-label">Fake</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{Math.round(urlResult.summary.avg_confidence * 100)}%</div>
                  <div className="stat-label">Avg Conf.</div>
                </div>
              </div>

              <div className="chart-container">
                {urlChartData && <Doughnut data={urlChartData} options={chartOptions} />}
              </div>

              <div className="bulk-table-wrapper">
                <table id="url-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Review (preview)</th>
                      <th>Label</th>
                      <th>Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {urlResult.results.map((r, i) => (
                      <tr key={`${i}-${r.text.slice(0, 12)}`}>
                        <td>{i + 1}</td>
                        <td title={r.text}>
                          {r.text.substring(0, 80)}{r.text.length > 80 ? "…" : ""}
                        </td>
                        <td className={`label-${r.label.toLowerCase()}`}>{r.label}</td>
                        <td>{Math.round(r.confidence * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </section>
      </div>
    </section>
  );
}

function AboutPage({ members }) {
  return (
    <section className="section">
      <div className="page-header">
        <div>
          <p className="eyebrow">About Us</p>
          <h2>Meet the Team</h2>
          <p className="section-subtitle">
            A multidisciplinary team combining ML, backend, frontend, and research expertise.
          </p>
        </div>
      </div>

      <div className="team-grid">
        {members.map((member) => (
          <div className="team-card" key={member.name}>
            <div className="team-avatar">{member.name.charAt(0)}</div>
            <div>
              <h3>{member.name}</h3>
              <p className="team-role">{member.role}</p>
              <p className="team-bio">{member.bio}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("text");
  const [loading, setLoading] = useState({ show: false, text: "" });
  const [toast, setToast] = useState({ show: false, message: "" });
  const toastTimeout = useRef(null);

  const [singleText, setSingleText] = useState("");
  const [singleModelKey, setSingleModelKey] = useState("ml:svm");
  const [singleResult, setSingleResult] = useState(null);
  const [singleConfidencePct, setSingleConfidencePct] = useState(0);

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [bulkModelKey, setBulkModelKey] = useState("ml:svm");
  const [bulkResult, setBulkResult] = useState(null);
  const [lastDownloadId, setLastDownloadId] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const [urlInput, setUrlInput] = useState("");
  const [urlModelType, setUrlModelType] = useState("ml");
  const [urlResult, setUrlResult] = useState(null);

  const singleResultRef = useRef(null);
  const bulkResultRef = useRef(null);
  const urlResultRef = useRef(null);

  useEffect(() => {
    if (singleResult && singleResultRef.current) {
      singleResultRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [singleResult]);

  useEffect(() => {
    if (bulkResult && bulkResultRef.current) {
      bulkResultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [bulkResult]);

  useEffect(() => {
    if (urlResult && urlResultRef.current) {
      urlResultRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [urlResult]);

  const showToast = (message) => {
    setToast({ show: true, message });
    if (toastTimeout.current) clearTimeout(toastTimeout.current);
    toastTimeout.current = setTimeout(() => {
      setToast({ show: false, message: "" });
    }, 3500);
  };

  const showLoading = (text) => setLoading({ show: true, text });
  const hideLoading = () => setLoading((prev) => ({ ...prev, show: false }));

  const animateConfidence = (pct) => {
    setSingleConfidencePct(0);
    requestAnimationFrame(() => {
      requestAnimationFrame(() => setSingleConfidencePct(pct));
    });
  };

  const handleFile = (file) => {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["csv", "xlsx"].includes(ext)) {
      showToast("Please upload a CSV or XLSX file.");
      return;
    }
    setSelectedFile(file);
    setFileName(file.name);
  };

  const clearFile = () => {
    setSelectedFile(null);
    setFileName("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyzeText = async () => {
    const text = singleText.trim();
    if (!text) {
      showToast("Please enter a review.");
      return;
    }

    const { type, name } = parseModelKey(singleModelKey);
    showLoading("Analyzing review…");

    try {
      const payload = { text, model_type: type };
      if (name) payload.model_name = name;
      const res = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setSingleResult(data.result);
      animateConfidence(Math.round(data.result.confidence * 100));
    } catch (err) {
      showToast(err.message || "Request failed.");
    } finally {
      hideLoading();
    }
  };

  const analyzeBulk = async () => {
    if (!selectedFile) {
      showToast("Please upload a file first.");
      return;
    }

    const { type, name } = parseModelKey(bulkModelKey);
    const params = new URLSearchParams({ model_type: type });
    if (name) params.set("model_name", name);
    const formData = new FormData();
    formData.append("file", selectedFile);

    showLoading("Processing file…");

    try {
      const res = await fetch(`${API}/api/predict/bulk?${params.toString()}`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setBulkResult(data);
      setLastDownloadId(data.download_id);
    } catch (err) {
      showToast(err.message || "Request failed.");
    } finally {
      hideLoading();
    }
  };

  const analyzeURL = async () => {
    const url = urlInput.trim();
    if (!url) {
      showToast("Please enter a URL.");
      return;
    }

    showLoading("Scraping & analyzing…");

    try {
      const payload = { url, model_type: urlModelType };
      const res = await fetch(`${API}/api/predict/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const data = await res.json();
      setUrlResult(data);
    } catch (err) {
      showToast(err.message || "Request failed.");
    } finally {
      hideLoading();
    }
  };

  const downloadLabelled = () => {
    if (!lastDownloadId) {
      showToast("No file to download.");
      return;
    }
    window.open(`${API}/api/download/${lastDownloadId}`, "_blank");
  };

  const bulkChartData = useMemo(() => {
    if (!bulkResult?.summary) return null;
    return {
      labels: ["Real", "Fake"],
      datasets: [{
        data: [bulkResult.summary.real_count, bulkResult.summary.fake_count],
        backgroundColor: ["#2ce6a9", "#ff6b6b"],
        borderColor: ["rgba(44,230,169,0.3)", "rgba(255,107,107,0.3)"],
        borderWidth: 2,
        hoverOffset: 6,
      }],
    };
  }, [bulkResult]);

  const urlChartData = useMemo(() => {
    if (!urlResult?.summary) return null;
    return {
      labels: ["Real", "Fake"],
      datasets: [{
        data: [urlResult.summary.real_count, urlResult.summary.fake_count],
        backgroundColor: ["#2ce6a9", "#ff6b6b"],
        borderColor: ["rgba(44,230,169,0.3)", "rgba(255,107,107,0.3)"],
        borderWidth: 2,
        hoverOffset: 6,
      }],
    };
  }, [urlResult]);

  return (
    <Router>
      <ScrollToTop />

      <div className="bg-orb bg-orb-1"></div>
      <div className="bg-orb bg-orb-2"></div>
      <div className="bg-grid"></div>

      <header className="site-header">
        <div className="nav-shell">
          <NavLink className="logo" to="/">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </div>
            <div className="logo-text">
              <span>Trust</span>Review
            </div>
          </NavLink>
          <nav className="nav-links">
            <NavLink className="nav-link" to="/">Home</NavLink>
            <NavLink className="nav-link" to="/project">Our Project</NavLink>
            <NavLink className="nav-link" to="/about">About Us</NavLink>
            <NavLink className="nav-cta" to="/project">Try Demo</NavLink>
          </nav>
        </div>
      </header>

      <main className="app">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route
            path="/project"
            element={(
              <ProjectPage
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                singleText={singleText}
                setSingleText={setSingleText}
                singleModelKey={singleModelKey}
                setSingleModelKey={setSingleModelKey}
                analyzeText={analyzeText}
                singleResult={singleResult}
                singleConfidencePct={singleConfidencePct}
                singleResultRef={singleResultRef}
                selectedFile={selectedFile}
                dragOver={dragOver}
                setDragOver={setDragOver}
                handleFile={handleFile}
                fileName={fileName}
                clearFile={clearFile}
                fileInputRef={fileInputRef}
                bulkModelKey={bulkModelKey}
                setBulkModelKey={setBulkModelKey}
                analyzeBulk={analyzeBulk}
                bulkResult={bulkResult}
                bulkResultRef={bulkResultRef}
                bulkChartData={bulkChartData}
                downloadLabelled={downloadLabelled}
                urlInput={urlInput}
                setUrlInput={setUrlInput}
                urlModelType={urlModelType}
                setUrlModelType={setUrlModelType}
                analyzeURL={analyzeURL}
                urlResult={urlResult}
                urlResultRef={urlResultRef}
                urlChartData={urlChartData}
              />
            )}
          />
          <Route path="/about" element={<AboutPage members={TEAM_MEMBERS} />} />
        </Routes>
      </main>

      {loading.show && (
        <div className="loading-overlay">
          <div className="spinner"></div>
          <p>{loading.text}</p>
        </div>
      )}

      <div className={`toast ${toast.show ? "show" : "hidden"}`}>
        {toast.message}
      </div>

      <footer>
        <p>Trust Review v1.0 — Capstone Project · Powered by ML &amp; DistilBERT</p>
      </footer>
    </Router>
  );
}

export default App;
