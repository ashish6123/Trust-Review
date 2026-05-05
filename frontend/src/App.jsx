import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HashRouter as Router, NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Bar, Doughnut } from "react-chartjs-2";
import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import homeScreenTwo from "./assets/home-screen-2.png";
import homeScreenThree from "./assets/home-screen-3.png";

ChartJS.register(ArcElement, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const API = "";
const HISTORY_KEY = "trust_review_history_v1";
const THEME_KEY = "trust_review_theme";
const HISTORY_LIMIT = 25;

const FALLBACK_MODEL_CHOICES = [
  { key: "ml:svm", label: "SVM (calibrated)", group: "ml" },
  { key: "ml:logistic_regression", label: "Logistic Regression", group: "ml" },
  { key: "ml:random_forest", label: "Random Forest", group: "ml" },
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
    bio: "FastAPI backend, prediction endpoint, model serving, and CSV handling.",
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

const prettyModelName = (rawName) => {
  if (!rawName) return "";
  return rawName
    .split("_")
    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
    .join(" ");
};

const chartOptions = {
  responsive: true,
  cutout: "65%",
  plugins: {
    legend: {
      position: "bottom",
      labels: { color: "var(--text-dim)", font: { family: "'Inter'" } },
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

/* ───────── Home ───────── */
function HomePage({ serverInfo }) {
  const topMetric = useMemo(() => {
    const models = serverInfo?.metrics?.models;
    if (!models?.length) return null;
    return [...models].sort((a, b) => (b.f1 || 0) - (a.f1 || 0))[0];
  }, [serverInfo]);

  return (
    <div className="home-page">
      <section className="section hero-section">
        <div className="hero-content">
          <p className="eyebrow">Trust Review Platform</p>
          <h1>AI-powered defense against fake reviews.</h1>
          <p className="lead">
            Trust Review protects platforms and shoppers by classifying review authenticity in
            seconds. We combine traditional ML with transformer models for high signal, low latency,
            and transparent confidence scores.
          </p>
          <div className="hero-actions">
            <NavLink className="btn btn-primary" to="/project">Launch Demo</NavLink>
            <NavLink className="btn btn-ghost" to="/about">Meet the Team</NavLink>
          </div>
          <div className="hero-metrics">
            {topMetric ? (
              <>
                <div>
                  <span className="metric-value">{(topMetric.f1 * 100).toFixed(1)}%</span>
                  <span className="metric-label">Best F1 ({topMetric.name})</span>
                </div>
                <div>
                  <span className="metric-value">{(topMetric.accuracy * 100).toFixed(1)}%</span>
                  <span className="metric-label">Test Accuracy</span>
                </div>
                <div>
                  <span className="metric-value">{(topMetric.roc_auc * 100).toFixed(1)}%</span>
                  <span className="metric-label">ROC-AUC</span>
                </div>
              </>
            ) : (
              <>
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
              </>
            )}
          </div>
        </div>
        <div className="hero-panels">
          <div className="info-card">
            <h3>Problem Statement</h3>
            <p>
              Fake reviews distort buyer trust and platform integrity. Manual moderation cannot keep
              up with scale, speed, and adversarial tactics.
            </p>
          </div>
          <div className="info-card accent-card">
            <h3>Our Solution</h3>
            <p>
              A unified pipeline that cleans, analyzes, and labels reviews using proven ML baselines
              and deep learning for stronger generalization.
            </p>
          </div>
          <div className="info-card">
            <h3>Impact</h3>
            <p>
              Accurate labels, confidence insights, and bulk automation help teams maintain trust
              and compliance at scale.
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
          <figure className="gallery-card">
            <img src={homeScreenTwo} alt="Single review analysis screen" loading="lazy" />
          </figure>
          <figure className="gallery-card">
            <img src={homeScreenThree} alt="Prediction result example" loading="lazy" />
          </figure>
        </div>
      </section>

      {serverInfo?.metrics?.models?.length > 0 && (
        <section className="section">
          <div className="page-header">
            <div>
              <p className="eyebrow">Benchmarks</p>
              <h2>Model Performance</h2>
              <p className="section-subtitle">
                Latest evaluation on the held-out test set (run{" "}
                <code>python -m training.compare_models</code> to refresh).
              </p>
            </div>
          </div>
          <MetricsTable models={serverInfo.metrics.models} />
        </section>
      )}
    </div>
  );
}

function MetricsTable({ models }) {
  return (
    <div className="metrics-table-wrap">
      <table className="metrics-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {[...models]
            .sort((a, b) => (b.f1 || 0) - (a.f1 || 0))
            .map((m) => (
              <tr key={m.name}>
                <td>{m.name}</td>
                <td>{(m.accuracy * 100).toFixed(2)}%</td>
                <td>{(m.precision * 100).toFixed(2)}%</td>
                <td>{(m.recall * 100).toFixed(2)}%</td>
                <td>
                  <strong>{(m.f1 * 100).toFixed(2)}%</strong>
                </td>
                <td>{(m.roc_auc * 100).toFixed(2)}%</td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}

/* ───────── Token-level explanation ───────── */
function ExplanationView({ explanation }) {
  if (!explanation) return null;
  if (!explanation.tokens?.length) {
    return (
      <p className="explain-empty">
        No token-level explanation available for this prediction.
      </p>
    );
  }

  const maxAbs = Math.max(...explanation.tokens.map((t) => Math.abs(t.weight)), 1e-9);
  const direction = explanation.result.label;

  return (
    <div className="explain-card">
      <h4>Why this label?</h4>
      <p className="subtitle">
        Tokens that pushed the model toward <strong>{direction}</strong> (red) or away (green).
      </p>
      <div className="token-cloud" role="list">
        {explanation.tokens.map((t, idx) => {
          const isFakeDirection = direction === "Fake";
          const towardLabel = isFakeDirection ? t.weight > 0 : t.weight < 0;
          const intensity = Math.min(1, Math.abs(t.weight) / maxAbs);
          const className = towardLabel
            ? "token toward-fake"
            : "token toward-real";
          return (
            <span
              key={`${t.token}-${idx}`}
              role="listitem"
              className={className}
              style={{ "--intensity": intensity }}
              title={`${t.token}: ${t.weight.toFixed(4)}`}
            >
              {t.token}
            </span>
          );
        })}
      </div>
      <p className="explain-meta">
        Method: <code>{explanation.explanation_kind}</code>
      </p>
    </div>
  );
}

/* ───────── History panel ───────── */
function HistoryPanel({ items, onSelect, onClear }) {
  if (!items.length) {
    return (
      <div className="history-empty">
        <p>Recent predictions will appear here.</p>
      </div>
    );
  }
  return (
    <div className="history-list">
      <div className="history-header">
        <h4>Recent</h4>
        <button className="btn-text" onClick={onClear}>Clear</button>
      </div>
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <button className="history-item" onClick={() => onSelect(item)}>
              <span className={`history-badge ${item.label.toLowerCase()}`}>{item.label}</span>
              <span className="history-text">{item.text.slice(0, 80)}{item.text.length > 80 ? "…" : ""}</span>
              <span className="history-meta">
                {Math.round(item.confidence * 100)}% · {prettyModelName(item.modelUsed)}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* ───────── Comparison view ───────── */
function ComparePanel({
  compareText,
  setCompareText,
  runCompare,
  compareResults,
  compareLoading,
}) {
  return (
    <div className="card input-card">
      <h2>Compare Models</h2>
      <p className="subtitle">
        Run the same review through every available model and compare their verdicts and confidence.
      </p>
      <textarea
        id="compare-input"
        placeholder="Paste a review and compare all models side-by-side…"
        rows="4"
        value={compareText}
        onChange={(e) => setCompareText(e.target.value)}
      />
      <div className="controls-row">
        <button
          className="btn btn-primary"
          onClick={runCompare}
          disabled={compareLoading}
        >
          {compareLoading ? "Comparing…" : "Compare All Models"}
        </button>
      </div>

      {compareResults.length > 0 && (
        <div className="compare-grid" aria-live="polite">
          {compareResults.map((r) => (
            <div className="compare-cell" key={r.modelKey}>
              <div className="compare-cell-header">
                <span className="compare-model">{r.modelLabel}</span>
                {r.error ? (
                  <span className="badge fake">error</span>
                ) : (
                  <span className={`badge ${r.label.toLowerCase()}`}>{r.label}</span>
                )}
              </div>
              {r.error ? (
                <p className="compare-error">{r.error}</p>
              ) : (
                <>
                  <div className="confidence-track">
                    <div
                      className={`confidence-fill ${r.label.toLowerCase()}`}
                      style={{ width: `${Math.round(r.confidence * 100)}%` }}
                    />
                  </div>
                  <div className="compare-conf">
                    {Math.round(r.confidence * 100)}% confident
                  </div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ───────── Project (demo) page ───────── */
function ProjectPage(props) {
  const {
    activeTab, setActiveTab,
    modelChoices,
    singleText, setSingleText,
    singleModelKey, setSingleModelKey,
    analyzeText,
    singleResult, singleConfidencePct, singleResultRef,
    explanation,
    history, onSelectHistoryItem, onClearHistory,
    selectedFile, dragOver, setDragOver,
    handleFile, fileName, clearFile, fileInputRef,
    bulkModelKey, setBulkModelKey, analyzeBulk,
    bulkResult, bulkResultRef, bulkChartData, bulkConfidenceChartData, downloadLabelled,
    urlInput, setUrlInput, urlModelKey, setUrlModelKey, analyzeURL,
    urlResult, urlResultRef, urlChartData,
    compareText, setCompareText, runCompare, compareResults, compareLoading,
  } = props;

  const dropzoneActivate = (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInputRef.current?.click();
    }
  };

  return (
    <section className="section">
      <div className="page-header">
        <div>
          <p className="eyebrow">Live Demo</p>
          <h2>Our Project</h2>
          <p className="section-subtitle">
            Analyze a single review, compare models, process bulk uploads, or scrape reviews
            directly from URLs.
          </p>
        </div>
      </div>

      <div className="project-shell">
        <div className="tab-row" role="tablist">
          {[
            ["text", "Single Review"],
            ["compare", "Compare Models"],
            ["bulk", "Bulk Upload"],
            ["url", "URL Scrape"],
          ].map(([key, label]) => (
            <button
              key={key}
              role="tab"
              aria-selected={activeTab === key}
              className={`tab ${activeTab === key ? "active" : ""}`}
              onClick={() => setActiveTab(key)}
            >
              {label}
            </button>
          ))}
        </div>

        <section className={`panel ${activeTab === "text" ? "active" : ""}`} role="tabpanel">
          <div className="single-grid">
            <div className="single-main">
              <div className="card input-card">
                <h2>Analyze a Review</h2>
                <p className="subtitle">
                  Paste any review text to check if it&apos;s genuine or fabricated.
                </p>
                <textarea
                  id="review-input"
                  placeholder="Enter or paste a review here…"
                  rows="5"
                  value={singleText}
                  onChange={(e) => setSingleText(e.target.value)}
                />
                <div className="controls-row">
                  <div className="model-select">
                    <label htmlFor="model-type-text">Model</label>
                    <select
                      id="model-type-text"
                      value={singleModelKey}
                      onChange={(e) => setSingleModelKey(e.target.value)}
                    >
                      {modelChoices.map((choice) => (
                        <option key={choice.key} value={choice.key} disabled={choice.disabled}>
                          {choice.label}{choice.disabled ? " — not loaded" : ""}
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
                <div
                  className="card result-card"
                  ref={singleResultRef}
                  aria-live="polite"
                >
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
                      />
                    </div>
                    <div className="confidence-value">{singleConfidencePct}%</div>
                  </div>
                  <div className="result-meta">
                    Model: <strong>{prettyModelName(singleResult.model_used)}</strong>
                  </div>
                  <ExplanationView explanation={explanation} />
                </div>
              )}
            </div>

            <aside className="single-side" aria-label="Recent predictions">
              <HistoryPanel
                items={history}
                onSelect={onSelectHistoryItem}
                onClear={onClearHistory}
              />
            </aside>
          </div>
        </section>

        <section className={`panel ${activeTab === "compare" ? "active" : ""}`} role="tabpanel">
          <ComparePanel
            compareText={compareText}
            setCompareText={setCompareText}
            runCompare={runCompare}
            compareResults={compareResults}
            compareLoading={compareLoading}
          />
        </section>

        <section className={`panel ${activeTab === "bulk" ? "active" : ""}`} role="tabpanel">
          <div className="card input-card">
            <h2>Bulk Analysis</h2>
            <p className="subtitle">Upload a CSV or XLSX file with a &quot;text&quot; column.</p>

            {!selectedFile && (
              <div
                className={`dropzone ${dragOver ? "dragover" : ""}`}
                role="button"
                tabIndex={0}
                aria-label="Upload CSV or XLSX file"
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={dropzoneActivate}
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
                <p>Drag &amp; drop your file here, or <span className="link">browse</span></p>
                <p className="hint">Supports CSV, XLSX (max 10 MB)</p>
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
                <button className="btn-icon" onClick={clearFile} aria-label="Remove selected file">
                  ✕
                </button>
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
                  {modelChoices.map((choice) => (
                    <option key={choice.key} value={choice.key} disabled={choice.disabled}>
                      {choice.label}{choice.disabled ? " — not loaded" : ""}
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
            <div className="card result-card" ref={bulkResultRef} aria-live="polite">
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

              <div className="charts-row">
                <div className="chart-container">
                  {bulkChartData && <Doughnut data={bulkChartData} options={chartOptions} />}
                </div>
                <div className="chart-container">
                  {bulkConfidenceChartData && (
                    <Bar data={bulkConfidenceChartData} options={confidenceBarOptions} />
                  )}
                </div>
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

        <section className={`panel ${activeTab === "url" ? "active" : ""}`} role="tabpanel">
          <div className="card input-card">
            <h2>Scrape &amp; Analyze</h2>
            <p className="subtitle">Enter a URL to scrape reviews and analyze them automatically.</p>
            <div className="url-input-row">
              <input
                type="url"
                placeholder="https://example.com/product/reviews"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                aria-label="URL to scrape"
              />
            </div>
            <div className="controls-row">
              <div className="model-select">
                <label htmlFor="model-type-url">Model</label>
                <select
                  id="model-type-url"
                  value={urlModelKey}
                  onChange={(e) => setUrlModelKey(e.target.value)}
                >
                  {modelChoices.map((choice) => (
                    <option key={choice.key} value={choice.key} disabled={choice.disabled}>
                      {choice.label}{choice.disabled ? " — not loaded" : ""}
                    </option>
                  ))}
                </select>
              </div>
              <button className="btn btn-primary" onClick={analyzeURL}>
                Scrape &amp; Analyze
              </button>
            </div>
          </div>

          {urlResult && (
            <div className="card result-card" ref={urlResultRef} aria-live="polite">
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

const confidenceBarOptions = {
  responsive: true,
  plugins: {
    legend: { display: false },
    tooltip: { mode: "index", intersect: false },
  },
  scales: {
    x: { ticks: { color: "var(--text-dim)" }, grid: { display: false } },
    y: {
      beginAtZero: true,
      ticks: { color: "var(--text-dim)", precision: 0 },
      grid: { color: "rgba(127, 140, 175, 0.1)" },
    },
  },
};

/* ───────── About ───────── */
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

/* ───────── App ───────── */
function App() {
  const [activeTab, setActiveTab] = useState("text");
  const [loading, setLoading] = useState({ show: false, text: "" });
  const [toast, setToast] = useState({ show: false, message: "", kind: "info" });
  const toastTimeout = useRef(null);

  const [serverInfo, setServerInfo] = useState(null);
  const [theme, setTheme] = useState(() => {
    if (typeof window === "undefined") return "dark";
    return window.localStorage.getItem(THEME_KEY) || "dark";
  });

  const [singleText, setSingleText] = useState("");
  const [singleModelKey, setSingleModelKey] = useState("ml:svm");
  const [singleResult, setSingleResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [singleConfidencePct, setSingleConfidencePct] = useState(0);

  const [history, setHistory] = useState(() => {
    if (typeof window === "undefined") return [];
    try {
      return JSON.parse(window.localStorage.getItem(HISTORY_KEY) || "[]");
    } catch {
      return [];
    }
  });

  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [bulkModelKey, setBulkModelKey] = useState("ml:svm");
  const [bulkResult, setBulkResult] = useState(null);
  const [lastDownloadId, setLastDownloadId] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const [urlInput, setUrlInput] = useState("");
  const [urlModelKey, setUrlModelKey] = useState("ml:svm");
  const [urlResult, setUrlResult] = useState(null);

  const [compareText, setCompareText] = useState("");
  const [compareResults, setCompareResults] = useState([]);
  const [compareLoading, setCompareLoading] = useState(false);

  const singleResultRef = useRef(null);
  const bulkResultRef = useRef(null);
  const urlResultRef = useRef(null);

  /* Theme */
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_KEY, theme);
    }
  }, [theme]);

  /* Persist history */
  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    }
  }, [history]);

  /* Fetch server info on mount */
  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/api/info`)
      .then((res) => (res.ok ? res.json() : null))
      .then((info) => {
        if (cancelled || !info) return;
        setServerInfo(info);
        // If the previously selected model is no longer available, fall back.
        const choices = buildModelChoices(info);
        const activeKeys = new Set(choices.filter((c) => !c.disabled).map((c) => c.key));
        const fallback = choices.find((c) => !c.disabled)?.key || "ml:svm";
        for (const [val, set] of [
          [singleModelKey, setSingleModelKey],
          [bulkModelKey, setBulkModelKey],
          [urlModelKey, setUrlModelKey],
        ]) {
          if (!activeKeys.has(val)) set(fallback);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const modelChoices = useMemo(
    () => (serverInfo ? buildModelChoices(serverInfo) : FALLBACK_MODEL_CHOICES),
    [serverInfo]
  );

  /* Auto-scroll to results */
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

  /* Toast / loading helpers */
  const showToast = useCallback((message, kind = "info") => {
    setToast({ show: true, message, kind });
    if (toastTimeout.current) clearTimeout(toastTimeout.current);
    toastTimeout.current = setTimeout(() => {
      setToast({ show: false, message: "", kind });
    }, 3500);
  }, []);

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
      showToast("Please upload a CSV or XLSX file.", "error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showToast("File is larger than 10 MB.", "error");
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

  const pushHistory = (entry) => {
    setHistory((prev) => {
      const next = [entry, ...prev.filter((p) => p.id !== entry.id)].slice(0, HISTORY_LIMIT);
      return next;
    });
  };

  const onSelectHistoryItem = (item) => {
    setActiveTab("text");
    setSingleText(item.text);
    setSingleModelKey(item.modelKey || "ml:svm");
  };

  const onClearHistory = () => setHistory([]);

  /* API calls */
  const analyzeText = async () => {
    const text = singleText.trim();
    if (!text) {
      showToast("Please enter a review.", "error");
      return;
    }

    const { type, name } = parseModelKey(singleModelKey);
    showLoading("Analyzing review…");

    try {
      const payload = { text, model_type: type, top_k: 12 };
      if (name) payload.model_name = name;

      const res = await fetch(`${API}/api/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        // Fall back to /api/predict if explain fails (e.g. DL).
        const fallbackRes = await fetch(`${API}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, model_type: type, ...(name ? { model_name: name } : {}) }),
        });
        if (!fallbackRes.ok) {
          const err = await fallbackRes.json().catch(() => ({}));
          throw new Error(err.detail || fallbackRes.statusText);
        }
        const data = await fallbackRes.json();
        setSingleResult(data.result);
        setExplanation(null);
        animateConfidence(Math.round(data.result.confidence * 100));
        pushHistory({
          id: `${Date.now()}`,
          text: data.result.text,
          label: data.result.label,
          confidence: data.result.confidence,
          modelUsed: data.result.model_used,
          modelKey: singleModelKey,
        });
        return;
      }
      const data = await res.json();
      setSingleResult(data.result);
      setExplanation(data);
      animateConfidence(Math.round(data.result.confidence * 100));
      pushHistory({
        id: `${Date.now()}`,
        text: data.result.text,
        label: data.result.label,
        confidence: data.result.confidence,
        modelUsed: data.result.model_used,
        modelKey: singleModelKey,
      });
    } catch (err) {
      showToast(err.message || "Request failed.", "error");
    } finally {
      hideLoading();
    }
  };

  const analyzeBulk = async () => {
    if (!selectedFile) {
      showToast("Please upload a file first.", "error");
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
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      setBulkResult(data);
      setLastDownloadId(data.download_id);
    } catch (err) {
      showToast(err.message || "Request failed.", "error");
    } finally {
      hideLoading();
    }
  };

  const analyzeURL = async () => {
    const url = urlInput.trim();
    if (!url) {
      showToast("Please enter a URL.", "error");
      return;
    }

    const { type, name } = parseModelKey(urlModelKey);
    showLoading("Scraping & analyzing…");

    try {
      const payload = { url, model_type: type };
      if (name) payload.model_name = name;
      const res = await fetch(`${API}/api/predict/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || res.statusText);
      }
      const data = await res.json();
      setUrlResult(data);
    } catch (err) {
      showToast(err.message || "Request failed.", "error");
    } finally {
      hideLoading();
    }
  };

  const runCompare = async () => {
    const text = compareText.trim();
    if (!text) {
      showToast("Please enter a review to compare.", "error");
      return;
    }
    const usable = modelChoices.filter((c) => !c.disabled);
    if (!usable.length) {
      showToast("No models available to compare.", "error");
      return;
    }

    setCompareLoading(true);
    setCompareResults(usable.map((c) => ({ modelKey: c.key, modelLabel: c.label })));

    const results = await Promise.all(
      usable.map(async (choice) => {
        const { type, name } = parseModelKey(choice.key);
        try {
          const res = await fetch(`${API}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text,
              model_type: type,
              ...(name ? { model_name: name } : {}),
            }),
          });
          if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            return {
              modelKey: choice.key,
              modelLabel: choice.label,
              error: err.detail || res.statusText,
            };
          }
          const data = await res.json();
          return {
            modelKey: choice.key,
            modelLabel: choice.label,
            label: data.result.label,
            confidence: data.result.confidence,
          };
        } catch (e) {
          return {
            modelKey: choice.key,
            modelLabel: choice.label,
            error: e.message || "Request failed",
          };
        }
      })
    );

    setCompareResults(results);
    setCompareLoading(false);
  };

  const downloadLabelled = () => {
    if (!lastDownloadId) {
      showToast("No file to download.", "error");
      return;
    }
    window.open(`${API}/api/download/${lastDownloadId}`, "_blank");
  };

  /* Charts */
  const bulkChartData = useMemo(() => {
    if (!bulkResult?.summary) return null;
    return {
      labels: ["Real", "Fake"],
      datasets: [
        {
          data: [bulkResult.summary.real_count, bulkResult.summary.fake_count],
          backgroundColor: ["#2ce6a9", "#ff6b6b"],
          borderColor: ["rgba(44,230,169,0.3)", "rgba(255,107,107,0.3)"],
          borderWidth: 2,
          hoverOffset: 6,
        },
      ],
    };
  }, [bulkResult]);

  const bulkConfidenceChartData = useMemo(() => {
    if (!bulkResult?.results) return null;
    const buckets = [0, 0, 0, 0, 0]; // 0-20, 20-40, 40-60, 60-80, 80-100
    for (const r of bulkResult.results) {
      const idx = Math.min(4, Math.floor(r.confidence * 5));
      buckets[idx] += 1;
    }
    return {
      labels: ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
      datasets: [
        {
          label: "Reviews",
          data: buckets,
          backgroundColor: "rgba(108, 125, 255, 0.6)",
          borderColor: "rgba(108, 125, 255, 1)",
          borderWidth: 1,
          borderRadius: 6,
        },
      ],
    };
  }, [bulkResult]);

  const urlChartData = useMemo(() => {
    if (!urlResult?.summary) return null;
    return {
      labels: ["Real", "Fake"],
      datasets: [
        {
          data: [urlResult.summary.real_count, urlResult.summary.fake_count],
          backgroundColor: ["#2ce6a9", "#ff6b6b"],
          borderColor: ["rgba(44,230,169,0.3)", "rgba(255,107,107,0.3)"],
          borderWidth: 2,
          hoverOffset: 6,
        },
      ],
    };
  }, [urlResult]);

  const dlReady = serverInfo?.dl_available ?? false;

  return (
    <Router>
      <ScrollToTop />

      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-grid" />

      <header className="site-header">
        <div className="nav-shell">
          <NavLink className="logo" to="/">
            <div className="logo-icon" aria-hidden>
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
            <button
              type="button"
              className="theme-toggle"
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
              onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? "☾" : "☀"}
            </button>
            <NavLink className="nav-cta" to="/project">Try Demo</NavLink>
          </nav>
        </div>
        {!dlReady && serverInfo && (
          <div className="status-banner">
            DistilBERT model is not loaded — only ML models are available.
          </div>
        )}
      </header>

      <main className="app">
        <Routes>
          <Route path="/" element={<HomePage serverInfo={serverInfo} />} />
          <Route
            path="/project"
            element={(
              <ProjectPage
                activeTab={activeTab}
                setActiveTab={setActiveTab}
                modelChoices={modelChoices}
                singleText={singleText}
                setSingleText={setSingleText}
                singleModelKey={singleModelKey}
                setSingleModelKey={setSingleModelKey}
                analyzeText={analyzeText}
                singleResult={singleResult}
                singleConfidencePct={singleConfidencePct}
                singleResultRef={singleResultRef}
                explanation={explanation}
                history={history}
                onSelectHistoryItem={onSelectHistoryItem}
                onClearHistory={onClearHistory}
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
                bulkConfidenceChartData={bulkConfidenceChartData}
                downloadLabelled={downloadLabelled}
                urlInput={urlInput}
                setUrlInput={setUrlInput}
                urlModelKey={urlModelKey}
                setUrlModelKey={setUrlModelKey}
                analyzeURL={analyzeURL}
                urlResult={urlResult}
                urlResultRef={urlResultRef}
                urlChartData={urlChartData}
                compareText={compareText}
                setCompareText={setCompareText}
                runCompare={runCompare}
                compareResults={compareResults}
                compareLoading={compareLoading}
              />
            )}
          />
          <Route path="/about" element={<AboutPage members={TEAM_MEMBERS} />} />
        </Routes>
      </main>

      {loading.show && (
        <div className="loading-overlay" role="status" aria-live="polite">
          <div className="spinner" />
          <p>{loading.text}</p>
        </div>
      )}

      <div
        className={`toast ${toast.show ? "show" : "hidden"} toast-${toast.kind}`}
        role="status"
        aria-live="polite"
      >
        {toast.message}
      </div>

      <footer>
        <p>Trust Review v1.1 — Capstone Project · Powered by ML &amp; DistilBERT</p>
      </footer>
    </Router>
  );
}

function buildModelChoices(info) {
  const choices = [];
  const mlNames = info?.available_ml_models || [];
  const ordered = [
    ["svm", "SVM (calibrated)"],
    ["logistic_regression", "Logistic Regression"],
    ["random_forest", "Random Forest"],
  ];
  for (const [name, label] of ordered) {
    if (mlNames.includes(name)) {
      choices.push({ key: `ml:${name}`, label, group: "ml" });
    }
  }
  // Any unknown ml models the server reports
  for (const name of mlNames) {
    if (!ordered.find(([n]) => n === name)) {
      choices.push({ key: `ml:${name}`, label: prettyModelName(name), group: "ml" });
    }
  }
  if (info?.dl_available) {
    choices.push({ key: "dl", label: "DistilBERT (transformer)", group: "dl" });
  } else {
    choices.push({ key: "dl", label: "DistilBERT", group: "dl", disabled: true });
  }
  if (!choices.length) return FALLBACK_MODEL_CHOICES;
  return choices;
}

export default App;
