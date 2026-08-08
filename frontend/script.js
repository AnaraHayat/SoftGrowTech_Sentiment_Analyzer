const API_BASE = "http://localhost:5000";

const textarea = document.getElementById("input-text");
const analyzeBtn = document.getElementById("analyze-btn");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error-msg");
const stampEl = document.getElementById("stamp");
const confidenceText = document.getElementById("confidence-text");
const mlBars = document.getElementById("ml-bars");
const tbReadout = document.getElementById("tb-readout");
const apiStatus = document.getElementById("api-status");

const SAMPLES = {
  positive: "This is hands down the best purchase I've made all year — it works flawlessly and the support team was incredibly helpful.",
  negative: "Completely disappointed. It broke after two days and customer service never responded to my emails.",
  neutral: "The device measures 14 centimeters and ships in a plain cardboard box with a printed manual."
};

document.querySelectorAll(".sample-links button").forEach((btn) => {
  btn.addEventListener("click", () => {
    textarea.value = SAMPLES[btn.dataset.sample];
    textarea.focus();
  });
});

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error();
    apiStatus.classList.add("live");
    apiStatus.classList.remove("down");
    apiStatus.innerHTML = "<i></i>API connected";
  } catch (e) {
    apiStatus.classList.add("down");
    apiStatus.classList.remove("live");
    apiStatus.innerHTML = "<i></i>API offline — run backend/app.py";
  }
}
checkHealth();

function labelClass(label) {
  if (label === "positive") return "positive";
  if (label === "negative") return "negative";
  return "neutral";
}

function renderBar(container, label, value, cls) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `
    <span class="bar-label">${label}</span>
    <span class="bar-track"><span class="bar-fill ${cls}" style="width:${(value * 100).toFixed(1)}%"></span></span>
    <span class="bar-value">${(value * 100).toFixed(1)}%</span>
  `;
  container.appendChild(row);
}

async function analyze() {
  const text = textarea.value.trim();
  errorBox.classList.remove("show");
  errorBox.textContent = "";

  if (!text) {
    errorBox.textContent = "Type or paste some text first.";
    errorBox.classList.add("show");
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing…";

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.error || `Request failed (${res.status})`);
    }

    const data = await res.json();

    // Single source of truth: the ML model prediction.
    // The stamp, verdict, confidence, and probability bars all use this same result.
    const verdict = data.ml_model.sentiment;
    stampEl.textContent = verdict;
    stampEl.className = `stamp ${labelClass(verdict)}`;
    confidenceText.innerHTML = `model verdict <b>${verdict}</b> · confidence <b>${(data.ml_model.confidence * 100).toFixed(1)}%</b>`;

    // ML probability bars
    mlBars.innerHTML = "";
    const probs = data.ml_model.class_probabilities;
    Object.keys(probs).sort().forEach((cls) => {
      renderBar(mlBars, cls, probs[cls], labelClass(cls));
    });

    // TextBlob is diagnostic only. It no longer provides a competing verdict.
    const tb = data.textblob;
    tbReadout.innerHTML = `
      <div class="bar-row">
        <span class="bar-label">ML verdict</span>
        <span style="color: var(--ink); font-weight:600;">${verdict}</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">TextBlob check</span>
        <span style="color: var(--muted);">${tb.sentiment}</span>
      </div>
      <div class="bar-row">
        <span class="bar-label">polarity</span>
        <span style="color: var(--muted);">${tb.polarity > 0 ? "+" : ""}${tb.polarity}</span>
      </div>
    `;

    resultBox.classList.add("show");
  } catch (err) {
    errorBox.textContent = `Couldn't reach the API: ${err.message}. Make sure backend/app.py is running on port 5000.`;
    errorBox.classList.add("show");
    resultBox.classList.remove("show");
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Analyze sentiment";
  }
}

analyzeBtn.addEventListener("click", analyze);
textarea.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
});
