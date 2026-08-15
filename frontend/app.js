const API_URL = "http://127.0.0.1:8000";

const essayText = document.getElementById("essayText");
const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");

const analyzeButton = document.getElementById("analyzeButton");
const clearButton = document.getElementById("clearButton");

const buttonText = document.getElementById("buttonText");
const spinner = document.getElementById("spinner");

const errorMessage = document.getElementById("errorMessage");
const resultsSection = document.getElementById("resultsSection");

const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidenceBar");

const aiProbability = document.getElementById("aiProbability");
const humanProbability = document.getElementById("humanProbability");
const hybridProbability = document.getElementById("hybridProbability");

const aiBar = document.getElementById("aiBar");
const humanBar = document.getElementById("humanBar");
const hybridBar = document.getElementById("hybridBar");

const featureCount = document.getElementById("featureCount");
const characterCount = document.getElementById("characterCount");
const wordCount = document.getElementById("wordCount");


// ---------------------------------------------------------------------------
// File upload
// ---------------------------------------------------------------------------

fileInput.addEventListener("change", async () => {

    const file = fileInput.files[0];

    if (!file) {
        fileName.textContent = "No file selected";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".txt")) {
        showError("Only .txt files are supported.");
        fileInput.value = "";
        return;
    }

    fileName.textContent = file.name;

    try {
        const text = await file.text();
        essayText.value = text;
    } catch (error) {
        showError("Could not read the selected file.");
    }
});


// ---------------------------------------------------------------------------
// Clear
// ---------------------------------------------------------------------------

clearButton.addEventListener("click", () => {

    essayText.value = "";

    fileInput.value = "";

    fileName.textContent = "No file selected";

    resultsSection.classList.add("hidden");

    hideError();
});


// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

analyzeButton.addEventListener("click", analyzeText);


async function analyzeText() {

    const text = essayText.value.trim();

    hideError();

    if (!text) {
        showError("Please enter some text.");
        return;
    }

    if (text.length < 20) {
        showError(
            "Please enter at least 20 characters."
        );
        return;
    }

    setLoading(true);

    try {

        // ---------------------------------------------------------------
        // Overall prediction
        // ---------------------------------------------------------------

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    text: text,
                }),
            }
        );

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }


        // ---------------------------------------------------------------
        // Sentence predictions
        // ---------------------------------------------------------------

        const sentenceResponse = await fetch(
            `${API_URL}/predict/sentences`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    text: text,
                }),
            }
        );

        const sentenceData =
            await sentenceResponse.json();

        if (!sentenceResponse.ok) {
            throw new Error(
                sentenceData.detail ||
                "Sentence analysis failed."
            );
        }


        // ---------------------------------------------------------------
        // Render
        // ---------------------------------------------------------------

        renderPrediction(data);

        renderStatistics(
            text,
            data
        );

        renderSentenceAnalysis(
            sentenceData.sentences
        );

        resultsSection.classList.remove(
            "hidden"
        );

    } catch (error) {

        console.error(error);

        showError(
            error.message ||
            "Unable to analyze text."
        );

    } finally {

        setLoading(false);
    }
}


// ---------------------------------------------------------------------------
// Overall prediction
// ---------------------------------------------------------------------------

function renderPrediction(data) {

    const label =
        capitalize(data.prediction);

    prediction.textContent = label;

    confidence.textContent =
        `Confidence: ${(data.confidence * 100).toFixed(2)}%`;

    confidenceBar.style.width =
        `${data.confidence * 100}%`;


    const probabilities =
        data.probabilities || {};

    const ai =
        (probabilities.ai || 0) * 100;

    const human =
        (probabilities.human || 0) * 100;

    const hybrid =
        (probabilities.hybrid || 0) * 100;


    aiProbability.textContent =
        `${ai.toFixed(2)}%`;

    humanProbability.textContent =
        `${human.toFixed(2)}%`;

    hybridProbability.textContent =
        `${hybrid.toFixed(2)}%`;


    aiBar.style.width =
        `${ai}%`;

    humanBar.style.width =
        `${human}%`;

    hybridBar.style.width =
        `${hybrid}%`;
}


// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

function renderStatistics(text, data) {

    featureCount.textContent =
        data.feature_count ?? "—";

    characterCount.textContent =
        text.length.toLocaleString();

    wordCount.textContent =
        countWords(text).toLocaleString();
}


function countWords(text) {

    return text
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .length;
}


// ---------------------------------------------------------------------------
// Sentence highlighting
// ---------------------------------------------------------------------------

function renderSentenceAnalysis(sentences) {

    let existing =
        document.getElementById(
            "sentenceAnalysis"
        );

    if (!existing) {

        existing =
            document.createElement("div");

        existing.id =
            "sentenceAnalysis";

        existing.className =
            "card sentence-analysis";

        resultsSection.appendChild(
            existing
        );
    }


    existing.innerHTML = `
        <div class="card-header">
            <div>
                <h2>Sentence Analysis</h2>

                <p>
                    Independent sentence-level model predictions.
                </p>
            </div>
        </div>

        <div class="highlighted-text">
            ${sentences
    .map(sentence => {

        const colors = blendSentenceColor(sentence.probabilities);

        const probsText = Object.entries(sentence.probabilities)
            .map(([k, v]) => `${capitalize(k)}: ${(v * 100).toFixed(1)}%`)
            .join(" · ");

        return `
            <span
                class="sentence"
                style="background: ${colors.background}; box-shadow: inset 0 -3px 0 ${colors.border};"
                title="${probsText}"
            >${escapeHtml(sentence.text)}</span>
        `;

    })
    .join(" ")}
        </div>

        <div class="legend legend-full">
    <div class="legend-group">
        <span class="legend-title">Pure signals</span>
        <span><i class="legend-ai"></i>AI-leaning</span>
        <span><i class="legend-human"></i>Human-leaning</span>
        <span><i class="legend-hybrid"></i>Hybrid-leaning</span>
    </div>
    <div class="legend-group">
        <span class="legend-title">Mixed / uncertain</span>
        ${legendSwatch({ai: 0.5, human: 0.5, hybrid: 0}, "AI + Human tied")}
        ${legendSwatch({ai: 0, human: 0.5, hybrid: 0.5}, "Human + Hybrid tied")}
        ${legendSwatch({ai: 0.5, human: 0, hybrid: 0.5}, "AI + Hybrid tied")}
        ${legendSwatch({ai: 0.34, human: 0.33, hybrid: 0.33}, "All three tied")}
    </div>
</div>
    `;
}

// ---------------------------------------------------------------------------
// Probability-weighted color blending
// ---------------------------------------------------------------------------

const CLASS_COLORS = {
    ai: [231, 111, 81],      // matches --ai in style.css
    human: [76, 201, 160],   // matches --human
    hybrid: [139, 124, 255], // matches --hybrid
};

function blendSentenceColor(probabilities) {
    let r = 0, g = 0, b = 0;

    for (const key of Object.keys(CLASS_COLORS)) {
        const weight = probabilities[key] || 0;
        const [cr, cg, cb] = CLASS_COLORS[key];
        r += cr * weight;
        g += cg * weight;
        b += cb * weight;
    }

    return {
        background: `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, 0.4)`,
        border: `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, 0.9)`,
    };
}
function legendSwatch(probs, label) {
    const colors = blendSentenceColor(probs);
    return `
        <span>
            <i style="background: ${colors.border}; width:10px; height:10px; display:inline-block; border-radius:3px;"></i>
            ${label}
        </span>
    `;
}
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function capitalize(value) {

    if (!value) {
        return "";
    }

    return value.charAt(0).toUpperCase() +
        value.slice(1);
}


function escapeHtml(value) {

    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


function setLoading(loading) {

    analyzeButton.disabled = loading;

    spinner.classList.toggle(
        "hidden",
        !loading
    );

    buttonText.textContent =
        loading
            ? "Analyzing..."
            : "Analyze Text";
}


function showError(message) {

    errorMessage.textContent = message;

    errorMessage.classList.remove(
        "hidden"
    );
}


function hideError() {

    errorMessage.classList.add(
        "hidden"
    );

    errorMessage.textContent = "";
}