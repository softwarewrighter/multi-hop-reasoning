/**
 * KG Reward Demo - Main Application
 *
 * Visualizes training episodes with:
 * - Knowledge graph with path highlighting
 * - Model output with entity highlighting
 * - Reward breakdown bars
 */

// State
let state = {
    runs: [],
    currentRun: null,
    kg: null,
    episodes: [],
    currentEpisode: 0,
    currentPhase: 'base',
    autoplayInterval: null,
    autoplaySpeed: 2000 // ms between episodes
};

// DOM elements
const elements = {
    runSelect: document.getElementById('run-select'),
    phaseSelect: document.getElementById('phase-select'),
    prevBtn: document.getElementById('prev-btn'),
    nextBtn: document.getElementById('next-btn'),
    autoplayBtn: document.getElementById('autoplay-btn'),
    episodeCounter: document.getElementById('episode-counter'),
    graphCanvas: document.getElementById('graph-canvas'),
    questionText: document.getElementById('question-text'),
    optionsList: document.getElementById('options-list'),
    traceText: document.getElementById('trace-text'),
    answerText: document.getElementById('answer-text'),
    correctnessBar: document.getElementById('correctness-bar'),
    correctnessValue: document.getElementById('correctness-value'),
    pathBar: document.getElementById('path-bar'),
    pathValue: document.getElementById('path-value'),
    penaltyBar: document.getElementById('penalty-bar'),
    penaltyValue: document.getElementById('penalty-value'),
    totalBar: document.getElementById('total-bar'),
    totalValue: document.getElementById('total-value'),
    metricAccuracy: document.getElementById('metric-accuracy'),
    metricCoverage: document.getElementById('metric-coverage'),
    metricReward: document.getElementById('metric-reward')
};

// API calls
async function fetchRuns() {
    const res = await fetch('/api/runs');
    const data = await res.json();
    return data.runs || [];
}

async function fetchKG() {
    const res = await fetch('/api/kg');
    return await res.json();
}

async function fetchEpisodes(runId) {
    const res = await fetch(`/api/episodes/${runId}`);
    const data = await res.json();
    return data.episodes || [];
}

// Initialize
async function init() {
    // Load runs
    state.runs = await fetchRuns();
    populateRunSelect();

    // Load KG
    state.kg = await fetchKG();

    // Event listeners
    elements.runSelect.addEventListener('change', onRunChange);
    elements.phaseSelect.addEventListener('change', onPhaseChange);
    elements.prevBtn.addEventListener('click', prevEpisode);
    elements.nextBtn.addEventListener('click', nextEpisode);
    elements.autoplayBtn.addEventListener('click', toggleAutoplay);

    // Initial render
    if (state.runs.length > 0) {
        elements.runSelect.value = state.runs[0].id;
        await onRunChange();
    }
}

function populateRunSelect() {
    state.runs.forEach(run => {
        const option = document.createElement('option');
        option.value = run.id;
        option.textContent = run.id;
        elements.runSelect.appendChild(option);
    });
}

async function onRunChange() {
    const runId = elements.runSelect.value;
    if (!runId) return;

    state.currentRun = state.runs.find(r => r.id === runId);
    state.episodes = await fetchEpisodes(runId);
    state.currentEpisode = 0;

    updateMetrics();
    filterAndRender();
}

function onPhaseChange() {
    state.currentPhase = elements.phaseSelect.value;
    state.currentEpisode = 0;
    filterAndRender();
}

function getFilteredEpisodes() {
    return state.episodes.filter(ep => ep.phase === state.currentPhase);
}

function filterAndRender() {
    const filtered = getFilteredEpisodes();
    if (filtered.length > 0) {
        renderEpisode(filtered[state.currentEpisode]);
    }
    updateCounter();
}

function prevEpisode() {
    const filtered = getFilteredEpisodes();
    if (state.currentEpisode > 0) {
        state.currentEpisode--;
        renderEpisode(filtered[state.currentEpisode]);
        updateCounter();
    }
}

function nextEpisode() {
    const filtered = getFilteredEpisodes();
    if (state.currentEpisode < filtered.length - 1) {
        state.currentEpisode++;
        renderEpisode(filtered[state.currentEpisode]);
        updateCounter();
    }
}

function toggleAutoplay() {
    if (state.autoplayInterval) {
        clearInterval(state.autoplayInterval);
        state.autoplayInterval = null;
        document.body.classList.remove('autoplay-active');
        elements.autoplayBtn.textContent = '▶▶ Auto';
    } else {
        document.body.classList.add('autoplay-active');
        elements.autoplayBtn.textContent = '⏸ Stop';
        state.autoplayInterval = setInterval(() => {
            const filtered = getFilteredEpisodes();
            state.currentEpisode = (state.currentEpisode + 1) % filtered.length;
            renderEpisode(filtered[state.currentEpisode]);
            updateCounter();
        }, state.autoplaySpeed);
    }
}

function updateCounter() {
    const filtered = getFilteredEpisodes();
    elements.episodeCounter.textContent = `${state.currentEpisode + 1}/${filtered.length}`;
}

function updateMetrics() {
    if (!state.currentRun?.metrics?.splits?.eval) return;

    const phase = state.currentPhase;
    const m = state.currentRun.metrics.splits.eval[phase];

    if (m) {
        elements.metricAccuracy.textContent = `${(m.accuracy * 100).toFixed(1)}%`;
        elements.metricCoverage.textContent = `${(m.avg_path_coverage * 100).toFixed(1)}%`;
        elements.metricReward.textContent = m.avg_total_reward.toFixed(2);
    }
}

function renderEpisode(episode) {
    if (!episode) return;

    // Question
    elements.questionText.textContent = extractQuestion(episode.prompt);

    // Options
    renderOptions(episode);

    // Trace with entity highlighting
    renderTrace(episode);

    // Answer
    const isCorrect = episode.reward.correctness > 0;
    elements.answerText.textContent = `ANSWER: ${episode.parsed.answer}`;
    elements.answerText.className = 'answer ' + (isCorrect ? 'correct' : 'incorrect');

    // Reward bars
    renderRewardBars(episode.reward);

    // Graph
    renderGraph(episode);
}

function extractQuestion(prompt) {
    const match = prompt.match(/Question:\s*(.+?)(?=\n[A-D]\)|$)/s);
    return match ? match[1].trim() : prompt;
}

function renderOptions(episode) {
    elements.optionsList.innerHTML = '';

    const optionMatch = episode.prompt.match(/([A-D])\)\s*(.+?)(?=\n[A-D]\)|$)/gs);
    if (!optionMatch) return;

    optionMatch.forEach(opt => {
        const [, letter, text] = opt.match(/([A-D])\)\s*(.+)/s) || [];
        if (!letter) return;

        const div = document.createElement('div');
        div.className = 'option';
        div.textContent = `${letter}) ${text.trim()}`;

        // TODO: Get correct answer from example data
        if (episode.parsed.answer === letter) {
            div.classList.add('selected');
        }
        if (episode.reward.correctness > 0 && episode.parsed.answer === letter) {
            div.classList.add('correct');
        }

        elements.optionsList.appendChild(div);
    });
}

function renderTrace(episode) {
    let trace = episode.parsed.trace_text || '';

    // Highlight entities
    const pathEntities = new Set(episode.parsed.path_entities || []);
    const traceEntities = episode.parsed.trace_entities || [];

    // Sort by length (longest first) to avoid partial matches
    const allEntities = [...new Set([...pathEntities, ...traceEntities])];
    allEntities.sort((a, b) => b.length - a.length);

    allEntities.forEach(entity => {
        const isPath = pathEntities.has(entity);
        const className = isPath ? 'entity path' : 'entity';
        const regex = new RegExp(entity.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g');
        trace = trace.replace(regex, `<span class="${className}">${entity}</span>`);
    });

    elements.traceText.innerHTML = `TRACE: ${trace}`;
}

function renderRewardBars(reward) {
    // Correctness (scale: -2 to +1 -> 0% to 100%)
    const correctPct = ((reward.correctness + 2) / 3) * 100;
    elements.correctnessBar.style.width = `${Math.max(0, correctPct)}%`;
    elements.correctnessBar.classList.toggle('negative', reward.correctness < 0);
    elements.correctnessValue.textContent = reward.correctness.toFixed(1);

    // Path coverage (0 to 1 -> 0% to 100%)
    const pathPct = reward.path_coverage * 100;
    elements.pathBar.style.width = `${pathPct}%`;
    elements.pathValue.textContent = reward.path_coverage.toFixed(2);

    // Penalty (0 to 0.5 -> 0% to 100%)
    const penaltyPct = (reward.spam_penalty / 0.5) * 100;
    elements.penaltyBar.style.width = `${penaltyPct}%`;
    elements.penaltyValue.textContent = reward.spam_penalty.toFixed(2);

    // Total (scale: -2.5 to +1.5 -> 0% to 100%)
    const totalPct = ((reward.total + 2.5) / 4) * 100;
    elements.totalBar.style.width = `${Math.max(0, Math.min(100, totalPct))}%`;
    elements.totalBar.classList.toggle('negative', reward.total < 0);
    elements.totalValue.textContent = reward.total.toFixed(2);
}

function renderGraph(episode) {
    const canvas = elements.graphCanvas;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    // Clear
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!state.kg || !state.kg.entities) return;

    // Get relevant entities for this episode
    const pathEntities = new Set(episode.parsed.path_entities || []);
    const traceEntities = new Set(episode.parsed.trace_entities || []);

    // Simple force-directed layout (placeholder)
    // TODO: Implement proper force simulation
    const nodes = [];
    const nodeMap = {};

    // Add path entities as nodes
    let i = 0;
    pathEntities.forEach(entityId => {
        const entity = state.kg.entities.find(e => e.id === entityId);
        if (entity) {
            const angle = (i / pathEntities.size) * Math.PI * 2;
            const radius = Math.min(canvas.width, canvas.height) * 0.3;
            nodes.push({
                id: entity.id,
                label: entity.label || entity.id,
                x: canvas.width / 2 + Math.cos(angle) * radius,
                y: canvas.height / 2 + Math.sin(angle) * radius,
                isPath: true,
                isMentioned: traceEntities.has(entity.id)
            });
            nodeMap[entity.id] = nodes.length - 1;
        }
        i++;
    });

    // Draw edges
    ctx.strokeStyle = '#ffd700';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);

    const pathList = episode.parsed.path_entities || [];
    for (let i = 0; i < pathList.length - 1; i++) {
        const fromIdx = nodeMap[pathList[i]];
        const toIdx = nodeMap[pathList[i + 1]];
        if (fromIdx !== undefined && toIdx !== undefined) {
            const from = nodes[fromIdx];
            const to = nodes[toIdx];

            ctx.beginPath();
            ctx.moveTo(from.x, from.y);
            ctx.lineTo(to.x, to.y);
            ctx.stroke();

            // Arrow
            const angle = Math.atan2(to.y - from.y, to.x - from.x);
            const arrowLen = 15;
            ctx.beginPath();
            ctx.moveTo(to.x - 25 * Math.cos(angle), to.y - 25 * Math.sin(angle));
            ctx.lineTo(
                to.x - 25 * Math.cos(angle) - arrowLen * Math.cos(angle - 0.3),
                to.y - 25 * Math.sin(angle) - arrowLen * Math.sin(angle - 0.3)
            );
            ctx.moveTo(to.x - 25 * Math.cos(angle), to.y - 25 * Math.sin(angle));
            ctx.lineTo(
                to.x - 25 * Math.cos(angle) - arrowLen * Math.cos(angle + 0.3),
                to.y - 25 * Math.sin(angle) - arrowLen * Math.sin(angle + 0.3)
            );
            ctx.stroke();
        }
    }

    // Draw nodes
    nodes.forEach(node => {
        // Node circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, 20, 0, Math.PI * 2);

        if (node.isMentioned) {
            ctx.fillStyle = '#4ecca3';
        } else {
            ctx.fillStyle = '#555';
        }
        ctx.fill();

        ctx.strokeStyle = node.isPath ? '#ffd700' : '#888';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Label
        ctx.fillStyle = '#eaeaea';
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(node.label, node.x, node.y + 35);
    });
}

// Start
init();
