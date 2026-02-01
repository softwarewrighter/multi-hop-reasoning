/**
 * Multi-Hop Reasoning Demo
 * Training: Graph actively scores outputs
 * Inference: Graph removed, model reasons alone
 */

const TOTAL_STEPS = 50; // Reduced for demo speed

const state = {
    kg: null,
    episodes: [],
    trainingData: [],
    testData: [],
    isRunning: false,
    trainStep: 0,
    accuracyHistory: [],
    testIndex: 0,
    testCorrect: 0,
    testWrong: 0,
    speed: 1,
    currentPhase: 'sft' // 'sft' or 'rsft'
};

const el = {
    trainingTab: document.getElementById('training-tab'),
    inferenceTab: document.getElementById('inference-tab'),
    trainingView: document.getElementById('training-view'),
    inferenceView: document.getElementById('inference-view'),
    speedSlider: document.getElementById('speed-slider'),
    speedLabel: document.getElementById('speed-label'),
    // Training
    graphContainer: document.getElementById('graph-container'),
    trainStep: document.getElementById('train-step'),
    trainQuestion: document.getElementById('train-question'),
    trainOutput: document.getElementById('train-output'),
    trainAnswer: document.getElementById('train-answer'),
    rewardCorrect: document.getElementById('reward-correct'),
    rewardPath: document.getElementById('reward-path'),
    rewardTotal: document.getElementById('reward-total'),
    rewardDecision: document.getElementById('reward-decision'),
    progressCanvas: document.getElementById('progress-canvas'),
    currentAccuracy: document.getElementById('current-accuracy'),
    examplesSeen: document.getElementById('examples-seen'),
    startTrainingBtn: document.getElementById('start-training-btn'),
    skipBtn: document.getElementById('skip-btn'),
    phaseSft: document.getElementById('phase-sft'),
    phaseRsft: document.getElementById('phase-rsft'),
    phaseBase: document.getElementById('phase-base'),
    // Inference
    testNum: document.getElementById('test-num'),
    testQuestion: document.getElementById('test-question'),
    testOptions: document.getElementById('test-options'),
    testTrace: document.getElementById('test-trace'),
    testAnswer: document.getElementById('test-answer'),
    correctCount: document.getElementById('correct-count'),
    wrongCount: document.getElementById('wrong-count'),
    testAccuracy: document.getElementById('test-accuracy'),
    testProgressBar: document.getElementById('test-progress-bar'),
    runTestBtn: document.getElementById('run-test-btn'),
    skipTestBtn: document.getElementById('skip-test-btn')
};

// ===== INIT =====
async function init() {
    try {
        const [kgRes, epRes] = await Promise.all([
            fetch('/multi-hop-reasoning/kg.json'),
            fetch('/multi-hop-reasoning/episodes.json')
        ]);
        state.kg = await kgRes.json();
        const data = await epRes.json();
        state.episodes = data.episodes || [];
        state.trainingData = state.episodes.filter(e => e.phase === 'sft');
        state.testData = state.episodes.filter(e => e.phase === 'rsft');
    } catch (e) {
        console.error('Failed to load:', e);
    }

    el.trainingTab.addEventListener('click', () => switchView('training'));
    el.inferenceTab.addEventListener('click', () => switchView('inference'));
    el.startTrainingBtn.addEventListener('click', toggleTraining);
    el.skipBtn.addEventListener('click', skipToEnd);
    el.runTestBtn.addEventListener('click', toggleTest);
    el.skipTestBtn.addEventListener('click', skipTest);
    el.speedSlider.addEventListener('input', updateSpeed);

    updateSpeed();
    drawEmptyProgress();
}

function switchView(view) {
    el.trainingTab.classList.toggle('active', view === 'training');
    el.inferenceTab.classList.toggle('active', view === 'inference');
    el.trainingView.classList.toggle('active', view === 'training');
    el.inferenceView.classList.toggle('active', view === 'inference');
}

function updateSpeed() {
    const val = parseInt(el.speedSlider.value);
    state.speed = val;
    el.speedLabel.textContent = val + 'x';
}

function getDelay(base) {
    return Math.max(10, base / state.speed);
}

// ===== TRAINING =====
function toggleTraining() {
    if (state.isRunning) {
        stopTraining();
    } else {
        startTraining();
    }
}

function startTraining() {
    state.isRunning = true;
    state.trainStep = 0;
    state.accuracyHistory = [{step: 0, acc: 0}];
    state.currentPhase = 'sft';

    el.startTrainingBtn.textContent = '⏹ Stop';
    el.startTrainingBtn.classList.add('running');
    el.phaseSft.classList.add('active');
    el.phaseRsft.classList.remove('active');
    // Base stays skipped (not complete) - we start from SFT

    runTrainingStep();
}

function stopTraining() {
    state.isRunning = false;
    el.startTrainingBtn.textContent = '▶ Start';
    el.startTrainingBtn.classList.remove('running');
}

function skipToEnd() {
    state.isRunning = false;

    // Jump to end state
    state.trainStep = TOTAL_STEPS;
    state.accuracyHistory = [
        {step: 0, acc: 0},
        {step: 25, acc: 30},
        {step: 50, acc: 75}
    ];

    el.currentAccuracy.textContent = '75%';
    el.examplesSeen.textContent = TOTAL_STEPS;
    el.trainStep.textContent = `Step ${TOTAL_STEPS} / ${TOTAL_STEPS}`;

    el.phaseSft.classList.remove('active');
    el.phaseSft.classList.add('complete');
    el.phaseRsft.classList.add('active');
    // Base stays skipped - we don't train from scratch

    drawProgress();
    stopTraining();
}

async function runTrainingStep() {
    if (!state.isRunning) return;

    const ep = state.trainingData[state.trainStep % state.trainingData.length];
    state.trainStep++;

    // Update phase at midpoint
    if (state.trainStep === Math.floor(TOTAL_STEPS / 2)) {
        state.currentPhase = 'rsft';
        el.phaseSft.classList.remove('active');
        el.phaseSft.classList.add('complete');
        el.phaseRsft.classList.add('active');
    }

    el.trainStep.textContent = `Step ${state.trainStep} / ${TOTAL_STEPS}`;

    // 1. Show question
    el.trainQuestion.textContent = extractQuestion(ep.prompt);
    el.trainOutput.innerHTML = '';
    el.trainAnswer.textContent = '';
    el.trainAnswer.className = 'answer-badge';
    clearRewardSteps();

    // 2. Render graph (vertical DOM nodes)
    renderGraph(ep.parsed.path_entities || []);

    // Exact timing at 1x speed: text=200ms, choice=1500ms, evaluation=4500ms

    // 3. Typewriter with node highlighting (fast)
    const trace = ep.parsed.trace_text || 'Analyzing...';
    const traceEntities = ep.parsed.trace_entities || [];
    await typewriterWithHighlight(el.trainOutput, trace, traceEntities, getDelay(5));

    await sleep(getDelay(200)); // 200ms after text

    // 4. Show answer (1.5 seconds to read)
    const isCorrect = ep.reward.correctness > 0;
    el.trainAnswer.textContent = `ANSWER: ${ep.parsed.answer}`;
    el.trainAnswer.classList.add(isCorrect ? 'correct' : 'incorrect');

    await sleep(getDelay(1500)); // 1.5s for choice

    // 5. Reward calculation (4.5 seconds total = 1125ms per step)
    animateReward(1, isCorrect ? '+1.0 ✓' : '-2.0 ✗', isCorrect);
    await sleep(getDelay(1125));

    animateReward(2, `+${ep.reward.path_coverage.toFixed(2)}`, true);
    await sleep(getDelay(1125));

    const total = ep.reward.total;
    animateReward(3, total >= 0 ? `+${total.toFixed(2)}` : total.toFixed(2), total >= 0);
    await sleep(getDelay(1125));

    // Decision
    if (total > 0) {
        el.rewardDecision.textContent = '✓ KEEP for training';
        el.rewardDecision.className = 'reward-decision keep';
    } else {
        el.rewardDecision.textContent = '✗ DISCARD';
        el.rewardDecision.className = 'reward-decision discard';
    }

    await sleep(getDelay(1125)); // 4th step of evaluation

    // 6. Update progress
    const simAcc = state.currentPhase === 'sft'
        ? Math.min(30, state.trainStep * 1.2)
        : Math.min(75, 30 + (state.trainStep - TOTAL_STEPS/2) * 1.8);

    state.accuracyHistory.push({step: state.trainStep, acc: simAcc});
    el.currentAccuracy.textContent = `${Math.round(simAcc)}%`;
    el.examplesSeen.textContent = state.trainStep;
    drawProgress();

    await sleep(getDelay(300));

    // Continue or stop
    if (state.isRunning && state.trainStep < TOTAL_STEPS) {
        runTrainingStep();
    } else if (state.trainStep >= TOTAL_STEPS) {
        el.phaseRsft.classList.add('complete');
        stopTraining();
    }
}

function renderGraph(pathEntities) {
    el.graphContainer.innerHTML = '';

    if (!pathEntities || pathEntities.length === 0) {
        el.graphContainer.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 20px;">Waiting for example...</div>';
        return;
    }

    pathEntities.forEach((id, i) => {
        const entity = state.kg?.entities?.find(e => e.id === id);
        const label = entity?.label || id.replace(/([A-Z])/g, ' $1').trim();

        const nodeDiv = document.createElement('div');
        nodeDiv.className = 'graph-node highlighted';
        nodeDiv.id = `node-${id}`;
        nodeDiv.innerHTML = `
            <div class="node-circle">${i + 1}</div>
            <div class="node-label">${label}</div>
        `;
        el.graphContainer.appendChild(nodeDiv);

        // Arrow (except last)
        if (i < pathEntities.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'node-arrow';
            arrow.textContent = '↓';
            el.graphContainer.appendChild(arrow);
        }
    });
}

function highlightNode(entityId) {
    const node = document.getElementById(`node-${entityId}`);
    if (node) {
        node.classList.add('active');
        setTimeout(() => node.classList.remove('active'), 800);
    }
}

async function typewriterWithHighlight(element, text, entities, charDelay) {
    element.innerHTML = '';
    let displayed = '';

    for (let i = 0; i < text.length; i++) {
        displayed = text.substring(0, i + 1);
        element.innerHTML = displayed + '<span class="cursor">|</span>';

        // Check if we just completed typing an entity name
        for (const ent of entities) {
            if (displayed.endsWith(ent)) {
                highlightNode(ent);
            }
        }

        await sleep(charDelay);
        if (!state.isRunning) break;
    }
    element.innerHTML = text;
}

function clearRewardSteps() {
    for (let i = 1; i <= 3; i++) {
        const step = document.getElementById(`reward-step-${i}`);
        step.classList.remove('active');
        step.querySelector('.step-value').textContent = '--';
        step.querySelector('.step-value').className = 'step-value';
    }
    el.rewardDecision.textContent = '';
    el.rewardDecision.className = 'reward-decision';
}

function animateReward(num, value, positive) {
    const step = document.getElementById(`reward-step-${num}`);
    const val = step.querySelector('.step-value');
    step.classList.add('active');
    val.textContent = value;
    val.classList.add(positive ? 'positive' : 'negative');
}

// ===== INFERENCE =====
function toggleTest() {
    if (state.isRunning) {
        stopTest();
    } else {
        startTest();
    }
}

function startTest() {
    state.isRunning = true;
    state.testIndex = 0;
    state.testCorrect = 0;
    state.testWrong = 0;

    el.runTestBtn.textContent = '⏹ Stop';
    el.runTestBtn.classList.add('running');
    el.testProgressBar.style.width = '0%';

    runTestStep();
}

function stopTest() {
    state.isRunning = false;
    el.runTestBtn.textContent = '▶ Run Test';
    el.runTestBtn.classList.remove('running');
}

function skipTest() {
    state.isRunning = false;

    // Show final results
    const total = state.testData.length;
    const correct = Math.round(total * 0.75);
    const wrong = total - correct;

    el.correctCount.textContent = correct;
    el.wrongCount.textContent = wrong;
    el.testAccuracy.textContent = '75%';
    el.testProgressBar.style.width = '100%';
    el.testNum.textContent = `${total} / ${total}`;

    stopTest();
}

async function runTestStep() {
    if (!state.isRunning || state.testIndex >= state.testData.length) {
        stopTest();
        return;
    }

    const ep = state.testData[state.testIndex];
    state.testIndex++;

    el.testNum.textContent = `${state.testIndex} / ${state.testData.length}`;
    el.testProgressBar.style.width = `${(state.testIndex / state.testData.length) * 100}%`;

    // Question
    el.testQuestion.textContent = extractQuestion(ep.prompt);
    el.testOptions.innerHTML = '';
    el.testTrace.innerHTML = '';
    el.testAnswer.textContent = '';
    el.testAnswer.className = 'answer-display';

    const options = parseOptions(ep.prompt);
    options.forEach(opt => {
        const div = document.createElement('div');
        div.className = 'option-item';
        div.id = `opt-${opt.letter}`;
        div.textContent = `${opt.letter}) ${opt.text}`;
        el.testOptions.appendChild(div);
    });

    // Timing: text=200ms, choice=1500ms, result=3000ms

    // Typewriter (fast, 5ms per char at 1x)
    const trace = ep.parsed.trace_text || 'Thinking...';
    await typewriter(el.testTrace, trace, getDelay(5));

    await sleep(getDelay(200)); // 200ms after text

    // Answer (1.5 seconds to read the choice)
    const answer = ep.parsed.answer;
    const isCorrect = ep.reward.correctness > 0;

    el.testAnswer.textContent = `ANSWER: ${answer}`;
    el.testAnswer.classList.add(isCorrect ? 'correct' : 'incorrect');

    await sleep(getDelay(1500)); // 1.5s for choice

    // Highlight correct/wrong option (3 seconds to see result)
    const selected = document.getElementById(`opt-${answer}`);
    if (selected) {
        selected.classList.add('selected');
        if (isCorrect) selected.classList.add('correct');
        else selected.classList.add('wrong');
    }

    if (isCorrect) state.testCorrect++;
    else state.testWrong++;

    el.correctCount.textContent = state.testCorrect;
    el.wrongCount.textContent = state.testWrong;

    const acc = ((state.testCorrect / state.testIndex) * 100).toFixed(0);
    el.testAccuracy.textContent = `${acc}%`;

    await sleep(getDelay(3000)); // 3s to see green/red result

    if (state.isRunning) runTestStep();
}

// ===== PROGRESS CHART =====
function drawEmptyProgress() {
    const canvas = el.progressCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawProgress() {
    const canvas = el.progressCanvas;
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const p = {left: 30, right: 10, top: 5, bottom: 15};
    const w = canvas.width - p.left - p.right;
    const h = canvas.height - p.top - p.bottom;

    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Axes
    ctx.strokeStyle = '#30363d';
    ctx.beginPath();
    ctx.moveTo(p.left, p.top);
    ctx.lineTo(p.left, p.top + h);
    ctx.lineTo(p.left + w, p.top + h);
    ctx.stroke();

    // Labels
    ctx.fillStyle = '#7d8590';
    ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('75%', p.left - 4, p.top + 5);
    ctx.fillText('0%', p.left - 4, p.top + h);

    if (state.accuracyHistory.length < 2) return;

    // Line
    ctx.strokeStyle = '#3fb950';
    ctx.lineWidth = 2;
    ctx.beginPath();

    state.accuracyHistory.forEach((pt, i) => {
        const x = p.left + (pt.step / TOTAL_STEPS) * w;
        const y = p.top + h - (pt.acc / 80) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Current point
    const last = state.accuracyHistory[state.accuracyHistory.length - 1];
    ctx.beginPath();
    ctx.arc(p.left + (last.step / TOTAL_STEPS) * w, p.top + h - (last.acc / 80) * h, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#3fb950';
    ctx.fill();
}

// ===== UTILS =====
function extractQuestion(prompt) {
    const m = prompt.match(/Question:\s*(.+?)(?=\n[A-D]\)|$)/s);
    return m ? m[1].trim() : 'Loading...';
}

function parseOptions(prompt) {
    const opts = [];
    const re = /([A-D])\)\s*(.+?)(?=\n[A-D]\)|$)/gs;
    let m;
    while ((m = re.exec(prompt)) !== null) {
        opts.push({letter: m[1], text: m[2].trim()});
    }
    return opts;
}

async function typewriter(el, text, delay) {
    el.innerHTML = '';
    for (let i = 0; i < text.length; i++) {
        el.innerHTML = text.substring(0, i + 1) + '<span class="cursor">|</span>';
        await sleep(delay);
        if (!state.isRunning) break;
    }
    el.innerHTML = text;
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

init();
