// ── Canvas Drawing Setup ──
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Initialize canvas with black background
function clearCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('predicted-digit').textContent = '?';
    resetChart();
}

// Drawing settings
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#fff';
clearCanvas();

// ── Mouse Events ──
canvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    ctx.beginPath();
    const pos = getMousePos(e);
    ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
});

canvas.addEventListener('mouseup', () => { isDrawing = false; });
canvas.addEventListener('mouseleave', () => { isDrawing = false; });

// ── Touch Events (mobile) ──
canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    ctx.beginPath();
    const pos = getTouchPos(e);
    ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const pos = getTouchPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
});

canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    isDrawing = false;
});

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (touch.clientX - rect.left) * scaleX,
        y: (touch.clientY - rect.top) * scaleY
    };
}

// ── Button Handlers ──
document.getElementById('btn-clear').addEventListener('click', clearCanvas);
document.getElementById('btn-check').addEventListener('click', checkDigit);

async function checkDigit() {
    const btn = document.getElementById('btn-check');
    btn.classList.add('loading');
    btn.textContent = '...';

    try {
        const imageData = canvas.toDataURL('image/png');
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
        });

        const result = await response.json();

        if (result.error) {
            document.getElementById('predicted-digit').textContent = '!';
            console.error(result.error);
        } else {
            document.getElementById('predicted-digit').textContent = result.digit;
            updateChart(result.probabilities, result.digit);
        }
    } catch (err) {
        console.error('Prediction failed:', err);
        document.getElementById('predicted-digit').textContent = '!';
    } finally {
        btn.classList.remove('loading');
        btn.textContent = 'CHECK';
    }
}

// ── Bar Chart ──
function initChart() {
    const barsContainer = document.getElementById('chart-bars');
    const labelsContainer = document.getElementById('chart-labels');

    for (let i = 0; i < 10; i++) {
        // Percentage label
        const label = document.createElement('span');
        label.id = `label-${i}`;
        label.textContent = '0.0%';
        labelsContainer.appendChild(label);

        // Bar wrapper + bar
        const wrapper = document.createElement('div');
        wrapper.className = 'bar-wrapper';
        const bar = document.createElement('div');
        bar.className = 'bar';
        bar.id = `bar-${i}`;
        bar.style.height = '3px';
        wrapper.appendChild(bar);
        barsContainer.appendChild(wrapper);
    }
}

function updateChart(probabilities, winnerDigit) {
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`bar-${i}`);
        const label = document.getElementById(`label-${i}`);
        const prob = probabilities[i];

        // Height: min 3px, max 100% of chart height
        const height = Math.max(2, prob) + '%';
        bar.style.height = height;

        // Highlight winner
        if (i === winnerDigit) {
            bar.classList.add('active');
        } else {
            bar.classList.remove('active');
        }

        label.textContent = prob.toFixed(1) + '%';
    }
}

function resetChart() {
    for (let i = 0; i < 10; i++) {
        const bar = document.getElementById(`bar-${i}`);
        const label = document.getElementById(`label-${i}`);
        if (bar) {
            bar.style.height = '3px';
            bar.classList.remove('active');
        }
        if (label) label.textContent = '0.0%';
    }
}

// ── Twinkling Stars ──
function generateStars() {
    const container = document.getElementById('stars-container');
    const count = 55;

    for (let i = 0; i < count; i++) {
        const star = document.createElement('div');
        star.className = 'star';

        const size = Math.random() * 2 + 1;
        const x = Math.random() * 100;
        const y = Math.random() * 100;
        const duration = Math.random() * 3 + 2;
        const delay = Math.random() * 5;
        const maxOpacity = Math.random() * 0.5 + 0.2;

        star.style.width = size + 'px';
        star.style.height = size + 'px';
        star.style.left = x + '%';
        star.style.top = y + '%';
        star.style.setProperty('--duration', duration + 's');
        star.style.setProperty('--delay', delay + 's');
        star.style.setProperty('--max-opacity', maxOpacity);

        container.appendChild(star);
    }
}

// ── Initialize ──
initChart();
generateStars();
