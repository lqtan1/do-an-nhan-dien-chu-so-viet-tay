const canvas = document.getElementById('digitCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultContainer = document.getElementById('resultContainer');
const predictionResult = document.getElementById('predictionResult');
const confidenceValue = document.getElementById('confidenceValue');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize canvas with white background
function initCanvas() {
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Setting for drawing
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
}

initCanvas();

function getCoords(e) {
    const rect = canvas.getBoundingClientRect();
    let x, y;
    if (e.type.startsWith('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }
    return { x, y };
}

function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getCoords(e);
    [lastX, lastY] = [x, y];
}

function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    
    const { x, y } = getCoords(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
}

// Event Listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch Support
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

// Clear canvas
clearBtn.addEventListener('click', () => {
    // Add a quick flash effect for feedback
    canvas.style.opacity = '0';
    setTimeout(() => {
        initCanvas();
        canvas.style.opacity = '1';
    }, 150);
    
    resultContainer.classList.add('hidden');
});

// Predict digit
predictBtn.addEventListener('click', async () => {
    // Check if canvas is empty (basic check: all white pixels)
    // For now, let's just proceed
    
    predictBtn.innerText = 'Analyzing...';
    predictBtn.disabled = true;
    
    try {
        // Prepare image data
        const imageData = canvas.toDataURL('image/png');
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ image: imageData })
        });
        
        if (!response.ok) throw new Error('API Error');
        
        const result = await response.json();
        
        // Update UI
        predictionResult.innerText = result.digit;
        confidenceValue.innerText = (result.confidence * 100).toFixed(1);
        
        resultContainer.classList.remove('hidden');
        
        // Scroll to result if on mobile
        if (window.innerWidth < 600) {
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        }
        
    } catch (error) {
        alert('Prediction Error: Please ensure the server is running.');
        console.error(error);
    } finally {
        predictBtn.innerText = 'Predict';
        predictBtn.disabled = false;
    }
});
