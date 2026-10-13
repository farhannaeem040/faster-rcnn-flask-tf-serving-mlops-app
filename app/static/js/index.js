/**
 * index.js — Upload Form Interaction & Live System Pulse Sparkline
 *
 * Responsibilities:
 *   - File input change handler: renders an inline image preview and
 *     displays the selected filename beneath the upload area.
 *   - Form submit handler: activates the progress bar and disables the
 *     submit button to prevent duplicate inference submissions.
 *   - Pulse sparkline: reads latency history from the #pulse-data JSON
 *     island and renders a compact Chart.js line chart.
 */
document.addEventListener('DOMContentLoaded', function () {

    // --- Wire up file-selection listener ---
    const fileInput = document.getElementById('image');
    if (fileInput) {
        fileInput.addEventListener('change', function () {
            updateFileName(this);
        });
    }

    // --- Wire up form submit listener ---
    const form = document.getElementById('upload-form');
    if (form) {
        form.addEventListener('submit', function () {
            showLoading();
        });
    }

    // --- Render the Pulse Sparkline ---
    // historyData is read from the JSON data island injected by Jinja2
    const pulseDataEl = document.getElementById('pulse-data');
    if (!pulseDataEl) return;

    const historyData = JSON.parse(pulseDataEl.textContent);
    if (!historyData || historyData.length === 0) return;

    // Extract latency values and build reverse-chronological labels
    const latencies = historyData.map(item => item.latency_ms);
    // T - 3, T - 2, T - 1, T - 0
    // T - 3 is the oldest, T - 0 is the most recent
    const labels = historyData.map((_, i) => `T - ${historyData.length - i}`);

    // Set Chart.js global defaults to match the app's colour palette
    Chart.defaults.color = '#58677c';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // Create the pulse sparkline
    // It requires at least one data point to render
    new Chart(document.getElementById('pulseChart').getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Latency (ms) Pulse',
                data: latencies,
                borderColor: '#00937F',                    // Liver Green
                backgroundColor: 'rgba(0, 147, 127, 0.1)',
                borderWidth: 2,
                pointRadius: 0,                            // Hidden points — cleaner sparkline look
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                y: { display: false, min: 0 },
                x: { display: false }
            },
            interaction: { mode: null }
        }
    });
});

/**
 * updateFileName — reads the selected file via FileReader and
 * replaces the upload prompt UI with an inline image preview.
 *
 * @param {HTMLInputElement} input - The file input element.
 */
function updateFileName(input) {
    const display = document.getElementById('file-name-display');
    const preview = document.getElementById('image-preview');
    const uploadIcon = document.getElementById('upload-icon');
    const uploadText = document.getElementById('upload-text');
    const uploadSubtext = document.getElementById('upload-subtext');

    if (input.files && input.files[0]) {
        const file = input.files[0];
        display.textContent = 'Selected: ' + file.name;

        // Use FileReader to render a base64 preview without a server round-trip
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.src = e.target.result;
            preview.classList.remove('hidden');
            uploadIcon.classList.add('hidden');
            uploadText.classList.add('hidden');
            uploadSubtext.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    } else {
        // Reset the upload area back to its default prompt state
        display.textContent = '';
        preview.classList.add('hidden');
        preview.src = '';
        uploadIcon.classList.remove('hidden');
        uploadText.classList.remove('hidden');
        uploadSubtext.classList.remove('hidden');
    }
}

/**
 * showLoading — validates file selection then activates the progress
 * bar and disables the submit button to prevent duplicate submissions.
 *
 * @returns {boolean|undefined} true on success, undefined if no file selected.
 */
function showLoading() {
    const fileInput = document.getElementById('image');

    // Basic validation: allow native HTML5 validation to block if no file
    if (!fileInput.files || fileInput.files.length === 0) return;

    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('loading-spinner');
    const submitBtn = document.getElementById('submit-btn');
    const container = document.getElementById('loading-container');

    if (btnText) btnText.textContent = 'Processing...';
    if (spinner) spinner.classList.remove('hidden');
    if (submitBtn) submitBtn.classList.add('btn-disabled');
    if (container) container.classList.remove('hidden');

    return true;
}
