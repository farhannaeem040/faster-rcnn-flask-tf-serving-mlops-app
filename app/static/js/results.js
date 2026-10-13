/**
 * results.js — Probability Distribution Chart for the Results Page
 *
 * Reads detection data from the #results-data JSON island injected by
 * Jinja2 and renders a Chart.js bar chart showing the confidence score
 * (%) for each object detected during the inference run.
 */
document.addEventListener('DOMContentLoaded', function () {

    // --- Read server-side detections from the JSON data island ---
    const dataEl = document.getElementById('results-data');
    if (!dataEl) return; // Island missing (no detections) — nothing to render

    const detections = JSON.parse(dataEl.textContent);
    if (!detections || detections.length === 0) return;

    // Build labels: "label [index]" differentiates duplicate classes (e.g. two 'person' entries)
    const labels = detections.map((d, i) => `${d.label} [${i + 1}]`);
    const scores = detections.map(d => Math.round(d.score * 1000) / 10); // Precision to 1 d.p.

    // Set Chart.js global defaults to match the app's colour palette
    Chart.defaults.color = '#58677c';
    Chart.defaults.font.family = 'Inter, sans-serif';

    new Chart(document.getElementById('confidenceChart').getContext('2d'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: scores,
                backgroundColor: '#E3006B',                // LJMU Magenta
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,                               // Confidence is always in 0–100% range
                    grid: { color: '#d3dae3' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
});
