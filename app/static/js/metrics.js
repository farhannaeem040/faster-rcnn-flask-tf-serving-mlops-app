/**
 * metrics.js — Chart.js Rendering for the Platform Analytics Dashboard
 *
 * Reads structured metrics data from the #metrics-data JSON island that
 * is injected into metrics.html by Jinja2 at render time, then builds
 * three Chart.js visualisations:
 *   1. Inference Latency Trend  — line chart
 *   2. Targets per Inference     — bar chart
 *   3. Success vs Failure        — doughnut chart
 */
document.addEventListener('DOMContentLoaded', function () {

    // --- Read server-side data from the JSON data island ---
    const dataEl = document.getElementById('metrics-data');
    if (!dataEl) return; // Island missing — nothing to render

    const metrics = JSON.parse(dataEl.textContent);
    const historyData = metrics.history;
    const successCount = metrics.successful_requests;
    const failCount = metrics.failed_requests;

    // Guard: nothing to render if history is empty
    if (!historyData || historyData.length === 0) return;

    // --- Theme overrides: align Chart.js with the Inter typography and LJMU palette ---
    Chart.defaults.color = '#58677c';
    Chart.defaults.font.family = 'Inter, sans-serif';

    // Build sequential labels and extract data arrays from history
    const labels = historyData.map((item, i) => 'Req ' + (historyData.length - i));
    const latencies = historyData.map(item => item.latency_ms);
    const detections = historyData.map(item => item.detection_count);

    // -----------------------------------------------------------------
    // Chart 1 — Inference Latency Trend (line)
    // -----------------------------------------------------------------
    const ctxLatency = document.getElementById('latencyChart').getContext('2d');
    new Chart(ctxLatency, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Processing Time (ms)',
                data: latencies,
                borderColor: '#E3006B',                    // LJMU Magenta
                backgroundColor: 'rgba(227, 0, 107, 0.1)',
                borderWidth: 3,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#E3006B',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: '#e2e8f0' } },
                x: { grid: { display: false } }
            }
        }
    });

    // -----------------------------------------------------------------
    // Chart 2 — Detected Targets per Inference (bar)
    // -----------------------------------------------------------------
    const ctxDetections = document.getElementById('detectionsChart').getContext('2d');
    new Chart(ctxDetections, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Detected Items',
                data: detections,
                backgroundColor: '#041E42',                // LJMU Deep Blue
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: '#e2e8f0' } },
                x: { grid: { display: false } }
            }
        }
    });

    // -----------------------------------------------------------------
    // Chart 3 — Success vs Failure Distribution (doughnut)
    // Only rendered when at least one request has been recorded
    // -----------------------------------------------------------------
    if (successCount > 0 || failCount > 0) {
        const ctxSuccess = document.getElementById('successChart').getContext('2d');
        new Chart(ctxSuccess, {
            type: 'doughnut',
            data: {
                labels: ['Operational', 'Failed / Errors'],
                datasets: [{
                    data: [successCount, failCount],
                    backgroundColor: ['#00937F', '#dc2626'], // Liver Green / Red
                    borderWidth: 2,
                    borderColor: '#ffffff',
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '75%',                             // Hollow-centre doughnut style
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }
});
