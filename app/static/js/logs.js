/**
 * logs.js — Server Application Logs Page Interactions
 *
 * Responsibilities:
 *   - Auto-scrolling: Scrolls the terminal-style log container to the bottom
 *     on initial page load so the most recent log entries are visible.
 */
document.addEventListener('DOMContentLoaded', function () {
    // Target the scrollable terminal container by ID
    const container = document.getElementById('logs-container');

    if (container) {
        // Set scroll position to the bottom of the container
        container.scrollTop = container.scrollHeight;
    }
});
