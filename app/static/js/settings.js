/**
 * settings.js — Model Upload UI for the Model Management Console
 *
 * Responsibilities:
 *   - File input change handler: updates the upload area icon and
 *     displays the selected archive filename.
 *   - showLoading(): validates file selection, swaps button label,
 *     shows spinner and progress bar, and disables the button to
 *     prevent duplicate form submissions during the upload.
 */

/**
 * showLoading — validates that a file has been selected, then swaps
 * the button label and reveals the progress bar to prevent duplicate
 * submissions and reassure the user the upload is in progress.
 *
 * Called via onsubmit on the model upload form.
 *
 * @returns {boolean|undefined} true on success, undefined if no file selected.
 */
function showLoading() {
    const fileInput = document.getElementById('model_zip');

    // Abort early if no file is attached; let native validation handle the message
    if (!fileInput.files || fileInput.files.length === 0) return;

    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('loading-spinner');
    const submitBtn = document.getElementById('submit-btn');
    const container = document.getElementById('loading-container');

    // Update button copy to signal processing has started
    if (btnText) btnText.textContent = 'Decoupling Archive...';
    // Show the inline spinner next to the button label
    if (spinner) spinner.classList.remove('hidden');
    // Disable submit button to prevent duplicate form submissions
    if (submitBtn) submitBtn.classList.add('btn-disabled');
    // Reveal the full progress bar and status message
    if (container) container.classList.remove('hidden');

    return true;
}

document.addEventListener('DOMContentLoaded', function () {
    const fileInput = document.getElementById('model_zip');
    const display = document.getElementById('zip-file-name-display');
    const textLabel = document.getElementById('zip-upload-text');
    const uploadIcon = document.getElementById('upload-icon');

    if (!fileInput) return;

    // Update the upload area UI when the user picks a file
    fileInput.addEventListener('change', function () {
        if (this.files && this.files.length > 0) {
            // Show the selected archive name and swap icon to a checkmark
            display.textContent = 'Archive Attached: ' + this.files[0].name;
            textLabel.textContent = 'Archive Ready';
            uploadIcon.textContent = '✅';
            uploadIcon.classList.remove('text-color-primary');
            uploadIcon.classList.add('text-success');
        } else {
            // Reset prompt text and icon if selection is cleared
            display.textContent = '';
            textLabel.textContent = 'Select Compiled Framework (.zip)';
            uploadIcon.textContent = '📦';
            uploadIcon.classList.remove('text-success');
            uploadIcon.classList.add('text-color-primary');
        }
    });
});
