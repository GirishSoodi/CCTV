/**
 * Main JavaScript file for CCTV Individual Detection System
 */

// Global variables
let systemStatus = {
    activeCameras: 0,
    detectedIndividuals: 0,
    crossCameraMatches: 0,
    lastUpdated: null
};

// Update system status from server
function updateSystemStatus() {
    fetch('/get_matches')
        .then(response => response.json())
        .then(data => {
            // Count individuals with unique IDs
            const uniqueIds = new Set();
            data.forEach(match => uniqueIds.add(match.id));
            
            // Count cross-camera matches
            const crossMatches = data.filter(match => match.cameras.length > 1).length;
            
            // Update status object
            systemStatus.detectedIndividuals = uniqueIds.size;
            systemStatus.crossCameraMatches = crossMatches;
            systemStatus.lastUpdated = new Date();
            
            // Update UI elements if they exist
            const elemIndividuals = document.getElementById('detectedIndividuals');
            const elemMatches = document.getElementById('crossCameraMatches');
            const elemUpdated = document.getElementById('lastUpdated');
            
            if (elemIndividuals) elemIndividuals.textContent = systemStatus.detectedIndividuals;
            if (elemMatches) elemMatches.textContent = systemStatus.crossCameraMatches;
            if (elemUpdated) elemUpdated.textContent = systemStatus.lastUpdated.toLocaleTimeString();
        })
        .catch(error => {
            console.error('Error updating system status:', error);
        });
}

// Initialize tooltips and popovers if Bootstrap is available
document.addEventListener('DOMContentLoaded', function() {
    // Check if Bootstrap is available
    if (typeof bootstrap !== 'undefined') {
        // Initialize tooltips
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
        
        // Initialize popovers
        const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
        popoverTriggerList.map(function(popoverTriggerEl) {
            return new bootstrap.Popover(popoverTriggerEl);
        });
    }
    
    // Set up auto-refresh for system status if on main page
    if (document.getElementById('detectedIndividuals')) {
        updateSystemStatus();
        setInterval(updateSystemStatus, 5000); // Update every 5 seconds
    }
});
