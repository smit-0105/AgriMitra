document.addEventListener('DOMContentLoaded', () => {

    const API_BASE_URL = 'http://localhost:8000/api';

    const navLinks = document.querySelectorAll('.nav-link');
    const contentSections = document.querySelectorAll('.content-section');

    const diseaseForm = document.getElementById('disease-form');
    const diseaseFileInput = document.getElementById('disease-file-input');
    const imagePreview = document.getElementById('image-preview');

    const fertilizerForm = document.getElementById('fertilizer-form');

    const diseaseResultsContent = document.getElementById('results-content-disease');
    const fertilizerResultsContent = document.getElementById('results-content-fertilizer');

    // --- Navigation Tabs ---
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-target');

            navLinks.forEach(nav => nav.classList.remove('active'));
            contentSections.forEach(sec => sec.classList.remove('active'));

            link.classList.add('active');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // --- Disease Image Preview ---
    diseaseFileInput.addEventListener('change', () => {
        const file = diseaseFileInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = e => {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            imagePreview.src = '';
            imagePreview.style.display = 'none';
        }
    });

    // --- Disease Form ---
    diseaseForm.addEventListener('submit', async e => {
        e.preventDefault();
        const file = diseaseFileInput.files[0];
        if (!file) return showError("Please select an image file.", diseaseResultsContent);

        const formData = new FormData();
        formData.append('file', file);
        showLoading("Analyzing image...", diseaseResultsContent);

        try {
            const response = await fetch(`${API_BASE_URL}/disease/predict`, { method: 'POST', body: formData });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');
            showDiseaseResults(data);
        } catch (error) {
            showError(error.message, diseaseResultsContent);
        }
    });

    // --- Fertilizer Form ---
    fertilizerForm.addEventListener('submit', async e => {
        e.preventDefault();

        const payload = {
            cropType: document.getElementById('crop-type').value,
            pH_Value: parseFloat(document.getElementById('ph-level').value),
            soil_moisture: parseFloat(document.getElementById('moisture').value),
            Temperature: parseFloat(document.getElementById('temperature').value),
            Humidity: parseFloat(document.getElementById('humidity').value),
            Rainfall: parseFloat(document.getElementById('rainfall').value),
        };

        if (Object.values(payload).some(v => v === "" || v === null || isNaN(v))) {
            return showError("Please fill all fields with valid numbers.", fertilizerResultsContent);
        }

        showLoading("Calculating recommendation...", fertilizerResultsContent);

        try {
            const response = await fetch(`${API_BASE_URL}/fertilizer/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Server error');
            showFertilizerResults(data);
        } catch (error) {
            showError(error.message, fertilizerResultsContent);
        }
    });

    // --- Helper Functions ---
    function showLoading(message, target) { target.innerHTML = `<p>${message}</p>`; }
    function showError(message, target) { target.innerHTML = `<p class="error">Error: ${message}</p>`; }

    function showDiseaseResults(data) {
        diseaseResultsContent.innerHTML = `
            <p><strong>Detected:</strong> ${data.disease_detected}</p>
            <p><strong>Confidence:</strong> ${data.confidence}</p>
            <p><strong>Remedy:</strong> ${data.remedy}</p>
        `;
    }

    function showFertilizerResults(data) {
        fertilizerResultsContent.innerHTML = `
            <p><strong>Nitrogen (N):</strong> ${data.N_recommendation_kg_ha} kg/ha</p>
            <p><strong>Phosphorus (P):</strong> ${data.P_recommendation_kg_ha} kg/ha</p>
            <p><strong>Potassium (K):</strong> ${data.K_recommendation_kg_ha} kg/ha</p>
        `;
    }
});
