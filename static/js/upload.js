document.addEventListener('DOMContentLoaded', function () {
    function setLoading(isLoading) {
        const spinner = document.getElementById('loadingSpinner');
        const progressContainer = document.getElementById('progressContainer');
        spinner.style.display = isLoading ? 'block' : 'none';
        progressContainer.style.display = isLoading ? 'block' : 'none';
    }

    function updateProgress(percent) {
        const progressBar = document.getElementById('progressBar');
        progressBar.style.width = percent + '%';
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = percent + '%';
    }

    document.querySelector('form').addEventListener('submit', function (event) {
        event.preventDefault();
        const formData = new FormData(this);

        fetch('/upload', {
            method: 'POST',
            body: formData
        }).then(response => {
            setLoading(false);
            if (response.ok) {
                return response.json();
            } else {
                throw new Error('Upload failed');
            }
        }).then(data => {
            document.getElementById('clusterCount').textContent = data.clusters;
            document.getElementById('modalTitle').value = data.title;
            document.getElementById('modalDescription').value = data.description;
            document.getElementById('modalMethod').value = data.method;
            const modal = new bootstrap.Modal(document.getElementById('exampleModal'));
            modal.show();
        }).catch(error => {
            setLoading(false);
            alert(error.message);
        });

        setLoading(true);
    });

    function toggleSubmitButton() {
        const submitBtn = document.getElementById('submitBtn');
        const filesInput = document.getElementById('files');
        submitBtn.disabled = filesInput.files.length === 0;
    }

    document.getElementById('files').addEventListener('change', toggleSubmitButton);
});
