try {
    var form = document.getElementById('analysisForm');
    var submitBtn = document.getElementById('submitBtn');
    var loader = document.getElementById('loader');
    var resultsSection = document.getElementById('resultsSection');
    var welcomeSection = document.getElementById('welcomeSection');
    var errorSection = document.getElementById('errorSection');
    var resultTabsContent = document.getElementById('resultTabsContent');
    var loadingMessages = [
        "Préparation des données...",
        "Analyse des caractéristiques...",
        "Entraînement des modèles...",
        "Évaluation des performances...",
        "Génération des visualisations...",
        "Finalisation des résultats..."
    ];
    var messageIndex = 0;

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        loader.classList.add('show');
        submitBtn.disabled = true;
        welcomeSection.style.display = 'none';
        errorSection.style.display = 'none';
        resultsSection.style.display = 'none';

        var loadingMessage = document.getElementById('loadingMessage');
        function updateLoadingMessage() {
            loadingMessage.textContent = loadingMessages[messageIndex];
            messageIndex = (messageIndex + 1) % loadingMessages.length;
        }
        updateLoadingMessage();
        var messageInterval = setInterval(updateLoadingMessage, 3000);

        var formData = new FormData(form);
        fetch('/machine_interface', {
            method: 'POST',
            body: formData
        })
        .then(function(response) {
            clearInterval(messageInterval);
            loader.classList.remove('show');
            submitBtn.disabled = false;

            if (!response.ok) {
                return response.text().then(function(text) {
                    throw new Error('Erreur HTTP ' + response.status + ': ' + text);
                });
            }
            return response.json();
        })
        .then(function(result) {
            if (result.error) {
                errorSection.textContent = 'Erreur: ' + result.error;
                errorSection.style.display = 'block';
                return;
            }
            resultsSection.style.display = 'block';
            renderResults(result);
        })
        .catch(function(error) {
            clearInterval(messageInterval);
            loader.classList.remove('show');
            submitBtn.disabled = false;
            errorSection.textContent = 'Erreur: ' + error.message;
            errorSection.style.display = 'block';
            console.error('Erreur lors de la soumission:', error);
        });
    });

    function renderResults(result) {
        // Summary Tab
        var summaryContent = [
            '<div class="tab-pane fade show active" id="summary" role="tabpanel" aria-labelledby="summary-tab">',
            '<div class="row">',
            '<div class="col-md-6">',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h5 class="card-title"><i class="fas fa-database me-2"></i>Informations du dataset</h5>',
            '<ul class="list-group list-group-flush">',
            '<li class="list-group-item d-flex justify-content-between align-items-center">Nombre de lignes<span class="badge bg-primary rounded-pill">' + result.n_rows + '</span></li>',
            '<li class="list-group-item d-flex justify-content-between align-items-center">Nombre de colonnes<span class="badge bg-primary rounded-pill">' + result.n_cols + '</span></li>',
            '<li class="list-group-item d-flex justify-content-between align-items-center">Type d\'analyse<span class="badge bg-secondary rounded-pill">' + result.task + '</span></li>',
            (result.target ? '<li class="list-group-item d-flex justify-content-between align-items-center">Colonne cible<span class="badge bg-info rounded-pill">' + result.target + '</span></li>' : ''),
            '</ul>',
            '</div>',
            '</div>',
            '</div>',
            '<div class="col-md-6">',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h5 class="card-title"><i class="fas fa-trophy me-2"></i>Meilleur modèle</h5>',
            '<div class="text-center my-3">',
            '<div class="display-6 fw-bold text-primary">' + result.best_model + '</div>',
            '<div class="mt-2">Score: <span class="fw-bold">' + parseFloat(result.models[result.best_model]).toFixed(4) + '</span></div>',
            '</div>',
            '<div class="d-grid gap-2 mt-3">',
            '<a href="/static/' + result.model_path + '" class="btn btn-primary"><i class="fas fa-download me-1"></i> Télécharger le modèle</a>',
            '</div>',
            '</div>',
            '</div>',
            '</div>',
            '</div>',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h5 class="card-title"><i class="fas fa-chart-bar me-2"></i>Comparaison des modèles</h5>',
            '<div class="chart-container" id="modelsComparisonChart"></div>',
            '<div class="mt-3">'
        ];
        for (var name in result.models) {
            if (result.models.hasOwnProperty(name)) {
                summaryContent.push(
                    '<div class="model-score-card ' + (name === result.best_model ? 'best-model-card' : '') + '">' +
                    '<span class="model-name">' + name + '</span>' +
                    '<span class="model-score">' + parseFloat(result.models[name]).toFixed(4) + '</span>' +
                    '</div>'
                );
            }
        }
        summaryContent.push(
            '</div>',
            '</div>',
            '</div>',
            '<div class="mt-4">',
            '<h5><i class="fas fa-file-download me-2"></i>Téléchargements</h5>',
            '<div class="mt-3">',
            '<a href="/static/' + result.report_path + '" class="btn btn-outline-primary download-btn"><i class="fas fa-file-pdf me-1"></i> Rapport PDF</a>',
            '<a href="/static/' + result.model_path + '" class="btn btn-outline-primary download-btn"><i class="fas fa-cogs me-1"></i> Modèle</a>',
            '</div>',
            '</div>',
            '</div>'
        );

        // Data Tab
        var dataContent = [
            '<div class="tab-pane fade" id="data" role="tabpanel" aria-labelledby="data-tab">',
            '<h5 class="mb-3"><i class="fas fa-table me-2"></i>Aperçu des données</h5>',
            '<div class="table-container">',
            '<table class="table table-striped table-hover">',
            '<thead><tr>'
        ];
        for (var i = 0; i < result.columns.length; i++) {
            dataContent.push('<th>' + result.columns[i] + '</th>');
        }
        dataContent.push(
            '</tr></thead>',
            '<tbody>'
        );
        for (var j = 0; j < result.preview.length; j++) {
            dataContent.push('<tr>');
            for (var k = 0; k < result.columns.length; k++) {
                var value = result.preview[j][result.columns[k]] || '';
                dataContent.push('<td>' + value + '</td>');
            }
            dataContent.push('</tr>');
        }
        dataContent.push(
            '</tbody>',
            '</table>',
            '</div>',
            '<h5 class="mt-4 mb-3"><i class="fas fa-info-circle me-2"></i>Structure des données</h5>',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h6 class="card-title">Colonnes et types</h6>',
            '<ul class="list-group list-group-flush">'
        );
        for (var l = 0; l < result.columns.length; l++) {
            dataContent.push(
                '<li class="list-group-item d-flex justify-content-between align-items-center">' +
                result.columns[l] +
                '<span class="badge bg-light text-dark">' + result.dtypes[result.columns[l]] + '</span>' +
                '</li>'
            );
        }
        dataContent.push(
            '</ul>',
            '</div>',
            '</div>',
            '</div>'
        );

        // Visualizations Tab
        var visualizationsContent = [
            '<div class="tab-pane fade" id="visualizations" role="tabpanel" aria-labelledby="visualizations-tab">',
            '<div class="row">',
            '<div class="col-md-6">',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h5 class="card-title">Graphique principal</h5>',
            (result.plot ? '<img src="/static/' + result.plot + '" class="img-fluid" style="max-width: 500px;">' : '<p class="text-muted">Graphique non disponible</p>'),
            '</div>',
            '</div>',
            '</div>',
            '<div class="col-md-6">',
            '<div class="card mb-3">',
            '<div class="card-body">',
            '<h5 class="card-title">Graphique brut</h5>',
            (result.raw_plot ? '<img src="/static/' + result.raw_plot + '" class="img-fluid" style="max-width: 500px;">' : '<p class="text-muted">Graphique non disponible</p>'),
            '</div>',
            '</div>',
            '</div>',
            '</div>',
            '</div>'
        );

        resultTabsContent.innerHTML = summaryContent.join('') + dataContent.join('') + visualizationsContent.join('');

        // Initialize Chart
        var modelsComparisonChart = document.getElementById('modelsComparisonChart');
        if (modelsComparisonChart && typeof Chart !== 'undefined') {
            var ctx = document.createElement('canvas');
            modelsComparisonChart.appendChild(ctx);
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: Object.keys(result.models),
                    datasets: [{
                        label: 'Score',
                        data: Object.values(result.models),
                        backgroundColor: Object.keys(result.models).map(function(name) {
                            return name === result.best_model ? '#4cc9f0' : '#4361ee';
                        }),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        yAxes: [{
                            ticks: {
                                beginAtZero: true,
                                max: 1
                            }
                        }]
                    }
                }
            });
        }
    }

    function handleTaskChange() {
        var task = document.getElementById('task').value;
        document.getElementById('clusteringOptions').style.display = task === 'clustering' ? 'block' : 'none';
    }

    window.onload = function() {
        handleTaskChange();
    };
} catch (e) {
    console.error('Erreur dans le script principal:', e);
    var errorSection = document.getElementById('errorSection');
    errorSection.textContent = 'Erreur JavaScript: ' + e.message;
    errorSection.style.display = 'block';
}