const messageContainer = document.getElementById('message');
const clearResultsLink = document.getElementById('clear-results');
const resultsContainer = document.getElementById('results');
// const scoringModeLink = document.getElementById('scoring-mode');
const twoClassResultsBody = document.getElementById('two-class-table-content');
const multiClassResultsBody = document.getElementById('multi-class-table-content');
const regressionResultsBody = document.getElementById('regression-table-content');
const scoringJsonForm = document.getElementById('scoring-json-form');
const scoringForm = document.getElementById('scoring-form');
const scoringButton = document.getElementById('scoring-button');
const scoringTextarea = document.getElementById('scoring-textarea');
const scoringUrl = getWebAppBackendUrl('score');
let datasetSchemaUrl;
let datasetSchema;
let defaultValues;
let formFeatures = false;
let dateFormatOptions = {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
};

let persistResults = false;
let inputModelId = '3MdtfR6C';
let inputDatasetName = 'recipes_topic_modeling';

// HELPERS

function computeTwoClassScoringDetails(row, records) {
    let details = `<td colspan="4"><strong>Model:</strong> ${inputModelId}
    <strong>Dataset:</strong> ${inputDatasetName}
    <strong>Percentile:</strong> ${row.probaPercentile}
    <strong>Features</strong>
    <pre><code>${JSON.stringify(records, undefined, 1)}</code></pre>
    </td>`;
    return details.replace(/\n/g, "<br>");
}

function computeMultiClassScoringDetails(row, records) {
    let details = `<td colspan="4"><strong>Model:</strong> ${inputModelId}
    <strong>Dataset:</strong> ${inputDatasetName}
    <strong>Full results</strong>
    <pre><code>${JSON.stringify(row.probas, undefined, 1)}</code></pre>
    <strong>Features</strong>
    <pre><code>${JSON.stringify(records, undefined, 1)}</code></pre>
    </td>`;
    return details.replace(/\n/g, "<br>");
}

function computeRegressionScoringDetails(row, records) {
    let details = `<td colspan="4"><strong>Model:</strong> ${inputModelId}
    <strong>Dataset:</strong> ${inputDatasetName}
    <strong>Features</strong>
    <pre><code>${JSON.stringify(records, undefined, 1)}</code></pre>
    </td>`;
    return details.replace(/\n/g, "<br>");
}

function displayTwoClassResults(rows, records) {
    resultsContainer.className += ' two-class-scoring--visible';

    rows.forEach(row => {
        let summaryRow = document.createElement('tr');
        let detailsRow = document.createElement('tr');
        summaryRow.classList.add('result-summary');
        detailsRow.classList.add('result-details');
        rowContent = `<td>${new Date().toLocaleString('en-us', dateFormatOptions)}</td>`;
        Object.keys(row.probas).forEach((proba, index) => {
            let domColumnValue = document.querySelector('#class-' + index);
            domColumnValue.innerHTML = proba;
            rowContent += `<td>${row.probas[proba]}</td>`;
        });
        rowContent += `<td><strong>${row.prediction}</strong></td>`;
        summaryRow.innerHTML = rowContent;
        detailsRow.innerHTML = computeTwoClassScoringDetails(row, records);
        twoClassResultsBody.insertAdjacentElement('afterbegin', detailsRow);
        twoClassResultsBody.insertAdjacentElement('afterbegin', summaryRow);
        summaryRow.addEventListener('click', () => {
            summaryRow.classList.toggle('result-details--visible');
        });
    })
    clearResultsLink.style.display = 'block';
}

function displayTwoClassResults(rows, records) {
    resultsContainer.className += ' two-class-scoring--visible';

    rows.forEach(row => {
        let summaryRow = document.createElement('tr');
        let detailsRow = document.createElement('tr');
        summaryRow.classList.add('result-summary');
        detailsRow.classList.add('result-details');
        rowContent = `<td>${new Date().toLocaleString('en-us', dateFormatOptions)}</td>`;
        Object.keys(row.probas).forEach((proba, index) => {
            let domColumnValue = document.querySelector('#class-' + index);
            domColumnValue.innerHTML = proba;
            rowContent += `<td>${row.probas[proba]}</td>`;
        });
        rowContent += `<td><strong>${row.prediction}</strong></td>`;
        summaryRow.innerHTML = rowContent;
        detailsRow.innerHTML = computeTwoClassScoringDetails(row, records);
        twoClassResultsBody.insertAdjacentElement('afterbegin', detailsRow);
        twoClassResultsBody.insertAdjacentElement('afterbegin', summaryRow);
        summaryRow.addEventListener('click', () => {
            summaryRow.classList.toggle('result-details--visible');
        });
    })
    clearResultsLink.style.display = 'block';
}

function displayMultiClassResults(rows, records) {
    resultsContainer.className += ' multi-class-scoring--visible';

    if (persistResults === false) {
        multiClassResultsBody.innerHTML = '';
    }

    rows.forEach(row => {
        let summaryRow = document.createElement('tr');
        let detailsRow = document.createElement('tr');
        summaryRow.classList.add('result-summary');
        detailsRow.classList.add('result-details');
        summaryRow.innerHTML = `<td>${new Date().toLocaleString('en-us', dateFormatOptions)}</td><td><strong>${row.prediction}</strong></td><td>${row.probas[row.prediction]}</td>`;
        detailsRow.innerHTML = computeMultiClassScoringDetails(row, records);
        multiClassResultsBody.insertAdjacentElement('afterbegin', detailsRow);
        multiClassResultsBody.insertAdjacentElement('afterbegin', summaryRow);
        summaryRow.addEventListener('click', () => {
            summaryRow.classList.toggle('result-details--visible');
        });
    })
    clearResultsLink.style.display = 'block';
}

function displayRegressionResults(rows, records) {
    resultsContainer.className += ' regression-scoring--visible';

    if (persistResults === false) {
        regressionResultsBody.innerHTML = '';
    }

    rows.forEach(row => {
        let summaryRow = document.createElement('tr');
        let detailsRow = document.createElement('tr');
        summaryRow.classList.add('result-summary');
        detailsRow.classList.add('result-details');
        summaryRow.innerHTML = `<td>${new Date().toLocaleString('en-us', dateFormatOptions)}</td><td><strong>${row.prediction}</strong></td>`;
        detailsRow.innerHTML = computeRegressionScoringDetails(row, records);
        regressionResultsBody.insertAdjacentElement('afterbegin', detailsRow);
        regressionResultsBody.insertAdjacentElement('afterbegin', summaryRow);
        summaryRow.addEventListener('click', () => {
            summaryRow.classList.toggle('result-details--visible');
        });
    })
    clearResultsLink.style.display = 'block';
}

function displayResults(predictionData, records) {
    resultsContainer.className += ' results-table--visible';
    clearError();

    if (predictionData.regression) {
        displayRegressionResults(predictionData.regression, records);
    } else if (predictionData.classification) {
        if (predictionData.classification[0] && predictionData.classification[0].probaPercentile) {
            displayTwoClassResults(predictionData.classification, records);
        } else {
            displayMultiClassResults(predictionData.classification, records);
        }
    } // else nothing
}

function displayError() {
    messageContainer.innerHTML = 'Please add features to score.';
    (persistResults === false) && resultsContainer.classList.remove('results-table--visible');
}

function displayDatasetError() {
    console.error(arguments)
    messageContainer.innerHTML = 'The dataset schema cannot be retrieved. Please check the logs.';
    (persistResults === false) && resultsContainer.classList.remove('results-table--visible');
}

function displayScoringError() {
    messageContainer.innerHTML = 'Scoring failed. Please check the logs or change the inputs.';
    (persistResults === false) && resultsContainer.classList.remove('results-table--visible');
}

function clearError() {
    messageContainer.innerHTML = '';
}

function readForm(form) {
    let features = {};
    let elements = form.querySelectorAll('input, select, textarea');
    Array.prototype.slice.call(elements).forEach(element => {
        if (element.type && element.type === 'checkbox') {
            features[element.name] = [element.checked];
        } else {
            features[element.name] = [element.value == 'null' ? null : element.value];
        }
    });
    return features;
}

function score(event) {
    event && event.preventDefault();

    let records;
    if (formFeatures) {
        records = readForm(scoringForm);
    } else {
        records = JSON.parse(scoringTextarea.value);
    }
    if (!records) {
        displayError();
        return false;
    }

    fetch(scoringUrl, {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({records: records})
    })
        .then(response => response.json())
        .then(response => displayResults(response.prediction, records))
        .catch(displayScoringError);
}


function updateTextarea() {
    scoringTextarea.value = '';
    scoringTextarea.value += '{\n';
    datasetSchema.forEach((feature, index) => {
        let val = defaultValues[feature.name][0];
        if (val == null) {
            scoringTextarea.value += ` "${feature.name}": [null],\n`;
        } else {
            scoringTextarea.value += ` "${feature.name}": ["${defaultValues[feature.name][0]}"],\n`;
        }
    });
    scoringTextarea.value = scoringTextarea.value.substring(0, scoringTextarea.value.length - 2) + '\n}';
}


function updateForm() {
    let content = '';
    for (feature of datasetSchema) {
        let value = defaultValues[feature.name.toString()][0];
        switch (feature.computedType) {
            case 'double':
            case 'bigint':
                content += `<div class="field"><label for="${feature.name}">${feature.name}</label><input type="number" step="any" name="${feature.name}" value="${value}"></input>`;
                break;

            case 'string':
                content += `<div class="field"><label for="${feature.name}">${feature.name}</label><input type="text" name="${feature.name}" value="${value}"></input></div>`;
                break;

            case 'boolean':
                let checked = value === true;
                if (checked === true) {
                    content += `<div class="field"><label for="${feature.name}">${feature.name}</label><input checked type="checkbox" tabindex="0" name="${feature.name}" value="${feature.name}"></input></div>`;
                } else {
                    content += `<div class="field"><label for="${feature.name}">${feature.name}</label><input type="checkbox" tabindex="0" name="${feature.name}" value="${feature.name}"></input></div>`;
                }

                break;
            case 'categorical':
                let options = '';
                if (!feature.values.includes(value)) {
                    feature.values.push(value);
                }
                for (featureValue of feature.values) {
                    if (featureValue === value) {
                        options += ` <option value="${value}" selected>${value}</option> `;
                    } else {
                        options += ` <option value="${featureValue}">${featureValue}</option> `;
                    }
                }
                content += `<div class="field"><label for="${feature.name}">${feature.name}</label><div class="select"><select tabindex="0" name="${feature.name}">${options}</select></div></div>`;
                break;
        }
    }

    scoringForm.innerHTML = content;
    Array.prototype.slice.call(scoringForm.querySelectorAll('input, select, textarea')).forEach(element => {
        element.addEventListener('change', updateDefaultValuesFromForm);
    });
}


function updateSample() {
    datasetSchemaUrl = getWebAppBackendUrl('get-dataset-schema') + '?dataset_name=' + inputDatasetName;
    scoringTextarea.value = '';
    fetch(datasetSchemaUrl).then(response => response.json()).then(response => {
        datasetSchema = response.schema;
        defaultValues = JSON.parse(response.defaultValues);
        clearError();

        updateTextarea();
        updateForm();
    }).catch(displayDatasetError);
}

function updateSampleFromJson() {
    defaultValues = JSON.parse(scoringTextarea.value);
    updateForm();
}


function updateDefaultValuesFromForm() {
    defaultValues = readForm(scoringForm);
    updateTextarea();
}


function switchScoringMode() {
    formFeatures = !formFeatures;
    scoringJsonForm.style.display = 'none';
    scoringForm.style.display = 'block';
    // if (formFeatures) {
    //     scoringModeLink.innerHTML = 'Switch to JSON view';
    //     scoringJsonForm.style.display = 'none';
    //     scoringForm.style.display = 'block';
    // } else {
    //     scoringModeLink.innerHTML = 'Switch to form view';
    //     scoringJsonForm.style.display = 'block';
    //     scoringForm.style.display = 'none';
    // }
}


function clearResults() {
    twoClassResultsBody.innerHTML = '';
    multiClassResultsBody.innerHTML = '';
    regressionResultsBody.innerHTML = '';
    resultsContainer.classList.remove('results-table--visible');
    clearResultsLink.style.display = 'none';
}


// Attach listeners
// scoringModeLink.addEventListener('click', switchScoringMode);
scoringButton.addEventListener('click', score);
scoringTextarea.addEventListener('change', updateSampleFromJson);
clearResultsLink.addEventListener('click', clearResults);

updateSample();
switchScoringMode();
