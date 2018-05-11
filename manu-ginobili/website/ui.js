/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import embed from 'vega-embed';

import {IRIS_CLASSES, IRIS_NUM_CLASSES} from './data';

/**
 * Clear the evaluation table.
 */
export function clearEvaluateTable() {
  const tableBody = document.getElementById('evaluate-tbody');
  while (tableBody.children.length > 1) {
    tableBody.removeChild(tableBody.children[1]);
  }
}

/**
 * Plot new loss values.
 *
 * @param lossValues An `Array` of data to append to.
 * @param epoch Training epoch number.
 * @param newTrainLoss The new training loss, as a single `Number`.
 * @param newValidationLoss The new validation loss, as a single `Number`.
 */
export function plotLosses(lossValues, epoch, newTrainLoss, newValidationLoss) {
  lossValues.push({'epoch': epoch, 'loss': newTrainLoss, 'set': 'train'});
  lossValues.push(
      {'epoch': epoch, 'loss': newValidationLoss, 'set': 'validation'});
  embed(
      '#lossCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': lossValues},
        'mark': 'line',
        'encoding': {
          'x': {'field': 'epoch', 'type': 'ordinal'},
          'y': {'field': 'loss', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal'},
        }
      },
      {});
}

/**
 * Plot new accuracy values.
 *
 * @param lossValues An `Array` of data to append to.
 * @param epoch Training epoch number.
 * @param newTrainLoss The new training accuracy, as a single `Number`.
 * @param newValidationLoss The new validation accuracy, as a single `Number`.
 */
export function plotAccuracies(
    accuracyValues, epoch, newTrainAccuracy, newValidationAccuracy) {
  accuracyValues.push(
      {'epoch': epoch, 'accuracy': newTrainAccuracy, 'set': 'train'});
  accuracyValues.push(
      {'epoch': epoch, 'accuracy': newValidationAccuracy, 'set': 'validation'});
  embed(
      '#accuracyCanvas', {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'data': {'values': accuracyValues},
        'mark': 'line',
        'encoding': {
          'x': {'field': 'epoch', 'type': 'ordinal'},
          'y': {'field': 'accuracy', 'type': 'quantitative'},
          'color': {'field': 'set', 'type': 'nominal'},
        }
      },
      {});
}

/**
 * Get manually input Iris data from the input boxes.
 */
export function getManualInputData() {
  return [
    Number(document.getElementById('petal-length').value),
    Number(document.getElementById('petal-width').value),
    Number(document.getElementById('sepal-length').value),
    Number(document.getElementById('sepal-width').value),
  ];
}

export function setManualInputWinnerMessage(message) {
  const winnerElement = document.getElementById('winner');
  winnerElement.textContent = message;
}

function logitsToSpans(logits) {
  let idxMax = -1;
  let maxLogit = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < logits.length; ++i) {
    if (logits[i] > maxLogit) {
      maxLogit = logits[i];
      idxMax = i;
    }
  }
  const spans = [];
  for (let i = 0; i < logits.length; ++i) {
    const logitSpan = document.createElement('span');
    logitSpan.textContent = logits[i].toFixed(3);
    if (i === idxMax) {
      logitSpan.style['font-weight'] = 'bold';
    }
    logitSpan.classList = ['logit-span'];
    spans.push(logitSpan);
  }
  return spans;
}

function renderLogits(logits, parentElement) {
  while (parentElement.firstChild) {
    parentElement.removeChild(parentElement.firstChild);
  }
  logitsToSpans(logits).map(logitSpan => {
    parentElement.appendChild(logitSpan);
  });
}

export function renderLogitsForManualInput(logits) {
  const logitsElement = document.getElementById('logits');
  renderLogits(logits, logitsElement);
}

export function renderEvaluateTable(xData, yTrue, yPred, logits) {
  const tableBody = document.getElementById('evaluate-tbody');

  for (let i = 0; i < yTrue.length; ++i) {
    const row = document.createElement('tr');
    for (let j = 0; j < 4; ++j) {
      const cell = document.createElement('td');
      cell.textContent = xData[4 * i + j].toFixed(1);
      row.appendChild(cell);
    }
    const truthCell = document.createElement('td');
    truthCell.textContent = IRIS_CLASSES[yTrue[i]];
    row.appendChild(truthCell);
    const predCell = document.createElement('td');
    predCell.textContent = IRIS_CLASSES[yPred[i]];
    predCell.classList =
        yPred[i] === yTrue[i] ? ['correct-prediction'] : ['wrong-prediction'];
    row.appendChild(predCell);
    const logitsCell = document.createElement('td');
    const exampleLogits =
        logits.slice(i * IRIS_NUM_CLASSES, (i + 1) * IRIS_NUM_CLASSES);
    logitsToSpans(exampleLogits).map(logitSpan => {
      logitsCell.appendChild(logitSpan);
    });
    row.appendChild(logitsCell);
    tableBody.appendChild(row);
  }
}

export function wireUpEvaluateTableCallbacks(predictOnManualInputCallback) {
  const petalLength = document.getElementById('petal-length');
  const petalWidth = document.getElementById('petal-width');
  const sepalLength = document.getElementById('sepal-length');
  const sepalWidth = document.getElementById('sepal-width');

  const increment = 0.1;
  document.getElementById('petal-length-inc').addEventListener('click', () => {
    petalLength.value = (Number(petalLength.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-length-dec').addEventListener('click', () => {
    petalLength.value = (Number(petalLength.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width-inc').addEventListener('click', () => {
    petalWidth.value = (Number(petalWidth.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width-dec').addEventListener('click', () => {
    petalWidth.value = (Number(petalWidth.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-length-inc').addEventListener('click', () => {
    sepalLength.value = (Number(sepalLength.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-length-dec').addEventListener('click', () => {
    sepalLength.value = (Number(sepalLength.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width-inc').addEventListener('click', () => {
    sepalWidth.value = (Number(sepalWidth.value) + increment).toFixed(1);
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width-dec').addEventListener('click', () => {
    sepalWidth.value = (Number(sepalWidth.value) - increment).toFixed(1);
    predictOnManualInputCallback();
  });

  document.getElementById('petal-length').addEventListener('change', () => {
    predictOnManualInputCallback();
  });
  document.getElementById('petal-width').addEventListener('change', () => {
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-length').addEventListener('change', () => {
    predictOnManualInputCallback();
  });
  document.getElementById('sepal-width').addEventListener('change', () => {
    predictOnManualInputCallback();
  });
}

export function loadTrainParametersFromUI() {
  return {
    epochs: Number(document.getElementById('train-epochs').value),
    learningRate: Number(document.getElementById('learning-rate').value)
  };
}

export function status(statusText) {
  console.log(statusText);
  document.getElementById('demo-status').textContent = statusText;
}

export function disableLoadModelButtons() {
  document.getElementById('load-pretrained-remote').style.display = 'none';
  document.getElementById('load-pretrained-local').style.display = 'none';
}
