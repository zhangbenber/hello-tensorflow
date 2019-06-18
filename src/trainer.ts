import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';
import { maxWordLength, getWordsData, testWords } from './common';

const modelName = 'model';
const modelURI = `file://${path.resolve(__dirname, modelName)}`;

const threshold = .5;
console.log(modelURI);

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.conv1d({
    filters: 100,
    kernelSize: 4,
    activation: 'relu',
    inputShape: [maxWordLength, 26]
  }));
  model.add(tf.layers.maxPooling1d({ poolSize: 27 }));
  model.add(tf.layers.flatten({ }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.summary();
  return model;
}

function trainModel(model: tf.Sequential, inputs: tf.Tensor, labels: tf.Tensor, validationInputs: tf.Tensor, validationLabels: tf.Tensor) {
  return new Promise(r => {
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['accuracy']
    });
    model.fit(inputs, labels, {
      shuffle: true,
      validationData: [validationInputs, validationLabels],
      epochs: 8,
      batchSize: 96,
      callbacks: {
        onTrainEnd: r
      }
    });
  })
}

function getModel() {
  return new Promise(r => {
    const words = fs.readFileSync('./words.txt', { encoding: 'utf-8' }).split('\r\n');
    words.shift();
    tf.util.shuffle(words);
    const noise = words.map(t => new Array(~~(Math.pow(t.length / 30, 1) * 30)).fill(undefined).map(_ =>
      String.fromCharCode(~~(Math.random() * 26) + 97)).join('')
    );
    const wordData = getWordsData(tf, words);
    const noiseData = getWordsData(tf, noise);
    const trainDataLength = ~~(words.length * .8);
    const wordLabels = tf.ones([wordData.shape[0]])
    const noiseLabels = tf.zeros([noiseData.shape[0]]);
    const trainData = wordData.slice(0, trainDataLength).concat(noiseData.slice(0, trainDataLength));
    const trainLabels = wordLabels.slice(0, trainDataLength).concat(noiseLabels.slice(0, trainDataLength));
    const validationData = wordData.slice(trainDataLength).concat(noiseData.slice(trainDataLength));
    const validationLabels = wordLabels.slice(trainDataLength).concat(noiseLabels.slice(trainDataLength));
    const model = createModel();
    trainModel(model, trainData, trainLabels, validationData, validationLabels).then(e => {
      model.save(modelURI);
      r(model);
    });
  });
}

function testModel(model: tf.Sequential) {
  console.log(`threshold = ${threshold}`);
  const result = (model.predict(getWordsData(tf, testWords)) as tf.Tensor).as1D().arraySync();
  const resultString = testWords.map((w, i) => {
    const passed = result[i] > threshold;
    return `${passed ? '\x1b[0;32m' : '\x1b[0;31m'}${result[i].toFixed(3)} \x1b[1m${w.padEnd(22)}\x1b[0m`
  });
  for (let i = 0; i < resultString.length;) {
    console.log(resultString[i++], resultString[i++] || '', resultString[i++] || '', resultString[i++] || '', resultString[i++] || '');
  }

  console.log('============================');

  const randomWords = new Array(10000).fill(undefined).map(() => {
    const x = Math.random() * 2;
    const y = Math.pow((Math.pow(x, 4) + Math.pow(x, .5) * 9) / 30, 1.5) * 30 + 1;
    return new Array(~~y).fill(undefined).map(t => String.fromCharCode(~~(Math.random() * 26) + 97)).join('')
  });
  const resultForRandomWords = (model.predict(getWordsData(tf, randomWords)) as tf.Tensor).as1D().arraySync();
  const randomWordsPair = randomWords.map((w, i) => ({
    word: w,
    result: resultForRandomWords[i]
  }));
  randomWordsPair.sort((a, b) => b.result - a.result);
  console.log(randomWordsPair.slice(0, 30).map(p => p.word).join(' '));
}

tf.loadLayersModel(`${modelURI}/model.json`).then(model => {
  testModel(model as tf.Sequential);
}, () => {
  getModel().then(model => {
    testModel(model as tf.Sequential);
  });
});