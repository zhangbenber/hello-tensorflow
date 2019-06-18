import * as React from 'react';
import * as tf from '@tensorflow/tfjs';
import { getWordsData, testWords } from './common';
import './App.css';

import logo from './logo.svg';

class App extends React.Component {
  public render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
        <p className="App-intro">
          To get started, edit <code>src/App.tsx</code> and save to reload.
        </p>
      </div>
    );
  }

  public componentDidMount() {
    tf.loadLayersModel('./model/model.json').then((model: tf.Sequential) => {
      (window as any).test = this.testWords.bind(this, model);
      this.testWords(model, ['test']);
      this.testWords(model, ['gfsdsdfg', 'dfg', 'dsg']);
      this.testWords(model, ['some', 'more', 'english', 'words', 'for', 'performance', 'test']);
      this.testWords(model, ['its', 'faster', 'if', 'items', 'have', 'equal', 'length']);
      this.testWords(model, testWords);
      this.testWords(model, testWords);
    }, console.error);
  }

  private testWords(model: tf.Sequential, words: string[]) {
    const threshold = .5;
    console.log(`threshold = ${threshold}`);
    const startTime = +new Date();
    const result = (model.predict(getWordsData(tf, words)) as tf.Tensor).as1D().arraySync();
    const resultString = words.map((w, i) => {
      const passed = result[i] > threshold;
      return `${passed ? '+' : '-'} ${result[i].toFixed(3)} ${w}`
    });
    console.log(+new Date() - startTime);
    console.log(resultString.join('\n'));
  }

}

export default App;
