import * as React from 'react';
import * as tf from '@tensorflow/tfjs';
import { getWordsData } from './common';
import './App.css';

class App extends React.Component {
  public state = {
    word: '',
    result: 0,
    passed: false
  }
  private model?: tf.Sequential;

  constructor(props: {}) {
    super(props);
    this.handleChange = this.handleChange.bind(this);
  }

  public render() {
    return (
      <div className="App">
        <p className={this.state.passed ? 'pass' : 'fail'}>
          <span>Result for</span>
          <input type="text" onChange={this.handleChange} value={this.state.word} spellCheck={false} />
          <span>is</span>
          <code>{(this.state.result * 100).toFixed(1)}%</code>
        </p>
      </div>
    );
  }

  public componentDidMount() {
    tf.loadLayersModel('./model/model.json').then((model: tf.Sequential) => {
      (window as any).test = this.testWords.bind(this, model);
      this.model = model;
    }, console.error);
  }

  private handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const word = e.target.value.toLowerCase().replace(/[^a-z]/, '').substr(0, 30);
    this.setState({
      word
    }, () => {
      setTimeout(() => {
        const data = this.testWords([word])[0];
        this.setState({
          result: data.result,
          passed: data.passed
        });
      });
    });
  }

  private testWords(words: string[]) {
    if (!this.model) {
      return [];
    }
    const threshold = .5;
    const startTime = +new Date();
    const result = (this.model.predict(getWordsData(tf, words)) as tf.Tensor).as1D().arraySync();
    console.log(+new Date() - startTime);
    return words.map((w, i) => ({
      passed: result[i] > threshold,
      word: w,
      result: result[i]
    }));
  }

}

export default App;
