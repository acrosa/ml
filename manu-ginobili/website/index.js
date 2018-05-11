"use strict";

import * as tf from "@tensorflow/tfjs";
import React, { Component } from "react";
import { render } from "react-dom";
import * as data from "./data";
import * as loader from "./loader";
import * as ui from "./ui";
import * as STATS from "./stats.json";

class App extends Component {
  async buildStats() {
    let stats = [];
    for (const stat in STATS) {
      stats.push(STATS[stat]);
    }
    return stats;
  }

  async createModel(params) {
    const { learningRate } = params;
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 3, inputShape: [3] }));
    model.add(tf.layers.dense({ units: 1, inputShape: [3] }));

    const optimizer = tf.train.adam(learningRate);

    await model.compile({
      loss: "meanSquaredError",
      optimizer: optimizer,
      metrics: ["accuracy"]
    });

    return model;
  }

  constructor(props) {
    super(props);

    this.state = {
      model: null,
      minutes: 30,
      age: 41,
      training: false,
      epochs: 10
    };
  }

  async componentDidMount() {
    const model = await this.createModel({ learningRate: 0.01 });
    const stats = await this.buildStats();

    this.setState({ model: model, stats: stats });
  }

  async train() {
    const { model, stats, epochs } = this.state;

    this.setState({ training: true });

    // Generate some synthetic data for training.
    const xs = tf.tensor2d(stats.map(stat => [stat.G, stat.MP, stat.Age]), [
      stats.length,
      3
    ]);
    const ys = tf.tensor2d(stats.map(stat => stat.PTS), [stats.length, 1]);

    const history = model
      .fit(xs, ys, {
        epochs: epochs,
        validationData: [xs, ys],
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            console.log("Loss: " + logs.loss);
            console.log("Accuracy: " + logs.acc * 100);
            console.log("Val Accuracy: " + logs.val_acc * 100);
            console.log("---------------------");
            await tf.nextFrame();
          }
        }
      })
      .then(() => {
        this.setState({ trainedModel: model }, () => this.predict());
      });
  }

  async predict() {
    const { trainedModel, minutes, age } = this.state;
    const results = trainedModel
      .predict(tf.tensor2d([1, minutes, age], [1, 3]))
      .dataSync();
    this.setState({ predicted: results });
    console.log(results);
  }

  setAge(age) {
    this.setState({ age: age });
  }

  render() {
    const { predicted, training, age } = this.state;
    const setAge = this.setAge.bind(this);
    const train = this.train.bind(this);
    const predict = this.predict.bind(this);

    return (
      <div>
        <h1>{this.props.title}</h1>
        <label htmlFor="age-slider">Age</label>
        <input
          id="age-slider"
          type={"range"}
          min={25}
          max={41}
          value={age}
          className="slider"
          onChange={e => setAge(e.target.value)}
        />

        {training ? (
          "Training..."
        ) : (
          <div className="actions">
            <button onClick={train}>Train</button>
            <button onClick={predict}>Predict</button>
          </div>
        )}
        <div>{predicted}</div>
      </div>
    );
  }
}

render(<App title={"#ELPIBEDE41"} />, document.getElementById("app"));
