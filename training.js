"use strict";

// train model


let network;

function setup() {
  createCanvas(640, 480);
  let options = {
    inputs: 34,
    outputs: 5,
    task: 'classification',
    debug: true
  };
  network = ml5.neuralNetwork(options);
  network.loadData('actions.json', dataReady);
}

function dataReady() {
  network.normalizeData();
  network.train({
    epochs: 50
  }, finished);
}

function finished() {
  console.log('you finished your training session!');
  network.save();
}

//references
// ml5.js: Pose Classification
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/Courses/ml5-beginners-guide/7.2-pose-classification.html
// https://youtu.be/FYgYyq-xqAw