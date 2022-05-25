"use strict";

// train model


let video;
let poseNet;
let pose;
let skeleton;
let emoji;
let img;
let pic;

let network;
let poseLabel = "u";


function setup() {
  createCanvas(640, 480);
  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);
  img = loadImage('images/standup.jpg');
  emoji = loadImage('images/smiley.png');

  let options = {
    inputs: 34,
    outputs: 5,
    task: 'classification',
    debug: true
  };
  network = ml5.neuralNetwork(options);
  const modelInfo = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin',
  };
  network.load(modelInfo, networkLoaded);

}

function networkLoaded() {
  console.log('pose classification ready!');
  classifyPose();
}

function classifyPose() {
  if (pose) {
    let inputs = [];
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      inputs.push(x);
      inputs.push(y);
    }
    network.classify(inputs, gotResult);
  } else {
    setTimeout(classifyPose, 100);
  }
}

function gotResult(error, results) {

  if (results[0].confidence > 0.8) {
    poseLabel = results[0].label;
    console.log(poseLabel);
  }

  classifyPose();

}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  // translate(video.width, 0);
  // scale(-1, 1);
  // image(video, 0, 0, video.width, video.height);
  // if (poseLabel != "R") {
  //   clear();
  //   image(emoji, 0, 0, emoji.width / 2, emoji.height / 2);
  // } else if (poseLabel == "R") {
  //   setTimeout(standUp, 10000);
  // } 

  if (poseLabel != "r") {
    clear();
    pic = image(emoji, 0, 0, emoji.width / 2, emoji.height / 2);
  } else {
    clear();
    pic = image(img, 0, 0);
  }

}


//references
// ml5.js: Pose Classification
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/Courses/ml5-beginners-guide/7.2-pose-classification.html
// https://youtu.be/FYgYyq-xqAw