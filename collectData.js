"use strict";

// collect 5 different states: Sitting (r), drinking(d), yes(arm up)(y), no(arm to side)(n), standing(u)


let video;
let poseNet;
let pose;
let skeleton;

let network;

let state = 'waiting';
let targetLabel;

function keyPressed() {
  if (key == 's') {
    network.saveData();
  } else {
    targetLabel = key;
    console.log(targetLabel);
    setTimeout(function () {
      console.log('collecting');
      state = 'collecting';
      setTimeout(function () {
        console.log('done collecting');
        state = 'waiting';
      }, 10000);
    }, 10000);
  }
}

function setup() {
  createCanvas(640, 480);

  video = createCapture(VIDEO);
  video.hide();
  poseNet = ml5.poseNet(video, modelLoaded);
  poseNet.on('pose', gotPoses);

  let options = {
    inputs: 34,
    outputs: 5,
    task: 'classification',
    debug: true
  };
  network = ml5.neuralNetwork(options);
}

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
    if (state == 'collecting') {
      let inputs = [];
      for (let i = 0; i < pose.keypoints.length; i++) {
        let x = pose.keypoints[i].position.x;
        let y = pose.keypoints[i].position.y;
        inputs.push(x);
        inputs.push(y);
      }
      let target = [targetLabel];
      network.addData(inputs, target);
    }
  }
}


function modelLoaded() {
  console.log('poseNet ready');
}

function draw() {
  translate(video.width, 0);
  scale(-1, 1);
  image(video, 0, 0, video.width, video.height);

  if (pose) {
    for (let i = 0; i < skeleton.length; i++) {
      let a = skeleton[i][0];
      let b = skeleton[i][1];
      strokeWeight(1);
      stroke(0);

      line(a.position.x, a.position.y, b.position.x, b.position.y);
    }
    for (let i = 0; i < pose.keypoints.length; i++) {
      let x = pose.keypoints[i].position.x;
      let y = pose.keypoints[i].position.y;
      fill(0);
      ellipse(x, y, 10, 10);
    }
  }
}

//references
// ml5.js: Pose Classification
// The Coding Train / Daniel Shiffman
// https://thecodingtrain.com/Courses/ml5-beginners-guide/7.2-pose-classification.html
// https://youtu.be/FYgYyq-xqAw