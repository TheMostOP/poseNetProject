let video;
let poseNet;
let pose;
let skeleton;

let brain;

let state = 'waiting';
let targetLabel;

let predicticedCharacter = '???';

function keyPressed() {
    if (key == 's') {
        brain.saveData();
    } else {
        targetLabel = key;
        console.log(targetLabel);
        setTimeout(function () {
            console.log('collecting')
            state = 'collecting';
            setTimeout(function () {
                console.log('not collecting');
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
        outputs: 4,
        task: 'classification',
        debug: true
    }
    brain = ml5.neuralNetwork(options);
    brain = ml5.neuralNetwork(options);
    const modelInfo = {
        model: 'model/model.json',
        metadata: 'model/model_meta.json',
        weights: 'model/model.weights.bin',
    };
    brain.load(modelInfo, brainLoaded);
    // brain.loadData('tpose.json', dataReady);
}

function brainLoaded() {
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
        brain.classify(inputs, gotResult);
    } else {
        setTimeout(classifyPose, 100);
    }
}

function gotResult(error, results) {
    console.log(results);
    console.log(results[0].label);
    //display the name of the closest pose
    switch (results[0].label) {
        case 'q':
            predicticedCharacter = 'Mario';
            document.querySelector('#Mario').checked = true;
            break;
        case 'w':
            predicticedCharacter = 'Donkey Kong';
            document.querySelector('#DonkeyKong').checked = true;
            break;
        case 'e':
            predicticedCharacter = 'Link';
            document.querySelector('#Link').checked = true;
            break;
        case 'r':
            predicticedCharacter = 'Samus';
            document.querySelector('#Samus').checked = true;
            break;
        case 't':
            predicticedCharacter = 'Dark Samus';
            document.querySelector('#DarkSamus').checked = true;
            break;
        case 'y':
            predicticedCharacter = 'Yoshi';
            document.querySelector('#Yoshi').checked = true;
            break;
        case 'u':
            predicticedCharacter = 'Kirby';
            document.querySelector('#Kirby').checked = true;
            break;
        case 'i':
            predicticedCharacter = 'Fox';
            document.querySelector('#Fox').checked = true;
            break;
        case 'o':
            predicticedCharacter = 'Pikachu';
            document.querySelector('#Pikachu').checked = true;
            break;
        case 'p':
            predicticedCharacter = 'Luigi';
            document.querySelector('#Luigi').checked = true;
            break;
        case 'a':
            predicticedCharacter = 'Ness';
            document.querySelector('#Ness').checked = true;
            break;
        default:
            predicticedCharacter = '???';
            break;
    }
    document.querySelector('.character').innerHTML = predicticedCharacter;
    classifyPose();
}

function dataReady() {
    brain.normalizeData();
    brain.train({ epochs: 50 }, finished);
}

function finished() {
    console.log('model trained');
    brain.save();
}

function gotPoses(poses) {
    //console.log(poses);
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
            brain.addData(inputs, target);
        }
    }
}

function modelLoaded() {
    console.log('poseNet ready');
}

function draw() {
    push();
    translate(video.width, 0);
    scale(-1, 1);
    image(video, 0, 0, video.width, video.height);


    if (pose) {
        for (let i = 0; i < skeleton.length; i++) {
            let a = skeleton[i][0];
            let b = skeleton[i][1];
            strokeWeight(2);
            stroke(0);

            line(a.position.x, a.position.y, b.position.x, b.position.y);
        }
        for (let i = 0; i < pose.keypoints.length; i++) {
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            fill(0);
            stroke(255);
            ellipse(x, y, 16, 16);
        }
    }
}