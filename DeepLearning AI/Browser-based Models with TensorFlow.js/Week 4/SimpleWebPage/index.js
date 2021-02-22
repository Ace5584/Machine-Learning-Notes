let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
  }

async function train(){
    model = tf.sequential({
        layers: [
          tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
          tf.layers.dense({ units: 100, activation: 'relu'}),
          tf.layers.dense({ units: 3, activation: 'softmax'})
        ]
      });
}

async function init(){
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
}



init();