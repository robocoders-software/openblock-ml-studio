/* ── ML Engine ────────────────────────────────────────────────────
   Transfer-learning pipeline:
     1. Backbone  – MobileNetV1 α=0.25, served locally on port 20112
                    loaded via tf.loadLayersModel() and wrapped with a
                    mobilenet-package-compatible .infer(imgEl, true) API
                    so the VM extension works unchanged.
     2. Head      – small trainable dense network (dense-relu → dropout → dense)
     3. Persistence – TF.js built-in IndexedDB model saving
     4. Interface – classifier.predictClass(logits) is KNN-compatible so the
                    VM extension works unchanged
   ─────────────────────────────────────────────────────────────── */
import {
    saveImageToIDB, getImageFromIDB,
    idbPut, idbGet, idbDelete,
    deleteProjectIDB as _idbDeleteProject
} from './idb.js';

/* ── Singletons ── */
let _tf         = null;
let _mobileNet  = null;   // the wrapped backbone object (has .infer())
let _mobileNetP = null;

/* ── Local model: served by the resource server on port 20112.
   Run  scripts/download-mobilenet.js  once to populate
   external-resources/models/mobilenet_v1_0.25_224/
── */
const LOCAL_MOBILENET_URL  = 'http://localhost:20112/models/mobilenet_v1_0.25_224/model.json';
const MOBILENET_INPUT_SIZE = 224;
const EMBEDDING_LAYER_NAME = 'global_average_pooling2d_1';   // output: [batch, 256]

/* ── Active model (set when user clicks "Use in Blocks") ── */
let _activeModel = null;
export const setActiveModel = model => {
    _activeModel = model;
    if (typeof window !== 'undefined') window.__openblockMLModel = model;
};
export const getActiveModel = () => _activeModel;

/* ── Public: initialised TF.js singleton ── */
export const getTF = async () => {
    if (_tf) return _tf;
    const m = await import(/* webpackChunkName: "tfjs" */ '@tensorflow/tfjs');
    _tf = m.default || m;
    await _tf.ready();
    return _tf;
};

/* ── Public: MobileNetV1 feature extractor (cached singleton) ──
   Loads the Keras layers model from the local resource server and wraps it
   in an object whose .infer(imgEl, embedding) API matches the
   @tensorflow-models/mobilenet package — so the VM extension (which calls
   localModel.mobileNet.infer(canvas, true)) works without any changes.

   .infer(imgEl):
     • Converts imgEl (HTMLImageElement / HTMLVideoElement / HTMLCanvasElement)
       to a tensor, resizes to 224×224, normalises to [-1, 1]
     • Runs the embedding sub-model (GlobalAveragePooling2D output: [1, 256])
     • Returns the tensor — caller must .dispose() it
── */
export const getMobileNet = async onStatus => {
    if (_mobileNet) return _mobileNet;
    if (_mobileNetP) return _mobileNetP;

    _mobileNetP = (async () => {
        const tf = await getTF();
        onStatus && onStatus('Loading MobileNetV1 backbone…');

        let fullModel;
        try {
            fullModel = await tf.loadLayersModel(LOCAL_MOBILENET_URL);
        } catch (e) {
            throw new Error(
                `Failed to load MobileNetV1 from local server (${LOCAL_MOBILENET_URL}). ` +
                'Make sure the desktop app is running and the resource server (port 20112) is active. ' +
                `Original error: ${e.message}`
            );
        }

        // Build embedding sub-model: input → GlobalAveragePooling2D → [batch, 256]
        const embLayer     = fullModel.getLayer(EMBEDDING_LAYER_NAME);
        const embeddingNet = tf.model({inputs: fullModel.inputs, outputs: embLayer.output});
        // Note: fullModel intentionally not disposed — embeddingNet shares its layers/weights

        /* ── Wrapper: exposes .infer(imgEl) matching mobilenet package API ──
           The VM extension calls: localModel.mobileNet.infer(canvas, true)
           The TestingPanel uses:  predictImages(videoEl, classifier, mobileNet, labels)
                                   → net.infer(imgEl, true)
           In-blocks training:     net.infer(img, true) per training image
           All three paths converge here. ── */
        const wrapper = {
            _model: embeddingNet,

            /* imgEl: HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | ImageData
               embedding param is ignored — we always return the embedding tensor */
            infer (imgEl /*, embedding = true */) {
                return _tf.tidy(() => {
                    const imgTensor  = _tf.browser.fromPixels(imgEl);             // [H, W, 3]
                    const resized    = _tf.image.resizeBilinear(
                        imgTensor, [MOBILENET_INPUT_SIZE, MOBILENET_INPUT_SIZE]   // [224, 224, 3]
                    );
                    const normalised = resized.toFloat().div(127.5).sub(1.0);     // [-1, 1]
                    const batched    = normalised.expandDims(0);                  // [1, 224, 224, 3]
                    return embeddingNet.predict(batched);                         // [1, 256]
                    // tidy disposes all intermediates; the returned [1,256] tensor survives
                });
            }
        };

        _mobileNet  = wrapper;
        _mobileNetP = null;
        onStatus && onStatus('MobileNetV1 ready (local)');
        return wrapper;
    })();

    return _mobileNetP;
};

/* ── Internal: load an image element from a data-URL ── */
const loadImg = src =>
    new Promise(resolve => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload  = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = src;
    });

/* ── Internal: build classification head ──
   Architecture mirrors ML for Kids:
     dense(relu) → dropout(0.2) → dense(logits — no activation)
   Loss is softmaxCrossEntropy (from_logits equivalent) with label smoothing.
── */
const buildHead = (tf, inputDim, numClasses) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [inputDim],
        units:      128,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.dropout({rate: 0.2}));
    model.add(tf.layers.dense({
        units:      numClasses,
        activation: 'linear',              // raw logits; softmax applied in loss and wrapHead
        kernelInitializer: 'varianceScaling'
    }));
    return model;
};

/* ── Internal: wrap a trained head so its interface matches KNN predictClass ──
   predictClass(logits) is called by:
     • The VM extension:  classifier.predictClass(logits)   where logits = mobileNet.infer(canvas)
     • predictImages():   classifier.predictClass(logits)   where logits = net.infer(imgEl)
   Returns { label, classIndex, confidences:{[String(i)]: probability} }
── */
const wrapHead = (head, labels) => ({
    _head: head,
    async predictClass (logits) {
        const out = _tf.tidy(() => {
            const flat = logits.reshape([1, logits.shape[logits.shape.length - 1]]);
            return _tf.softmax(head.predict(flat));
        });
        const arr = await out.data();
        out.dispose();
        const confidences = {};
        arr.forEach((p, i) => { confidences[String(i)] = p; });
        const topI = arr.indexOf(Math.max(...arr));
        return {
            label:       (labels[topI] !== undefined && labels[topI] !== null) ? labels[topI] : String(topI),
            classIndex:  topI,
            confidences
        };
    },
    getNumClasses: () => labels.length,
    getClassifierDataset: () => ({})
});

/* ── Image training ──────────────────────────────────────────────
   Returns the trained classifier wrapper.
   onEpochEnd(epochIdx, valAcc) is called after every epoch.
── */
export const trainImages = async (
    labels, trainingData, projectId,
    onStatus, onProgress,
    {epochs = 20, batchSize = 32, learningRate = 0.005, onEpochEnd} = {}
) => {
    const tf  = await getTF();
    const net = await getMobileNet(onStatus);

    const total = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
    if (total < 2) throw new Error('Need at least 2 training images total.');
    if (!labels.every(l => (trainingData[l] || []).length >= 1)) {
        throw new Error('Every class needs at least 1 image.');
    }

    /* ── Phase 1 of 2: feature extraction (0–50 %) ── */
    onStatus && onStatus('Extracting MobileNetV1 features…');
    const embRows   = [];   // each entry: Float32Array of length embDim
    const labelIdxs = [];
    let done = 0;
    let embDim = 256;

    for (let i = 0; i < labels.length; i++) {
        for (const ex of (trainingData[labels[i]] || [])) {
            let src = ex.data && ex.data.startsWith('data:') ? ex.data : null;
            if (!src) src = await getImageFromIDB(projectId, ex.id);
            if (!src) { done++; onProgress && onProgress(Math.round(done / total * 50)); continue; }

            const img = await loadImg(src);
            if (!img) { done++; onProgress && onProgress(Math.round(done / total * 50)); continue; }

            const emb  = net.infer(img, true);  // [1, embDim]
            embDim     = emb.shape[emb.shape.length - 1];
            const data = await emb.data();       // read to CPU Float32Array
            emb.dispose();
            embRows.push(data);
            labelIdxs.push(i);

            done++;
            onProgress && onProgress(Math.round(done / total * 50));
        }
    }

    if (embRows.length === 0) throw new Error('No valid training images could be loaded.');

    /* ── Stack into training tensors (plain JS array → single tensor, no WebGL concat) ── */
    const flat = new Float32Array(embRows.length * embDim);
    embRows.forEach((row, r) => flat.set(row, r * embDim));
    const xs = tf.tensor2d(flat, [embRows.length, embDim]);  // [N, embDim]

    const ys = tf.oneHot(tf.tensor1d(labelIdxs, 'int32'), labels.length); // [N, C]

    /* ── Phase 2 of 2: train head (50–100 %) ── */
    onStatus && onStatus('Building & training classification head…');
    const head = buildHead(tf, embDim, labels.length);
    head.compile({
        optimizer: tf.train.sgd(learningRate),
        loss:      (ys_, logits) =>
            tf.losses.softmaxCrossEntropy(ys_, logits, undefined, 0.1), // label smoothing
        metrics:   ['accuracy']
    });

    const safeEpochs  = Math.max(1, Math.round(epochs));
    const valSplit    = Math.min(0.5, Math.max(0.15, 1.0 / xs.shape[0]));
    const trainCount  = Math.floor(xs.shape[0] * (1 - valSplit));
    const effectiveBs = Math.max(1, Math.min(Math.max(1, Math.round(batchSize)), trainCount));

    await head.fit(xs, ys, {
        epochs:          safeEpochs,
        batchSize:       effectiveBs,
        shuffle:         true,
        validationSplit: valSplit,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const valAcc   = logs.val_acc      || logs.val_accuracy  || 0;
                const trainAcc = logs.acc          || logs.accuracy      || 0;
                const valLoss  = logs.val_loss     || 0;
                const trainLoss = logs.loss        || 0;
                const displayAcc = valAcc || trainAcc;
                onStatus && onStatus(`Epoch ${epoch + 1}/${safeEpochs}  acc: ${(displayAcc * 100).toFixed(1)}%  loss: ${trainLoss.toFixed(4)}`);
                onProgress && onProgress(50 + Math.round(((epoch + 1) / safeEpochs) * 50));
                onEpochEnd && onEpochEnd(epoch, {trainAcc, valAcc, trainLoss, valLoss});
            }
        }
    });

    xs.dispose();
    ys.dispose();

    /* ── Persist head via TF.js built-in IDB model saving ── */
    await head.save(`indexeddb://ml-head-${projectId}`);
    await idbPut('models', `head-labels:${projectId}`, JSON.stringify(labels));

    return wrapHead(head, labels);
};

/* ── Evaluate image classifier on training data → confusion matrix + per-class metrics ── */
export const evaluateImageModel = async (labels, trainingData, projectId, classifier, net, onProgress) => {
    if (!classifier || !net) return null;
    const total = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
    if (total === 0) return null;

    // confusion[actual][predicted] = count
    const confusion = {};
    const samples   = {};   // label → [{src, predicted, correct}]
    for (const l of labels) {
        confusion[l] = {};
        samples[l]   = [];
        for (const p of labels) confusion[l][p] = 0;
    }

    let done = 0;
    for (const actual of labels) {
        for (const ex of (trainingData[actual] || [])) {
            let src = ex.data && ex.data.startsWith('data:') ? ex.data : null;
            if (!src) src = await getImageFromIDB(projectId, ex.id);
            done++;
            onProgress && onProgress(Math.round(done / total * 100));
            if (!src) continue;

            const img = await loadImg(src);
            if (!img) continue;

            const logits   = net.infer(img, true);
            const res      = await classifier.predictClass(logits);
            logits.dispose();

            const predIdx  = res.classIndex;
            const predicted = labels[predIdx] || labels[0];
            confusion[actual][predicted] = (confusion[actual][predicted] || 0) + 1;
            samples[actual].push({src, predicted, correct: predicted === actual});
        }
    }

    // Compute per-class metrics
    const classMetrics = labels.map(l => {
        const tp = confusion[l][l] || 0;
        const fnCount = labels.reduce((s, p) => p !== l ? s + (confusion[l][p] || 0) : s, 0);
        const fp = labels.reduce((s, a) => a !== l ? s + (confusion[a][l] || 0) : s, 0);
        const total_l = tp + fnCount;
        const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
        const recall    = total_l > 0   ? tp / total_l    : 0;
        const accuracy  = total_l > 0   ? tp / total_l    : 0;
        return {label: l, accuracy, precision, recall, samples: total_l, tp, fp, fn: fnCount};
    });

    return {confusion, classMetrics, samples};
};

/* ── Load persisted image classifier ── */
export const loadImageClassifier = async (projectId, labels) => {
    try {
        const tf       = await getTF();
        const head     = await tf.loadLayersModel(`indexeddb://ml-head-${projectId}`);
        const lblRaw   = await idbGet('models', `head-labels:${projectId}`);
        const savedLbl = lblRaw ? JSON.parse(lblRaw) : (labels || []);
        return wrapHead(head, savedLbl);
    } catch (_) {
        return null;
    }
};

/* ── Delete all project data (images + head model) ── */
export const deleteProjectData = async projectId => {
    await _idbDeleteProject(projectId);
    await idbDelete('models', `head-labels:${projectId}`);
    try {
        const tf = await getTF();
        await tf.io.removeModel(`indexeddb://ml-head-${projectId}`);
    } catch (_) {}
};

/* ── Prediction ──
   Used by TestingPanel: predictImages(imgEl, classifier, mobileNet, labels)
   The mobileNet here is our wrapper object (has .infer()).
── */
export const predictImages = async (imgEl, classifier, net, labels) => {
    if (!classifier || !net) return [];
    const logits = net.infer(imgEl, true);     // [1, 256]  — wrapper handles preprocessing
    const res    = await classifier.predictClass(logits);
    logits.dispose();
    return labels.map((lbl, i) => ({
        label: lbl,
        prob:  ((res.confidences[String(i)] !== undefined ? res.confidences[String(i)] : res.confidences[i]) || 0) * 100
    }));
};

/* ── Re-export IDB helpers ── */
export { saveImageToIDB, getImageFromIDB, deleteProjectIDB } from './idb.js';

/* ════════════════════════════════════════════════════════════════
   Audio / Speech-Commands Engine
   Uses @tensorflow-models/speech-commands loaded from CDN as UMD.
   Transfer-learning pipeline (matches ML for Kids approach):
     1. Base recognizer  – pre-trained BROWSER_FFT speech-commands model
     2. Transfer recognizer – per-project transfer-learning head
     3. Persistence – tf.io IndexedDB via transferRecognizer.save/load
   ════════════════════════════════════════════════════════════════ */

const SPEECH_COMMANDS_CDN =
    'https://cdn.jsdelivr.net/npm/@tensorflow-models/speech-commands@0.5.4/dist/speech-commands.min.js';

let _scLib            = null;   // window.speechCommands
let _baseRecognizer   = null;   // singleton base recognizer
let _transferRec      = null;   // current project's transfer recognizer
let _soundProjectId   = null;
let _soundModelInfo   = null;   // { numFrames, fftSize }

const loadSpeechCommandsLib = async () => {
    if (_scLib) return _scLib;
    if (typeof window !== 'undefined' && window.speechCommands) {
        _scLib = window.speechCommands;
        return _scLib;
    }
    const tf = await getTF();
    // speech-commands UMD needs window.tf
    if (typeof window !== 'undefined') window.tf = tf;
    await new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = SPEECH_COMMANDS_CDN;
        s.onload = resolve;
        s.onerror = () => reject(new Error(
            'Could not load speech-commands library. Check your internet connection.'
        ));
        document.head.appendChild(s);
    });
    _scLib = window.speechCommands;
    if (!_scLib) throw new Error('speech-commands did not initialise correctly.');
    return _scLib;
};

/* ── Public: init base + transfer recognizer ── */
export const initSpeechCommands = async (projectId, onStatus) => {
    onStatus && onStatus('Loading speech-commands library…');
    const sc = await loadSpeechCommandsLib();

    if (!_baseRecognizer) {
        onStatus && onStatus('Loading audio base model…');
        _baseRecognizer = sc.create('BROWSER_FFT');
        await _baseRecognizer.ensureModelLoaded();
    }

    if (_soundProjectId !== projectId || !_transferRec) {
        _transferRec    = _baseRecognizer.createTransfer('project-' + projectId);
        _soundProjectId = projectId;
        const shape     = _transferRec.modelInputShape();
        _soundModelInfo = {numFrames: shape[1], fftSize: shape[2]};
    }

    onStatus && onStatus('');
    return {recognizer: _transferRec, modelInfo: _soundModelInfo};
};

export const getSoundModelInfo = () => _soundModelInfo;

/* ── Public: record one audio example (~1 second) ── */
export const collectAudioExample = async label => {
    if (!_transferRec) throw new Error('Audio engine not ready. Call initSpeechCommands first.');
    return _transferRec.collectExample(label);
};

/* ── Public: train transfer model from stored spectrogram data ── */
export const trainSounds = async (labels, trainingData, projectId, onStatus, onProgress, config = {}) => {
    const {epochs: rawEpochs = 50, batchSize: rawBatch = 32, onEpochEnd: epochCb} = config;
    const epochs    = Math.max(1, Math.round(rawEpochs));
    const batchSize = Math.max(1, Math.round(rawBatch));
    if (!_transferRec || !_soundModelInfo) throw new Error('Audio engine not initialised.');

    // Clear dataset and re-add all stored examples
    _transferRec.dataset.clear();
    try { _transferRec.dataset.label2Ids = {}; } catch (_) {}
    _transferRec.words = null;

    onStatus && onStatus('Loading audio samples…');
    let added = 0;
    for (const label of labels) {
        for (const sample of (trainingData[label] || [])) {
            if (!sample.spectrogramData) continue;
            _transferRec.dataset.addExample({
                label,
                spectrogram: {
                    frameSize: sample.frameSize || _soundModelInfo.fftSize,
                    data: new Float32Array(sample.spectrogramData)
                }
            });
            added++;
        }
    }
    if (added === 0) throw new Error('No audio samples found. Record some samples first.');

    _transferRec.collateTransferWords();

    onStatus && onStatus('Training audio model…');
    const valSplitAudio  = Math.min(0.5, Math.max(0.15, 1.0 / added));
    const trainCountAudio = Math.floor(added * (1 - valSplitAudio));
    const effectiveBsAudio = Math.max(1, Math.min(batchSize, trainCountAudio));
    await _transferRec.train({
        epochs,
        batchSize: effectiveBsAudio,
        validationSplit: valSplitAudio,
        windowHopRatio:  0.25,
        optimizer:       'sgd',
        callback: {
            onEpochEnd: async (epoch, logs) => {
                const pct = Math.round(((epoch + 1) / epochs) * 100);
                onProgress && onProgress(pct);
                const acc = logs.val_acc    !== undefined ? logs.val_acc
                          : logs.val_accuracy !== undefined ? logs.val_accuracy
                          : logs.acc          !== undefined ? logs.acc
                          : (logs.accuracy || 0);
                onStatus && onStatus(`Epoch ${epoch + 1}/${epochs}  acc: ${(acc * 100).toFixed(1)}%`);
                epochCb && epochCb(epoch, acc);
            }
        }
    });

    await _transferRec.save(`indexeddb://ml-sound-${projectId}`);
    await idbPut('models', `sound-labels:${projectId}`, JSON.stringify(labels));
    onStatus && onStatus('Training Complete');
    onProgress && onProgress(100);
};

/* ── Public: restore saved model ── */
export const loadSoundClassifier = async (projectId, labels) => {
    try {
        if (!_transferRec) return null;
        await _transferRec.load(`indexeddb://ml-sound-${projectId}`);
        _transferRec.words = Array.from(labels).sort();
        return _transferRec;
    } catch (_) {
        return null;
    }
};

/* ── Public: live mic classification ── */
export const startListening = async (callback, options = {}) => {
    if (!_transferRec) throw new Error('Audio engine not ready.');
    return _transferRec.listen(result => {
        const lbls = _transferRec.wordLabels();
        if (!lbls) return;
        const matches = lbls.map((lbl, i) => ({
            label: lbl,
            prob: (result.scores[i] || 0) * 100
        })).sort((a, b) => b.prob - a.prob);
        callback(matches);
    }, {probabilityThreshold: options.threshold || 0.5});
};

export const stopListening = async () => {
    if (_transferRec) {
        try { await _transferRec.stopListening(); } catch (_) {}
    }
};
