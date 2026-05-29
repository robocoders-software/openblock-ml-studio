/* ── ML Engine ────────────────────────────────────────────────────
   Transfer-learning pipeline:
     1. Backbone  – MobileNetV1 α=0.25, loaded via Electron custom protocol
                    robocoders-resource://models/mobilenet_v1_0.25_224/model.json
                    Files live in external-resources/ and are served directly from
                    disk — no HTTP server, no port, works offline indefinitely.
     2. Head      – small trainable dense network (dense-relu → dropout → dense)
     3. Persistence – TF.js built-in IndexedDB model saving
     4. Interface – classifier.predictClass(logits) is KNN-compatible so the
                    VM extension works unchanged
   ─────────────────────────────────────────────────────────────── */
import {
    makeFSModelHandler,
    fsWriteFile, fsReadFile, fsDeleteProject,
    getImageFromFS, saveImageToFS
} from './ml-fs.js';

/* ── Singletons ── */
let _tf         = null;
let _mobileNet  = null;   // the wrapped backbone object (has .infer())
let _mobileNetP = null;

/* ── Local model: loaded via Electron custom protocol (no HTTP server required).
   Files live in external-resources/models/mobilenet_v1_0.25_224/
   Run  scripts/download-mobilenet.js  once to populate that directory.
── */
const LOCAL_MOBILENET_URL  = 'robocoders-resource://models/mobilenet_v1_0.25_224/model.json';
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
                `Failed to load MobileNetV1 (${LOCAL_MOBILENET_URL}). ` +
                'Make sure external-resources/models/mobilenet_v1_0.25_224/ is present in the app directory. ' +
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
   Intentionally minimal: Dropout → Dense(logits).
   No hidden Dense(128) layer — with small datasets (5-20 images) that layer
   introduces ~32k extra parameters that overfit badly. ML for Kids and
   Teachable Machine both use this simpler structure.
── */
const buildHead = (tf, inputDim, numClasses) => {
    const model = tf.sequential();
    model.add(tf.layers.dropout({inputShape: [inputDim], rate: 0.2}));
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
    {epochs = 50, batchSize = 32, learningRate = 0.001, onEpochEnd} = {}
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
            if (!src) src = await getImageFromFS(projectId, ex.id);
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
        // Adam converges faster than plain SGD for small datasets; momentum is baked in.
        optimizer: tf.train.adam(learningRate),
        loss:      (ys_, logits) =>
            tf.losses.softmaxCrossEntropy(ys_, logits, undefined, 0.1),
        metrics:   ['accuracy']
    });

    const N          = xs.shape[0];
    const safeEpochs = Math.max(1, Math.round(epochs));
    // With fewer than 20 total images, a validation split wastes too many training examples.
    // ML for Kids also skips validation and trains on the full set.
    const valSplit    = N >= 20 ? Math.min(0.2, Math.max(0.1, 2.0 / N)) : 0;
    const trainCount  = valSplit > 0 ? Math.floor(N * (1 - valSplit)) : N;
    const effectiveBs = Math.max(1, Math.min(Math.max(1, Math.round(batchSize)), trainCount));

    await head.fit(xs, ys, {
        epochs:          safeEpochs,
        batchSize:       effectiveBs,
        shuffle:         true,
        validationSplit: valSplit,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const valAcc    = logs.val_acc   || logs.val_accuracy || 0;
                const trainAcc  = logs.acc       || logs.accuracy     || 0;
                const valLoss   = logs.val_loss  || 0;
                const trainLoss = logs.loss      || 0;
                const accLabel  = valSplit > 0 ? `val_acc: ${(valAcc * 100).toFixed(1)}%` : `acc: ${(trainAcc * 100).toFixed(1)}%`;
                onStatus && onStatus(`Epoch ${epoch + 1}/${safeEpochs}  ${accLabel}  loss: ${trainLoss.toFixed(4)}`);
                onProgress && onProgress(50 + Math.round(((epoch + 1) / safeEpochs) * 50));
                onEpochEnd && onEpochEnd(epoch, {trainAcc, valAcc, trainLoss, valLoss});
            }
        }
    });

    xs.dispose();
    ys.dispose();

    /* ── Persist head to filesystem ── */
    await head.save(makeFSModelHandler(projectId, 'image-model'));
    await fsWriteFile(projectId, 'image-model/labels.json', JSON.stringify(labels));

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
            if (!src) src = await getImageFromFS(projectId, ex.id);
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
        const head     = await tf.loadLayersModel(makeFSModelHandler(projectId, 'image-model'));
        const lblRaw   = await fsReadFile(projectId, 'image-model/labels.json');
        const savedLbl = lblRaw
            ? JSON.parse(Buffer.from(lblRaw).toString('utf8'))
            : (labels || []);
        return wrapHead(head, savedLbl);
    } catch (_) {
        return null;
    }
};

/* ── Delete all project data ── */
export const deleteProjectData = async projectId => {
    await fsDeleteProject(projectId);
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

/* ── Re-export FS helpers so callers can import from ml-engine ── */
export { saveImageToFS, getImageFromFS, fsDeleteProject as deleteProjectFS } from './ml-fs.js';

/* ════════════════════════════════════════════════════════════════
   Audio / Speech-Commands Engine
   Fully offline — all assets served via robocoders-resource://.
   Run  scripts/download-speech-commands.js  once to populate:
     external-resources/libs/speech-commands.min.js
     external-resources/models/speech-commands-browser-fft/
   Transfer-learning pipeline (matches ML for Kids approach):
     1. Base recognizer  – pre-trained BROWSER_FFT speech-commands model
     2. Transfer recognizer – per-project transfer-learning head
     3. Persistence – tf.io IndexedDB via transferRecognizer.save/load
   ════════════════════════════════════════════════════════════════ */

// speech-commands.min.js loaded via <script> tag — custom protocol works fine here.
const SPEECH_COMMANDS_URL = 'robocoders-resource://libs/speech-commands.min.js';

const _getIpc = () => {
    try { return window.require('electron').ipcRenderer; } catch (_) { return null; }
};

// speech-commands validates URL scheme internally (only accepts http/https/file) and uses
// browser fetch for http URLs. The resource server at port 20112 is LOCAL — it serves the
// bundled files from disk with no internet required. This is the only reliable approach for
// this library since passing raw ModelArtifacts via IPC causes binary serialization issues
// that corrupt the model and break transfer-learning layer freezing.
const BROWSER_FFT_MODEL_URL = 'http://localhost:20112/models/speech-commands-browser-fft/model.json';
const BROWSER_FFT_META_URL  = 'http://localhost:20112/models/speech-commands-browser-fft/metadata.json';

let _scLib            = null;   // window.speechCommands
let _baseRecognizer   = null;   // singleton base recognizer
let _transferRec      = null;   // current project's transfer recognizer
let _soundProjectId   = null;
let _soundModelInfo   = null;   // { numFrames, fftSize }

const _loadScriptTag = url => new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
});

const loadSpeechCommandsLib = async () => {
    if (_scLib) return _scLib;
    if (typeof window !== 'undefined' && window.speechCommands) {
        _scLib = window.speechCommands;
        return _scLib;
    }
    const tf = await getTF();
    // speech-commands UMD needs window.tf exposed globally before the script runs
    if (typeof window !== 'undefined') window.tf = tf;

    // 1. Try loading via the custom protocol — up to 3 attempts with 800ms gaps.
    //    Transient failures happen on first run (slow disk, protocol handler not warm)
    //    and are transparently recovered here without showing the error overlay.
    let loaded = false;
    for (let i = 0; i < 3; i++) {
        if (i > 0) await new Promise(r => setTimeout(r, 800));
        try {
            await _loadScriptTag(SPEECH_COMMANDS_URL);
            loaded = true;
            break;
        } catch (_) { /* try again */ }
    }

    // 2. Protocol consistently failed — fall back to IPC file read + blob URL.
    //    This bypasses the custom protocol entirely and is immune to protocol
    //    registration timing issues.
    if (!loaded) {
        const ipc = _getIpc();
        if (ipc) {
            try {
                const buf = await ipc.invoke('read-external-resource', 'libs/speech-commands.min.js');
                if (buf) {
                    const blob   = new Blob([buf], {type: 'text/javascript'});
                    const blobUrl = URL.createObjectURL(blob);
                    try {
                        await _loadScriptTag(blobUrl);
                        loaded = true;
                    } finally {
                        URL.revokeObjectURL(blobUrl);
                    }
                }
            } catch (_) { /* fall through to error below */ }
        }
    }

    if (!loaded) {
        throw new Error(
            'Could not load speech-commands library from local resources. ' +
            'Make sure external-resources/libs/speech-commands.min.js exists in the app directory.'
        );
    }

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
        const ipc = _getIpc();
        // Retry up to 3× — the resource server may still be starting on first launch.
        let lastErr = null;
        for (let attempt = 0; attempt < 3; attempt++) {
            try {
                if (ipc) await ipc.invoke('ensure-resource-server').catch(() => {});
                const rec = sc.create('BROWSER_FFT', undefined, BROWSER_FFT_MODEL_URL, BROWSER_FFT_META_URL);
                await rec.ensureModelLoaded();
                _baseRecognizer = rec;
                lastErr = null;
                break;
            } catch (e) {
                lastErr = e;
                _baseRecognizer = null;
                if (attempt < 2) {
                    onStatus && onStatus(`Retrying audio model load (${attempt + 1}/3)…`);
                    await new Promise(r => setTimeout(r, 1500));
                }
            }
        }
        if (lastErr) throw lastErr;
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

/* ── Internal: adaptive training config (mirrors ML for Kids' _prepareTrainingConfig) ──
   tiny:    avg < 15 samples/class OR total < 30  → 80 epochs, bs 64,  no val split, no augment
   default: avg 15–50 samples/class               → 50 epochs, bs 128, val 0.15, augment 0.3
   huge:    avg > 50 samples/class                → 40 epochs, bs 128, val 0.20, augment 0.3
── */
const _prepareSoundTrainingConfig = (totalSamples, numLabels, userEpochs, userBatch) => {
    const avg = totalSamples / Math.max(1, numLabels);
    const tiny    = avg < 15 || totalSamples < 30;
    const huge    = avg > 50;

    if (tiny) {
        return {
            epochs:                   userEpochs || 80,
            batchSize:                userBatch  || 64,
            validationSplit:          null,
            windowHopRatio:           0.25,
            augmentByMixingNoiseRatio: null,
            fineTuningEpochs:         null,
            optimizer:                'sgd'
        };
    }
    if (huge) {
        return {
            epochs:                   userEpochs || 40,
            batchSize:                userBatch  || 128,
            validationSplit:          0.2,
            windowHopRatio:           0.25,
            augmentByMixingNoiseRatio: 0.3,
            fineTuningEpochs:         12,
            optimizer:                'sgd'
        };
    }
    return {
        epochs:                   userEpochs || 50,
        batchSize:                userBatch  || 128,
        validationSplit:          0.15,
        windowHopRatio:           0.25,
        augmentByMixingNoiseRatio: 0.3,
        fineTuningEpochs:         15,
        optimizer:                'sgd'
    };
};

/* ── Public: train transfer model from stored spectrogram data ── */
export const trainSounds = async (labels, trainingData, projectId, onStatus, onProgress, config = {}) => {
    const {epochs: rawEpochs, batchSize: rawBatch, onEpochEnd: epochCb} = config;
    if (!_baseRecognizer || !_soundModelInfo) throw new Error('Audio engine not initialised.');

    /* Always create a fresh transfer recognizer before training.
       Reusing the same instance across multiple train() calls corrupts the
       internal layer state of the speech-commands library.
       A unique name suffix is required because the library keeps a global
       registry — reusing the same name throws "already exists". */
    const transferName = `project-${projectId}-${Date.now()}`;
    _transferRec    = _baseRecognizer.createTransfer(transferName);
    _soundProjectId = projectId;

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

    const tc = _prepareSoundTrainingConfig(added, labels.length, rawEpochs, rawBatch);
    const totalEpochs = tc.epochs + (tc.fineTuningEpochs || 0);

    onStatus && onStatus('Training audio model…');

    const trainCfg = {
        epochs:          tc.epochs,
        batchSize:       tc.batchSize,
        windowHopRatio:  tc.windowHopRatio,
        optimizer:       tc.optimizer,
        callback: {
            onEpochEnd: async (epoch, logs) => {
                const pct = Math.round(((epoch + 1) / totalEpochs) * 100);
                onProgress && onProgress(pct);
                const acc = logs.val_acc       !== undefined ? logs.val_acc
                          : logs.val_accuracy  !== undefined ? logs.val_accuracy
                          : logs.acc           !== undefined ? logs.acc
                          : (logs.accuracy || 0);
                onStatus && onStatus(`Epoch ${epoch + 1}/${tc.epochs}  acc: ${(acc * 100).toFixed(1)}%`);
                epochCb && epochCb(epoch, acc);
            }
        }
    };
    if (tc.validationSplit !== null) trainCfg.validationSplit = tc.validationSplit;
    if (tc.augmentByMixingNoiseRatio !== null) trainCfg.augmentByMixingNoiseRatio = tc.augmentByMixingNoiseRatio;
    if (tc.fineTuningEpochs !== null) trainCfg.fineTuningEpochs = tc.fineTuningEpochs;

    await _transferRec.train(trainCfg);

    await _transferRec.save(makeFSModelHandler(projectId, 'audio-model'));
    await fsWriteFile(projectId, 'audio-model/labels.json', JSON.stringify(labels));
    onStatus && onStatus('Training Complete');
    onProgress && onProgress(100);
};

/* ── Public: restore saved model ── */
export const loadSoundClassifier = async (projectId, labels) => {
    try {
        if (!_transferRec) return null;
        await _transferRec.load(makeFSModelHandler(projectId, 'audio-model'));
        _transferRec.words = Array.from(labels).sort();
        return _transferRec;
    } catch (_) {
        return null;
    }
};

/* ── Public: live mic classification ── */
export const startListening = async (callback, options = {}) => {
    if (!_transferRec) throw new Error('Audio engine not ready.');
    // Use 0.0 threshold so ALL predictions always fire (scores always sum to 1).
    // Callers that want to suppress low-confidence results should filter in the callback.
    // ML for Kids uses 0.70 for blocks (high confidence gate), but the testing
    // panel needs all scores to always update the bars.
    const threshold = options.threshold !== undefined ? options.threshold : 0.0;
    return _transferRec.listen(result => {
        const lbls = _transferRec.wordLabels();
        if (!lbls) return;
        const matches = lbls.map((lbl, i) => ({
            label: lbl,
            prob: (result.scores[i] || 0) * 100
        })).sort((a, b) => b.prob - a.prob);
        callback(matches);
    }, {probabilityThreshold: threshold});
};

export const stopListening = async () => {
    if (_transferRec) {
        try { await _transferRec.stopListening(); } catch (_) {}
    }
};

/* ═════════════════════════════════════════════════════════════════
   Model artifact export / import (used by project-persistence.js)
   Artifacts: { modelJson: string, weightData: ArrayBuffer }
   weightData is a raw binary ArrayBuffer — keep it out of JSON so
   it survives IPC via the structured-clone algorithm without base64
   inflation when possible; callers that must JSON-serialise it can
   call bufferToB64 / b64ToBuffer themselves.
   ═════════════════════════════════════════════════════════════════ */

/* Capture a TF.js LayersModel's topology + weights into plain objects. */
const _captureModelArtifacts = model =>
    new Promise((resolve, reject) => {
        model.save({
            save: async artifacts => {
                resolve({
                    modelJson: JSON.stringify({
                        modelTopology:    artifacts.modelTopology,
                        weightsManifest: [{
                            paths:   ['model.weights.bin'],
                            weights: artifacts.weightSpecs
                        }]
                    }),
                    weightData: artifacts.weightData   // ArrayBuffer
                });
                return {modelArtifactsInfo: {dateSaved: new Date(), modelTopologyType: 'JSON'}};
            }
        }).catch(reject);
    });

/* Restore a TF.js LayersModel from captured artifacts. */
const _modelFromArtifacts = async (modelJsonStr, weightData) => {
    const tf      = await getTF();
    const parsed  = JSON.parse(modelJsonStr);
    const buffer  = weightData instanceof ArrayBuffer ? weightData
        : (weightData && weightData.buffer) ? weightData.buffer
        : weightData;
    return tf.loadLayersModel(tf.io.fromMemory(
        {
            modelTopology: parsed.modelTopology,
            weightSpecs:   parsed.weightsManifest[0].weights,
            weightData:    buffer
        }
    ));
};

/* ── Image head export / import ── */
export const exportTrainedHead = head => _captureModelArtifacts(head);

export const loadHeadFromArtifacts = async (modelJsonStr, weightData, labels) => {
    const head = await _modelFromArtifacts(modelJsonStr, weightData);
    return wrapHead(head, labels);
};

/* ── Audio (speech-commands transfer) export / import ── */

/* Export the currently trained transfer recognizer.
   Returns null if no model has been trained/saved yet. */
export const exportAudioModelArtifacts = async () => {
    if (!_soundProjectId) return null;
    const tf = await getTF();
    let model;
    try {
        model = await tf.loadLayersModel(`indexeddb://ml-sound-${_soundProjectId}`);
    } catch (_) {
        return null;
    }
    try {
        return await _captureModelArtifacts(model);
    } finally {
        model.dispose();
    }
};

/* Load audio model from artifacts back into the speech-commands pipeline.
   Saves the restored LayersModel to the FS so loadSoundClassifier() can pick it up.
   Returns true on success. */
export const loadAudioModelFromArtifacts = async (modelJsonStr, weightData, labels) => {
    const projectId = _soundProjectId;
    if (!projectId) return false;
    try {
        const model = await _modelFromArtifacts(modelJsonStr, weightData);
        await model.save(makeFSModelHandler(projectId, 'audio-model'));
        model.dispose();
        await fsWriteFile(projectId, 'audio-model/labels.json', JSON.stringify(labels));
        return true;
    } catch (e) {
        console.warn('[ml-engine] loadAudioModelFromArtifacts failed:', e.message);
        return false;
    }
};

/* ═════════════════════════════════════════════════════════════════
   TEXT CLASSIFICATION ENGINE  (Naive Bayes + Porter stemmer)
   Zero GPU dependency, zero model downloads, instantaneous training.
   Uses the same bayes-classifier package as PictoBlox:
     addDocument(text, label) → tokenise+stem → train() → classify()
   Model state is a plain JSON object — no weights.bin, no TF.js IO.
   ═════════════════════════════════════════════════════════════════ */

let _bayesClassifier = null;
let _textLabels      = null;

const _loadBayes = () =>
    import(/* webpackChunkName: "bayes" */ 'bayes-classifier')
        .then(m => m.default || m);

export const trainText = async (labels, trainingData, projectId, onStatus, onProgress) => {
    const total = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
    if (total < 2)
        throw new Error('Need at least 2 training examples total.');
    if (!labels.every(l => (trainingData[l] || []).length >= 1))
        throw new Error('Every class needs at least 1 example.');

    onProgress && onProgress(10);
    onStatus && onStatus('Training text classifier…');

    const BayesClassifier = await _loadBayes();
    const classifier = new BayesClassifier();

    for (const lbl of labels) {
        for (const ex of (trainingData[lbl] || [])) {
            const text = (ex.text || '').trim();
            if (text) classifier.addDocument(text, lbl);
        }
    }

    classifier.train();

    onProgress && onProgress(80);

    /* Persist: plain JSON state — no binary weights file required */
    const state = {
        docs:          classifier.docs,
        lastAdded:     classifier.docs.length,
        features:      classifier.features,
        classFeatures: classifier.classFeatures,
        classTotals:   classifier.classTotals,
        totalExamples: classifier.totalExamples,
        smoothing:     classifier.smoothing
    };
    await fsWriteFile(projectId, 'text-model/classifier.json',
        JSON.stringify({state, labels}));

    _bayesClassifier = classifier;
    _textLabels      = labels;

    onStatus && onStatus('Training Complete');
    onProgress && onProgress(100);
    return {classifier, labels};
};

/* ── Public: classify a text string using the Naive Bayes model ── */
export const classifyText = async text => {
    if (!_bayesClassifier)
        throw new Error('No text model loaded. Train a model first.');

    const clsns = _bayesClassifier.getClassifications((text || '').trim());
    const labels = _textLabels;

    /* getClassifications returns raw probability scores; normalise to sum=1 */
    const sum = clsns.reduce((s, c) => s + c.value, 0) || 1;
    const confidences = {};
    let topI = 0;
    labels.forEach((lbl, i) => {
        const c = clsns.find(x => x.label === lbl);
        confidences[String(i)] = c ? c.value / sum : 0;
        if (confidences[String(i)] > (confidences[String(topI)] || 0)) topI = i;
    });

    return {label: labels[topI] || '', classIndex: topI, confidences};
};

/* ── Text model export / import (JSON — no binary weights) ── */
export const exportTextModelArtifacts = async () => {
    if (!_bayesClassifier || !_textLabels) return null;
    const state = {
        docs:          _bayesClassifier.docs,
        lastAdded:     _bayesClassifier.docs.length,
        features:      _bayesClassifier.features,
        classFeatures: _bayesClassifier.classFeatures,
        classTotals:   _bayesClassifier.classTotals,
        totalExamples: _bayesClassifier.totalExamples,
        smoothing:     _bayesClassifier.smoothing
    };
    return {modelJson: JSON.stringify({state, labels: _textLabels}), weightData: null};
};

export const loadTextModelFromArtifacts = async (modelJsonStr, _weightData, labels, projectId) => {
    try {
        const BayesClassifier = await _loadBayes();
        const {state, labels: savedLbls} = JSON.parse(modelJsonStr);
        const classifier = new BayesClassifier();
        classifier.restore(state);
        const effectiveLabels = labels || savedLbls;
        if (projectId) {
            await fsWriteFile(projectId, 'text-model/classifier.json',
                JSON.stringify({state, labels: effectiveLabels}));
        }
        _bayesClassifier = classifier;
        _textLabels      = effectiveLabels;
        return true;
    } catch (e) {
        console.warn('[ml-engine] loadTextModelFromArtifacts failed:', e.message);
        return false;
    }
};

/* ── Public: restore saved text classifier from filesystem ── */
export const loadTextClassifier = async (projectId, labels) => {
    try {
        const raw = await fsReadFile(projectId, 'text-model/classifier.json');
        if (!raw) return null;
        const BayesClassifier = await _loadBayes();
        const {state, labels: savedLbls} = JSON.parse(
            Buffer.from(raw).toString('utf8')
        );
        const classifier = new BayesClassifier();
        classifier.restore(state);
        _bayesClassifier = classifier;
        _textLabels      = labels || savedLbls;
        return {classifier, labels: _textLabels};
    } catch (_) {
        return null;
    }
};
