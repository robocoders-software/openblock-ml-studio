/* ── ML Engine ────────────────────────────────────────────────────
   Transfer-learning pipeline:
     1. Backbone  – MobileNetV2 (frozen, loaded from @tensorflow-models/mobilenet)
     2. Head      – small trainable dense network (2 layers)
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
let _mobileNet  = null;
let _mobileNetP = null;

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

/* ── Public: MobileNetV2 feature extractor (cached singleton) ──
   Uses version 2 / alpha 1.0 for maximum embedding quality.
   Weights are downloaded from TF Hub on first use then browser-cached.
── */
export const getMobileNet = async onStatus => {
    if (_mobileNet) return _mobileNet;
    if (_mobileNetP) return _mobileNetP;

    _mobileNetP = (async () => {
        const tf = await getTF();  // eslint-disable-line no-unused-vars
        onStatus?.('Loading MobileNetV2 backbone…');
        const mod = await import(/* webpackChunkName: "mobilenet" */ '@tensorflow-models/mobilenet');
        const lib = mod.default || mod;
        const net = await lib.load({version: 2, alpha: 1.0});
        _mobileNet  = net;
        _mobileNetP = null;
        onStatus?.('MobileNetV2 ready');
        return net;
    })();

    return _mobileNetP;
};

/* ── Internal: extract embedding → shape [1, embDim] (caller must dispose) ── */
const extractEmbedding = (tf, net, imgEl) => {
    const raw      = net.infer(imgEl, true);              // e.g. [1,1280] or [1,1,1,1280]
    const last     = raw.shape[raw.shape.length - 1];
    const reshaped = tf.reshape(raw, [1, last]);
    const out      = reshaped.clone();                    // independent buffer
    raw.dispose();
    reshaped.dispose();
    return out;
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
        activation: 'linear',              // raw logits, softmax applied in loss
        kernelInitializer: 'varianceScaling'
    }));
    return model;
};

/* ── Internal: wrap a trained head so its interface matches KNN predictClass ── */
const wrapHead = (head, labels) => ({
    _head: head,
    async predictClass (logits) {
        // head outputs raw logits — apply softmax to get probabilities
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
            label:       labels[topI] ?? String(topI),
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

    const total   = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
    if (total < 2) throw new Error('Need at least 2 training images total.');
    if (!labels.every(l => (trainingData[l] || []).length >= 1)) {
        throw new Error('Every class needs at least 1 image.');
    }

    /* ── Phase 1 of 2: feature extraction (0-50 %) ── */
    onStatus?.('Extracting MobileNetV2 features…');
    const embeddings = [];
    const labelIdxs  = [];
    let done = 0;

    for (let i = 0; i < labels.length; i++) {
        for (const ex of (trainingData[labels[i]] || [])) {
            let src = ex.data && ex.data.startsWith('data:') ? ex.data : null;
            if (!src) src = await getImageFromIDB(projectId, ex.id);
            if (!src) { done++; onProgress?.(Math.round(done / total * 50)); continue; }

            const img = await loadImg(src);
            if (!img) { done++; onProgress?.(Math.round(done / total * 50)); continue; }

            const emb = extractEmbedding(tf, net, img);
            embeddings.push(emb);
            labelIdxs.push(i);

            done++;
            onProgress?.(Math.round(done / total * 50));
        }
    }

    if (embeddings.length === 0) throw new Error('No valid training images could be loaded.');

    /* ── Stack into training tensors ── */
    const xs = tf.concat(embeddings, 0);      // [N, embDim]
    embeddings.forEach(e => e.dispose());
    const embDim = xs.shape[1];

    const ys = tf.oneHot(tf.tensor1d(labelIdxs, 'int32'), labels.length); // [N, C]

    /* ── Phase 2 of 2: train head (50-100 %) ── */
    onStatus?.('Building & training classification head…');
    const head = buildHead(tf, embDim, labels.length);
    head.compile({
        optimizer: tf.train.sgd(learningRate),
        loss:      (labels, logits) =>
            tf.losses.softmaxCrossEntropy(labels, logits, undefined, 0.1), // label smoothing 0.1
        metrics:   ['accuracy']
    });

    const useVal       = xs.shape[0] >= 8;
    const valSplit     = useVal ? 0.15 : 0;
    const effectiveBs  = Math.min(batchSize, xs.shape[0]);

    await head.fit(xs, ys, {
        epochs,
        batchSize: effectiveBs,
        shuffle:   true,
        validationSplit: valSplit,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                const acc  = useVal ? (logs.val_acc ?? logs.val_accuracy ?? logs.acc ?? logs.accuracy ?? 0)
                                    : (logs.acc ?? logs.accuracy ?? 0);
                const loss = logs.loss ?? 0;
                onStatus?.(`Epoch ${epoch + 1}/${epochs}  acc: ${(acc * 100).toFixed(1)}%  loss: ${loss.toFixed(4)}`);
                onProgress?.(50 + Math.round(((epoch + 1) / epochs) * 50));
                onEpochEnd?.(epoch, acc);
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

/* ── Prediction ── */
export const predictImages = async (imgEl, classifier, net, labels) => {
    if (!classifier || !net) return [];
    const tf     = await getTF();
    const logits = extractEmbedding(tf, net, imgEl);
    const res    = await classifier.predictClass(logits);
    logits.dispose();
    return labels.map((lbl, i) => ({
        label: lbl,
        prob:  ((res.confidences[String(i)] ?? res.confidences[i]) || 0) * 100
    }));
};

/* ── Re-export IDB helpers ── */
export { saveImageToIDB, getImageFromIDB, deleteProjectIDB } from './idb.js';
