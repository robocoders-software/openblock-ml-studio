/* ── IndexedDB wrapper ───────────────────────────────────────────
   Stores:
     'images'  – key = "{projectId}:{exampleId}"  value = { data: dataUrl }
     'models'  – key = "knn:{projectId}"          value = serialized JSON string
   ─────────────────────────────────────────────────────────────── */
const DB_NAME    = 'openblock_ml_studio';
const DB_VERSION = 3;

let dbPromise = null;

const getDB = () => {
    if (dbPromise) return dbPromise;
    dbPromise = new Promise((resolve, reject) => {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = evt => {
            const db = evt.target.result;
            ['images', 'models', 'audio-samples', 'audio-thumbs', 'audio-blobs'].forEach(name => {
                if (!db.objectStoreNames.contains(name)) {
                    db.createObjectStore(name);
                }
            });
        };
        req.onsuccess = evt => resolve(evt.target.result);
        req.onerror  = ()  => { dbPromise = null; reject(req.error); };
    });
    return dbPromise;
};

export const idbGet = async (store, key) => {
    const db = await getDB();
    return new Promise((resolve, reject) => {
        const tx  = db.transaction(store, 'readonly');
        const req = tx.objectStore(store).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror   = () => reject(req.error);
    });
};

export const idbPut = async (store, key, value) => {
    const db = await getDB();
    return new Promise((resolve, reject) => {
        const tx  = db.transaction(store, 'readwrite');
        const req = tx.objectStore(store).put(value, key);
        req.onsuccess = () => resolve();
        req.onerror   = () => reject(req.error);
    });
};

export const idbDelete = async (store, key) => {
    const db = await getDB();
    return new Promise((resolve, reject) => {
        const tx  = db.transaction(store, 'readwrite');
        const req = tx.objectStore(store).delete(key);
        req.onsuccess = () => resolve();
        req.onerror   = () => reject(req.error);
    });
};

export const idbGetAllKeys = async (store, prefix) => {
    const db = await getDB();
    return new Promise((resolve, reject) => {
        const tx  = db.transaction(store, 'readonly');
        const req = tx.objectStore(store).getAllKeys();
        req.onsuccess = () => resolve(
            prefix ? req.result.filter(k => String(k).startsWith(prefix)) : req.result
        );
        req.onerror = () => reject(req.error);
    });
};

/* ── Convenience: save / load / delete all images for a project ── */
export const saveImageToIDB = async (projectId, id, dataUrl) => {
    await idbPut('images', `${projectId}:${id}`, { data: dataUrl });
};

export const getImageFromIDB = async (projectId, id) => {
    const rec = await idbGet('images', `${projectId}:${id}`);
    return rec ? rec.data : null;
};

export const loadProjectImages = async (projectId, trainingData) => {
    /* Returns a new trainingData object with image .data fields populated from IDB */
    const out = {};
    for (const [label, examples] of Object.entries(trainingData || {})) {
        out[label] = await Promise.all(
            (examples || []).map(async ex => {
                if (ex.type !== 'image') return ex;
                const data = await getImageFromIDB(projectId, ex.id);
                return { ...ex, data: data || '' };
            })
        );
    }
    return out;
};

export const deleteProjectIDB = async (projectId) => {
    const imageKeys = await idbGetAllKeys('images', `${projectId}:`);
    for (const k of imageKeys) await idbDelete('images', k);
    const audioKeys = await idbGetAllKeys('audio-samples', `${projectId}:`);
    for (const k of audioKeys) await idbDelete('audio-samples', k);
    const thumbKeys = await idbGetAllKeys('audio-thumbs', `${projectId}:`);
    for (const k of thumbKeys) await idbDelete('audio-thumbs', k);
    const blobKeys  = await idbGetAllKeys('audio-blobs', `${projectId}:`);
    for (const k of blobKeys) await idbDelete('audio-blobs', k);
    await idbDelete('models', `knn:${projectId}`);
    await idbDelete('models', `knn:text:${projectId}`);
    await idbDelete('models', `knn:numbers:${projectId}`);
    await idbDelete('models', `vocab:${projectId}`);
    await idbDelete('models', `sound-labels:${projectId}`);
};

/* ── Audio sample helpers ── */
export const saveAudioToIDB = async (projectId, id, spectrogramData, frameSize) => {
    await idbPut('audio-samples', `${projectId}:${id}`, {data: Array.from(spectrogramData), frameSize});
};

export const getAudioFromIDB = async (projectId, id) => {
    return idbGet('audio-samples', `${projectId}:${id}`);
};

export const saveAudioThumbToIDB = async (projectId, id, dataUrl) => {
    await idbPut('audio-thumbs', `${projectId}:${id}`, dataUrl);
};

export const getAudioThumbFromIDB = async (projectId, id) => {
    return idbGet('audio-thumbs', `${projectId}:${id}`);
};

/* ── Raw audio blob helpers (MediaRecorder output for playback / WAV export) ──
   Stores as {buffer: ArrayBuffer, type: string} rather than a raw Blob because
   Electron's IDB implementation deserializes Blobs as plain objects, breaking
   URL.createObjectURL on retrieval. ArrayBuffer round-trips correctly everywhere. */
export const saveAudioBlobToIDB = async (projectId, id, blob) => {
    const buffer = await blob.arrayBuffer();
    await idbPut('audio-blobs', `${projectId}:${id}`, {buffer, type: blob.type || 'audio/webm'});
};

export const getAudioBlobFromIDB = async (projectId, id) => {
    const stored = await idbGet('audio-blobs', `${projectId}:${id}`);
    if (!stored) return null;
    /* Reconstruct a proper Blob from the stored ArrayBuffer */
    return new Blob([stored.buffer], {type: stored.type || 'audio/webm'});
};

export const deleteAudioBlobFromIDB = async (projectId, id) => {
    await idbDelete('audio-blobs', `${projectId}:${id}`);
};

/* ── Bulk read helpers for persistence (project export) ── */
export const getAllProjectImages = async (projectId) => {
    const keys = await idbGetAllKeys('images', `${projectId}:`);
    const map = {};
    for (const k of keys) {
        const rec = await idbGet('images', k);
        const id  = k.slice(projectId.length + 1);
        if (rec) map[id] = rec.data;
    }
    return map;
};

export const getAllProjectAudio = async (projectId) => {
    const keys = await idbGetAllKeys('audio-samples', `${projectId}:`);
    const map = {};
    for (const k of keys) {
        const rec = await idbGet('audio-samples', k);
        const id  = k.slice(projectId.length + 1);
        if (rec) map[id] = rec;
    }
    return map;
};

export const loadProjectAudio = async (projectId, trainingData) => {
    const out = {};
    for (const [label, samples] of Object.entries(trainingData || {})) {
        out[label] = await Promise.all(
            (samples || []).map(async ex => {
                if (ex.type !== 'audio') return ex;
                const stored = await getAudioFromIDB(projectId, ex.id);
                if (!stored) return ex;
                return {...ex, spectrogramData: stored.data, frameSize: stored.frameSize};
            })
        );
    }
    return out;
};

/* ── KNN serialization helpers ── */
const serializeKNN = async (classifier) => {
    const dataset = classifier.getClassifierDataset();
    const serial  = {};
    for (const [k, tensor] of Object.entries(dataset)) {
        serial[k] = { data: Array.from(tensor.dataSync()), shape: tensor.shape };
    }
    return JSON.stringify(serial);
};

const deserializeKNN = (raw, tf, knnLib) => {
    const serial     = JSON.parse(raw);
    const classifier = knnLib.create();
    const dataset    = {};
    for (const [k, {data, shape}] of Object.entries(serial)) {
        dataset[k] = tf.tensor(data, shape);
    }
    classifier.setClassifierDataset(dataset);
    return classifier;
};

/* ── Image KNN model ── */
export const saveKNNToIDB = async (projectId, classifier, tf) => { // eslint-disable-line no-unused-vars
    await idbPut('models', `knn:${projectId}`, await serializeKNN(classifier));
};

export const loadKNNFromIDB = async (projectId, tf, knnLib) => {
    const raw = await idbGet('models', `knn:${projectId}`);
    if (!raw) return null;
    return deserializeKNN(raw, tf, knnLib);
};

/* ── Text KNN model + vocabulary ── */
export const saveTextModelToIDB = async (projectId, classifier, vocab) => {
    await idbPut('models', `knn:text:${projectId}`, await serializeKNN(classifier));
    await idbPut('models', `vocab:${projectId}`, JSON.stringify(vocab));
};

export const loadTextModelFromIDB = async (projectId, tf, knnLib) => {
    const rawKNN   = await idbGet('models', `knn:text:${projectId}`);
    const rawVocab = await idbGet('models', `vocab:${projectId}`);
    if (!rawKNN || !rawVocab) return null;
    const classifier = deserializeKNN(rawKNN, tf, knnLib);
    const vocab      = JSON.parse(rawVocab);
    return { classifier, vocab };
};

/* ── Numbers KNN model ── */
export const saveNumbersModelToIDB = async (projectId, classifier) => {
    await idbPut('models', `knn:numbers:${projectId}`, await serializeKNN(classifier));
};

export const loadNumbersModelFromIDB = async (projectId, tf, knnLib) => {
    const raw = await idbGet('models', `knn:numbers:${projectId}`);
    if (!raw) return null;
    return deserializeKNN(raw, tf, knnLib);
};
