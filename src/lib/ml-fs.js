/* ── ML Filesystem Storage ─────────────────────────────────────────
   Every ML project is stored under:
     <userData>/ml-projects/<projectId>/
       images/<id>               – image data-URL (UTF-8 text)
       audio/<id>.json           – {data:[...], frameSize} (UTF-8 text)
       thumbs/<id>               – waveform thumbnail data-URL (UTF-8 text)
       blobs/<id>.meta           – MIME type string
       blobs/<id>.bin            – raw audio blob binary
       image-model/model.json    – TF.js topology
       image-model/weights.bin   – TF.js weight binary
       image-model/labels.json   – label array
       audio-model/model.json
       audio-model/weights.bin
       audio-model/labels.json
       project.json              – {id, name, type, labels, disabledLabels, trained}

   All main-process file operations go through IPC so the renderer
   never touches the filesystem directly.
   ─────────────────────────────────────────────────────────────── */

const getIpc = () => {
    try { return window.require('electron').ipcRenderer; } catch (_) { return null; }
};

/* ── Raw IPC wrappers ── */
const writeRaw = (projectId, rel, data) =>
    getIpc().invoke('ml-write-file', projectId, rel, data);

const readRaw = (projectId, rel) =>
    getIpc().invoke('ml-read-file', projectId, rel);

const delRaw = (projectId, rel) =>
    getIpc().invoke('ml-delete-file', projectId, rel);

const listRaw = (projectId, sub) =>
    getIpc().invoke('ml-list-files', projectId, sub || '');

/* ── Typed read helpers ── */
const readText = async (projectId, rel) => {
    const data = await readRaw(projectId, rel);
    if (!data) return null;
    return typeof data === 'string' ? data : Buffer.from(data).toString('utf8');
};

const readBinary = async (projectId, rel) => {
    const data = await readRaw(projectId, rel);
    if (!data) return null;
    return Buffer.isBuffer(data) ? data : Buffer.from(data);
};

/* ── Public: generic file ops (used by project-persistence.js) ── */
export const fsWriteFile = (projectId, rel, data) => writeRaw(projectId, rel, data);
export const fsReadFile  = (projectId, rel)       => readRaw(projectId, rel);
export const fsDeleteProject = (projectId)        => getIpc().invoke('ml-delete-project', projectId);

/* ══════════════════════════════════════════════════════════════
   Image helpers
══════════════════════════════════════════════════════════════ */
export const saveImageToFS = (projectId, id, dataUrl) =>
    writeRaw(projectId, `images/${id}`, dataUrl);

export const getImageFromFS = async (projectId, id) =>
    readText(projectId, `images/${id}`);

export const deleteImageFromFS = (projectId, id) =>
    delRaw(projectId, `images/${id}`).catch(() => {});

export const getAllImagesFromFS = async (projectId) => {
    const files = await listRaw(projectId, 'images');
    const map   = {};
    for (const f of (files || [])) {
        const data = await readText(projectId, `images/${f}`);
        if (data) map[f] = data;
    }
    return map;
};

/* Enriches trainingData (meta-only) with .data fields from FS */
export const loadProjectImagesFromFS = async (projectId, trainingData) => {
    const out = {};
    for (const [label, examples] of Object.entries(trainingData || {})) {
        out[label] = await Promise.all(
            (examples || []).map(async ex => {
                if (ex.type !== 'image') return ex;
                const data = await getImageFromFS(projectId, ex.id);
                return {...ex, data: data || ''};
            })
        );
    }
    return out;
};

/* ══════════════════════════════════════════════════════════════
   Audio sample helpers (spectrogram Float32Array)
══════════════════════════════════════════════════════════════ */
export const saveAudioToFS = (projectId, id, spectrogramData, frameSize) =>
    writeRaw(projectId, `audio/${id}.json`,
        JSON.stringify({data: Array.from(spectrogramData), frameSize}));

export const getAudioFromFS = async (projectId, id) => {
    const text = await readText(projectId, `audio/${id}.json`);
    if (!text) return null;
    return JSON.parse(text);   // {data, frameSize}
};

export const getAllAudioFromFS = async (projectId) => {
    const files = await listRaw(projectId, 'audio');
    const map   = {};
    for (const f of (files || []).filter(f => f.endsWith('.json'))) {
        const id   = f.slice(0, -5);
        const text = await readText(projectId, `audio/${f}`);
        if (text) map[id] = JSON.parse(text);
    }
    return map;
};

/* Enriches trainingData (meta-only) with .spectrogramData/.frameSize fields */
export const loadProjectAudioFromFS = async (projectId, trainingData) => {
    const out = {};
    for (const [label, samples] of Object.entries(trainingData || {})) {
        out[label] = await Promise.all(
            (samples || []).map(async ex => {
                if (ex.type !== 'audio') return ex;
                const stored = await getAudioFromFS(projectId, ex.id);
                if (!stored) return ex;
                return {...ex, spectrogramData: stored.data, frameSize: stored.frameSize};
            })
        );
    }
    return out;
};

/* ══════════════════════════════════════════════════════════════
   Waveform thumbnail helpers
══════════════════════════════════════════════════════════════ */
export const saveAudioThumbToFS = (projectId, id, dataUrl) =>
    writeRaw(projectId, `thumbs/${id}`, dataUrl);

export const getAudioThumbFromFS = (projectId, id) =>
    readText(projectId, `thumbs/${id}`);

/* ══════════════════════════════════════════════════════════════
   Raw audio blob helpers (MediaRecorder output for playback / WAV export)
   Stored as {meta: mimeType string, bin: Buffer} to avoid Electron
   IDB blob-deserialisation issues — filesystem has no such problem.
══════════════════════════════════════════════════════════════ */
export const saveAudioBlobToFS = async (projectId, id, blob) => {
    const buffer = await blob.arrayBuffer();
    await writeRaw(projectId, `blobs/${id}.meta`, blob.type || 'audio/webm');
    await writeRaw(projectId, `blobs/${id}.bin`,  Buffer.from(buffer));
};

export const getAudioBlobFromFS = async (projectId, id) => {
    const [meta, bin] = await Promise.all([
        readText(projectId, `blobs/${id}.meta`),
        readBinary(projectId, `blobs/${id}.bin`)
    ]);
    if (!bin) return null;
    const type = meta || 'audio/webm';
    /* ArrayBuffer from Buffer for Blob constructor */
    const ab = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
    return new Blob([ab], {type});
};

export const deleteAudioBlobFromFS = async (projectId, id) => {
    await delRaw(projectId, `blobs/${id}.meta`).catch(() => {});
    await delRaw(projectId, `blobs/${id}.bin`).catch(() => {});
};

/* ══════════════════════════════════════════════════════════════
   TF.js IOHandler — reads/writes model.json + weights.bin via IPC.
   Usage:
     await model.save(makeFSModelHandler(projectId, 'image-model'));
     const m = await tf.loadLayersModel(makeFSModelHandler(projectId, 'image-model'));
     await transferRec.save(makeFSModelHandler(projectId, 'audio-model'));
     await transferRec.load(makeFSModelHandler(projectId, 'audio-model'));
══════════════════════════════════════════════════════════════ */
export const makeFSModelHandler = (projectId, subDir) => ({
    save: async artifacts => {
        const modelJson = JSON.stringify({
            modelTopology:    artifacts.modelTopology,
            weightsManifest: [{paths: ['weights.bin'], weights: artifacts.weightSpecs}]
        });
        await writeRaw(projectId, `${subDir}/model.json`, modelJson);
        await writeRaw(projectId, `${subDir}/weights.bin`, Buffer.from(artifacts.weightData));
        return {modelArtifactsInfo: {dateSaved: new Date(), modelTopologyType: 'JSON'}};
    },
    load: async () => {
        const [jsonText, weightsBuf] = await Promise.all([
            readText(projectId, `${subDir}/model.json`),
            readBinary(projectId, `${subDir}/weights.bin`)
        ]);
        if (!jsonText || !weightsBuf) throw new Error(`Model not found: ${subDir}`);
        const parsed = JSON.parse(jsonText);
        const ab = weightsBuf.buffer.slice(
            weightsBuf.byteOffset, weightsBuf.byteOffset + weightsBuf.byteLength
        );
        return {
            modelTopology: parsed.modelTopology,
            weightSpecs:   parsed.weightsManifest[0].weights,
            weightData:    ab
        };
    }
});
