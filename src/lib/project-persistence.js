/* ── ML Project Persistence (.ob format) ────────────────────────────
   ML data is stored inside the .ob ZIP file under the ml/ prefix.

   Flow:
     • After training/changes: cacheMLData() → ipc 'ml-set-pending-data'
       Main process holds it in memory; injected into .ob on next blocks save
       OR when user clicks Save button (which calls saveXxxProject with showDialog:true).
     • Explicit save button: saveXxxProject(...) → ipc 'ml-save-ob-file' (shows dialog)
     • On mount: loadXxxProject() → ipc 'ml-get-loaded-data' (from opened .ob file)
   ─────────────────────────────────────────────────────────────── */

import { getAllProjectImages, getAllProjectAudio, saveImageToIDB, saveAudioToIDB } from './idb.js';
import {
    exportTrainedHead,
    loadHeadFromArtifacts,
    exportAudioModelArtifacts,
    loadAudioModelFromArtifacts,
    getMobileNet
} from './ml-engine.js';

const getIpc = () => {
    try { return window.require('electron').ipcRenderer; } catch (_) { return null; }
};

/* ═══════════════════════════════════════════════════════════════
   IMAGE CLASSIFIER
═══════════════════════════════════════════════════════════════ */

const buildImageMLData = async (project, labels, disabledLabels, trainingData, classifier) => {
    const allImages = await getAllProjectImages(project.id);

    for (const [label, examples] of Object.entries(trainingData)) {
        for (const ex of (examples || [])) {
            if (ex.data && !allImages[ex.id]) allImages[ex.id] = ex.data;
        }
    }

    const trainingIndex = {};
    const imageMap      = {};
    for (let li = 0; li < labels.length; li++) {
        const label = labels[li];
        trainingIndex[label] = [];
        for (const ex of (trainingData[label] || [])) {
            const src = allImages[ex.id];
            if (!src) continue;
            trainingIndex[label].push(ex.id);
            imageMap[`${li}/${ex.id}`] = src;
        }
    }

    let modelJson = null, modelWeights = null;
    if (classifier && classifier._head) {
        try {
            const arts = await exportTrainedHead(classifier._head);
            modelJson    = arts.modelJson;
            modelWeights = arts.weightData;
        } catch (e) {
            console.warn('[persistence] model export failed:', e.message);
        }
    }

    return {
        metadata: {
            id: project.id,
            name: project.name,
            type: project.type || 'images',
            labels,
            disabledLabels,
            trainingIndex,
            trained:  !!classifier,
            savedAt:  Date.now(),
            version:  '1.0'
        },
        images:       imageMap,
        audio:        {},
        modelJson,
        modelWeights
    };
};

/**
 * Save image project.
 * @param {object}   project
 * @param {string[]} labels
 * @param {string[]} disabledLabels
 * @param {object}   trainingData
 * @param {object}   classifier
 * @param {object}   [opts]
 * @param {boolean}  [opts.showDialog=true]  – true = show save-file dialog; false = cache only
 */
export const saveImageProject = async (project, labels, disabledLabels, trainingData, classifier, opts) => {
    const showDialog = !opts || opts.showDialog !== false;
    const ipc = getIpc();
    if (!ipc) return { success: false, error: 'no ipc' };

    const mlData = await buildImageMLData(project, labels, disabledLabels, trainingData, classifier);

    // Always update the in-memory cache so the normal blocks save picks it up
    ipc.send('ml-set-pending-data', mlData);

    if (showDialog) {
        return ipc.invoke('ml-save-ob-file', mlData);
    }
    return { success: true };
};

/**
 * Load image project from an .ob file that was opened via the blocks editor.
 * Returns null if no ML data was loaded.
 */
export const loadImageProject = async () => {
    const ipc = getIpc();
    if (!ipc) return null;

    const loaded = await ipc.invoke('ml-get-loaded-data');
    if (!loaded || !loaded.metadata) return null;

    const { metadata, images, modelJson, modelWeights } = loaded;
    const savedLabels = metadata.labels || [];
    const projectId   = metadata.id;

    const trainingData = {};
    for (const label of savedLabels) trainingData[label] = [];

    for (const [imgKey, dataUrl] of Object.entries(images || {})) {
        const slash    = imgKey.indexOf('/');
        const labelIdx = parseInt(imgKey.slice(0, slash), 10);
        const exId     = imgKey.slice(slash + 1);
        const label    = savedLabels[labelIdx];
        if (!label) continue;

        await saveImageToIDB(projectId, exId, dataUrl);
        trainingData[label].push({ id: exId, type: 'image' });
    }

    let classifier = null, net = null;
    if (modelJson && modelWeights) {
        try {
            net        = await getMobileNet();
            classifier = await loadHeadFromArtifacts(modelJson, modelWeights, savedLabels);
        } catch (e) {
            console.warn('[persistence] model load failed:', e.message);
        }
    }

    return {
        labels:         savedLabels,
        disabledLabels: metadata.disabledLabels || [],
        trainingData,
        classifier,
        net
    };
};

/* ═══════════════════════════════════════════════════════════════
   AUDIO CLASSIFIER
═══════════════════════════════════════════════════════════════ */

const buildAudioMLData = async (project, labels, disabledLabels, trainingData, isTrained) => {
    const allAudio = await getAllProjectAudio(project.id);

    const trainingIndex = {};
    const audioMap      = {};
    for (let li = 0; li < labels.length; li++) {
        const label = labels[li];
        trainingIndex[label] = [];
        for (const ex of (trainingData[label] || [])) {
            const rec = allAudio[ex.id] || (ex.spectrogramData
                ? { data: ex.spectrogramData, frameSize: ex.frameSize }
                : null);
            if (!rec) continue;
            trainingIndex[label].push(ex.id);
            audioMap[`${li}/${ex.id}`] = { data: Array.from(rec.data), frameSize: rec.frameSize };
        }
    }

    let modelJson = null, modelWeights = null;
    if (isTrained) {
        try {
            const arts = await exportAudioModelArtifacts();
            if (arts) { modelJson = arts.modelJson; modelWeights = arts.weightData; }
        } catch (e) {
            console.warn('[persistence] audio model export failed:', e.message);
        }
    }

    return {
        metadata: {
            id: project.id,
            name: project.name,
            type: 'sounds',
            labels,
            disabledLabels,
            trainingIndex,
            trained: isTrained,
            savedAt: Date.now(),
            version: '1.0'
        },
        images:       {},
        audio:        audioMap,
        modelJson,
        modelWeights
    };
};

/**
 * Save audio project.
 * @param {object}   opts  – { showDialog: boolean } default true
 */
export const saveAudioProject = async (project, labels, disabledLabels, trainingData, isTrained, opts) => {
    const showDialog = !opts || opts.showDialog !== false;
    const ipc = getIpc();
    if (!ipc) return { success: false, error: 'no ipc' };

    const mlData = await buildAudioMLData(project, labels, disabledLabels, trainingData, isTrained);
    ipc.send('ml-set-pending-data', mlData);

    if (showDialog) {
        return ipc.invoke('ml-save-ob-file', mlData);
    }
    return { success: true };
};

/**
 * Load audio project from an .ob file opened via the blocks editor.
 */
export const loadAudioProject = async () => {
    const ipc = getIpc();
    if (!ipc) return null;

    const loaded = await ipc.invoke('ml-get-loaded-data');
    if (!loaded || !loaded.metadata) return null;

    const { metadata, audio, modelJson, modelWeights } = loaded;
    const savedLabels = metadata.labels || [];
    const projectId   = metadata.id;

    const trainingData = {};
    for (const label of savedLabels) trainingData[label] = [];

    for (const [audioKey, { data, frameSize }] of Object.entries(audio || {})) {
        const slash    = audioKey.indexOf('/');
        const labelIdx = parseInt(audioKey.slice(0, slash), 10);
        const exId     = audioKey.slice(slash + 1);
        const label    = savedLabels[labelIdx];
        if (!label) continue;

        await saveAudioToIDB(projectId, exId, data, frameSize);
        trainingData[label].push({ id: exId, type: 'audio', spectrogramData: data, frameSize });
    }

    let modelRestored = false;
    if (modelJson && modelWeights) {
        modelRestored = await loadAudioModelFromArtifacts(modelJson, modelWeights, savedLabels);
    }

    return {
        labels:         savedLabels,
        disabledLabels: metadata.disabledLabels || [],
        trainingData,
        modelRestored
    };
};
