/* ── ML Project Persistence (.ob format) ────────────────────────────
   All ML assets live on disk under:
     <userData>/ml-projects/<projectId>/

   The main process reads/writes that directory directly.
   The .ob ZIP is produced by bundling that directory (will-download
   hook or explicit save dialog); opening an .ob extracts it back.

   Flow:
     • Any ML write (add image, record audio, train) goes to FS in real-time.
     • saveXxxProject({showDialog:false}) — just refreshes project.json and
       tells main which project is active for the next blocks save.
     • saveXxxProject({showDialog:true})  — shows a Save As dialog; main bundles
       the entire ML directory into the chosen .ob file.
     • loadXxxProject(projectId)  — tells main to extract ml/ from the CLI-opened
       .ob file into <userData>/ml-projects/<projectId>/; returns project.json.
   ─────────────────────────────────────────────────────────────── */

import {fsWriteFile, loadProjectImagesFromFS, loadProjectAudioFromFS} from './ml-fs.js';
import {
    loadImageClassifier, getMobileNet,
    loadSoundClassifier,
    loadTextClassifier, classifyText,
    setActiveModel
} from './ml-engine.js';

const getIpc = () => {
    try { return window.require('electron').ipcRenderer; } catch (_) { return null; }
};

/* ═══════════════════════════════════════════════════════════════
   IMAGE CLASSIFIER
═══════════════════════════════════════════════════════════════ */

/**
 * Persist image project metadata and notify main process.
 * @param {boolean} [opts.showDialog=true]
 */
export const saveImageProject = async (project, labels, disabledLabels, trainingData, classifier, opts) => {
    const showDialog = !opts || opts.showDialog !== false;
    const ipc = getIpc();
    if (!ipc) return {success: false, error: 'no ipc'};

    /* Build index: label → [sampleId, ...] so samples survive a reload */
    const trainingIndex = {};
    for (const [label, samples] of Object.entries(trainingData || {})) {
        trainingIndex[label] = (samples || []).map(s => s.id);
    }

    /* Keep project.json on disk up-to-date */
    const now = Date.now();
    try {
        await fsWriteFile(project.id, 'project.json', JSON.stringify({
            id:            project.id,
            name:          project.name,
            type:          project.type || 'images',
            labels,
            disabledLabels,
            trainingIndex,
            trained:       !!classifier,
            createdAt:     project.createdAt || now,
            updatedAt:     now,
            savedAt:       now
        }));
    } catch (e) {
        console.error('[persistence] saveImageProject: failed to write project.json:', e);
        return {success: false, error: e.message};
    }

    if (showDialog) {
        return ipc.invoke('ml-save-ob-file', project.id, project.name);
    }
    return {success: true};
};

/**
 * Restore image project from an .ob file opened via the blocks editor.
 * Returns null if the .ob contains no ML data.
 */
export const loadImageProject = async (projectId) => {
    const ipc = getIpc();
    if (!ipc) return null;

    /* Main extracts ml/ from the .ob into <userData>/ml-projects/<projectId>/ */
    const meta = await ipc.invoke('ml-get-loaded-data', projectId);
    // Distinguish "no ML data in file" from "load error"
    if (!meta || meta.noMlData) return null;
    if (meta.loadError) {
        console.error('[persistence] ml-get-loaded-data error:', meta.loadError);
        return null;
    }
    if (!meta.labels) return null;

    const savedLabels = meta.labels;

    /* Rebuild lightweight trainingData from the training index */
    const trainingData = {};
    for (const label of savedLabels) trainingData[label] = [];
    for (const [label, ids] of Object.entries(meta.trainingIndex || {})) {
        if (!trainingData[label]) trainingData[label] = [];
        for (const id of ids) {
            trainingData[label].push({id, type: 'image'});
        }
    }

    /* Restore classifier if a trained model was bundled */
    let classifier = null, net = null;
    if (meta.trained) {
        try {
            net        = await getMobileNet();
            classifier = await loadImageClassifier(projectId, savedLabels);
        } catch (e) {
            console.warn('[persistence] image model load failed:', e.message);
        }
    }

    /* Expose the restored model so the blocks editor VM extension can use it immediately,
       fixing the race condition where blocks render before window.__openblockMLModel is set. */
    if (classifier && net) {
        setActiveModel({
            projectId,
            type:          meta.type || 'images',
            labels:        savedLabels,
            classifier,
            mobileNet:     net,
            trainingStatus: 'ready'
        });
    }

    return {
        name:           meta.name || '',
        labels:         savedLabels,
        disabledLabels: meta.disabledLabels || [],
        trainingData,
        classifier,
        net
    };
};

/* ═══════════════════════════════════════════════════════════════
   AUDIO CLASSIFIER
═══════════════════════════════════════════════════════════════ */

/**
 * Persist audio project metadata and notify main process.
 * @param {boolean} [opts.showDialog=true]
 */
export const saveAudioProject = async (project, labels, disabledLabels, trainingData, isTrained, opts) => {
    const showDialog = !opts || opts.showDialog !== false;
    const ipc = getIpc();
    if (!ipc) return {success: false, error: 'no ipc'};

    const trainingIndex = {};
    for (const [label, samples] of Object.entries(trainingData || {})) {
        trainingIndex[label] = (samples || []).map(s => s.id);
    }

    const now = Date.now();
    try {
        await fsWriteFile(project.id, 'project.json', JSON.stringify({
            id:            project.id,
            name:          project.name,
            type:          'sounds',
            labels,
            disabledLabels,
            trainingIndex,
            trained:       isTrained,
            createdAt:     project.createdAt || now,
            updatedAt:     now,
            savedAt:       now
        }));
    } catch (e) {
        console.error('[persistence] saveAudioProject: failed to write project.json:', e);
        return {success: false, error: e.message};
    }

    if (showDialog) {
        return ipc.invoke('ml-save-ob-file', project.id, project.name);
    }
    return {success: true};
};

/**
 * Restore audio project from an .ob file opened via the blocks editor.
 * Returns null if the .ob contains no ML data.
 */
export const loadAudioProject = async (projectId) => {
    const ipc = getIpc();
    if (!ipc) return null;

    const meta = await ipc.invoke('ml-get-loaded-data', projectId);
    if (!meta || meta.noMlData) return null;
    if (meta.loadError) {
        console.error('[persistence] ml-get-loaded-data error (audio):', meta.loadError);
        return null;
    }
    if (!meta.labels) return null;

    const savedLabels = meta.labels;

    const trainingData = {};
    for (const label of savedLabels) trainingData[label] = [];
    for (const [label, ids] of Object.entries(meta.trainingIndex || {})) {
        if (!trainingData[label]) trainingData[label] = [];
        for (const id of ids) {
            trainingData[label].push({id, type: 'audio'});
        }
    }

    /* Enrich trainingData with spectrogram data from FS */
    const enriched = await loadProjectAudioFromFS(projectId, trainingData);

    /* Restore the transfer recognizer from the bundled model */
    let modelRestored = false;
    if (meta.trained) {
        try {
            const cls = await loadSoundClassifier(projectId, savedLabels);
            modelRestored = !!cls;
        } catch (e) {
            console.warn('[persistence] audio model load failed:', e.message);
        }
    }

    return {
        name:           meta.name || '',
        labels:         savedLabels,
        disabledLabels: meta.disabledLabels || [],
        trainingData:   enriched,
        modelRestored
    };
};

/* ═══════════════════════════════════════════════════════════════
   TEXT CLASSIFIER
═══════════════════════════════════════════════════════════════ */

export const saveTextProject = async (project, labels, trainingData, isTrained, opts) => {
    const showDialog = !opts || opts.showDialog !== false;
    const ipc = getIpc();
    if (!ipc) return {success: false, error: 'no ipc'};

    /* Text samples are small — store text inline in trainingIndex */
    const trainingIndex = {};
    for (const [label, samples] of Object.entries(trainingData || {})) {
        trainingIndex[label] = (samples || []).map(s => ({id: s.id, text: s.text || ''}));
    }

    const now = Date.now();
    try {
        await fsWriteFile(project.id, 'project.json', JSON.stringify({
            id:           project.id,
            name:         project.name,
            type:         'text',
            labels,
            trainingIndex,
            trained:      isTrained,
            createdAt:    project.createdAt || now,
            updatedAt:    now,
            savedAt:      now
        }));
    } catch (e) {
        console.error('[persistence] saveTextProject: failed to write project.json:', e);
        return {success: false, error: e.message};
    }

    if (showDialog) {
        return ipc.invoke('ml-save-ob-file', project.id, project.name);
    }
    return {success: true};
};

export const loadTextProject = async (projectId) => {
    const ipc = getIpc();
    if (!ipc) return null;

    const meta = await ipc.invoke('ml-get-loaded-data', projectId);
    if (!meta || meta.noMlData) return null;
    if (meta.loadError) {
        console.error('[persistence] ml-get-loaded-data error (text):', meta.loadError);
        return null;
    }
    if (!meta.labels) return null;

    const savedLabels = meta.labels;

    /* Rebuild trainingData — text samples are stored inline in trainingIndex */
    const trainingData = {};
    for (const label of savedLabels) trainingData[label] = [];
    for (const [label, items] of Object.entries(meta.trainingIndex || {})) {
        if (!trainingData[label]) trainingData[label] = [];
        for (const item of (Array.isArray(items) ? items : [])) {
            if (item && item.text !== undefined) {
                trainingData[label].push({id: item.id, type: 'text', text: item.text});
            }
        }
    }

    let modelRestored = false;
    if (meta.trained) {
        try {
            const cls = await loadTextClassifier(projectId, savedLabels);
            modelRestored = !!cls;
            if (cls) {
                setActiveModel({
                    projectId,
                    type:           'text',
                    labels:         savedLabels,
                    trainingStatus: 'ready',
                    classifyText
                });
            }
        } catch (e) {
            console.warn('[persistence] text model load failed:', e.message);
        }
    }

    return {
        name:         meta.name || '',
        labels:       savedLabels,
        trainingData,
        modelRestored
    };
};
