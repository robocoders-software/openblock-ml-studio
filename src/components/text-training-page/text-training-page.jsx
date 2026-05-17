import React, {useState, useRef, useEffect, useLayoutEffect, useCallback} from 'react';
import PropTypes from 'prop-types';
import styles from './text-training-page.css';
import openblockLogo from '../openblock-logo.svg';
import Spinner from 'openblock-gui/src/components/spinner/spinner.jsx';
import MLLoader from '../ml-loader/ml-loader.jsx';

import {
    trainText,
    classifyText,
    loadTextClassifier,
    setActiveModel
} from '../../lib/ml-engine.js';
import {saveTextProject, loadTextProject} from '../../lib/project-persistence.js';

const CLASS_COLORS = ['#E05C3D', '#2EAA7E', '#004AAD', '#003A8C', '#F39C12', '#E91E63', '#1ABC9C', '#E67E22'];
const generateId   = () => Math.random().toString(36).slice(2, 10);

/* Sample CSV users can download as a starting point */
const SAMPLE_CSV_CONTENT = `text,label
I love this product it is amazing,Positive
This is the best thing ever made,Positive
That was really fun and exciting,Positive
Great job I am really happy,Positive
This is terrible and broken,Negative
I hate this so much,Negative
Awful experience would not recommend,Negative
This does not work at all,Negative`;

/* Validate that the CSV matches the expected text,label format */
const validateDatasetCSV = rawText => {
    const lines = rawText.split('\n').map(l => l.trim()).filter(Boolean);
    if (lines.length === 0) return 'The file is empty.';

    const firstLine = lines[0];
    const hasHeader = firstLine.toLowerCase() === 'text,label';

    /* If there is no header, the first line must still look like a data row */
    if (!hasHeader && !firstLine.includes(',')) {
        return 'Invalid format. The CSV must have two columns: text and label.\n' +
               'Download the sample file to see the correct format.';
    }

    const rows = [];
    for (const line of lines) {
        const t = line.trim();
        if (!t || t.toLowerCase() === 'text,label') continue;
        const lastComma = t.lastIndexOf(',');
        if (lastComma < 1) continue;
        const text  = t.slice(0, lastComma).trim();
        const label = t.slice(lastComma + 1).trim();
        if (text && label) rows.push({text, label});
    }

    if (rows.length === 0) {
        return 'No valid data rows found.\n' +
               'Each row must be: your text, Label\n' +
               'Download the sample file to see the correct format.';
    }

    const uniqueLabels = [...new Set(rows.map(r => r.label))];
    if (uniqueLabels.length < 2) {
        return `Only 1 class found ("${uniqueLabels[0]}"). ` +
               'The dataset must contain at least 2 different classes.';
    }

    return null; // valid
};

/* Parse CSV: text,label  (header row optional) */
const parseDatasetCSV = rawText => {
    const rows = [];
    const lines = rawText.split('\n');
    for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        /* skip header row */
        if (trimmed.toLowerCase() === 'text,label') continue;
        /* handle quoted: "some text, with comma",Label */
        let text, label;
        const quotedMatch = trimmed.match(/^"((?:[^"\\]|\\.)*)"\s*,\s*(.+)$/);
        if (quotedMatch) {
            text  = quotedMatch[1].replace(/\\"/g, '"').trim();
            label = quotedMatch[2].trim();
        } else {
            /* find last comma for split */
            const lastComma = trimmed.lastIndexOf(',');
            if (lastComma < 1) continue;
            text  = trimmed.slice(0, lastComma).trim();
            label = trimmed.slice(lastComma + 1).trim();
        }
        if (text && label) rows.push({text, label});
    }
    return rows;
};


/* ── TextChip — renders one sample as a chip ── */
const TextChip = ({text, onDelete}) => {
    const isLong = text.length > 28;
    const chipClass = isLong ? styles.textChipLong : styles.textChip;
    const display = text.length > 60 ? text.slice(0, 57) + '…' : text;
    return (
        <div className={chipClass} title={text}>
            <span className={styles.chipText}>{display}</span>
            <button className={styles.chipDeleteBtn} onClick={onDelete} title="Remove">&#10005;</button>
        </div>
    );
};
TextChip.propTypes = {text: PropTypes.string.isRequired, onDelete: PropTypes.func.isRequired};

/* ── TextClassCard ── */
const TextClassCard = React.forwardRef(({
    label, colorIdx, samples,
    onAddSample, onDeleteSample, onDeleteAllSamples,
    onRename, onDelete, canDelete, onToggleDisable, isDisabled,
    menuOpen, onOpenMenu, onCloseMenu
}, ref) => {
    const color = CLASS_COLORS[colorIdx % CLASS_COLORS.length];
    const [editing,       setEditing]  = useState(false);
    const [newName,       setNewName]  = useState(label);
    const [confirmAction, setConfirm]  = useState(null);
    const [inputText,     setInput]    = useState('');
    const menuRef  = useRef(null);
    const inputRef = useRef(null);

    useEffect(() => {
        if (!menuOpen) return;
        const handler = e => {
            if (menuRef.current && !menuRef.current.contains(e.target)) {
                onCloseMenu();
                setConfirm(null);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [menuOpen, onCloseMenu]);

    const commitRename = () => {
        const n = newName.trim();
        if (n && n !== label) onRename(label, n);
        setEditing(false);
    };

    const handleAdd = () => {
        const text = inputText.trim();
        if (!text) return;
        onAddSample(label, {id: generateId(), type: 'text', text});
        setInput('');
        inputRef.current && inputRef.current.focus();
    };

    return (
        <div className={`${styles.classCard}${isDisabled ? ` ${styles.classCardDisabled}` : ''}`} ref={ref}>
            {/* Header */}
            <div className={styles.classCardHeader} style={{background: color}}>
                <div className={styles.classCardNameRow}>
                    {editing ? (
                        <input
                            className={styles.nameEditInput}
                            value={newName}
                            autoFocus
                            onChange={e => setNewName(e.target.value)}
                            onKeyPress={e => e.key === 'Enter' && commitRename()}
                            onBlur={commitRename}
                        />
                    ) : (
                        <span className={styles.classCardTitle}>{label}</span>
                    )}
                    <button
                        className={styles.classCardEditBtn}
                        onClick={() => { setEditing(true); setNewName(label); }}
                        title="Rename"
                    >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                    {isDisabled && <span className={styles.disabledBadge}>DISABLED</span>}
                </div>
                <div style={{position: 'relative'}} ref={menuRef}>
                    <button
                        className={styles.classCardMenuBtn}
                        onClick={() => menuOpen ? onCloseMenu() : onOpenMenu()}
                    >&#8942;</button>
                    {menuOpen && (
                        <div className={styles.classCardMenu}>
                            {confirmAction ? (
                                <div className={styles.menuConfirm}>
                                    <p className={styles.menuConfirmText}>
                                        {confirmAction === 'deleteAll' ? 'Delete all samples?' : 'Delete this class?'}
                                    </p>
                                    <div className={styles.menuConfirmBtns}>
                                        <button className={styles.menuConfirmYes} onClick={() => {
                                            if (confirmAction === 'deleteAll') onDeleteAllSamples(label);
                                            else onDelete(label);
                                            onCloseMenu();
                                            setConfirm(null);
                                        }}>Delete</button>
                                        <button className={styles.menuConfirmNo} onClick={() => setConfirm(null)}>Cancel</button>
                                    </div>
                                </div>
                            ) : (
                                <>
                                    <button onClick={() => { setEditing(true); setNewName(label); onCloseMenu(); }}>Rename class</button>
                                    <button onClick={() => { onToggleDisable(label); onCloseMenu(); }}>
                                        {isDisabled ? '✓ Enable class' : 'Disable class'}
                                    </button>
                                    <div className={styles.menuDivider}/>
                                    <button className={styles.danger} onClick={() => setConfirm('deleteAll')}>Delete all samples</button>
                                    {canDelete && (
                                        <button className={styles.danger} onClick={() => setConfirm('deleteClass')}>Delete class</button>
                                    )}
                                </>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Body */}
            <div className={styles.classCardBody}>
                <p className={styles.sampleCountLabel}>{samples.length} sample{samples.length !== 1 ? 's' : ''}</p>
                <div className={styles.chipsArea}>
                    {samples.length === 0 ? (
                        <span className={styles.emptyHint}>Type examples below and press Enter</span>
                    ) : (
                        samples.map(s => (
                            <TextChip
                                key={s.id}
                                text={s.text}
                                onDelete={() => onDeleteSample(label, s.id)}
                            />
                        ))
                    )}
                </div>
                <div className={styles.textInputRow}>
                    <input
                        ref={inputRef}
                        className={styles.sampleInput}
                        type="text"
                        placeholder={`Add a "${label}" example…`}
                        value={inputText}
                        onChange={e => setInput(e.target.value)}
                        onKeyPress={e => e.key === 'Enter' && handleAdd()}
                    />
                    <button className={styles.addSampleBtn} onClick={handleAdd} disabled={!inputText.trim()} title="Add">
                        +
                    </button>
                </div>
            </div>
        </div>
    );
});
TextClassCard.displayName = 'TextClassCard';
TextClassCard.propTypes = {
    label:              PropTypes.string.isRequired,
    colorIdx:           PropTypes.number.isRequired,
    samples:            PropTypes.array.isRequired,
    onAddSample:        PropTypes.func.isRequired,
    onDeleteSample:     PropTypes.func.isRequired,
    onDeleteAllSamples: PropTypes.func.isRequired,
    onRename:           PropTypes.func.isRequired,
    onDelete:           PropTypes.func.isRequired,
    canDelete:          PropTypes.bool.isRequired,
    onToggleDisable:    PropTypes.func.isRequired,
    isDisabled:         PropTypes.bool.isRequired,
    menuOpen:           PropTypes.bool.isRequired,
    onOpenMenu:         PropTypes.func.isRequired,
    onCloseMenu:        PropTypes.func.isRequired
};

/* ── TextTestingPanel ── */
const TextTestingPanel = ({isTrained, labels, labelColorMap}) => {
    const [inputText,     setInput]     = useState('');
    const [results,       setResults]   = useState([]);
    const [isClassifying, setClassify]  = useState(false);
    const [error,         setError]     = useState('');

    const doClassify = useCallback(async text => {
        if (!text || !isTrained) return;
        setClassify(true);
        setError('');
        try {
            const res = await classifyText(text);
            /* Winner-takes-all: top class = 100%, rest = 0%.
               Always keep original label order so bars never jump. */
            const arr = labels.map(lbl => ({
                label: lbl,
                prob:  res.label === lbl ? 100 : 0,
                isTop: res.label === lbl
            }));
            setResults(arr);
        } catch (e) {
            setError(e.message || 'Classification failed');
        } finally {
            setClassify(false);
        }
    }, [isTrained, labels]);

    /* Auto-classify on every keystroke (200 ms debounce) */
    useEffect(() => {
        const text = inputText.trim();
        if (!text || !isTrained) { setResults([]); return; }
        const timer = setTimeout(() => doClassify(text), 200);
        return () => clearTimeout(timer);
    }, [inputText, isTrained, doClassify]);

    if (!isTrained) {
        return <p className={styles.testingPlaceholder}>Train a model first, then test it here.</p>;
    }

    return (
        <div className={styles.testPanelContent}>
            <p className={styles.testInputLabel}>Input</p>
            <div className={styles.testInputRow}>
                <input
                    className={styles.testTextInput}
                    type="text"
                    placeholder="Type text here…"
                    value={inputText}
                    onChange={e => setInput(e.target.value)}
                />
            </div>
            <button
                className={styles.classifyBtn}
                onClick={() => doClassify(inputText.trim())}
                disabled={!inputText.trim() || isClassifying}
            >
                {isClassifying ? <Spinner small level="info"/> : 'Classify'}
            </button>

            {results.length > 0 && (
                <div className={styles.testOutputSection}>
                    <p className={styles.testOutputLabel}>Output</p>
                    {results.map(r => {
                        const origIdx = labels.indexOf(r.label);
                        const color   = (labelColorMap && labelColorMap[r.label]) || CLASS_COLORS[origIdx % CLASS_COLORS.length];
                        const isTop   = r.prob === 100;
                        return (
                            <div key={r.label} className={styles.testResultBar}>
                                <div className={styles.testResultLabelRow}>
                                    <span style={isTop ? {fontWeight: 700} : {}}>{r.label}</span>
                                    <span style={isTop ? {fontWeight: 700, color} : {}}>{r.prob}%</span>
                                </div>
                                <div className={styles.testResultTrack}>
                                    <div
                                        className={styles.testResultFill}
                                        style={{width: `${r.prob}%`, background: color}}
                                    />
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}

            {error && <p style={{fontSize: 11, color: '#e53935', margin: 0}}>{error}</p>}
        </div>
    );
};
TextTestingPanel.propTypes = {
    isTrained:     PropTypes.bool.isRequired,
    labels:        PropTypes.array.isRequired,
    labelColorMap: PropTypes.object.isRequired
};

/* ══════════════════════════════════════════════════════════════
   Main TextTrainingPage
══════════════════════════════════════════════════════════════ */
const TextTrainingPage = ({
    project, onBack, onUseInBlocks, onUpdateProject,
    onNewProject, onNewMLProject, onOpenMLProject
}) => {
    const [labels,        setLabels]    = useState(project.labels || ['Class 1', 'Class 2']);
    const [trainingData,  setData]      = useState({});
    const [loadingData,   setLoading]   = useState(true);
    const [isTrained,     setTrained]   = useState(!!project.trained);
    const [isTraining,    setTraining]  = useState(false);
    const [trainPct,      setTrainPct]  = useState(0);
    const [trainStatus,   setStatus]    = useState('');
    const [disabledLabels, setDisabled] = useState([]);
    const [openMenuLabel, setOpenMenu]  = useState(null);
    const [fileMenuOpen,  setFileMenu]  = useState(false);
    const [saveStatus,    setSaveStatus] = useState('idle');
    const [uploadError,   setUploadError] = useState('');
    const [renamingProject, setRenamingProject] = useState(false);
    const [renameValue,     setRenameValue]     = useState(project.name);

    const fileMenuRef      = useRef(null);
    const canvasRef        = useRef(null);
    const classesColRef    = useRef(null);
    const classCardRefs    = useRef([]);
    const trainCardRef     = useRef(null);
    const testCardRef      = useRef(null);
    const trainingAPIRef = useRef(null);
    const [svgPaths,     setSvgPaths]  = useState([]);
    const csvInputRef    = useRef(null);

    /* Close file menu when clicking outside */
    useEffect(() => {
        if (!fileMenuOpen) return;
        const handler = e => {
            if (fileMenuRef.current && !fileMenuRef.current.contains(e.target)) {
                setFileMenu(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [fileMenuOpen]);

    /* Load project data */
    const loadFromDisk = useCallback(async (signal = {cancelled: false}) => {
        let finalLabels = project.labels || labels;
        let trained = false;
        try {
            setLoading(true);
            const fromOb = await loadTextProject(project.id);
            if (fromOb && !signal.cancelled) {
                if (fromOb.name) onUpdateProject({...project, name: fromOb.name});
                if (fromOb.labels && fromOb.labels.length >= 2) {
                    setLabels(fromOb.labels);
                    finalLabels = fromOb.labels;
                }
                if (!signal.cancelled) setData(fromOb.trainingData || {});
                trained = !!(fromOb.modelRestored && !signal.cancelled);
                setTrained(trained);
            } else {
                /* Use in-memory data from project object (localStorage) */
                if (!signal.cancelled) setData(project.trainingData || {});
                if (project.trained) {
                    const loaded = await loadTextClassifier(project.id, project.labels || labels);
                    trained = !!(loaded && !signal.cancelled);
                    if (trained) setTrained(true);
                } else {
                    setTrained(false);
                }
            }
            /* Always register model type so the blocks editor knows this is a TEXT project,
               even before training — prevents stale image model from persisting. */
            if (!signal.cancelled) {
                setActiveModel({
                    projectId:      project.id,
                    projectName:    project.name,
                    type:           'text',
                    labels:         finalLabels,
                    trainingStatus: trained ? 'ready' : 'idle',
                    classifyText,
                    _trainingAPI:   trainingAPIRef
                });
            }
        } catch (err) {
            console.error('[TextPage] loadFromDisk:', err);
        } finally {
            if (!signal.cancelled) setLoading(false);
        }
    }, [project.id]); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        const signal = {cancelled: false};
        loadFromDisk(signal);
        return () => { signal.cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    /* Reload when a new .ob file is opened */
    useEffect(() => {
        try {
            const ipc = window.require('electron').ipcRenderer;
            const handler = () => {
                const signal = {cancelled: false};
                loadFromDisk(signal);
            };
            ipc.on('ml-project-file-changed', handler);
            return () => ipc.removeListener('ml-project-file-changed', handler);
        } catch (_) { /* not in Electron */ }
    }, [loadFromDisk]);

    /* Auto-save to disk whenever labels/data/trained changes so training data
       persists even if the user navigates back without explicitly saving. */
    useEffect(() => {
        if (loadingData) return;
        onUpdateProject({
            ...project,
            labels,
            trainingData: trainingData,
            trained:      isTrained,
            updatedAt:    Date.now()
        });
        saveTextProject(project, labels, trainingData, isTrained, {showDialog: false}).catch(() => {});
    }, [labels, trainingData, isTrained]); // eslint-disable-line react-hooks/exhaustive-deps

    /* ── CRUD helpers ── */
    const addSample = useCallback((label, sample) => {
        setData(d => ({...d, [label]: [...(d[label] || []), sample]}));
        setTrained(false);
    }, []);

    const deleteSample = useCallback((label, sampleId) => {
        setData(d => ({...d, [label]: (d[label] || []).filter(s => s.id !== sampleId)}));
        setTrained(false);
    }, []);

    const deleteAllSamples = useCallback(label => {
        setData(d => ({...d, [label]: []}));
        setTrained(false);
    }, []);

    const toggleDisable = useCallback(label => {
        setDisabled(prev =>
            prev.includes(label) ? prev.filter(l => l !== label) : [...prev, label]
        );
        setTrained(false);
    }, []);

    const renameClass = useCallback((oldName, newName) => {
        if (!newName || newName === oldName || labels.includes(newName)) return;
        setLabels(l => l.map(x => x === oldName ? newName : x));
        setData(d => {
            const c = {...d};
            if (c[oldName]) { c[newName] = c[oldName]; delete c[oldName]; }
            return c;
        });
        setTrained(false);
    }, [labels]);

    const deleteClass = useCallback(name => {
        if (labels.length <= 2) return;
        setLabels(l => l.filter(x => x !== name));
        setData(d => { const c = {...d}; delete c[name]; return c; });
        setTrained(false);
    }, [labels]);

    const addClass = () => {
        let n = labels.length + 1;
        while (labels.includes(`Class ${n}`)) n++;
        setLabels(l => [...l, `Class ${n}`]);
        setTrained(false);
    };

    /* ── Train ── */
    const activeLabels = labels.filter(l => !disabledLabels.includes(l));

    const canTrain = !isTraining &&
        activeLabels.length >= 2 &&
        activeLabels.every(l => (trainingData[l] || []).length >= 1);

    const trainModel = async () => {
        setTraining(true);
        setTrainPct(0);
        setStatus('Preparing…');
        try {
            await trainText(
                activeLabels, trainingData, project.id,
                s => setStatus(s),
                pct => setTrainPct(pct)
            );
            setTrained(true);
            setTrainPct(100);
            saveTextProject(project, labels, trainingData, true, {showDialog: false}).catch(() => {});
            setActiveModel({
                projectId:      project.id,
                projectName:    project.name,
                type:           'text',
                labels:         activeLabels,
                trainingStatus: 'ready',
                classifyText,
                _trainingAPI:   trainingAPIRef
            });
        } catch (err) {
            setStatus(`Error: ${err.message}`);
            console.error('[TextTrain]', err);
        } finally {
            setTraining(false);
        }
    };

    /* ── Project rename ── */
    const commitProjectRename = useCallback(() => {
        const trimmed = renameValue.trim();
        setRenamingProject(false);
        if (!trimmed || trimmed === project.name) return;
        onUpdateProject({...project, name: trimmed});
        const ipc = (() => { try { return window.require('electron').ipcRenderer; } catch (_) { return null; } })();
        if (ipc) {
            ipc.invoke('ml-write-file', project.id, 'project.json', JSON.stringify({
                id: project.id, name: trimmed, type: project.type,
                labels: labels || [], trained: isTrained,
                createdAt: project.createdAt, updatedAt: Date.now(),
                savedAt: project.savedAt || Date.now()
            })).catch(() => {});
        }
    }, [renameValue, project, labels, isTrained, onUpdateProject]);

    /* ── Save ── */
    const handleSave = useCallback(async () => {
        setSaveStatus('saving');
        try {
            const result = await saveTextProject(project, labels, trainingData, isTrained, {showDialog: false});
            if (result && result.success === false) throw new Error(result.error || 'Save failed');
            setSaveStatus('saved');
        } catch (err) {
            console.error('[TextPage] save failed:', err);
            setSaveStatus('error');
        } finally {
            setTimeout(() => setSaveStatus('idle'), 2000);
        }
    }, [project, labels, trainingData, isTrained]);

    /* Keep _trainingAPI ref fresh on every render so blocks always get latest closures */
    trainingAPIRef.current = {
        addTrainingText: (label, text) => {
            if (!label || !text) return;
            addSample(label, {id: generateId(), type: 'text', text});
        },
        startTraining:  trainModel,
        clearTraining:  () => labels.forEach(l => deleteAllSamples(l)),
        getStatus:      () => isTraining ? 'training' : isTrained ? 'ready' : 'idle'
    };

    /* ── Deploy / Export ── */
    const deployModel = () => {
        setActiveModel({
            projectId:      project.id,
            projectName:    project.name,
            type:           'text',
            labels,
            trainingStatus: isTrained ? 'ready' : 'idle',
            classifyText,
            _trainingAPI:   trainingAPIRef
        });
    };

    const handleExport = () => {
        deployModel();
        try { window.require('electron').ipcRenderer.send('ml-set-pending-project', project.id); } catch (_) {}
        onUseInBlocks();
    };

    /* ── CSV upload ── */
    const handleCSVUpload = e => {
        const file = e.target.files[0];
        if (!file) return;
        setUploadError('');

        /* Validate extension */
        if (!file.name.toLowerCase().endsWith('.csv')) {
            setUploadError('Invalid file type. Please upload a .csv file.');
            e.target.value = '';
            return;
        }

        const reader = new FileReader();
        reader.onload = evt => {
            const raw = evt.target.result || '';

            /* Validate format */
            const validationMsg = validateDatasetCSV(raw);
            if (validationMsg) {
                setUploadError(validationMsg);
                e.target.value = '';
                return;
            }

            const rows = parseDatasetCSV(raw);

            /* Build merged data using current state directly (closure) */
            const mergedData = {...trainingData};
            for (const {text, label} of rows) {
                if (!mergedData[label]) mergedData[label] = [];
                mergedData[label].push({id: generateId(), type: 'text', text});
            }

            /* Merge labels: existing + new from CSV */
            const csvLabels = [...new Set(rows.map(r => r.label).filter(Boolean))];
            const mergedLabels = [...labels];
            for (const lbl of csvLabels) {
                if (!mergedLabels.includes(lbl)) mergedLabels.push(lbl);
            }

            /* Remove classes that are empty after the upload (keep ≥ 2) */
            const nonEmpty = mergedLabels.filter(l => (mergedData[l] || []).length > 0);
            const finalLabels = nonEmpty.length >= 2 ? nonEmpty : mergedLabels;

            setLabels(finalLabels);
            setData(mergedData);
            setTrained(false);
        };
        reader.readAsText(file);
        /* reset so the same file can be re-uploaded */
        e.target.value = '';
    };

    /* ── CSV download ── */
    const handleDownloadSample = () => {
        const blob = new Blob([SAMPLE_CSV_CONTENT], {type: 'text/csv'});
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        a.download = 'sample_text_dataset.csv';
        a.click();
        URL.revokeObjectURL(url);
    };

    /* ── SVG bezier curves connecting cards ──
       Canvas no longer scrolls (overflow: hidden); only classesColumn scrolls.
       getBoundingClientRect() already returns live viewport positions, so no
       scroll-offset correction needed — just subtract canvas origin. */
    const recalcCurves = useCallback(() => {
        if (!canvasRef.current || !trainCardRef.current || !testCardRef.current) return;
        const wrap      = canvasRef.current;
        const wrapRect  = wrap.getBoundingClientRect();
        const trainRect = trainCardRef.current.getBoundingClientRect();
        const testRect  = testCardRef.current.getBoundingClientRect();
        const paths = [];

        classCardRefs.current.forEach((r, i) => {
            if (!r) return;
            const rect = r.getBoundingClientRect();
            const x1 = rect.right  - wrapRect.left;
            const y1 = rect.top + rect.height / 2 - wrapRect.top;
            const x2 = trainRect.left - wrapRect.left;
            const y2 = trainRect.top + trainRect.height / 2 - wrapRect.top;
            const cx = Math.max(40, (x2 - x1) / 2);
            paths.push({
                color: CLASS_COLORS[i % CLASS_COLORS.length],
                d: `M ${x1} ${y1} C ${x1 + cx} ${y1} ${x2 - cx} ${y2} ${x2} ${y2}`
            });
        });

        const tx1 = trainRect.right - wrapRect.left;
        const ty1 = trainRect.top + trainRect.height / 2 - wrapRect.top;
        const tx2 = testRect.left  - wrapRect.left;
        const ty2 = testRect.top + testRect.height / 2 - wrapRect.top;
        const tcx = Math.max(40, (tx2 - tx1) / 2);
        paths.push({
            color: '#004AAD',
            d: `M ${tx1} ${ty1} C ${tx1 + tcx} ${ty1} ${tx2 - tcx} ${ty2} ${tx2} ${ty2}`
        });

        setSvgPaths(prev =>
            prev.length === paths.length && prev.every((p, i) => p.d === paths[i].d)
                ? prev : paths
        );
    }, []);

    useLayoutEffect(() => { recalcCurves(); });
    useEffect(() => {
        window.addEventListener('resize', recalcCurves);
        return () => window.removeEventListener('resize', recalcCurves);
    }, [recalcCurves]);

    if (loadingData) return <MLLoader message="Loading project data…" />;

    const labelColorMap = Object.fromEntries(
        labels.map((l, i) => [l, CLASS_COLORS[i % CLASS_COLORS.length]])
    );

    return (
        <div className={styles.page}>
            {/* Top header */}
            <div className={styles.header}>
                <img src={openblockLogo} alt="RoboCoders Studio" className={styles.headerLogo} draggable={false}/>
                <span className={styles.headerTitle}>Machine Learning Environment</span>
                <nav className={styles.headerNav}>
                    <div className={styles.fileMenuWrap} ref={fileMenuRef}>
                        <button className={styles.navBtn} onClick={() => setFileMenu(o => !o)}>File</button>
                        {fileMenuOpen && (
                            <div className={styles.navDropdown}>
                                <button onClick={() => { setFileMenu(false); (onNewProject || onBack)(); }}>New</button>
                                <button onClick={() => { setFileMenu(false); (onNewMLProject || onBack)(); }}>New ML Project</button>
                            </div>
                        )}
                    </div>
                    <button className={styles.navBtn}>Help</button>
                </nav>
                <div className={styles.headerSpacer}/>
                <button className={styles.headerBackBtn} onClick={onBack}>&#8592; Back</button>
            </div>

            {/* Sub-header */}
            <div className={styles.subHeader}>
                {/* Text Classifier label + info icon */}
                <span className={styles.subHeaderType}>Text Classifier</span>
                <span className={styles.infoIcon} title="Train a model to classify text — then use it in your Blocks project.">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="#004AAD" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="11"/>
                        <circle cx="12" cy="8" r="1.3" fill="white"/>
                        <rect x="10.7" y="11" width="2.6" height="6" rx="1.3" fill="white"/>
                    </svg>
                </span>
                <div className={styles.divider}/>
                {renamingProject ? (
                    <input
                        className={styles.projectNameInput}
                        autoFocus
                        value={renameValue}
                        onChange={e => setRenameValue(e.target.value)}
                        onBlur={commitProjectRename}
                        onKeyDown={e => {
                            if (e.key === 'Enter') { e.preventDefault(); commitProjectRename(); }
                            if (e.key === 'Escape') { setRenameValue(project.name); setRenamingProject(false); }
                        }}
                    />
                ) : (
                    <span
                        className={styles.projectNamePill}
                        title="Click to rename"
                        onClick={() => { setRenameValue(project.name); setRenamingProject(true); }}
                    >
                        {project.name}
                    </span>
                )}
                <div className={styles.divider}/>

                {/* CSV upload */}
                <button className={styles.csvUploadBtn} onClick={() => { setUploadError(''); csvInputRef.current && csvInputRef.current.click(); }}>
                    Upload dataset from .csv
                </button>
                <input
                    ref={csvInputRef}
                    type="file"
                    accept=".csv,text/csv"
                    style={{display: 'none'}}
                    onChange={handleCSVUpload}
                />
                {uploadError && (
                    <span className={styles.uploadError} title={uploadError}>
                        &#9888; Invalid format
                        <span className={styles.uploadErrorTooltip}>{uploadError}</span>
                    </span>
                )}

                {/* Save icon */}
                <button
                    className={styles.saveBtn}
                    title={saveStatus === 'saving' ? 'Saving…' : saveStatus === 'saved' ? 'Saved!' : saveStatus === 'error' ? 'Save failed' : 'Save ML Project'}
                    disabled={saveStatus === 'saving'}
                    onClick={handleSave}
                >
                    {saveStatus === 'saving' ? '⏳' : saveStatus === 'saved' ? '✓' : saveStatus === 'error' ? '✗' : '💾'}
                </button>

                <div className={styles.spacer}/>

                {/* Download sample CSV */}
                <button className={styles.downloadSampleBtn} onClick={handleDownloadSample}>
                    Download Sample
                </button>
            </div>

            {/* Canvas */}
            <div className={styles.canvas} ref={canvasRef}>
                <svg className={styles.curveSvg}>
                    {svgPaths.map((p, i) => (
                        <path key={i} d={p.d} stroke={p.color} strokeWidth="2.5" fill="none" opacity="0.85"/>
                    ))}
                </svg>

                {/* Classes column — scrolls independently */}
                <div className={styles.classesColumn} ref={classesColRef} onScroll={recalcCurves}>
                    {labels.map((lbl, i) => (
                        <TextClassCard
                            key={lbl}
                            ref={el => { classCardRefs.current[i] = el; }}
                            label={lbl}
                            colorIdx={i}
                            samples={trainingData[lbl] || []}
                            canDelete={labels.length > 2}
                            isDisabled={disabledLabels.includes(lbl)}
                            menuOpen={openMenuLabel === lbl}
                            onOpenMenu={() => setOpenMenu(lbl)}
                            onCloseMenu={() => setOpenMenu(null)}
                            onAddSample={addSample}
                            onDeleteSample={deleteSample}
                            onDeleteAllSamples={deleteAllSamples}
                            onRename={renameClass}
                            onDelete={deleteClass}
                            onToggleDisable={toggleDisable}
                        />
                    ))}
                    <button className={styles.addClassCard} onClick={addClass}>+ Add a Class</button>
                </div>

                {/* Training column — stays fixed */}
                <div className={styles.trainingColumn}>
                <div className={styles.trainingCard} ref={trainCardRef}>
                    <div className={styles.trainingHeader}><span>Training</span></div>
                    {disabledLabels.length > 0 && (
                        <div className={styles.activeClassInfo}>
                            {activeLabels.length} / {labels.length} classes active
                        </div>
                    )}
                    <div className={styles.trainingBody}>
                        {trainStatus && (
                            <p className={
                                isTrained && trainStatus === 'Training Complete'
                                    ? styles.trainCompletedText
                                    : styles.trainStatusText
                            }>{trainStatus}</p>
                        )}
                        {isTraining && (
                            <div className={styles.progressTrack}>
                                <div className={styles.progressFill} style={{width: `${trainPct}%`}}/>
                            </div>
                        )}
                        {isTrained ? (
                            <button className={styles.trainAgainBtn} onClick={trainModel} disabled={isTraining || !canTrain}>
                                Train Again
                            </button>
                        ) : (
                            <button className={styles.trainModelBtn} onClick={trainModel} disabled={!canTrain}>
                                {isTraining ? <Spinner small level="info"/> : 'Train Model'}
                            </button>
                        )}
                    </div>
                </div>

                </div>

                {/* Testing column — stays fixed */}
                <div className={styles.testingColumn}>
                <div className={styles.testingCard} ref={testCardRef}>
                    <div className={styles.testingHeader}>
                        <span>Testing</span>
                        {isTrained && (
                            <button className={styles.exportModelBtn} onClick={handleExport}>
                                &#8593; Export Model
                            </button>
                        )}
                    </div>
                    <div className={styles.testingBody}>
                        <TextTestingPanel
                            isTrained={isTrained}
                            labels={labels}
                            labelColorMap={labelColorMap}
                        />
                    </div>
                </div>
                </div>
            </div>
        </div>
    );
};

TextTrainingPage.propTypes = {
    project:         PropTypes.object.isRequired,
    onBack:          PropTypes.func.isRequired,
    onUseInBlocks:   PropTypes.func.isRequired,
    onUpdateProject: PropTypes.func.isRequired,
    onNewProject:    PropTypes.func,
    onNewMLProject:  PropTypes.func,
    onOpenMLProject: PropTypes.func
};

export default TextTrainingPage;
