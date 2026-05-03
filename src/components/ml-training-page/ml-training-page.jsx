import React, {useState, useRef, useEffect, useLayoutEffect, useCallback} from 'react';
import PropTypes from 'prop-types';
import styles from './ml-training-page.css';

import {
    getMobileNet,
    trainImages,
    loadImageClassifier,
    predictImages,
    saveImageToIDB,
    getImageFromIDB,
    setActiveModel
} from '../../lib/ml-engine.js';
import {loadProjectImages} from '../../lib/idb.js';

const CLASS_COLORS = ['#E05C3D', '#2EAA7E', '#3498DB', '#9B59B6', '#F39C12', '#E91E63', '#1ABC9C', '#E67E22'];
const MAX_THUMBS   = 9;
const generateId   = () => Math.random().toString(36).slice(2, 10);

/* ──────────────────────────────────────────────
   Accuracy vs Epochs SVG chart
────────────────────────────────────────────── */
const AccuracyChart = ({points}) => {
    if (!points || points.length < 2) return null;
    const W = 220, H = 150;
    const pL = 30, pR = 10, pT = 10, pB = 30;
    const cW = W - pL - pR;
    const cH = H - pT - pB;
    const maxX = points[points.length - 1].x || 1;
    const sx = x => pL + (x / maxX) * cW;
    const sy = y => pT + (1 - Math.min(1, Math.max(0, y))) * cH;
    const lineD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(' ');
    const yTicks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
    const xCount = Math.min(maxX, 10);
    return (
        <svg width={W} height={H} style={{display: 'block', margin: '0 auto', overflow: 'visible'}}>
            {yTicks.map(y => (
                <g key={y}>
                    <line x1={pL} y1={sy(y)} x2={W - pR} y2={sy(y)} stroke="#e8daf5" strokeWidth="1"/>
                    <text x={pL - 4} y={sy(y) + 3} textAnchor="end" fontSize="8" fill="#aaa">{y.toFixed(1)}</text>
                </g>
            ))}
            {Array.from({length: xCount + 1}, (_, i) => Math.round((i / xCount) * maxX)).map(x => (
                <text key={x} x={sx(x)} y={H - pB + 13} textAnchor="middle" fontSize="8" fill="#aaa">{x}</text>
            ))}
            <line x1={pL} y1={pT} x2={pL} y2={H - pB} stroke="#ccc" strokeWidth="1"/>
            <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke="#ccc" strokeWidth="1"/>
            <path d={lineD} stroke="#3d7cf5" strokeWidth="2.5" fill="none" strokeLinejoin="round" strokeLinecap="round"/>
            <circle cx={sx(points[points.length - 1].x)} cy={sy(points[points.length - 1].y)} r="4" fill="#3d7cf5"/>
            <text x={pL + cW / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="#888">Accuracy Vs Epochs</text>
        </svg>
    );
};
AccuracyChart.propTypes = {points: PropTypes.array};

/* ──────────────────────────────────────────────
   Class card with inline webcam
────────────────────────────────────────────── */
const ClassCard = React.forwardRef(({
    label, colorIdx, samples, onAddImages, onDeleteSample, onRename, onDelete, canDelete, projectId, selectedDeviceId
}, ref) => {
    const color = CLASS_COLORS[colorIdx % CLASS_COLORS.length];
    const [editing,    setEditing]   = useState(false);
    const [newName,    setNewName]   = useState(label);
    const [menuOpen,   setMenu]      = useState(false);
    const [thumbData,  setThumbData] = useState([]); // [{src, id}]
    const [showWebcam, setShowWebcam] = useState(false);
    const [camErr,     setCamErr]    = useState('');

    const fileRef   = useRef(null);
    const videoRef  = useRef(null);
    const streamRef = useRef(null);
    const holdRef   = useRef(null);

    /* Load thumbnails with id tracking for deletion */
    useEffect(() => {
        let cancelled = false;
        const toLoad = samples.filter(s => s.type === 'image').slice(0, MAX_THUMBS);
        Promise.all(toLoad.map(async s => {
            const src = (s.data && s.data.startsWith('data:')) ? s.data : await getImageFromIDB(projectId, s.id);
            return src ? {src, id: s.id} : null;
        })).then(data => { if (!cancelled) setThumbData(data.filter(Boolean)); });
        return () => { cancelled = true; };
    }, [samples, projectId]);

    /* Cleanup on unmount */
    useEffect(() => () => {
        if (holdRef.current) clearInterval(holdRef.current);
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    }, []);

    /* Attach stream to video element after it renders */
    useEffect(() => {
        if (showWebcam && videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
        }
    }, [showWebcam]);

    const startWebcam = async () => {
        setCamErr('');
        const constraints = {video: selectedDeviceId ? {deviceId: {exact: selectedDeviceId}} : true};
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;
            setShowWebcam(true); // video element renders, then effect above attaches stream
        } catch (e) { setCamErr(e.message); }
    };

    const stopWebcam = () => {
        if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
        setShowWebcam(false);
        setCamErr('');
    };

    const captureOne = useCallback(() => {
        const v = videoRef.current;
        if (!v || !v.videoWidth || !v.videoHeight) return;
        const c = document.createElement('canvas');
        c.width = v.videoWidth; c.height = v.videoHeight;
        c.getContext('2d').drawImage(v, 0, 0);
        onAddImages(label, [c.toDataURL('image/jpeg', 0.85)]);
    }, [label, onAddImages]);

    const holdStart = e => {
        e.preventDefault();
        if (holdRef.current) return; // already holding
        captureOne();
        holdRef.current = setInterval(captureOne, 200);
    };

    const holdEnd = e => {
        e.preventDefault();
        if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
    };

    const commitRename = () => {
        const n = newName.trim();
        if (n && n !== label) onRename(label, n);
        setEditing(false);
    };

    const handleFileUpload = e => {
        const files = [...(e.target.files || [])];
        Promise.all(files.map(f => new Promise(res => {
            const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f);
        }))).then(urls => onAddImages(label, urls));
        e.target.value = '';
    };

    return (
        <div className={styles.classCard} ref={ref}>
            <div className={styles.classCardHeader} style={{background: color}}>
                {editing ? (
                    <input className={styles.nameEditInput} value={newName} autoFocus
                        onChange={e => setNewName(e.target.value)}
                        onKeyPress={e => e.key === 'Enter' && commitRename()}
                        onBlur={commitRename}/>
                ) : (
                    <span className={styles.classCardTitle}>{label}</span>
                )}
                <button className={styles.classCardEditBtn} onClick={() => { setEditing(true); setNewName(label); }} title="Rename">&#9998;</button>
                <div style={{position: 'relative'}}>
                    <button className={styles.classCardMenuBtn} onClick={() => setMenu(m => !m)}>&#8942;</button>
                    {menuOpen && (
                        <div className={styles.classCardMenu}>
                            <button onClick={() => { setEditing(true); setNewName(label); setMenu(false); }}>Rename class</button>
                            {canDelete && (
                                <button className={styles.danger} onClick={() => { onDelete(label); setMenu(false); }}>Delete class</button>
                            )}
                        </div>
                    )}
                </div>
            </div>

            <div className={styles.classCardBody}>
                <div className={styles.classCardLeft}>
                    {showWebcam ? (
                        <div className={styles.inlineWebcam}>
                            <div className={styles.inlineWebcamHeader}>
                                <span className={styles.webcamSectionLabel}>Webcam</span>
                                <button className={styles.camBackBtn} onClick={stopWebcam} title="Back to upload">&#8592;</button>
                            </div>
                            {camErr ? (
                                <p style={{color: '#e53935', fontSize: 11, padding: '4px 0'}}>{camErr}</p>
                            ) : (
                                <video ref={videoRef} autoPlay playsInline muted className={styles.inlineVideo}/>
                            )}
                            <div className={styles.holdRow}>
                                <button
                                    className={styles.holdBtn}
                                    onMouseDown={holdStart}
                                    onMouseUp={holdEnd}
                                    onMouseLeave={holdEnd}
                                    onTouchStart={holdStart}
                                    onTouchEnd={holdEnd}
                                    disabled={!!camErr}
                                >
                                    Hold to Record
                                </button>
                                <button className={styles.gearBtn} title="Settings">&#9881;</button>
                            </div>
                        </div>
                    ) : (
                        <>
                            <p className={styles.addSamplesLabel}>Add Image Samples</p>
                            <div className={styles.addBtnsRow}>
                                <label className={styles.addBtn}>
                                    <svg className={styles.addBtnIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                                        <polyline points="17 8 12 3 7 8"/>
                                        <line x1="12" y1="3" x2="12" y2="15"/>
                                    </svg>
                                    Upload
                                    <input type="file" multiple accept="image/*" style={{display: 'none'}} ref={fileRef} onChange={handleFileUpload}/>
                                </label>
                                <button className={styles.addBtn} onClick={startWebcam}>
                                    <svg className={styles.addBtnIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                        <path d="M23 7l-7 5 7 5V7z"/>
                                        <rect x="1" y="5" width="15" height="14" rx="2"/>
                                    </svg>
                                    Webcam
                                </button>
                            </div>
                            {camErr && <p style={{color: '#e53935', fontSize: 11, marginTop: 4}}>{camErr}</p>}
                        </>
                    )}
                </div>

                <div className={styles.classCardRight}>
                    <p className={styles.sampleCountLabel}>{samples.length} Image Sample{samples.length !== 1 ? 's' : ''}</p>
                    {thumbData.length > 0 && (
                        <div className={styles.thumbnailGrid}>
                            {thumbData.map((item, i) => {
                                const isLast = i === MAX_THUMBS - 1 && samples.length > MAX_THUMBS;
                                return (
                                    <div key={item.id} className={styles.thumb}>
                                        <img src={item.src} alt=""/>
                                        <button
                                            className={styles.thumbDeleteBtn}
                                            onClick={e => { e.stopPropagation(); onDeleteSample(label, item.id); }}
                                            title="Remove sample"
                                        >&#10005;</button>
                                        {isLast && <div className={styles.thumbMore}>+{samples.length - MAX_THUMBS}</div>}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
});
ClassCard.displayName = 'ClassCard';
ClassCard.propTypes = {
    label:            PropTypes.string.isRequired,
    colorIdx:         PropTypes.number.isRequired,
    samples:          PropTypes.array.isRequired,
    onAddImages:      PropTypes.func.isRequired,
    onDeleteSample:   PropTypes.func.isRequired,
    onRename:         PropTypes.func.isRequired,
    onDelete:         PropTypes.func.isRequired,
    canDelete:        PropTypes.bool.isRequired,
    projectId:        PropTypes.string.isRequired,
    selectedDeviceId: PropTypes.string
};

/* ──────────────────────────────────────────────
   Testing panel (Upload / Webcam)
────────────────────────────────────────────── */
const TestingPanel = ({isTrained, classifierRef, mobileNetRef, labels, selectedDeviceId}) => {
    const videoRef    = useRef(null);
    const streamRef   = useRef(null);
    const intervalRef = useRef(null);
    const [mode,      setMode]    = useState('idle');
    const [results,   setResults] = useState([]);
    const [testImg,   setTestImg] = useState(null);

    const stopCam = useCallback(() => {
        clearInterval(intervalRef.current);
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
        setMode('idle');
        setResults([]);
    }, []);

    const startWebcam = useCallback(async () => {
        const constraints = {video: selectedDeviceId ? {deviceId: {exact: selectedDeviceId}} : true};
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;
            if (videoRef.current) videoRef.current.srcObject = stream;
            setMode('webcam');
            intervalRef.current = setInterval(async () => {
                if (!videoRef.current || !classifierRef.current || !mobileNetRef.current) return;
                try {
                    const probs = await predictImages(videoRef.current, classifierRef.current, mobileNetRef.current, labels);
                    setResults(probs);
                } catch (_) { /* non-fatal */ }
            }, 400);
        } catch (e) { console.error('[TestPanel]', e); }
    }, [selectedDeviceId, classifierRef, mobileNetRef, labels]);

    const handleUpload = useCallback(async e => {
        const file = e.target.files[0];
        if (!file || !classifierRef.current || !mobileNetRef.current) return;
        const dataUrl = await new Promise(res => {
            const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(file);
        });
        setTestImg(dataUrl);
        const img = new Image();
        img.src = dataUrl;
        await new Promise(res => { img.onload = res; });
        try {
            const probs = await predictImages(img, classifierRef.current, mobileNetRef.current, labels);
            setResults(probs);
            setMode('upload');
        } catch (_) { /* non-fatal */ }
        e.target.value = '';
    }, [classifierRef, mobileNetRef, labels]);

    useEffect(() => () => stopCam(), [stopCam]);

    if (!isTrained) {
        return <p className={styles.testingPlaceholder}>You must train a model on the left before you can test it here.</p>;
    }

    return (
        <div className={styles.testPanelContent}>
            {mode === 'webcam' && (
                <div className={styles.testVideoWrap}>
                    <video ref={videoRef} autoPlay playsInline muted className={styles.testVideo}/>
                </div>
            )}
            {mode === 'upload' && testImg && (
                <div className={styles.testVideoWrap}>
                    <img src={testImg} alt="test" style={{width: '100%', borderRadius: 8, display: 'block'}}/>
                </div>
            )}
            {mode === 'idle' && <p className={styles.testByLabel}>Test Image By</p>}
            <div className={styles.testOptionRow}>
                <label className={styles.testOptionBtn}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="26" height="26">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    Upload
                    <input type="file" accept="image/*" style={{display: 'none'}} onChange={handleUpload}/>
                </label>
                <button className={styles.testOptionBtn} onClick={mode === 'webcam' ? stopCam : startWebcam}>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="26" height="26">
                        <path d="M23 7l-7 5 7 5V7z"/>
                        <rect x="1" y="5" width="15" height="14" rx="2"/>
                    </svg>
                    {mode === 'webcam' ? 'Stop' : 'Webcam'}
                </button>
            </div>
            {results.map(r => (
                <div key={r.label} className={styles.testResultBar}>
                    <div className={styles.testResultLabelRow}>
                        <span>{r.label}</span>
                        <span>{(r.prob || 0).toFixed(1)}%</span>
                    </div>
                    <div className={styles.testResultTrack}>
                        <div className={styles.testResultFill} style={{width: `${r.prob || 0}%`}}/>
                    </div>
                </div>
            ))}
        </div>
    );
};
TestingPanel.propTypes = {
    isTrained:        PropTypes.bool.isRequired,
    classifierRef:    PropTypes.object.isRequired,
    mobileNetRef:     PropTypes.object.isRequired,
    labels:           PropTypes.array.isRequired,
    selectedDeviceId: PropTypes.string
};

/* ──────────────────────────────────────────────
   Main training page
────────────────────────────────────────────── */
const MLTrainingPage = ({project, onBack, onUseInBlocks, onUpdateProject}) => {
    const [labels,         setLabels]      = useState(project.labels || ['Class 1', 'Class 2']);
    const [trainingData,   setData]        = useState({});
    const [loadingData,    setLoadingData] = useState(true);
    const [isTrained,      setTrained]     = useState(!!project.trained);
    const [isTraining,     setTraining]    = useState(false);
    const [trainPct,       setTrainPct]    = useState(0);
    const [trainStatus,    setStatus]      = useState('');
    const [accuracyPoints, setAccPoints]   = useState([]);
    const [showAdvanced,   setAdvanced]    = useState(false);
    const [epochs,         setEpochs]      = useState(20);
    const [batchSize,      setBatchSize]   = useState(16);
    const [learningRate,   setLR]          = useState(0.001);

    const [cameras,     setCameras]    = useState([]);
    const [selectedCam, setSelectedCam] = useState('');

    const classifierRef = useRef(null);
    const mobileNetRef  = useRef(null);

    /* ── Live training API ref — always current, exposed via bridge ── */
    const trainingAPIRef = useRef({});

    const canvasRef     = useRef(null);
    const classCardRefs = useRef([]);
    const trainCardRef  = useRef(null);
    const testCardRef   = useRef(null);
    const [svgPaths,    setSvgPaths]   = useState([]);

    /* ── Keep trainingAPI ref live (updated every render — no stale closures) ── */
    trainingAPIRef.current = {
        addTrainingImage: addImages,
        startTraining:    trainModel,
        getStatus:        () => isTraining ? 'training' : isTrained ? 'ready' : 'idle',
        clearTraining:    () => { setData({}); setTrained(false); },
        labels,
        trainingData
    };

    /* ── Push live status updates to the bridge when training state changes ── */
    useEffect(() => {
        const model = typeof window !== 'undefined' && window.__openblockMLModel;
        if (!model || model.projectId !== project.id) return;
        model.trainingStatus = isTraining ? 'training' : isTrained ? 'ready' : 'idle';
        model.labels         = labels;
        model.classifier     = classifierRef.current;
        model.mobileNet      = mobileNetRef.current;
    }, [isTraining, isTrained, labels]); // eslint-disable-line react-hooks/exhaustive-deps

    /* ── Enumerate cameras ── */
    useEffect(() => {
        navigator.mediaDevices.enumerateDevices()
            .then(devs => {
                const cams = devs.filter(d => d.kind === 'videoinput');
                setCameras(cams);
                if (cams.length > 0) setSelectedCam(cams[0].deviceId);
            })
            .catch(() => {});
    }, []);

    /* ── Load project data from IDB ── */
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const enriched = await loadProjectImages(project.id, project.trainingData || {});
                if (!cancelled) setData(enriched);
                if (project.trained) {
                    const restored = await loadImageClassifier(project.id, project.labels || ['Class 1', 'Class 2']);
                    if (restored && !cancelled) {
                        classifierRef.current = restored;
                        getMobileNet(s => setStatus(s)).then(net => {
                            if (!cancelled) mobileNetRef.current = net;
                        }).catch(() => {});
                        setTrained(true);
                    }
                }
            } finally {
                if (!cancelled) setLoadingData(false);
            }
        })();
        return () => { cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    /* ── Persist metadata ── */
    useEffect(() => {
        if (loadingData) return;
        const metaOnly = {};
        for (const [lbl, exs] of Object.entries(trainingData)) {
            metaOnly[lbl] = (exs || []).map(ex => ex.type === 'image' ? {id: ex.id, type: 'image'} : ex);
        }
        onUpdateProject({...project, labels, trainingData: metaOnly, trained: isTrained, updatedAt: Date.now()});
    }, [labels, trainingData, isTrained]); // eslint-disable-line react-hooks/exhaustive-deps

    /* ── Add images ── */
    const addImages = useCallback(async (label, dataUrls) => {
        if (!dataUrls.length) return;
        const newExs = dataUrls.map(data => ({id: generateId(), type: 'image', data}));
        for (const ex of newExs) await saveImageToIDB(project.id, ex.id, ex.data);
        setData(d => ({...d, [label]: [...(d[label] || []), ...newExs]}));
        setTrained(false);
    }, [project.id]);

    /* ── Rename / Delete / Add class ── */
    const renameClass = useCallback((oldName, newName) => {
        if (!newName || newName === oldName || labels.includes(newName)) return;
        setLabels(l => l.map(x => x === oldName ? newName : x));
        setData(d => { const c = {...d}; if (c[oldName]) { c[newName] = c[oldName]; delete c[oldName]; } return c; });
        setTrained(false);
    }, [labels]);

    const deleteClass = useCallback(name => {
        if (labels.length <= 2) return;
        setLabels(l => l.filter(x => x !== name));
        setData(d => { const c = {...d}; delete c[name]; return c; });
        setTrained(false);
    }, [labels]);

    const deleteSample = useCallback((label, sampleId) => {
        setData(d => ({...d, [label]: (d[label] || []).filter(ex => ex.id !== sampleId)}));
        setTrained(false);
    }, []);

    const addClass = () => {
        let n = labels.length + 1;
        while (labels.includes(`Class ${n}`)) n++;
        setLabels(l => [...l, `Class ${n}`]);
        setTrained(false);
    };

    /* ── Upload from folder ── */
    const folderInputRef = useRef(null);
    const handleFolderUpload = async e => {
        const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'));
        const groups = {};
        for (const f of files) {
            const parts = f.webkitRelativePath ? f.webkitRelativePath.split('/') : [f.name];
            const cls = parts.length > 2 ? parts[1] : (parts.length === 2 ? parts[0] : 'Class 1');
            if (!groups[cls]) groups[cls] = [];
            groups[cls].push(f);
        }
        const allLabels = new Set(labels);
        for (const cls of Object.keys(groups)) allLabels.add(cls);
        setLabels([...allLabels]);
        for (const [cls, fs] of Object.entries(groups)) {
            const urls = await Promise.all(fs.map(f => new Promise(res => {
                const r = new FileReader(); r.onload = () => res(r.result); r.readAsDataURL(f);
            })));
            await addImages(cls, urls);
        }
        e.target.value = '';
    };

    /* ── Train model ── */
    const trainModel = async () => {
        const total = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
        if (total < 2 || !labels.every(l => (trainingData[l] || []).length >= 1)) {
            setStatus('Every class needs at least 1 image.');
            return;
        }
        setTraining(true);
        setTrainPct(0);
        setAccPoints([{x: 0, y: 0}]);
        setStatus('Loading MobileNetV2…');

        try {
            const net = await getMobileNet(s => setStatus(s));
            mobileNetRef.current = net;

            const classifier = await trainImages(
                labels, trainingData, project.id,
                s => setStatus(s),
                pct => setTrainPct(pct),
                {
                    epochs,
                    batchSize,
                    learningRate,
                    onEpochEnd: (epochIdx, acc) => {
                        setAccPoints(pts => [...pts, {x: epochIdx + 1, y: acc}]);
                    }
                }
            );

            classifierRef.current = classifier;
            setTrained(true);
            setTrainPct(100);
            setStatus('Training Complete');
        } catch (err) {
            setStatus(`Error: ${err.message}`);
            console.error('[ML Train]', err);
        } finally {
            setTraining(false);
        }
    };

    /* ── Deploy model to window.__openblockMLModel ── */
    const deployModel = () => {
        setActiveModel({
            projectId:     project.id,
            projectName:   project.name,
            type:          'images',
            labels,
            classifier:    classifierRef.current,
            mobileNet:     mobileNetRef.current,
            trainingStatus: isTraining ? 'training' : isTrained ? 'ready' : 'idle',
            // Pass ref itself so VM extension always calls through to latest closures
            _trainingAPI:  trainingAPIRef
        });
    };

    /* ── Export model: deploy + download manifest JSON ── */
    const handleExportModel = async () => {
        if (!classifierRef.current) return;
        deployModel();
        try {
            const payload = JSON.stringify({
                version:     2,
                type:        'images',
                backbone:    'mobilenet_v2_100_224',
                labels,
                projectName: project.name,
                savedAt:     new Date().toISOString()
            }, null, 2);
            const blob = new Blob([payload], {type: 'application/json'});
            const url  = URL.createObjectURL(blob);
            const a    = document.createElement('a');
            a.href = url;
            a.download = `${project.name || 'ml-model'}.json`;
            a.click();
            URL.revokeObjectURL(url);
        } catch (e) { console.error('[Export]', e); }
        onUseInBlocks();
    };

    /* ── Use in Blocks ── */
    const handleUseInBlocks = () => {
        deployModel();
        onUseInBlocks();
    };

    /* ── SVG curves ── */
    const recalcCurves = useCallback(() => {
        if (!canvasRef.current || !trainCardRef.current || !testCardRef.current) return;
        const canvasEl   = canvasRef.current;
        const canvasRect = canvasEl.getBoundingClientRect();
        const trainRect  = trainCardRef.current.getBoundingClientRect();
        const testRect   = testCardRef.current.getBoundingClientRect();
        const scrollLeft = canvasEl.scrollLeft;
        const scrollTop  = canvasEl.scrollTop;
        const paths = [];

        classCardRefs.current.forEach((r, i) => {
            if (!r) return;
            const rect = r.getBoundingClientRect();
            const x1 = rect.right - canvasRect.left + scrollLeft;
            const y1 = rect.top + rect.height / 2 - canvasRect.top + scrollTop;
            const x2 = trainRect.left - canvasRect.left + scrollLeft;
            const y2 = trainRect.top + trainRect.height / 2 - canvasRect.top + scrollTop;
            const cx = Math.max(40, (x2 - x1) / 2);
            paths.push({
                color: CLASS_COLORS[i % CLASS_COLORS.length],
                d: `M ${x1} ${y1} C ${x1 + cx} ${y1} ${x2 - cx} ${y2} ${x2} ${y2}`
            });
        });

        const tx1 = trainRect.right - canvasRect.left + scrollLeft;
        const ty1 = trainRect.top + trainRect.height / 2 - canvasRect.top + scrollTop;
        const tx2 = testRect.left - canvasRect.left + scrollLeft;
        const ty2 = testRect.top + testRect.height / 2 - canvasRect.top + scrollTop;
        const tcx = Math.max(40, (tx2 - tx1) / 2);
        paths.push({
            color: '#6a00b0',
            d: `M ${tx1} ${ty1} C ${tx1 + tcx} ${ty1} ${tx2 - tcx} ${ty2} ${tx2} ${ty2}`
        });

        setSvgPaths(prev => {
            if (prev.length === paths.length && prev.every((p, i) => p.d === paths[i].d && p.color === paths[i].color)) return prev;
            return paths;
        });
    }, []);

    useLayoutEffect(() => { recalcCurves(); });
    useEffect(() => {
        window.addEventListener('resize', recalcCurves);
        return () => window.removeEventListener('resize', recalcCurves);
    }, [recalcCurves]);

    const canTrain = !isTraining &&
        labels.every(l => (trainingData[l] || []).length >= 1) &&
        labels.reduce((s, l) => s + (trainingData[l] || []).length, 0) >= 2;

    if (loadingData) {
        return (
            <div className={styles.page} style={{alignItems: 'center', justifyContent: 'center', background: '#ebebeb'}}>
                <p style={{color: '#666', fontSize: 15}}>Loading project data…</p>
            </div>
        );
    }

    return (
        <div className={styles.page}>
            {/* Header */}
            <div className={styles.header}>
                <span style={{fontSize: 22}}>🐻</span>
                <span className={styles.headerTitle}>Machine Learning Environment</span>
                <nav className={styles.headerNav}>
                    <span>File</span>
                    <span>Tutorials</span>
                    <span>Help</span>
                </nav>
                <button className={styles.headerBackBtn} onClick={onBack}>&#8592; Back</button>
            </div>

            {/* Sub-header */}
            <div className={styles.subHeader}>
                <div className={styles.subHeaderIcon}>🖼</div>
                <span className={styles.subHeaderType}>Image Classifier</span>
                <span className={styles.infoIcon} title="Train an image classification model.">i</span>
                <div className={styles.divider}/>
                <span className={styles.projectNamePill}>{project.name}</span>
                <div className={styles.divider}/>
                <button className={styles.uploadFolderBtn} onClick={() => folderInputRef.current && folderInputRef.current.click()}>
                    Upload Classes from Folder
                </button>
                <input type="file" webkitdirectory="" multiple accept="image/*" style={{display: 'none'}} ref={folderInputRef} onChange={handleFolderUpload}/>
                <button className={styles.saveBtn} title="Project auto-saves">💾</button>
                <div className={styles.spacer}/>
                <span className={styles.webcamLabel}>Select Webcam:</span>
                <select className={styles.webcamSelect} value={selectedCam} onChange={e => setSelectedCam(e.target.value)}>
                    {cameras.length === 0 && <option value="">No cameras found</option>}
                    {cameras.map(c => (
                        <option key={c.deviceId} value={c.deviceId}>{c.label || `Camera (${c.deviceId.slice(0, 8)})`}</option>
                    ))}
                </select>
            </div>

            {/* Canvas */}
            <div className={styles.canvas} ref={canvasRef}>
                <svg className={styles.curveSvg}>
                    {svgPaths.map((p, i) => (
                        <path key={i} d={p.d} stroke={p.color} strokeWidth="2.5" fill="none" opacity="0.85"/>
                    ))}
                </svg>

                {/* Classes column */}
                <div className={styles.classesColumn}>
                    {labels.map((lbl, i) => (
                        <ClassCard
                            key={lbl}
                            ref={el => { classCardRefs.current[i] = el; }}
                            label={lbl}
                            colorIdx={i}
                            samples={trainingData[lbl] || []}
                            projectId={project.id}
                            selectedDeviceId={selectedCam}
                            canDelete={labels.length > 2}
                            onAddImages={addImages}
                            onDeleteSample={deleteSample}
                            onRename={renameClass}
                            onDelete={deleteClass}
                        />
                    ))}
                    <button className={styles.addClassCard} onClick={addClass}>+ Add Class</button>
                </div>

                {/* Right: Training + Testing */}
                <div className={styles.rightArea}>
                    {/* Training card */}
                    <div className={styles.trainingCard} ref={trainCardRef}>
                        <div className={styles.trainingHeader}>
                            <span>Training</span>
                            <div className={styles.langToggle}>
                                <span>🐍</span>
                                <div className={styles.langToggleSwitch}/>
                                <span style={{color: '#ffe000', fontWeight: 700}}>JS</span>
                            </div>
                        </div>
                        <div className={styles.trainingBody}>
                            {/* Accuracy chart */}
                            {accuracyPoints.length >= 2 && (
                                <div className={styles.chartWrap}>
                                    <AccuracyChart points={accuracyPoints}/>
                                </div>
                            )}

                            {/* Status */}
                            {trainStatus && (
                                <p className={isTrained && trainStatus === 'Training Complete' ? styles.trainCompletedText : styles.trainStatusText}>
                                    {trainStatus}
                                </p>
                            )}

                            {/* Progress bar while training */}
                            {isTraining && (
                                <div className={styles.progressTrack}>
                                    <div className={styles.progressFill} style={{width: `${trainPct}%`}}/>
                                </div>
                            )}

                            {/* Train / Train Again button */}
                            {isTrained ? (
                                <button className={styles.trainAgainBtn} onClick={trainModel} disabled={isTraining || !canTrain}>
                                    Train Again
                                </button>
                            ) : (
                                <button className={styles.trainModelBtn} onClick={trainModel} disabled={!canTrain}>
                                    {isTraining ? '⏳ Training…' : '⚡ Train Model'}
                                </button>
                            )}

                            {/* Use in Blocks */}
                            {isTrained && (
                                <button className={styles.useInBlocksBtn} onClick={handleUseInBlocks}>
                                    🧩 Use in Blocks
                                </button>
                            )}
                        </div>

                        {/* Advanced accordion */}
                        <div className={styles.trainingFooter} onClick={() => setAdvanced(a => !a)}>
                            <span>Advanced</span>
                            <span>{showAdvanced ? '▲' : '▼'}</span>
                        </div>
                        {showAdvanced && (
                            <div className={styles.advancedSection}>
                                <div className={styles.advancedRow}>
                                    <span>Epochs</span>
                                    <input className={styles.advancedInput} type="number" value={epochs} min={1} max={200}
                                        onChange={e => setEpochs(Number(e.target.value))}/>
                                </div>
                                <div className={styles.advancedRow}>
                                    <span>Batch Size</span>
                                    <input className={styles.advancedInput} type="number" value={batchSize} min={1} max={512}
                                        onChange={e => setBatchSize(Number(e.target.value))}/>
                                </div>
                                <div className={styles.advancedRow}>
                                    <span>Learning Rate</span>
                                    <input className={styles.advancedInput} type="number" value={learningRate} step="0.0001" min={0.0001}
                                        onChange={e => setLR(Number(e.target.value))}/>
                                </div>
                                <div className={styles.advancedBtnsRow}>
                                    <button className={styles.trainReportBtn} onClick={() => alert(`Model: Image Classifier\nClasses: ${labels.join(', ')}\nSamples: ${labels.reduce((s, l) => s + (trainingData[l] || []).length, 0)}\nStatus: ${isTrained ? 'Trained' : 'Not trained'}`)}>
                                        Train Report
                                    </button>
                                    <button className={styles.resetBtn} onClick={() => { setEpochs(20); setBatchSize(16); setLR(0.001); }}>
                                        Reset
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Testing card */}
                    <div className={styles.testingCard} ref={testCardRef}>
                        <div className={styles.testingHeader}>
                            <span>Testing</span>
                            {isTrained && (
                                <button className={styles.exportModelBtn} onClick={handleExportModel} title="Export trained model as JSON">
                                    &#8593; Export Model
                                </button>
                            )}
                        </div>
                        <div className={styles.testingBody}>
                            <TestingPanel
                                isTrained={isTrained}
                                classifierRef={classifierRef}
                                mobileNetRef={mobileNetRef}
                                labels={labels}
                                selectedDeviceId={selectedCam}
                            />
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

MLTrainingPage.propTypes = {
    project:         PropTypes.object.isRequired,
    onBack:          PropTypes.func.isRequired,
    onUseInBlocks:   PropTypes.func.isRequired,
    onUpdateProject: PropTypes.func.isRequired
};

export default MLTrainingPage;
