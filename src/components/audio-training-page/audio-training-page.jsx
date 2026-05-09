import React, {useState, useRef, useEffect, useLayoutEffect, useCallback} from 'react';
import PropTypes from 'prop-types';
import styles from './audio-training-page.css';
import openblockLogo from '../openblock-logo.svg';
import Loader from 'openblock-gui/src/components/loader/loader.jsx';
import Spinner from 'openblock-gui/src/components/spinner/spinner.jsx';

import {
    initSpeechCommands,
    collectAudioExample,
    trainSounds,
    loadSoundClassifier,
    startListening,
    stopListening,
    setActiveModel
} from '../../lib/ml-engine.js';
import {
    saveAudioToIDB,
    saveAudioThumbToIDB,
    getAudioThumbFromIDB,
    loadProjectAudio
} from '../../lib/idb.js';

const CLASS_COLORS = ['#E05C3D', '#2EAA7E', '#9966FF', '#774DCB', '#F39C12', '#E91E63', '#1ABC9C', '#E67E22'];
const MAX_THUMBS   = 9;
const generateId   = () => Math.random().toString(36).slice(2, 10);

/* ── Accuracy vs Epochs SVG chart ── */
const AccuracyChart = ({points}) => {
    if (!points || points.length < 2) return null;
    const W = 220, H = 150;
    const pL = 30, pR = 10, pT = 10, pB = 30;
    const cW = W - pL - pR, cH = H - pT - pB;
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
            <path d={lineD} stroke="#2453ff" strokeWidth="2.5" fill="none" strokeLinejoin="round" strokeLinecap="round"/>
            <circle cx={sx(points[points.length - 1].x)} cy={sy(points[points.length - 1].y)} r="4" fill="#2453ff"/>
            <text x={pL + cW / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="#888">Accuracy Vs Epochs</text>
        </svg>
    );
};
AccuracyChart.propTypes = {points: PropTypes.array};

/* ── AudioClassCard ── */
const AudioClassCard = React.forwardRef(({
    label, colorIdx, samples, onRecorded, onDeleteSample, onDeleteAllSamples,
    onRename, onDelete, canDelete, onToggleDisable, isDisabled, projectId, selectedMicId, isEngineReady,
    menuOpen, onOpenMenu, onCloseMenu,
    isMicActive, onRequestMic, onReleaseMic
}, ref) => {
    const color = CLASS_COLORS[colorIdx % CLASS_COLORS.length];
    const [editing,       setEditing]  = useState(false);
    const [newName,       setNewName]  = useState(label);
    const [confirmAction, setConfirm]  = useState(null);
    const [showMic,       setShowMic]  = useState(false);
    const [isRecording,   setRecording] = useState(false);
    const [thumbData,     setThumbData] = useState([]);
    const [micError,      setMicError]  = useState('');

    const liveCanvasRef  = useRef(null);
    const analyserRef    = useRef(null);
    const vizStreamRef   = useRef(null);
    const animFrameRef   = useRef(null);
    const isHoldingRef   = useRef(false);
    const isRecordingRef = useRef(false);
    const menuRef        = useRef(null);

    /* Close menu when clicking outside */
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

    /* Load waveform thumbnails from IDB */
    useEffect(() => {
        let cancelled = false;
        (async () => {
            const thumbs = await Promise.all(
                samples.map(async s => {
                    const src = await getAudioThumbFromIDB(projectId, s.id);
                    return src ? {src, id: s.id} : null;
                })
            );
            if (!cancelled) setThumbData(thumbs.filter(Boolean));
        })();
        return () => { cancelled = true; };
    }, [samples, projectId]);

    /* Stop mic when another class takes ownership */
    useEffect(() => {
        if (!isMicActive && vizStreamRef.current) {
            isHoldingRef.current = false;
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
            vizStreamRef.current.getTracks().forEach(t => t.stop());
            vizStreamRef.current = null;
            analyserRef.current  = null;
            setShowMic(false);
            setRecording(false);
            setMicError('');
        }
    }, [isMicActive]);

    /* Open visualization mic stream */
    const friendlyMicError = e => {
        const n = e.name || '';
        if (n === 'NotAllowedError' || n === 'PermissionDeniedError')
            return {title: 'Permission Denied', msg: 'Microphone access was blocked. Allow microphone permission in your browser or system settings.'};
        if (n === 'NotFoundError' || n === 'DevicesNotFoundError')
            return {title: 'No Microphone Found', msg: 'No microphone device was detected. Connect a microphone and try again.'};
        if (n === 'NotReadableError' || n === 'TrackStartError')
            return {title: 'Microphone Busy', msg: 'Your microphone might be open in another application. Close it and try again.'};
        return {title: 'Microphone Error', msg: e.message || 'Could not start the microphone. Please try again.'};
    };

    const startMic = async () => {
        onRequestMic();
        setMicError('');
        setShowMic(true);
        try {
            const constraints = {
                audio: selectedMicId ? {deviceId: {exact: selectedMicId}} : true,
                video: false
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            vizStreamRef.current = stream;
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source   = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            analyserRef.current = analyser;
        } catch (e) {
            const {title, msg} = friendlyMicError(e);
            setMicError(`${title}|||${msg}`);
            onReleaseMic();
        }
    };

    const stopMic = () => {
        isHoldingRef.current = false;
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        if (vizStreamRef.current) vizStreamRef.current.getTracks().forEach(t => t.stop());
        vizStreamRef.current = null;
        analyserRef.current  = null;
        setShowMic(false);
        setRecording(false);
        setMicError('');
        onReleaseMic();
    };

    /* Live waveform animation */
    useEffect(() => {
        if (!showMic || !analyserRef.current) return;
        const analyser = analyserRef.current;
        const buf      = new Uint8Array(analyser.frequencyBinCount);
        const draw = () => {
            animFrameRef.current = requestAnimationFrame(draw);
            const canvas = liveCanvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            analyser.getByteTimeDomainData(buf);
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = color;
            ctx.lineWidth   = 2;
            ctx.beginPath();
            const sliceW = canvas.width / buf.length;
            let x = 0;
            for (let i = 0; i < buf.length; i++) {
                const v = buf[i] / 128.0;
                const y = (v * canvas.height) / 2;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                x += sliceW;
            }
            ctx.stroke();
        };
        draw();
        return () => { if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current); };
    }, [showMic, color]);

    /* Cleanup on unmount */
    useEffect(() => () => stopMic(), []); // eslint-disable-line react-hooks/exhaustive-deps

    /* Record one sample, loop while button held */
    const doRecord = useCallback(async () => {
        if (!isHoldingRef.current || isRecordingRef.current) return;
        isRecordingRef.current = true;
        setRecording(true);
        try {
            const specData = await collectAudioExample(label);
            /* Snapshot the live canvas as the waveform thumbnail */
            const thumbUrl = liveCanvasRef.current
                ? liveCanvasRef.current.toDataURL('image/png')
                : null;
            const id = generateId();
            await saveAudioToIDB(projectId, id, Array.from(specData.data), specData.frameSize);
            if (thumbUrl) await saveAudioThumbToIDB(projectId, id, thumbUrl);
            onRecorded(label, {
                id,
                type: 'audio',
                spectrogramData: Array.from(specData.data),
                frameSize: specData.frameSize
            });
        } catch (err) {
            console.error('[AudioCard] recording error:', err);
        }
        isRecordingRef.current = false;
        setRecording(false);
        if (isHoldingRef.current) doRecord(); // continue while holding
    }, [label, projectId, onRecorded]); // eslint-disable-line react-hooks/exhaustive-deps

    const startHold = useCallback(e => {
        e.preventDefault();
        if (!isEngineReady) return;
        isHoldingRef.current = true;
        doRecord();
    }, [isEngineReady, doRecord]);

    const endHold = useCallback(e => {
        e.preventDefault();
        isHoldingRef.current = false;
    }, []);

    const commitRename = () => {
        const n = newName.trim();
        if (n && n !== label) onRename(label, n);
        setEditing(false);
    };

    return (
        <div className={`${styles.classCard}${isDisabled ? ` ${styles.classCardDisabled}` : ''}`} ref={ref}>
            {/* Header */}
            <div className={styles.classCardHeader} style={{background: color}}>
                <div className={styles.classCardNameRow}>
                    {editing ? (
                        <input className={styles.nameEditInput} value={newName} autoFocus
                            onChange={e => setNewName(e.target.value)}
                            onKeyPress={e => e.key === 'Enter' && commitRename()}
                            onBlur={commitRename}/>
                    ) : (
                        <span className={styles.classCardTitle}>{label}</span>
                    )}
                    <button className={styles.classCardEditBtn}
                        onClick={() => { setEditing(true); setNewName(label); }}
                        title="Rename">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>
                        </svg>
                    </button>
                </div>
                {isDisabled && <span className={styles.disabledBadge}>DISABLED</span>}
                <div style={{position: 'relative'}} ref={menuRef}>
                    <button className={styles.classCardMenuBtn} onClick={() => menuOpen ? onCloseMenu() : onOpenMenu()}>&#8942;</button>
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
                                            onCloseMenu(); setConfirm(null);
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
            <div className={`${styles.classCardBody}${showMic ? ` ${styles.classCardBodyCam}` : ''}`}>
                {/* Left: mic controls */}
                <div className={styles.classCardLeft}>
                    {showMic ? (
                        <div className={styles.micRecordView}>
                            <div className={styles.micHeader}>
                                <span className={styles.micLabel}>Microphone</span>
                                <button className={styles.camBackBtn} onClick={stopMic} title="Back">&#8592;</button>
                            </div>
                            {micError ? (() => {
                                const [title, msg] = micError.split('|||');
                                return (
                                    <div className={styles.resourceError}>
                                        <div className={styles.resourceErrorIcon}>!</div>
                                        <p className={styles.resourceErrorTitle}>{title}</p>
                                        <p className={styles.resourceErrorMsg}>{msg}</p>
                                        <button className={styles.resourceRetryBtn} onClick={startMic}>Try Again</button>
                                    </div>
                                );
                            })() : (
                                <>
                                    <canvas ref={liveCanvasRef} className={styles.liveWaveform} width={200} height={110}/>
                                    <button
                                        className={`${styles.holdBtn}${isRecording ? ` ${styles.holdBtnRecording}` : ''}`}
                                        onMouseDown={startHold}
                                        onMouseUp={endHold}
                                        onMouseLeave={endHold}
                                        onTouchStart={startHold}
                                        onTouchEnd={endHold}
                                        disabled={!isEngineReady}
                                    >
                                        {isRecording ? '● Recording…' : 'Hold to record'}
                                    </button>
                                </>
                            )}
                        </div>
                    ) : (
                        <div className={styles.idleAddMode}>
                            <p className={styles.addSamplesLabel}>Add Audio Samples</p>
                            <button className={styles.micIdleBtn} onClick={startMic}>
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="28" height="28">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                    <line x1="12" y1="19" x2="12" y2="23"/>
                                    <line x1="8" y1="23" x2="16" y2="23"/>
                                </svg>
                                Microphone
                            </button>
                        </div>
                    )}
                </div>

                {/* Right: sample thumbnails */}
                <div className={styles.classCardRight}>
                    <p className={styles.sampleCountLabel}>{samples.length} Audio Sample{samples.length !== 1 ? 's' : ''}</p>
                    {thumbData.length > 0 && (
                        <div className={styles.thumbnailGrid}>
                            {thumbData.map((item) => (
                                <div key={item.id} className={styles.thumb}>
                                    <img src={item.src} alt="" className={styles.waveThumb}/>
                                    <button
                                        className={styles.thumbDeleteBtn}
                                        onClick={e => { e.stopPropagation(); onDeleteSample(label, item.id); }}
                                        title="Remove">&#10005;</button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
});
AudioClassCard.displayName = 'AudioClassCard';
AudioClassCard.propTypes = {
    label:              PropTypes.string.isRequired,
    colorIdx:           PropTypes.number.isRequired,
    samples:            PropTypes.array.isRequired,
    onRecorded:         PropTypes.func.isRequired,
    onDeleteSample:     PropTypes.func.isRequired,
    onDeleteAllSamples: PropTypes.func.isRequired,
    onRename:           PropTypes.func.isRequired,
    onDelete:           PropTypes.func.isRequired,
    canDelete:          PropTypes.bool.isRequired,
    onToggleDisable:    PropTypes.func.isRequired,
    isDisabled:         PropTypes.bool.isRequired,
    projectId:          PropTypes.string.isRequired,
    selectedMicId:      PropTypes.string,
    isEngineReady:      PropTypes.bool.isRequired,
    menuOpen:           PropTypes.bool.isRequired,
    onOpenMenu:         PropTypes.func.isRequired,
    onCloseMenu:        PropTypes.func.isRequired,
    isMicActive:        PropTypes.bool.isRequired,
    onRequestMic:       PropTypes.func.isRequired,
    onReleaseMic:       PropTypes.func.isRequired
};

/* ── Audio Testing Panel ── */
const AudioTestingPanel = ({isTrained}) => {
    const [isListening, setListening] = useState(false);
    const [results,     setResults]   = useState([]);
    const [listenErr,   setListenErr] = useState('');

    const toggle = async () => {
        if (isListening) {
            setListening(false);
            setResults([]);
            await stopListening();
        } else {
            setListenErr('');
            try {
                await startListening(matches => setResults(matches));
                setListening(true);
            } catch (e) {
                setListenErr(e.message || 'Could not start listening');
            }
        }
    };

    useEffect(() => () => { stopListening(); }, []);

    if (!isTrained) {
        return <p className={styles.testingPlaceholder}>Train a model first, then test it here.</p>;
    }

    return (
        <div className={styles.testPanelContent}>
            {!isListening && <p className={styles.testByLabel}>Test Audio By</p>}
            <div className={styles.testOptionRow}>
                <button
                    className={`${styles.testOptionBtn}${isListening ? ` ${styles.testOptionBtnActive}` : ''}`}
                    onClick={toggle}
                >
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="26" height="26">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                    </svg>
                    {isListening ? 'Stop' : 'Microphone'}
                </button>
            </div>
            {listenErr && <p style={{color: '#e53935', fontSize: 11}}>{listenErr}</p>}
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
AudioTestingPanel.propTypes = {isTrained: PropTypes.bool.isRequired};

/* ── Main Audio Training Page ── */
const AudioTrainingPage = ({project, onBack, onUseInBlocks, onUpdateProject, onNewProject, onNewMLProject, onOpenMLProject}) => {
    const [labels,        setLabels]   = useState(project.labels || ['Class 1', 'Class 2']);
    const [trainingData,  setData]     = useState({});
    const [loadingData,   setLoading]  = useState(true);
    const [engineReady,   setReady]    = useState(false);
    const [engineStatus,  setEngStatus] = useState('Initializing audio engine…');
    const [isTrained,     setTrained]  = useState(!!project.trained);
    const [isTraining,    setTraining] = useState(false);
    const [trainPct,      setTrainPct] = useState(0);
    const [trainStatus,   setStatus]  = useState('');
    const [accPoints,     setAccPts]  = useState([]);
    const [showAdvanced,  setAdvanced] = useState(false);
    const [epochs,        setEpochs]  = useState(25);
    const [batchSize,     setBatch]   = useState(16);
    const [mics,           setMics]          = useState([]);
    const [selectedMic,    setSelMic]         = useState('');
    const [activeMicLabel, setActiveMicLabel] = useState(null);
    const [openMenuLabel,  setOpenMenuLabel]  = useState(null);
    const [fileMenuOpen,  setFileMenuOpen] = useState(false);
    const fileMenuRef = useRef(null);

    useEffect(() => {
        if (!fileMenuOpen) return;
        const handler = e => {
            if (fileMenuRef.current && !fileMenuRef.current.contains(e.target)) {
                setFileMenuOpen(false);
            }
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, [fileMenuOpen]);

    const canvasRef      = useRef(null);
    const classCardRefs  = useRef([]);
    const trainCardRef   = useRef(null);
    const testCardRef    = useRef(null);
    const [svgPaths,     setSvgPaths] = useState([]);

    /* Enumerate microphones */
    useEffect(() => {
        navigator.mediaDevices.enumerateDevices()
            .then(devs => {
                const ms = devs.filter(d => d.kind === 'audioinput');
                setMics(ms);
                if (ms.length > 0) setSelMic(ms[0].deviceId);
            })
            .catch(() => {});
    }, []);

    /* Init speech-commands + load stored audio samples */
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                await initSpeechCommands(project.id, s => { if (!cancelled) setEngStatus(s); });
                if (!cancelled) { setReady(true); setEngStatus(''); }

                const enriched = await loadProjectAudio(project.id, project.trainingData || {});
                if (!cancelled) setData(enriched);

                if (project.trained) {
                    const loaded = await loadSoundClassifier(
                        project.id, project.labels || labels
                    );
                    if (loaded && !cancelled) setTrained(true);
                }
            } catch (err) {
                if (!cancelled) setEngStatus(`Error: ${err.message}`);
                console.error('[AudioPage] init:', err);
            } finally {
                if (!cancelled) setLoading(false);
            }
        })();
        return () => { cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    /* Persist metadata on change */
    useEffect(() => {
        if (loadingData) return;
        const metaOnly = {};
        for (const [lbl, exs] of Object.entries(trainingData)) {
            metaOnly[lbl] = (exs || []).map(ex => ({id: ex.id, type: ex.type}));
        }
        onUpdateProject({...project, labels, trainingData: metaOnly, trained: isTrained, updatedAt: Date.now()});
    }, [labels, trainingData, isTrained]); // eslint-disable-line react-hooks/exhaustive-deps

    /* CRUD helpers */
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

    const [disabledLabels, setDisabledLabels] = useState([]);
    const toggleDisableClass = useCallback(label => {
        setDisabledLabels(prev =>
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

    /* Train */
    const trainModel = async () => {
        const activeTrainLabels = labels.filter(l => !disabledLabels.includes(l));
        const total = activeTrainLabels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
        if (activeTrainLabels.length < 2 || !activeTrainLabels.every(l => (trainingData[l] || []).length >= 10)) {
            setStatus('Need at least 2 enabled classes, each with at least 10 audio samples.');
            return;
        }
        setTraining(true);
        setTrainPct(0);
        setAccPts([{x: 0, y: 0}]);
        setStatus('Preparing…');
        try {
            await trainSounds(
                activeTrainLabels, trainingData, project.id,
                s => setStatus(s),
                pct => setTrainPct(pct),
                {
                    epochs,
                    batchSize,
                    onEpochEnd: (epochIdx, acc) => {
                        setAccPts(pts => [...pts, {x: epochIdx + 1, y: acc}]);
                    }
                }
            );
            setTrained(true);
            setTrainPct(100);
        } catch (err) {
            setStatus(`Error: ${err.message}`);
            console.error('[AudioTrain]', err);
        } finally {
            setTraining(false);
        }
    };

    /* Deploy */
    const deployModel = () => {
        setActiveModel({
            projectId:      project.id,
            projectName:    project.name,
            type:           'sounds',
            labels,
            trainingStatus: isTrained ? 'ready' : 'idle',
            startListening,
            stopListening
        });
    };

    const handleExport = () => {
        deployModel();
        onUseInBlocks();
    };

    /* SVG bezier curves between cards */
    const recalcCurves = useCallback(() => {
        if (!canvasRef.current || !trainCardRef.current || !testCardRef.current) return;
        const wrap      = canvasRef.current;
        const wrapRect  = wrap.getBoundingClientRect();
        const trainRect = trainCardRef.current.getBoundingClientRect();
        const testRect  = testCardRef.current.getBoundingClientRect();
        const sl = wrap.scrollLeft, st = wrap.scrollTop;
        const paths = [];

        classCardRefs.current.forEach((r, i) => {
            if (!r) return;
            const rect = r.getBoundingClientRect();
            const x1 = rect.right  - wrapRect.left + sl;
            const y1 = rect.top + rect.height / 2 - wrapRect.top + st;
            const x2 = trainRect.left - wrapRect.left + sl;
            const y2 = trainRect.top + trainRect.height / 2 - wrapRect.top + st;
            const cx = Math.max(40, (x2 - x1) / 2);
            paths.push({
                color: CLASS_COLORS[i % CLASS_COLORS.length],
                d: `M ${x1} ${y1} C ${x1 + cx} ${y1} ${x2 - cx} ${y2} ${x2} ${y2}`
            });
        });

        const tx1 = trainRect.right - wrapRect.left + sl;
        const ty1 = trainRect.top + trainRect.height / 2 - wrapRect.top + st;
        const tx2 = testRect.left  - wrapRect.left + sl;
        const ty2 = testRect.top + testRect.height / 2 - wrapRect.top + st;
        const tcx = Math.max(40, (tx2 - tx1) / 2);
        paths.push({
            color: '#9966FF',
            d: `M ${tx1} ${ty1} C ${tx1 + tcx} ${ty1} ${tx2 - tcx} ${ty2} ${tx2} ${ty2}`
        });

        setSvgPaths(prev =>
            prev.length === paths.length && prev.every((p, i) => p.d === paths[i].d) ? prev : paths
        );
    }, []);

    useLayoutEffect(() => { recalcCurves(); });
    useEffect(() => {
        window.addEventListener('resize', recalcCurves);
        return () => window.removeEventListener('resize', recalcCurves);
    }, [recalcCurves]);

    const activeLabels = labels.filter(l => !disabledLabels.includes(l));
    const canTrain = !isTraining && engineReady &&
        activeLabels.length >= 2 &&
        activeLabels.every(l => (trainingData[l] || []).length >= 10);

    if (loadingData) {
        return <Loader messageId="gui.loader.headline" />;
    }

    return (
        <div className={styles.page}>
            {/* Top header */}
            <div className={styles.header}>
                <img
                    src={openblockLogo}
                    alt="RoboCoders Studio"
                    className={styles.headerLogo}
                    draggable={false}
                />
                <span className={styles.headerTitle}>Machine Learning Environment</span>
                <nav className={styles.headerNav}>
                    <div className={styles.fileMenuWrap} ref={fileMenuRef}>
                        <button className={styles.navBtn} onClick={() => setFileMenuOpen(o => !o)}>File</button>
                        {fileMenuOpen && (
                            <div className={styles.navDropdown}>
                                <button onClick={() => { setFileMenuOpen(false); (onNewProject || onBack)(); }}>New</button>
                                <button onClick={() => { setFileMenuOpen(false); (onNewMLProject || onBack)(); }}>New ML Project</button>
                                <button onClick={() => { setFileMenuOpen(false); (onOpenMLProject || onBack)(); }}>Open ML Project</button>
                                <button onClick={() => { setFileMenuOpen(false); document.documentElement.requestFullscreen && document.documentElement.requestFullscreen(); }}>Full Screen Recording</button>
                                <button onClick={() => setFileMenuOpen(false)}>Examples</button>
                            </div>
                        )}
                    </div>
                    <button className={styles.navBtn}>Tutorials</button>
                    <button className={styles.navBtn}>Help</button>
                </nav>
                <div className={styles.headerSpacer} />
                <button className={styles.headerBackBtn} onClick={onBack}>&#8592; Back</button>
            </div>

            {/* Sub-header */}
            <div className={styles.subHeader}>
                <span className={styles.subHeaderType}>Audio Classifier</span>
                <span className={styles.infoIcon} title="Train a model to recognise sounds from your microphone — then use it in your Blocks project.">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="#9966FF" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="11"/>
                        <circle cx="12" cy="8" r="1.3" fill="white"/>
                        <rect x="10.7" y="11" width="2.6" height="6" rx="1.3" fill="white"/>
                    </svg>
                </span>
                <div className={styles.divider}/>
                <span className={styles.projectNamePill}>{project.name}</span>
                <div className={styles.divider}/>
                {engineStatus && <span style={{fontSize: 12, color: '#9966FF'}}>{engineStatus}</span>}
                <button className={styles.saveBtn} title="Project auto-saves">💾</button>
                <div className={styles.spacer}/>
                <span className={styles.webcamLabel}>Select Microphones:</span>
                <select className={styles.webcamSelect} value={selectedMic} onChange={e => setSelMic(e.target.value)}>
                    {mics.length === 0 && <option value="">No microphones found</option>}
                    {mics.map(m => (
                        <option key={m.deviceId} value={m.deviceId}>
                            {m.label || `Microphone (${m.deviceId.slice(0, 8)})`}
                        </option>
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
                        <AudioClassCard
                            key={lbl}
                            ref={el => { classCardRefs.current[i] = el; }}
                            label={lbl}
                            colorIdx={i}
                            samples={trainingData[lbl] || []}
                            projectId={project.id}
                            selectedMicId={selectedMic}
                            isEngineReady={engineReady}
                            canDelete={labels.length > 2}
                            isDisabled={disabledLabels.includes(lbl)}
                            menuOpen={openMenuLabel === lbl}
                            onOpenMenu={() => setOpenMenuLabel(lbl)}
                            onCloseMenu={() => setOpenMenuLabel(null)}
                            isMicActive={activeMicLabel === lbl}
                            onRequestMic={() => setActiveMicLabel(lbl)}
                            onReleaseMic={() => setActiveMicLabel(null)}
                            onRecorded={addSample}
                            onDeleteSample={deleteSample}
                            onDeleteAllSamples={deleteAllSamples}
                            onRename={renameClass}
                            onDelete={deleteClass}
                            onToggleDisable={toggleDisableClass}
                        />
                    ))}
                    <button className={styles.addClassCard} onClick={addClass}>+ Add Class</button>
                </div>

                {/* Right: Training + Testing */}
                <div className={styles.rightArea}>
                    {/* Training card */}
                    <div className={styles.trainingCard} ref={trainCardRef}>
                        <div className={styles.trainingHeader}><span>Training</span></div>
                        {disabledLabels.length > 0 && (
                            <div className={styles.activeClassInfo}>
                                {activeLabels.length} / {labels.length} classes active
                            </div>
                        )}
                        <div className={styles.trainingBody}>
                            {accPoints.length >= 2 && (
                                <div className={styles.chartWrap}>
                                    <AccuracyChart points={accPoints}/>
                                </div>
                            )}
                            {trainStatus && (
                                <p className={isTrained && trainStatus === 'Training Complete'
                                    ? styles.trainCompletedText : styles.trainStatusText}>
                                    {trainStatus}
                                </p>
                            )}
                            {isTraining && (
                                <div className={styles.progressTrack}>
                                    <div className={styles.progressFill} style={{width: `${trainPct}%`}}/>
                                </div>
                            )}
                            {isTrained ? (
                                <button className={styles.trainAgainBtn} onClick={trainModel}
                                    disabled={isTraining || !canTrain}>Train Again</button>
                            ) : (
                                <button className={styles.trainModelBtn} onClick={trainModel} disabled={!canTrain}>
                                    {isTraining ? <Spinner small level="info" /> : '⚡ Train Model'}
                                </button>
                            )}
                        </div>
                        <div className={styles.trainingFooter} onClick={() => setAdvanced(a => !a)}>
                            <span>Advanced</span>
                            <span>{showAdvanced ? '▲' : '▼'}</span>
                        </div>
                        {showAdvanced && (
                            <div className={styles.advancedSection}>
                                <div className={styles.advancedRow}>
                                    <span>Epochs</span>
                                    <input className={styles.advancedInput} type="number" value={epochs}
                                        min={1} max={200}
                                        onChange={e => setEpochs(Number(e.target.value))}
                                        onBlur={e => setEpochs(Math.max(1, Math.min(200, Number(e.target.value) || 25)))}/>
                                </div>
                                <div className={styles.advancedRow}>
                                    <span>Batch Size</span>
                                    <input className={styles.advancedInput} type="number" value={batchSize}
                                        min={1} max={128}
                                        onChange={e => setBatch(Number(e.target.value))}
                                        onBlur={e => setBatch(Math.max(1, Math.min(128, Number(e.target.value) || 16)))}/>
                                </div>
                                <div className={styles.advancedBtnsRow}>
                                    <button className={styles.trainReportBtn} onClick={() => {
                                        const totalSamples = labels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
                                        alert(`Model: Audio Classifier\nClasses: ${labels.join(', ')}\nTotal Samples: ${totalSamples}\nEpochs: ${epochs}  Batch Size: ${batchSize}\nStatus: ${isTrained ? 'Trained ✓' : 'Not trained'}`);
                                    }}>Train Report</button>
                                    <button className={styles.resetBtn}
                                        onClick={() => { setEpochs(25); setBatch(16); }}>Reset</button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Testing card */}
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
                            <AudioTestingPanel isTrained={isTrained}/>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

AudioTrainingPage.propTypes = {
    project:          PropTypes.object.isRequired,
    onBack:           PropTypes.func.isRequired,
    onUseInBlocks:    PropTypes.func.isRequired,
    onUpdateProject:  PropTypes.func.isRequired,
    onNewProject:     PropTypes.func,
    onNewMLProject:   PropTypes.func,
    onOpenMLProject:  PropTypes.func
};

export default AudioTrainingPage;
