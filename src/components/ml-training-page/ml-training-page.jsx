import React, {useState, useRef, useEffect, useLayoutEffect, useCallback} from 'react';
import PropTypes from 'prop-types';
import styles from './ml-training-page.css';
import AudioTrainingPage from '../audio-training-page/audio-training-page.jsx';
import TextTrainingPage  from '../text-training-page/text-training-page.jsx';
import openblockLogo from '../openblock-logo.svg';
import MLLoader from '../ml-loader/ml-loader.jsx';
import Spinner from 'openblock-gui/src/components/spinner/spinner.jsx';

import {
    getMobileNet,
    trainImages,
    loadImageClassifier,
    predictImages,
    evaluateImageModel,
    setActiveModel
} from '../../lib/ml-engine.js';
import {
    saveImageToFS        as saveImageToIDB,
    getImageFromFS       as getImageFromIDB,
    loadProjectImagesFromFS as loadProjectImages
} from '../../lib/ml-fs.js';
import {saveImageProject, loadImageProject} from '../../lib/project-persistence.js';
import TrainReportModal from './TrainReportModal.jsx';

const CLASS_COLORS = ['#E05C3D', '#2EAA7E', '#9966FF', '#774DCB', '#F39C12', '#E91E63', '#1ABC9C', '#E67E22'];
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
            <path d={lineD} stroke="#9966FF" strokeWidth="2.5" fill="none" strokeLinejoin="round" strokeLinecap="round"/>
            <circle cx={sx(points[points.length - 1].x)} cy={sy(points[points.length - 1].y)} r="4" fill="#9966FF"/>
            <text x={pL + cW / 2} y={H - 2} textAnchor="middle" fontSize="9" fill="#888">Accuracy Vs Epochs</text>
        </svg>
    );
};
AccuracyChart.propTypes = {points: PropTypes.array};

/* ──────────────────────────────────────────────
   Class card with inline webcam
────────────────────────────────────────────── */
const ClassCard = React.forwardRef(({
    label, colorIdx, samples, onAddImages, onDeleteSample, onDeleteAllSamples,
    onRename, onDelete, canDelete, onToggleDisable, isDisabled, projectId, selectedDeviceId,
    menuOpen, onOpenMenu, onCloseMenu,
    isWebcamActive, onRequestWebcam, onReleaseWebcam
}, ref) => {
    const color = CLASS_COLORS[colorIdx % CLASS_COLORS.length];
    const [editing,       setEditing]   = useState(false);
    const [newName,       setNewName]   = useState(label);
    const [confirmAction, setConfirm]   = useState(null);
    const [thumbData,     setThumbData] = useState([]);
    const [showWebcam,    setShowWebcam] = useState(false);
    const [camErr,        setCamErr]    = useState('');
    const [showSettings,  setShowSettings] = useState(false);

    /* ── Capture settings ── */
    const [fps,             setFps]           = useState(15);
    const [autoRecord,      setAutoRecord]     = useState(false); // false = hold mode, true = timed
    const [captureDelay,    setCaptureDelay]   = useState(1);
    const [captureDuration, setCaptureDuration] = useState(4);
    /* Temp edit state while settings panel is open */
    const [editFps,      setEditFps]      = useState(15);
    const [editAuto,     setEditAuto]     = useState(false);
    const [editDelay,    setEditDelay]    = useState(1);
    const [editDuration, setEditDuration] = useState(4);
    /* Auto-record runtime */
    const [autoCountdown, setAutoCountdown] = useState(0); // >0 = delaying
    const [autoRunning,   setAutoRunning]   = useState(false);
    const autoTimerRef = useRef(null);

    const fileRef   = useRef(null);
    const videoRef  = useRef(null);
    const streamRef = useRef(null);
    const holdRef   = useRef(null);
    const menuRef   = useRef(null);

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

    /* Stop webcam when another class takes ownership */
    useEffect(() => {
        if (!isWebcamActive && streamRef.current) {
            if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
            setShowWebcam(false);
            setCamErr('');
        }
    }, [isWebcamActive]);

    /* Load thumbnails with id tracking for deletion */
    useEffect(() => {
        let cancelled = false;
        const toLoad = samples.filter(s => s.type === 'image');
        Promise.all(toLoad.map(async s => {
            const src = (s.data && s.data.startsWith('data:')) ? s.data : await getImageFromIDB(projectId, s.id);
            return src ? {src, id: s.id} : null;
        })).then(data => { if (!cancelled) setThumbData(data.filter(Boolean)); });
        return () => { cancelled = true; };
    }, [samples, projectId]);

    /* Cleanup on unmount */
    useEffect(() => () => {
        if (holdRef.current) clearInterval(holdRef.current);
        if (autoTimerRef.current) { clearInterval(autoTimerRef.current); clearTimeout(autoTimerRef.current); }
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    }, []);

    /* Attach stream to video element after it renders (also re-fires when settings panel closes) */
    useEffect(() => {
        if (!showSettings && showWebcam && videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
        }
    }, [showWebcam, showSettings]);

    const friendlyCamError = e => {
        const n = e.name || '';
        if (n === 'NotAllowedError' || n === 'PermissionDeniedError')
            return {title: 'Permission Denied', msg: 'Camera access was blocked. Allow camera permission in your browser or system settings.'};
        if (n === 'NotFoundError' || n === 'DevicesNotFoundError')
            return {title: 'No Camera Found', msg: 'No camera device was detected. Connect a webcam and try again.'};
        if (n === 'NotReadableError' || n === 'TrackStartError')
            return {title: 'Camera Busy', msg: 'Your camera might be open in another application. Close it and try again.'};
        if (n === 'OverconstrainedError')
            return {title: 'Camera Unavailable', msg: 'The selected camera does not support the required settings.'};
        return {title: 'Camera Error', msg: e.message || 'Could not start the camera. Please try again.'};
    };

    const startWebcam = async () => {
        onRequestWebcam();
        setCamErr('');
        setShowWebcam(true); // show panel immediately (video attaches after stream)
        const constraints = {video: selectedDeviceId ? {deviceId: {exact: selectedDeviceId}} : true};
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;
            if (videoRef.current) videoRef.current.srcObject = stream;
        } catch (e) {
            const {title, msg} = friendlyCamError(e);
            setCamErr(`${title}|||${msg}`);
            onReleaseWebcam();
        }
    };

    const stopWebcam = () => {
        if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
        setShowWebcam(false);
        setCamErr('');
        onReleaseWebcam();
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
        if (holdRef.current) return;
        captureOne();
        holdRef.current = setInterval(captureOne, Math.round(1000 / fps));
    };

    const holdEnd = e => {
        e.preventDefault();
        if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
    };

    const startAutoRecord = () => {
        if (autoRunning || autoCountdown > 0) return;
        if (captureDelay > 0) {
            setAutoCountdown(captureDelay);
            let remaining = captureDelay;
            autoTimerRef.current = setInterval(() => {
                remaining -= 1;
                setAutoCountdown(remaining);
                if (remaining <= 0) {
                    clearInterval(autoTimerRef.current);
                    autoTimerRef.current = null;
                    runAutoCapture();
                }
            }, 1000);
        } else {
            runAutoCapture();
        }
    };

    const runAutoCapture = () => {
        setAutoRunning(true);
        setAutoCountdown(0);
        const interval = Math.round(1000 / fps);
        holdRef.current = setInterval(captureOne, interval);
        autoTimerRef.current = setTimeout(() => {
            if (holdRef.current) { clearInterval(holdRef.current); holdRef.current = null; }
            setAutoRunning(false);
            autoTimerRef.current = null;
        }, captureDuration * 1000);
    };

    const openSettings = () => {
        setEditFps(fps);
        setEditAuto(autoRecord);
        setEditDelay(captureDelay);
        setEditDuration(captureDuration);
        setShowSettings(true);
    };

    const saveSettings = () => {
        const f = Math.max(1, Math.min(30, Number(editFps) || 15));
        const d = Math.max(0, Number(editDelay) || 0);
        const dur = Math.max(1, Number(editDuration) || 4);
        setFps(f);
        setAutoRecord(editAuto);
        setCaptureDelay(d);
        setCaptureDuration(dur);
        setShowSettings(false);
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
        <div className={`${styles.classCard}${isDisabled ? ` ${styles.classCardDisabled}` : ''}`} ref={ref}>
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
                    <button className={styles.classCardEditBtn} onClick={() => { setEditing(true); setNewName(label); }} title="Rename">
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

            <div className={`${styles.classCardBody}${showWebcam ? ` ${styles.classCardBodyCam}` : ''}`}>
                <div className={styles.classCardLeft}>
                    {showSettings ? (
                        <div className={styles.settingsPanel}>
                            <div className={styles.settingsPanelHeader}>
                                <span>Settings</span>
                                <button className={styles.camBackBtn} onClick={() => setShowSettings(false)}>&#8592;</button>
                            </div>
                            <div className={styles.settingsForm}>
                                <div className={styles.settingsRow}>
                                    <label className={styles.settingsLabel}>FPS:</label>
                                    <input
                                        type="number" min="1" max="30"
                                        className={styles.settingsInput}
                                        value={editFps}
                                        onChange={e => setEditFps(e.target.value)}
                                    />
                                </div>
                                <div className={styles.settingsRow}>
                                    <label className={styles.settingsLabel}>Hold to Record:</label>
                                    <button
                                        className={`${styles.toggleSwitch}${editAuto ? ` ${styles.toggleOn}` : ''}`}
                                        onClick={() => setEditAuto(v => !v)}
                                    >
                                        <span className={styles.toggleThumb}/>
                                    </button>
                                </div>
                                {editAuto && (<>
                                    <div className={styles.settingsRow}>
                                        <label className={styles.settingsLabel}>Delay:</label>
                                        <input
                                            type="number" min="0" max="10"
                                            className={styles.settingsInput}
                                            value={editDelay}
                                            onChange={e => setEditDelay(e.target.value)}
                                        />
                                        <span className={styles.settingsUnit}>seconds</span>
                                    </div>
                                    <div className={styles.settingsRow}>
                                        <label className={styles.settingsLabel}>Duration:</label>
                                        <input
                                            type="number" min="1" max="60"
                                            className={styles.settingsInput}
                                            value={editDuration}
                                            onChange={e => setEditDuration(e.target.value)}
                                        />
                                        <span className={styles.settingsUnit}>seconds</span>
                                    </div>
                                </>)}
                                <button className={styles.saveSettingsBtn} onClick={saveSettings}>Save Settings</button>
                            </div>
                        </div>
                    ) : showWebcam ? (
                        <div className={styles.inlineWebcam}>
                            <div className={styles.inlineWebcamHeader}>
                                <span className={styles.webcamSectionLabel}>Webcam</span>
                                <button className={styles.camBackBtn} onClick={stopWebcam} title="Back">&#8592;</button>
                            </div>
                            {camErr ? (() => {
                                const [title, msg] = camErr.split('|||');
                                return (
                                    <div className={styles.resourceError}>
                                        <div className={styles.resourceErrorIcon}>!</div>
                                        <p className={styles.resourceErrorTitle}>{title}</p>
                                        <p className={styles.resourceErrorMsg}>{msg}</p>
                                        <button className={styles.resourceRetryBtn} onClick={startWebcam}>Try Again</button>
                                    </div>
                                );
                            })() : (
                                <video ref={videoRef} autoPlay playsInline muted className={styles.inlineVideo}/>
                            )}
                            {!camErr && (
                                <div className={styles.holdRow}>
                                    {autoRecord ? (
                                        <button
                                            className={`${styles.holdBtn}${autoRunning ? ` ${styles.holdBtnRecording}` : ''}`}
                                            onClick={startAutoRecord}
                                            disabled={autoRunning || autoCountdown > 0}
                                        >
                                            {autoCountdown > 0
                                                ? `Starting in ${autoCountdown}s…`
                                                : autoRunning
                                                    ? '● Recording…'
                                                    : `Record ${captureDuration} Seconds`}
                                        </button>
                                    ) : (
                                        <button
                                            className={styles.holdBtn}
                                            onMouseDown={holdStart}
                                            onMouseUp={holdEnd}
                                            onMouseLeave={holdEnd}
                                            onTouchStart={holdStart}
                                            onTouchEnd={holdEnd}
                                        >
                                            Hold to Record
                                        </button>
                                    )}
                                    <button className={styles.gearBtn} title="Settings" onClick={openSettings}>&#9881;</button>
                                </div>
                            )}
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
                        </>
                    )}
                </div>

                <div className={styles.classCardRight}>
                    <p className={styles.sampleCountLabel}>{samples.length} Image Sample{samples.length !== 1 ? 's' : ''}</p>
                    {thumbData.length > 0 && (
                        <div className={styles.thumbnailGrid}>
                            {thumbData.map((item) => (
                                <div key={item.id} className={styles.thumb}>
                                    <img src={item.src} alt=""/>
                                    <button
                                        className={styles.thumbDeleteBtn}
                                        onClick={e => { e.stopPropagation(); onDeleteSample(label, item.id); }}
                                        title="Remove sample"
                                    >&#10005;</button>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
});
ClassCard.displayName = 'ClassCard';
ClassCard.propTypes = {
    label:              PropTypes.string.isRequired,
    colorIdx:           PropTypes.number.isRequired,
    samples:            PropTypes.array.isRequired,
    onAddImages:        PropTypes.func.isRequired,
    onDeleteSample:     PropTypes.func.isRequired,
    onDeleteAllSamples: PropTypes.func.isRequired,
    onRename:           PropTypes.func.isRequired,
    onDelete:           PropTypes.func.isRequired,
    canDelete:          PropTypes.bool.isRequired,
    onToggleDisable:    PropTypes.func.isRequired,
    isDisabled:         PropTypes.bool.isRequired,
    projectId:          PropTypes.string.isRequired,
    selectedDeviceId:   PropTypes.string,
    menuOpen:           PropTypes.bool.isRequired,
    onOpenMenu:         PropTypes.func.isRequired,
    onCloseMenu:        PropTypes.func.isRequired,
    isWebcamActive:     PropTypes.bool.isRequired,
    onRequestWebcam:    PropTypes.func.isRequired,
    onReleaseWebcam:    PropTypes.func.isRequired
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
    const [camError,  setCamError] = useState('');

    /* ── Stop webcam cleanly ── */
    const stopCam = useCallback(() => {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop());
            streamRef.current = null;
        }
        if (videoRef.current) videoRef.current.srcObject = null;
        setMode('idle');
        setResults([]);
        setCamError('');
    }, []);

    /* ── KEY FIX: attach stream AFTER the <video> element renders ──
       The video element only appears in the DOM when mode === 'webcam'.
       startWebcam calls setMode('webcam') which triggers a re-render,
       then this effect attaches the stream to the newly mounted element.
    ── */
    useEffect(() => {
        if (mode === 'webcam' && videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
        }
    }, [mode]);

    /* ── Start continuous webcam prediction ── */
    const startWebcam = useCallback(async () => {
        setCamError('');
        const constraints = {video: selectedDeviceId ? {deviceId: {exact: selectedDeviceId}} : true};
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            streamRef.current = stream;

            // Switch to webcam mode — useEffect above will attach srcObject
            // after React renders the <video> element.
            setMode('webcam');

            // Start prediction interval — guard against unready video frames
            clearInterval(intervalRef.current);
            intervalRef.current = setInterval(async () => {
                const video = videoRef.current;
                // Skip if video not ready, no dimensions, or model not loaded
                if (!video || !video.videoWidth || !video.videoHeight) return;
                if (!classifierRef.current || !mobileNetRef.current) return;
                try {
                    const probs = await predictImages(
                        video,
                        classifierRef.current,
                        mobileNetRef.current,
                        labels
                    );
                    setResults(probs);
                } catch (err) {
                    console.warn('[TestPanel] prediction error:', err.message);
                }
            }, 400);
        } catch (e) {
            setCamError(e.message || 'Could not access camera');
            console.error('[TestPanel] getUserMedia error:', e);
        }
    }, [selectedDeviceId, classifierRef, mobileNetRef, labels]);

    /* ── Upload single image and classify ── */
    const handleUpload = useCallback(async e => {
        const file = e.target.files[0];
        if (!file) return;
        if (!classifierRef.current || !mobileNetRef.current) {
            alert('Train a model first before testing.');
            return;
        }
        const dataUrl = await new Promise(res => {
            const r = new FileReader();
            r.onload = () => res(r.result);
            r.readAsDataURL(file);
        });
        setTestImg(dataUrl);
        setResults([]);
        const img = new Image();
        img.src = dataUrl;
        await new Promise(res => { img.onload = res; img.onerror = res; });
        try {
            const probs = await predictImages(
                img,
                classifierRef.current,
                mobileNetRef.current,
                labels
            );
            setResults(probs);
            setMode('upload');
        } catch (err) {
            console.error('[TestPanel] upload classify error:', err);
        }
        e.target.value = '';
    }, [classifierRef, mobileNetRef, labels]);

    /* ── Cleanup on unmount ── */
    useEffect(() => () => {
        clearInterval(intervalRef.current);
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    }, []);

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
const MLTrainingPage = ({project, onBack, onUseInBlocks, onUpdateProject, onNewProject, onNewMLProject, onOpenMLProject}) => {
    if (project.type === 'sounds') {
        return (
            <AudioTrainingPage
                project={project}
                onBack={onBack}
                onUseInBlocks={onUseInBlocks}
                onUpdateProject={onUpdateProject}
                onNewProject={onNewProject}
                onNewMLProject={onNewMLProject}
                onOpenMLProject={onOpenMLProject}
            />
        );
    }
    if (project.type === 'text') {
        return (
            <TextTrainingPage
                project={project}
                onBack={onBack}
                onUseInBlocks={onUseInBlocks}
                onUpdateProject={onUpdateProject}
                onNewProject={onNewProject}
                onNewMLProject={onNewMLProject}
                onOpenMLProject={onOpenMLProject}
            />
        );
    }
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
    const [epochMetrics,   setEpochMetrics] = useState([]);
    const [showReport,     setShowReport]  = useState(false);
    const [reportData,     setReportData]  = useState(null);
    /* savedReport persists the last completed report so the button stays enabled
       even after adding new images — cleared only when a new training run starts */
    const [savedReport,    setSavedReport] = useState(null);

    const [cameras,          setCameras]         = useState([]);
    const [selectedCam,      setSelectedCam]      = useState('');
    const [activeWebcamLabel, setActiveWebcamLabel] = useState(null);
    const [openMenuLabel,    setOpenMenuLabel]    = useState(null);
    const [fileMenuOpen,     setFileMenuOpen]     = useState(false);
    const [saveStatus,       setSaveStatus]       = useState('idle'); // 'idle' | 'saving' | 'saved' | 'error'
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

    const classifierRef = useRef(null);
    const mobileNetRef  = useRef(null);

    /* ── Live training API ref — always current, exposed via bridge ── */
    const trainingAPIRef = useRef({});

    const canvasRef     = useRef(null);
    const classCardRefs = useRef([]);
    const trainCardRef  = useRef(null);
    const testCardRef   = useRef(null);
    const [svgPaths,    setSvgPaths]   = useState([]);

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

    /* ── Load project data (try opened .ob file first, fall back to FS) ── */
    const loadFromDisk = useCallback(async (signal = {cancelled: false}) => {
        let finalLabels = project.labels || ['Class 1', 'Class 2'];
        let trained = false;
        try {
            setLoadingData(true);
            classifierRef.current = null;
            mobileNetRef.current  = null;
            const fromOb = await loadImageProject(project.id);
            if (fromOb && !signal.cancelled) {
                if (fromOb.name) onUpdateProject({...project, name: fromOb.name});
                if (fromOb.labels && fromOb.labels.length >= 2) {
                    setLabels(fromOb.labels);
                    finalLabels = fromOb.labels;
                }
                const enriched = await loadProjectImages(project.id, fromOb.trainingData || {});
                if (!signal.cancelled) setData(enriched);
                if (fromOb.classifier && fromOb.net) {
                    classifierRef.current = fromOb.classifier;
                    mobileNetRef.current  = fromOb.net;
                    trained = true;
                    setTrained(true);
                } else {
                    setTrained(false);
                }
            } else {
                const enriched = await loadProjectImages(project.id, project.trainingData || {});
                if (!signal.cancelled) setData(enriched);
                if (project.trained) {
                    const restored = await loadImageClassifier(project.id, project.labels || ['Class 1', 'Class 2']);
                    if (restored && !signal.cancelled) {
                        classifierRef.current = restored;
                        getMobileNet(s => setStatus(s)).then(net => {
                            if (!signal.cancelled) mobileNetRef.current = net;
                        }).catch(() => {});
                        trained = true;
                        setTrained(true);
                    }
                } else {
                    setTrained(false);
                }
            }
            /* Always register model type so the blocks editor knows this is an IMAGE project. */
            if (!signal.cancelled) {
                setActiveModel({
                    projectId:     project.id,
                    projectName:   project.name,
                    type:          'images',
                    labels:        finalLabels,
                    classifier:    classifierRef.current,
                    mobileNet:     mobileNetRef.current,
                    trainingStatus: trained ? 'ready' : 'idle',
                    _trainingAPI:  trainingAPIRef
                });
            }
        } finally {
            if (!signal.cancelled) setLoadingData(false);
        }
    }, [project.id]); // eslint-disable-line react-hooks/exhaustive-deps

    useEffect(() => {
        const signal = {cancelled: false};
        loadFromDisk(signal);
        return () => { signal.cancelled = true; };
    }, []); // eslint-disable-line react-hooks/exhaustive-deps

    /* Re-load when a new .ob file is opened in the blocks editor */
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

    /* ── Persist metadata — auto-save so training data survives navigation ── */
    useEffect(() => {
        if (loadingData) return;
        const metaOnly = {};
        for (const [lbl, exs] of Object.entries(trainingData)) {
            metaOnly[lbl] = (exs || []).map(ex => ex.type === 'image' ? {id: ex.id, type: 'image'} : ex);
        }
        onUpdateProject({...project, labels, trainingData: metaOnly, trained: isTrained, updatedAt: Date.now()});
        saveImageProject(project, labels, disabledLabels, trainingData, classifierRef.current, {showDialog: false})
            .catch(() => {});
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
        const activeLabels = labels.filter(l => !disabledLabels.includes(l));
        const total = activeLabels.reduce((s, l) => s + (trainingData[l] || []).length, 0);
        if (activeLabels.length < 2 || !activeLabels.every(l => (trainingData[l] || []).length >= 10)) {
            setStatus('Need at least 2 enabled classes, each with at least 10 images.');
            return;
        }
        setTraining(true);
        setTrainPct(0);
        setAccPoints([{x: 0, y: 0}]);
        setEpochMetrics([]);
        setReportData(null);
        setSavedReport(null);
        setStatus('Loading MobileNetV1…');

        try {
            const net = await getMobileNet(s => setStatus(s));
            mobileNetRef.current = net;

            const collectedMetrics = [];
            const classifier = await trainImages(
                activeLabels, trainingData, project.id,
                s => setStatus(s),
                pct => setTrainPct(pct),
                {
                    epochs,
                    batchSize,
                    learningRate,
                    onEpochEnd: (epochIdx, metrics) => {
                        const {trainAcc, valAcc} = metrics;
                        const displayAcc = valAcc || trainAcc;
                        const pt = {epoch: epochIdx + 1, ...metrics};
                        collectedMetrics.push(pt);
                        setAccPoints(pts => [...pts, {x: epochIdx + 1, y: displayAcc}]);
                        setEpochMetrics(pts => [...pts, pt]);
                    }
                }
            );

            classifierRef.current = classifier;
            setTrained(true);
            setTrainPct(100);
            /* Cache ML data so the next normal blocks save includes the trained model */
            saveImageProject(project, labels, disabledLabels, trainingData, classifier, {showDialog: false})
                .catch(() => {});
            setStatus('Evaluating model…');

            // Run evaluation for report (non-blocking)
            evaluateImageModel(activeLabels, trainingData, project.id, classifier, net,
                pct => setStatus(`Evaluating… ${pct}%`)
            ).then(result => {
                setReportData(result);
                setSavedReport({epochMetrics: collectedMetrics, reportData: result, labels: activeLabels});
                setStatus('Training Complete');
            }).catch(() => setStatus('Training Complete'));

        } catch (err) {
            setStatus(`Error: ${err.message}`);
            console.error('[ML Train]', err);
        } finally {
            setTraining(false);
        }
    };

    /* ── Keep trainingAPI ref live (updated every render — no stale closures) ── */
    trainingAPIRef.current = {
        addTrainingImage: addImages,
        startTraining:    trainModel,
        getStatus:        () => isTraining ? 'training' : isTrained ? 'ready' : 'idle',
        clearTraining:    () => { setData({}); setTrained(false); },
        labels,
        trainingData
    };

    /* ── Save project with user feedback ── */
    const handleSave = useCallback(async () => {
        setSaveStatus('saving');
        try {
            const result = await saveImageProject(
                project, labels, disabledLabels, trainingData, classifierRef.current, {showDialog: false}
            );
            if (result && result.success === false) throw new Error(result.error || 'Save failed');
            setSaveStatus('saved');
        } catch (e) {
            console.error('[MLPage] save failed:', e);
            setSaveStatus('error');
        } finally {
            setTimeout(() => setSaveStatus('idle'), 2000);
        }
    }, [project, labels, disabledLabels, trainingData]);

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

    /* ── Export: deploy model to blocks and navigate ── */
    const handleExportModel = () => {
        deployModel();
        try { window.require('electron').ipcRenderer.send('ml-set-pending-project', project.id); } catch (_) {}
        onUseInBlocks();
    };

    /* ── SVG curves — canvas no longer scrolls, only classesColumn does ── */
    const recalcCurves = useCallback(() => {
        if (!canvasRef.current || !trainCardRef.current || !testCardRef.current) return;
        const canvasEl   = canvasRef.current;
        const canvasRect = canvasEl.getBoundingClientRect();
        const trainRect  = trainCardRef.current.getBoundingClientRect();
        const testRect   = testCardRef.current.getBoundingClientRect();
        const paths = [];

        classCardRefs.current.forEach((r, i) => {
            if (!r) return;
            const rect = r.getBoundingClientRect();
            const x1 = rect.right - canvasRect.left;
            const y1 = rect.top + rect.height / 2 - canvasRect.top;
            const x2 = trainRect.left - canvasRect.left;
            const y2 = trainRect.top + trainRect.height / 2 - canvasRect.top;
            const cx = Math.max(40, (x2 - x1) / 2);
            paths.push({
                color: CLASS_COLORS[i % CLASS_COLORS.length],
                d: `M ${x1} ${y1} C ${x1 + cx} ${y1} ${x2 - cx} ${y2} ${x2} ${y2}`
            });
        });

        const tx1 = trainRect.right - canvasRect.left;
        const ty1 = trainRect.top + trainRect.height / 2 - canvasRect.top;
        const tx2 = testRect.left - canvasRect.left;
        const ty2 = testRect.top + testRect.height / 2 - canvasRect.top;
        const tcx = Math.max(40, (tx2 - tx1) / 2);
        paths.push({
            color: '#9966FF',
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

    const activeLabels = labels.filter(l => !disabledLabels.includes(l));
    const canTrain = !isTraining &&
        activeLabels.length >= 2 &&
        activeLabels.every(l => (trainingData[l] || []).length >= 10);

    if (loadingData) return <MLLoader message="Loading project data…" />;

    return (
        <>
        <div className={styles.page}>
            {/* Header */}
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
                <span className={styles.subHeaderType}>Image Classifier</span>
                <span className={styles.infoIcon} title="Train a model to recognise images from your webcam or uploaded photos — then use it in your Blocks project.">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="#9966FF" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="11"/>
                        <circle cx="12" cy="8" r="1.3" fill="white"/>
                        <rect x="10.7" y="11" width="2.6" height="6" rx="1.3" fill="white"/>
                    </svg>
                </span>
                <div className={styles.divider}/>
                <span className={styles.projectNamePill}>{project.name}</span>
                <div className={styles.divider}/>
                <button className={styles.uploadFolderBtn} onClick={() => folderInputRef.current && folderInputRef.current.click()}>
                    Upload Classes from Folder
                </button>
                <input type="file" webkitdirectory="" multiple accept="image/*" style={{display: 'none'}} ref={folderInputRef} onChange={handleFolderUpload}/>
                <button
                    className={styles.saveBtn}
                    title={saveStatus === 'saving' ? 'Saving…' : saveStatus === 'saved' ? 'Saved!' : saveStatus === 'error' ? 'Save failed' : 'Save ML Project'}
                    disabled={saveStatus === 'saving'}
                    onClick={handleSave}
                >
                    {saveStatus === 'saving' ? '⏳' : saveStatus === 'saved' ? '✓' : saveStatus === 'error' ? '✗' : '💾'}
                </button>
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

                {/* Classes column — scrolls independently */}
                <div className={styles.classesColumn} onScroll={recalcCurves}>
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
                            isDisabled={disabledLabels.includes(lbl)}
                            menuOpen={openMenuLabel === lbl}
                            onOpenMenu={() => setOpenMenuLabel(lbl)}
                            onCloseMenu={() => setOpenMenuLabel(null)}
                            isWebcamActive={activeWebcamLabel === lbl}
                            onRequestWebcam={() => setActiveWebcamLabel(lbl)}
                            onReleaseWebcam={() => setActiveWebcamLabel(null)}
                            onAddImages={addImages}
                            onDeleteSample={deleteSample}
                            onDeleteAllSamples={deleteAllSamples}
                            onRename={renameClass}
                            onDelete={deleteClass}
                            onToggleDisable={toggleDisableClass}
                        />
                    ))}
                    <button className={styles.addClassCard} onClick={addClass}>+ Add Class</button>
                </div>

                {/* Training column — stays fixed */}
                <div className={styles.trainingColumn}>
                    <div className={styles.trainingCard} ref={trainCardRef}>
                        <div className={styles.trainingHeader}>
                            <span>Training</span>
                        </div>
                        {disabledLabels.length > 0 && (
                            <div className={styles.activeClassInfo}>
                                {activeLabels.length} / {labels.length} classes active
                            </div>
                        )}
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
                                    {isTraining ? <Spinner small level="info" /> : '⚡ Train Model'}
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
                                        onChange={e => setEpochs(Number(e.target.value))}
                                        onBlur={e => setEpochs(Math.max(1, Math.min(200, Number(e.target.value) || 20)))}/>
                                </div>
                                <div className={styles.advancedRow}>
                                    <span>Batch Size</span>
                                    <input className={styles.advancedInput} type="number" value={batchSize} min={1} max={512}
                                        onChange={e => setBatchSize(Number(e.target.value))}
                                        onBlur={e => setBatchSize(Math.max(1, Math.min(512, Number(e.target.value) || 16)))}/>
                                </div>
                                <div className={styles.advancedRow}>
                                    <span>Learning Rate</span>
                                    <input className={styles.advancedInput} type="number" value={learningRate} step="0.0001" min={0.0001} max={1}
                                        onChange={e => setLR(Number(e.target.value))}
                                        onBlur={e => setLR(Math.max(0.0001, Math.min(1, Number(e.target.value) || 0.001)))}/>
                                </div>
                                <div className={styles.advancedBtnsRow}>
                                    <button className={styles.trainReportBtn}
                                        disabled={!savedReport}
                                        onClick={() => setShowReport(true)}>
                                        Train Report
                                    </button>
                                    <button className={styles.resetBtn} onClick={() => { setEpochs(20); setBatchSize(16); setLR(0.001); }}>
                                        Reset
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                </div>

                {/* Testing column — stays fixed */}
                <div className={styles.testingColumn}>
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

        {showReport && savedReport && (
            <TrainReportModal
                onClose={() => setShowReport(false)}
                epochMetrics={savedReport.epochMetrics}
                reportData={savedReport.reportData}
                labels={savedReport.labels}
            />
        )}
        </>
    );
};

MLTrainingPage.propTypes = {
    project:          PropTypes.object.isRequired,
    onBack:           PropTypes.func.isRequired,
    onUseInBlocks:    PropTypes.func.isRequired,
    onUpdateProject:  PropTypes.func.isRequired,
    onNewProject:     PropTypes.func,
    onNewMLProject:   PropTypes.func,
    onOpenMLProject:  PropTypes.func
};

export default MLTrainingPage;
