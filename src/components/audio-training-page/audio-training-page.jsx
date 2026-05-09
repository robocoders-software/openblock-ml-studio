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
    loadProjectAudio,
    saveAudioBlobToIDB,
    getAudioBlobFromIDB,
    deleteAudioBlobFromIDB
} from '../../lib/idb.js';
import {WaveformRenderer} from './waveform-renderer.js';
import {computeRMS, blobToWavBlob} from '../../lib/audio-utils.js';

const CLASS_COLORS = ['#E05C3D', '#2EAA7E', '#9966FF', '#774DCB', '#F39C12', '#E91E63', '#1ABC9C', '#E67E22'];
const MAX_THUMBS   = 9;
const generateId   = () => Math.random().toString(36).slice(2, 10);

/* Render spectrogram data as a colored waveform thumbnail (bar-chart of RMS per frame).
   Normalises to the loudest frame so quiet recordings still show a visible waveform. */
const renderSpectrumThumb = (spectrogramData, frameSize, color) => {
    const W = 200, H = 80;
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, W, H);
    const data      = spectrogramData instanceof Float32Array ? spectrogramData : new Float32Array(spectrogramData);
    const numFrames = frameSize > 0 ? Math.floor(data.length / frameSize) : 0;
    if (numFrames === 0) return canvas.toDataURL('image/png');

    /* Compute RMS per frame */
    const rmsValues = new Float32Array(numFrames);
    for (let f = 0; f < numFrames; f++) {
        let sum = 0;
        const off = f * frameSize;
        for (let i = 0; i < frameSize; i++) sum += data[off + i] ** 2;
        rmsValues[f] = Math.sqrt(sum / frameSize);
    }

    /* Normalise so the loudest frame fills 90% of height */
    const maxRms = Math.max(...rmsValues, 1e-6);
    const midY   = H / 2;
    const barW   = W / numFrames;

    ctx.fillStyle = color;
    for (let f = 0; f < numFrames; f++) {
        const barH = Math.max(2, (rmsValues[f] / maxRms) * H * 0.9);
        ctx.fillRect(f * barW, midY - barH / 2, Math.max(1, barW - 0.5), barH);
    }
    return canvas.toDataURL('image/png');
};

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
    const noiseFillRef  = useRef(null);  // direct DOM ref — avoids re-renders for meter
    const [recTime,       setRecTime]   = useState(0);    // seconds elapsed
    const [playingId,     setPlayingId] = useState(null); // sample id being played
    const playingIdRef    = useRef(null);                 // mirrors playingId without closure capture

    const liveCanvasRef    = useRef(null);
    const analyserRef      = useRef(null);
    const vizStreamRef     = useRef(null);
    const animFrameRef     = useRef(null);
    const isHoldingRef     = useRef(false);
    const isRecordingRef   = useRef(false);
    const menuRef          = useRef(null);
    const rendererRef      = useRef(null);
    const recTimerRef      = useRef(null);
    const mediaRecorderRef = useRef(null);
    const playingAudioRef  = useRef(null);

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
            clearInterval(recTimerRef.current);
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
            vizStreamRef.current.getTracks().forEach(t => t.stop());
            vizStreamRef.current   = null;
            analyserRef.current    = null;
            rendererRef.current    = null;
            setShowMic(false);
            setRecording(false);
            if (noiseFillRef.current) { noiseFillRef.current.style.width = '0%'; }
            setRecTime(0);
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
            analyser.fftSize = 1024; // more sample points → denser, richer waveform
            source.connect(analyser);
            analyserRef.current = analyser;
            // Set showMic AFTER analyser is ready so the useEffect sees it immediately.
            setShowMic(true);
        } catch (e) {
            const {title, msg} = friendlyMicError(e);
            setMicError(`${title}|||${msg}`);
            setShowMic(true); // show error UI
            onReleaseMic();
        }
    };

    const stopMic = () => {
        isHoldingRef.current = false;
        clearInterval(recTimerRef.current);
        if (playingAudioRef.current) {
            playingAudioRef.current.pause();
            if (playingAudioRef.current._revokeUrl) URL.revokeObjectURL(playingAudioRef.current._revokeUrl);
            playingAudioRef.current = null;
        }
        if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        if (vizStreamRef.current) vizStreamRef.current.getTracks().forEach(t => t.stop());
        vizStreamRef.current  = null;
        analyserRef.current   = null;
        rendererRef.current   = null;
        playingIdRef.current  = null;
        setShowMic(false);
        setRecording(false);
        if (noiseFillRef.current) { noiseFillRef.current.style.width = '0%'; }
        setRecTime(0);
        setPlayingId(null);
        setMicError('');
        onReleaseMic();
    };

    /* Live waveform — DAW-style scrolling min/max ring-buffer (WaveformRenderer) */
    useEffect(() => {
        if (!showMic || !analyserRef.current) return;
        const canvas = liveCanvasRef.current;
        if (!canvas) return;

        const analyser = analyserRef.current;
        const renderer = new WaveformRenderer(canvas, {color, bgColor: '#111'});
        rendererRef.current = renderer;

        const buf  = new Float32Array(analyser.fftSize);
        const draw = () => {
            animFrameRef.current = requestAnimationFrame(draw);
            if (!analyserRef.current) return;
            analyserRef.current.getFloatTimeDomainData(buf);
            renderer.push(buf);
            renderer.draw();
            /* Update noise meter directly on the DOM element — no React re-renders */
            if (noiseFillRef.current) {
                const rms = computeRMS(buf);
                noiseFillRef.current.style.width = `${Math.min(100, rms * 400)}%`;
                noiseFillRef.current.style.background =
                    rms > 0.25 ? '#e53935' : rms > 0.08 ? '#F39C12' : '#4caf50';
            }
        };
        draw();

        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
            rendererRef.current = null;
        };
    }, [showMic, color]);

    /* Cleanup on unmount */
    useEffect(() => () => stopMic(), []); // eslint-disable-line react-hooks/exhaustive-deps

    /* Record one sample; also captures raw audio blob for playback / WAV export */
    const doRecord = useCallback(async () => {
        if (!isHoldingRef.current || isRecordingRef.current) return;
        isRecordingRef.current = true;
        setRecording(true);

        /* Recording timer */
        const start = Date.now();
        recTimerRef.current = setInterval(
            () => setRecTime(Math.floor((Date.now() - start) / 1000)),
            200
        );

        /* MediaRecorder — capture raw audio alongside the spectrogram */
        const chunks = [];
        let mr = null;
        if (vizStreamRef.current) {
            try {
                mr = new MediaRecorder(vizStreamRef.current);
                mr.ondataavailable = e => { if (e.data && e.data.size > 0) chunks.push(e.data); };
                mr.start();
                mediaRecorderRef.current = mr;
            } catch (_) { mr = null; }
        }

        try {
            const specData = await collectAudioExample(label);

            /* Stop MediaRecorder and collect blob */
            let mediaBlob = null;
            if (mr && mr.state !== 'inactive') {
                mediaBlob = await new Promise(resolve => {
                    mr.addEventListener('stop', () => {
                        resolve(chunks.length ? new Blob(chunks, {type: mr.mimeType}) : null);
                    }, {once: true});
                    mr.stop();
                });
            }
            mediaRecorderRef.current = null;

            const thumbUrl = renderSpectrumThumb(specData.data, specData.frameSize, color);
            const id = generateId();
            await saveAudioToIDB(projectId, id, Array.from(specData.data), specData.frameSize);
            if (thumbUrl) await saveAudioThumbToIDB(projectId, id, thumbUrl);
            if (mediaBlob) await saveAudioBlobToIDB(projectId, id, mediaBlob);
            onRecorded(label, {
                id,
                type: 'audio',
                spectrogramData: Array.from(specData.data),
                frameSize: specData.frameSize
            });
        } catch (err) {
            console.error('[AudioCard] recording error:', err);
        }

        clearInterval(recTimerRef.current);
        setRecTime(0);
        isRecordingRef.current = false;
        setRecording(false);
        if (isHoldingRef.current) doRecord();
    }, [label, projectId, color, onRecorded]); // eslint-disable-line react-hooks/exhaustive-deps

    /* Play (or stop) the raw audio blob for a recorded sample.
       Uses playingIdRef so the callback is stable (no playingId dep). */
    const playAudio = useCallback(async sampleId => {
        if (playingAudioRef.current) {
            playingAudioRef.current.pause();
            if (playingAudioRef.current._revokeUrl) {
                URL.revokeObjectURL(playingAudioRef.current._revokeUrl);
            }
            playingAudioRef.current = null;
            const wasPlaying = playingIdRef.current;
            playingIdRef.current = null;
            setPlayingId(null);
            if (wasPlaying === sampleId) return;
        }
        const blob = await getAudioBlobFromIDB(projectId, sampleId);
        if (!blob) return; // sample pre-dates blob capture — silently skip
        const url   = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio._revokeUrl    = url;
        playingAudioRef.current = audio;
        playingIdRef.current    = sampleId;
        setPlayingId(sampleId);
        audio.play().catch(() => {});
        audio.onended = () => {
            URL.revokeObjectURL(url);
            playingAudioRef.current = null;
            playingIdRef.current    = null;
            setPlayingId(null);
        };
    }, [projectId]);

    /* Export a single sample as WAV */
    const exportWAV = useCallback(async (sampleId) => {
        const blob = await getAudioBlobFromIDB(projectId, sampleId);
        if (!blob) return;
        const wavBlob = await blobToWavBlob(blob);
        if (!wavBlob) return;
        const url = URL.createObjectURL(wavBlob);
        const a   = document.createElement('a');
        a.href     = url;
        a.download = `sample_${sampleId}.wav`;
        a.click();
        URL.revokeObjectURL(url);
    }, [projectId]);

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
                                    <canvas ref={liveCanvasRef} className={styles.liveWaveform} width={200} height={130}/>
                                    <div className={styles.recInfoRow}>
                                        <div className={styles.noiseMeter}>
                                            <span className={styles.noiseMeterLabel}>Level</span>
                                            <div className={styles.noiseMeterTrack}>
                                                <div ref={noiseFillRef} className={styles.noiseMeterFill} style={{width: '0%', background: '#4caf50'}}/>
                                            </div>
                                        </div>
                                        {isRecording && (
                                            <span className={styles.recTimer}>
                                                {`● ${String(Math.floor(recTime / 60)).padStart(2, '0')}:${String(recTime % 60).padStart(2, '0')}`}
                                            </span>
                                        )}
                                    </div>
                                    <button
                                        className={`${styles.holdBtn}${isRecording ? ` ${styles.holdBtnRecording}` : ''}`}
                                        onMouseDown={startHold}
                                        onMouseUp={endHold}
                                        onMouseLeave={endHold}
                                        onTouchStart={startHold}
                                        onTouchEnd={endHold}
                                        disabled={!isEngineReady}
                                    >
                                        {isRecording ? 'Release to Save' : 'Hold to Record'}
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
                            {thumbData.map(item => (
                                <div key={item.id}
                                    className={`${styles.thumb}${playingId === item.id ? ` ${styles.thumbActive}` : ''}`}>
                                    <img src={item.src} alt="" className={styles.waveThumb}/>
                                    <button
                                        className={styles.thumbPlayBtn}
                                        onClick={e => { e.stopPropagation(); playAudio(item.id); }}
                                        title={playingId === item.id ? 'Stop' : 'Play'}>
                                        {playingId === item.id ? '■' : '▶'}
                                    </button>
                                    <button
                                        className={styles.thumbDeleteBtn}
                                        onClick={e => { e.stopPropagation(); onDeleteSample(label, item.id); }}
                                        title="Remove">&#10005;</button>
                                    <button
                                        className={styles.thumbExportBtn}
                                        onClick={e => { e.stopPropagation(); exportWAV(item.id); }}
                                        title="Export WAV">&#8659;</button>
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
const AudioTestingPanel = ({isTrained, labels, labelColorMap}) => {
    const [isListening, setListening] = useState(false);
    const [results,     setResults]   = useState([]);
    const [listenErr,   setListenErr] = useState('');

    const testCanvasRef    = useRef(null);
    const testAnalyserRef  = useRef(null);
    const testStreamRef    = useRef(null);
    const testAnimRef      = useRef(null);
    const testRendererRef  = useRef(null);
    const testNoiseFillRef = useRef(null);

    /* Canvas is only in the DOM when isListening=true, so init renderer inside
       a useEffect that fires after the canvas mounts. */
    useEffect(() => {
        if (!isListening || !testCanvasRef.current || !testAnalyserRef.current) return;

        const analyser = testAnalyserRef.current;
        const renderer = new WaveformRenderer(testCanvasRef.current, {
            color:   '#9966FF',
            bgColor: '#1a0a2e'
        });
        testRendererRef.current = renderer;

        const buf = new Float32Array(analyser.fftSize);
        const draw = () => {
            testAnimRef.current = requestAnimationFrame(draw);
            if (!testAnalyserRef.current) return;
            testAnalyserRef.current.getFloatTimeDomainData(buf);
            renderer.push(buf);
            renderer.draw();
            if (testNoiseFillRef.current) {
                const rms = computeRMS(buf);
                testNoiseFillRef.current.style.width = `${Math.min(100, rms * 400)}%`;
                testNoiseFillRef.current.style.background =
                    rms > 0.25 ? '#e53935' : rms > 0.08 ? '#F39C12' : '#4caf50';
            }
        };
        draw();

        return () => {
            if (testAnimRef.current) cancelAnimationFrame(testAnimRef.current);
            testRendererRef.current = null;
        };
    }, [isListening]);

    const startTestViz = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
            testStreamRef.current = stream;
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            const source   = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 1024;
            source.connect(analyser);
            testAnalyserRef.current = analyser;
            /* Renderer is started by useEffect([isListening]) once canvas mounts */
        } catch (_) { /* visualization is optional */ }
    };

    const stopTestViz = () => {
        if (testAnimRef.current) cancelAnimationFrame(testAnimRef.current);
        if (testStreamRef.current) testStreamRef.current.getTracks().forEach(t => t.stop());
        testStreamRef.current   = null;
        testAnalyserRef.current = null;
        testRendererRef.current = null;
        if (testNoiseFillRef.current) testNoiseFillRef.current.style.width = '0%';
    };

    const toggle = async () => {
        if (isListening) {
            setListening(false);
            setResults([]);
            stopTestViz();
            await stopListening();
        } else {
            setListenErr('');
            try {
                await startListening(matches => setResults(matches));
                await startTestViz();
                setListening(true);
            } catch (e) {
                setListenErr(e.message || 'Could not start listening');
            }
        }
    };

    useEffect(() => () => { stopTestViz(); stopListening(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

    if (!isTrained) {
        return <p className={styles.testingPlaceholder}>Train a model first, then test it here.</p>;
    }

    return (
        <div className={styles.testPanelContent}>
            {isListening ? (
                <>
                    <div className={styles.testMicHeader}>
                        <span className={styles.micLabel}>Microphone</span>
                        <button className={styles.camBackBtn} onClick={toggle} title="Stop">&#8592;</button>
                    </div>
                    <canvas ref={testCanvasRef} className={styles.testLiveWaveform} width={200} height={130}/>
                    <div className={styles.recInfoRow}>
                        <div className={styles.noiseMeter}>
                            <span className={styles.noiseMeterLabel}>Level</span>
                            <div className={styles.noiseMeterTrack}>
                                <div ref={testNoiseFillRef} className={styles.noiseMeterFill} style={{width: '0%', background: '#4caf50'}}/>
                            </div>
                        </div>
                    </div>
                    {results.length > 0 && (
                        <div className={styles.testOutputSection}>
                            <p className={styles.testOutputLabel}>Output</p>
                            {/* Render in fixed label order so rows never swap positions */}
                            {labels.map(lbl => {
                                const r     = results.find(x => x.label === lbl);
                                const prob  = r ? (r.prob || 0) : 0;
                                const color = (labelColorMap && labelColorMap[lbl]) || '#9966FF';
                                return (
                                    <div key={lbl} className={styles.testResultBar}>
                                        <div className={styles.testResultLabelRow}>
                                            <span>{lbl}</span>
                                            <span>{Math.round(prob)}%</span>
                                        </div>
                                        <div className={styles.testResultTrack}>
                                            <div className={styles.testResultFill}
                                                style={{width: `${prob}%`, background: color}}/>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </>
            ) : (
                <>
                    <p className={styles.testByLabel}>Test Audio By</p>
                    <div className={styles.testOptionRow}>
                        <button className={styles.testOptionBtn} onClick={toggle}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" width="26" height="26">
                                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                                <line x1="12" y1="19" x2="12" y2="23"/>
                                <line x1="8" y1="23" x2="16" y2="23"/>
                            </svg>
                            Microphone
                        </button>
                    </div>
                </>
            )}
            {listenErr && <p style={{color: '#e53935', fontSize: 11}}>{listenErr}</p>}
        </div>
    );
};
AudioTestingPanel.propTypes = {
    isTrained:     PropTypes.bool.isRequired,
    labels:        PropTypes.array.isRequired,
    labelColorMap: PropTypes.object.isRequired
};

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
        deleteAudioBlobFromIDB(project.id, sampleId).catch(() => {});
    }, [project.id]);

    const deleteAllSamples = useCallback(label => {
        setData(d => {
            const samples = d[label] || [];
            samples.forEach(s => deleteAudioBlobFromIDB(project.id, s.id).catch(() => {}));
            return {...d, [label]: []};
        });
        setTrained(false);
    }, [project.id]);

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
                            <AudioTestingPanel
                                isTrained={isTrained}
                                labels={labels}
                                labelColorMap={Object.fromEntries(
                                    labels.map((l, i) => [l, CLASS_COLORS[i % CLASS_COLORS.length]])
                                )}
                            />
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
