import React, { useState, useMemo } from 'react';
import styles from './TrainReportModal.css';

/* ── Tiny line chart ── */
const LineChart = ({ data, title, yLabel, lines }) => {
    const W = 560, H = 320;
    const pL = 52, pR = 20, pT = 18, pB = 46;
    const cW = W - pL - pR, cH = H - pT - pB;

    const allVals = lines.flatMap(l => data.map(d => d[l.key] || 0));
    const maxY = Math.max(...allVals, 0.01);
    const minY = 0;
    const epochMax = data.length > 0 ? data[data.length - 1].epoch : 1;

    const sx = x => pL + (x / epochMax) * cW;
    const sy = y => pT + cH - ((y - minY) / (maxY - minY || 1)) * cH;

    const gridY = [0, 0.2, 0.4, 0.6, 0.8, 1.0].filter(v => v <= maxY + 0.05);

    const makePath = key => {
        const pts = data.filter(d => d[key] !== undefined);
        if (pts.length < 1) return '';
        return pts.map((d, i) =>
            `${i === 0 ? 'M' : 'L'}${sx(d.epoch).toFixed(1)},${sy(d[key]).toFixed(1)}`
        ).join(' ');
    };

    return (
        <div className={styles.chartBox}>
            <div className={styles.chartTitle}>{title}</div>
            <svg width="100%" viewBox={`0 0 ${W} ${H}`} className={styles.chartSvg}>
                {/* Grid */}
                {gridY.map(v => (
                    <g key={v}>
                        <line x1={pL} y1={sy(v)} x2={W - pR} y2={sy(v)} stroke="#e8e0f5" strokeWidth="1"/>
                        <text x={pL - 6} y={sy(v) + 4} textAnchor="end" fontSize="10" fill="#999">{v.toFixed(2)}</text>
                    </g>
                ))}
                {/* Axes */}
                <line x1={pL} y1={pT} x2={pL} y2={H - pB} stroke="#bbb" strokeWidth="1"/>
                <line x1={pL} y1={H - pB} x2={W - pR} y2={H - pB} stroke="#bbb" strokeWidth="1"/>
                {/* Lines */}
                {lines.map(l => (
                    <path key={l.key} d={makePath(l.key)} stroke={l.color} strokeWidth="2.5"
                        fill="none" strokeLinejoin="round" strokeLinecap="round"/>
                ))}
                {/* Last point dots */}
                {lines.map(l => {
                    const last = [...data].reverse().find(d => d[l.key] !== undefined);
                    if (!last) return null;
                    return <circle key={l.key} cx={sx(last.epoch)} cy={sy(last[l.key])} r="4" fill={l.color}/>;
                })}
                {/* Labels */}
                <text x={pL + cW / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#666">Epochs</text>
                <text x={14} y={pT + cH / 2} textAnchor="middle" fontSize="11" fill="#666"
                    transform={`rotate(-90, 14, ${pT + cH / 2})`}>{yLabel}</text>
                {/* Arrow tips */}
                <polygon points={`${W-pR},${H-pB-4} ${W-pR+5},${H-pB} ${W-pR},${H-pB+4}`} fill="#bbb"/>
                <polygon points={`${pL-4},${pT} ${pL},${pT-5} ${pL+4},${pT}`} fill="#bbb"/>
            </svg>
            {/* Legend */}
            <div className={styles.chartLegend}>
                {lines.map(l => (
                    <span key={l.key} className={styles.legendItem}>
                        <span className={styles.legendLine} style={{background: l.color}}/>
                        {l.label}
                    </span>
                ))}
            </div>
        </div>
    );
};

/* ── Confusion matrix ── */
const ConfusionMatrix = ({ labels, confusion }) => {
    const maxVal = Math.max(...labels.flatMap(a => labels.map(p => (confusion[a] && confusion[a][p]) || 0)), 1);

    const cellColor = (actual, predicted) => {
        const v = (confusion[actual] && confusion[actual][predicted]) || 0;
        const intensity = v / maxVal;
        return `rgba(220, 50, 50, ${intensity.toFixed(2)})`;
    };

    return (
        <div className={styles.confusionWrap}>
            <div className={styles.sectionHeader}>Confusion Matrix</div>
            <div className={styles.confusionBody}>
                <div className={styles.confusionLeft}>
                    <table className={styles.confusionTable}>
                        <thead>
                            <tr>
                                <th className={styles.confusionCorner}/>
                                {labels.map(l => <th key={l} className={styles.confusionColHead}>{l}</th>)}
                            </tr>
                        </thead>
                        <tbody>
                            {labels.map(actual => (
                                <tr key={actual}>
                                    <td className={styles.confusionRowHead}>{actual}</td>
                                    {labels.map(predicted => {
                                        const v = (confusion[actual] && confusion[actual][predicted]) || 0;
                                        return (
                                            <td key={predicted} className={styles.confusionCell}
                                                style={{background: cellColor(actual, predicted)}}>
                                                <span style={{color: v / maxVal > 0.4 ? 'white' : '#333'}}>{v}</span>
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    <div className={styles.confusionXLabel}>Prediction</div>
                </div>
                <div className={styles.colorScale}>
                    <span className={styles.scaleMax}>{maxVal}</span>
                    <div className={styles.scaleBar}/>
                    <span className={styles.scaleMin}>0</span>
                </div>
            </div>
        </div>
    );
};

/* ── Classwise analysis ── */
const ClasswiseAnalysis = ({ labels, classMetrics, samples }) => {
    const [selectedClass, setSelectedClass] = useState(labels[0]);
    const classSamples = (samples && samples[selectedClass]) || [];
    const truePos  = classSamples.filter(s => s.correct);
    const falseNeg = classSamples.filter(s => !s.correct);

    const rowColor = idx => {
        const hues = ['#e05c3a', '#3ab878', '#3a7be0', '#9b3ae0', '#e0b83a', '#3ae0d4'];
        return hues[idx % hues.length];
    };

    return (
        <div className={styles.classwiseWrap}>
            <div className={styles.sectionHeader}>Classwise Analysis</div>
            <div className={styles.classwiseBody}>
                {/* Left: image gallery */}
                <div className={styles.classwiseLeft}>
                    <div className={styles.selectedClassTitle}>Selected Class: {selectedClass}</div>

                    <div className={styles.analysisSection}>
                        <div className={styles.analysisSectionHeader} style={{background: '#e8d6f5'}}>
                            <span style={{color: '#6633cc', fontWeight: 700}}>True Positive</span>
                            <span className={styles.analysisSectionSub}>
                                Your model correctly predicted {selectedClass} on these images
                            </span>
                        </div>
                        <div className={styles.imageGallery}>
                            {truePos.length === 0
                                ? <p className={styles.noImages}>No correctly predicted images</p>
                                : truePos.map((s, i) => (
                                    <div key={i} className={styles.galleryItem}>
                                        <img src={s.src} alt="" className={styles.galleryImg}/>
                                        <span className={styles.galleryLabel}>Prediction: {s.predicted}</span>
                                    </div>
                                ))}
                        </div>
                    </div>

                    <div className={styles.analysisSection}>
                        <div className={styles.analysisSectionHeader} style={{background: '#e8d6f5'}}>
                            <span style={{color: '#6633cc', fontWeight: 700}}>False Negatives</span>
                            <span className={styles.analysisSectionSub}>
                                Your model failed to predict an accurate class for {selectedClass} on these images
                            </span>
                        </div>
                        <div className={styles.imageGallery}>
                            {falseNeg.length === 0
                                ? <p className={styles.noImages}>No misclassified images ✓</p>
                                : falseNeg.map((s, i) => (
                                    <div key={i} className={styles.galleryItem}>
                                        <img src={s.src} alt="" className={styles.galleryImg}/>
                                        <span className={styles.galleryLabel}>Prediction: {s.predicted}</span>
                                    </div>
                                ))}
                        </div>
                    </div>
                </div>

                {/* Right: metrics table */}
                <div className={styles.classwiseRight}>
                    <table className={styles.metricsTable}>
                        <thead>
                            <tr>
                                <th>Class</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>#Samples</th>
                            </tr>
                        </thead>
                        <tbody>
                            {classMetrics.map((m, i) => (
                                <tr key={m.label}
                                    className={m.label === selectedClass ? styles.metricsRowActive : styles.metricsRow}
                                    onClick={() => setSelectedClass(m.label)}
                                    style={{borderLeft: `4px solid ${rowColor(i)}`}}
                                >
                                    <td style={{background: rowColor(i), color: 'white', padding: '6px 10px', fontWeight: 700}}>{m.label}</td>
                                    <td>{m.accuracy.toFixed(4)}</td>
                                    <td>{m.precision.toFixed(4)}</td>
                                    <td>{m.recall.toFixed(4)}</td>
                                    <td>{m.samples}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

/* ── Main modal ── */
const TrainReportModal = ({ onClose, epochMetrics, reportData, labels }) => {
    const accLines  = [{key: 'trainAcc', label: 'train', color: '#7ecfcf'}, {key: 'valAcc', label: 'test', color: '#f5a623'}];
    const lossLines = [{key: 'trainLoss', label: 'train', color: '#7ecfcf'}, {key: 'valLoss', label: 'test', color: '#f5a623'}];

    const hasReport = reportData && reportData.confusion && reportData.classMetrics;

    return (
        <div className={styles.overlay} onClick={e => {
            if (e.target !== e.currentTarget) return;
            const r = e.currentTarget.getBoundingClientRect();
            if (e.clientX < r.left || e.clientX > r.right - 17) return; // ignore scrollbar clicks
            onClose();
        }}>
            <div className={styles.modal}>
                <div className={styles.modalHeader}>
                    <span className={styles.modalTitle}>Report</span>
                    <div className={styles.modalHeaderBtns}>
                        <button className={styles.modalHeaderBtn} title="Close" onClick={onClose}>✕</button>
                    </div>
                </div>

                <div className={styles.modalBody}>
                    {/* Charts row */}
                    <div className={styles.chartsRow}>
                        <LineChart
                            data={epochMetrics}
                            title="Accuracy per epoch"
                            yLabel="Accuracy"
                            lines={accLines}
                        />
                        <LineChart
                            data={epochMetrics}
                            title="Loss per epoch"
                            yLabel="Loss"
                            lines={lossLines}
                        />
                    </div>

                    {hasReport ? (<>
                        <ConfusionMatrix labels={labels} confusion={reportData.confusion}/>
                        <ClasswiseAnalysis
                            labels={labels}
                            classMetrics={reportData.classMetrics}
                            samples={reportData.samples}
                        />
                    </>) : (
                        <div className={styles.evaluatingMsg}>
                            <div className={styles.evalSpinner}/>
                            <p>Evaluating model on training data…</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TrainReportModal;
