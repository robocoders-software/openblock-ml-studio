import React, {useState, useCallback} from 'react';
import PropTypes from 'prop-types';
import styles from './home-screen.css';
import MLProjectsPage    from '../ml-projects-page/ml-projects-page.jsx';
import CreateProjectModal from '../create-project-modal/create-project-modal.jsx';
import MLTrainingPage    from '../ml-training-page/ml-training-page.jsx';
import {deleteProjectData} from '../../lib/ml-engine.js';

/* ─── localStorage helpers ───────────────────────── */
const STORAGE_KEY = 'robocoders_ml_projects';
const loadProjects = () => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); }
    catch (_) { return []; }
};

/* Strip image data URLs before writing to localStorage – actual bytes live in IndexedDB */
const stripProject = p => ({
    ...p,
    trainingData: Object.fromEntries(
        Object.entries(p.trainingData || {}).map(([lbl, exs]) => [
            lbl,
            (exs || []).map(ex => ex.type === 'image' ? {id: ex.id, type: 'image'} : ex)
        ])
    )
});

const saveProjects = ps => {
    try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(ps.map(stripProject)));
    } catch (e) {
        console.warn('[MLStudio] localStorage save failed (quota?):', e.message);
    }
};

const generateId = () => Math.random().toString(36).slice(2, 10);

/* ─── Inline SVG icons ───────────────────────────── */
const BlocksIcon = () => (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="4"  y="28" width="24" height="16" rx="4" fill="#FF8C42"/>
        <rect x="4"  y="28" width="10" height="6"  rx="2" fill="#FF6B00"/>
        <rect x="36" y="16" width="24" height="16" rx="4" fill="#4C97FF"/>
        <rect x="36" y="16" width="10" height="6"  rx="2" fill="#1A73E8"/>
        <rect x="20" y="40" width="24" height="16" rx="4" fill="#9B59B6"/>
        <rect x="20" y="40" width="10" height="6"  rx="2" fill="#7D3C98"/>
    </svg>
);

const AIIcon = () => (
    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="24" cy="28" rx="16" ry="18" fill="#9B59B6" opacity="0.9"/>
        <ellipse cx="40" cy="28" rx="16" ry="18" fill="#7D3C98" opacity="0.9"/>
        <ellipse cx="32" cy="28" rx="8"  ry="18" fill="#B07FD4"/>
        <circle cx="18" cy="22" r="3" fill="white" opacity="0.8"/>
        <circle cx="26" cy="32" r="3" fill="white" opacity="0.8"/>
        <circle cx="38" cy="22" r="3" fill="white" opacity="0.8"/>
        <circle cx="46" cy="32" r="3" fill="white" opacity="0.8"/>
        <circle cx="32" cy="18" r="3" fill="white" opacity="0.8"/>
        <line x1="18" y1="22" x2="26" y2="32" stroke="white" strokeWidth="1.5" opacity="0.5"/>
        <line x1="26" y1="32" x2="38" y2="22" stroke="white" strokeWidth="1.5" opacity="0.5"/>
        <line x1="38" y1="22" x2="46" y2="32" stroke="white" strokeWidth="1.5" opacity="0.5"/>
        <line x1="18" y1="22" x2="32" y2="18" stroke="white" strokeWidth="1.5" opacity="0.5"/>
        <line x1="32" y1="18" x2="46" y2="32" stroke="white" strokeWidth="1.5" opacity="0.5"/>
        <line x1="28" y1="46" x2="36" y2="46" stroke="#FF8C1A" strokeWidth="2" strokeLinecap="round"/>
        <line x1="22" y1="50" x2="42" y2="50" stroke="#FF8C1A" strokeWidth="2" strokeLinecap="round"/>
        <line x1="28" y1="46" x2="22" y2="50" stroke="#FF8C1A" strokeWidth="2" strokeLinecap="round"/>
        <line x1="36" y1="46" x2="42" y2="50" stroke="#FF8C1A" strokeWidth="2" strokeLinecap="round"/>
    </svg>
);

/* ─── Home screen cards ──────────────────────────── */
const BLOCK_CARDS = [{id: 'blocks', title: 'Blocks', description: 'Code with playful puzzle-shaped blocks', age: 'Ages 7+', icon: <BlocksIcon />}];
const AI_CARDS    = [{id: 'aiml',   title: 'AI & Machine Learning', description: 'Train custom AI models to recognise images, text & numbers — then use them in your projects', age: 'Ages 10+', icon: <AIIcon />}];

/* ─── Main component ─────────────────────────────── */
const HomeScreen = ({onSelectMode}) => {
    /* Navigation state */
    const [view,             setView]    = useState('home');       // 'home' | 'mlProjects' | 'mlTraining'
    const [projects,         setProjects] = useState(loadProjects);
    const [activeProject,   setActive]  = useState(null);
    const [showCreateModal,  setShowCreate] = useState(false);

    /* ── Project CRUD ── */
    const createProject = useCallback(({name, type}) => {
        const p = {
            id: generateId(),
            name,
            type,
            labels: ['Class 1', 'Class 2'],
            trainingData: {},
            trained: false,
            createdAt: Date.now(),
            updatedAt: Date.now()
        };
        const updated = [...projects, p];
        setProjects(updated);
        saveProjects(updated);
        setShowCreate(false);
        setActive(p);
        setView('mlTraining');
    }, [projects]);

    const deleteProject = useCallback(projectOrId => {
        const id = typeof projectOrId === 'string' ? projectOrId : projectOrId.id;
        const updated = projects.filter(p => p.id !== id);
        setProjects(updated);
        saveProjects(updated);
        deleteProjectData(id).catch(() => {});
    }, [projects]);

    const updateProject = useCallback(updated => {
        setProjects(prev => {
            const next = prev.map(p => p.id === updated.id ? {...updated, updatedAt: Date.now()} : p);
            saveProjects(next);
            return next;
        });
        setActive(updated);
    }, []);

    const importProject = useCallback(() => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json,.mlproject';
        input.onchange = e => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = evt => {
                try {
                    const p = JSON.parse(evt.target.result);
                    if (!p.name || !p.type) throw new Error('Invalid project file');
                    const imported = {
                        ...p,
                        id: generateId(),
                        createdAt: Date.now(),
                        updatedAt: Date.now(),
                        trained: false,
                        trainingData: {}
                    };
                    const updated = [...projects, imported];
                    setProjects(updated);
                    saveProjects(updated);
                    setActive(imported);
                    setView('mlTraining');
                } catch (err) {
                    console.error('[MLStudio] Import failed:', err);
                }
            };
            reader.readAsText(file);
        };
        input.click();
    }, [projects]);

    /* ── Card click ── */
    const handleCard = id => {
        if (id === 'aiml') { setView('mlProjects'); return; }
        onSelectMode(id);
    };

    /* ── Use in Blocks from training page ── */
    const handleUseInBlocks = () => {
        onSelectMode('blocks');
    };

    /* ── Render sub-views ── */
    if (view === 'mlProjects') {
        return (
            <>
                <MLProjectsPage
                    projects={projects}
                    onBack={() => setView('home')}
                    onCreate={() => setShowCreate(true)}
                    onOpen={p => { setActive(p); setView('mlTraining'); }}
                    onDelete={deleteProject}
                    onImport={importProject}
                />
                {showCreateModal && (
                    <CreateProjectModal
                        onCancel={() => setShowCreate(false)}
                        onCreate={createProject}
                    />
                )}
            </>
        );
    }

    if (view === 'mlTraining' && activeProject) {
        return (
            <MLTrainingPage
                project={activeProject}
                onBack={() => setView('mlProjects')}
                onUseInBlocks={handleUseInBlocks}
                onUpdateProject={updateProject}
            />
        );
    }

    /* ── Home screen ── */
    return (
        <div className={styles.homeWrapper}>
            <div className={styles.card}>
                <h1 className={styles.heading}>What would you like to do?</h1>

                <div className={styles.section}>
                    <p className={styles.sectionTitle}>Block Coding</p>
                    <div className={styles.modeGrid}>
                        {BLOCK_CARDS.map(m => <ModeCard key={m.id} mode={m} onClick={handleCard} />)}
                    </div>
                </div>

                <div className={styles.section}>
                    <p className={styles.sectionTitle}>AI & Machine Learning</p>
                    <div className={styles.modeGrid}>
                        {AI_CARDS.map(m => <ModeCard key={m.id} mode={m} onClick={handleCard} />)}
                    </div>
                </div>
            </div>
        </div>
    );
};

const ModeCard = ({mode, onClick}) => (
    <div
        className={styles.modeCard}
        onClick={() => onClick(mode.id)}
        role="button"
        tabIndex={0}
        onKeyPress={e => e.key === 'Enter' && onClick(mode.id)}
    >
        {mode.age && <span className={styles.ageBadge}>{mode.age}</span>}
        <div className={styles.modeIcon}>{mode.icon}</div>
        <p className={styles.modeTitle}>{mode.title}</p>
        <p className={styles.modeDescription}>{mode.description}</p>
    </div>
);

HomeScreen.propTypes = { onSelectMode: PropTypes.func.isRequired };
ModeCard.propTypes   = { mode: PropTypes.object.isRequired, onClick: PropTypes.func.isRequired };

export default HomeScreen;
