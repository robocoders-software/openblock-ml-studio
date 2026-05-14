import React, {useState, useCallback} from 'react';
import PropTypes from 'prop-types';
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

/* ─── MLStudioApp ─────────────────────────────────
   Manages the ML studio experience: project list and
   training. Navigation back to the home screen is
   delegated upward via onBack — this component has
   no knowledge of HomeScreen or the blocks editor.
──────────────────────────────────────────────────── */
const MLStudioApp = ({onEnterBlocks, onBack}) => {
    const [view,            setView]      = useState('mlProjects');
    const [projects,        setProjects]  = useState(loadProjects);
    const [activeProject,   setActive]    = useState(null);
    const [showCreateModal, setShowCreate] = useState(false);

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
            const next = prev.map(p =>
                p.id === updated.id ? {...updated, updatedAt: Date.now()} : p
            );
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

    /* ── View routing ── */
    if (view === 'mlTraining' && activeProject) {
        return (
            <MLTrainingPage
                project={activeProject}
                onBack={() => setView('mlProjects')}
                onUseInBlocks={onEnterBlocks}
                onUpdateProject={updateProject}
            />
        );
    }

    return (
        <>
            <MLProjectsPage
                projects={projects}
                onBack={onBack}
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
};

MLStudioApp.propTypes = {
    onEnterBlocks: PropTypes.func.isRequired,
    onBack: PropTypes.func.isRequired
};

export default MLStudioApp;
