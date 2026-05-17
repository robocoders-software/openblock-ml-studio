import React, {useState, useCallback, useEffect} from 'react';
import PropTypes from 'prop-types';
import MLProjectsPage    from '../ml-projects-page/ml-projects-page.jsx';
import CreateProjectModal from '../create-project-modal/create-project-modal.jsx';
import MLTrainingPage    from '../ml-training-page/ml-training-page.jsx';
import MLLoader          from '../ml-loader/ml-loader.jsx';
import {deleteProjectData} from '../../lib/ml-engine.js';
import {saveTextProject, saveImageProject, saveAudioProject} from '../../lib/project-persistence.js';

const getIpc = () => {
    try { return window.require('electron').ipcRenderer; } catch (_) { return null; }
};

const generateId = () => Math.random().toString(36).slice(2, 10);

/* ─── MLStudioApp ─────────────────────────────────
   Project list is persisted on disk:
     <userData>/ml-projects/<projectId>/project.json

   On mount we ask the main process to scan that
   directory and return all project metadata.
   Writes happen whenever a training page saves.
──────────────────────────────────────────────────── */
const MLStudioApp = ({onEnterBlocks, onBack}) => {
    const [view,            setView]      = useState('mlProjects');
    const [projects,        setProjects]  = useState([]);
    const [loading,         setLoading]   = useState(true);
    const [loadingMsg,      setLoadingMsg] = useState('Loading projects…');
    const [activeProject,   setActive]    = useState(null);
    const [showCreateModal, setShowCreate] = useState(false);

    /* ── Load project list from disk on mount ── */
    useEffect(() => {
        const ipc = getIpc();
        if (!ipc) { setLoading(false); return; }
        ipc.invoke('ml-list-projects').then(list => {
            setProjects(Array.isArray(list) ? list : []);
        }).catch(err => {
            console.error('[MLStudio] ml-list-projects failed:', err);
        }).finally(() => {
            setLoading(false);
        });
    }, []);

    /* ── Refresh project list from disk (called after save/delete) ── */
    const refreshProjects = useCallback(() => {
        const ipc = getIpc();
        if (!ipc) return;
        ipc.invoke('ml-list-projects').then(list => {
            setProjects(Array.isArray(list) ? list : []);
        }).catch(err => {
            console.error('[MLStudio] ml-list-projects refresh failed:', err);
        });
    }, []);

    /* ── Project CRUD ── */
    const createProject = useCallback(({name, type}) => {
        const p = {
            id:          generateId(),
            name,
            type,
            labels:      ['Class 1', 'Class 2'],
            trainingData: {},
            trained:     false,
            createdAt:   Date.now(),
            updatedAt:   Date.now()
        };
        /* Write the initial project.json to disk immediately so it shows up in the list */
        const ipc = getIpc();
        if (ipc) {
            const saveData = JSON.stringify({
                id:        p.id,
                name:      p.name,
                type:      p.type,
                labels:    p.labels,
                trained:   false,
                createdAt: p.createdAt,
                updatedAt: p.updatedAt,
                savedAt:   p.createdAt
            });
            ipc.invoke('ml-write-file', p.id, 'project.json', saveData).catch(() => {});
        }
        setShowCreate(false);
        setActive(p);
        setView('mlTraining');
        /* Refresh list after a short delay so the new project.json is on disk */
        setTimeout(refreshProjects, 500);
    }, [refreshProjects]);

    const deleteProject = useCallback(projectOrId => {
        const id = typeof projectOrId === 'string' ? projectOrId : projectOrId.id;
        deleteProjectData(id).catch(() => {});
        const ipc = getIpc();
        if (ipc) ipc.invoke('ml-delete-project', id).catch(() => {});
        // Clear pending project in main process so will-download doesn't try to bundle deleted dir
        if (ipc) ipc.send('ml-clear-pending-project', id);
        /* If the deleted project was the active model, clear the global ref and
           notify app.jsx so it doesn't restore the deleted model on Back. */
        if (typeof window !== 'undefined' &&
            window.__openblockMLModel &&
            window.__openblockMLModel.projectId === id) {
            window.__openblockMLModel = null;
            window.dispatchEvent(new CustomEvent('robocoders:ml-model-deleted', {detail: {projectId: id}}));
        }
        /* Optimistically remove from list, then refresh from disk */
        setProjects(prev => prev.filter(p => p.id !== id));
        setTimeout(refreshProjects, 300);
    }, [refreshProjects]);

    const updateProject = useCallback(updated => {
        setActive(updated);
        /* The training pages write project.json to disk themselves via saveXxxProject.
           We just refresh the in-memory list so the dashboard stays current. */
        setProjects(prev => {
            const exists = prev.some(p => p.id === updated.id);
            if (exists) {
                return prev.map(p => p.id === updated.id
                    ? {...p, name: updated.name, labels: updated.labels, trained: updated.trained, updatedAt: Date.now()}
                    : p
                );
            }
            return [...prev, updated];
        });
    }, []);

    /* When returning to the project list, refresh from disk to pick up any saves */
    const handleBackToList = useCallback(() => {
        setView('mlProjects');
        refreshProjects();
    }, [refreshProjects]);

    /* ── Import .ob file ── */
    const importProject = useCallback(() => {
        const ipc = getIpc();
        if (!ipc) return;
        /* Use Electron's file dialog to pick an .ob file */
        ipc.invoke('ml-open-ob-file').then(result => {
            if (!result || result.canceled || !result.projectId) return;
            /* The main process extracted the .ob and we have the projectId */
            refreshProjects();
        }).catch(() => {
            /* Fallback: just open a file input for JSON import */
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
                            id:           generateId(),
                            createdAt:    Date.now(),
                            updatedAt:    Date.now(),
                            trained:      false,
                            trainingData: {}
                        };
                        setActive(imported);
                        setView('mlTraining');
                    } catch (err) {
                        console.error('[MLStudio] Import failed:', err);
                    }
                };
                reader.readAsText(file);
            };
            input.click();
        });
    }, [refreshProjects]);

    /* Show branded loader during initial project list fetch */
    if (loading) return <MLLoader message={loadingMsg} />;

    /* Open a project with a brief transition loader */
    const handleOpenProject = p => {
        setLoadingMsg('Opening project…');
        setLoading(true);
        setActive(p);
        /* Let the loader render for one frame, then switch view */
        setTimeout(() => {
            setView('mlTraining');
            setLoading(false);
        }, 0);
    };

    /* ── View routing ── */
    if (view === 'mlTraining' && activeProject) {
        return (
            <MLTrainingPage
                project={activeProject}
                onBack={handleBackToList}
                onUseInBlocks={onEnterBlocks}
                onUpdateProject={updateProject}
            />
        );
    }

    return (
        <>
            <MLProjectsPage
                projects={projects}
                loading={false}
                onBack={onBack}
                onCreate={() => setShowCreate(true)}
                onOpen={handleOpenProject}
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
    onBack:        PropTypes.func.isRequired
};

export default MLStudioApp;
