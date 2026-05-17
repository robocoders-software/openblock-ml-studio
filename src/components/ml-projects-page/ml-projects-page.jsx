import React, { useState, useRef, useEffect } from 'react';
import styles from './ml-projects-page.css';
import openblockLogo from '../openblock-logo.svg';

const TYPE_LABELS = {
    images: 'Image Classifier',
    text: 'Text Classifier',
    numbers: 'Number Classifier',
    sounds: 'Sound Classifier'
};

const MLProjectsPage = ({ projects = [], loading = false, onBack, onCreate, onOpen, onDelete, onRename, onUpdateDescription, onImport, onExport }) => {
    const [search, setSearch] = useState('');
    const [menuOpenId, setMenuOpenId] = useState(null);
    const [fileMenuOpen, setFileMenuOpen] = useState(false);
    const [renamingId, setRenamingId] = useState(null);
    const [renameValue, setRenameValue] = useState('');
    const [descEditId, setDescEditId] = useState(null);
    const [descValue, setDescValue] = useState('');
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

    const filtered = projects.filter(p =>
        p.name && p.name.toLowerCase().includes(search.toLowerCase())
    );

    const handleRowClick = (project, e) => {
        if (e.target.closest(`.${styles.menuCell}`)) return;
        onOpen && onOpen(project);
    };

    const handleMenuToggle = (e, id) => {
        e.stopPropagation();
        setMenuOpenId(prev => (prev === id ? null : id));
    };

    const handleDelete = (e, project) => {
        e.stopPropagation();
        setMenuOpenId(null);
        onDelete && onDelete(project);
    };

    const handleRenameStart = (e, project) => {
        e.stopPropagation();
        setMenuOpenId(null);
        setRenamingId(project.id || project.name);
        setRenameValue(project.name);
    };

    const commitRename = project => {
        const trimmed = renameValue.trim();
        if (trimmed && trimmed !== project.name) {
            onRename && onRename(project, trimmed);
        }
        setRenamingId(null);
    };

    const handleRenameKey = (e, project) => {
        if (e.key === 'Enter') { e.preventDefault(); commitRename(project); }
        if (e.key === 'Escape') { setRenamingId(null); }
    };

    const handleDescEdit = (e, project) => {
        e.stopPropagation();
        setDescEditId(project.id || project.name);
        setDescValue(project.description || '');
    };

    const commitDescEdit = project => {
        const trimmed = descValue.trim();
        if (trimmed !== (project.description || '').trim()) {
            onUpdateDescription && onUpdateDescription(project, trimmed);
        }
        setDescEditId(null);
    };

    const handleDescKey = (e, project) => {
        if (e.key === 'Enter') { e.preventDefault(); commitDescEdit(project); }
        if (e.key === 'Escape') { setDescEditId(null); }
    };

    const handleExport = (e, project) => {
        e.stopPropagation();
        setMenuOpenId(null);
        if (onExport) { onExport(project); return; }
        /* Default: download project metadata as JSON */
        const data = {
            name: project.name,
            type: project.type,
            labels: project.labels || [],
            createdAt: project.createdAt
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${project.name || 'ml-project'}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    const formatDate = project => {
        const raw = project.updatedAt || project.createdAt || project.savedAt;
        if (!raw) return '—';
        return new Date(raw).toLocaleString();
    };

    return (
        <div className={styles.page}>
            {/* Top bar */}
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
                                <button onClick={() => { setFileMenuOpen(false); onCreate && onCreate(); }}>New ML Project</button>
                            </div>
                        )}
                    </div>
                    <button className={styles.navBtn}>Help</button>
                </nav>
                <div className={styles.headerSpacer} />
                <button className={styles.backBtn} onClick={onBack}>&#8592; Back</button>
            </div>

            {/* Sub-header */}
            <div className={styles.subHeader}>
                <h2 className={styles.myProjects}>My Projects</h2>
                <div className={styles.searchWrap}>
                    <svg
                        width="16"
                        height="16"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="#999"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    >
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <input
                        type="text"
                        placeholder="Search projects…"
                        value={search}
                        onChange={e => setSearch(e.target.value)}
                    />
                </div>
                <button className={styles.createBtn} onClick={onCreate}>
                    Create New Project
                </button>
            </div>

            {/* Main content */}
            <div className={styles.tableWrap}>
                {loading ? (
                    <div className={styles.emptyState}>Loading projects…</div>
                ) : filtered.length === 0 ? (
                    <div className={styles.emptyState}>
                        {search
                            ? `No projects match "${search}".`
                            : 'No projects yet. Create your first ML project!'}
                    </div>
                ) : (
                    <table className={styles.table}>
                        <thead>
                            <tr>
                                <th>Project Details</th>
                                <th>Type</th>
                                <th>No. of Classes</th>
                                <th>Last Updated</th>
                                <th>Status</th>
                                <th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.map(project => {
                                const rowKey = project.id || project.name;
                                return (
                                    <tr
                                        key={rowKey}
                                        onClick={e => handleRowClick(project, e)}
                                    >
                                        <td>
                                            {renamingId === rowKey ? (
                                                <input
                                                    className={styles.renameInput}
                                                    autoFocus
                                                    value={renameValue}
                                                    onChange={e => setRenameValue(e.target.value)}
                                                    onBlur={() => commitRename(project)}
                                                    onKeyDown={e => handleRenameKey(e, project)}
                                                    onClick={e => e.stopPropagation()}
                                                />
                                            ) : (
                                                <>
                                                    <div className={styles.projName}>{project.name}</div>
                                                    {descEditId === rowKey ? (
                                                        <input
                                                            className={styles.descEditInput}
                                                            autoFocus
                                                            value={descValue}
                                                            placeholder="Add a description…"
                                                            onChange={e => setDescValue(e.target.value)}
                                                            onBlur={() => commitDescEdit(project)}
                                                            onKeyDown={e => handleDescKey(e, project)}
                                                            onClick={e => e.stopPropagation()}
                                                        />
                                                    ) : (
                                                        <div
                                                            className={project.description ? styles.projSub : styles.projSubAdd}
                                                            onClick={e => handleDescEdit(e, project)}
                                                            title={project.description ? 'Click to edit description' : 'Click to add description'}
                                                        >
                                                            {project.description || '+ Add description'}
                                                        </div>
                                                    )}
                                                </>
                                            )}
                                        </td>
                                        <td>
                                            <span className={styles.typeBadge}>
                                                {TYPE_LABELS[project.type] || project.type || '—'}
                                            </span>
                                        </td>
                                        <td>
                                            {Array.isArray(project.labels)
                                                ? project.labels.length
                                                : '—'}
                                        </td>
                                        <td>{formatDate(project)}</td>
                                        <td>
                                            <span className={styles.statusText}>
                                                {project.trained ? 'Model Trained' : 'Model Not Trained'}
                                            </span>
                                        </td>
                                        <td className={styles.menuCell}>
                                            <button
                                                className={styles.menuBtn}
                                                onClick={e => handleMenuToggle(e, rowKey)}
                                            >
                                                &#8942;
                                            </button>
                                            {menuOpenId === rowKey && (
                                                <div className={styles.dropdown}>
                                                    <button
                                                        className={styles.dropdownAction}
                                                        onClick={e => handleRenameStart(e, project)}
                                                    >
                                                        Rename
                                                    </button>
                                                    <button
                                                        className={styles.dropdownAction}
                                                        onClick={e => handleExport(e, project)}
                                                    >
                                                        Export
                                                    </button>
                                                    <button
                                                        className={styles.dropdownDanger}
                                                        onClick={e => handleDelete(e, project)}
                                                    >
                                                        Delete
                                                    </button>
                                                </div>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                )}
            </div>
        </div>
    );
};

export default MLProjectsPage;
