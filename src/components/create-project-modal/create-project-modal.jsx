import React, {useState} from 'react';
import PropTypes from 'prop-types';
import styles from './create-project-modal.css';

const PROJECT_TYPES = [
    {
        id: 'images',
        icon: '🖼️',
        title: 'Images',
        desc: 'Train a model to recognise pictures from your camera or files'
    },
    {
        id: 'text',
        icon: '📝',
        title: 'Text',
        desc: 'Train a model to understand different kinds of phrases or sentences'
    },
    {
        id: 'numbers',
        icon: '🔢',
        title: 'Numbers',
        desc: 'Train a model to make decisions from number values'
    },
    {
        id: 'sounds',
        icon: '🔊',
        title: 'Sounds',
        desc: 'Train a model to recognise different sounds from your microphone'
    }
];

const CreateProjectModal = ({onCancel, onCreate}) => {
    const [name, setName] = useState('');
    const [type, setType] = useState('images');

    const handleCreate = () => {
        if (!name.trim()) return;
        onCreate({name: name.trim(), type});
    };

    return (
        <div className={styles.overlay}>
            <div className={styles.modal}>
                <h2 className={styles.modalTitle}>Create a new ML project</h2>
                <p className={styles.modalSubtitle}>
                    Choose a name and the type of data your model will learn from.
                </p>

                <label className={styles.label}>Project name</label>
                <input
                    className={styles.nameInput}
                    type="text"
                    placeholder="e.g. Detect Cats and Dogs"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    onKeyPress={e => e.key === 'Enter' && handleCreate()}
                    autoFocus
                />

                <label className={styles.label}>What kind of data will you use to train your model?</label>
                <div className={styles.typeGrid}>
                    {PROJECT_TYPES.map(t => (
                        <div
                            key={t.id}
                            className={`${styles.typeCard} ${type === t.id ? styles.typeCardSelected : ''}`}
                            onClick={() => setType(t.id)}
                            role="radio"
                            aria-checked={type === t.id}
                            tabIndex={0}
                            onKeyPress={e => e.key === 'Enter' && setType(t.id)}
                        >
                            <div className={styles.typeCardIcon}>{t.icon}</div>
                            <p className={styles.typeCardTitle}>{t.title}</p>
                            <p className={styles.typeCardDesc}>{t.desc}</p>
                        </div>
                    ))}
                </div>

                <div className={styles.footer}>
                    <button className={styles.cancelBtn} onClick={onCancel}>Cancel</button>
                    <button
                        className={styles.createBtn}
                        onClick={handleCreate}
                        disabled={!name.trim()}
                    >
                        Create project
                    </button>
                </div>
            </div>
        </div>
    );
};

CreateProjectModal.propTypes = {
    onCancel: PropTypes.func.isRequired,
    onCreate: PropTypes.func.isRequired
};

export default CreateProjectModal;
