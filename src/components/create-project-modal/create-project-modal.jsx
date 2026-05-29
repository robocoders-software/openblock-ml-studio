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
        desc: 'Train a model to make decisions from number values',
        comingSoon: true
    },
    {
        id: 'sounds',
        icon: '🔊',
        title: 'Sounds',
        desc: 'Train a model to recognise different sounds from your microphone'
    }
];

const CreateProjectModal = ({onCancel, onCreate, existingNames = []}) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [type, setType] = useState('images');

    const nameTrimmed = name.trim();
    const isDuplicate = nameTrimmed.length > 0 &&
        existingNames.some(n => n.toLowerCase() === nameTrimmed.toLowerCase());

    const handleCreate = () => {
        if (!nameTrimmed || isDuplicate) return;
        onCreate({name: nameTrimmed, type, description: description.trim()});
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
                    className={`${styles.nameInput} ${isDuplicate ? styles.nameInputError : ''}`}
                    type="text"
                    placeholder="e.g. Detect Cats and Dogs"
                    value={name}
                    onChange={e => setName(e.target.value)}
                    onKeyPress={e => e.key === 'Enter' && !isDuplicate && handleCreate()}
                    autoFocus
                />
                {isDuplicate && (
                    <p className={styles.errorMsg}>A project with this name already exists.</p>
                )}

                <label className={styles.label}>
                    Description <span className={styles.optionalTag}>(optional)</span>
                </label>
                <textarea
                    className={styles.descInput}
                    placeholder="Briefly describe what your model will do…"
                    value={description}
                    onChange={e => setDescription(e.target.value)}
                    rows={2}
                />

                <label className={styles.label}>What kind of data will you use to train your model?</label>
                <div className={styles.typeGrid}>
                    {PROJECT_TYPES.map(t => (
                        <div
                            key={t.id}
                            className={[
                                styles.typeCard,
                                t.comingSoon ? styles.typeCardComingSoon : '',
                                !t.comingSoon && type === t.id ? styles.typeCardSelected : ''
                            ].join(' ')}
                            onClick={() => !t.comingSoon && setType(t.id)}
                            role="radio"
                            aria-checked={!t.comingSoon && type === t.id}
                            tabIndex={t.comingSoon ? -1 : 0}
                            onKeyPress={e => !t.comingSoon && e.key === 'Enter' && setType(t.id)}
                        >
                            {t.comingSoon && (
                                <span className={styles.comingSoonBadge}>Coming Soon</span>
                            )}
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
                        disabled={!nameTrimmed || isDuplicate}
                    >
                        Create project
                    </button>
                </div>
            </div>
        </div>
    );
};

CreateProjectModal.propTypes = {
    onCancel:      PropTypes.func.isRequired,
    onCreate:      PropTypes.func.isRequired,
    existingNames: PropTypes.arrayOf(PropTypes.string)
};

export default CreateProjectModal;
