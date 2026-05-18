import React from 'react';
import PropTypes from 'prop-types';
import styles from './app-dialog.css';

const ICONS = {
    info:     'i',
    warning:  '!',
    error:    '✕',
    question: '?'
};

const ICON_COLORS = {
    info:     {bg: 'rgba(0,74,173,0.12)',   fg: '#004AAD'},
    warning:  {bg: 'rgba(255,140,0,0.12)',  fg: '#FF8C00'},
    error:    {bg: 'rgba(229,57,53,0.12)',  fg: '#e53935'},
    question: {bg: 'rgba(0,74,173,0.12)',   fg: '#004AAD'}
};

const DANGER_WORDS = ["don't save", 'discard', 'delete', 'remove', 'overwrite'];

const isDanger = label => DANGER_WORDS.some(w => label.toLowerCase().includes(w));

const AppDialog = ({type, title, message, detail, buttons, defaultId, onButtonClick}) => {
    const colors = ICON_COLORS[type] || ICON_COLORS.info;
    const symbol = ICONS[type] || 'i';

    return (
        <div className={styles.overlay}>
            <div className={styles.modal}>
                <div className={styles.header}>
                    <div
                        className={styles.iconWrap}
                        style={{background: colors.bg}}
                    >
                        <span
                            className={styles.iconSymbol}
                            style={{color: colors.fg}}
                        >
                            {symbol}
                        </span>
                    </div>
                    <div className={styles.titleBlock}>
                        <h2 className={styles.title}>{title}</h2>
                        <p className={styles.message}>{message}</p>
                        {detail && <p className={styles.detail}>{detail}</p>}
                    </div>
                </div>
                <div className={styles.actions}>
                    {buttons.map((label, idx) => {
                        let cls = styles.btnSecondary;
                        if (idx === defaultId) cls = styles.btnPrimary;
                        else if (isDanger(label)) cls = styles.btnDanger;
                        return (
                            <button
                                key={`${label}-${idx}`}
                                className={cls}
                                onClick={() => onButtonClick(idx)}
                            >
                                {label}
                            </button>
                        );
                    })}
                </div>
            </div>
        </div>
    );
};

AppDialog.propTypes = {
    type:          PropTypes.oneOf(['info', 'warning', 'error', 'question']),
    title:         PropTypes.string,
    message:       PropTypes.string,
    detail:        PropTypes.string,
    buttons:       PropTypes.arrayOf(PropTypes.string),
    defaultId:     PropTypes.number,
    onButtonClick: PropTypes.func.isRequired
};

AppDialog.defaultProps = {
    type:      'info',
    title:     '',
    message:   '',
    detail:    null,
    buttons:   ['OK'],
    defaultId: 0
};

export default AppDialog;
