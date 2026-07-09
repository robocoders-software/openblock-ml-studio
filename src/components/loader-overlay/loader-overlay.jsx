import React from 'react';
import PropTypes from 'prop-types';
import Spinner from 'openblock-gui/src/components/spinner/spinner.jsx';

/* Lightweight, blocking loader shown while a short action runs (upload, save, delete, …).
   Distinct from MLLoader (the full-screen branded loader used for model load/training):
   this is a small centred card over a dim backdrop, PictoBlox-style. Self-contained inline
   styles — no CSS module — so it can be dropped in anywhere.
   It is `position: fixed`, so it covers the whole window regardless of where it is mounted. */
const OVERLAY_STYLE = {
    position: 'fixed',
    inset: 0,
    zIndex: 10000,
    background: 'rgba(15, 23, 42, 0.35)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
};

const CARD_STYLE = {
    background: '#ffffff',
    borderRadius: '12px',
    padding: '22px 30px',
    minWidth: '190px',
    maxWidth: '80vw',
    boxShadow: '0 10px 34px rgba(0, 0, 0, 0.20)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    gap: '12px'
};

const MESSAGE_STYLE = {
    margin: 0,
    fontSize: '14px',
    lineHeight: 1.4,
    color: '#1f2937',
    fontWeight: 500,
    textAlign: 'center'
};

const LoaderOverlay = ({message = 'Please wait…'}) => (
    <div style={OVERLAY_STYLE} aria-live="polite" aria-busy="true">
        <div style={CARD_STYLE} role="status">
            <Spinner level="info" />
            {message && <p style={MESSAGE_STYLE}>{message}</p>}
        </div>
    </div>
);

LoaderOverlay.propTypes = {
    message: PropTypes.string
};

export default LoaderOverlay;
