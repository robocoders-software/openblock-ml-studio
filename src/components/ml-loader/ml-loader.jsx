import React from 'react';
import PropTypes from 'prop-types';
import styles from './ml-loader.css';

const MLLoader = ({message = 'Loading…'}) => (
    <div className={styles.overlay}>
        <div className={styles.card}>
            <div className={styles.spinnerRing}>
                <div className={styles.spinnerTrack} />
                <div className={styles.spinnerFill} />
            </div>
            <p className={styles.appName}>Machine Learning Environment</p>
            {message && <p className={styles.message}>{message}</p>}
            <p className={styles.tagline}>
                {'Code • Build • Innovate'}
                <br />
                {'with '}
                <span className={styles.taglineBrand}>{'RoboCoders Studio'}</span>
            </p>
        </div>
    </div>
);

MLLoader.propTypes = {
    message: PropTypes.string
};

export default MLLoader;
