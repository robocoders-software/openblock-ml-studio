/* ── Audio utility functions ── */

/**
 * Encode a Float32Array of mono PCM samples into a WAV Blob.
 * @param {Float32Array} samples  PCM in [-1, 1]
 * @param {number}       sampleRate
 * @returns {Blob}  audio/wav
 */
export const float32ToWavBlob = (samples, sampleRate) => {
    const n   = samples.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const v   = new DataView(buf);

    const str = (off, s) => {
        for (let i = 0; i < s.length; i++) v.setUint8(off + i, s.charCodeAt(i));
    };

    str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true);
    str(8, 'WAVE'); str(12, 'fmt ');
    v.setUint32(16, 16, true);       // PCM chunk size
    v.setUint16(20, 1,  true);       // PCM format
    v.setUint16(22, 1,  true);       // mono
    v.setUint32(24, sampleRate, true);
    v.setUint32(28, sampleRate * 2, true);  // byte rate (SR * channels * BPS/8)
    v.setUint16(32, 2,  true);       // block align
    v.setUint16(34, 16, true);       // bits per sample
    str(36, 'data'); v.setUint32(40, n * 2, true);

    let off = 44;
    for (let i = 0; i < n; i++) {
        const x = Math.max(-1, Math.min(1, samples[i]));
        v.setInt16(off, x < 0 ? x * 32768 : x * 32767, true);
        off += 2;
    }
    return new Blob([buf], {type: 'audio/wav'});
};

/**
 * Convert a MediaRecorder Blob (webm/ogg) to a WAV Blob via AudioContext decode.
 * Returns null on failure.
 */
export const blobToWavBlob = async blob => {
    try {
        const arrayBuf = await blob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded  = await audioCtx.decodeAudioData(arrayBuf);
        const pcm      = decoded.getChannelData(0);
        audioCtx.close();
        return float32ToWavBlob(pcm, decoded.sampleRate);
    } catch (_) {
        return null;
    }
};

/**
 * Compute the RMS (root-mean-square) of a Float32Array frame.
 * Result is in [0, 1] — use as a noise/level indicator.
 */
export const computeRMS = buf => {
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    return Math.sqrt(sum / buf.length);
};
