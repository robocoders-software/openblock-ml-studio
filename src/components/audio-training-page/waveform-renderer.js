/**
 * WaveformRenderer — DAW-style scrolling PCM waveform on a canvas.
 *
 * Maintains a ring buffer of {min, max} per pixel column and renders
 * a dense vertical-bar waveform (Audacity / Adobe Audition style).
 * Newest samples appear on the right; the history scrolls leftward.
 */
export class WaveformRenderer {
    constructor(canvas, options) {
        const o      = options || {};
        this.canvas  = canvas;
        this.color   = o.color    || '#9966FF';
        this.bgColor = o.bgColor  || '#111';
        this.histMs  = o.historyMs || 4000;
        this._hist   = [];
        this._lastTs = 0;
    }

    /* Push a Float32Array frame of PCM samples [-1..1].
       Returns true if a new entry was added to the ring buffer. */
    push(floatBuf) {
        const now   = performance.now();
        const W     = this.canvas.clientWidth || 300;
        const msPpx = this.histMs / W;
        if (now - this._lastTs < msPpx) return false;
        this._lastTs = now;

        let mn = Infinity, mx = -Infinity;
        for (let i = 0; i < floatBuf.length; i++) {
            if (floatBuf[i] < mn) mn = floatBuf[i];
            if (floatBuf[i] > mx) mx = floatBuf[i];
        }
        this._hist.push({mn, mx});
        if (this._hist.length > W) this._hist.shift();
        return true;
    }

    /* Render the current ring buffer to the canvas. */
    draw() {
        const canvas = this.canvas;
        const W = canvas.clientWidth  || 300;
        const H = canvas.clientHeight || 130;
        if (canvas.width  !== W) canvas.width  = W;
        if (canvas.height !== H) canvas.height = H;

        const ctx  = canvas.getContext('2d');
        const midY = H / 2;

        /* Background */
        ctx.fillStyle = this.bgColor;
        ctx.fillRect(0, 0, W, H);

        /* Subtle horizontal grid at ±33% and ±66% */
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth   = 1;
        [0.33, 0.66].forEach(f => {
            const y1 = midY - midY * f, y2 = midY + midY * f;
            ctx.beginPath(); ctx.moveTo(0, y1); ctx.lineTo(W, y1); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, y2); ctx.lineTo(W, y2); ctx.stroke();
        });
        /* Center line */
        ctx.strokeStyle = 'rgba(255,255,255,0.14)';
        ctx.beginPath(); ctx.moveTo(0, midY); ctx.lineTo(W, midY); ctx.stroke();

        if (!this._hist.length) return;

        /* Auto-gain: scale so the loudest entry fills 90% of half-height */
        let maxAbs = 0;
        for (let i = 0; i < this._hist.length; i++) {
            const a = Math.abs(this._hist[i].mn), b = Math.abs(this._hist[i].mx);
            if (a > maxAbs) maxAbs = a;
            if (b > maxAbs) maxAbs = b;
        }
        const scale = maxAbs > 0.01 ? (midY * 0.9) / maxAbs : midY * 0.03;

        /* Draw waveform bars — right-aligned, newest on right.
           Batched into a single path + fill instead of N fillRect calls. */
        const startX = W - this._hist.length;
        ctx.fillStyle   = this.color;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        for (let i = 0; i < this._hist.length; i++) {
            const {mn, mx} = this._hist[i];
            const x  = startX + i;
            const y1 = midY - mx * scale;
            const y2 = midY - mn * scale;
            ctx.rect(x, Math.min(y1, y2), 1, Math.abs(y2 - y1) || 1);
        }
        ctx.fill();
        ctx.globalAlpha = 1;

        /* Bright playhead at newest position */
        const last = this._hist[this._hist.length - 1];
        const ply1 = midY - last.mx * scale, ply2 = midY - last.mn * scale;
        ctx.fillStyle   = '#ffffff';
        ctx.globalAlpha = 0.55;
        ctx.fillRect(W - 2, Math.min(ply1, ply2), 2, Math.abs(ply2 - ply1) || 1);
        ctx.globalAlpha = 1;
    }

    /* Clear history for a fresh recording session. */
    reset() {
        this._hist   = [];
        this._lastTs = 0;
    }
}
