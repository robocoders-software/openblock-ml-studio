/* ── One-shot audio classification — PURE logic ───────────────────────
   Extracted from ml-engine.js so it has ZERO dependencies (no tf, no
   window, no DOM) and can be unit-tested in isolation under plain Node.

   These helpers own the two things that were historically buggy:
     1. turning a speech-commands listen() result into user-facing matches
        (align scores → wordLabels, drop the internal noise class, sort);
     2. deciding, over a short sampling window, WHICH frame to return — so a
        one-shot waits for the actual sound instead of grabbing the first
        (usually silent) frame, and never hangs.
   ─────────────────────────────────────────────────────────────────── */

/* Convert a raw listen() result's score vector + the recognizer's word labels
   into sorted, noise-filtered matches with probabilities as 0-100 percentages.
   scores[i] MUST correspond to labels[i] (both come from the transfer model). */
export const reduceFrameToMatches = (scores, labels) => {
    if (!scores || !Array.isArray(labels)) return [];
    return labels
        .map((label, i) => ({label, prob: (scores[i] || 0) * 100}))
        // '_background_noise_' is an internal training aid, never a user-facing class.
        .filter(m => m.label !== '_background_noise_')
        .sort((a, b) => b.prob - a.prob);
};

/* A deterministic, timer-free state machine for one one-shot capture.
   The caller drives it: feed each frame via offer(matches, nowMs); when a
   safety timeout fires or the listener ends, call flush(). Keeping time and
   I/O OUT of here is what makes it testable with a fake clock.

     maxMs     – how long to keep sampling before returning the best frame
     earlyExit – a frame whose top prob (%) reaches this ends sampling at once */
export const createOneShotSampler = ({maxMs = 2000, earlyExit = 75, startTime = 0} = {}) => {
    let best = null;
    const consider = matches => {
        if (matches && matches.length && (!best || matches[0].prob > best[0].prob)) {
            best = matches;
        }
    };
    return {
        /* Feed one frame. Returns a result { matches, reason } when sampling
           should STOP, or null to keep going. */
        offer (matches, nowMs) {
            consider(matches);
            if (matches && matches.length && matches[0].prob >= earlyExit) {
                return {matches, reason: 'confident'};
            }
            if (nowMs - startTime >= maxMs) {
                return {matches: best || [], reason: 'timeout'};
            }
            return null;
        },
        /* Force a result now (safety timeout, or the stream ended): the best
           frame seen so far, or [] if none arrived. */
        flush () {
            return {matches: best || [], reason: 'flush'};
        },
        peekBest () { return best; }
    };
};
