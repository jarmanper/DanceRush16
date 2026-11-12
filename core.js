/* Dance Rush '16: browser port of pose_engine.py feature math + the Random Forest.
 * Works in the browser (window.DanceCore) and in Node (module.exports) for testing. */
(function (root) {
  const MOTION_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  const FRAMES = 12; // RESAMPLED_FRAMES in settings.py

  /** Translation/scale normalization (same as normalize_landmarks in Python). */
  function normalizeLandmarks(lm, aspect) {
    const xy = lm.map((p) => [p.x * aspect, p.y]);
    const pelvis = [(xy[23][0] + xy[24][0]) / 2, (xy[23][1] + xy[24][1]) / 2];
    const shoulders = [(xy[11][0] + xy[12][0]) / 2, (xy[11][1] + xy[12][1]) / 2];
    const torso = Math.hypot(shoulders[0] - pelvis[0], shoulders[1] - pelvis[1]);
    const shoulderWidth = Math.hypot(xy[11][0] - xy[12][0], xy[11][1] - xy[12][1]);
    const scale = Math.max(torso, shoulderWidth, 0.08);
    return xy.map((p) => [Math.fround((p[0] - pelvis[0]) / scale), Math.fround((p[1] - pelvis[1]) / scale)]);
  }

  /** Linear resample of a variable-FPS clip to FRAMES poses (np.interp equivalent). */
  function resample(ts, poses, count) {
    const n = ts.length;
    if (n < 3 || ts[n - 1] <= ts[0]) return null;
    for (let i = 1; i < n; i++) if (ts[i] <= ts[i - 1]) return null;
    const out = [];
    let seg = 0;
    for (let k = 0; k < count; k++) {
      const g = ts[0] + ((ts[n - 1] - ts[0]) * k) / (count - 1);
      while (seg < n - 2 && g > ts[seg + 1]) seg++;
      const w = Math.min(1, Math.max(0, (g - ts[seg]) / (ts[seg + 1] - ts[seg])));
      const frame = [];
      for (let j = 0; j < 33; j++) {
        const a = poses[seg][j], b = poses[seg + 1][j];
        frame.push([a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w]);
      }
      out.push(frame);
    }
    return out;
  }

  function mean(arr) { return arr.reduce((s, v) => s + v, 0) / arr.length; }
  function std(arr) { const m = mean(arr); return Math.sqrt(mean(arr.map((v) => (v - m) * (v - m)))); }

  /** 468-number feature vector, same order as sequence_features in Python. */
  function sequenceFeatures(ts, poses) {
    const a = resample(ts, poses, FRAMES);
    if (!a) return null;
    const f = [];
    // mean + std over time of all 33 joints (x, y)
    for (let j = 0; j < 33; j++) for (let c = 0; c < 2; c++) f.push(mean(a.map((fr) => fr[j][c])));
    for (let j = 0; j < 33; j++) for (let c = 0; c < 2; c++) f.push(std(a.map((fr) => fr[j][c])));
    // full trajectory of the 12 motion joints
    for (let t = 0; t < FRAMES; t++) for (const j of MOTION_JOINTS) for (let c = 0; c < 2; c++) f.push(a[t][j][c]);
    // velocity statistics
    const diffs = (j, c) => { const d = []; for (let t = 1; t < FRAMES; t++) d.push(a[t][j][c] - a[t - 1][j][c]); return d; };
    for (const j of MOTION_JOINTS) for (let c = 0; c < 2; c++) f.push(mean(diffs(j, c).map(Math.abs)));
    for (const j of MOTION_JOINTS) for (let c = 0; c < 2; c++) f.push(std(diffs(j, c)));
    return Float32Array.from(f);
  }

  /** Average of every tree's leaf class fractions (= sklearn predict_proba). */
  function predictProba(model, x) {
    const nClass = model.classes.length;
    const sum = new Array(nClass).fill(0);
    for (const tree of model.trees) {
      let node = 0;
      while (tree.l[node] !== -1) node = x[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
      const v = tree.v[node];
      for (let k = 0; k < nClass; k++) sum[k] += v[k];
    }
    return sum.map((s) => s / model.trees.length);
  }

  const api = { MOTION_JOINTS, FRAMES, normalizeLandmarks, resample, sequenceFeatures, predictProba };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.DanceCore = api;
})(typeof window !== "undefined" ? window : globalThis);
