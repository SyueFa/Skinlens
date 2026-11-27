/* 網頁即時頭部姿勢偵測 (MediaPipe FaceLandmarker 網頁版)，給拍照流程 (capture.js) 引導使用者正對鏡頭。
 * 角度算法和後端 analysis/preprocess.head_pose 相同：
 *   yaw   左右轉 (度)：負 = 鼻子朝影像左邊 = 本人往自己的右邊轉 (相機影像不鏡像)
 *   pitch 上下 (度)：負 = 抬下巴 / 手機拿太低；正 = 低頭 / 手機拿太高
 * 模型與程式都放在自己的伺服器 (scripts/download_models.py 下載)，畫面不會送到外部。
 * 用法：await SLPose.load() (失敗回傳 false，拍照流程退回只按秒數倒數)
 *       SLPose.detect(video) → null (沒偵測到臉) 或 {yaw, pitch, box:{cx, cy, w, h}, pts}
 *         box、pts 是影像上的比例座標 (0–1，不鏡像)
 *       SLPose.draw(canvas, video, pose, ok) 在預覽上畫特徵點 (配合 CSS 的鏡像與 object-fit: cover)
 */
window.SLPose = (function () {
  const BASE = "/static/vendor/mediapipe";
  const MODEL = "/models/face_landmarker.task";
  let landmarker = null, loading = null, lastTs = -1;

  function load() {
    if (landmarker) return Promise.resolve(true);
    if (loading) return loading;
    loading = (async () => {
      try {
        const {FaceLandmarker, FilesetResolver} = await import(`${BASE}/vision_bundle.mjs`);
        const fileset = await FilesetResolver.forVisionTasks(`${BASE}/wasm`);
        const opts = (delegate) => ({
          baseOptions: {modelAssetPath: MODEL, delegate},
          runningMode: "VIDEO", numFaces: 1, outputFacialTransformationMatrixes: true,
        });
        try {
          landmarker = await FaceLandmarker.createFromOptions(fileset, opts("GPU"));
        } catch (e) {
          landmarker = await FaceLandmarker.createFromOptions(fileset, opts("CPU"));   // 沒有 WebGL
        }
        return true;
      } catch (e) {
        console.warn("臉部偵測載入失敗，改用秒數倒數：", e);
        loading = null;
        return false;
      }
    })();
    return loading;
  }

  // 4×4 轉換矩陣 → 角度。MediaPipe 網頁版的 data 可能是行優先或列優先：
  // 平移量 (臉離相機約 −30 ~ −80 公分，z 絕對值最大) 在哪就知道排法
  function angles(m) {
    const d = m.data, colMajor = Math.abs(d[14]) > Math.abs(d[11]);
    const R = (i, j) => colMajor ? d[j * 4 + i] : d[i * 4 + j];
    const deg = r => r * 180 / Math.PI;
    return {
      yaw: deg(Math.asin(Math.max(-1, Math.min(1, -R(2, 0))))),
      pitch: deg(Math.atan2(R(2, 1), R(2, 2))),
    };
  }

  function detect(video) {
    if (!landmarker || video.readyState < 2) return null;
    let ts = performance.now();
    if (ts <= lastTs) ts = lastTs + 1;   // VIDEO 模式的時間戳必須遞增
    lastTs = ts;
    let res;
    try { res = landmarker.detectForVideo(video, ts); } catch (e) { return null; }
    if (!res.faceLandmarks?.length || !res.facialTransformationMatrixes?.length) return null;
    const pts = res.faceLandmarks[0];
    const l = pts[234], r = pts[454], top = pts[10], chin = pts[152];
    return {
      ...angles(res.facialTransformationMatrixes[0]),
      box: {cx: (l.x + r.x) / 2, cy: (top.y + chin.y) / 2, w: Math.abs(r.x - l.x), h: Math.abs(chin.y - top.y)},
      pts,
    };
  }

  // 影像比例座標 → 預覽框的像素座標 (object-fit: cover 會裁掉一部分，CSS 又左右鏡像)
  function mapper(canvas, video) {
    const cw = canvas.width, ch = canvas.height, vw = video.videoWidth, vh = video.videoHeight;
    const s = Math.max(cw / vw, ch / vh), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
    return (x, y) => [cw - (ox + x * vw * s), oy + y * vh * s];
  }

  function draw(canvas, video, pose, ok) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(rect.width * dpr)) {
      canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!pose || !video.videoWidth) return;
    const map = mapper(canvas, video);
    ctx.fillStyle = ok ? "rgba(80, 220, 160, .75)" : "rgba(255, 255, 255, .55)";
    const r = 1.3 * dpr;
    for (let i = 0; i < pose.pts.length; i += 2) {   // 一半的點就夠有掃描感，也比較省
      const [x, y] = map(pose.pts[i].x, pose.pts[i].y);
      ctx.fillRect(x - r / 2, y - r / 2, r, r);
    }
  }

  // 影像比例座標 → 預覽框的比例座標 (0–1)，用來和引導框比對位置與大小
  function toView(container, video, x, y) {
    const cw = container.clientWidth, ch = container.clientHeight, vw = video.videoWidth, vh = video.videoHeight;
    const s = Math.max(cw / vw, ch / vh), ox = (cw - vw * s) / 2, oy = (ch - vh * s) / 2;
    return [1 - (ox + x * vw * s) / cw, (oy + y * vh * s) / ch];
  }

  return {load, detect, draw, toView, angles, ready: () => !!landmarker};
})();
