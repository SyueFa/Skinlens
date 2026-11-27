/* 網頁拍照流程：開相機 → 對正 (即時偵測頭部角度，臉正對鏡頭、在框內才開始)
 *   → 睜眼看鏡頭 5 秒 (背景擷取影格，後端用眼白校色) → 閉眼 3 秒 → 確認還是正臉才拍
 *   → (選用) 眼睛繼續閉著，轉向左邊、右邊，轉夠了才各拍一張側臉。
 * 閉眼時看不到螢幕，所以倒數、姿勢提示與完成都用嗶聲 + 語音。
 * 臉部偵測 (facepose.js) 載入失敗時退回只按秒數倒數，後端仍會檢查角度並提醒。
 * 用法：SLCapture.start({video, canvas, guide, onStatus(title, count), onDone({still, frames, sides}), onError(msg), withSides})
 *       SLCapture.stop()
 */
window.SLCapture = (function () {
  const OPEN_SEC = 5, CLOSE_SEC = 3, SIDE_SEC = 3;
  const FRAME_EVERY_MS = 600, FRAME_START_MS = 1000;   // 前 1 秒讓自動曝光穩定，之後每 0.6 秒抓一張
  const FRAME_MAX_SIDE = 1280, STILL_MAX_SIDE = 2400;   // 影格只用來看眼白，不必太大
  // 姿勢引導：正臉容許的角度 (度)。後端超過 15 度才警告，這裡抓嚴一點；一般自拍上下約 −5 ~ −11 度
  const POSE_OK = {yaw: 10, pitch: 12};
  const SIDE_MIN_YAW = 25;                  // 側臉至少轉這麼多度才拍 (目標約 45 度)
  // 引導框 (CSS .sl-cam-guide：中心 50%/45%、高 64%) 對應的「額頭特徵點 → 下巴」範圍：
  // 框也包含頭髮，所以特徵點的中心比框的中心低一點、高度約框的 0.8
  const GUIDE = {cx: 0.5, cy: 0.48, h: 0.64};
  const ALIGN_HOLD_MS = 800, ALIGN_MAX_MS = 30000;   // 對正後要穩定多久才開始；最多等多久
  const SHOT_WAIT_MS = 4000, SIDE_WAIT_MS = 7000;    // 閉眼拍照前 / 側臉等姿勢的上限
  const VOICE_GAP_MS = 2500;                // 語音提示的最短間隔
  let stream = null, timers = [], audio = null, cancelled = false;
  let poseOn = false, cur = {pose: null, ok: false, hint: ""}, mode = "front", raf = 0, lastVoice = 0, lastHint = "";

  function beep(freq = 880, ms = 150) {
    try {
      audio = audio || new (window.AudioContext || window.webkitAudioContext)();
      const o = audio.createOscillator(), g = audio.createGain();
      o.frequency.value = freq; g.gain.value = 0.15;
      o.connect(g); g.connect(audio.destination);
      o.start(); o.stop(audio.currentTime + ms / 1000);
    } catch (e) { /* 沒有聲音也能拍，只是少了提示 */ }
  }

  function say(text) {
    try {
      if (!window.speechSynthesis) return;
      speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "zh-TW"; u.rate = 1.05;
      speechSynthesis.speak(u);
    } catch (e) { /* 不支援語音就只有嗶聲 */ }
  }

  const wait = ms => new Promise(res => timers.push(setTimeout(res, ms)));

  function grab(video, maxSide, quality) {
    const w = video.videoWidth, h = video.videoHeight, s = Math.min(1, maxSide / Math.max(w, h));
    const c = document.createElement("canvas");
    c.width = Math.round(w * s); c.height = Math.round(h * s);
    c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);   // 不鏡像：存真實方向
    return new Promise(res => c.toBlob(res, "image/jpeg", quality));
  }

  async function lockWhiteBalance(track) {
    // Android Chrome 等支援的裝置鎖住白平衡，睜眼影格和閉眼照片的顏色才一致；
    // 不支援 (例如 iPhone Safari) 也沒關係，後端會用膚色對齊兩者
    try {
      const cap = track.getCapabilities ? track.getCapabilities() : {};
      if ((cap.whiteBalanceMode || []).includes("manual")) {
        await track.applyConstraints({advanced: [{whiteBalanceMode: "manual"}]});
        return true;
      }
    } catch (e) { /* 忽略 */ }
    return false;
  }

  // ---------------------------------------------------------------- 姿勢引導
  // 正臉：臉要在引導框中間、大小合適、左右轉與上下仰俯都在容許範圍內。回傳 {ok, hint}
  function judgeFront(pose, video, box) {
    if (!pose) return {ok: false, hint: "把臉放進框裡"};
    const [vx, vy] = SLPose.toView(box, video, pose.box.cx, pose.box.cy);
    const [, y0] = SLPose.toView(box, video, pose.box.cx, pose.box.cy - pose.box.h / 2);
    const [, y1] = SLPose.toView(box, video, pose.box.cx, pose.box.cy + pose.box.h / 2);
    const rel = (y1 - y0) / GUIDE.h;
    if (rel < 0.6) return {ok: false, hint: "靠近一點"};
    if (rel > 1.2) return {ok: false, hint: "離遠一點"};
    // 預覽是鏡像的 (像照鏡子)：畫面上偏左 = 本人偏左，方向直接照本人的感覺說
    if (vx < GUIDE.cx - 0.12) return {ok: false, hint: "往右移一點"};
    if (vx > GUIDE.cx + 0.12) return {ok: false, hint: "往左移一點"};
    if (vy < GUIDE.cy - 0.13) return {ok: false, hint: "臉往下移一點"};
    if (vy > GUIDE.cy + 0.13) return {ok: false, hint: "臉往上移一點"};
    // yaw 負 = 鼻子朝影像左邊 = 本人往自己的右邊轉 → 請他往左轉回來
    if (pose.yaw < -POSE_OK.yaw) return {ok: false, hint: "臉往左轉一點"};
    if (pose.yaw > POSE_OK.yaw) return {ok: false, hint: "臉往右轉一點"};
    if (pose.pitch < -POSE_OK.pitch) return {ok: false, hint: "手機舉高到眼睛的高度"};
    if (pose.pitch > POSE_OK.pitch) return {ok: false, hint: "手機放低到眼睛的高度"};
    return {ok: true, hint: ""};
  }

  // 側臉：往本人的左邊轉 = 鼻子朝影像右邊 = yaw 為正
  function judgeSide(pose, dir) {
    if (!pose) return {ok: false, hint: "轉回來一點，鏡頭看不到臉"};
    const y = dir === "左" ? pose.yaw : -pose.yaw;
    if (y < SIDE_MIN_YAW) return {ok: false, hint: `再往${dir}轉一點`};
    return {ok: true, hint: ""};
  }

  // 每一幀偵測姿勢、畫特徵點、更新引導框顏色；結果放在 cur 給流程判斷
  function track(video, canvas, guide, box) {
    let last = 0;
    const loop = now => {
      if (cancelled) return;
      raf = requestAnimationFrame(loop);
      if (now - last < 66) return;   // 約每秒 15 次，手機也跑得動
      last = now;
      const pose = SLPose.detect(video);
      const j = mode === "front" ? judgeFront(pose, video, box) : judgeSide(pose, mode);
      cur = {pose, ...j};
      if (canvas) SLPose.draw(canvas, video, pose, j.ok);
      if (guide) {
        guide.classList.toggle("ok", j.ok);
        guide.classList.toggle("bad", !j.ok && !!pose && mode === "front");
        guide.hidden = mode !== "front";
      }
    };
    raf = requestAnimationFrame(loop);
  }

  // 提示改變、或間隔夠久才念出來 (閉眼時唯一的回饋)
  function voiceHint(hint) {
    const now = performance.now();
    if (!hint || (hint === lastHint && now - lastVoice < VOICE_GAP_MS * 2) || now - lastVoice < VOICE_GAP_MS) return;
    say(hint); lastVoice = now; lastHint = hint;
  }

  // 等到姿勢 ok 並維持 holdMs；逾時回傳 false。onTick 每 100ms 呼叫一次 (更新畫面 / 語音)
  async function waitPose(holdMs, maxMs, onTick) {
    const t0 = performance.now();
    let since = null;
    while (!cancelled) {
      const now = performance.now();
      if (cur.ok) { since = since ?? now; if (now - since >= holdMs) return true; } else since = null;
      if (now - t0 > maxMs) return false;
      onTick && onTick(now - t0);
      await wait(100);
    }
    return false;
  }

  function stop() {
    cancelled = true;
    cancelAnimationFrame(raf);
    timers.forEach(t => { clearTimeout(t); clearInterval(t); }); timers = [];
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    try { window.speechSynthesis && speechSynthesis.cancel(); } catch (e) { /* 忽略 */ }
  }

  async function start({video, canvas, guide, onStatus, onDone, onError, withSides = true}) {
    cancelled = false; poseOn = false; mode = "front"; cur = {pose: null, ok: false, hint: ""};
    lastVoice = 0; lastHint = "";
    if (guide) { guide.classList.remove("ok", "bad"); guide.hidden = false; }
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    const poseLoad = window.SLPose ? SLPose.load() : Promise.resolve(false);   // 和開相機同時載入
    beep(660, 1);   // 在按鈕事件當下啟動音效 (瀏覽器規定要有使用者操作，等相機權限之後再啟動 Safari 會擋)
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      onError("這個瀏覽器無法使用相機 (需要 https 或 localhost)，請改用上傳照片"); return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode: "user", width: {ideal: 1920}, height: {ideal: 1440}}, audio: false});
    } catch (e) {
      onError(e.name === "NotAllowedError" ? "沒有相機權限，請在瀏覽器允許使用相機後再試" : "無法開啟相機：" + e.message);
      return;
    }
    if (cancelled) { stop(); return; }   // 等權限時就按了取消：相機打開後立刻關掉
    video.srcObject = stream;
    try {
      await video.play();
    } catch (e) {
      stop(); onError("相機畫面無法播放：" + e.message); return;
    }

    // 0. 對正：臉在框內、正對鏡頭並穩定一下才開始 (偵測載入失敗就跳過)
    onStatus("準備臉部偵測…", "");
    poseOn = await Promise.race([poseLoad, wait(10000).then(() => false)]);
    if (cancelled) return;
    const box = video.parentElement;
    if (poseOn) {
      track(video, canvas, guide, box);
      say("請把臉放進框裡，正對鏡頭");
      lastVoice = performance.now();
      await waitPose(ALIGN_HOLD_MS, ALIGN_MAX_MS, () => {
        onStatus(cur.ok ? "很好，保持不動" : cur.hint, "");
        if (!cur.ok) voiceHint(cur.hint);
      });
      if (cancelled) return;
      beep(1320, 120);
    }

    // 1. 睜眼看鏡頭：背景擷取影格
    onStatus("請看著鏡頭，眼睛張開", OPEN_SEC);
    say("請看著鏡頭，眼睛張開");
    const frames = [];
    let grabbing = false;
    timers.push(setTimeout(() => {
      lockWhiteBalance(stream?.getVideoTracks()[0]);
      const iv = setInterval(async () => {
        if (grabbing || cancelled) return;
        grabbing = true;
        const b = await grab(video, FRAME_MAX_SIDE, 0.9);
        if (b) frames.push(b);
        grabbing = false;
      }, FRAME_EVERY_MS);
      timers.push(iv);
    }, FRAME_START_MS));
    for (let s = OPEN_SEC * 10; s > 0; s--) {   // 每 0.1 秒更新：跑掉時把提示換成姿勢提示
      const title = poseOn && !cur.ok && cur.hint ? cur.hint : "請看著鏡頭，眼睛張開";
      onStatus(title, Math.ceil(s / 10));
      await wait(100);
      if (cancelled) return;
    }
    timers.forEach(t => { clearTimeout(t); clearInterval(t); }); timers = [];

    // 2. 閉眼倒數 (看不到螢幕 → 每秒嗶一聲)
    say("請閉上眼睛");
    for (let s = CLOSE_SEC; s > 0; s--) {
      onStatus("請閉上眼睛", s);
      await wait(1000);
      if (cancelled) return;
      beep(880, 120);
    }

    // 3. 拍正臉：閉眼時頭常會不自覺動，跑掉就用語音提示，等回正再拍 (最多等幾秒)
    if (poseOn && !cur.ok) {
      lastVoice = 0;
      await waitPose(300, SHOT_WAIT_MS, () => { onStatus(cur.hint || "保持不動", ""); voiceHint(cur.hint); });
      if (cancelled) return;
    }
    onStatus("拍照中…", "");
    const still = await grab(video, STILL_MAX_SIDE, 0.95);
    const shots = [];

    // 4. (選用) 側臉：眼睛繼續閉著，轉向左、右各拍一張 (看魚尾紋與外側臉頰，像 VISIA 的側面照)
    //    有臉部偵測：轉夠角度才拍；沒有：照秒數倒數
    if (withSides) {
      for (const [dir, sec] of [["左", SIDE_SEC], ["右", SIDE_SEC + 1]]) {   // 從左轉到右要多一秒
        beep(1320, 150);
        say(`眼睛繼續閉著，請把臉慢慢轉向${dir}邊`);
        if (poseOn) {
          mode = dir;
          lastVoice = performance.now() + 1500;   // 先讓開頭的指示念完、給時間轉頭
          await wait(1200);
          if (cancelled) return;
          await waitPose(400, SIDE_WAIT_MS, () => {
            onStatus(cur.ok ? "很好，保持不動" : `眼睛閉著，臉轉向${dir}邊 (約 45 度)`, "");
            if (!cur.ok) voiceHint(cur.hint);
          });
          if (cancelled) return;
          beep(880, 120);
        } else {
          for (let s = sec; s > 0; s--) {
            onStatus(`眼睛閉著，臉轉向${dir}邊 (約 45 度)`, s);
            await wait(1000);
            if (cancelled) return;
            beep(880, 120);
          }
        }
        onStatus("拍照中…", "");
        shots.push(await grab(video, STILL_MAX_SIDE, 0.95));
      }
    }

    beep(1320, 400);
    say("好了，可以睜開眼睛");
    stop();
    onDone({still, frames, sides: shots});
  }

  // 網頁相機拍的照片沒有 EXIF，從瀏覽器資訊推測裝置 (Android 會帶型號代碼，iPhone 只知道是 iPhone)，
  // 讓歷史紀錄能提醒「兩次用不同手機拍」
  function deviceLabel() {
    const ua = navigator.userAgent;
    const android = ua.match(/Android [\d.]+; ([^;)]+?)(?: Build|\))/);
    if (android && android[1] !== "K") return android[1].trim();   // Chrome 精簡 UA 時型號會變成 "K"
    if (/iPhone/.test(ua)) return "iPhone";
    if (/iPad/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1)) return "iPad";
    if (/Android/.test(ua)) return "Android";
    return /Mac/.test(ua) ? "Mac 網路攝影機" : /Windows/.test(ua) ? "Windows 網路攝影機" : null;
  }

  return {start, stop, deviceLabel};
})();
