/* 網頁拍照流程：開相機 → 睜眼看鏡頭 5 秒 (背景擷取影格，後端用眼白校色) → 閉眼 3 秒 → 拍正臉
 *   → (選用) 眼睛繼續閉著，轉向左邊、右邊各拍一張側臉。
 * 閉眼時看不到螢幕，所以倒數與完成都用嗶聲 + 語音提示。
 * 用法：SLCapture.start({video, onStatus(title, count), onDone({still, frames, sides}), onError(msg), withSides})
 *       SLCapture.stop()
 */
window.SLCapture = (function () {
  const OPEN_SEC = 5, CLOSE_SEC = 3, SIDE_SEC = 3;
  const FRAME_EVERY_MS = 600, FRAME_START_MS = 1000;   // 前 1 秒讓自動曝光穩定，之後每 0.6 秒抓一張
  const FRAME_MAX_SIDE = 1280, STILL_MAX_SIDE = 2400;   // 影格只用來看眼白，不必太大
  let stream = null, timers = [], audio = null, cancelled = false;

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

  function stop() {
    cancelled = true;
    timers.forEach(t => { clearTimeout(t); clearInterval(t); }); timers = [];
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    try { window.speechSynthesis && speechSynthesis.cancel(); } catch (e) { /* 忽略 */ }
  }

  async function start({video, onStatus, onDone, onError, withSides = true}) {
    cancelled = false;
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
    for (let s = OPEN_SEC; s > 0; s--) {
      onStatus("請看著鏡頭，眼睛張開", s);
      await wait(1000);
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

    // 3. 拍正臉
    onStatus("拍照中…", "");
    const still = await grab(video, STILL_MAX_SIDE, 0.95);
    const shots = [];

    // 4. (選用) 側臉：眼睛繼續閉著，轉向左、右各拍一張 (看魚尾紋與外側臉頰，像 VISIA 的側面照)
    if (withSides) {
      for (const [dir, sec] of [["左", SIDE_SEC], ["右", SIDE_SEC + 1]]) {   // 從左轉到右要多一秒
        beep(1320, 150);
        say(`眼睛繼續閉著，請把臉慢慢轉向${dir}邊`);
        for (let s = sec; s > 0; s--) {
          onStatus(`眼睛閉著，臉轉向${dir}邊 (約 45 度)`, s);
          await wait(1000);
          if (cancelled) return;
          beep(880, 120);
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
