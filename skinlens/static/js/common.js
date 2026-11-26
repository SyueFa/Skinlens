/* 共用工具：API 呼叫、提示訊息、登出、報告畫面 (首頁與歷史詳情共用) */
const SL = (() => {
  const LEVEL_COLORS = ["var(--sl-l0)", "var(--sl-l1)", "var(--sl-l2)", "var(--sl-l3)", "var(--sl-l4)"];
  const REGIONS = {forehead: "額頭", glabella: "眉間", crow_l: "左側眼尾", crow_r: "右側眼尾", under_eye: "眼下",
                   nose: "鼻子", cheek_l: "左臉頰", cheek_r: "右臉頰", nasolabial: "法令紋", chin: "下巴"};
  const LIGHTING = {standard: "一般光源", cross_polarized: "交叉偏光", uv: "UV 光源"};

  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

  /** 呼叫 JSON API；失敗時丟出含中文訊息的 Error */
  function api(method, url, body) {
    return $.ajax({url, method, contentType: body ? "application/json" : undefined,
                   data: body ? JSON.stringify(body) : undefined, dataType: "json"})
      .catch(xhr => { throw new Error(xhr.responseJSON?.error || `連線失敗 (${xhr.status || "網路錯誤"})`); });
  }

  function toast(msg, type = "success") {
    const icon = type === "success" ? "check-circle" : type === "danger" ? "exclamation-triangle" : "info-circle";
    const $t = $(`<div class="toast align-items-center text-bg-${type} border-0" role="status">
      <div class="d-flex"><div class="toast-body"><i class="bi bi-${icon}"></i> ${esc(msg)}</div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="關閉"></button></div></div>`);
    $("#toasts").append($t);
    new bootstrap.Toast($t[0], {delay: 3500}).show();
    $t.on("hidden.bs.toast", () => $t.remove());
  }

  const fmtDate = iso => new Date(iso).toLocaleString("zh-TW", {year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"});
  const levelBadge = ind => `<span class="sl-level sl-l${ind.level}">${esc(ind.level_name)}</span>`;

  /**
   * 畫出完整報告。data = /api/analyze 或 /api/history/<id> 的回傳。
   * opts.faceUrl：歷史紀錄沒有 base64 圖時，用這張照片
   */
  function renderReport($root, data, opts = {}) {
    const ringColor = LEVEL_COLORS[Math.min(4, Math.floor((100 - data.overall_score) / 20))];
    const q = data.quality || {};
    const warnings = [...(q.warnings || []), ...(data.unavailable || [])];
    const hasImages = !!data.original;
    const inds = data.indicators;
    $root.off("click");   // 重新分析時清掉上一次的事件，避免重複綁定

    const views = hasImages ? [
      {name: "校色後", img: data.preprocessed, cap: "白平衡與對比校正後"},
      {name: "分區", img: data.regions_preview, cap: "臉部 10 個分區與皮膚遮罩"},
      ...inds.map(i => ({key: i.key, name: i.name, img: i.overlay, cap: `${i.name}：標示處為偵測到的位置`}))
    ] : [];

    $root.html(`
      <div class="sl-card p-4 mb-3">
        <div class="d-flex flex-wrap gap-4 align-items-center">
          <div class="sl-score-ring" style="--v:${data.overall_score};--c:${ringColor}">
            <div><div><strong class="tabular">${Math.round(data.overall_score)}</strong><div class="small text-body-secondary">綜合分數</div></div></div>
          </div>
          <div class="flex-grow-1" style="min-width:220px">
            <div class="sl-eyebrow mb-1">分析結果${data.created_at ? " · " + fmtDate(data.created_at) : ""}</div>
            <h2 class="h4 fw-bold mb-2">${esc(data.summary)}</h2>
            <div class="small text-body-secondary">
              ${esc(LIGHTING[data.lighting] || data.lighting)} · 臉部偵測：${esc(q.detector === "mediapipe" ? "MediaPipe 478 點" : "OpenCV")}
              · 清晰度 ${q.blur ?? "-"} · 亮度 ${q.brightness ?? "-"}
            </div>
            ${data.saved_to_history ? `<div class="small mt-2"><i class="bi bi-check2-circle text-success"></i> 已存到 <a href="/history">我的紀錄</a></div>` : ""}
            ${data.saved_to_history === false && !window.CURRENT_USER ? `<div class="small mt-2 text-body-secondary"><i class="bi bi-info-circle"></i> <a href="/register">註冊會員</a>就能保存每次結果、追蹤保養前後變化</div>` : ""}
          </div>
        </div>
        ${warnings.map(w => `<div class="alert alert-warning py-2 px-3 mt-3 mb-0 small"><i class="bi bi-exclamation-triangle"></i> ${esc(w)}</div>`).join("")}
      </div>

      <div class="row g-3 mb-3">
        <div class="col-lg-7">
          <div class="sl-card p-3 h-100">
            ${hasImages ? `
              <div class="sl-view-tabs d-flex flex-wrap gap-1 mb-3">${views.map((v, i) =>
                `<button class="btn btn-sm btn-outline-secondary" data-view="${i}">${esc(v.name)}</button>`).join("")}</div>
              <div class="row g-2 sl-viewer">
                <figure class="col-6 m-0"><img src="${data.original}" alt="原始照片 (已轉正裁切)"><figcaption>原始 (已轉正裁切)</figcaption></figure>
                <figure class="col-6 m-0"><img id="view-img" alt=""><figcaption id="view-cap"></figcaption></figure>
              </div>` : opts.faceUrl ? `
              <div class="sl-viewer text-center"><img src="${opts.faceUrl}" alt="當次照片" style="max-width:360px;margin:auto"><figcaption>當次照片 (前處理後)</figcaption></div>` :
              `<p class="text-body-secondary small m-0">這筆紀錄沒有保存照片。</p>`}
          </div>
        </div>
        <div class="col-lg-5">
          <div class="sl-card p-3 h-100">
            <h3 class="h6 fw-bold mb-1">各項指標</h3>
            <p class="small text-body-secondary mb-2">分數越高越好 (100 = 沒有這個問題)，點一下看說明</p>
            <div class="sl-ind-grid">${inds.map(i => `
              <button class="sl-ind-tile" data-key="${i.key}">
                <div class="d-flex justify-content-between align-items-start"><span class="fw-bold">${esc(i.name)}</span>${levelBadge(i)}</div>
                <div class="num tabular">${Math.round(i.score)}</div>
                <div class="sl-bar"><div style="width:${i.score}%;background:${LEVEL_COLORS[i.level]}"></div></div>
              </button>`).join("")}</div>
          </div>
        </div>
      </div>

      <div class="sl-card p-4 mb-3">
        <h3 class="h5 fw-bold mb-3">各指標代表什麼、怎麼改善</h3>
        <div class="accordion" id="ind-acc">${inds.map((i, idx) => `
          <div class="accordion-item" id="acc-${i.key}">
            <h4 class="accordion-header">
              <button class="accordion-button ${idx < 2 ? "" : "collapsed"}" type="button" data-bs-toggle="collapse" data-bs-target="#c-${i.key}">
                <span class="me-2 fw-bold">${esc(i.name)}</span><small class="text-body-secondary me-2">${esc(i.en)}</small>
                ${levelBadge(i)}<span class="ms-auto me-3 tabular small">${Math.round(i.score)} 分</span>
              </button>
            </h4>
            <div id="c-${i.key}" class="accordion-collapse collapse ${idx < 2 ? "show" : ""}">
              <div class="accordion-body sl-advice">
                <div class="small text-body-secondary mb-2">
                  ${i.percentile != null ? `同齡百分位：贏過 ${i.percentile}% 的同齡者 · ` : ""}
                  ${i.count ? `偵測到 ${i.count} 處 · ` : ""}
                  ${i.worst_regions?.length ? `較明顯的位置：${i.worst_regions.map(r => REGIONS[r] || r).join("、")} · ` : ""}
                  ${i.source === "model" ? `<span class="sl-source-model"><i class="bi bi-patch-check"></i> 已用 AI 模型校正</span>` : "演算法估計值"}
                </div>
                <p class="mb-1">${esc(i.what)}</p>
                <h6>可能原因</h6><ul>${i.causes.map(c => `<li>${esc(c)}</li>`).join("")}</ul>
                <h6>建議怎麼做 (${esc(i.level_name)})</h6><ul>${i.advice.map(c => `<li>${esc(c)}</li>`).join("")}</ul>
                ${i.doctor ? `<div class="sl-doctor mt-2"><i class="bi bi-hospital"></i> ${esc(i.doctor)}</div>` : ""}
              </div>
            </div>
          </div>`).join("")}</div>
      </div>

      <div class="sl-card p-4">
        <h3 class="h6 fw-bold">共同保養原則</h3>
        <ul class="mb-3">${(data.general_tips || []).map(t => `<li>${esc(t)}</li>`).join("")}</ul>
        <p class="small text-body-secondary mb-0"><i class="bi bi-info-circle"></i> ${esc(data.disclaimer)}</p>
      </div>`);

    if (hasImages) {
      const show = i => {
        $root.find("#view-img").attr({src: views[i].img, alt: views[i].cap});
        $root.find("#view-cap").text(views[i].cap);
        $root.find("[data-view]").removeClass("active btn-secondary").addClass("btn-outline-secondary")
             .filter(`[data-view=${i}]`).addClass("active btn-secondary").removeClass("btn-outline-secondary");
        $root.find(".sl-ind-tile").removeClass("active").filter(`[data-key="${views[i].key}"]`).addClass("active");
      };
      $root.on("click", "[data-view]", e => show(+$(e.currentTarget).data("view")));
      show(2);   // 預設顯示最需要注意的指標
    }
    $root.on("click", ".sl-ind-tile", e => {
      const key = $(e.currentTarget).data("key");
      if (hasImages) show_(key);
      const el = document.getElementById("c-" + key);
      bootstrap.Collapse.getOrCreateInstance(el).show();
      document.getElementById("acc-" + key).scrollIntoView({behavior: "smooth", block: "center"});
    });
    function show_(key) { const idx = views.findIndex(v => v.key === key); if (idx >= 0) $root.find(`[data-view=${idx}]`).trigger("click"); }
  }

  $(document).on("click", "#btn-logout", () => api("POST", "/api/auth/logout").then(() => location.href = "/"));

  return {api, toast, esc, fmtDate, renderReport, LEVEL_COLORS, REGIONS, LIGHTING};
})();
