/* 後台：總覽統計、照片標註 (0~4 級)、模型訓練 */
$(function () {
  const IND = {spots: "斑點", wrinkles: "皺紋", texture: "紋理", pores: "毛孔", brown_spots: "棕色斑",
               red_areas: "紅色區", uv_spots: "紫外線色斑", porphyrins: "紫質"};
  const LV = ["無", "輕微", "中等", "明顯", "嚴重"];
  let items = [], skip = 0, current = null;

  // ---------------- 總覽
  function loadStats() {
    SL.api("GET", "/api/admin/stats").then(s => {
      const card = (icon, label, n) => `<div class="col-6 col-md-3"><div class="sl-card p-3">
        <div class="small text-body-secondary"><i class="bi bi-${icon}"></i> ${label}</div><div class="fs-3 fw-bold tabular">${n.toLocaleString()}</div></div></div>`;
      $("#stat-cards").html(card("people", "會員", s.users) + card("clipboard-data", "分析次數", s.analyses) +
                            card("images", "訓練照片", s.images) + card("tags", "資料來源", Object.keys(s.sources).length));
      const max = Math.max(1, ...Object.values(s.labeled));
      $("#labeled").html(Object.entries(s.labeled).map(([k, n]) => `
        <div class="d-flex align-items-center gap-2 mb-1 small">
          <span style="width:6rem">${IND[k]}</span>
          <div class="sl-bar flex-grow-1"><div style="width:${100 * n / max}%;background:var(--sl-primary)"></div></div>
          <span class="tabular text-end" style="width:4rem">${n}</span>
        </div>`).join(""));
    }).catch(err => SL.toast(err.message, "danger"));
  }

  // ---------------- 標註
  function loadImages(reset) {
    if (reset) { skip = 0; items = []; $("#img-list").empty(); }
    SL.api("GET", `/api/admin/images?limit=50&skip=${skip}&unlabeled=${$("#only-unlabeled").is(":checked")}`).then(r => {
      items = items.concat(r.items); skip += r.items.length;
      $("#img-total").text(`共 ${r.total.toLocaleString()} 張`);
      $("#img-list").append(r.items.map(it => `
        <button class="list-group-item list-group-item-action d-flex justify-content-between" data-id="${it.id}">
          <span>${SL.esc(it.source || "upload")} · ${it.id.slice(-6)}</span>
          <span class="badge rounded-pill ${Object.keys(it.annotations).length ? "text-bg-success" : "text-bg-light border"}">${Object.keys(it.annotations).length}</span>
        </button>`).join(""));
      $("#more").toggle(skip < r.total);
    }).catch(err => SL.toast(err.message, "danger"));
  }

  function showImage(id) {
    current = items.find(i => i.id === id);
    $("#img-list .list-group-item").removeClass("active").filter(`[data-id="${id}"]`).addClass("active");
    $("#label-panel").html(`
      <div class="row g-3">
        <div class="col-md-5"><img src="/api/admin/images/${id}/file" class="img-fluid rounded" alt="待標註照片">
          <div class="small text-body-secondary mt-1">${SL.esc(current.source)} · ${current.split}</div></div>
        <div class="col-md-7">
          <p class="small text-body-secondary">每個指標點選嚴重度 (0 = 無，4 = 嚴重)，沒把握的可以不點。</p>
          ${Object.entries(IND).map(([k, name]) => {
            const cur = current.annotations[k]?.severity;
            return `<div class="d-flex align-items-center gap-2 mb-2 sl-label-row" data-ind="${k}">
              <span class="small" style="width:5.5rem">${name}</span>
              <div class="btn-group btn-group-sm" role="group" aria-label="${name}">${LV.map((l, v) =>
                `<button class="btn ${cur === v ? "btn-primary" : "btn-outline-secondary"}" data-v="${v}" title="${l}">${v}</button>`).join("")}</div>
              ${current.annotations[k] ? `<span class="small text-body-secondary">目前 ${Math.round(current.annotations[k].score)} 分</span>` : ""}
            </div>`; }).join("")}
          <div class="d-flex gap-2 mt-3">
            <button class="btn btn-primary" id="save">儲存並下一張</button>
            <button class="btn btn-outline-danger ms-auto" id="del-img">刪除這張照片</button>
          </div>
        </div>
      </div>`);
  }

  $("#img-list").on("click", ".list-group-item", function () { showImage($(this).data("id")); });
  $("#only-unlabeled").on("change", () => loadImages(true));
  $("#more").on("click", () => loadImages(false));
  $("#label-panel").on("click", ".sl-label-row .btn", function () {
    const $b = $(this), on = $b.hasClass("btn-primary");
    $b.siblings().removeClass("btn-primary").addClass("btn-outline-secondary");
    $b.toggleClass("btn-primary", !on).toggleClass("btn-outline-secondary", on);
  });
  $("#label-panel").on("click", "#save", function () {
    const picks = $("#label-panel .sl-label-row").map((_, el) => {
      const $sel = $(el).find(".btn-primary");
      return $sel.length ? {indicator: $(el).data("ind"), severity: +$sel.data("v")} : null;
    }).get();
    if (!picks.length) { SL.toast("還沒有點選任何指標", "warning"); return; }
    SL.api("POST", "/api/admin/annotations", {image_id: current.id, items: picks}).then(() => {
      picks.forEach(p => current.annotations[p.indicator] = {severity: p.severity, score: p.severity * 25});
      $(`#img-list [data-id="${current.id}"] .badge`).text(Object.keys(current.annotations).length).attr("class", "badge rounded-pill text-bg-success");
      SL.toast("已儲存");
      const next = items[items.findIndex(i => i.id === current.id) + 1];
      if (next) showImage(next.id);
    }).catch(err => SL.toast(err.message, "danger"));
  });
  $("#label-panel").on("click", "#del-img", function () {
    const $b = $(this);
    if (!$b.hasClass("confirming")) { $b.addClass("confirming").text("再按一次確定刪除"); return; }
    SL.api("DELETE", `/api/admin/images/${current.id}`).then(() => {
      $(`#img-list [data-id="${current.id}"]`).remove();
      items = items.filter(i => i.id !== current.id);
      $("#label-panel").html('<p class="text-body-secondary m-0">已刪除。從左邊選下一張。</p>');
      SL.toast("已刪除");
    }).catch(err => SL.toast(err.message, "danger"));
  });

  // ---------------- 模型
  function metricRows(metrics) {
    return Object.entries(metrics).map(([k, m]) => `<tr>
      <td>${IND[k]}</td><td class="tabular">${m.n}</td>
      <td>${m.deployed ? '<span class="badge text-bg-success">已上線</span>' : `<span class="badge text-bg-light border">${SL.esc(m.status || "未上線")}</span>`}</td>
      <td class="tabular">${m.cv_mae ?? "-"}</td><td class="tabular">${m.baseline_mae ?? "-"}</td>
      <td class="tabular">${m.improvement != null ? m.improvement + "%" : "-"}</td>
      <td class="tabular small text-body-secondary">${m.improvement_range ? m.improvement_range.join(" ~ ") + "%" : "-"}</td>
      <td class="tabular">${m.pearson_r ?? "-"}</td></tr>`).join("");
  }
  const metricHead = `<thead><tr><th>指標</th><th>樣本</th><th>狀態</th><th>模型誤差</th><th>基準誤差</th><th>改善 (平均)</th><th>10 次分組範圍</th><th>相關係數 r</th></tr></thead>`;

  function loadVersions() {
    SL.api("GET", "/api/admin/models").then(vs => {
      $("#versions").html(`<thead><tr><th>版本</th><th>時間</th><th>上線</th><th>最多樣本</th></tr></thead><tbody>${vs.map(v => `
        <tr><td>${SL.esc(v.name)}</td><td>${SL.fmtDate(v.created_at)}</td>
        <td>${v.deployed ? '<i class="bi bi-check-circle-fill text-success"></i>' : "-"}</td><td class="tabular">${v.n_samples}</td></tr>`).join("")}</tbody>`);
    });
  }

  $("#train").on("click", function () {
    $(this).prop("disabled", true).html('<span class="spinner-border spinner-border-sm"></span> 訓練中…');
    SL.api("POST", "/api/admin/train").then(r => {
      $("#train-result").html(`<div class="alert ${r.trained ? "alert-success" : "alert-warning"} small">${SL.esc(r.message)}${r.version ? ` (${SL.esc(r.version)})` : ""}</div>
        ${r.metrics ? `<div class="table-responsive"><table class="table table-sm small">${metricHead}<tbody>${metricRows(r.metrics)}</tbody></table></div>` : ""}`);
      loadVersions(); loadStats();
    }).catch(err => SL.toast(err.message, "danger"))
      .always(() => $("#train").prop("disabled", false).html('<i class="bi bi-cpu"></i> 開始訓練'));
  });

  loadStats(); loadImages(true); loadVersions();
});
