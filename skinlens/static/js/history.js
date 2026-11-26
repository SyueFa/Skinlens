/* 我的紀錄：趨勢圖 (Chart.js) + 列表 + 任兩筆前後比較 */
$(function () {
  const NAMES = {overall: "綜合分數", spots: "斑點", wrinkles: "皺紋", texture: "紋理", pores: "毛孔",
                 brown_spots: "棕色斑", red_areas: "紅色區", uv_spots: "紫外線色斑", porphyrins: "紫質"};
  let rows = [], chart = null;
  const selected = new Set();

  SL.api("GET", "/api/history").then(data => {
    rows = data;
    if (!rows.length) { $("#empty").removeClass("d-none"); return; }
    $("#has-data").removeClass("d-none");
    const keys = ["overall", ...Object.keys(rows[0].scores)];
    $("#metric").html(keys.map(k => `<option value="${k}">${NAMES[k] || k}</option>`).join(""));
    drawChart("overall");
    drawList();
  }).catch(err => SL.toast(err.message, "danger"));

  $("#metric").on("change", function () { drawChart(this.value); });

  function value(r, k) { return k === "overall" ? r.overall_score : r.scores[k]; }

  function drawChart(k) {
    const pts = [...rows].reverse();
    const css = getComputedStyle(document.documentElement);
    const color = css.getPropertyValue("--sl-primary").trim() || "#22707a";
    if (chart) chart.destroy();
    chart = new Chart(document.getElementById("trend"), {
      type: "line",
      data: {labels: pts.map(r => SL.fmtDate(r.created_at)),
             datasets: [{label: NAMES[k] || k, data: pts.map(r => value(r, k)), borderColor: color,
                         backgroundColor: color + "22", fill: true, tension: .3, pointRadius: 4}]},
      options: {maintainAspectRatio: false, plugins: {legend: {display: false}},
                scales: {y: {min: 0, max: 100, title: {display: true, text: "分數 (越高越好)"}}}}
    });
  }

  function drawList() {
    $("#list").html(rows.map(r => `
      <div class="sl-hist-item d-flex gap-3 align-items-center" data-id="${r.id}">
        <input class="form-check-input flex-none" type="checkbox" value="${r.id}" aria-label="選擇這筆比較" ${selected.has(r.id) ? "checked" : ""}>
        <div class="flex-grow-1" style="min-width:0">
          <div class="small text-body-secondary">${SL.fmtDate(r.created_at)} · ${SL.LIGHTING[r.lighting] || r.lighting}</div>
          <div class="text-truncate">${SL.esc(r.summary)}</div>
        </div>
        <div class="text-end"><div class="fs-5 fw-bold tabular">${Math.round(r.overall_score)}</div></div>
        <div class="d-flex gap-1">
          <a class="btn btn-sm btn-outline-primary" href="/history/${r.id}">報告</a>
          <button class="btn btn-sm btn-outline-danger btn-del" aria-label="刪除"><i class="bi bi-trash"></i></button>
        </div>
      </div>`).join(""));
  }

  $("#list").on("change", "input[type=checkbox]", function () {
    if (this.checked) {
      if (selected.size >= 2) { const first = selected.values().next().value; selected.delete(first);
        $(`#list input[value="${first}"]`).prop("checked", false); }
      selected.add(this.value);
    } else selected.delete(this.value);
    drawCompare();
  });

  // 刪除：第一次點變成「確定？」，再點一次才刪
  $("#list").on("click", ".btn-del", function () {
    const $b = $(this), id = $b.closest("[data-id]").data("id");
    if (!$b.hasClass("confirming")) {
      $b.addClass("confirming btn-danger text-white").removeClass("btn-outline-danger").html("確定？");
      setTimeout(() => $b.removeClass("confirming btn-danger text-white").addClass("btn-outline-danger").html('<i class="bi bi-trash"></i>'), 3000);
      return;
    }
    SL.api("DELETE", `/api/history/${id}`).then(() => {
      rows = rows.filter(r => r.id !== id); selected.delete(id);
      SL.toast("已刪除");
      if (!rows.length) { $("#has-data").addClass("d-none"); $("#empty").removeClass("d-none"); return; }
      drawList(); drawChart($("#metric").val()); drawCompare();
    }).catch(err => SL.toast(err.message, "danger"));
  });

  function drawCompare() {
    if (selected.size < 2) { $("#compare-hint").removeClass("d-none"); $("#compare-body").empty(); return; }
    $("#compare-hint").addClass("d-none");
    const [a, b] = [...selected].map(id => rows.find(r => r.id === id))
                                .sort((x, y) => new Date(x.created_at) - new Date(y.created_at));
    const keys = ["overall", ...Object.keys(a.scores)];
    const line = k => {
      const d = Math.round(value(b, k) - value(a, k));
      const cls = d > 0 ? "sl-delta-up" : d < 0 ? "sl-delta-down" : "text-body-secondary";
      const icon = d > 0 ? "bi-arrow-up" : d < 0 ? "bi-arrow-down" : "bi-dash";
      return `<tr><td>${NAMES[k] || k}</td><td class="tabular text-end">${Math.round(value(a, k))}</td>
              <td class="tabular text-end">${Math.round(value(b, k))}</td>
              <td class="tabular text-end ${cls}"><i class="bi ${icon}"></i> ${Math.abs(d)}</td></tr>`;
    };
    const better = keys.filter(k => k !== "overall" && value(b, k) - value(a, k) >= 3).map(k => NAMES[k]);
    $("#compare-body").html(`
      <div class="small text-body-secondary mb-2">${SL.fmtDate(a.created_at)} → ${SL.fmtDate(b.created_at)}</div>
      <table class="table table-sm align-middle mb-2"><thead><tr><th>指標</th><th class="text-end">之前</th><th class="text-end">之後</th><th class="text-end">變化</th></tr></thead>
      <tbody>${keys.map(line).join("")}</tbody></table>
      <p class="small mb-0">${better.length ? `<i class="bi bi-emoji-smile text-success"></i> 進步：${better.join("、")}` : "各項指標變化不大；保養通常要 8~12 週才看得出差異。"}</p>
      <p class="small text-body-secondary mb-0 mt-1">拍攝光線不同會影響分數，比較時請盡量在同樣環境拍。</p>`);
  }
});
