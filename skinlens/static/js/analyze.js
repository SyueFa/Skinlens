/* 首頁：選照片 → 上傳 /api/analyze → 顯示報告 */
$(function () {
  let fileObj = null, timer = null;
  const prof = window.CURRENT_USER?.profile || {};
  if (prof.age) $("#age").val(prof.age);
  if (prof.gender) $("#gender").val(prof.gender);
  if (prof.skin_type) $("#skin_type").val(prof.skin_type);

  // 選 UV 光源時提醒：需要 365nm UV 燈拍攝，且尚未校正
  $("#lighting").on("change", function () { $("#uv-warn").prop("hidden", this.value !== "uv"); });

  function pick(f) {
    if (!f) return;
    if (!f.type.startsWith("image/")) { showErr("請選擇圖片檔 (JPG、PNG)"); return; }
    if (f.size > 15 * 1024 * 1024) { showErr("照片太大，請小於 15MB"); return; }
    fileObj = f;
    $("#preview").attr("src", URL.createObjectURL(f)).removeClass("d-none");
    $("#drop-empty").addClass("d-none");
    $("#drop").addClass("has-file");
    $("#go").prop("disabled", false);
    $("#err").addClass("d-none");
  }
  const showErr = msg => $("#err").text(msg).removeClass("d-none");

  $("#drop").on("click", () => $("#file").trigger("click"))
    .on("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#file").trigger("click"); } })
    .on("dragover", e => { e.preventDefault(); $("#drop").addClass("dragover"); })
    .on("dragleave drop", () => $("#drop").removeClass("dragover"))
    .on("drop", e => { e.preventDefault(); pick(e.originalEvent.dataTransfer.files[0]); });
  $("#file").on("click", e => e.stopPropagation()).on("change", e => pick(e.target.files[0]));

  // 分析進度 (後端是一次完成的，這裡依時間推進讓使用者知道在做什麼)
  function progress(start) {
    const $li = $("#progress li");
    clearInterval(timer);
    if (!start) { $("#progress").addClass("d-none"); return; }
    $("#progress").removeClass("d-none");
    let step = 0;
    const paint = () => $li.each((i, el) => {
      $(el).toggleClass("done", i < step).toggleClass("now", i === step)
           .find("i").attr("class", i < step ? "bi bi-check-circle-fill" : i === step ? "spinner-border spinner-border-sm" : "bi bi-circle");
    });
    paint();
    timer = setInterval(() => { if (step < $li.length - 1) { step++; paint(); } }, 900);
  }

  $("#go").on("click", function () {
    if (!fileObj) return;
    const fd = new FormData();
    fd.append("file", fileObj);
    ["lighting", "age", "gender", "skin_type"].forEach(k => { const v = $("#" + k).val(); if (v) fd.append(k, v); });
    fd.append("consent", $("#consent").is(":checked"));
    $("#go").prop("disabled", true).html('<span class="spinner-border spinner-border-sm"></span> 分析中…');
    $("#err").addClass("d-none");
    progress(true);
    $.ajax({url: "/api/analyze", method: "POST", data: fd, processData: false, contentType: false, dataType: "json"})
      .done(data => {
        $("#result-wrap").prop("hidden", false);
        SL.renderReport($("#result"), data);
        document.getElementById("result-wrap").scrollIntoView({behavior: "smooth"});
      })
      .fail(xhr => showErr(xhr.responseJSON?.error || "分析失敗，請換一張照片再試"))
      .always(() => {
        progress(false);
        $("#go").prop("disabled", false).html('<i class="bi bi-stars"></i> 重新分析');
      });
  });

  $("#again").on("click", () => {
    window.scrollTo({top: 0, behavior: "smooth"});
    $("#file").trigger("click");
  });
});
