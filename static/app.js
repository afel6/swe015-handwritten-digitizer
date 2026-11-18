"use strict";

const BOX_PX = 112;
const BRUSH_PX = 14;
const $ = (id) => document.getElementById(id);
const state = { strokes: [], current: null, stripB64: null, mode: "draw" };

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = String(text);
  if (className) node.className = className;
  return node;
}

function pngImg(b64, alt) {
  const img = document.createElement("img");
  img.alt = alt;
  if (typeof b64 === "string" && /^[A-Za-z0-9+/=]+$/.test(b64)) img.src = "data:image/png;base64," + b64;
  return img;
}

function showError(message) {
  const banner = $("error-banner");
  banner.textContent = message;
  banner.hidden = false;
}

function clearError() {
  $("error-banner").hidden = true;
  $("error-banner").textContent = "";
}

/* ---------------- drawing pad ---------------- */

function boxCount() {
  const n = parseInt($("box-count").value, 10);
  return Math.min(16, Math.max(1, Number.isFinite(n) ? n : 1));
}

function redraw() {
  const pad = $("pad");
  const ctx = pad.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.strokeStyle = "#c8c1b2";
  ctx.lineWidth = 1;
  for (let i = 1; i < boxCount(); i += 1) {
    ctx.beginPath();
    ctx.moveTo(i * BOX_PX + 0.5, 0);
    ctx.lineTo(i * BOX_PX + 0.5, pad.height);
    ctx.stroke();
  }
  drawStrokes(ctx, state.strokes.concat(state.current ? [state.current] : []));
}

function drawStrokes(ctx, strokes) {
  ctx.strokeStyle = "#111";
  ctx.lineWidth = BRUSH_PX;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const stroke of strokes) {
    ctx.beginPath();
    stroke.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    if (stroke.length === 1) ctx.lineTo(stroke[0].x + 0.1, stroke[0].y);
    ctx.stroke();
  }
}

function resizePad() {
  const pad = $("pad");
  pad.width = boxCount() * BOX_PX;
  pad.height = BOX_PX;
  state.strokes = state.strokes.filter((s) => s.every((p) => p.x < pad.width));
  redraw();
}

function padPoint(evt) {
  const rect = $("pad").getBoundingClientRect();
  return { x: (evt.clientX - rect.left) * ($("pad").width / rect.width), y: (evt.clientY - rect.top) * ($("pad").height / rect.height) };
}

function initPad() {
  const pad = $("pad");
  pad.addEventListener("pointerdown", (evt) => {
    pad.setPointerCapture(evt.pointerId);
    state.current = [padPoint(evt)];
    redraw();
  });
  pad.addEventListener("pointermove", (evt) => {
    if (!state.current) return;
    state.current.push(padPoint(evt));
    redraw();
  });
  const finish = () => {
    if (state.current) state.strokes.push(state.current);
    state.current = null;
    redraw();
  };
  pad.addEventListener("pointerup", finish);
  pad.addEventListener("pointercancel", finish);
  $("undo").addEventListener("click", () => { state.strokes.pop(); redraw(); });
  $("clear").addEventListener("click", () => { state.strokes = []; redraw(); });
  $("box-count").addEventListener("change", resizePad);
  resizePad();
}

function exportBoxes() {
  const out = [];
  const box = document.createElement("canvas");
  box.width = BOX_PX;
  box.height = BOX_PX;
  const ctx = box.getContext("2d");
  const strokesOnly = document.createElement("canvas");
  strokesOnly.width = $("pad").width;
  strokesOnly.height = BOX_PX;
  const sctx = strokesOnly.getContext("2d");
  sctx.fillStyle = "#fff";
  sctx.fillRect(0, 0, strokesOnly.width, BOX_PX);
  drawStrokes(sctx, state.strokes);
  for (let i = 0; i < boxCount(); i += 1) {
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, BOX_PX, BOX_PX);
    ctx.drawImage(strokesOnly, i * BOX_PX, 0, BOX_PX, BOX_PX, 0, 0, BOX_PX, BOX_PX);
    out.push(box.toDataURL("image/png").split(",")[1]);
  }
  return out;
}

/* ---------------- samples and upload ---------------- */

function setMode(mode) {
  state.mode = mode;
  $("tab-draw").setAttribute("aria-selected", String(mode === "draw"));
  $("tab-strip").setAttribute("aria-selected", String(mode === "strip"));
  $("panel-draw").hidden = mode !== "draw";
  $("panel-strip").hidden = mode !== "strip";
}

function showStrip(b64, boxes) {
  state.stripB64 = b64;
  $("strip-boxes").value = String(boxes);
  const preview = $("strip-preview");
  preview.src = "data:image/png;base64," + b64;
  preview.hidden = false;
  setMode("strip");
}

async function loadSamples() {
  try {
    const resp = await fetch("/api/samples");
    if (!resp.ok) throw new Error("samples");
    const samples = await resp.json();
    const select = $("sample-select");
    samples.forEach((s, i) => {
      const opt = el("option", `${s.id} · ${s.difficulty} · ${s.num_boxes} boxes`);
      opt.value = String(i);
      select.appendChild(opt);
    });
    select.addEventListener("change", () => {
      const s = samples[parseInt(select.value, 10)];
      if (!s) return;
      $("sentence").value = s.sentence;
      showStrip(s.strip_base64, s.num_boxes);
    });
  } catch (err) {
    showError("Could not load the sample worksheets.");
  }
}

function initUpload() {
  $("strip-file").addEventListener("change", () => {
    const file = $("strip-file").files[0];
    if (!file) return;
    if (!["image/png", "image/jpeg"].includes(file.type)) { showError("Please choose a PNG or JPEG image."); return; }
    const reader = new FileReader();
    reader.onload = () => showStrip(String(reader.result).split(",")[1], parseInt($("strip-boxes").value, 10) || 5);
    reader.readAsDataURL(file);
  });
}

/* ---------------- results ---------------- */

function badge(label, pred) {
  const row = el("div", null, "badge");
  row.appendChild(el("span", label));
  const top = pred.top2.map((t) => `${t.char} ${(t.prob * 100).toFixed(0)}%`).join(" · ");
  row.appendChild(el("b", top));
  return row;
}

function renderBoxes(boxes) {
  const wrap = $("boxes");
  wrap.replaceChildren();
  for (const b of boxes) {
    const card = el("div", null, b.empty ? "box empty" : "box");
    const imgs = el("div", null, "imgs");
    const pre = el("div");
    pre.appendChild(pngImg(b.preview_png_b64, `Box ${b.index + 1} as the model sees it`));
    pre.appendChild(el("div", "28×28 input", "cap"));
    imgs.appendChild(pre);
    if (b.gradcam_png_b64) {
      const cam = el("div");
      cam.appendChild(pngImg(b.gradcam_png_b64, `Grad-CAM heatmap for box ${b.index + 1}`));
      cam.appendChild(el("div", "Grad-CAM", "cap"));
      imgs.appendChild(cam);
    }
    card.appendChild(imgs);
    if (b.empty) {
      card.appendChild(el("div", "empty box (ignored)", "cap"));
    } else {
      const badges = el("div", null, "badges");
      badges.appendChild(badge("Dense", b.dense_pred));
      badges.appendChild(badge("CNN FP32", b.cnn_fp32_pred));
      badges.appendChild(badge("CNN INT8", b.cnn_int8_pred));
      card.appendChild(badges);
    }
    wrap.appendChild(card);
  }
}

function renderModels(models, primary) {
  const body = $("model-rows");
  body.replaceChildren();
  for (const m of models) {
    const tr = el("tr", null, m.name === primary ? "primary-row" : "");
    const name = el("td", m.name.replaceAll("_", " "));
    if (m.name === primary) name.appendChild(el("span", "App model", "tag"));
    tr.appendChild(name);
    tr.appendChild(el("td", m.precision));
    tr.appendChild(el("td", m.predicted_word, "word-cell"));
    tr.appendChild(el("td", m.top_candidates.map((c) => c.word).join(", ") || "—"));
    tr.appendChild(el("td", Number(m.latency_ms).toFixed(1)));
    body.appendChild(tr);
  }
}

function renderResult(data) {
  const primary = data.models.find((m) => m.name === data.primary_model) || data.models[data.models.length - 1];
  $("raw-word").textContent = primary.raw_word;
  $("final-word").textContent = primary.predicted_word;
  renderBoxes(data.boxes);
  renderModels(data.models, data.primary_model);
  $("results").hidden = false;
}

async function recognize() {
  clearError();
  const body = { sentence: $("sentence").value };
  if (state.mode === "draw") {
    if (!state.strokes.length) { showError("Draw at least one character first."); return; }
    body.boxes_base64 = exportBoxes();
  } else {
    if (!state.stripB64) { showError("Choose a sample or upload a strip image first."); return; }
    body.strip_base64 = state.stripB64;
    body.num_boxes = parseInt($("strip-boxes").value, 10);
  }
  const button = $("recognize");
  button.disabled = true;
  $("busy").textContent = "Reading…";
  try {
    const resp = await fetch("/api/recognize", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) { showError(typeof data.detail === "string" ? data.detail : "The request could not be processed."); return; }
    renderResult(data);
  } catch (err) {
    showError("Could not reach the local server. Is python app.py still running?");
  } finally {
    button.disabled = false;
    $("busy").textContent = "";
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initPad();
  initUpload();
  loadSamples();
  $("tab-draw").addEventListener("click", () => setMode("draw"));
  $("tab-strip").addEventListener("click", () => setMode("strip"));
  $("recognize").addEventListener("click", recognize);
});
