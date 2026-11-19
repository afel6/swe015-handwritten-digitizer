"use strict";

/* Top-level tool tabs and the optional full-page OCR tab (local Ollama + Qwen3-VL). */

const MAX_SIDE_PX = 2048;
const JPEG_QUALITY = 0.9;
const SLOW_HINT_MS = 3000;
const COPIED_MS = 1400;
const ocr = { b64: null, previewUrl: null, text: "", ranOnce: false, statusChecked: false, requestId: 0 };
const byId = (id) => document.getElementById(id);

/* ---------------- tool tabs ---------------- */

function setTool(name, focus) {
  const tools = ["reader", "ocr"];
  for (const t of tools) {
    const tab = byId(`tab-${t}`);
    const on = t === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    byId(`panel-${t}`).hidden = !on;
    if (on && focus) tab.focus();
  }
  const banner = byId("error-banner");
  banner.hidden = true;
  banner.textContent = "";
  if (name === "ocr" && !ocr.statusChecked) checkStatus();
}

function initToolTabs() {
  byId("tab-reader").addEventListener("click", () => setTool("reader"));
  byId("tab-ocr").addEventListener("click", () => setTool("ocr"));
  document.querySelector(".apptabs").addEventListener("keydown", (evt) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(evt.key)) return;
    evt.preventDefault();
    const next = byId("tab-reader").getAttribute("aria-selected") === "true" ? "ocr" : "reader";
    setTool(evt.key === "Home" ? "reader" : evt.key === "End" ? "ocr" : next, true);
  });
}

/* ---------------- status ---------------- */

function setPill(text, kind) {
  const pill = byId("ocr-state");
  pill.textContent = text;
  pill.className = `pill ${kind || ""}`.trim();
}

async function checkStatus() {
  ocr.statusChecked = true;
  setPill("Checking the local model…");
  try {
    const resp = await fetch("/api/ocr/status");
    const data = await resp.json();
    byId("ocr-setup").hidden = Boolean(data.model_ready);
    byId("ocr-setup-detail").textContent = data.model_ready ? "" : String(data.detail || "");
    setPill(data.model_ready ? "Model ready · runs on this computer" : "Model not set up", data.model_ready ? "ok" : "warn");
  } catch (err) {
    setPill("Could not reach the local server", "warn");
  }
}

/* ---------------- image in ---------------- */

async function downscale(file) {
  const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  const scale = Math.min(1, MAX_SIDE_PX / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close();
  return canvas.toDataURL("image/jpeg", JPEG_QUALITY).split(",")[1];
}

function showText(text, kind) {
  const box = byId("ocr-text");
  box.classList.remove("fresh", "error");
  box.replaceChildren();
  if (kind === "placeholder") {
    const span = document.createElement("span");
    span.className = "placeholder";
    span.textContent = text;
    box.appendChild(span);
  } else {
    box.textContent = text;
    if (kind === "error") box.classList.add("error");
  }
  void box.offsetWidth; // restart the fade for each new result
  box.classList.add("fresh");
}

async function pick(file) {
  if (!file || !["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    showText("Please choose a PNG, JPEG or WebP image.", "error");
    return;
  }
  byId("ocr-go").disabled = true; // no Transcribe while the new image is still loading
  ocr.requestId += 1; // a transcription still in flight belongs to the old image: drop its result
  byId("ocr-busy").textContent = "";
  ocr.b64 = null;
  try {
    ocr.b64 = await downscale(file);
  } catch (err) {
    showText("This image could not be opened. Try a PNG or JPEG.", "error");
    return;
  }
  if (ocr.previewUrl) URL.revokeObjectURL(ocr.previewUrl);
  ocr.previewUrl = URL.createObjectURL(file);
  const preview = byId("ocr-preview");
  preview.src = ocr.previewUrl;
  preview.hidden = false;
  byId("ocr-go").disabled = false;
  ocr.text = "";
  byId("ocr-copy").disabled = true;
  byId("ocr-meta").textContent = "";
  showText("Press Transcribe to read this image.", "placeholder");
}

function initInput() {
  const drop = byId("ocr-drop");
  byId("ocr-file").addEventListener("change", () => pick(byId("ocr-file").files[0]));
  drop.addEventListener("dragover", (evt) => { evt.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (evt) => { evt.preventDefault(); drop.classList.remove("over"); pick(evt.dataTransfer.files[0]); });
  document.addEventListener("paste", (evt) => {
    if (byId("panel-ocr").hidden) return;
    const item = [...evt.clipboardData.items].find((i) => i.type.startsWith("image/"));
    if (item) pick(item.getAsFile());
  });
}

/* ---------------- transcribe and copy ---------------- */

function metaLine(data) {
  const parts = [`${(Number(data.latency_ms) / 1000).toFixed(1)} s`];
  if (data.min_prob !== null && data.min_prob !== undefined) parts.push(`least certain character ${(data.min_prob * 100).toFixed(1)} %`);
  if (data.finish_reason === "length") parts.push("stopped at the length limit");
  return parts.join(" · ");
}

async function transcribe() {
  if (!ocr.b64) return;
  ocr.requestId += 1;
  const id = ocr.requestId;
  const button = byId("ocr-go");
  button.disabled = true;
  byId("ocr-busy").textContent = "Reading…";
  const slow = setTimeout(() => {
    if (!ocr.ranOnce && id === ocr.requestId) byId("ocr-busy").textContent = "Loading the model, about a minute the first time…";
  }, SLOW_HINT_MS);
  try {
    const resp = await fetch("/api/ocr", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ image_base64: ocr.b64 }) });
    const data = await resp.json().catch(() => ({}));
    if (id !== ocr.requestId) return;
    if (!resp.ok) {
      showText(typeof data.detail === "string" ? data.detail : "The image could not be read.", "error");
      if (resp.status === 503) checkStatus();
      return;
    }
    ocr.ranOnce = true;
    ocr.text = String(data.text || "");
    showText(ocr.text || "(no text found)", ocr.text ? "" : "placeholder");
    byId("ocr-meta").textContent = metaLine(data);
    byId("ocr-copy").disabled = !ocr.text;
  } catch (err) {
    if (id === ocr.requestId) showText("Could not reach the local server. Is python app.py still running?", "error");
  } finally {
    clearTimeout(slow);
    if (id === ocr.requestId) {
      button.disabled = !ocr.b64;
      byId("ocr-busy").textContent = "";
    }
  }
}

async function copyText() {
  const button = byId("ocr-copy");
  try {
    await navigator.clipboard.writeText(ocr.text);
    button.textContent = "Copied";
    button.classList.add("copied");
    setTimeout(() => { button.textContent = "Copy"; button.classList.remove("copied"); }, COPIED_MS);
  } catch (err) {
    window.getSelection().selectAllChildren(byId("ocr-text"));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initToolTabs();
  initInput();
  byId("ocr-go").addEventListener("click", transcribe);
  byId("ocr-copy").addEventListener("click", copyText);
  byId("ocr-recheck").addEventListener("click", checkStatus);
});
