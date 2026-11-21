"use strict";

const PAGE_WIDTH = 1000;
const PAGE_HEIGHT = 1400;
const BOX = 90;
const LEFT = 90;
const questions = [
  { sample: "test_001", sentence: "Around [MASK] people are struck by lightning each year.", boxes: 4, digits: true },
  { sample: "test_023", sentence: "First was the [MASK] of steam power to run the machinery.", boxes: 3, digits: false },
  { sample: "test_002", sentence: "Alexander Graham Bell first patented the telephone in [MASK].", boxes: 4, digits: true },
  { sample: "test_162", sentence: "Copper is [MASK] from the body easily.", boxes: 8, digits: false },
];
const byId = id => document.getElementById(id);
let run = null;

function rowTop(index) {
  return 253 + index * 275;
}

function makeTemplate() {
  const page = document.createElement("canvas");
  page.width = PAGE_WIDTH;
  page.height = PAGE_HEIGHT;
  const ctx = page.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, page.width, page.height);
  // The outside of the printed frame is the crop boundary, including on paper with printer margins.
  ctx.strokeStyle = "#111";
  ctx.lineWidth = 6;
  ctx.strokeRect(3, 3, page.width - 6, page.height - 6);
  ctx.fillStyle = "#111";
  ctx.font = "bold 48px sans-serif";
  ctx.fillText("Boxed answer worksheet", LEFT, 96);
  ctx.font = "25px sans-serif";
  ctx.fillText("Write one character in each box. Leave unused boxes empty.", LEFT, 151);
  ctx.strokeStyle = "#333";
  ctx.lineWidth = 2;
  questions.forEach((question, index) => {
    const top = rowTop(index);
    ctx.fillText((index + 1) + ". " + question.sentence.replace("[MASK]", "_______"), LEFT, top - 25);
    for (let col = 0; col < question.boxes; col += 1) ctx.strokeRect(LEFT + BOX * col, top, BOX, BOX);
  });
  return page;
}

function loadImage(url) {
  const image = new Image();
  image.src = url;
  return image.decode().then(() => image);
}

function cropAnswer(page, question, index) {
  const strip = document.createElement("canvas");
  strip.width = BOX * question.boxes;
  strip.height = BOX;
  const x = page.width / PAGE_WIDTH;
  const y = page.height / PAGE_HEIGHT;
  const ctx = strip.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, strip.width, strip.height);
  ctx.drawImage(page, LEFT * x, rowTop(index) * y, strip.width * x, BOX * y, 0, 0, strip.width, BOX);
  return strip.toDataURL("image/png");
}

async function demoPage() {
  const response = await fetch("/api/samples");
  if (!response.ok) throw new Error("Could not load the demo samples.");
  const samples = await response.json();
  const page = makeTemplate();
  const ctx = page.getContext("2d");
  await Promise.all(questions.map(async (question, index) => {
    const sample = samples.find(item => item.id === question.sample);
    if (!sample) throw new Error("A demo sample is missing.");
    const image = await loadImage("data:image/png;base64," + sample.strip_base64);
    ctx.drawImage(image, LEFT, rowTop(index), BOX * question.boxes, BOX);
  }));
  return page;
}

async function recognizeStrip(image, question) {
  const response = await fetch("/api/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence: question.sentence, strip_base64: image.split(",")[1],
      num_boxes: question.boxes, digits_only: question.digits, allow_blank: true }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Recognition failed.");
  if (data.empty_answer === true) return "";
  const result = data.models.find(model => model.name === data.primary_model);
  if (typeof result?.predicted_word !== "string") throw new Error("Recognition returned no transcription.");
  return result.predicted_word;
}

function renderRow(question, index, image, suggestion, warning) {
  const row = byId("worksheet-rows").insertRow();
  const label = document.createElement("th");
  label.scope = "row";
  label.textContent = (index + 1) + ". " + question.sentence.replace("[MASK]", "___");
  row.append(label);
  const preview = new Image();
  preview.alt = "Handwritten answer to question " + (index + 1);
  preview.src = image;
  row.insertCell().append(preview);
  const field = document.createElement("input");
  field.type = "text";
  field.maxLength = question.boxes;
  field.pattern = question.digits ? "[0-9]*" : "[A-Za-z0-9]*";
  field.inputMode = question.digits ? "numeric" : "text";
  field.autocomplete = "off";
  field.spellcheck = false;
  field.setAttribute("aria-label", "Transcription for question " + (index + 1));
  field.title = question.digits ? "Enter digits only." : "Enter letters or digits without spaces.";
  field.value = suggestion;
  const cell = row.insertCell();
  cell.append(field);
  if (warning) {
    const message = document.createElement("p");
    message.className = "hint";
    message.textContent = warning;
    cell.append(message);
  }
  return field;
}

async function start(loadPage, pageId, mode) {
  const started = performance.now();
  const status = byId("worksheet-status");
  const controls = ["worksheet-start", "load-demo", "worksheet-id", "worksheet-file", "worksheet-mode"].map(byId);
  controls.forEach(control => { control.disabled = true; });
  status.textContent = "Loading worksheet…";
  byId("worksheet-review").hidden = true;
  byId("worksheet-rows").replaceChildren();
  byId("worksheet-export-status").textContent = "";
  run = null;
  try {
    const page = await loadPage();
    const scale = (page.width / page.height) / (PAGE_WIDTH / PAGE_HEIGHT);
    if (Math.abs(scale - 1) > 0.04 || page.width < 400) {
      throw new Error("Use an upright image at least 400 pixels wide, cropped to the outside edge of the printed frame.");
    }
    const rows = [];
    let failures = 0;
    for (const [index, question] of questions.entries()) {
      const image = cropAnswer(page, question, index);
      let suggestion = "";
      let suggestionStatus = "";
      let warning = "";
      if (mode === "assisted") {
        status.textContent = "Reading answer " + (index + 1) + " of " + questions.length + "…";
        try {
          suggestion = await recognizeStrip(image, question);
          suggestionStatus = "ok";
        } catch (error) {
          suggestionStatus = "failed";
          warning = "Check this answer manually. " + error.message;
          failures += 1;
        }
      }
      rows.push({ suggestion, suggestionStatus, field: renderRow(question, index, image, suggestion, warning) });
    }
    run = { pageId, mode, started, rows, seconds: null };
    status.textContent = failures ? failures + " answers need manual entry." : "Worksheet ready to review.";
    byId("worksheet-review").hidden = false;
    rows[0].field.focus();
  } catch (error) {
    status.textContent = "Could not open worksheet. " + error.message;
  } finally {
    controls.forEach(control => { control.disabled = false; });
  }
}

function download(url, name) {
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
}

byId("download-template").addEventListener("click", () => download(makeTemplate().toDataURL("image/png"), "boxed-worksheet.png"));
byId("load-demo").addEventListener("click", () => start(demoPage, "demo", byId("worksheet-mode").value));
byId("worksheet-form").addEventListener("submit", event => {
  event.preventDefault();
  const file = byId("worksheet-file").files[0];
  const pageId = byId("worksheet-id").value;
  const mode = byId("worksheet-mode").value;
  start(async () => {
    if (!file || !["image/png", "image/jpeg"].includes(file.type) || file.size > 20 * 1024 * 1024) {
      throw new Error("Choose a PNG or JPEG up to 20 MB.");
    }
    const url = URL.createObjectURL(file);
    try {
      return await loadImage(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }, pageId, mode);
});
byId("worksheet-rows").addEventListener("input", () => {
  run.seconds = null;
  byId("worksheet-export-status").textContent = "";
});
byId("worksheet-export").addEventListener("submit", event => {
  event.preventDefault();
  if (!run) return;
  run.seconds ??= ((performance.now() - run.started) / 1000).toFixed(3);
  const rows = run.rows.map((row, index) => [run.pageId, index + 1, run.mode,
    questions[index].digits ? "digits" : "word", row.suggestion, row.field.value, run.seconds, row.suggestionStatus].join(","));
  const header = "page_id,question,method,kind,suggestion,transcription,seconds,suggestion_status";
  const url = URL.createObjectURL(new Blob([[header, ...rows].join("\r\n") + "\r\n"], { type: "text/csv" }));
  download(url, run.pageId + "-" + run.mode + ".csv");
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  byId("worksheet-export-status").textContent = "CSV exported. Session time " + run.seconds + " seconds.";
});
