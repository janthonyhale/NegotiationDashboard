async function fetchDoc(docId){
  const res = await fetch(`/api/doc/${docId}`);
  if(!res.ok) throw new Error("Doc not found");
  return await res.json();
}

function clamp(n, lo, hi){ return Math.max(lo, Math.min(hi, n)); }

function drawEmotionChart(canvas, emotionSeries){
  // Minimal chart without external libs (simple line-ish rendering).
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth;
  const H = canvas.height = canvas.clientHeight;

  ctx.clearRect(0,0,W,H);

  // axes
  ctx.globalAlpha = 0.6;
  ctx.strokeStyle = "#9fb6ff";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(40, 10);
  ctx.lineTo(40, H-30);
  ctx.lineTo(W-10, H-30);
  ctx.stroke();
  ctx.globalAlpha = 1;

  const seriesNames = ["anger","sadness","joy","fear"];
  const maxY = Math.max(1, ...emotionSeries.flatMap(d => seriesNames.map(k => d[k] || 0)));
  const N = Math.max(1, emotionSeries.length);

  function x(i){
    const left=50, right=W-15;
    return left + (right-left) * (i/(N-1 || 1));
  }
  function y(v){
    const top=15, bot=H-40;
    return bot - (bot-top) * (v/maxY);
  }

  // legend
  ctx.font = "12px system-ui";
  ctx.fillStyle = "rgba(232,238,252,.9)";
  ctx.fillText("anger  sadness  joy  fear (placeholder)", 50, H-10);

  // draw each series in a different alpha; keep default strokeStyle but vary alpha
  seriesNames.forEach((name, idx) => {
    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(122,162,255,1)";
    ctx.globalAlpha = 0.25 + idx * 0.18;

    emotionSeries.forEach((d, i) => {
      const xv = x(i);
      const yv = y(d[name] || 0);
      if(i === 0) ctx.moveTo(xv, yv);
      else ctx.lineTo(xv, yv);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  });
}

async function translate(text, target_lang){
  const res = await fetch("/api/translate", {
    method:"POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({text, target_lang})
  });
  return await res.json();
}

function renderTurn(state){
  const t = state.turns[state.idx];
  const iv = state.interventions[state.idx];

  document.getElementById("turnIdx").textContent = `#${t.idx}`;
  document.getElementById("turnSpeaker").textContent = t.speaker || "Unknown";
  document.getElementById("turnTs").textContent = t.ts ? `ts=${t.ts}` : "";
  document.getElementById("turnText").textContent = t.text || "";

  // Intervention UI
  const badge = document.getElementById("interveneBadge");
  const reasons = document.getElementById("interveneReasons");
  const suggestion = document.getElementById("interveneSuggestion");

  reasons.innerHTML = "";
  suggestion.textContent = "";

  if(iv && iv.should_intervene){
    badge.classList.remove("hidden");
    (iv.reason || []).forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      reasons.appendChild(li);
    });
    suggestion.textContent = iv.suggestion || "";
  } else {
    badge.classList.add("hidden");
  }

  // Reset translation box
  const box = document.getElementById("translationBox");
  box.classList.add("hidden");
  box.textContent = "";
}

function wireControls(state){
  const prevBtn = document.getElementById("prevBtn");
  const nextBtn = document.getElementById("nextBtn");
  const jumpTo = document.getElementById("jumpTo");
  const jumpBtn = document.getElementById("jumpBtn");
  const translateBtn = document.getElementById("translateBtn");
  const translateLang = document.getElementById("translateLang");
  const translationBox = document.getElementById("translationBox");

  prevBtn.addEventListener("click", () => {
    state.idx = clamp(state.idx - 1, 0, state.turns.length - 1);
    renderTurn(state);
  });

  nextBtn.addEventListener("click", () => {
    state.idx = clamp(state.idx + 1, 0, state.turns.length - 1);
    renderTurn(state);
  });

  jumpBtn.addEventListener("click", () => {
    const v = parseInt(jumpTo.value, 10);
    if(Number.isFinite(v)){
      state.idx = clamp(v, 0, state.turns.length - 1);
      renderTurn(state);
    }
  });

  translateBtn.addEventListener("click", async () => {
    const t = state.turns[state.idx];
    const lang = translateLang.value || "en";
    translateBtn.disabled = true;
    translateBtn.textContent = "Translating…";
    try{
      const out = await translate(t.text || "", lang);
      translationBox.textContent = out.translated_text || "(no output)";
      translationBox.classList.remove("hidden");
    } finally {
      translateBtn.disabled = false;
      translateBtn.textContent = "Translate (placeholder)";
    }
  });

  // keyboard shortcuts
  window.addEventListener("keydown", (e) => {
    if(e.key === "ArrowLeft"){
      state.idx = clamp(state.idx - 1, 0, state.turns.length - 1);
      renderTurn(state);
    }
    if(e.key === "ArrowRight"){
      state.idx = clamp(state.idx + 1, 0, state.turns.length - 1);
      renderTurn(state);
    }
  });
}

(async function init(){
  const docId = window.DOC_ID;
  const doc = await fetchDoc(docId);

  const state = {
    idx: 0,
    turns: doc.turns || [],
    emotion: doc.emotion || [],
    interventions: doc.interventions || []
  };

  // set jump max hint
  document.getElementById("jumpTo").value = "0";

  // chart
  const canvas = document.getElementById("emoChart");
  drawEmotionChart(canvas, state.emotion);

  // viewer
  wireControls(state);
  renderTurn(state);

  // redraw chart on resize
  window.addEventListener("resize", () => drawEmotionChart(canvas, state.emotion));
})();
