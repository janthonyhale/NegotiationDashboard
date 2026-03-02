let DOC = null;
let CUR = 0;
let chart = null;

function $(id){ return document.getElementById(id); }
function clamp(n,a,b){ return Math.max(a, Math.min(b,n)); }

function renderTranscript(){
  const el = $("transcript");
  el.innerHTML = "";
  if(!DOC) return;

  // Only render up to current (no future utterances)
  const shown = DOC.turns.filter(t => t.idx <= CUR);

  for(const t of shown){
    const row = document.createElement("div");
    const sp = (t.speaker || "Unknown");
    row.className = "turn " + (t.idx === CUR ? "current" : "");
    row.innerHTML = `
      <div class="meta">
        <span class="speaker ${sp.toLowerCase()}">${sp}</span>
        <span class="muted small">#${t.idx}</span>
      </div>
      <div class="text"></div>
    `;
    row.querySelector(".text").textContent = (t.text || "");
    el.appendChild(row);
  }

  const cur = DOC.turns[CUR];
  $("turnMeta").textContent = cur ? `${cur.speaker || "Unknown"} · turn ${CUR+1}/${DOC.turns.length}` : "—";

  // keep current in view
  const current = el.querySelector(".turn.current");
  if(current) current.scrollIntoView({block:"nearest"});
}

function emotionForIdx(idx){
  if(!DOC) return null;
  return DOC.emotion.find(e => e.idx === idx) || null;
}

function renderEmoNow(){
  const el = $("emoNow");
  el.innerHTML = "";
  const e = emotionForIdx(CUR);
  if(!e) return;
  const keys = ["anger","sadness","joy","fear","valence"];
  for(const k of keys){
    const box = document.createElement("div");
    box.className = "emoBox";
    box.innerHTML = `<div class="muted small">${k}</div><div class="big">${e[k]}</div>`;
    el.appendChild(box);
  }
}

function renderRisk(){
  if(!DOC) return;
  $("riskLabel").textContent = DOC.risk.label;
  $("riskScore").textContent = `risk=${DOC.risk.risk}`;
  $("riskSignals").textContent = `neg_hits=${DOC.risk.signals.neg_hits}, threat_hits=${DOC.risk.signals.threat_hits}`;
}

function renderAdvisor(){
  if(!DOC) return;
  const inter = (DOC.interventions || []).find(x => x.idx === CUR);
  const badge = $("interveneBadge");
  const short = $("interveneShort");
  const reasons = $("interveneReasons");
  const sugg = $("interveneSuggestion");
  const oddsText = $("oddsText");
  const oddsSuccess = $("oddsSuccess");
  const oddsWalk = $("oddsWalk");

  reasons.innerHTML = "";
  if(!inter){
    badge.classList.add("hidden");
    short.textContent = "—";
    sugg.textContent = "";
    oddsText.textContent = "—";
    oddsSuccess.style.width = "0%";
    oddsWalk.style.width = "0%";
    return;
  }

  if(inter.should_intervene){
    badge.classList.remove("hidden");
    badge.textContent = "Intervene";
  } else {
    badge.classList.add("hidden");
  }

  short.textContent = inter.recommended_action || "—";
  for(const r of (inter.reason || [])){
    const li = document.createElement("li");
    li.textContent = r;
    reasons.appendChild(li);
  }
  sugg.textContent = inter.suggestion || "";

  const s = Math.round((inter.success_odds || 0) * 100);
  const w = Math.round((inter.walkaway_odds || 0) * 100);
  oddsSuccess.style.width = `${s}%`;
  oddsWalk.style.width = `${w}%`;
  oddsText.textContent = `Success: ${s}% · Walkaway: ${w}%`;
}

function renderCountryPrediction(){
  if(!DOC) return;
  $("langVal").textContent = `${DOC.lang_country.language} (${DOC.lang_country.confidence})`;
  $("buyerCountryVal").textContent = `${DOC.lang_country.buyer_country} (${DOC.lang_country.buyer_country_confidence})`;
  $("sellerCountryVal").textContent = `${DOC.lang_country.seller_country} (${DOC.lang_country.seller_country_confidence})`;

  const map = $("countryMap");
  const buyer = DOC.lang_country.buyer_country;
  const seller = DOC.lang_country.seller_country;
  const cent = DOC.country_centroids || {};

  function project(lon, lat){
    const x = (lon + 180) * (300/360);
    const y = (90 - lat) * (150/180);
    return {x, y};
  }

  const pts = [];
  if(cent[buyer]) pts.push({code: buyer, ...project(...cent[buyer]), cls:"buyerDot"});
  if(cent[seller]) pts.push({code: seller, ...project(...cent[seller]), cls:"sellerDot"});

  map.innerHTML = `
    <svg viewBox="0 0 300 150" class="miniMap" aria-label="country map">
      <rect x="0" y="0" width="300" height="150" rx="10" ry="10" class="mapBg"></rect>
      <path class="land" d="M25,65 C55,40 80,40 100,60 C120,80 95,95 70,92 C50,90 35,82 25,65 Z"></path>
      <path class="land" d="M120,55 C145,35 175,35 190,55 C205,75 190,95 160,95 C135,90 125,75 120,55 Z"></path>
      <path class="land" d="M200,60 C225,45 255,50 270,70 C280,90 260,105 235,100 C220,95 205,80 200,60 Z"></path>
      ${pts.map(p => `<circle cx="${p.x}" cy="${p.y}" r="6" class="${p.cls}"><title>${p.code}</title></circle>`).join("")}
    </svg>
  `;
}

function buildEmotionChart(){
  if(!DOC) return;
  const sel = $("emoSelect");
  sel.innerHTML = "";
  for(const k of DOC.supported_emotions || ["anger","sadness","joy","fear","valence"]){
    const opt = document.createElement("option");
    opt.value = k;
    opt.textContent = k;
    sel.appendChild(opt);
  }

  const ctx = $("emoChart").getContext("2d");

  function seriesFor(key){
    const xs = DOC.turns.map(t => t.idx);
    const buyer = xs.map(i => {
      const e = emotionForIdx(i);
      const sp = (e && (e.speaker||"")).toLowerCase();
      if(sp !== "buyer") return null;
      return e[key];
    });
    const seller = xs.map(i => {
      const e = emotionForIdx(i);
      const sp = (e && (e.speaker||"")).toLowerCase();
      if(sp !== "seller") return null;
      return e[key];
    });
    return {xs, buyer, seller};
  }

  function render(key){
    const d = seriesFor(key);
    if(chart) chart.destroy();
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels: d.xs,
        datasets: [
          { label: "Buyer", data: d.buyer, spanGaps: false, tension: 0.25 },
          { label: "Seller", data: d.seller, spanGaps: false, tension: 0.25 },
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: {
          x: { title: { display: true, text: "Turn" } },
          y: { title: { display: true, text: "Score" } }
        }
      }
    });
  }

  sel.addEventListener("change", () => render(sel.value));
  render(sel.value);
}

function renderAll(){
  renderTranscript();
  renderEmoNow();
  renderRisk();
  renderAdvisor();
}

function step(delta){
  if(!DOC) return;
  CUR = clamp(CUR + delta, 0, DOC.turns.length - 1);
  renderAll();
}

async function init(){
  const res = await fetch(`/api/doc/${window.DOC_ID}`);
  DOC = await res.json();

  CUR = 0;
  renderCountryPrediction();
  buildEmotionChart();
  renderAll();

  $("prevBtn")?.addEventListener("click", () => step(-1));
  $("nextBtn")?.addEventListener("click", () => step(1));
  $("jumpBtn")?.addEventListener("click", () => {
    const v = parseInt(($("jumpTo").value || "0"), 10);
    CUR = clamp(v, 0, DOC.turns.length - 1);
    renderAll();
  });
}

document.addEventListener("DOMContentLoaded", init);
