const $=s=>document.querySelector(s), R="../results/";
const MODELS=["M_BASE","M_V","M_P","M_VP","LR_V"], COLOR={M_BASE:"var(--mut)",M_V:"var(--phys)",M_P:"var(--proc)",M_VP:"var(--both)",LR_V:"var(--phys)"};
const LABEL={M_BASE:"BASE",M_V:"PHYSIOLOGY (V)",M_P:"PROCESS (P)",M_VP:"PHYSIOLOGY + PROCESS",LR_V:"V · logistic baseline"};
const S={}; let view="question";
const fj=async f=>{try{const r=await fetch(R+f);return r.ok?await r.json():null}catch{return null}};
const f3=x=>x==null?"–":x.toFixed(3), el=(h)=>{const d=document.createElement("div");d.innerHTML=h;return d};
async function init(){
  [S.matrix,S.cases,S.stress,S.meta,S.audit]=await Promise.all(["transfer_matrix.json","attribution_cases.json","stress_test.json","model_metrics.json","audit.json"].map(fj));
  if(S.meta?.synthetic){const b=$("#banner");b.hidden=false;b.textContent="SYNTHETIC test data: numbers shown are not results and say nothing about real hospitals."}
  document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{view=b.dataset.v;document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on",x==b));render()});
  render()}
const empty=()=>`<div class="empty">No results found in <code>results/</code>. Obtain the dataset (README → Dataset) and run <code>make run</code>, then serve the repository root and reload.</div>`;
function render(){const m=$("#main");m.innerHTML="";m.append(el(({question:Q,transfer:T,case:C,stress:X})[view]()))}
function Q(){return `<h1>When a clinical model predicts sepsis, is it seeing the patient — or seeing what clinicians chose to measure?</h1>
<p class="mut" style="max-width:36em">Hospital records hold two kinds of information. We separate them, train matched models on each, then move the models between hospitals.</p>
<div class="two"><div class="col phys"><div class="k">Physiology</div><p><b>What was happening to the patient?</b></p><p class="mut">Vital signs and lab values, carried forward causally. No indicators of whether or when they were measured.</p></div>
<div class="col proc"><div class="k">Process</div><p><b>What did clinicians choose to measure?</b></p><p class="mut">Which variables were observed, how recently, how often, how large the lab panel was. Never the values themselves.</p></div></div>
<h2>The experiment</h2><p style="max-width:38em">Train on Hospital A and test on Hospital B, and the reverse. If the measurement process differs between hospitals, a model that leaned on it may degrade where a physiology model does not. The data decide — the hypothesis may not hold.</p>
<p class="mut sans" style="font-size:13px">PhysioNet/CinC Challenge 2019 · Set A (BIDMC) and Set B (Emory) · patient-grouped validation · patient-level bootstrap CIs</p>`}
let metric="auroc";
function T(){if(!S.matrix)return empty();
  const cell=(m,s)=>{const r=S.matrix[m][s],ci=r.ci95?.[metric];return `<b>${f3(r[metric])}</b><span>${ci?`[${f3(ci[0])}, ${f3(ci[1])}]`:""}</span>`};
  const delta=(m,s,w)=>{const a=S.matrix[m][s][metric],b=S.matrix[m][w][metric];return a==null||b==null?"":`Δ vs within-target ${(a-b>=0?"+":"")+(a-b).toFixed(3)}`};
  const sel=`<div class="ctl">Metric <select id="met">${["auroc","auprc","brier"].map(x=>`<option ${x==metric?"selected":""} value="${x}">${x.toUpperCase()}</option>`).join("")}</select> <span class="mut">${metric=="brier"?"lower is better":"higher is better"}</span></div>`;
  setTimeout(()=>{const s=$("#met");if(s)s.onchange=e=>{metric=e.target.value;render()}});
  return `<h1>Does the signal survive the move?</h1><p class="mut" style="max-width:38em">Diagonal: patient-grouped cross-validation within a hospital. Off-diagonal (bold border): trained on one hospital, tested on the other.</p>${sel}
  <div class="grid">${MODELS.map(m=>`<div><div class="k" style="color:${COLOR[m]}">${LABEL[m]}</div><table class="m"><tr><th></th><th>test A</th><th>test B</th></tr>
  <tr><th>train A</th><td>${cell(m,"within_A")}</td><td class="x">${cell(m,"A_to_B")}<span><br>${delta(m,"A_to_B","within_B")}</span></td></tr>
  <tr><th>train B</th><td class="x">${cell(m,"B_to_A")}<span><br>${delta(m,"B_to_A","within_A")}</span></td><td>${cell(m,"within_B")}</td></tr></table></div>`).join("")}</div>
  ${S.audit?`<p class="mut sans" style="font-size:12px;margin-top:36px">Leak audit — AUROC of predicting “${S.audit.A.variable} measured now” from physiology features alone: A ${f3(S.audit.A.auroc)}, B ${f3(S.audit.B.auroc)} (0.5 = no timing information recovered).</p>`:""}`}
let ci=0,mode="all",dir="A_to_B";
function C(){if(!S.cases)return empty();
  const cs=S.cases.cases.filter(c=>c.id.startsWith(dir)), c=cs[Math.min(ci,cs.length-1)];
  setTimeout(()=>{document.querySelectorAll("[data-m]").forEach(b=>b.onclick=()=>{mode=b.dataset.m;render()});
    $("#cs").onchange=e=>{ci=+e.target.value;render()};$("#dr").onchange=e=>{dir=e.target.value;ci=0;render()};draw(c)});
  return `<h1>Same patient timeline. Different information source.</h1>
  <div class="ctl"><select id="dr">${["A_to_B","B_to_A"].map(d=>`<option ${d==dir?"selected":""} value="${d}">${d.replace("_to_"," → ")}</option>`).join("")}</select>
  <select id="cs">${cs.map((x,i)=>`<option ${i==ci?"selected":""} value="${i}">patient ${x.pid} · ${x.ever_septic?"sepsis label present":"no sepsis label"}</option>`).join("")}</select>
  ${[["all","All information"],["physiology_only","Physiology only"],["process_only","Process only"]].map(([k,t])=>`<button data-m="${k}" class="${mode==k?"on":""}">${t}</button>`).join("")}</div>
  <p class="mut sans" style="font-size:12px">Patients are a seeded random sample, not hand-picked. Model-input counterfactuals: the combined model with one family replaced by a fixed reference. Not clinical counterfactuals; ablated inputs are off-distribution.</p><div id="cv"></div>`}
function draw(c){const W=980,pad=46,H=[170,230,170],n=c.hours.length,x=i=>pad+(W-2*pad)*(c.hours[i]-c.hours[0])/Math.max(1,c.hours.at(-1)-c.hours[0]);
  const hx=h=>pad+(W-2*pad)*(h-c.hours[0])/Math.max(1,c.hours.at(-1)-c.hours[0]);let s=`<svg viewBox="0 0 ${W} 640" width="100%">`;
  const v="HR",vals=c.values[v],pts=vals.map((y,i)=>[i,y]).filter(p=>p[1]!=null);
  if(pts.length){const lo=Math.min(...pts.map(p=>p[1])),hi=Math.max(...pts.map(p=>p[1])),y=u=>10+140*(1-(u-lo)/Math.max(1e-6,hi-lo));
    s+=`<text x="${pad}" y="12" style="fill:var(--phys)">PHYSIOLOGY — ${v} (dots = actual measurements)</text><polyline fill="none" stroke="var(--phys)" stroke-width="1.5" points="${pts.map(p=>x(p[0])+","+(y(p[1])+8)).join(" ")}"/>${pts.map(p=>`<circle cx="${x(p[0])}" cy="${y(p[1])+8}" r="2.6" fill="var(--phys)"/>`).join("")}`}
  const vars=Object.keys(c.observed_hours),rh=190/vars.length,oy=H[0]+30;
  s+=`<text x="${pad}" y="${oy-6}" style="fill:var(--proc)">PROCESS — each row is one variable; ticks = a measurement was taken</text>`;
  vars.forEach((k,j)=>c.observed_hours[k].forEach(h=>s+=`<rect x="${hx(h)-1}" y="${oy+j*rh}" width="2" height="${rh*.7}" fill="var(--proc)" opacity=".8"/>`));
  const py=oy+H[1]+10,ph=130,Y=p=>py+ph*(1-Math.min(1,p)),key={all:"all",physiology_only:"physiology_only",process_only:"process_only"}[mode];
  s+=`<text x="${pad}" y="${py-4}">MODEL PREDICTION (${({all:"all information",physiology_only:"process removed",process_only:"physiology removed"})[mode]}) · shaded = SepsisLabel=1</text>`;
  c.label.forEach((l,i)=>{if(l)s+=`<rect x="${x(i)-3}" y="${py}" width="6" height="${ph}" fill="var(--lab)" opacity=".13"/>`});
  const line=(arr,col,w,d)=>`<polyline fill="none" stroke="${col}" stroke-width="${w}" ${d?'stroke-dasharray="3 3"':""} points="${arr.map((p,i)=>x(i)+","+Y(p)).join(" ")}"/>`;
  s+=line(c.pred.all,"var(--both)",1,true)+line(c.pred[key],mode=="process_only"?"var(--proc)":mode=="physiology_only"?"var(--phys)":"var(--both)",2.2)+`<line x1="${pad}" x2="${W-pad}" y1="${py+ph}" y2="${py+ph}" stroke="var(--rule)"/><text x="${W-pad-170}" y="${py-4}">dashed = all-information prediction</text>`;
  const pr=c.pri.filter(p=>p!=null);s+=`</svg><p class="mut sans" style="font-size:12px">Hours with a defined Process Reliance Index: ${pr.length}/${n}${pr.length?` · median ${f3(pr.sort((a,b)=>a-b)[pr.length>>1])}`:""} (model-based, associational).</p>`;
  $("#cv").innerHTML=s}
function X(){if(!S.stress||!Object.keys(S.stress).length)return `<h1>Stress test</h1><div class="empty">Measurement-thinning results are not available. Run the pipeline without <code>--no-stress</code>.</div>`;
  const d=S.stress.A_to_B,ks=Object.keys(d),W=640,H=300;
  const val=(k,m)=>d[k][m].auroc;let s=`<svg viewBox="0 0 ${W} ${H+30}" width="${W}">`;
  const all=ks.flatMap(k=>MODELS.map(m=>val(k,m))).filter(v=>v!=null),lo=Math.min(...all)-.02,hi=Math.max(...all)+.02,Y=v=>H-(H-20)*(v-lo)/(hi-lo),Xp=i=>60+(W-120)*i/(ks.length-1);
  MODELS.forEach(m=>{s+=`<polyline fill="none" stroke="${COLOR[m]}" stroke-width="2" points="${ks.map((k,i)=>Xp(i)+","+Y(val(k,m))).join(" ")}"/><text x="${W-56}" y="${Y(val(ks.at(-1),m))+4}" style="fill:${COLOR[m]}">${m}</text>`});
  ks.forEach((k,i)=>s+=`<text x="${Xp(i)}" y="${H+20}" text-anchor="middle">${Math.round(k*100)}% removed</text>`);
  return `<h1>If the target hospital measured less</h1><p class="mut" style="max-width:38em">Models trained on A, evaluated on B after randomly deleting a fraction of B's measurements. AUROC shown; labels untouched.</p>${s}</svg>`}
init();
