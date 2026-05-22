from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model  = joblib.load(os.path.join(BASE_DIR, 'model', 'rf_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.pkl'))

FEATURES = [
    'age', 'sex', 'bmi', 'systolic_bp', 'diastolic_bp',
    'cholesterol', 'glucose', 'smoking', 'diabetes',
    'activity', 'alcohol', 'family_history', 'bp_medication'
]

HTML_LANDING = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HeartGuard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,sans-serif;background:#fafafa;}
nav{background:white;padding:14px 2rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #eee;position:sticky;top:0;z-index:10;}
.logo{display:flex;align-items:center;gap:10px;}
.logo-icon{width:36px;height:36px;background:#FCEBEB;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:18px;}
.logo-name{font-size:17px;font-weight:700;color:#1a1a1a;}
.nav-links{display:flex;gap:24px;font-size:14px;}
.nav-links a{text-decoration:none;color:#666;}
.nav-links a:hover{color:#e63946;}
.start-btn{background:#e63946;color:white;padding:9px 22px;border-radius:22px;font-size:14px;font-weight:700;border:none;cursor:pointer;text-decoration:none;}
.hero{text-align:center;padding:5rem 2rem 3rem;background:white;}
.hero-badge{display:inline-block;background:#FCEBEB;color:#791F1F;font-size:12px;font-weight:600;padding:5px 16px;border-radius:20px;margin-bottom:1.25rem;}
.hero h1{font-size:40px;font-weight:800;color:#1a1a1a;line-height:1.2;margin-bottom:1rem;}
.hero h1 span{color:#e63946;}
.hero p{font-size:16px;color:#666;max-width:520px;margin:0 auto 2rem;line-height:1.75;}
.hero-btns{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;}
.btn-main{background:#e63946;color:white;padding:14px 32px;border-radius:14px;font-size:15px;font-weight:700;border:none;cursor:pointer;text-decoration:none;}
.btn-main:hover{background:#c1121f;}
.btn-sec{background:white;color:#e63946;padding:14px 32px;border-radius:14px;font-size:15px;font-weight:600;border:2px solid #e63946;cursor:pointer;text-decoration:none;}
.stats-bar{display:flex;justify-content:center;gap:3rem;padding:2rem;background:#FCEBEB;flex-wrap:wrap;}
.stat{text-align:center;}
.stat-val{font-size:28px;font-weight:800;color:#c1121f;}
.stat-lab{font-size:13px;color:#A32D2D;margin-top:2px;}
.section{padding:3rem 2rem;max-width:900px;margin:0 auto;}
.section-title{font-size:24px;font-weight:700;color:#1a1a1a;text-align:center;margin-bottom:0.5rem;}
.section-sub{text-align:center;color:#888;font-size:14px;margin-bottom:2rem;}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;}
.feat{background:white;border-radius:16px;padding:1.5rem;border:1px solid #eee;}
.feat-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:1rem;}
.feat h3{font-size:15px;font-weight:700;color:#1a1a1a;margin-bottom:6px;}
.feat p{font-size:13px;color:#888;line-height:1.6;}
.how-steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;}
.step{text-align:center;padding:1.5rem 1rem;}
.step-num{width:44px;height:44px;background:#e63946;color:white;border-radius:50%;font-size:18px;font-weight:700;display:flex;align-items:center;justify-content:center;margin:0 auto 1rem;}
.step h3{font-size:14px;font-weight:700;color:#1a1a1a;margin-bottom:6px;}
.step p{font-size:13px;color:#888;}
.cta{background:#e63946;padding:3rem 2rem;text-align:center;}
.cta h2{font-size:26px;font-weight:800;color:white;margin-bottom:0.75rem;}
.cta p{color:rgba(255,255,255,0.85);font-size:14px;margin-bottom:1.5rem;}
.cta-btn{background:white;color:#e63946;padding:14px 36px;border-radius:14px;font-size:15px;font-weight:700;border:none;cursor:pointer;text-decoration:none;}
footer{text-align:center;padding:1.5rem;font-size:12px;color:#aaa;background:white;border-top:1px solid #eee;}
</style>
</head>
<body>
<nav>
  <div class="logo">
    <div class="logo-icon">&#10084;&#65039;</div>
    <div class="logo-name">HeartGuard</div>
  </div>
  <div class="nav-links">
    <a href="/">Home</a>
    <a href="#features">Features</a>
    <a href="#how">How it works</a>
  </div>
  <a href="/checkup" class="start-btn">Start Checkup &rarr;</a>
</nav>
<div class="hero">
  <div class="hero-badge">&#129302; Powered by Random Forest AI</div>
  <h1>Your heart health,<br><span>predicted by AI</span></h1>
  <p>HeartGuard analyses 13 health indicators and gives you an instant 10-year cardiovascular disease risk estimate.</p>
  <div class="hero-btns">
    <a href="/checkup" class="btn-main">&#129657; Start Free Checkup</a>
  </div>
</div>
<div class="stats-bar">
  <div class="stat"><div class="stat-val">150</div><div class="stat-lab">Decision trees</div></div>
  <div class="stat"><div class="stat-val">3,000</div><div class="stat-lab">Training records</div></div>
  <div class="stat"><div class="stat-val">13</div><div class="stat-lab">Health features</div></div>
  <div class="stat"><div class="stat-val">84%</div><div class="stat-lab">Model accuracy</div></div>
</div>
<div class="section" id="features">
  <div class="section-title">Why HeartGuard?</div>
  <div class="section-sub">Everything you need to understand your heart health</div>
  <div class="features">
    <div class="feat"><div class="feat-icon" style="background:#FCEBEB;">&#129504;</div><h3>AI-powered prediction</h3><p>Random Forest with 150 trees analyses your inputs and returns an accurate risk score instantly.</p></div>
    <div class="feat"><div class="feat-icon" style="background:#EAF3DE;">&#128202;</div><h3>Detailed risk breakdown</h3><p>See which health factors are driving your risk level up or down.</p></div>
    <div class="feat"><div class="feat-icon" style="background:#FAEEDA;">&#128203;</div><h3>Checkup history</h3><p>Every checkup is saved locally. Track how your risk changes over time.</p></div>
    <div class="feat"><div class="feat-icon" style="background:#EEEDFE;">&#9889;</div><h3>Instant results</h3><p>The model runs in milliseconds with a colour-coded risk gauge.</p></div>
  </div>
</div>
<div style="background:#f5f5f5;padding:3rem 2rem;" id="how">
  <div style="max-width:900px;margin:0 auto;">
    <div class="section-title">How it works</div>
    <div class="section-sub">Three simple steps to your risk score</div>
    <div class="how-steps">
      <div class="step"><div class="step-num">1</div><h3>Enter your details</h3><p>Fill in your health info — age, blood pressure, lifestyle habits</p></div>
      <div class="step"><div class="step-num">2</div><h3>AI analyses</h3><p>150 decision trees vote on your risk level in milliseconds</p></div>
      <div class="step"><div class="step-num">3</div><h3>Get your score</h3><p>See your risk %, colour-coded level, and key risk factors</p></div>
    </div>
  </div>
</div>
<div class="cta">
  <h2>Ready to check your heart health?</h2>
  <p>It takes less than 2 minutes. Free and powered by real ML.</p>
  <a href="/checkup" class="cta-btn">Start My Checkup &rarr;</a>
</div>
<footer>HeartGuard &mdash; For educational purposes only. Not a substitute for medical advice.</footer>
</body>
</html>"""

HTML_CHECKUP = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>HeartGuard — Checkup</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:system-ui,sans-serif;background:#f5f0ff;min-height:100vh;padding-bottom:3rem;}
nav{background:white;padding:14px 2rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #eee;}
.logo{display:flex;align-items:center;gap:10px;}
.logo-icon{width:36px;height:36px;background:#FCEBEB;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:18px;}
.logo-name{font-size:17px;font-weight:700;color:#1a1a1a;}
.back-link{font-size:14px;color:#e63946;text-decoration:none;font-weight:600;}
.wrap{max-width:740px;margin:0 auto;padding:2rem 1rem;}
.page-head{margin-bottom:1.5rem;}
.page-head h1{font-size:22px;font-weight:700;color:#1a1a1a;margin-bottom:4px;}
.page-head p{font-size:14px;color:#888;}
.card{background:white;border-radius:16px;padding:1.5rem;margin-bottom:1rem;border:1px solid #eee;}
.sec-lab{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#aaa;margin-bottom:12px;}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.field{display:flex;flex-direction:column;gap:5px;}
.field label{font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:0.04em;}
input,select{padding:9px 11px;border:1.5px solid #e8e8e8;border-radius:8px;font-size:14px;color:#222;background:#fafafa;width:100%;}
input:focus,select:focus{outline:none;border-color:#e63946;background:white;}
.tog-row{display:flex;gap:6px;flex-wrap:wrap;}
.tog{padding:6px 16px;border-radius:20px;border:1.5px solid #e8e8e8;font-size:12px;cursor:pointer;background:white;color:#888;font-weight:500;}
.tog.on{background:#FCEBEB;border-color:#E24B4A;color:#791F1F;}
.assess-btn{width:100%;padding:15px;background:#e63946;color:white;border:none;border-radius:14px;font-size:16px;font-weight:700;cursor:pointer;margin-top:0.5rem;}
.assess-btn:hover{background:#c1121f;}
.result-wrap{display:none;margin-top:1.5rem;}
.result-top{padding:1.75rem;border-radius:16px 16px 0 0;display:flex;align-items:center;justify-content:space-between;}
.result-score{font-size:52px;font-weight:800;line-height:1;}
.risk-pill{padding:8px 20px;border-radius:20px;font-size:14px;font-weight:700;color:white;}
.bar-section{background:white;padding:1rem 1.5rem;border-left:1px solid #eee;border-right:1px solid #eee;}
.bar-track{height:12px;border-radius:6px;background:linear-gradient(to right,#639922,#EF9F27,#E24B4A);position:relative;margin-bottom:6px;}
.bar-needle{position:absolute;top:-5px;width:4px;height:22px;background:#1a1a1a;border-radius:2px;transform:translateX(-50%);transition:left 0.8s ease;}
.bar-labels{display:flex;justify-content:space-between;font-size:11px;color:#aaa;}
.factors{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:1rem 1.5rem;background:white;border-left:1px solid #eee;border-right:1px solid #eee;}
.factor{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:10px;font-size:13px;font-weight:500;}
.f-bad{background:#FCEBEB;color:#791F1F;}
.f-ok{background:#EAF3DE;color:#27500A;}
.f-warn{background:#FAEEDA;color:#633806;}
.disclaimer{background:white;border-radius:0 0 16px 16px;border:1px solid #eee;border-top:none;padding:0.75rem 1.5rem;font-size:12px;color:#aaa;}
</style>
</head>
<body>
<nav>
  <div class="logo">
    <div class="logo-icon">&#10084;&#65039;</div>
    <div class="logo-name">HeartGuard</div>
  </div>
  <a href="/" class="back-link">&larr; Back to home</a>
</nav>
<div class="wrap">
  <div class="page-head">
    <h1>&#129657; Your health checkup</h1>
    <p>Fill in your details and get an instant AI-powered cardiovascular risk assessment</p>
  </div>
  <form id="form">
    <div class="card">
      <div class="sec-lab">&#128100; Personal information</div>
      <div class="grid3">
        <div class="field"><label>Your name</label><input type="text" id="name" placeholder="e.g. Anagh" required></div>
        <div class="field"><label>Age</label><input type="number" name="age" value="45" min="18" max="100"></div>
        <div class="field"><label>Sex</label><select name="sex"><option value="1">Male</option><option value="0">Female</option></select></div>
      </div>
      <div class="grid2" style="margin-top:12px;">
        <div class="field"><label>BMI</label><input type="number" name="bmi" value="24.5" step="0.1"></div>
        <div class="field"><label>BP medication</label>
          <div class="tog-row">
            <span class="tog on" data-group="bp_medication" data-val="0" onclick="pick(this)">No</span>
            <span class="tog" data-group="bp_medication" data-val="1" onclick="pick(this)">Yes</span>
          </div>
          <input type="hidden" name="bp_medication" value="0">
        </div>
      </div>
    </div>
    <div class="card">
      <div class="sec-lab">&#128200; Blood pressure & tests</div>
      <div class="grid2" style="margin-bottom:12px;">
        <div class="field"><label>Systolic BP (mmHg)</label><input type="number" name="systolic_bp" value="120"></div>
        <div class="field"><label>Diastolic BP (mmHg)</label><input type="number" name="diastolic_bp" value="80"></div>
      </div>
      <div class="grid2">
        <div class="field"><label>Cholesterol (mg/dL)</label><input type="number" name="cholesterol" value="200"></div>
        <div class="field"><label>Glucose (mg/dL)</label><input type="number" name="glucose" value="90"></div>
      </div>
    </div>
    <div class="card">
      <div class="sec-lab">&#127939; Lifestyle & history</div>
      <div class="grid2" style="gap:16px;">
        <div class="field"><label>Smoking</label>
          <div class="tog-row">
            <span class="tog on" data-group="smoking" data-val="0" onclick="pick(this)">No</span>
            <span class="tog" data-group="smoking" data-val="1" onclick="pick(this)">Yes</span>
          </div>
          <input type="hidden" name="smoking" value="0">
        </div>
        <div class="field"><label>Diabetes</label>
          <div class="tog-row">
            <span class="tog on" data-group="diabetes" data-val="0" onclick="pick(this)">No</span>
            <span class="tog" data-group="diabetes" data-val="1" onclick="pick(this)">Yes</span>
          </div>
          <input type="hidden" name="diabetes" value="0">
        </div>
        <div class="field"><label>Activity level</label>
          <div class="tog-row">
            <span class="tog" data-group="activity" data-val="0" onclick="pick(this)">Low</span>
            <span class="tog on" data-group="activity" data-val="1" onclick="pick(this)">Moderate</span>
            <span class="tog" data-group="activity" data-val="2" onclick="pick(this)">High</span>
          </div>
          <input type="hidden" name="activity" value="1">
        </div>
        <div class="field"><label>Alcohol use</label>
          <div class="tog-row">
            <span class="tog on" data-group="alcohol" data-val="0" onclick="pick(this)">No</span>
            <span class="tog" data-group="alcohol" data-val="1" onclick="pick(this)">Yes</span>
          </div>
          <input type="hidden" name="alcohol" value="0">
        </div>
        <div class="field"><label>Family history CVD</label>
          <div class="tog-row">
            <span class="tog on" data-group="family_history" data-val="0" onclick="pick(this)">No</span>
            <span class="tog" data-group="family_history" data-val="1" onclick="pick(this)">Yes</span>
          </div>
          <input type="hidden" name="family_history" value="0">
        </div>
      </div>
    </div>
    <button type="submit" class="assess-btn">&#129657; Assess My Risk</button>
  </form>
  <div class="result-wrap" id="result-wrap">
    <div class="result-top" id="result-top">
      <div>
        <div style="font-size:13px;margin-bottom:6px;" id="result-sub">10-year CVD risk score</div>
        <div class="result-score" id="score-display">--</div>
        <div style="font-size:13px;margin-top:4px;" id="advice-text"></div>
      </div>
      <div class="risk-pill" id="risk-pill"></div>
    </div>
    <div class="bar-section">
      <div class="bar-track"><div class="bar-needle" id="needle" style="left:0%"></div></div>
      <div class="bar-labels"><span>Low</span><span>Moderate</span><span>High</span></div>
    </div>
    <div class="factors" id="factors-grid"></div>
    <div class="disclaimer">&#9888;&#65039; Statistical estimate only — not a medical diagnosis. Always consult a doctor.</div>
  </div>
</div>
<script>
function pick(btn){
  const g=btn.dataset.group;
  document.querySelectorAll('[data-group="'+g+'"]').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelector('[name="'+g+'"]').value=btn.dataset.val;
}
document.getElementById('form').addEventListener('submit',async function(e){
  e.preventDefault();
  const name=document.getElementById('name').value||'Anonymous';
  const data={name};
  new FormData(e.target).forEach((v,k)=>{if(k!=='name')data[k]=parseFloat(v);});
  const btn=document.querySelector('.assess-btn');
  btn.textContent='Analysing...';
  btn.disabled=true;
  const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await res.json();
  btn.textContent='&#129657; Assess My Risk';
  btn.disabled=false;
  const prob=result.probability;
  const level=result.risk_level;
  const colors={Low:'#639922',Moderate:'#EF9F27',High:'#E24B4A'};
  const bgs={Low:'#EAF3DE',Moderate:'#FAEEDA',High:'#FCEBEB'};
  const textc={Low:'#27500A',Moderate:'#633806',High:'#791F1F'};
  const advice={Low:'Keep up your healthy habits!',Moderate:'Consider lifestyle improvements.',High:'Consult a cardiologist soon.'};
  document.getElementById('result-top').style.background=bgs[level];
  document.getElementById('score-display').style.color=textc[level];
  document.getElementById('score-display').textContent=prob+'%';
  document.getElementById('advice-text').style.color=textc[level];
  document.getElementById('advice-text').textContent=advice[level];
  const pill=document.getElementById('risk-pill');
  pill.textContent=level+' Risk';
  pill.style.background=colors[level];
  setTimeout(()=>{document.getElementById('needle').style.left=Math.min(prob,99)+'%';},100);
  const factors=[];
  if(data.smoking===1)factors.push({cls:'f-bad',icon:'&#128684;',text:'Smoking increases risk'});
  if(data.diabetes===1)factors.push({cls:'f-bad',icon:'&#129506;',text:'Diabetes raises CVD risk'});
  if(data.family_history===1)factors.push({cls:'f-warn',icon:'&#128106;',text:'Family history present'});
  if(data.alcohol===1)factors.push({cls:'f-warn',icon:'&#127867;',text:'Alcohol use noted'});
  if(data.activity>=2)factors.push({cls:'f-ok',icon:'&#127939;',text:'High activity is great'});
  if(data.activity===1)factors.push({cls:'f-ok',icon:'&#127939;',text:'Moderate activity helps'});
  if(data.bmi>30)factors.push({cls:'f-warn',icon:'&#9878;',text:'BMI above healthy range'});
  if(data.smoking===0&&data.diabetes===0)factors.push({cls:'f-ok',icon:'&#9989;',text:'No major risk factors'});
  document.getElementById('factors-grid').innerHTML=factors.slice(0,4).map(f=>'<div class="factor '+f.cls+'">'+f.icon+' '+f.text+'</div>').join('');
  const wrap=document.getElementById('result-wrap');
  wrap.style.display='block';
  wrap.scrollIntoView({behavior:'smooth',block:'start'});
});
</script>
</body>
</html>"""

@app.route('/')
def index():
    return HTML_LANDING

@app.route('/checkup')
def checkup():
    return HTML_CHECKUP

@app.route('/predict', methods=['POST'])
def predict():
    data        = request.get_json()
    features    = np.array([[data[f] for f in FEATURES]])
    features_sc = scaler.transform(features)
    probability = model.predict_proba(features_sc)[0][1]
    prob_pct    = round(float(probability) * 100, 1)
    risk_level  = 'High' if probability > 0.6 else 'Moderate' if probability > 0.35 else 'Low'
    return jsonify({'probability': prob_pct, 'risk_level': risk_level})