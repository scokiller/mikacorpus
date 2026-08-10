'use strict';

const ENGINE = '0.7-web.2';
const PROTOCOL = 'MC-WEB-1';
const TFJS_VERSION = '4.22.0';
const LOCAL_CHECKPOINT_EVERY = 5;
const CLOUD_CHECKPOINT_EVERY = 10;
const API_BASE = 'https://xifxrkjvsrzexmuqtsvw.supabase.co/functions/v1/mikacorpus-api';
const CORPUS_BASE = 'https://xifxrkjvsrzexmuqtsvw.supabase.co/storage/v1/object/public/mikacorpus-public/corpus/chromium-hevc-v2';
const CLOUD_TIMEOUT_MS = 25000;
const HANDOFF_COOKIE = 'mikacorpus_handoff';
const EXPERIMENTS = [
  { key:'foundation-c96-h128', context:96, hidden:128, batch:64, stepsPerEpoch:120, maxEpochs:8, lr:0.025, patience:4, minGain:0.002 },
  { key:'context-c64-h160', context:64, hidden:160, batch:64, stepsPerEpoch:120, maxEpochs:8, lr:0.025, patience:4, minGain:0.002 },
  { key:'context-c128-h112', context:128, hidden:112, batch:64, stepsPerEpoch:120, maxEpochs:8, lr:0.020, patience:4, minGain:0.002 },
  { key:'context-c192-h96', context:192, hidden:96, batch:64, stepsPerEpoch:120, maxEpochs:8, lr:0.018, patience:4, minGain:0.002 },
];
const VALIDATION_BATCHES = 24;
const VALIDATION_SEED = 0x19850212;
const INIT_SEED = 0x1985c0de;

const $ = (id) => document.getElementById(id);
const ui = {
  status:$('status'), detail:$('detail'), gpu:$('gpu'), progress:$('progress'),
  experiment:$('experiment'), epoch:$('epoch'), step:$('step'), train:$('train'), val:$('val'), best:$('best'),
  sync:$('sync'), start:$('start'), pause:$('pause'), log:$('log'), install:$('install')
};
let stopRequested = false;
let running = false;
let wakeLock = null;
let corpusCache = null;

function log(msg) {
  const t = new Date().toLocaleTimeString('fr-FR');
  ui.log.textContent = `[${t}] ${msg}\n` + ui.log.textContent.slice(0, 5000);
}
function setStatus(title, detail='') { ui.status.textContent=title; ui.detail.textContent=detail; }
function pct(v){ ui.progress.style.width=`${Math.max(0,Math.min(100,v))}%`; }
function fmt(v){ return Number.isFinite(v) ? Number(v).toFixed(4) : '—'; }
function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }

async function sha256Bytes(bytes) {
  const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return [...d].map(x=>x.toString(16).padStart(2,'0')).join('');
}
async function sha256Text(s){ return sha256Bytes(new TextEncoder().encode(s)); }
function bytesToB64(bytes) {
  let out=''; const CH=0x8000;
  for(let i=0;i<bytes.length;i+=CH) out += String.fromCharCode(...bytes.subarray(i, Math.min(bytes.length,i+CH)));
  return btoa(out);
}
function b64ToBytes(s) {
  const raw=atob(s), out=new Uint8Array(raw.length);
  for(let i=0;i<raw.length;i++) out[i]=raw.charCodeAt(i);
  return out;
}

function openDB(){
  return new Promise((resolve,reject)=>{
    const r=indexedDB.open('mikacorpus-mcweb1',1);
    r.onupgradeneeded=()=>{ const db=r.result; if(!db.objectStoreNames.contains('kv')) db.createObjectStore('kv'); };
    r.onsuccess=()=>resolve(r.result); r.onerror=()=>reject(r.error);
  });
}
async function dbGet(key){ const db=await openDB(); return new Promise((res,rej)=>{ const tx=db.transaction('kv','readonly'); const r=tx.objectStore('kv').get(key); r.onsuccess=()=>res(r.result); r.onerror=()=>rej(r.error); }); }
async function dbPut(key,val){ const db=await openDB(); return new Promise((res,rej)=>{ const tx=db.transaction('kv','readwrite'); tx.objectStore('kv').put(val,key); tx.oncomplete=()=>res(); tx.onerror=()=>rej(tx.error); }); }
async function dbCheckpoint(slot, payloadObj, sequence){
  const db=await openDB();
  return new Promise((res,rej)=>{
    const tx=db.transaction('kv','readwrite'); const s=tx.objectStore('kv');
    s.put({sequence,payload:payloadObj},`checkpoint:${slot}`); s.put({slot,sequence},'checkpoint:pointer');
    tx.oncomplete=()=>res(); tx.onerror=()=>rej(tx.error);
  });
}
async function dbLatestCheckpoint(){
  const p=await dbGet('checkpoint:pointer'); if(!p) return null;
  const a=await dbGet(`checkpoint:${p.slot}`); return a?.payload || null;
}

async function api(path, opts={}) {
  const capability = await dbGet('capability');
  const headers = {'content-type':'application/json',...(opts.headers||{})};
  if(capability) headers['x-mika-capability']=capability;
  const ctrl=new AbortController(); const timer=setTimeout(()=>ctrl.abort(),CLOUD_TIMEOUT_MS);
  try {
    const r=await fetch(`${API_BASE}${path}`,{...opts,headers,signal:ctrl.signal,cache:'no-store'});
    const text=await r.text(); let data; try{data=JSON.parse(text)}catch{data={raw:text}};
    if(!r.ok) throw new Error(`${r.status} ${data?.error||r.statusText}`);
    return data;
  } finally { clearTimeout(timer); }
}

function readHandoffCookie(){
  const prefix=HANDOFF_COOKIE+'=';
  for(const part of document.cookie.split(';')){ const v=part.trim(); if(v.startsWith(prefix)) return decodeURIComponent(v.slice(prefix.length)); }
  return null;
}
function writeHandoffCookie(cap){ document.cookie=`${HANDOFF_COOKIE}=${encodeURIComponent(cap)}; Path=/; Max-Age=900; SameSite=Strict; Secure`; }
function clearHandoffCookie(){ document.cookie=`${HANDOFF_COOKIE}=; Path=/; Max-Age=0; SameSite=Strict; Secure`; }
async function restoreCapability(){
  let cap=await dbGet('capability'); if(cap) return {cap,source:'indexeddb'};
  cap=readHandoffCookie();
  if(cap){
    await dbPut('capability',cap); clearHandoffCookie();
    log('Secret d’association transféré vers le stockage privé de la PWA.');
    return {cap,source:'install-cookie'};
  }
  return null;
}
async function claimIfNeeded(){
  const restored=await restoreCapability(); if(restored) return {cap:restored.cap,newlyClaimed:false};
  const q=new URLSearchParams(location.search); const bootstrap=q.get('claim');
  if(!bootstrap) throw new Error('Ce PWA n’est pas encore associé à cet iPhone. Ouvre le lien de claim fourni.');
  let install=await dbGet('installationId');
  if(!install){ install=crypto.randomUUID(); await dbPut('installationId',install); }
  setStatus('Association sécurisée','Enregistrement de cet iPhone comme appareil de calcul…');
  const r=await api('/claim',{method:'POST',body:JSON.stringify({bootstrap,installationId:install,label:`iPhone PWA ${navigator.platform||''}`})});
  await dbPut('capability',r.capability); await dbPut('deviceId',r.deviceId); writeHandoffCookie(r.capability);
  history.replaceState(null,'',location.pathname+location.hash);
  log('iPhone associé. Un cookie de transfert valable 15 minutes est prêt pour l’installation PWA.');
  return {cap:r.capability,newlyClaimed:true};
}

async function verifyBuild(){
  const b=await fetch('./build.json',{cache:'no-store'}).then(r=>r.json());
  const app=await fetch('./app.js',{cache:'no-store'}).then(r=>r.arrayBuffer());
  const actual=await sha256Bytes(app);
  if(actual!==b.app_sha256) throw new Error('Intégrité app.js refusée (SHA-256 différent).');
  return b;
}

async function gpuGate(){
  if(!('gpu' in navigator)) throw new Error('WebGPU absent dans Safari/iOS. Aucun fallback CPU ne sera utilisé.');
  if(!window.tf) throw new Error('TensorFlow.js non chargé.');
  if(typeof tf.setBackend!=='function' || typeof tf.ready!=='function' || typeof tf.tensor2d!=='function') {
    throw new Error(`TensorFlow.js incomplet (setBackend=${typeof tf.setBackend}, ready=${typeof tf.ready}, tensor2d=${typeof tf.tensor2d}). Boot=${JSON.stringify(globalThis.__mikaTfBoot||{}).slice(0,1200)}`);
  }
  try { if(tf.env().getFlags().WEBGPU_CPU_FORWARD !== undefined) tf.env().set('WEBGPU_CPU_FORWARD',false); } catch {}
  let ok=false;
  try { ok=await tf.setBackend('webgpu'); await tf.ready(); }
  catch(e){ throw new Error(`Initialisation TFJS WebGPU impossible: ${e instanceof Error?e.message:String(e)}`); }
  if(!ok || tf.getBackend()!=='webgpu') throw new Error(`Backend refusé: ${tf.getBackend()}`);
  const a=tf.tensor2d([1,2,3,4],[2,2]); const b=tf.tensor2d([2,0,0,2],[2,2]); const c=a.matMul(b); const cd=await c.data();
  if(cd[0]!==2 || cd[3]!==8) throw new Error('Matmul WebGPU incohérent.');
  a.dispose();b.dispose();c.dispose();
  const v=tf.variable(tf.tensor2d([1,2,3,4],[2,2]),true,'gateVar');
  const before=(await v.data())[0];
  const vg=tf.variableGrads(()=>v.square().sum(),[v]); const g=vg.grads[v.name];
  const updated=tf.tidy(()=>v.sub(g.mul(0.01))); v.assign(updated); updated.dispose(); vg.value.dispose(); g.dispose();
  const after=(await v.data())[0]; v.dispose();
  if(!(after < before)) throw new Error('Mise à jour de gradient WebGPU non fonctionnelle.');
  ui.gpu.textContent=`WEBGPU ✓ · TFJS ${tf.version.tfjs} · ${navigator.userAgent.match(/OS [^ )]+/)?.[0]||'iOS'}`;
  log('Gate GPU validé: backend webgpu + matmul + gradient SGD.');
}

function concatBytes(list){ const n=list.reduce((sum,a)=>sum+a.length,0); const out=new Uint8Array(n); let p=0; for(const a of list){out.set(a,p);p+=a.length;} return out; }
async function fetchCorpusFile(f){
  const ab=await fetch(`${CORPUS_BASE}/${encodeURIComponent(f.name)}`,{cache:'force-cache'}).then(r=>{if(!r.ok)throw new Error(`Corpus HTTP ${r.status}: ${f.name}`);return r.arrayBuffer()});
  const h=await sha256Bytes(ab); if(h!==f.sha256) throw new Error(`SHA corpus invalide: ${f.name}`);
  return new Uint8Array(ab);
}
async function loadCorpus(){
  if(corpusCache) return corpusCache;
  setStatus('Corpus','Téléchargement et vérification SHA-256 du corpus H.265 v2…');
  const manifest=await fetch(`${CORPUS_BASE}/manifest.json`,{cache:'no-store'}).then(r=>{if(!r.ok)throw new Error(`Manifest corpus HTTP ${r.status}`);return r.json()});
  const groups={train:[],validation:[]};
  const usable=manifest.files.filter(f=>f.split!=='benchmark');
  for(let i=0;i<usable.length;i++){
    const f=usable[i]; const bytes=await fetchCorpusFile(f); groups[f.split].push(bytes); pct(5+20*(i+1)/usable.length);
  }
  corpusCache={manifest,train:concatBytes(groups.train),validation:concatBytes(groups.validation)};
  log(`Corpus vérifié: ${manifest.counts.train} train (${corpusCache.train.length} octets) + ${manifest.counts.validation} validation (${corpusCache.validation.length} octets); benchmark non chargé.`);
  return corpusCache;
}
async function loadBenchmark(){
  const corp=await loadCorpus(); const files=corp.manifest.files.filter(f=>f.split==='benchmark'); const chunks=[];
  setStatus('Benchmark final','Chargement du jeu scellé, jamais utilisé pour les gradients ni l’early stopping…');
  for(let i=0;i<files.length;i++){chunks.push(await fetchCorpusFile(files[i]));pct(82+8*(i+1)/files.length);}
  return {bytes:concatBytes(chunks),files};
}

function mix32(x){ x|=0; x=Math.imul(x^(x>>>16),0x21f0aaad); x=Math.imul(x^(x>>>15),0x735a2d97); return (x^(x>>>15))>>>0; }
function makeBatch(bytes,cfg,globalStep,seed){
  const max=Math.max(1,bytes.length-cfg.context-1), xs=new Float32Array(cfg.batch*cfg.context), ys=new Int32Array(cfg.batch);
  for(let b=0;b<cfg.batch;b++){
    const start=mix32((seed ^ Math.imul((globalStep+1),0x9e3779b1) ^ Math.imul(b+1,0x85ebca6b))>>>0)%max;
    const off=b*cfg.context; for(let j=0;j<cfg.context;j++) xs[off+j]=(bytes[start+j]-127.5)/127.5;
    ys[b]=bytes[start+cfg.context];
  }
  return {xs,ys};
}
function seededFloats(n,seed,scale){
  const out=new Float32Array(n); let s=seed>>>0;
  for(let i=0;i<n;i++){ s=mix32(s+i+0x9e3779b9); out[i]=(((s/4294967296)*2)-1)*scale; }
  return out;
}
function createModel(cfg, serialized=null){
  const make=(name,shape,data)=>tf.variable(tf.tensor(data,shape,'float32'),true,name);
  if(serialized){
    const d=serialized.weights; return {
      W1:make('W1',[cfg.context,cfg.hidden],new Float32Array(b64ToBytes(d.W1).buffer)),
      b1:make('b1',[cfg.hidden],new Float32Array(b64ToBytes(d.b1).buffer)),
      W2:make('W2',[cfg.hidden,256],new Float32Array(b64ToBytes(d.W2).buffer)),
      b2:make('b2',[256],new Float32Array(b64ToBytes(d.b2).buffer)),
    };
  }
  const s=INIT_SEED ^ cfg.context ^ (cfg.hidden<<8);
  return {
    W1:make('W1',[cfg.context,cfg.hidden],seededFloats(cfg.context*cfg.hidden,s,Math.sqrt(2/cfg.context))),
    b1:make('b1',[cfg.hidden],new Float32Array(cfg.hidden)),
    W2:make('W2',[cfg.hidden,256],seededFloats(cfg.hidden*256,s^0xa5a5a5a5,Math.sqrt(2/cfg.hidden))),
    b2:make('b2',[256],new Float32Array(256)),
  };
}
function disposeModel(m){ Object.values(m).forEach(v=>v.dispose()); }
function forward(m,x){ return tf.tidy(()=>x.matMul(m.W1).add(m.b1).relu().matMul(m.W2).add(m.b2)); }
async function serializeModel(m,cfg){
  const w={}; for(const k of ['W1','b1','W2','b2']){ const d=await m[k].data(); w[k]=bytesToB64(new Uint8Array(d.buffer,d.byteOffset,d.byteLength)); }
  return {format:'mc-dense-f32-v1',context:cfg.context,hidden:cfg.hidden,weights:w};
}
async function cloneSerialized(s){ return JSON.parse(JSON.stringify(s)); }

async function trainStep(m,cfg,bytes,globalStep){
  const {xs,ys}=makeBatch(bytes,cfg,globalStep,0x51c0ffee); const x=tf.tensor2d(xs,[cfg.batch,cfg.context]); const y=tf.tensor1d(ys,'int32'); const oh=tf.oneHot(y,256);
  const vars=[m.W1,m.b1,m.W2,m.b2];
  const vg=tf.variableGrads(()=>{ const logits=forward(m,x); const loss=tf.losses.softmaxCrossEntropy(oh,logits).mean(); logits.dispose(); return loss; },vars);
  const loss=(await vg.value.data())[0];
  for(const v of vars){ const g=vg.grads[v.name]; const nv=tf.tidy(()=>v.sub(g.mul(cfg.lr))); v.assign(nv); nv.dispose(); g.dispose(); }
  vg.value.dispose(); x.dispose(); y.dispose(); oh.dispose();
  return loss/Math.LN2;
}
async function validationBpb(m,cfg,bytes){
  let total=0;
  for(let i=0;i<VALIDATION_BATCHES;i++){
    const {xs,ys}=makeBatch(bytes,cfg,i,VALIDATION_SEED); const x=tf.tensor2d(xs,[cfg.batch,cfg.context]); const y=tf.tensor1d(ys,'int32'); const oh=tf.oneHot(y,256); const logits=forward(m,x); const loss=tf.losses.softmaxCrossEntropy(oh,logits).mean();
    total+=(await loss.data())[0]/Math.LN2; x.dispose();y.dispose();oh.dispose();logits.dispose();loss.dispose();
    if(i%4===0) await tf.nextFrame();
  }
  return total/VALIDATION_BATCHES;
}

async function cloudCheckpoint(state){
  const payload=JSON.stringify(state); const hash=await sha256Text(payload); const slot=state.sequence%2;
  ui.sync.textContent='Cloud…';
  const r=await api('/checkpoint',{method:'POST',body:JSON.stringify({runId:state.runId,slot,sequence:state.sequence,epoch:state.epoch,step:state.step,globalStep:state.globalStep,sha256:hash,payload,stateMeta:{experimentIndex:state.experimentIndex,bestBpb:state.bestBpb}})});
  ui.sync.textContent=`Cloud ✓ #${r.sequence}`;
}
async function saveCheckpoint(state,syncCloud=true){
  const slot=state.sequence%2; await dbCheckpoint(slot,state,state.sequence); await dbPut('currentRunId',state.runId); await dbPut('experimentIndex',state.experimentIndex);
  ui.sync.textContent=`Local ✓ #${state.sequence}`;
  if(syncCloud){ try{ await cloudCheckpoint(state); }catch(e){ ui.sync.textContent='Local ✓ / cloud en retard'; log(`Cloud checkpoint différé: ${e.message}`); } }
}
async function recoverCheckpoint(){
  const local=await dbLatestCheckpoint(); const runId=local?.runId || await dbGet('currentRunId'); if(!runId) return local;
  try{
    const r=await api(`/checkpoint/latest?runId=${encodeURIComponent(runId)}`); const cloud=r.checkpoint?.payload ? JSON.parse(r.checkpoint.payload) : null;
    if(cloud && (!local || cloud.sequence>local.sequence)){ log(`Reprise depuis checkpoint cloud #${cloud.sequence}.`); return cloud; }
  }catch(e){ log(`Checkpoint cloud indisponible, reprise locale: ${e.message}`); }
  return local;
}

async function requestWakeLock(){ try{ wakeLock=await navigator.wakeLock?.request('screen'); log('Écran maintenu actif pendant le calcul.'); }catch{} }
async function releaseWakeLock(){ try{await wakeLock?.release();}catch{} wakeLock=null; }

document.addEventListener('visibilitychange',async()=>{
  if(document.visibilityState==='visible' && running) await requestWakeLock();
});

async function runExperiment(expIndex, recovered=null){
  const cfg=EXPERIMENTS[expIndex]; ui.experiment.textContent=cfg.key; setStatus('Préparation',`Expérience ${expIndex+1}/${EXPERIMENTS.length}`);
  const corp=await loadCorpus();
  let state=recovered && recovered.experimentIndex===expIndex ? recovered : null;
  let m, runId;
  if(state){
    m=createModel(cfg,state.currentModel); runId=state.runId; log(`État restauré: époque ${state.epoch+1}, étape ${state.step}, checkpoint #${state.sequence}.`);
  }else{
    m=createModel(cfg); const initial=await validationBpb(m,cfg,corp.validation); const ser=await serializeModel(m,cfg);
    const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:cfg.key,engineVersion:ENGINE,config:cfg,progress:{initialValidationBpb:initial,corpus:corp.manifest.canonical_sha256}})}); runId=rr.run.id;
    state={version:1,engine:ENGINE,protocol:PROTOCOL,experimentIndex:expIndex,runId,epoch:0,step:0,globalStep:0,sequence:0,runningLossSum:0,runningLossCount:0,bestBpb:initial,patience:0,currentModel:ser,bestModel:await cloneSerialized(ser),initialBpb:initial};
    await saveCheckpoint(state,true);
    await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'initial_validation_bpb',value:initial,epoch:0,step:0,globalStep:0})});
    const baselinePayload=JSON.stringify({engine:ENGINE,protocol:PROTOCOL,experiment:cfg.key,validationBpb:initial,model:state.bestModel,corpusManifestSha256:corp.manifest.canonical_sha256,baseline:true});
    await api('/artifact',{method:'POST',body:JSON.stringify({runId,kind:'champion',payload:baselinePayload,sha256:await sha256Text(baselinePayload),promote:true,validationBpb:initial,metadata:{epoch:0,context:cfg.context,hidden:cfg.hidden,baseline:true}})});
    log(`Baseline ${cfg.key}: ${initial.toFixed(4)} bpb, sauvegardée comme champion anti-régression.`);
  }

  ui.best.textContent=fmt(state.bestBpb);
  for(let epoch=state.epoch;epoch<cfg.maxEpochs;epoch++){
    let startStep=(epoch===state.epoch?state.step:0); if(epoch!==state.epoch){state.runningLossSum=0;state.runningLossCount=0;}
    for(let step=startStep;step<cfg.stepsPerEpoch;step++){
      if(stopRequested){
        state.epoch=epoch;state.step=step;state.currentModel=await serializeModel(m,cfg);state.sequence++; await saveCheckpoint(state);
        await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'paused',epoch,step,global_step:state.globalStep,progress:{reason:'user pause'}})}).catch(()=>{});
        disposeModel(m); return {paused:true,state};
      }
      const loss=await trainStep(m,cfg,corp.train,state.globalStep); state.runningLossSum+=loss;state.runningLossCount++;state.globalStep++;state.epoch=epoch;state.step=step+1;
      ui.epoch.textContent=`${epoch+1}/${cfg.maxEpochs}`; ui.step.textContent=`${step+1}/${cfg.stepsPerEpoch}`; ui.train.textContent=fmt(state.runningLossSum/state.runningLossCount); pct(25+55*((epoch*cfg.stepsPerEpoch+step+1)/(cfg.maxEpochs*cfg.stepsPerEpoch)));
      if(state.globalStep%LOCAL_CHECKPOINT_EVERY===0 || step===cfg.stepsPerEpoch-1){ state.currentModel=await serializeModel(m,cfg);state.sequence++;await saveCheckpoint(state,(state.globalStep%CLOUD_CHECKPOINT_EVERY===0)||step===cfg.stepsPerEpoch-1); }
      if(step%4===0) await tf.nextFrame();
    }
    setStatus('Validation',`Époque ${epoch+1} · données fixes jamais utilisées pour les gradients`);
    const vb=await validationBpb(m,cfg,corp.validation); const trainBpb=state.runningLossSum/Math.max(1,state.runningLossCount); ui.val.textContent=fmt(vb);
    let promoted=false;
    if(vb < state.bestBpb-cfg.minGain){ state.bestBpb=vb; state.bestModel=await serializeModel(m,cfg); state.patience=0; promoted=true; ui.best.textContent=fmt(vb); }
    else state.patience++;
    await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'validation_bpb',value:vb,epoch:epoch+1,step:cfg.stepsPerEpoch,globalStep:state.globalStep,payload:{trainBpb,promoted,patience:state.patience}})}).catch(()=>{});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'training',epoch:epoch+1,step:0,global_step:state.globalStep,best_validation_bpb:state.bestBpb,current_validation_bpb:vb,train_bpb:trainBpb,progress:{patience:state.patience,promoted}})}).catch(()=>{});
    if(promoted){
      const p=JSON.stringify({engine:ENGINE,protocol:PROTOCOL,experiment:cfg.key,validationBpb:vb,model:state.bestModel,corpusManifestSha256:corp.manifest.canonical_sha256}); const h=await sha256Text(p);
      await api('/artifact',{method:'POST',body:JSON.stringify({runId,kind:'champion',payload:p,sha256:h,promote:true,validationBpb:vb,metadata:{epoch:epoch+1,context:cfg.context,hidden:cfg.hidden}})});
      log(`Champion promu: ${vb.toFixed(4)} bpb.`);
    }
    state.epoch=epoch+1;state.step=0;state.runningLossSum=0;state.runningLossCount=0;state.currentModel=await serializeModel(m,cfg);state.sequence++;await saveCheckpoint(state);
    if(state.patience>=cfg.patience){ log(`Early stopping: ${state.patience} validations sans promotion.`); break; }
  }
  await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'completed',epoch:state.epoch,step:state.step,global_step:state.globalStep,best_validation_bpb:state.bestBpb,completed_at:new Date().toISOString()})}).catch(()=>{});
  disposeModel(m); return {paused:false,state};
}

async function benchmarkChampion(){
  const r=await api('/model'); if(!r.champion){log('Aucun champion disponible pour le benchmark.');return null;}
  const artifact=JSON.parse(r.champion.payload); const cfg=EXPERIMENTS.find(x=>x.key===artifact.experiment); if(!cfg) throw new Error(`Configuration champion inconnue: ${artifact.experiment}`);
  const b=await loadBenchmark(); const m=createModel(cfg,artifact.model); const bpb=await validationBpb(m,cfg,b.bytes); disposeModel(m);
  const report={engine:ENGINE,protocol:PROTOCOL,experiment:artifact.experiment,validationBpb:artifact.validationBpb,benchmarkBpb:bpb,benchmarkFiles:b.files.map(f=>({name:f.name,sha256:f.sha256,bytes:f.bytes})),corpusManifestSha256:(await loadCorpus()).manifest.canonical_sha256,createdAt:new Date().toISOString()};
  const payload=JSON.stringify(report); const sha=await sha256Text(payload);
  await api('/metric',{method:'POST',body:JSON.stringify({runId:r.champion.run_id,name:'benchmark_bpb',value:bpb,payload:{files:b.files.length}})}).catch(()=>{});
  await api('/artifact',{method:'POST',body:JSON.stringify({runId:r.champion.run_id,kind:'benchmark-report',payload,sha256:sha,metadata:{benchmarkBpb:bpb,files:b.files.length}})});
  log(`Benchmark final champion ${artifact.experiment}: ${bpb.toFixed(4)} bpb sur ${b.files.length} fichiers.`); return report;
}

async function startAutopilot(){
  if(running) return; running=true; stopRequested=false; ui.start.disabled=true; ui.pause.disabled=false;
  try{
    const pairing=await claimIfNeeded();
    if(pairing.newlyClaimed){
      ui.start.textContent='Lancer / reprendre';
      setStatus('iPhone associé','Maintenant: Partager → Sur l’écran d’accueil. Ouvre ensuite MikaCorpus depuis son icône et touche « Lancer / reprendre ».');
      return;
    }
    await verifyBuild(); await gpuGate(); await requestWakeLock();
    const h=await api('/health'); log(`Backend: ${h.storage}, ${h.checkpoint_scheme}.`);
    let recovered=await recoverCheckpoint(); let expIndex=recovered?.experimentIndex ?? (await dbGet('experimentIndex') ?? 0);
    while(expIndex<EXPERIMENTS.length && !stopRequested){
      const r=await runExperiment(expIndex,recovered); if(r.paused) break;
      expIndex++; recovered=null; await dbPut('experimentIndex',expIndex); await dbPut('currentRunId',null);
    }
    if(expIndex>=EXPERIMENTS.length){ const report=await benchmarkChampion(); setStatus('Cycle terminé',report?`Champion benchmark: ${report.benchmarkBpb.toFixed(4)} bpb. Tout est sauvegardé dans Supabase.`:'Toutes les expériences prévues ont été exécutées. Le meilleur champion reste conservé dans Supabase.'); pct(100); }
    else if(stopRequested) setStatus('En pause','Checkpoint enregistré. Relance plus tard: la reprise repartira du dernier état valide.');
  }catch(e){ setStatus('Arrêt sécurisé',e.message); log(`ERREUR: ${e.stack||e}`); }
  finally{ running=false; ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock(); }
}
ui.start.addEventListener('click',startAutopilot);
ui.pause.addEventListener('click',()=>{stopRequested=true;ui.pause.disabled=true;setStatus('Pause demandée','Je termine le mini-batch puis j’écris un checkpoint avant de m’arrêter.');});

let deferredPrompt=null;
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();deferredPrompt=e;ui.install.hidden=false;});
ui.install.addEventListener('click',async()=>{if(deferredPrompt){deferredPrompt.prompt();deferredPrompt=null;ui.install.hidden=true;}});

(async function boot(){
  try{
    if('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js',{scope:'./'});
    const b=await verifyBuild(); ui.detail.textContent=`Build ${b.engine} · protocole ${b.protocol}`;
    const existing=await restoreCapability();
    if(existing) setStatus('Prêt','Appuie sur « Lancer / reprendre ».');
    else if(new URLSearchParams(location.search).get('claim')) {
      ui.start.textContent='Associer cet iPhone';
      setStatus('Prêt à associer','Ouvre impérativement ce lien dans Safari, puis touche « Associer cet iPhone ».');
    } else setStatus('Lien de claim requis','Utilise le lien privé fourni pour la première ouverture.');
  }catch(e){setStatus('Intégrité refusée',e.message);}
})();

// MikaCorpus runtime revision R5 - CSP TensorFlow eval compatibility 2026-08-07T10:15:49.523Z

// MikaCorpus runtime revision R5 - CSP TensorFlow eval compatibility 2026-08-07T10:16:29.410Z
