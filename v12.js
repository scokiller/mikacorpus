'use strict';

// Mika Corpus v0.12 — diversified corpus generalization gate.
// Train/validation use reference-lossless-v12.1. Full Canterbury benchmark remains sealed.
const V12_ENGINE='0.12-web.1-reference-generalization';
const V12_CORPUS_VERSION='reference-lossless-v12.1';
const V12_CORPUS_BASE='./corpus-v12';
const V12_PROFILE={key:'h192-e48-s128-b8',embed:48,hidden:192,seq:128,batch:8};
const V12_LR=3e-4; // v09CreateModel uses this exact Adam LR.
const V12_SEEDS=[0x19851201,0x19851202,0x19851203];
const V12_TOKEN_BUDGET=1500000;
const V12_VALID_FINAL_BATCHES=128;       // 131,072 held-out target bytes per validation.
const V12_VALID_TREND_BATCHES=32;
const V12_CHECKPOINT_EVERY=64;
const V12_CLOUD_EVERY=128;
const V12_STATE_KEY='v12:state';
const V12_DONE_KEY='v12:completed';
let v12Stop=false;
let v12ManifestText='';
let v12ManifestSha='';

function v12Log(s){try{log(`v0.12 · ${s}`);}catch{}}
function v12Status(t,d=''){try{setStatus(t,d);}catch{}}
function v12Mean(a){return a.length?a.reduce((x,y)=>x+y,0)/a.length:NaN;}
function v12Sd(a){if(a.length<2)return 0;const m=v12Mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/(a.length-1));}
function v12Steps(){return Math.ceil(V12_TOKEN_BUDGET/(V12_PROFILE.batch*V12_PROFILE.seq));}
function v12Fresh(){return {version:1,seedIndex:0,current:null,results:[],baselines:null,startedAt:new Date().toISOString()};}
async function v12LoadState(){return (await dbGet(V12_STATE_KEY))||v12Fresh();}
async function v12SaveState(s){await dbPut(V12_STATE_KEY,s);}

async function v12FetchCorpusFile(f){
  const url=`${V12_CORPUS_BASE}/${f.path.split('/').map(encodeURIComponent).join('/')}`;
  const ab=await fetch(url,{cache:'force-cache'}).then(r=>{if(!r.ok)throw new Error(`Corpus v0.12 HTTP ${r.status}: ${f.path}`);return r.arrayBuffer();});
  const h=await sha256Bytes(ab);if(h!==f.sha256)throw new Error(`SHA-256 corpus v0.12 invalide: ${f.path}`);
  return new Uint8Array(ab);
}

// Override legacy HEVC loader for v0.12 and later in this page session.
loadCorpus=async function v12LoadCorpus(){
  if(corpusCache?.manifest?.version===V12_CORPUS_VERSION)return corpusCache;
  corpusCache=null;
  v12Status('Corpus v0.12','Chargement du corpus généraliste et vérification SHA-256…');
  const r=await fetch(`${V12_CORPUS_BASE}/manifest.json`,{cache:'no-store'});if(!r.ok)throw new Error(`Manifest v0.12 HTTP ${r.status}`);
  v12ManifestText=await r.text();v12ManifestSha=await sha256Text(v12ManifestText);
  const manifest=JSON.parse(v12ManifestText);if(manifest.version!==V12_CORPUS_VERSION)throw new Error(`Version corpus inattendue: ${manifest.version}`);
  const usable=manifest.files.filter(f=>f.split==='train'||f.split==='validation');
  const groups={train:[],validation:[]};
  for(let i=0;i<usable.length;i++){
    const f=usable[i],bytes=await v12FetchCorpusFile(f);groups[f.split].push(bytes);pct(3+12*(i+1)/usable.length);await sleep(0);
  }
  corpusCache={manifest,manifestSha256:v12ManifestSha,train:concatBytes(groups.train),validation:concatBytes(groups.validation)};
  if(corpusCache.train.length!==manifest.counts.train.bytes||corpusCache.validation.length!==manifest.counts.validation.bytes)throw new Error('Taille corpus v0.12 incohérente après concaténation.');
  v12Log(`Corpus ${manifest.version} ✓ · train ${(corpusCache.train.length/1048576).toFixed(2)} MiB · validation ${(corpusCache.validation.length/1048576).toFixed(2)} MiB · Canterbury non chargé.`);
  return corpusCache;
};

// Hard gate: v0.12 is forbidden from touching the Canterbury benchmark.
loadBenchmark=async function v12SealedBenchmark(){throw new Error('Benchmark Canterbury scellé pendant v0.12.');};

function v12Counts(bytes){const c=new Float64Array(256);for(let i=0;i<bytes.length;i++)c[bytes[i]]++;return c;}
function v12EntropyFromCounts(c,n){let h=0;for(let i=0;i<256;i++)if(c[i]){const p=c[i]/n;h-=p*Math.log2(p);}return h;}
function v12CrossEntropy(trainCounts,trainN,valCounts,valN){const alpha=.5,den=trainN+alpha*256;let h=0;for(let i=0;i<256;i++)if(valCounts[i]){const p=(trainCounts[i]+alpha)/den;h-=valCounts[i]*Math.log2(p);}return h/valN;}
async function v12GzipBpb(bytes){
  if(typeof CompressionStream!=='function')return null;
  try{
    const cs=new CompressionStream('gzip');
    const writer=cs.writable.getWriter();await writer.write(bytes);await writer.close();
    const out=await new Response(cs.readable).arrayBuffer();return out.byteLength*8/bytes.length;
  }catch(e){v12Log(`baseline gzip indisponible: ${e.message}`);return null;}
}
async function v12Baselines(corpus){
  v12Status('v0.12 · Baselines','Mesure entropie zéro-ordre et gzip sur la validation, sans charger Canterbury…');
  const tc=v12Counts(corpus.train),vc=v12Counts(corpus.validation);
  const b={
    rawBpb:8,
    validationZeroOrderEntropyBpb:v12EntropyFromCounts(vc,corpus.validation.length),
    trainUnigramCrossEntropyBpb:v12CrossEntropy(tc,corpus.train.length,vc,corpus.validation.length),
    gzipValidationBpb:await v12GzipBpb(corpus.validation),
    trainBytes:corpus.train.length,
    validationBytes:corpus.validation.length,
    benchmarkBytes:corpus.manifest.counts.benchmark.bytes,
    manifestSha256:corpus.manifestSha256
  };
  v12Log(`Baselines · brut 8.0000 · H0 val ${b.validationZeroOrderEntropyBpb.toFixed(4)} · unigram train→val ${b.trainUnigramCrossEntropyBpb.toFixed(4)}${Number.isFinite(b.gzipValidationBpb)?` · gzip ${b.gzipValidationBpb.toFixed(4)}`:''} bpb.`);
  return b;
}

async function v12CloudCheckpoint(cur,snapshot){
  const payload=JSON.stringify({engine:V12_ENGINE,corpusVersion:V12_CORPUS_VERSION,seedIndex:cur.seedIndex,runId:cur.runId,step:cur.step,tokensSeen:cur.tokensSeen,sequence:cur.sequence,initialValidationBpb:cur.initialValidationBpb,snapshot});
  const hash=await sha256Text(payload),slot=cur.sequence%2;
  await api('/checkpoint',{method:'POST',body:JSON.stringify({runId:cur.runId,slot,sequence:cur.sequence,epoch:0,step:cur.step,globalStep:cur.step,sha256:hash,payload,stateMeta:{family:'v12-reference-generalization',seedIndex:cur.seedIndex,tokensSeen:cur.tokensSeen,corpusVersion:V12_CORPUS_VERSION}})});
}
async function v12Checkpoint(state,ctx,cloud=false){
  const cur=state.current;if(!cur)return;
  cur.snapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);cur.sequence=(cur.sequence||0)+1;cur.savedAt=new Date().toISOString();await v12SaveState(state);ui.sync.textContent=`Local v0.12 ✓ #${cur.sequence}`;
  if(cloud){try{await v12CloudCheckpoint(cur,cur.snapshot);ui.sync.textContent=`Cloud v0.12 ✓ #${cur.sequence}`;}catch(e){ui.sync.textContent='Local ✓ / cloud en retard';v12Log(`checkpoint cloud différé: ${e.message}`);}}
}
async function v12RestoreCloud(cur){
  if(!cur?.runId)return null;
  try{const r=await api(`/checkpoint/latest?runId=${encodeURIComponent(cur.runId)}`);if(!r.checkpoint?.payload)return null;const p=JSON.parse(r.checkpoint.payload);return p.engine===V12_ENGINE&&p.snapshot?p:null;}catch(e){v12Log(`reprise cloud indisponible: ${e.message}`);return null;}
}

async function v12RunSeed(state,seedIndex,corpus){
  const seed=V12_SEEDS[seedIndex]>>>0,total=v12Steps(),tps=V12_PROFILE.batch*V12_PROFILE.seq;
  let cur=state.current;
  if(!cur||cur.seedIndex!==seedIndex){
    const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:`v12-reference-h192-e48-s128-b8-s${seedIndex+1}`,engineVersion:V12_ENGINE,config:{profile:V12_PROFILE,seed,learningRate:V12_LR,tokenBudget:V12_TOKEN_BUDGET,totalSteps:total,validationTargetBytes:V12_VALID_FINAL_BATCHES*tps,corpusVersion:V12_CORPUS_VERSION,manifestSha256:corpus.manifestSha256,benchmarkSealed:true},progress:{stage:'training',seedIndex,tokensSeen:0}})});
    cur={seedIndex,runId:rr.run.id,step:0,tokensSeen:0,sequence:0,initialValidationBpb:null,snapshot:null};state.current=cur;await v12SaveState(state);
  }
  const ctx=v09CreateModel(V12_PROFILE,seed);let resumed=false;
  try{
    let snap=cur.snapshot;
    if(!snap){const cloud=await v12RestoreCloud(cur);if(cloud){snap=cloud.snapshot;cur.step=cloud.step||0;cur.tokensSeen=cloud.tokensSeen||0;cur.sequence=cloud.sequence||0;cur.initialValidationBpb=cloud.initialValidationBpb??cur.initialValidationBpb;}}
    if(snap){await v08RestoreSnapshot(ctx.model,ctx.opt,snap,true);resumed=true;v12Log(`seed ${seedIndex+1}: reprise étape ${cur.step}/${total}.`);}
    if(cur.initialValidationBpb==null){
      v12Status('v0.12 · Validation initiale',`Seed ${seedIndex+1}/3 · échantillon fixe de ${V12_VALID_FINAL_BATCHES*tps} octets cibles…`);
      cur.initialValidationBpb=await v09Validate(ctx,V12_PROFILE,corpus.validation,seed,V12_VALID_FINAL_BATCHES);await v12SaveState(state);v12Log(`seed ${seedIndex+1}: initial ${cur.initialValidationBpb.toFixed(5)} bpb.`);
    }
    const times=[];const marks=new Set([Math.floor(total*.25),Math.floor(total*.5),Math.floor(total*.75)]);
    for(let i=cur.step;i<total;i++){
      if(v12Stop){cur.step=i;await v12Checkpoint(state,ctx,true);await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'paused',step:i,global_step:i,progress:{stage:'paused',tokensSeen:cur.tokensSeen}})}).catch(()=>{});return {paused:true};}
      const t0=performance.now(),r=await v09Step(ctx,V12_PROFILE,corpus.train,i,seed);await tf.nextFrame();times.push(performance.now()-t0);
      cur.step=i+1;cur.tokensSeen=Math.min(V12_TOKEN_BUDGET,(i+1)*tps);
      ui.experiment.textContent=`Référence v0.12 · seed ${seedIndex+1}/3`;ui.epoch.textContent=`Seed ${seedIndex+1}/3`;ui.step.textContent=`${i+1}/${total}`;ui.train.textContent=fmt(r.bpb);
      pct(15+80*((seedIndex+(i+1)/total)/V12_SEEDS.length));
      if((i+1)%V12_CHECKPOINT_EVERY===0||i===total-1)await v12Checkpoint(state,ctx,((i+1)%V12_CLOUD_EVERY===0)||i===total-1);
      if(marks.has(i+1)){
        const vb=await v09Validate(ctx,V12_PROFILE,corpus.validation,seed,V12_VALID_TREND_BATCHES);ui.val.textContent=fmt(vb);
        await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v12_validation_trend_bpb',value:vb,globalStep:i+1,payload:{tokensSeen:cur.tokensSeen,seedIndex,validationBatches:V12_VALID_TREND_BATCHES}})}).catch(()=>{});
      }
    }
    v12Status('v0.12 · Validation finale',`Seed ${seedIndex+1}/3 · mesure large et fixe…`);
    const finalValidationBpb=await v09Validate(ctx,V12_PROFILE,corpus.validation,seed,V12_VALID_FINAL_BATCHES);
    const medianStepMs=times.length?[...times].sort((a,b)=>a-b)[Math.floor(times.length/2)]:null;
    const tokensPerSec=medianStepMs? tps/(medianStepMs/1000):null;
    const result={seedIndex,seed,runId:cur.runId,initialValidationBpb:cur.initialValidationBpb,finalValidationBpb,improvementBpb:cur.initialValidationBpb-finalValidationBpb,tokensSeen:cur.tokensSeen,totalSteps:total,medianStepMs,tokensPerSec,resumed};
    state.results=state.results.filter(x=>x.seedIndex!==seedIndex);state.results.push(result);state.current=null;await v12SaveState(state);
    await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v12_final_validation_bpb',value:finalValidationBpb,globalStep:total,payload:result})});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'completed',step:total,global_step:total,current_validation_bpb:finalValidationBpb,best_validation_bpb:finalValidationBpb,progress:{stage:'completed',tokensSeen:cur.tokensSeen,benchmarkLoaded:false},completed_at:new Date().toISOString()})});
    v12Log(`seed ${seedIndex+1}: ${cur.initialValidationBpb.toFixed(5)} → ${finalValidationBpb.toFixed(5)} bpb${tokensPerSec?` · ${Math.round(tokensPerSec)} tok/s`:''}.`);
    return {paused:false};
  }finally{v09Dispose(ctx);await tf.nextFrame();}
}

async function v12Run(){
  if(running)return;running=true;v12Stop=false;ui.start.disabled=true;ui.pause.disabled=false;
  try{
    await claimIfNeeded();await verifyBuild();await gpuGate();const locked=await requestWakeLock();if(!locked)throw new Error('Wake Lock non acquis');
    const corpus=await loadCorpus();let state=await v12LoadState();const done=await dbGet(V12_DONE_KEY);
    if(done){v12Status('v0.12 déjà terminé',`Validation moyenne ${done.summary.meanFinalValidationBpb.toFixed(4)} ± ${done.summary.sdFinalValidationBpb.toFixed(4)} bpb.`);pct(100);return;}
    if(!state.baselines){state.baselines=await v12Baselines(corpus);await v12SaveState(state);}
    for(let s=state.seedIndex;s<V12_SEEDS.length;s++){
      state=await v12LoadState();state.seedIndex=s;await v12SaveState(state);v12Status('v0.12 · Généralisation',`Seed ${s+1}/3 · 1,5 M tokens · corpus généraliste · benchmark Canterbury scellé.`);
      const rr=await v12RunSeed(state,s,corpus);if(rr.paused){v12Status('Pause sûre v0.12','Checkpoint modèle + Adam sauvegardé.');return;}
      state=await v12LoadState();state.seedIndex=s+1;await v12SaveState(state);
    }
    state=await v12LoadState();const rs=state.results.sort((a,b)=>a.seedIndex-b.seedIndex);if(rs.length!==V12_SEEDS.length)throw new Error(`Résultats incomplets: ${rs.length}/3.`);
    const finals=rs.map(x=>x.finalValidationBpb),imps=rs.map(x=>x.improvementBpb),speeds=rs.map(x=>x.tokensPerSec).filter(Number.isFinite);
    const summary={meanFinalValidationBpb:v12Mean(finals),sdFinalValidationBpb:v12Sd(finals),meanImprovementBpb:v12Mean(imps),sdImprovementBpb:v12Sd(imps),meanTokensPerSec:v12Mean(speeds),allSeedsBelowRaw:finals.every(x=>x<8),beatsTrainUnigramMean:v12Mean(finals)<state.baselines.trainUnigramCrossEntropyBpb};
    const report={engine:V12_ENGINE,createdAt:new Date().toISOString(),corpusVersion:V12_CORPUS_VERSION,manifestSha256:corpus.manifestSha256,profile:V12_PROFILE,learningRate:V12_LR,tokenBudgetPerSeed:V12_TOKEN_BUDGET,results:rs,baselines:state.baselines,summary,benchmark:{name:'Canterbury Corpus',files:corpus.manifest.counts.benchmark.files,bytes:corpus.manifest.counts.benchmark.bytes,loaded:false,sealed:true},note:'v0.12 measures generalization only. Canterbury benchmark bytes are never fetched by this runtime.'};
    await dbPut(V12_DONE_KEY,report);await dbPut(V12_STATE_KEY,null);
    const last=rs[rs.length-1].runId,payload=JSON.stringify(report),hash=await sha256Text(payload);
    await api('/artifact',{method:'POST',body:JSON.stringify({runId:last,kind:'reference-generalization-v12',payload,sha256:hash,promote:false,metadata:{meanFinalValidationBpb:summary.meanFinalValidationBpb,sd:summary.sdFinalValidationBpb,corpusVersion:V12_CORPUS_VERSION,benchmarkLoaded:false}})}).catch(()=>{});
    pct(100);ui.sync.textContent='Supabase ✓';ui.experiment.textContent='v0.12 référence';ui.val.textContent=fmt(summary.meanFinalValidationBpb);ui.best.textContent=`±${summary.sdFinalValidationBpb.toFixed(4)}`;ui.train.textContent=Number.isFinite(summary.meanTokensPerSec)?`${Math.round(summary.meanTokensPerSec)} tok/s`:'—';
    v12Status('v0.12 Généralisation terminée',`Validation ${summary.meanFinalValidationBpb.toFixed(4)} ± ${summary.sdFinalValidationBpb.toFixed(4)} bpb · amélioration moyenne ${summary.meanImprovementBpb.toFixed(4)} bpb · Canterbury toujours scellé.`);
  }catch(e){const msg=e instanceof Error?e.message:String(e);v12Status(v12Stop?'v0.12 en pause':'Arrêt sécurisé v0.12',msg);v12Log(`ERREUR: ${e?.stack||e}`);}
  finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v12Install(){
  corpusCache=null;
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;
  ui.start.textContent='Lancer / reprendre v0.12';ui.pause.textContent='Pause sûre';ui.start.addEventListener('click',v12Run);ui.pause.addEventListener('click',()=>{v12Stop=true;ui.pause.disabled=true;v12Status('Pause demandée v0.12','Fin du mini-batch puis checkpoint exact modèle + Adam.');});
  setTimeout(async()=>{try{const d=await dbGet(V12_DONE_KEY),st=await dbGet(V12_STATE_KEY);if(d?.summary){v12Status('v0.12 déjà terminé',`Corpus généraliste · ${d.summary.meanFinalValidationBpb.toFixed(4)} ± ${d.summary.sdFinalValidationBpb.toFixed(4)} bpb · Canterbury scellé.`);ui.experiment.textContent='v0.12 terminé';}else if(st?.current){v12Status('v0.12 reprise prête',`Seed ${st.current.seedIndex+1}/3 · étape ${st.current.step}/${v12Steps()} · corpus généraliste.`);ui.experiment.textContent='v0.12 reprise';}else{v12Status('v0.12 Généralisation prête','12,73 MiB train · 1,05 MiB validation · 3 seeds × 1,5 M tokens · Canterbury (11 fichiers) reste scellé.');ui.experiment.textContent='Référence généraliste';}}catch{}},1050);
})();
