'use strict';

// Mika Corpus v0.10 — equal-token tournament on profiles proven stable by v0.9.2.
// Goal: distinguish raw throughput from real held-out learning under the same token budget.
const V10_ENGINE='0.10-web.1-equal-token-tournament';
const V10_TOKEN_BUDGET=1000000;
const V10_SEEDS=[0x19851001,0x19851002];
const V10_VALID_BATCHES=12;
const V10_CHECKPOINT_EVERY=64;
const V10_CLOUD_EVERY=128;
const V10_STATE_KEY='v10:state';
const V10_DONE_KEY='v10:completed';
const V10_PROFILES=[
  {key:'h192-e48-s128-b8',embed:48,hidden:192,seq:128,batch:8},
  {key:'h256-e64-s160-b12',embed:64,hidden:256,seq:160,batch:12},
  {key:'h256-e64-s192-b16',embed:64,hidden:256,seq:192,batch:16}
];
let v10Stop=false;

function v10Log(s){try{log(`v0.10 · ${s}`);}catch{}}
function v10Status(t,d=''){try{setStatus(t,d);}catch{}}
function v10Steps(cfg){return Math.ceil(V10_TOKEN_BUDGET/(cfg.batch*cfg.seq));}
function v10Median(a){const x=[...a].sort((p,q)=>p-q);return x.length?x[Math.floor(x.length/2)]:NaN;}
function v10Fresh(){return {version:1,profileIndex:0,seedIndex:0,current:null,results:[],startedAt:new Date().toISOString()};}
async function v10Load(){return (await dbGet(V10_STATE_KEY))||v10Fresh();}
async function v10Save(s){await dbPut(V10_STATE_KEY,s);}

async function v10CloudCheckpoint(current,snapshot){
  const payload=JSON.stringify({engine:V10_ENGINE,profileIndex:current.profileIndex,seedIndex:current.seedIndex,runId:current.runId,step:current.step,sequence:current.sequence,initialValidationBpb:current.initialValidationBpb,tokensSeen:current.tokensSeen,snapshot});
  const hash=await sha256Text(payload);const slot=current.sequence%2;
  await api('/checkpoint',{method:'POST',body:JSON.stringify({runId:current.runId,slot,sequence:current.sequence,epoch:0,step:current.step,globalStep:current.step,sha256:hash,payload,stateMeta:{family:'v10-equal-token',profile:current.profileKey,seedIndex:current.seedIndex,tokensSeen:current.tokensSeen}})});
}

async function v10Checkpoint(state,ctx,cloud=false){
  const cur=state.current;if(!cur)return;
  const snapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);
  cur.snapshot=snapshot;cur.sequence=(cur.sequence||0)+1;cur.savedAt=new Date().toISOString();
  await v10Save(state);ui.sync.textContent=`Local v0.10 ✓ #${cur.sequence}`;
  if(cloud){
    try{await v10CloudCheckpoint(cur,snapshot);ui.sync.textContent=`Cloud v0.10 ✓ #${cur.sequence}`;}
    catch(e){ui.sync.textContent='Local ✓ / cloud en retard';v10Log(`checkpoint cloud différé: ${e.message}`);}
  }
}

async function v10RestoreFromCloud(cur){
  if(!cur?.runId)return null;
  try{
    const r=await api(`/checkpoint/latest?runId=${encodeURIComponent(cur.runId)}`);
    if(!r.checkpoint?.payload)return null;
    const p=JSON.parse(r.checkpoint.payload);
    if(p.engine!==V10_ENGINE||!p.snapshot)return null;
    return p;
  }catch(e){v10Log(`checkpoint cloud indisponible: ${e.message}`);return null;}
}

async function v10RunOne(state,cfg,profileIndex,seedIndex,corpus){
  const seed=V10_SEEDS[seedIndex]>>>0;const totalSteps=v10Steps(cfg);const tokensPerStep=cfg.batch*cfg.seq;
  let cur=state.current;
  if(!cur||cur.profileIndex!==profileIndex||cur.seedIndex!==seedIndex){
    const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:`v10-${cfg.key}-seed${seedIndex+1}`,engineVersion:V10_ENGINE,config:{profile:cfg,seed,tokenBudget:V10_TOKEN_BUDGET,totalSteps,validationBatches:V10_VALID_BATCHES,equalTokenBudget:true},progress:{stage:'training',profileIndex,seedIndex}})});
    cur={profileIndex,seedIndex,profileKey:cfg.key,runId:rr.run.id,step:0,tokensSeen:0,sequence:0,initialValidationBpb:null,snapshot:null};state.current=cur;await v10Save(state);
  }

  const ctx=v09CreateModel(cfg,seed);let restored=false;
  try{
    let snap=cur.snapshot;
    if(!snap){const cloud=await v10RestoreFromCloud(cur);if(cloud){snap=cloud.snapshot;cur.step=cloud.step||0;cur.tokensSeen=cloud.tokensSeen||0;cur.sequence=cloud.sequence||0;cur.initialValidationBpb=cloud.initialValidationBpb??cur.initialValidationBpb;}}
    if(snap){await v08RestoreSnapshot(ctx.model,ctx.opt,snap,true);restored=true;v10Log(`${cfg.key} seed ${seedIndex+1}: reprise étape ${cur.step}/${totalSteps}.`);}
    if(cur.initialValidationBpb==null){cur.initialValidationBpb=await v09Validate(ctx,cfg,corpus.validation,seed,V10_VALID_BATCHES);await v10Save(state);v10Log(`${cfg.key} seed ${seedIndex+1}: validation initiale ${cur.initialValidationBpb.toFixed(5)} bpb.`);}

    const times=[];let lastMetricAt=cur.step;
    for(let i=cur.step;i<totalSteps;i++){
      if(v10Stop){cur.step=i;await v10Checkpoint(state,ctx,true);await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'paused',step:i,global_step:i,progress:{stage:'paused',tokensSeen:cur.tokensSeen}})}).catch(()=>{});return {paused:true};}
      const t0=performance.now();const r=await v09Step(ctx,cfg,corpus.train,i,seed);await tf.nextFrame();times.push(performance.now()-t0);
      cur.step=i+1;cur.tokensSeen=Math.min(V10_TOKEN_BUDGET,(i+1)*tokensPerStep);
      ui.experiment.textContent=`${cfg.key} · seed ${seedIndex+1}/2`;ui.epoch.textContent=`Profil ${profileIndex+1}/3`;ui.step.textContent=`${i+1}/${totalSteps}`;ui.train.textContent=fmt(r.bpb);
      const overall=((profileIndex*V10_SEEDS.length+seedIndex)+(i+1)/totalSteps)/(V10_PROFILES.length*V10_SEEDS.length);pct(5+88*overall);
      if((i+1)%V10_CHECKPOINT_EVERY===0||i===totalSteps-1)await v10Checkpoint(state,ctx,((i+1)%V10_CLOUD_EVERY===0)||i===totalSteps-1);
      if((i+1)-lastMetricAt>=Math.max(64,Math.floor(totalSteps/4))){
        lastMetricAt=i+1;const vb=await v09Validate(ctx,cfg,corpus.validation,seed,V10_VALID_BATCHES);ui.val.textContent=fmt(vb);
        await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v10_validation_bpb',value:vb,globalStep:i+1,payload:{tokensSeen:cur.tokensSeen,profile:cfg.key,seedIndex}})}).catch(()=>{});
      }
    }

    const finalValidationBpb=await v09Validate(ctx,cfg,corpus.validation,seed,V10_VALID_BATCHES);
    const medianStepMs=v10Median(times);const tokensPerSec=Number.isFinite(medianStepMs)&&medianStepMs>0?tokensPerStep/(medianStepMs/1000):null;
    const result={profileKey:cfg.key,...cfg,seedIndex,seed,runId:cur.runId,initialValidationBpb:cur.initialValidationBpb,finalValidationBpb,improvementBpb:cur.initialValidationBpb-finalValidationBpb,tokensSeen:cur.tokensSeen,totalSteps,medianStepMs,tokensPerSec,resumed:restored};
    state.results=state.results.filter(x=>!(x.profileKey===cfg.key&&x.seedIndex===seedIndex));state.results.push(result);state.current=null;await v10Save(state);
    await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v10_final_validation_bpb',value:finalValidationBpb,globalStep:totalSteps,payload:result})});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'completed',step:totalSteps,global_step:totalSteps,current_validation_bpb:finalValidationBpb,best_validation_bpb:finalValidationBpb,progress:{stage:'completed',tokensSeen:cur.tokensSeen},completed_at:new Date().toISOString()})});
    v10Log(`${cfg.key} seed ${seedIndex+1}: ${cur.initialValidationBpb.toFixed(5)} → ${finalValidationBpb.toFixed(5)} bpb${tokensPerSec?` · ${Math.round(tokensPerSec)} tok/s`:''}.`);
    return {paused:false,result};
  }finally{v09Dispose(ctx);await tf.nextFrame();}
}

async function v10Tournament(){
  if(running)return;running=true;v10Stop=false;ui.start.disabled=true;ui.pause.disabled=false;
  try{
    await claimIfNeeded();await verifyBuild();await gpuGate();const locked=await requestWakeLock();if(!locked)throw new Error('Wake Lock non acquis');
    const corpus=await loadCorpus();let state=await v10Load();
    const done=await dbGet(V10_DONE_KEY);if(done){v10Status('v0.10 déjà terminé',`Vainqueur: ${done.winner.profileKey} · médiane ${done.winner.medianFinalValidationBpb.toFixed(5)} bpb.`);pct(100);return;}
    for(let p=state.profileIndex;p<V10_PROFILES.length;p++){
      for(let s=(p===state.profileIndex?state.seedIndex:0);s<V10_SEEDS.length;s++){
        state.profileIndex=p;state.seedIndex=s;await v10Save(state);
        v10Status('v0.10 · Tournoi égalitaire',`Profil ${p+1}/3 · seed ${s+1}/2 · ~1 million de tokens · checkpoints A/B.`);
        const rr=await v10RunOne(state,V10_PROFILES[p],p,s,corpus);if(rr.paused){v10Status('Pause sûre v0.10','Checkpoint modèle + Adam sauvegardé.');return;}
        state=await v10Load();state.seedIndex=s+1;await v10Save(state);
      }
      state=await v10Load();state.profileIndex=p+1;state.seedIndex=0;await v10Save(state);
    }

    state=await v10Load();const summaries=V10_PROFILES.map(cfg=>{
      const rs=state.results.filter(x=>x.profileKey===cfg.key);return {profileKey:cfg.key,...cfg,seeds:rs,medianFinalValidationBpb:v10Median(rs.map(x=>x.finalValidationBpb)),medianImprovementBpb:v10Median(rs.map(x=>x.improvementBpb)),medianTokensPerSec:v10Median(rs.map(x=>x.tokensPerSec).filter(Number.isFinite))};
    }).filter(x=>x.seeds.length===V10_SEEDS.length).sort((a,b)=>a.medianFinalValidationBpb-b.medianFinalValidationBpb);
    if(!summaries.length)throw new Error('Aucun profil n’a terminé ses deux seeds.');
    const winner=summaries[0];const report={engine:V10_ENGINE,createdAt:new Date().toISOString(),tokenBudgetPerSeed:V10_TOKEN_BUDGET,seeds:V10_SEEDS.length,summaries,winner,note:'Tournoi à budget de tokens identique; benchmark final toujours scellé. Le vainqueur devient le profil d’entraînement, pas encore le champion codec.'};
    await dbPut(V10_DONE_KEY,report);await dbPut(V10_STATE_KEY,null);
    const lastRun=winner.seeds[winner.seeds.length-1]?.runId;const payload=JSON.stringify(report);const hash=await sha256Text(payload);
    if(lastRun)await api('/artifact',{method:'POST',body:JSON.stringify({runId:lastRun,kind:'profile-tournament-v10',payload,sha256:hash,promote:false,metadata:{winner:winner.profileKey,medianFinalValidationBpb:winner.medianFinalValidationBpb}})}).catch(()=>{});
    pct(100);ui.sync.textContent='Supabase ✓';ui.experiment.textContent=winner.profileKey;ui.val.textContent=fmt(winner.medianFinalValidationBpb);ui.best.textContent=fmt(winner.medianImprovementBpb);ui.train.textContent=Number.isFinite(winner.medianTokensPerSec)?`${Math.round(winner.medianTokensPerSec)} tok/s`:'—';
    v10Status('v0.10 Tournoi terminé',`Vainqueur: ${winner.profileKey} · validation médiane ${winner.medianFinalValidationBpb.toFixed(5)} bpb · ${Math.round(winner.medianTokensPerSec||0)} tokens/s.`);
  }catch(e){const msg=e instanceof Error?e.message:String(e);v10Status(v10Stop?'v0.10 en pause':'Arrêt sécurisé v0.10',msg);v10Log(`ERREUR: ${e?.stack||e}`);}
  finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v10Install(){
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;
  ui.start.textContent='Lancer / reprendre v0.10';ui.pause.textContent='Pause sûre';ui.start.addEventListener('click',v10Tournament);ui.pause.addEventListener('click',()=>{v10Stop=true;ui.pause.disabled=true;v10Status('Pause demandée v0.10','Fin du mini-batch puis checkpoint exact modèle + Adam.');});
  setTimeout(async()=>{try{const d=await dbGet(V10_DONE_KEY),st=await dbGet(V10_STATE_KEY);if(d?.winner){v10Status('v0.10 déjà terminé',`Vainqueur ${d.winner.profileKey} · ${d.winner.medianFinalValidationBpb.toFixed(5)} bpb.`);ui.experiment.textContent=d.winner.profileKey;}else if(st?.current){v10Status('v0.10 reprise prête',`${st.current.profileKey} · seed ${st.current.seedIndex+1}/2 · étape ${st.current.step}.`);ui.experiment.textContent='Tournoi reprise';}else{v10Status('v0.10 Tournoi prêt','3 profils stables · 2 seeds chacun · même budget de 1 million de tokens · checkpoints local + Supabase.');ui.experiment.textContent='3 profils × 2 seeds';}}catch{}},850);
})();
