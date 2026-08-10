'use strict';

// Mika Corpus v0.11 — corrected statistics + LR duel + independent confirmation.
const V11_ENGINE='0.11-web.1-lr-duel-confirm';
const V11_SCREEN_TOKENS=500000;
const V11_CONFIRM_TOKENS=1200000;
const V11_LRS=[1.5e-4,2e-4,3e-4];
const V11_SCREEN_SEEDS=[0x19851101,0x19851102];
const V11_CONFIRM_SEEDS=[0x19851111,0x19851112,0x19851113,0x19851114];
const V11_VALID_BATCHES=16;
const V11_LOCAL_EVERY=64;
const V11_CLOUD_EVERY=128;
const V11_STATE='v11:state';
const V11_DONE='v11:completed';
const V11_PROFILES=[
  {key:'h192-e48-s128-b8',embed:48,hidden:192,seq:128,batch:8},
  {key:'h256-e64-s160-b12',embed:64,hidden:256,seq:160,batch:12}
];
let v11Stop=false;

function v11Log(s){try{log(`v0.11 · ${s}`);}catch{}}
function v11Status(t,d=''){try{setStatus(t,d);}catch{}}
function v11Mean(a){return a.length?a.reduce((s,x)=>s+x,0)/a.length:NaN;}
function v11Sd(a){if(a.length<2)return 0;const m=v11Mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)*(x-m),0)/(a.length-1));}
function v11Median(a){if(!a.length)return NaN;const x=[...a].sort((p,q)=>p-q),n=x.length,m=Math.floor(n/2);return n%2?x[m]:(x[m-1]+x[m])/2;}
function v11Steps(cfg,budget){return Math.ceil(budget/(cfg.batch*cfg.seq));}
function v11CandidateId(profileKey,lr){return `${profileKey}@${lr}`;}
function v11Profile(key){return V11_PROFILES.find(x=>x.key===key);}
function v11CreateModel(cfg,seed,lr){const ctx=v09CreateModel(cfg,seed);try{ctx.opt.dispose();}catch{}ctx.opt=tf.train.adam(lr,0.9,0.999,1e-7);return ctx;}
function v11ScreenJobs(){const out=[];for(let s=0;s<V11_SCREEN_SEEDS.length;s++)for(const lr of V11_LRS)for(const cfg of V11_PROFILES)out.push({phase:'screen',profileKey:cfg.key,lr,seed:V11_SCREEN_SEEDS[s]>>>0,seedIndex:s,budget:V11_SCREEN_TOKENS});return out;}
function v11Fresh(){return {version:1,phase:'screen',screenIndex:0,screenResults:[],winner:null,confirmIndex:0,confirmResults:[],current:null,startedAt:new Date().toISOString()};}
async function v11Load(){return (await dbGet(V11_STATE))||v11Fresh();}
async function v11Save(s){await dbPut(V11_STATE,s);}

async function v11CloudCheckpoint(cur,snapshot){
  const payload=JSON.stringify({engine:V11_ENGINE,job:cur.job,runId:cur.runId,step:cur.step,sequence:cur.sequence,tokensSeen:cur.tokensSeen,initialValidationBpb:cur.initialValidationBpb,snapshot});
  const hash=await sha256Text(payload),slot=cur.sequence%2;
  await api('/checkpoint',{method:'POST',body:JSON.stringify({runId:cur.runId,slot,sequence:cur.sequence,epoch:0,step:cur.step,globalStep:cur.step,sha256:hash,payload,stateMeta:{family:'v11-lr-duel',phase:cur.job.phase,profile:cur.job.profileKey,lr:cur.job.lr,tokensSeen:cur.tokensSeen}})});
}
async function v11Checkpoint(state,ctx,cloud=false){
  const cur=state.current;if(!cur)return;const snapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);cur.snapshot=snapshot;cur.sequence=(cur.sequence||0)+1;cur.savedAt=new Date().toISOString();await v11Save(state);ui.sync.textContent=`Local v0.11 ✓ #${cur.sequence}`;
  if(cloud){try{await v11CloudCheckpoint(cur,snapshot);ui.sync.textContent=`Cloud v0.11 ✓ #${cur.sequence}`;}catch(e){ui.sync.textContent='Local ✓ / cloud en retard';v11Log(`checkpoint cloud différé: ${e.message}`);}}
}
async function v11CloudRestore(cur){
  if(!cur?.runId)return null;try{const r=await api(`/checkpoint/latest?runId=${encodeURIComponent(cur.runId)}`);if(!r.checkpoint?.payload)return null;const p=JSON.parse(r.checkpoint.payload);return p.engine===V11_ENGINE&&p.snapshot?p:null;}catch(e){v11Log(`reprise cloud indisponible: ${e.message}`);return null;}
}

async function v11RunJob(state,job,ordinal,totalJobs,corpus){
  const cfg=v11Profile(job.profileKey);if(!cfg)throw new Error(`profil inconnu ${job.profileKey}`);const totalSteps=v11Steps(cfg,job.budget),tokensPerStep=cfg.batch*cfg.seq;
  const jobKey=`${job.phase}-${job.profileKey}-lr${String(job.lr).replace('.','p')}-s${job.seedIndex+1}`;
  let cur=state.current;
  if(!cur||cur.jobKey!==jobKey){
    const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:`v11-${jobKey}`,engineVersion:V11_ENGINE,config:{profile:cfg,learningRate:job.lr,seed:job.seed,tokenBudget:job.budget,totalSteps,validationBatches:V11_VALID_BATCHES,statistic:'mean+sample-sd'},progress:{stage:job.phase}})});
    cur={jobKey,job,runId:rr.run.id,step:0,tokensSeen:0,sequence:0,initialValidationBpb:null,snapshot:null};state.current=cur;await v11Save(state);
  }
  const ctx=v11CreateModel(cfg,job.seed,job.lr);let resumed=false;
  try{
    let snap=cur.snapshot;if(!snap){const cloud=await v11CloudRestore(cur);if(cloud){snap=cloud.snapshot;cur.step=cloud.step||0;cur.tokensSeen=cloud.tokensSeen||0;cur.sequence=cloud.sequence||0;cur.initialValidationBpb=cloud.initialValidationBpb??cur.initialValidationBpb;}}
    if(snap){await v08RestoreSnapshot(ctx.model,ctx.opt,snap,true);resumed=true;v11Log(`${jobKey}: reprise étape ${cur.step}/${totalSteps}.`);}
    if(cur.initialValidationBpb==null){cur.initialValidationBpb=await v09Validate(ctx,cfg,corpus.validation,job.seed,V11_VALID_BATCHES);await v11Save(state);}
    const times=[];let nextValidation=Math.max(64,Math.floor(totalSteps/3));
    for(let i=cur.step;i<totalSteps;i++){
      if(v11Stop){cur.step=i;await v11Checkpoint(state,ctx,true);await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'paused',step:i,global_step:i,progress:{stage:'paused',tokensSeen:cur.tokensSeen}})}).catch(()=>{});return {paused:true};}
      const t0=performance.now();const tr=await v09Step(ctx,cfg,corpus.train,i,job.seed);await tf.nextFrame();times.push(performance.now()-t0);cur.step=i+1;cur.tokensSeen=Math.min(job.budget,(i+1)*tokensPerStep);
      ui.experiment.textContent=`${cfg.key} · LR ${job.lr}`;ui.epoch.textContent=`${ordinal+1}/${totalJobs}`;ui.step.textContent=`${i+1}/${totalSteps}`;ui.train.textContent=fmt(tr.bpb);pct(4+91*((ordinal+(i+1)/totalSteps)/totalJobs));
      if((i+1)%V11_LOCAL_EVERY===0||i===totalSteps-1)await v11Checkpoint(state,ctx,((i+1)%V11_CLOUD_EVERY===0)||i===totalSteps-1);
      if(i+1>=nextValidation&&i<totalSteps-1){const vb=await v09Validate(ctx,cfg,corpus.validation,job.seed,V11_VALID_BATCHES);ui.val.textContent=fmt(vb);await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v11_validation_bpb',value:vb,globalStep:i+1,payload:{phase:job.phase,profile:cfg.key,lr:job.lr,tokensSeen:cur.tokensSeen}})}).catch(()=>{});nextValidation+=Math.max(64,Math.floor(totalSteps/3));}
    }
    const finalValidationBpb=await v09Validate(ctx,cfg,corpus.validation,job.seed,V11_VALID_BATCHES);const medianStepMs=v11Median(times),tokensPerSec=Number.isFinite(medianStepMs)&&medianStepMs>0?tokensPerStep/(medianStepMs/1000):null;
    const result={phase:job.phase,profileKey:cfg.key,lr:job.lr,seed:job.seed,seedIndex:job.seedIndex,runId:cur.runId,budget:job.budget,initialValidationBpb:cur.initialValidationBpb,finalValidationBpb,improvementBpb:cur.initialValidationBpb-finalValidationBpb,tokensPerSec,medianStepMs,resumed};
    state.current=null;await v11Save(state);await api('/metric',{method:'POST',body:JSON.stringify({runId:cur.runId,name:'v11_final_validation_bpb',value:finalValidationBpb,globalStep:totalSteps,payload:result})});await api('/run/update',{method:'POST',body:JSON.stringify({runId:cur.runId,status:'completed',step:totalSteps,global_step:totalSteps,current_validation_bpb:finalValidationBpb,best_validation_bpb:finalValidationBpb,progress:{stage:'completed',phase:job.phase,tokensSeen:job.budget},completed_at:new Date().toISOString()})});
    v11Log(`${jobKey}: ${cur.initialValidationBpb.toFixed(5)}→${finalValidationBpb.toFixed(5)} bpb.`);return {paused:false,result};
  }finally{v09Dispose(ctx);await tf.nextFrame();await sleep(400);}
}

function v11SummarizeScreen(results){
  const groups=[];for(const cfg of V11_PROFILES)for(const lr of V11_LRS){const rs=results.filter(x=>x.profileKey===cfg.key&&x.lr===lr);if(rs.length!==V11_SCREEN_SEEDS.length)continue;const finals=rs.map(x=>x.finalValidationBpb),imps=rs.map(x=>x.improvementBpb);groups.push({profileKey:cfg.key,lr,seeds:rs,meanFinalValidationBpb:v11Mean(finals),sdFinalValidationBpb:v11Sd(finals),meanImprovementBpb:v11Mean(imps),meanTokensPerSec:v11Mean(rs.map(x=>x.tokensPerSec).filter(Number.isFinite))});}
  for(const g of groups)g.selectionScore=g.meanFinalValidationBpb+0.25*g.sdFinalValidationBpb;groups.sort((a,b)=>a.selectionScore-b.selectionScore);return groups;
}

async function v11Tournament(){
  if(running)return;running=true;v11Stop=false;ui.start.disabled=true;ui.pause.disabled=false;
  try{
    await claimIfNeeded();await verifyBuild();await gpuGate();const locked=await requestWakeLock();if(!locked)throw new Error('Wake Lock non acquis');const corpus=await loadCorpus();let state=await v11Load();const done=await dbGet(V11_DONE);if(done){v11Status('v0.11 déjà terminé',`Configuration confirmée: ${done.confirmed.profileKey} · LR ${done.confirmed.lr}.`);pct(100);return;}
    const screenJobs=v11ScreenJobs();
    if(state.phase==='screen'){
      for(let i=state.screenIndex;i<screenJobs.length;i++){state.screenIndex=i;await v11Save(state);v11Status('v0.11 · Duel + LR',`Criblage ${i+1}/${screenJobs.length} · 500k tokens · alternance thermique.`);const rr=await v11RunJob(state,screenJobs[i],i,screenJobs.length+V11_CONFIRM_SEEDS.length,corpus);if(rr.paused)return;state=await v11Load();state.screenResults=state.screenResults.filter(x=>!(x.profileKey===rr.result.profileKey&&x.lr===rr.result.lr&&x.seedIndex===rr.result.seedIndex));state.screenResults.push(rr.result);state.screenIndex=i+1;await v11Save(state);}
      const groups=v11SummarizeScreen(state.screenResults);if(!groups.length)throw new Error('Criblage incomplet.');state.winner={profileKey:groups[0].profileKey,lr:groups[0].lr,screen:groups};state.phase='confirm';state.confirmIndex=0;await v11Save(state);v11Log(`criblage: ${groups[0].profileKey} LR ${groups[0].lr} retenu · score ${groups[0].selectionScore.toFixed(6)}.`);
    }
    state=await v11Load();const winner=state.winner;if(!winner)throw new Error('Vainqueur de criblage absent.');
    for(let i=state.confirmIndex;i<V11_CONFIRM_SEEDS.length;i++){const job={phase:'confirm',profileKey:winner.profileKey,lr:winner.lr,seed:V11_CONFIRM_SEEDS[i]>>>0,seedIndex:i,budget:V11_CONFIRM_TOKENS};state.confirmIndex=i;await v11Save(state);v11Status('v0.11 · Confirmation indépendante',`Seed ${i+1}/${V11_CONFIRM_SEEDS.length} · 1,2 M tokens · ${winner.profileKey} · LR ${winner.lr}.`);const rr=await v11RunJob(state,job,screenJobs.length+i,screenJobs.length+V11_CONFIRM_SEEDS.length,corpus);if(rr.paused)return;state=await v11Load();state.confirmResults=state.confirmResults.filter(x=>x.seedIndex!==i);state.confirmResults.push(rr.result);state.confirmIndex=i+1;await v11Save(state);}
    state=await v11Load();if(state.confirmResults.length!==V11_CONFIRM_SEEDS.length)throw new Error('Confirmation incomplète.');const finals=state.confirmResults.map(x=>x.finalValidationBpb),imps=state.confirmResults.map(x=>x.improvementBpb);const confirmed={profileKey:winner.profileKey,lr:winner.lr,seeds:state.confirmResults,meanFinalValidationBpb:v11Mean(finals),sdFinalValidationBpb:v11Sd(finals),medianFinalValidationBpb:v11Median(finals),meanImprovementBpb:v11Mean(imps),meanTokensPerSec:v11Mean(state.confirmResults.map(x=>x.tokensPerSec).filter(Number.isFinite))};
    const report={engine:V11_ENGINE,createdAt:new Date().toISOString(),v10Correction:{reason:'v0.10 even-N median used upper middle instead of averaging central pair',correctedMeans:{'h192-e48-s128-b8':7.996237643521083,'h256-e64-s160-b12':7.998056446045283,'h256-e64-s192-b16':8.001737820678325}},screening:winner.screen,confirmed,note:'Configuration d’entraînement confirmée par 4 nouvelles seeds; benchmark codec final toujours scellé.'};await dbPut(V11_DONE,report);await dbPut(V11_STATE,null);const lastRun=state.confirmResults[state.confirmResults.length-1].runId,payload=JSON.stringify(report),hash=await sha256Text(payload);await api('/artifact',{method:'POST',body:JSON.stringify({runId:lastRun,kind:'lr-duel-confirm-v11',payload,sha256:hash,promote:false,metadata:{profile:confirmed.profileKey,lr:confirmed.lr,meanFinalValidationBpb:confirmed.meanFinalValidationBpb,sd:confirmed.sdFinalValidationBpb}})}).catch(()=>{});
    pct(100);ui.sync.textContent='Supabase ✓';ui.experiment.textContent=`${confirmed.profileKey} · LR ${confirmed.lr}`;ui.val.textContent=fmt(confirmed.meanFinalValidationBpb);ui.best.textContent=fmt(confirmed.meanImprovementBpb);ui.train.textContent=Number.isFinite(confirmed.meanTokensPerSec)?`${Math.round(confirmed.meanTokensPerSec)} tok/s`:'—';v11Status('v0.11 terminé',`Confirmé: ${confirmed.profileKey} · LR ${confirmed.lr} · moyenne ${confirmed.meanFinalValidationBpb.toFixed(5)} ± ${confirmed.sdFinalValidationBpb.toFixed(5)} bpb.`);
  }catch(e){const msg=e instanceof Error?e.message:String(e);v11Status(v11Stop?'v0.11 en pause':'Arrêt sécurisé v0.11',msg);v11Log(`ERREUR: ${e?.stack||e}`);}finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v11Install(){
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;ui.start.textContent='Lancer / reprendre v0.11';ui.pause.textContent='Pause sûre';ui.start.addEventListener('click',v11Tournament);ui.pause.addEventListener('click',()=>{v11Stop=true;ui.pause.disabled=true;v11Status('Pause demandée v0.11','Fin du mini-batch puis checkpoint exact modèle + Adam.');});
  setTimeout(async()=>{try{const d=await dbGet(V11_DONE),st=await dbGet(V11_STATE);if(d?.confirmed){v11Status('v0.11 déjà terminé',`${d.confirmed.profileKey} · LR ${d.confirmed.lr} · ${d.confirmed.meanFinalValidationBpb.toFixed(5)} bpb.`);ui.experiment.textContent=d.confirmed.profileKey;}else if(st?.current){v11Status('v0.11 reprise prête',`${st.current.job.profileKey} · LR ${st.current.job.lr} · étape ${st.current.step}.`);ui.experiment.textContent='Duel reprise';}else{v11Status('v0.11 Duel statistique prêt','2 profils × 3 learning rates × 2 seeds, puis 4 seeds indépendantes de confirmation. Checkpoints local + Supabase.');ui.experiment.textContent='Duel + LR';}}catch{}},1050);
})();
