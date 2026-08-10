'use strict';

const V09_ENGINE='0.9-web.1-hw-autotune';
const V09_SEED=0x19850909;
const V09_PROBE_STEPS=4;
const V09_LEARN_STEPS=24;
const V09_VALID_BATCHES=4;
const V09_LR=3e-4;
const V09_GRAD_CLIP=1.0;
let v09Stop=false;

const V09_CANDIDATES=[
  {key:'h192-e48-s128-b8',embed:48,hidden:192,seq:128,batch:8},
  {key:'h256-e64-s160-b12',embed:64,hidden:256,seq:160,batch:12},
  {key:'h256-e64-s192-b16',embed:64,hidden:256,seq:192,batch:16},
  {key:'h320-e64-s192-b12',embed:64,hidden:320,seq:192,batch:12},
  {key:'h320-e80-s256-b8',embed:80,hidden:320,seq:256,batch:8},
  {key:'h384-e80-s256-b6',embed:80,hidden:384,seq:256,batch:6},
  {key:'h448-e96-s256-b4',embed:96,hidden:448,seq:256,batch:4},
  {key:'h384-e96-s320-b4',embed:96,hidden:384,seq:320,batch:4}
];

function v09Log(s){log(`v0.9 · ${s}`);}
function v09Status(t,d=''){setStatus(t,d);}
function v09Mix32(x){x|=0;x=Math.imul(x^(x>>>16),0x21f0aaad);x=Math.imul(x^(x>>>15),0x735a2d97);return (x^(x>>>15))>>>0;}
function v09Median(a){const x=[...a].sort((p,q)=>p-q);return x[Math.floor(x.length/2)]||0;}

function v09MakeBatch(bytes,cfg,globalStep,seed){
  const max=Math.max(1,bytes.length-cfg.seq-1);
  const x=new Int32Array(cfg.batch*cfg.seq),y=new Int32Array(cfg.batch*cfg.seq);
  for(let b=0;b<cfg.batch;b++){
    const st=v09Mix32((seed^Math.imul(globalStep+1,0x9e3779b1)^Math.imul(b+1,0x85ebca6b))>>>0)%max;
    const o=b*cfg.seq;
    for(let j=0;j<cfg.seq;j++){x[o+j]=bytes[st+j];y[o+j]=bytes[st+j+1];}
  }
  return {x,y};
}

function v09CreateModel(cfg,seed){
  const input=tf.input({shape:[null],dtype:'int32',name:`v09_bytes_${cfg.key}_${seed>>>0}`});
  const emb=tf.layers.embedding({inputDim:256,outputDim:cfg.embed,embeddingsInitializer:tf.initializers.randomUniform({minval:-0.05,maxval:0.05,seed:seed>>>0}),name:`v09_emb_${cfg.key}`}).apply(input);
  const rec=tf.layers.gru({units:cfg.hidden,returnSequences:true,activation:'tanh',recurrentActivation:'sigmoid',resetAfter:false,kernelInitializer:tf.initializers.glorotUniform({seed:(seed+1)>>>0}),recurrentInitializer:tf.initializers.glorotUniform({seed:(seed+2)>>>0}),biasInitializer:'zeros',name:`v09_gru_${cfg.key}`}).apply(emb);
  const logits=tf.layers.dense({units:256,kernelInitializer:tf.initializers.glorotUniform({seed:(seed+3)>>>0}),biasInitializer:'zeros',name:`v09_out_${cfg.key}`}).apply(rec);
  const model=tf.model({inputs:input,outputs:logits,name:`v09_${cfg.key}`});
  const opt=tf.train.adam(V09_LR,0.9,0.999,1e-7);
  return {model,opt};
}

function v09Loss(yTrue,yPred){
  return tf.tidy(()=>{
    const labels=yTrue.cast('int32').reshape([-1]);
    const logits=yPred.reshape([-1,256]);
    const oh=tf.oneHot(labels,256);
    return tf.losses.softmaxCrossEntropy(oh,logits).mean();
  });
}

async function v09Step(ctx,cfg,bytes,step,seed){
  const b=v09MakeBatch(bytes,cfg,step,seed);
  const x=tf.tensor2d(b.x,[cfg.batch,cfg.seq],'int32');
  const y=tf.tensor2d(b.y,[cfg.batch,cfg.seq],'int32');
  const vars=ctx.model.trainableWeights.map(w=>w.val);
  try{
    const vg=tf.variableGrads(()=>tf.tidy(()=>{const raw=ctx.model.apply(x,{training:true});const pred=Array.isArray(raw)?raw[0]:raw;return v09Loss(y,pred);}),vars);
    try{
      const loss=(await vg.value.data())[0];
      if(!Number.isFinite(loss))throw new Error('loss non finie');
      const entries=Object.entries(vg.grads);
      const normSq=tf.tidy(()=>tf.addN(entries.map(([,g])=>g.square().sum())));
      const ns=(await normSq.data())[0];normSq.dispose();
      const gradNorm=Math.sqrt(ns);
      if(!Number.isFinite(gradNorm))throw new Error('gradient non fini');
      const scale=Math.min(1,V09_GRAD_CLIP/(gradNorm+1e-12));
      const clipped={};
      try{for(const [name,g] of entries)clipped[name]=tf.tidy(()=>g.mul(scale));ctx.opt.applyGradients(clipped);}finally{Object.values(clipped).forEach(t=>t.dispose());}
      if(tf.getBackend()!=='webgpu')throw new Error(`backend changé: ${tf.getBackend()}`);
      return {bpb:loss/Math.LN2,gradNorm,scale};
    }finally{vg.value.dispose();Object.values(vg.grads).forEach(g=>g.dispose());}
  }finally{x.dispose();y.dispose();}
}

async function v09Validate(ctx,cfg,bytes,seed,batches=V09_VALID_BATCHES){
  let total=0;
  for(let i=0;i<batches;i++){
    if(v09Stop)throw new Error('pause demandée');
    const b=v09MakeBatch(bytes,cfg,i,seed^0x51515151);
    const x=tf.tensor2d(b.x,[cfg.batch,cfg.seq],'int32');
    const y=tf.tensor2d(b.y,[cfg.batch,cfg.seq],'int32');
    const pred=ctx.model.predict(x);const loss=v09Loss(y,pred);
    total+=(await loss.data())[0]/Math.LN2;
    x.dispose();y.dispose();pred.dispose();loss.dispose();
    await tf.nextFrame();
  }
  return total/batches;
}

function v09Dispose(ctx){try{ctx.model.dispose();}catch{}try{ctx.opt.dispose();}catch{}}

async function v09AdapterInfo(){
  const out={userAgent:navigator.userAgent,deviceMemory:navigator.deviceMemory??null,hardwareConcurrency:navigator.hardwareConcurrency??null};
  try{
    const a=await navigator.gpu?.requestAdapter();
    if(a){out.features=[...a.features];out.limits={maxBufferSize:Number(a.limits?.maxBufferSize||0),maxStorageBufferBindingSize:Number(a.limits?.maxStorageBufferBindingSize||0),maxComputeWorkgroupStorageSize:Number(a.limits?.maxComputeWorkgroupStorageSize||0)};out.info=a.info?{vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description}:null;}
  }catch(e){out.adapterError=String(e?.message||e);}
  return out;
}

async function v09HardwareProbe(cfg,bytes,index,total){
  v09Status('v0.9 · Autotune matériel',`${index+1}/${total} · ${cfg.key} · test débit WebGPU`);
  ui.experiment.textContent=cfg.key;
  const ctx=v09CreateModel(cfg,V09_SEED^index);
  const timings=[];let last=null;const mem0=tf.memory();
  try{
    await v09Step(ctx,cfg,bytes,0,V09_SEED);await tf.nextFrame();
    for(let i=0;i<V09_PROBE_STEPS;i++){
      if(v09Stop)throw new Error('pause demandée');
      const t0=performance.now();last=await v09Step(ctx,cfg,bytes,i+1,V09_SEED);await tf.nextFrame();timings.push(performance.now()-t0);
      pct(5+30*((index+(i+1)/V09_PROBE_STEPS)/total));
    }
    const ms=v09Median(timings);const tok=cfg.batch*cfg.seq;const tokensPerSec=tok/(ms/1000);const slowdown=timings.length>1?timings[timings.length-1]/Math.max(1,timings[0]):1;const mem1=tf.memory();
    return {key:cfg.key,...cfg,ok:true,medianStepMs:ms,tokensPerSec,slowdown,lastBpb:last?.bpb??null,tensorBytesBefore:mem0.numBytes,tensorBytesAfter:mem1.numBytes,numTensors:mem1.numTensors};
  }catch(e){return {key:cfg.key,...cfg,ok:false,error:String(e?.message||e)};}
  finally{v09Dispose(ctx);await tf.nextFrame();await sleep(120);}
}

async function v09LearningProbe(cfg,bytes,index,total){
  v09Status('v0.9 · Test d’apprentissage',`${index+1}/${total} · ${cfg.key} · ${V09_LEARN_STEPS} mini-batchs`);
  ui.experiment.textContent=cfg.key;
  const ctx=v09CreateModel(cfg,V09_SEED^0x77770000^index);let initial,final;const times=[];
  try{
    initial=await v09Validate(ctx,cfg,bytes.validation,V09_SEED^index);
    for(let i=0;i<V09_LEARN_STEPS;i++){
      if(v09Stop)throw new Error('pause demandée');
      const t0=performance.now();await v09Step(ctx,cfg,bytes.train,i,V09_SEED^index);await tf.nextFrame();times.push(performance.now()-t0);
      pct(38+47*((index+(i+1)/V09_LEARN_STEPS)/total));
    }
    final=await v09Validate(ctx,cfg,bytes.validation,V09_SEED^index);
    const improvement=initial-final;const ms=v09Median(times);const tokensPerSec=(cfg.batch*cfg.seq)/(ms/1000);
    return {key:cfg.key,...cfg,ok:true,initialValidationBpb:initial,finalValidationBpb:final,improvementBpb:improvement,medianStepMs:ms,tokensPerSec,score:(improvement*1000)+(Math.log10(Math.max(1,tokensPerSec))*0.01)};
  }catch(e){return {key:cfg.key,...cfg,ok:false,error:String(e?.message||e)};}
  finally{v09Dispose(ctx);await tf.nextFrame();await sleep(180);}
}

async function v09Autotune(){
  if(running)return;running=true;v09Stop=false;ui.start.disabled=true;ui.pause.disabled=false;
  let runId=null;
  try{
    await claimIfNeeded();await verifyBuild();await gpuGate();
    const locked=await requestWakeLock();if(!locked)throw new Error('Wake Lock non acquis');
    const corpus=await loadCorpus();
    const adapter=await v09AdapterInfo();
    const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:'hardware-autotune-v09',engineVersion:V09_ENGINE,config:{probeSteps:V09_PROBE_STEPS,learnSteps:V09_LEARN_STEPS,candidates:V09_CANDIDATES,adapter},progress:{stage:'hardware-probe'}})});runId=rr.run.id;
    v09Log(`run ${runId} créé · ${V09_CANDIDATES.length} profils.`);

    const hardware=[];
    for(let i=0;i<V09_CANDIDATES.length;i++){
      const r=await v09HardwareProbe(V09_CANDIDATES[i],corpus.train,i,V09_CANDIDATES.length);hardware.push(r);v09Log(`${r.key}: ${r.ok?`${Math.round(r.tokensPerSec)} tok/s · ${r.medianStepMs.toFixed(0)} ms/step · chauffe x${r.slowdown.toFixed(2)}`:`échec ${r.error}`}`);
      await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'hardware_probe_tokens_per_sec',value:r.ok?r.tokensPerSec:null,globalStep:i,payload:r})}).catch(()=>{});
    }
    const successful=hardware.filter(x=>x.ok&&Number.isFinite(x.tokensPerSec));
    if(successful.length<2)throw new Error(`Seulement ${successful.length} profil(s) WebGPU stable(s).`);
    successful.sort((a,b)=>b.tokensPerSec-a.tokensPerSec);
    const selected=[];
    for(const r of successful){if(selected.length>=4)break;if(!selected.some(x=>x.hidden===r.hidden&&x.seq===r.seq))selected.push(r);}
    for(const r of successful){if(selected.length>=4)break;if(!selected.some(x=>x.key===r.key))selected.push(r);}
    v09Log(`profils retenus: ${selected.map(x=>x.key).join(', ')}.`);
    await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'training',progress:{stage:'learning-probe',selected:selected.map(x=>x.key)}})}).catch(()=>{});

    const learning=[];
    for(let i=0;i<selected.length;i++){
      const cfg=V09_CANDIDATES.find(x=>x.key===selected[i].key);const r=await v09LearningProbe(cfg,corpus,i,selected.length);learning.push(r);v09Log(`${r.key}: ${r.ok?`${r.initialValidationBpb.toFixed(4)}→${r.finalValidationBpb.toFixed(4)} bpb · gain ${r.improvementBpb.toFixed(4)}`:`échec ${r.error}`}`);
      await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'learning_probe_gain_bpb',value:r.ok?r.improvementBpb:null,globalStep:i,payload:r})}).catch(()=>{});
    }
    const good=learning.filter(x=>x.ok&&Number.isFinite(x.score)).sort((a,b)=>b.score-a.score);
    if(!good.length)throw new Error('Aucun profil d’apprentissage stable.');
    const winner=good[0];
    const report={engine:V09_ENGINE,createdAt:new Date().toISOString(),adapter,hardware,learning,recommended:{key:winner.key,embed:winner.embed,hidden:winner.hidden,seq:winner.seq,batch:winner.batch,tokensPerSec:winner.tokensPerSec,initialValidationBpb:winner.initialValidationBpb,finalValidationBpb:winner.finalValidationBpb,improvementBpb:winner.improvementBpb},note:'Profil matériel recommandé pour les expériences suivantes; ce run ne remplace pas le champion de compression.'};
    await dbPut('v09:hardwareProfile',report);
    const payload=JSON.stringify(report);const hash=await sha256Text(payload);
    await api('/artifact',{method:'POST',body:JSON.stringify({runId,kind:'hardware-profile-v09',payload,sha256:hash,promote:false,metadata:{recommended:winner.key}})});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'completed',global_step:V09_CANDIDATES.length+selected.length,progress:{stage:'completed',recommended:winner.key},completed_at:new Date().toISOString()})});
    pct(100);ui.sync.textContent='Supabase ✓';ui.experiment.textContent=winner.key;ui.train.textContent=`${Math.round(winner.tokensPerSec)} tok/s`;ui.val.textContent=fmt(winner.finalValidationBpb);ui.best.textContent=fmt(winner.improvementBpb);
    v09Status('v0.9 Autotune terminé',`Profil recommandé: ${winner.key} · ${Math.round(winner.tokensPerSec)} tokens/s · gain court ${winner.improvementBpb.toFixed(4)} bpb. Résultat sauvegardé dans Supabase.`);
  }catch(e){
    const msg=e instanceof Error?e.message:String(e);v09Status(v09Stop?'v0.9 en pause':'Arrêt sécurisé v0.9',msg);v09Log(`ERREUR: ${e?.stack||e}`);if(runId)await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:v09Stop?'paused':'failed',progress:{error:msg}})}).catch(()=>{});
  }finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v09Install(){
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;
  ui.start.textContent='Lancer v0.9 Autotune';ui.pause.textContent='Pause sûre';ui.start.addEventListener('click',v09Autotune);ui.pause.addEventListener('click',()=>{v09Stop=true;ui.pause.disabled=true;v09Status('Pause demandée v0.9','Fin de l’opération WebGPU en cours puis arrêt propre.');});
  setTimeout(async()=>{try{const r=await dbGet('v09:hardwareProfile');if(r?.recommended){v09Status('v0.9 Autotune déjà mesuré',`Profil précédent: ${r.recommended.key} · ${Math.round(r.recommended.tokensPerSec)} tokens/s. Relancer permet de remesurer les performances actuelles.`);ui.experiment.textContent=r.recommended.key;}else{v09Status('v0.9 Hardware Autotune prêt','Mesure le débit WebGPU, la stabilité thermique et la vitesse d’apprentissage de 8 profils, puis choisit automatiquement le meilleur pour cet iPhone.');ui.experiment.textContent='Autotune 8 profils';}}catch{}},450);
})();
