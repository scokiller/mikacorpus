'use strict';

const V08_ENGINE='0.8-web.5-bytegru';
const V08_ARCH='ByteGRU-e48-h192';
const V08_EMBED=48;
const V08_HIDDEN=192;
const V08_SEQ=128;
const V08_BATCH=8;
const V08_STEPS=48;
const V08_EPOCHS=10;
const V08_LR=3e-4;
const V08_GRAD_CLIP=1.0;
const V08_PATIENCE=4;
const V08_MIN_GAIN=0.002;
const V08_REFERENCE_CONTEXT=64;
const V08_REFERENCE_BATCH=64;
const V08_REFERENCE_BATCHES=24;
const V08_GLOBAL_GAIN=0.005;
const V08_BASELINE_BENCHMARK_BPB=8.02609414701239;
const V08_SEEDS=[0x19850001,0x19850002];
const V08_LOCAL_EVERY=5;
const V08_CLOUD_EVERY=10;
let v08Stop=false;

function v08SetStatus(title,detail=''){setStatus(title,detail);}
function v08Log(s){log(`v0.8 · ${s}`);}
function v08Mix32(x){x|=0;x=Math.imul(x^(x>>>16),0x21f0aaad);x=Math.imul(x^(x>>>15),0x735a2d97);return (x^(x>>>15))>>>0;}

function v08DecodeTyped(bytes,dtype){
  if(dtype==='float32')return new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
  if(dtype==='int32')return new Int32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);
  if(dtype==='bool')return new Uint8Array(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  throw new Error(`dtype checkpoint non supporté: ${dtype}`);
}
async function v08TensorObj(t){
  const d=await t.data();const bytes=new Uint8Array(d.buffer,d.byteOffset,d.byteLength);
  return {shape:[...t.shape],dtype:t.dtype,data:bytesToB64(bytes)};
}
function v08TensorFromObj(o){const bytes=b64ToBytes(o.data);return tf.tensor(v08DecodeTyped(bytes,o.dtype),o.shape,o.dtype);}
async function v08SerializeSnapshot(model,opt,includeOptimizer=true){
  const modelWeights=[];for(const w of model.weights)modelWeights.push(await v08TensorObj(w.val));
  const out={format:'mc-bytegru-tfjs-v1',architecture:V08_ARCH,embedding:V08_EMBED,hidden:V08_HIDDEN,modelWeights};
  if(includeOptimizer){
    const ow=await opt.getWeights();const optimizerWeights=[];
    try{for(const n of ow)optimizerWeights.push({name:n.name,tensor:await v08TensorObj(n.tensor)});}finally{const iter=ow.find(n=>n.name==='iter');if(iter)iter.tensor.dispose();}
    out.optimizerWeights=optimizerWeights;
  }
  return out;
}
async function v08EnsureOptimizerSlots(model,opt){const grads={};for(const w of model.trainableWeights)grads[w.val.name]=tf.zerosLike(w.val);try{opt.applyGradients(grads);}finally{Object.values(grads).forEach(t=>t.dispose());}}
async function v08RestoreSnapshot(model,opt,snap,withOptimizer=true){
  if(!snap||snap.format!=='mc-bytegru-tfjs-v1')throw new Error('Checkpoint ByteGRU inconnu.');
  const mw=snap.modelWeights.map(v08TensorFromObj);try{model.setWeights(mw);}finally{mw.forEach(t=>t.dispose());}
  if(withOptimizer&&snap.optimizerWeights){
    if(snap.optimizerWeights.length>1)v08EnsureOptimizerSlots(model,opt);
    const ow=snap.optimizerWeights.map(x=>({name:x.name,tensor:v08TensorFromObj(x.tensor)}));
    try{await opt.setWeights(ow);}catch(e){throw new Error(`Restauration exacte Adam refusée: ${e instanceof Error?e.message:String(e)}`);}finally{ow.forEach(x=>x.tensor.dispose());}
  }
}
function v08RawModelBytes(snapshot){let n=0;for(const w of snapshot.modelWeights)n+=b64ToBytes(w.data).length;return n;}

function v08SeqLoss(yTrue,yPred){
  return tf.tidy(()=>{
    const labels=yTrue.cast('int32').reshape([-1]);
    const logits=yPred.reshape([-1,256]);
    const oh=tf.oneHot(labels,256);
    return tf.losses.softmaxCrossEntropy(oh,logits).mean();
  });
}
function v08CreateModel(seed){
  const input=tf.input({shape:[null],dtype:'int32',name:`bytes_${seed>>>0}`});
  const embLayer=tf.layers.embedding({inputDim:256,outputDim:V08_EMBED,embeddingsInitializer:tf.initializers.randomUniform({minval:-0.05,maxval:0.05,seed:seed>>>0}),name:`embedding_${seed>>>0}`});
  const gruLayer=tf.layers.gru({units:V08_HIDDEN,returnSequences:true,activation:'tanh',recurrentActivation:'sigmoid',resetAfter:false,kernelInitializer:tf.initializers.glorotUniform({seed:(seed+1)>>>0}),recurrentInitializer:tf.initializers.glorotUniform({seed:(seed+2)>>>0}),biasInitializer:'zeros',name:`gru_${seed>>>0}`});
  const denseLayer=tf.layers.dense({units:256,kernelInitializer:tf.initializers.glorotUniform({seed:(seed+3)>>>0}),biasInitializer:'zeros',name:`out_${seed>>>0}`});
  const emb=embLayer.apply(input);const recurrent=gruLayer.apply(emb);const logits=denseLayer.apply(recurrent);
  const model=tf.model({inputs:input,outputs:logits,name:`bytegru_${seed>>>0}`});
  const opt=tf.train.adam(V08_LR,0.9,0.999,1e-7);
  return {model,opt};
}
function v08DisposeModel(x){try{x.model.dispose();}catch{}try{x.opt.dispose();}catch{}}

function v08MakeSeqBatch(bytes,globalStep,seed,batch=V08_BATCH,seq=V08_SEQ){
  const max=Math.max(1,bytes.length-seq-1),x=new Int32Array(batch*seq),y=new Int32Array(batch*seq);
  for(let b=0;b<batch;b++){
    const st=v08Mix32((seed^Math.imul(globalStep+1,0x9e3779b1)^Math.imul(b+1,0x85ebca6b))>>>0)%max;
    const o=b*seq;for(let j=0;j<seq;j++){x[o+j]=bytes[st+j];y[o+j]=bytes[st+j+1];}
  }
  return {x,y,batch,seq};
}
function v08TrainableVars(ctx){const vars=ctx.model.trainableWeights.map(w=>w.val);if(vars.length!==6)throw new Error(`Poids GRU entraînables inattendus: ${vars.length}`);return vars;}
async function v08CoreStep(ctx,x,y){
  const vars=v08TrainableVars(ctx);
  const vg=tf.variableGrads(()=>tf.tidy(()=>{const raw=ctx.model.apply(x,{training:true});const pred=Array.isArray(raw)?raw[0]:raw;return v08SeqLoss(y,pred);}),vars);
  try{
    const loss=(await vg.value.data())[0];
    if(!Number.isFinite(loss))throw new Error('loss GRU non finie avant mise à jour');
    const entries=Object.entries(vg.grads);
    const normSq=tf.tidy(()=>tf.addN(entries.map(([,g])=>g.square().sum())));
    const ns=(await normSq.data())[0];normSq.dispose();
    const gradNorm=Math.sqrt(ns);
    if(!Number.isFinite(gradNorm))throw new Error('gradient GRU non fini');
    const scale=Math.min(1,V08_GRAD_CLIP/(gradNorm+1e-12));
    const clipped={};
    try{for(const [name,g] of entries)clipped[name]=tf.tidy(()=>g.mul(scale));ctx.opt.applyGradients(clipped);}finally{Object.values(clipped).forEach(t=>t.dispose());}
    return {loss,gradNorm,scale};
  }finally{vg.value.dispose();Object.values(vg.grads).forEach(g=>g.dispose());}
}
async function v08TrainBatch(ctx,bytes,globalStep,seed){
  const b=v08MakeSeqBatch(bytes,globalStep,seed);const x=tf.tensor2d(b.x,[b.batch,b.seq],'int32');const y=tf.tensor2d(b.y,[b.batch,b.seq],'int32');
  try{const r=await v08CoreStep(ctx,x,y);if(tf.getBackend()!=='webgpu')throw new Error(`backend changé pendant GRU: ${tf.getBackend()}`);return r.loss/Math.LN2;}finally{x.dispose();y.dispose();}
}
async function v08SequenceValidation(ctx,bytes,seed,batches=12){
  let total=0;
  for(let i=0;i<batches;i++){
    const b=v08MakeSeqBatch(bytes,i,seed^VALIDATION_SEED);const x=tf.tensor2d(b.x,[b.batch,b.seq],'int32');const y=tf.tensor(b.y,[b.batch,b.seq,1],'int32');
    const pred=ctx.model.predict(x);const loss=v08SeqLoss(y,pred);total+=(await loss.data())[0]/Math.LN2;x.dispose();y.dispose();pred.dispose();loss.dispose();if(i%2===0)await tf.nextFrame();
  }
  return total/batches;
}
function v08ReferenceBatch(bytes,batchIndex){
  const max=Math.max(1,bytes.length-V08_REFERENCE_CONTEXT-1),x=new Int32Array(V08_REFERENCE_BATCH*V08_REFERENCE_CONTEXT),y=new Int32Array(V08_REFERENCE_BATCH);
  for(let b=0;b<V08_REFERENCE_BATCH;b++){
    const st=v08Mix32((VALIDATION_SEED^Math.imul(batchIndex+1,0x9e3779b1)^Math.imul(b+1,0x85ebca6b))>>>0)%max;
    const o=b*V08_REFERENCE_CONTEXT;for(let j=0;j<V08_REFERENCE_CONTEXT;j++)x[o+j]=bytes[st+j];y[b]=bytes[st+V08_REFERENCE_CONTEXT];
  }
  return {x,y};
}
async function v08ReferenceBpb(ctx,bytes){
  let total=0;
  for(let i=0;i<V08_REFERENCE_BATCHES;i++){
    const b=v08ReferenceBatch(bytes,i);const x=tf.tensor2d(b.x,[V08_REFERENCE_BATCH,V08_REFERENCE_CONTEXT],'int32');const y=tf.tensor1d(b.y,'int32');const pred=ctx.model.predict(x);
    const last=pred.slice([0,V08_REFERENCE_CONTEXT-1,0],[V08_REFERENCE_BATCH,1,256]).reshape([V08_REFERENCE_BATCH,256]);const oh=tf.oneHot(y,256);const loss=tf.losses.softmaxCrossEntropy(oh,last).mean();total+=(await loss.data())[0]/Math.LN2;
    x.dispose();y.dispose();pred.dispose();last.dispose();oh.dispose();loss.dispose();if(i%3===0)await tf.nextFrame();
  }
  return total/V08_REFERENCE_BATCHES;
}

async function v08Gate(corpus){
  v08SetStatus('Gate ByteGRU R11','12 mini-batchs complets · clipping gradient · contrôle NaN/Inf sur WebGPU…');
  const ctx=v08CreateModel(0x1985ffff);let firstLoss=null,lastLoss=null,maxGrad=0,minScale=1;
  try{
    for(let i=0;i<12;i++){
      const b=v08MakeSeqBatch(corpus.train,i,0x55aa55aa,V08_BATCH,V08_SEQ);
      const x=tf.tensor2d(b.x,[b.batch,b.seq],'int32'),y=tf.tensor2d(b.y,[b.batch,b.seq],'int32');
      try{
        const pred=ctx.model.predict(x);const shp=pred.shape.join(',');pred.dispose();if(shp!==String(V08_BATCH)+','+String(V08_SEQ)+',256')throw new Error('sortie GRU inattendue: ['+shp+']');
        const r=await v08CoreStep(ctx,x,y);if(firstLoss===null)firstLoss=r.loss;lastLoss=r.loss;maxGrad=Math.max(maxGrad,r.gradNorm);minScale=Math.min(minScale,r.scale);
        if(!Number.isFinite(r.loss)||!Number.isFinite(r.gradNorm))throw new Error('gate numérique non fini');
      }finally{x.dispose();y.dispose();}
      await tf.nextFrame();
    }
    if(tf.getBackend()!=='webgpu')throw new Error('backend après gate='+tf.getBackend());
    const ow=await ctx.opt.getWeights(),names=ow.map(n=>n.name);const iter=ow.find(n=>n.name==='iter');if(iter)iter.tensor.dispose();if(ow.length!==13||names[0]!=='iter')throw new Error('État Adam inattendu: '+names.join(','));
    v08Log('Gate R11 ✓ · 12/12 · loss '+(firstLoss/Math.LN2).toFixed(4)+'→'+(lastLoss/Math.LN2).toFixed(4)+' bpb · grad max '+maxGrad.toFixed(3)+' · scale min '+minScale.toFixed(3)+'.');
    return {firstLoss,lastLoss,maxGrad,minScale};
  }finally{v08DisposeModel(ctx);}
}
async function v08DbCheckpoint(state){
  const db=await openDB();return new Promise((res,rej)=>{const tx=db.transaction('kv','readwrite'),s=tx.objectStore('kv'),slot=state.sequence%2;s.put({sequence:state.sequence,payload:state},`v08:checkpoint:${slot}`);s.put({slot,sequence:state.sequence},'v08:checkpoint:pointer');s.put(state.runId,'v08:currentRunId');tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error);});
}
async function v08DbLatest(){const p=await dbGet('v08:checkpoint:pointer');if(!p)return null;const a=await dbGet(`v08:checkpoint:${p.slot}`);return a?.payload||null;}
async function v08CloudCheckpoint(state){const payload=JSON.stringify(state),hash=await sha256Text(payload),slot=state.sequence%2;ui.sync.textContent='Cloud v0.8…';const r=await api('/checkpoint',{method:'POST',body:JSON.stringify({runId:state.runId,slot,sequence:state.sequence,epoch:state.epoch,step:state.step,globalStep:state.globalStep,sha256:hash,payload,stateMeta:{family:'ByteGRU',seed:state.seed,bestBpb:state.bestBpb}})});ui.sync.textContent=`Cloud v0.8 ✓ #${r.sequence}`;}
async function v08SaveCheckpoint(state,cloud=true){await v08DbCheckpoint(state);ui.sync.textContent=`Local v0.8 ✓ #${state.sequence}`;if(cloud){try{await v08CloudCheckpoint(state);}catch(e){ui.sync.textContent='Local ✓ / cloud en retard';v08Log(`checkpoint cloud différé: ${e.message}`);}}}
async function v08Recover(){const local=await v08DbLatest();const runId=local?.runId||await dbGet('v08:currentRunId');if(!runId)return local;try{const r=await api(`/checkpoint/latest?runId=${encodeURIComponent(runId)}`);const cloud=r.checkpoint?.payload?JSON.parse(r.checkpoint.payload):null;if(cloud&&(!local||cloud.sequence>local.sequence)){v08Log(`reprise cloud #${cloud.sequence}`);return cloud;}}catch(e){v08Log(`cloud indisponible, reprise locale: ${e.message}`);}return local;}

async function v08RunSeed(seedIndex,recovered,corpus,baseline){
  const seed=V08_SEEDS[seedIndex]>>>0,experiment=`bytegru-e48-h192-s128-seed${seedIndex+1}`;ui.experiment.textContent=experiment;
  let state=recovered&&recovered.seedIndex===seedIndex?recovered:null;let ctx=v08CreateModel(seed),runId;
  try{
    if(state){await v08RestoreSnapshot(ctx.model,ctx.opt,state.currentSnapshot,true);runId=state.runId;v08Log(`seed ${seedIndex+1} restaurée époque ${state.epoch+1}, étape ${state.step}, cp #${state.sequence}`);}
    else{
      const initial=await v08SequenceValidation(ctx,corpus.validation,seed);const referenceInitial=await v08ReferenceBpb(ctx,corpus.validation);const snap=await v08SerializeSnapshot(ctx.model,ctx.opt,true);const best=await v08SerializeSnapshot(ctx.model,ctx.opt,false);
      const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:experiment,engineVersion:V08_ENGINE,config:{architecture:V08_ARCH,embedding:V08_EMBED,hidden:V08_HIDDEN,sequence:V08_SEQ,batch:V08_BATCH,stepsPerEpoch:V08_STEPS,maxEpochs:V08_EPOCHS,lr:V08_LR,seed},progress:{initialValidationBpb:initial,referenceInitialBpb:referenceInitial,baselineChampion:baseline}})});runId=rr.run.id;
      state={format:'mc-v08-state-v1',engine:V08_ENGINE,seedIndex,seed,runId,epoch:0,step:0,globalStep:0,sequence:0,bestBpb:initial,bestReferenceBpb:referenceInitial,patience:0,runningLossSum:0,runningLossCount:0,currentSnapshot:snap,bestSnapshot:best};
      await v08SaveCheckpoint(state,true);await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'v08_initial_validation_bpb',value:initial,epoch:0,step:0,globalStep:0,payload:{referenceBpb:referenceInitial,seed}})}).catch(()=>{});
      v08Log(`seed ${seedIndex+1}: initial ${initial.toFixed(4)} bpb · référence ${referenceInitial.toFixed(4)}.`);
    }
    for(let epoch=state.epoch;epoch<V08_EPOCHS;epoch++){
      if(epoch!==state.epoch){state.runningLossSum=0;state.runningLossCount=0;}
      const start=epoch===state.epoch?state.step:0;
      for(let step=start;step<V08_STEPS;step++){
        if(v08Stop){state.epoch=epoch;state.step=step;state.currentSnapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);state.sequence++;await v08SaveCheckpoint(state,true);await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'paused',epoch,step,global_step:state.globalStep,progress:{family:'ByteGRU',reason:'user pause'}})}).catch(()=>{});return {paused:true,state};}
        const loss=await v08TrainBatch(ctx,corpus.train,state.globalStep,seed^0x51c0ffee);state.runningLossSum+=loss;state.runningLossCount++;state.globalStep++;state.epoch=epoch;state.step=step+1;
        ui.epoch.textContent=`${epoch+1}/${V08_EPOCHS}`;ui.step.textContent=`${step+1}/${V08_STEPS}`;ui.train.textContent=fmt(state.runningLossSum/state.runningLossCount);pct(10+60*((seedIndex*V08_EPOCHS*V08_STEPS+epoch*V08_STEPS+step+1)/(V08_SEEDS.length*V08_EPOCHS*V08_STEPS)));
        if(state.globalStep%V08_LOCAL_EVERY===0||step===V08_STEPS-1){state.currentSnapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);state.sequence++;await v08SaveCheckpoint(state,(state.globalStep%V08_CLOUD_EVERY===0)||step===V08_STEPS-1);}
        if(step%2===0)await tf.nextFrame();
      }
      v08SetStatus('Validation ByteGRU',`Seed ${seedIndex+1}/2 · époque ${epoch+1} · validation fixe hors gradients`);
      const vb=await v08SequenceValidation(ctx,corpus.validation,seed),ref=await v08ReferenceBpb(ctx,corpus.validation),train=state.runningLossSum/Math.max(1,state.runningLossCount);ui.val.textContent=fmt(ref);
      let promoted=false;if(vb<state.bestBpb-V08_MIN_GAIN){state.bestBpb=vb;state.bestReferenceBpb=ref;state.bestSnapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,false);state.patience=0;promoted=true;}else state.patience++;
      ui.best.textContent=fmt(state.bestReferenceBpb);
      await api('/metric',{method:'POST',body:JSON.stringify({runId,name:'v08_validation_bpb',value:vb,epoch:epoch+1,step:V08_STEPS,globalStep:state.globalStep,payload:{trainBpb:train,referenceBpb:ref,bestReferenceBpb:state.bestReferenceBpb,promoted,patience:state.patience,seed}})}).catch(()=>{});
      await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'training',epoch:epoch+1,step:0,global_step:state.globalStep,best_validation_bpb:state.bestBpb,current_validation_bpb:vb,train_bpb:train,progress:{family:'ByteGRU',seed,referenceBpb:ref,bestReferenceBpb:state.bestReferenceBpb,patience:state.patience,promoted}})}).catch(()=>{});
      state.epoch=epoch+1;state.step=0;state.runningLossSum=0;state.runningLossCount=0;state.currentSnapshot=await v08SerializeSnapshot(ctx.model,ctx.opt,true);state.sequence++;await v08SaveCheckpoint(state,true);
      if(state.patience>=V08_PATIENCE){v08Log(`seed ${seedIndex+1}: early stop après ${state.patience} validations sans gain.`);break;}
    }
    await v08RestoreSnapshot(ctx.model,ctx.opt,state.bestSnapshot,false);const finalRef=await v08ReferenceBpb(ctx,corpus.validation);const rawBytes=v08RawModelBytes(state.bestSnapshot);
    const payload=JSON.stringify({engine:V08_ENGINE,protocol:PROTOCOL,architecture:V08_ARCH,seed,seedIndex,sequence:V08_SEQ,embedding:V08_EMBED,hidden:V08_HIDDEN,internalValidationBpb:state.bestBpb,referenceValidationBpb:finalRef,modelRawBytes:rawBytes,model:state.bestSnapshot,corpusManifestSha256:corpus.manifest.canonical_sha256});
    await api('/artifact',{method:'POST',body:JSON.stringify({runId,kind:'bytegru-candidate',payload,sha256:await sha256Text(payload),promote:false,metadata:{architecture:V08_ARCH,seed,sequence:V08_SEQ,referenceValidationBpb:finalRef,modelRawBytes:rawBytes}})});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId,status:'completed',epoch:state.epoch,step:state.step,global_step:state.globalStep,best_validation_bpb:state.bestBpb,current_validation_bpb:state.bestBpb,completed_at:new Date().toISOString(),progress:{family:'ByteGRU',seed,referenceValidationBpb:finalRef,modelRawBytes:rawBytes}})}).catch(()=>{});
    const result={seedIndex,seed,runId,internalValidationBpb:state.bestBpb,referenceValidationBpb:finalRef,modelRawBytes:rawBytes,snapshot:state.bestSnapshot};v08Log(`seed ${seedIndex+1} terminée: référence ${finalRef.toFixed(4)} bpb, modèle ${rawBytes.toLocaleString('fr-FR')} o.`);return {paused:false,result};
  }finally{v08DisposeModel(ctx);}
}

async function v08Finalize(results,corpus,baseline){
  if(results.length!==2)throw new Error('Deux initialisations ByteGRU sont requises avant benchmark.');
  const bothImprove=results.every(r=>r.referenceValidationBpb<=baseline.validationBpb-V08_GLOBAL_GAIN);const delta=Math.abs(results[0].referenceValidationBpb-results[1].referenceValidationBpb);const chosen=[...results].sort((a,b)=>a.referenceValidationBpb-b.referenceValidationBpb)[0];
  v08SetStatus('Benchmark scellé','Les deux initialisations sont terminées. Chargement du benchmark seulement maintenant…');
  const b=await loadBenchmark();const ctx=v08CreateModel(chosen.seed);try{await v08RestoreSnapshot(ctx.model,ctx.opt,chosen.snapshot,false);const benchmarkBpb=await v08ReferenceBpb(ctx,b.bytes);const benchmarkGain=V08_BASELINE_BENCHMARK_BPB-benchmarkBpb;const validationGain=baseline.validationBpb-chosen.referenceValidationBpb;const reproducible=delta<=0.05;const promote=bothImprove&&reproducible&&validationGain>=V08_GLOBAL_GAIN&&benchmarkGain>=V08_GLOBAL_GAIN;
    const report={engine:V08_ENGINE,architecture:V08_ARCH,seeds:results.map(r=>({seed:r.seed,referenceValidationBpb:r.referenceValidationBpb,internalValidationBpb:r.internalValidationBpb,modelRawBytes:r.modelRawBytes,runId:r.runId})),chosenSeed:chosen.seed,referenceValidationBpb:chosen.referenceValidationBpb,benchmarkBpb,baselineValidationBpb:baseline.validationBpb,baselineBenchmarkBpb:V08_BASELINE_BENCHMARK_BPB,validationGain,benchmarkGain,reproducibilityDeltaBpb:delta,reproducible,promotionGate:promote,mkc5Core:'round-trip exact',benchmarkFiles:b.files.map(f=>({name:f.name,sha256:f.sha256,bytes:f.bytes})),createdAt:new Date().toISOString()};
    const rp=JSON.stringify(report);await api('/artifact',{method:'POST',body:JSON.stringify({runId:chosen.runId,kind:'v08-benchmark-report',payload:rp,sha256:await sha256Text(rp),promote:false,metadata:{benchmarkBpb,validationGain,benchmarkGain,reproducibilityDeltaBpb:delta,promotionGate:promote}})});await api('/metric',{method:'POST',body:JSON.stringify({runId:chosen.runId,name:'v08_reference_benchmark_bpb',value:benchmarkBpb,payload:{baseline:V08_BASELINE_BENCHMARK_BPB,gain:benchmarkGain,reproducibilityDeltaBpb:delta,promotionGate:promote}})}).catch(()=>{});
    if(promote){const cp=JSON.stringify({engine:V08_ENGINE,protocol:PROTOCOL,experiment:`bytegru-e48-h192-s128-seed${chosen.seedIndex+1}`,validationBpb:chosen.referenceValidationBpb,benchmarkBpb,architecture:V08_ARCH,model:chosen.snapshot,corpusManifestSha256:corpus.manifest.canonical_sha256,reproducibility:{deltaBpb:delta,seeds:V08_SEEDS}});await api('/artifact',{method:'POST',body:JSON.stringify({runId:chosen.runId,kind:'champion',payload:cp,sha256:await sha256Text(cp),promote:true,validationBpb:chosen.referenceValidationBpb,metadata:{architecture:V08_ARCH,embedding:V08_EMBED,hidden:V08_HIDDEN,sequence:V08_SEQ,benchmarkBpb,reproducibilityDeltaBpb:delta,modelRawBytes:chosen.modelRawBytes}})});}
    return report;
  }finally{v08DisposeModel(ctx);}
}

async function v08VerifyBuild(){
  const b=await fetch('./build.json',{cache:'no-store'}).then(r=>r.json());
  for(const [path,key] of [['v08.js','v08_sha256'],['mkc5.js','mkc5_sha256']]){const ab=await fetch(`./${path}`,{cache:'no-store'}).then(r=>r.arrayBuffer());const h=await sha256Bytes(ab);if(h!==b[key])throw new Error(`Intégrité ${path} refusée.`);}
  if(b.engine!==V08_ENGINE)throw new Error(`Build attendu ${V08_ENGINE}, reçu ${b.engine}`);return b;
}

async function v08Autopilot(){
  if(running)return;running=true;v08Stop=false;ui.start.disabled=true;ui.pause.disabled=false;
  try{
    await claimIfNeeded();await verifyBuild();await v08VerifyBuild();await gpuGate();await requestWakeLock();
    const mk=await MKC5.selfTest();v08Log(`MKC5 core ✓ · header ${mk.headerBytes} o · SHA exact.`);
    const corpus=await loadCorpus();await v08Gate(corpus);
    let baseline=await dbGet('v08:baseline');if(!baseline){const cm=await api('/model');if(!cm.champion)throw new Error('Champion v0.7 introuvable.');baseline={validationBpb:Number(cm.champion.validationBpb),experimentKey:cm.champion.experimentKey,runId:cm.champion.run_id,capturedAt:new Date().toISOString(),benchmarkBpb:V08_BASELINE_BENCHMARK_BPB};await dbPut('v08:baseline',baseline);}v08Log(`référence v0.7: validation ${baseline.validationBpb.toFixed(4)}, benchmark ${baseline.benchmarkBpb.toFixed(4)}.`);
    const done=await dbGet('v08:completed');if(done){v08SetStatus('Cycle v0.8 déjà terminé',`ByteGRU ${done.promotionGate?'promu':'non promu'} · validation ${done.referenceValidationBpb.toFixed(4)} · benchmark ${done.benchmarkBpb.toFixed(4)} bpb.`);pct(100);return;}
    let results=await dbGet('v08:seedResults')||[],recovered=await v08Recover();if(recovered&&recovered.engine!==V08_ENGINE){v08Log('Ancien checkpoint R10 rejeté volontairement après instabilité numérique.');recovered=null;await dbPut('v08:checkpoint:pointer',null);await dbPut('v08:currentRunId',null);}let startIndex=recovered?.seedIndex??results.length;
    for(let i=startIndex;i<V08_SEEDS.length;i++){
      v08SetStatus('Entraînement ByteGRU',`Initialisation indépendante ${i+1}/2 · Embedding 48 · GRU 192 · séquence ${V08_SEQ}`);
      const rr=await v08RunSeed(i,recovered,corpus,baseline);if(rr.paused){v08SetStatus('Pause sûre v0.8','État ByteGRU + Adam sauvegardé localement et dans Supabase.');return;}results=results.filter(x=>x.seedIndex!==i);results.push(rr.result);results.sort((a,b)=>a.seedIndex-b.seedIndex);await dbPut('v08:seedResults',results);await dbPut('v08:currentRunId',null);await dbPut('v08:checkpoint:pointer',null);recovered=null;
    }
    const report=await v08Finalize(results,corpus,baseline);await dbPut('v08:completed',report);pct(100);ui.val.textContent=fmt(report.referenceValidationBpb);ui.best.textContent=fmt(report.referenceValidationBpb);ui.sync.textContent='Supabase ✓';
    if(report.promotionGate)v08SetStatus('v0.8 validée',`ByteGRU promu · validation ${report.referenceValidationBpb.toFixed(4)} bpb · benchmark ${report.benchmarkBpb.toFixed(4)} · MKC5 core ✓.`);else v08SetStatus('v0.8 terminée sans promotion',`Le champion v0.7 est conservé. ByteGRU: validation ${report.referenceValidationBpb.toFixed(4)}, benchmark ${report.benchmarkBpb.toFixed(4)} bpb.`);
    v08Log(`cycle fini · gain validation ${report.validationGain.toFixed(4)} · gain benchmark ${report.benchmarkGain.toFixed(4)} · delta seeds ${report.reproducibilityDeltaBpb.toFixed(4)}.`);
  }catch(e){v08SetStatus('Arrêt sécurisé v0.8',e instanceof Error?e.message:String(e));v08Log(`ERREUR: ${e.stack||e}`);}finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v08InstallControls(){
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;ui.start.textContent='Lancer v0.8 / reprendre';ui.pause.textContent='Pause sûre';ui.start.addEventListener('click',v08Autopilot);ui.pause.addEventListener('click',()=>{v08Stop=true;ui.pause.disabled=true;v08SetStatus('Pause demandée v0.8','Fin du mini-batch puis checkpoint exact modèle + état Adam.');});
  setTimeout(async()=>{try{const done=await dbGet('v08:completed');if(done)v08SetStatus('v0.8 prête / déjà calculée',`Résultat sauvegardé: validation ${done.referenceValidationBpb.toFixed(4)} · benchmark ${done.benchmarkBpb.toFixed(4)}.`);else v08SetStatus('v0.8 ByteGRU prête','Deux initialisations indépendantes, checkpoints exacts Adam A/B, benchmark scellé et MKC5 core.');ui.experiment.textContent='ByteGRU e48 · h192';}catch{}},150);
})();

// MikaCorpus R8 target rank fix 2026-08-07T11:21:54.333Z

// MikaCorpus R10 dense-output-bias gradient probe

// R11 numeric stability
