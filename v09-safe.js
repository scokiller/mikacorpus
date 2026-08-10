'use strict';

// Mika Corpus v0.9.2 — crash-resilient WebGPU hardware autotune for iOS PWA.
// A synchronous crash marker is written before every risky GPU probe. If WebKit/iOS
// terminates the process, the next launch blacklists that exact profile and resumes.
const V09S_ENGINE='0.9-web.2-hw-autotune-safe';
const V09S_STATE='mika-v09safe-state-v2';
const V09S_INFLIGHT='mika-v09safe-inflight-v2';
const V09S_KNOWN_UNSAFE=new Set(['h320-e64-s192-b12']); // killed WebKit twice on this iPhone
const V09S_CANDIDATES=[
  {key:'h192-e48-s128-b8',embed:48,hidden:192,seq:128,batch:8},
  {key:'h256-e64-s160-b12',embed:64,hidden:256,seq:160,batch:12},
  {key:'h256-e64-s192-b16',embed:64,hidden:256,seq:192,batch:16},
  {key:'h288-e64-s192-b12',embed:64,hidden:288,seq:192,batch:12},
  {key:'h320-e64-s192-b8',embed:64,hidden:320,seq:192,batch:8},
  {key:'h320-e80-s256-b6',embed:80,hidden:320,seq:256,batch:6},
  {key:'h384-e80-s192-b6',embed:80,hidden:384,seq:192,batch:6},
  {key:'h384-e80-s256-b4',embed:80,hidden:384,seq:256,batch:4},
  {key:'h448-e96-s192-b4',embed:96,hidden:448,seq:192,batch:4}
];
let v09sStop=false;

function v09sLog(s){try{log(`v0.9.2 · ${s}`);}catch{}}
function v09sStatus(t,d=''){try{setStatus(t,d);}catch{}}
function v09sLoad(){try{return JSON.parse(localStorage.getItem(V09S_STATE)||'null');}catch{return null;}}
function v09sSave(s){localStorage.setItem(V09S_STATE,JSON.stringify(s));}
function v09sMarker(m){localStorage.setItem(V09S_INFLIGHT,JSON.stringify(m));}
function v09sClearMarker(){localStorage.removeItem(V09S_INFLIGHT);}
function v09sReadMarker(){try{return JSON.parse(localStorage.getItem(V09S_INFLIGHT)||'null');}catch{return null;}}
function v09sFresh(adapter){return {version:2,runId:null,phase:'hardware',hardwareIndex:0,hardware:[],selected:[],learningIndex:0,learning:[],blacklist:[...V09S_KNOWN_UNSAFE],adapter,startedAt:new Date().toISOString()};}
function v09sCfg(key){return V09S_CANDIDATES.find(x=>x.key===key);}

async function v09sRecoverCrash(state){
  const marker=v09sReadMarker();
  if(!marker)return state;
  const key=String(marker.key||'unknown');
  if(!state.blacklist.includes(key))state.blacklist.push(key);
  if(marker.phase==='hardware'){
    if(!state.hardware.some(x=>x.key===key))state.hardware.push({...(v09sCfg(key)||{key}),ok:false,error:'process WebKit/iOS interrompu pendant ce profil',processCrash:true});
    state.hardwareIndex=Math.max(state.hardwareIndex,Number(marker.index||0)+1);
  }else if(marker.phase==='learning'){
    if(!state.learning.some(x=>x.key===key))state.learning.push({...(v09sCfg(key)||{key}),ok:false,error:'process WebKit/iOS interrompu pendant apprentissage',processCrash:true});
    state.learningIndex=Math.max(state.learningIndex,Number(marker.index||0)+1);
  }
  v09sClearMarker();v09sSave(state);
  v09sLog(`crash précédent détecté: ${key} blacklisté et reprise automatique.`);
  if(state.runId)await api('/metric',{method:'POST',body:JSON.stringify({runId:state.runId,name:'autotune_process_crash',value:1,globalStep:Number(marker.index||0),payload:{key,phase:marker.phase,detectedAt:new Date().toISOString()}})}).catch(()=>{});
  return state;
}

async function v09sStartOrResume(){
  if(running)return;running=true;v09sStop=false;ui.start.disabled=true;ui.pause.disabled=false;
  let state=null;
  try{
    await claimIfNeeded();await verifyBuild();await gpuGate();
    const locked=await requestWakeLock();if(!locked)throw new Error('Wake Lock non acquis');
    const corpus=await loadCorpus();
    const adapter=await v09AdapterInfo();
    state=v09sLoad()||v09sFresh(adapter);state.adapter=adapter;state=await v09sRecoverCrash(state);
    if(!state.runId){
      const rr=await api('/run/create',{method:'POST',body:JSON.stringify({experimentKey:'hardware-autotune-v09-safe',engineVersion:V09S_ENGINE,config:{candidates:V09S_CANDIDATES,knownUnsafe:[...V09S_KNOWN_UNSAFE],adapter,crashResume:true},progress:{stage:'hardware-probe',resumeSafe:true}})});
      state.runId=rr.run.id;v09sSave(state);v09sLog(`run ${state.runId} créé.`);
    }

    state.phase='hardware';v09sSave(state);
    for(let i=state.hardwareIndex;i<V09S_CANDIDATES.length;i++){
      if(v09sStop)throw new Error('pause demandée');
      const cfg=V09S_CANDIDATES[i];
      if(state.blacklist.includes(cfg.key)){
        if(!state.hardware.some(x=>x.key===cfg.key))state.hardware.push({...cfg,ok:false,skipped:true,error:'profil blacklisté après crash WebKit'});
        state.hardwareIndex=i+1;v09sSave(state);continue;
      }
      v09sMarker({phase:'hardware',index:i,key:cfg.key,at:Date.now()});
      state.hardwareIndex=i;v09sSave(state);
      const r=await v09HardwareProbe(cfg,corpus.train,i,V09S_CANDIDATES.length);
      v09sClearMarker();state.hardware=state.hardware.filter(x=>x.key!==cfg.key);state.hardware.push(r);state.hardwareIndex=i+1;v09sSave(state);
      v09sLog(`${cfg.key}: ${r.ok?`${Math.round(r.tokensPerSec)} tok/s · ${r.medianStepMs.toFixed(0)} ms`:`échec ${r.error}`}`);
      await api('/metric',{method:'POST',body:JSON.stringify({runId:state.runId,name:'hardware_probe_tokens_per_sec',value:r.ok?r.tokensPerSec:null,globalStep:i,payload:{...r,resumeSafe:true}})}).catch(()=>{});
      await api('/run/update',{method:'POST',body:JSON.stringify({runId:state.runId,status:'training',global_step:i+1,progress:{stage:'hardware-probe',next:i+1,last:cfg.key}})}).catch(()=>{});
    }

    const successful=state.hardware.filter(x=>x.ok&&Number.isFinite(x.tokensPerSec)&&!state.blacklist.includes(x.key)).sort((a,b)=>b.tokensPerSec-a.tokensPerSec);
    if(successful.length<2)throw new Error(`Seulement ${successful.length} profil(s) WebGPU stable(s).`);
    if(!state.selected.length){
      const selected=[];
      for(const r of successful){if(selected.length>=3)break;if((r.slowdown??1)<1.35&&!selected.some(x=>x.hidden===r.hidden&&x.seq===r.seq))selected.push(r);}
      for(const r of successful){if(selected.length>=3)break;if(!selected.some(x=>x.key===r.key))selected.push(r);}
      state.selected=selected.map(x=>x.key);state.phase='learning';state.learningIndex=0;v09sSave(state);
      v09sLog(`apprentissage retenu: ${state.selected.join(', ')}.`);
    }

    state.phase='learning';v09sSave(state);
    await api('/run/update',{method:'POST',body:JSON.stringify({runId:state.runId,status:'training',progress:{stage:'learning-probe',selected:state.selected}})}).catch(()=>{});
    for(let i=state.learningIndex;i<state.selected.length;i++){
      if(v09sStop)throw new Error('pause demandée');
      const cfg=v09sCfg(state.selected[i]);if(!cfg){state.learningIndex=i+1;v09sSave(state);continue;}
      if(state.blacklist.includes(cfg.key)){state.learningIndex=i+1;v09sSave(state);continue;}
      v09sMarker({phase:'learning',index:i,key:cfg.key,at:Date.now()});state.learningIndex=i;v09sSave(state);
      const r=await v09LearningProbe(cfg,corpus,i,state.selected.length);
      v09sClearMarker();state.learning=state.learning.filter(x=>x.key!==cfg.key);state.learning.push(r);state.learningIndex=i+1;v09sSave(state);
      await api('/metric',{method:'POST',body:JSON.stringify({runId:state.runId,name:'learning_probe_gain_bpb',value:r.ok?r.improvementBpb:null,globalStep:i,payload:{...r,resumeSafe:true}})}).catch(()=>{});
    }

    const good=state.learning.filter(x=>x.ok&&Number.isFinite(x.score)).sort((a,b)=>b.score-a.score);if(!good.length)throw new Error('Aucun profil d’apprentissage stable.');
    const winner=good[0];
    const report={engine:V09S_ENGINE,createdAt:new Date().toISOString(),adapter:state.adapter,hardware:state.hardware,learning:state.learning,blacklist:state.blacklist,recommended:{key:winner.key,embed:winner.embed,hidden:winner.hidden,seq:winner.seq,batch:winner.batch,tokensPerSec:winner.tokensPerSec,initialValidationBpb:winner.initialValidationBpb,finalValidationBpb:winner.finalValidationBpb,improvementBpb:winner.improvementBpb},crashResume:true,note:'Profil matériel recommandé; aucun profil ayant provoqué un crash WebKit n’est retesté.'};
    await dbPut('v09:hardwareProfile',report);const payload=JSON.stringify(report);const hash=await sha256Text(payload);
    await api('/artifact',{method:'POST',body:JSON.stringify({runId:state.runId,kind:'hardware-profile-v09-safe',payload,sha256:hash,promote:false,metadata:{recommended:winner.key,blacklist:state.blacklist}})});
    await api('/run/update',{method:'POST',body:JSON.stringify({runId:state.runId,status:'completed',global_step:V09S_CANDIDATES.length+state.selected.length,progress:{stage:'completed',recommended:winner.key,blacklist:state.blacklist},completed_at:new Date().toISOString()})});
    localStorage.removeItem(V09S_STATE);v09sClearMarker();pct(100);ui.sync.textContent='Supabase ✓';ui.experiment.textContent=winner.key;ui.train.textContent=`${Math.round(winner.tokensPerSec)} tok/s`;ui.val.textContent=fmt(winner.finalValidationBpb);ui.best.textContent=fmt(winner.improvementBpb);
    v09sStatus('v0.9.2 Autotune terminé',`Profil recommandé: ${winner.key} · ${Math.round(winner.tokensPerSec)} tokens/s · gain court ${winner.improvementBpb.toFixed(4)} bpb.`);
  }catch(e){
    const msg=e instanceof Error?e.message:String(e);v09sStatus(v09sStop?'v0.9.2 en pause':'Arrêt sécurisé v0.9.2',msg);v09sLog(`ERREUR: ${e?.stack||e}`);
    if(state?.runId)await api('/run/update',{method:'POST',body:JSON.stringify({runId:state.runId,status:v09sStop?'paused':'failed',progress:{error:msg,resumable:true}})}).catch(()=>{});
  }finally{running=false;ui.start.disabled=false;ui.pause.disabled=true;await releaseWakeLock();}
}

(function v09sInstall(){
  const s=ui.start.cloneNode(true),p=ui.pause.cloneNode(true);ui.start.replaceWith(s);ui.pause.replaceWith(p);ui.start=s;ui.pause=p;
  ui.start.textContent='Lancer v0.9.2 Autotune sûr';ui.pause.textContent='Pause sûre';
  ui.start.addEventListener('click',v09sStartOrResume);ui.pause.addEventListener('click',()=>{v09sStop=true;ui.pause.disabled=true;v09sStatus('Pause demandée v0.9.2','Fin de l’opération WebGPU en cours puis sauvegarde de la progression.');});
  setTimeout(()=>{
    const marker=v09sReadMarker(),st=v09sLoad();
    if(marker)v09sStatus('Crash GPU détecté',`${marker.key} sera automatiquement exclu; touche « Lancer » pour reprendre après ce profil.`);
    else if(st?.runId)v09sStatus('v0.9.2 reprise prête',`Progression autotune sauvegardée · prochain profil ${st.hardwareIndex+1}/${V09S_CANDIDATES.length}.`);
    else v09sStatus('v0.9.2 Hardware Autotune sûr prêt','Exploration WebGPU progressive avec reprise après crash et blacklist automatique des profils incompatibles.');
    ui.experiment.textContent=st?.runId?'Autotune reprise':'Autotune sûr';
  },700);
})();
