'use strict';

const API_BASE='https://xifxrkjvsrzexmuqtsvw.supabase.co/functions/v1/mikacorpus-api';
const HANDOFF_COOKIE='mikacorpus_handoff';
const $=id=>document.getElementById(id);
const ui={
  status:$('status'),detail:$('detail'),gpu:$('gpu'),progress:$('progress'),
  experiment:$('experiment'),epoch:$('epoch'),step:$('step'),train:$('train'),val:$('val'),best:$('best'),
  sync:$('sync'),start:$('start'),pause:$('pause'),log:$('log'),install:$('install')
};

function log(msg){
  const t=new Date().toLocaleTimeString('fr-FR');
  ui.log.textContent=`[${t}] ${msg}\n`+ui.log.textContent.slice(0,7000);
}
function setStatus(title,detail=''){
  ui.status.textContent=title;ui.detail.textContent=detail;
  if(/terminé|arrêté|erreur/i.test(String(title))) releaseWakeLock().catch(()=>{});
}
function pct(v){ui.progress.style.width=`${Math.max(0,Math.min(100,Number(v)||0))}%`}
function sleep(ms){return new Promise(r=>setTimeout(r,ms))}
async function sha256Bytes(bytes){
  const d=new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));
  return [...d].map(x=>x.toString(16).padStart(2,'0')).join('');
}
async function sha256Text(s){return sha256Bytes(new TextEncoder().encode(s))}

function openDB(){
  return new Promise((resolve,reject)=>{
    const r=indexedDB.open('mikacorpus-mcweb1',1);
    r.onupgradeneeded=()=>{const db=r.result;if(!db.objectStoreNames.contains('kv'))db.createObjectStore('kv')};
    r.onsuccess=()=>resolve(r.result);r.onerror=()=>reject(r.error);
  });
}
async function dbGet(key){const db=await openDB();return new Promise((res,rej)=>{const tx=db.transaction('kv','readonly');const r=tx.objectStore('kv').get(key);r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error)})}
async function dbPut(key,val){const db=await openDB();return new Promise((res,rej)=>{const tx=db.transaction('kv','readwrite');tx.objectStore('kv').put(val,key);tx.oncomplete=()=>res();tx.onerror=()=>rej(tx.error)})}

async function api(path,opts={}){
  const capability=await dbGet('capability');
  const headers={'content-type':'application/json',...(opts.headers||{})};
  if(capability)headers['x-mika-capability']=capability;
  const r=await fetch(`${API_BASE}${path}`,{...opts,headers,cache:'no-store'});
  const text=await r.text();let data;try{data=JSON.parse(text)}catch{data={raw:text}};
  if(!r.ok)throw new Error(`${r.status} ${data?.error||data?.raw||r.statusText}`);
  return data;
}
function readHandoffCookie(){const prefix=HANDOFF_COOKIE+'=';for(const part of document.cookie.split(';')){const v=part.trim();if(v.startsWith(prefix))return decodeURIComponent(v.slice(prefix.length))}return null}
function clearHandoffCookie(){document.cookie=`${HANDOFF_COOKIE}=; Path=/; Max-Age=0; SameSite=Strict; Secure`}
async function restoreCapability(){
  let cap=await dbGet('capability');if(cap)return {cap,source:'indexeddb'};
  cap=readHandoffCookie();if(cap){await dbPut('capability',cap);clearHandoffCookie();return {cap,source:'cookie'}}
  return null;
}
async function claimIfNeeded(){
  const restored=await restoreCapability();if(restored)return {cap:restored.cap,newlyClaimed:false};
  const q=new URLSearchParams(location.search);let bootstrap=String(q.get('claim')||'').trim();
  if(!bootstrap)bootstrap=String(window.prompt('Mika Video Lab · accès privé\n\nColle le code d’association à usage unique.')||'').trim();
  if(!bootstrap)throw new Error('Association privée requise.');
  let install=await dbGet('installationId');if(!install){install=crypto.randomUUID();await dbPut('installationId',install)}
  setStatus('Association sécurisée','Association de cet iPhone au laboratoire privé…');
  const r=await api('/claim',{method:'POST',body:JSON.stringify({bootstrap,installationId:install,label:`iPhone PWA ${navigator.platform||''}`})});
  await dbPut('capability',r.capability);await dbPut('deviceId',r.deviceId);history.replaceState(null,'',location.pathname+location.hash);
  log('iPhone associé au laboratoire privé.');return {cap:r.capability,newlyClaimed:true};
}

let mikaWake=null,mikaWakeWanted=false;
async function requestWakeLock(){
  mikaWakeWanted=true;
  if(document.visibilityState!=='visible'||!navigator.wakeLock?.request)return false;
  if(mikaWake&&!mikaWake.released)return true;
  try{
    mikaWake=await navigator.wakeLock.request('screen');
    mikaWake.addEventListener('release',()=>{mikaWake=null;if(mikaWakeWanted&&document.visibilityState==='visible')setTimeout(()=>requestWakeLock().catch(()=>{}),500)},{once:true});
    return true;
  }catch{return false}
}
async function releaseWakeLock(){mikaWakeWanted=false;const w=mikaWake;mikaWake=null;try{if(w&&!w.released)await w.release()}catch{}}
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='visible'&&mikaWakeWanted)requestWakeLock().catch(()=>{})});
document.addEventListener('click',e=>{if(e.target?.id==='start')requestWakeLock().catch(()=>{})},true);

let deferredInstall=null;
window.addEventListener('beforeinstallprompt',e=>{e.preventDefault();deferredInstall=e;if(ui.install)ui.install.hidden=false});
ui.install?.addEventListener('click',async()=>{if(!deferredInstall)return;deferredInstall.prompt();deferredInstall=null;ui.install.hidden=true});

setStatus('Mika Video Lab','Chargement du module neural privé actif…');
ui.experiment.textContent='Neural Restore v0.8';
ui.sync.textContent='Privé';
ui.gpu.textContent=navigator.gpu?'WebGPU détecté · initialisation neural…':'WebGPU absent';
ui.pause.disabled=true;ui.pause.textContent='—';
log('Boot unique Mika Video Lab R31. Aucun ancien moteur lossless n’est exécuté.');

if('serviceWorker' in navigator){
  window.addEventListener('load',async()=>{
    try{const reg=await navigator.serviceWorker.register('./sw.js',{updateViaCache:'none'});await reg.update()}catch(e){log(`Service worker: ${e instanceof Error?e.message:String(e)}`)}
  });
}
