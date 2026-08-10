'use strict';

// Single-owner gate for the installed Mika Corpus PWA.
// The one-time claim secret is never embedded in the static site.
claimIfNeeded = async function privateClaimIfNeeded(){
  const restored = await restoreCapability();
  if(restored) return {cap:restored.cap,newlyClaimed:false};

  const q = new URLSearchParams(location.search);
  let bootstrap = q.get('claim');
  if(!bootstrap){
    bootstrap = window.prompt('Mika Corpus · accès privé\n\nColle le code d’association à usage unique.');
  }
  bootstrap = String(bootstrap || '').trim();
  if(!bootstrap) throw new Error('Association requise. Seul l’iPhone autorisé peut lancer Mika Corpus.');

  let install = await dbGet('installationId');
  if(!install){
    install = crypto.randomUUID();
    await dbPut('installationId', install);
  }

  setStatus('Association sécurisée','Réassociation de cet iPhone au coffre Supabase existant…');
  const r = await api('/claim',{
    method:'POST',
    body:JSON.stringify({
      bootstrap,
      installationId:install,
      label:`iPhone PWA ${navigator.platform || ''}`
    })
  });

  await dbPut('capability',r.capability);
  await dbPut('deviceId',r.deviceId);
  history.replaceState(null,'',location.pathname+location.hash);
  log('Accès propriétaire associé à cet iPhone. Secret stocké uniquement dans IndexedDB.');
  ui.start.textContent='Lancer v0.8 / reprendre';
  return {cap:r.capability,newlyClaimed:true};
};

setTimeout(async()=>{
  try{
    const paired = await restoreCapability();
    if(!paired){
      setStatus('Accès privé','Cet iPhone doit être associé une seule fois avant le calcul.');
      ui.start.textContent='Associer cet iPhone';
      ui.sync.textContent='Verrouillé';
      ui.experiment.textContent='Privé';
    }
  }catch{}
},350);
