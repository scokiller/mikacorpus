'use strict';
(()=>{
  const ENDPOINT='https://xifxrkjvsrzexmuqtsvw.supabase.co/functions/v1/mikavideo-private/module?name=mikavideo-v10';
  function describeError(e){
    try{
      if(e instanceof Error)return e.message||e.name||e.stack||'Error sans message';
      if(typeof e==='string')return e||'Erreur chaîne vide';
      if(e===null)return 'Erreur null';
      if(e===undefined)return 'Erreur undefined';
      const j=JSON.stringify(e);return j&&j!=='{}'?j:String(e);
    }catch{return String(e)}
  }
  async function boot(){
    try{
      const cap=await dbGet('capability');if(!cap)return;
      let r;
      try{r=await fetch(ENDPOINT,{headers:{'x-mika-capability':cap},cache:'no-store'})}
      catch(e){throw new Error(`fetch Structure Direct Lab: ${describeError(e)}`)}
      if(!r.ok){const body=await r.text().catch(()=> '');throw new Error(`module privé HTTP ${r.status}${body?` · ${body}`:''}`)}
      const bytes=new Uint8Array(await r.arrayBuffer());
      const expected=r.headers.get('x-mikavideo-sha256')||'';
      const actual=await sha256Bytes(bytes);
      if(!expected)throw new Error('header SHA Structure Direct absent');
      if(actual!==expected)throw new Error(`SHA-256 Structure Direct refusé · attendu ${expected.slice(0,12)} · reçu ${actual.slice(0,12)}`);
      const src=new TextDecoder().decode(bytes);
      try{(new Function(src))()}catch(e){throw new Error(`exécution module: ${describeError(e)}`)}
      try{log(`Mika Structure Direct v1.0 privé chargé · r${r.headers.get('x-mikavideo-revision')||'?'} · SHA ✓`)}catch{}
    }catch(e){try{log(`Mika Structure Direct non chargé: ${describeError(e)}`)}catch{}}
  }
  setTimeout(boot,1500);
})();
