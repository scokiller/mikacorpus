'use strict';
(()=>{
  const ENDPOINT='https://xifxrkjvsrzexmuqtsvw.supabase.co/functions/v1/mikavideo-private/module?name=mikavideo-v01';
  async function boot(){
    try{
      const cap=await dbGet('capability');if(!cap)return;
      const r=await fetch(ENDPOINT,{headers:{'x-mika-capability':cap},cache:'no-store'});if(!r.ok)throw new Error(`module privé HTTP ${r.status}`);
      const src=await r.text(),expected=r.headers.get('x-mikavideo-sha256')||'';
      const actual=await sha256Text(src);if(!expected||actual!==expected)throw new Error('SHA-256 module vidéo refusé');
      (new Function(src))();
      try{log(`Mika Video privé chargé · r${r.headers.get('x-mikavideo-revision')||'?'} · SHA ✓`);}catch{}
    }catch(e){try{log(`Mika Video privé non chargé: ${e instanceof Error?e.message:String(e)}`);}catch{}}
  }
  setTimeout(boot,1500);
})();
