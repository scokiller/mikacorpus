const CACHE='mikacorpus-0.8-web.5-r11';const SHELL=["./","./index.html","./app.js","./v08.js","./mkc5.js","./build.json","./manifest.webmanifest","./icon-192.png","./icon-512.png","./pre-tf-r4.js","./vendor/tfjs-4.22.0-r4.min.js","./post-tf-core-r4.js","./vendor/tfjs-backend-webgpu-4.22.0-r4.js","./post-tf-webgpu-r4.js"];const FRESH=new Set(['/','/index.html','/app.js','/v08.js','/mkc5.js','/build.json','/manifest.webmanifest','/sw.js']);self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL)).then(()=>self.skipWaiting())));self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==CACHE).map(x=>caches.delete(x)))).then(()=>self.clients.claim())));self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;const u=new URL(e.request.url);if(u.origin===location.origin&&FRESH.has(u.pathname)){e.respondWith((async()=>{const c=await caches.open(CACHE);try{const r=await fetch(e.request,{cache:'no-store'});if(r.ok)await c.put(e.request,r.clone());return r}catch{const h=await c.match(e.request);if(h)return h;throw new Error('offline')}})());return}e.respondWith((async()=>{const c=await caches.open(CACHE);const h=await c.match(e.request);if(h)return h;const r=await fetch(e.request);if(r.ok&&u.origin===location.origin)try{await c.put(e.request,r.clone())}catch{}return r})())});
// MikaCorpus R7 ByteGRU reset-before 2026-08-07T11:14:17.475Z

// MikaCorpus SW R8 2026-08-07T11:21:54.704Z

// MikaCorpus service worker R10

// R11 numeric stability
