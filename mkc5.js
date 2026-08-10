'use strict';
(function(global){
  const MAGIC='MKC5', VERSION=1, HEADER_BYTES=84;
  const METHOD_RAW=0, METHOD_BROTLI=1, METHOD_ZSTD=2, METHOD_LZMA=3, METHOD_NEURAL=4;
  const DEFAULT_TOTAL_FREQ=1<<20;
  const FULL=4294967296, MASK=FULL-1, HALF=2147483648, QUARTER=1073741824, THREE=3221225472;

  class BitOutput{
    constructor(){this.data=[];this.current=0;this.count=0;}
    write(bit){this.current=this.current*2+(bit&1);this.count++;if(this.count===8){this.data.push(this.current);this.current=0;this.count=0;}}
    finish(){if(this.count){this.current*=2**(8-this.count);this.data.push(this.current);}return new Uint8Array(this.data);}
  }
  class BitInput{
    constructor(data){this.data=data;this.bitIndex=0;}
    read(){const bi=Math.floor(this.bitIndex/8);if(bi>=this.data.length){this.bitIndex++;return 0;}const shift=7-(this.bitIndex%8);this.bitIndex++;return (this.data[bi]>>shift)&1;}
  }
  class ArithmeticEncoder{
    constructor(){this.low=0;this.high=MASK;this.pending=0;this.out=new BitOutput();}
    emit(bit){this.out.write(bit);for(let i=0;i<this.pending;i++)this.out.write(bit^1);this.pending=0;}
    write(cum,symbol){
      const total=cum[cum.length-1];const range=this.high-this.low+1;const old=this.low;
      this.low=old+Math.floor(cum[symbol]*range/total);
      this.high=old+Math.floor(cum[symbol+1]*range/total)-1;
      for(;;){
        if(this.high<HALF)this.emit(0);
        else if(this.low>=HALF){this.emit(1);this.low-=HALF;this.high-=HALF;}
        else if(this.low>=QUARTER&&this.high<THREE){this.pending++;this.low-=QUARTER;this.high-=QUARTER;}
        else break;
        this.low=(this.low*2)%FULL;this.high=(this.high*2+1)%FULL;
      }
    }
    finish(){this.pending++;this.emit(this.low<QUARTER?0:1);return this.out.finish();}
  }
  class ArithmeticDecoder{
    constructor(data){this.inp=new BitInput(data);this.low=0;this.high=MASK;this.code=0;for(let i=0;i<32;i++)this.code=(this.code*2+this.inp.read())%FULL;}
    read(cum){
      const total=cum[cum.length-1],range=this.high-this.low+1;
      const value=Math.floor(((this.code-this.low+1)*total-1)/range);
      let lo=0,hi=cum.length-1;while(lo+1<hi){const m=(lo+hi)>>1;if(cum[m]<=value)lo=m;else hi=m;}const symbol=lo;
      const old=this.low;this.low=old+Math.floor(cum[symbol]*range/total);this.high=old+Math.floor(cum[symbol+1]*range/total)-1;
      for(;;){
        if(this.high<HALF){}
        else if(this.low>=HALF){this.low-=HALF;this.high-=HALF;this.code-=HALF;}
        else if(this.low>=QUARTER&&this.high<THREE){this.low-=QUARTER;this.high-=QUARTER;this.code-=QUARTER;}
        else break;
        this.low=(this.low*2)%FULL;this.high=(this.high*2+1)%FULL;this.code=(this.code*2+this.inp.read())%FULL;
      }
      return symbol;
    }
  }
  function uniformCum(total=DEFAULT_TOTAL_FREQ){const c=new Float64Array(257);const base=Math.floor(total/256),rem=total-base*256;let s=0;c[0]=0;for(let i=0;i<256;i++){s+=base+(i<rem?1:0);c[i+1]=s;}return c;}
  function probsToCum(probs,total=DEFAULT_TOTAL_FREQ){
    if(total<=256)throw new Error('total frequency must exceed 256');
    const scaled=new Float64Array(256),floors=new Int32Array(256),freq=new Int32Array(256);let sum=0;
    for(let i=0;i<256;i++){scaled[i]=Math.max(0,probs[i])*(total-256);floors[i]=Math.floor(scaled[i]);freq[i]=floors[i]+1;sum+=freq[i];}
    let remaining=total-sum;
    if(remaining>0){const idx=Array.from({length:256},(_,i)=>i).sort((a,b)=>(scaled[b]-floors[b])-(scaled[a]-floors[a])||a-b);for(let k=0;k<remaining;k++)freq[idx[k%256]]++;}
    const cum=new Float64Array(257);let s=0;for(let i=0;i<256;i++){cum[i]=s;s+=freq[i];}cum[256]=s;if(s!==total)throw new Error('invalid CDF total');return cum;
  }
  function logitsToCum(logits,total=DEFAULT_TOTAL_FREQ){let max=-Infinity;for(const v of logits)if(v>max)max=v;const p=new Float64Array(256);let sum=0;for(let i=0;i<256;i++){p[i]=Math.exp(logits[i]-max);sum+=p[i];}for(let i=0;i<256;i++)p[i]/=sum;return probsToCum(p,total);}
  async function sha256Bytes(bytes){return new Uint8Array(await crypto.subtle.digest('SHA-256',bytes));}
  function concat(list){let n=0;for(const a of list)n+=a.length;const out=new Uint8Array(n);let p=0;for(const a of list){out.set(a,p);p+=a.length;}return out;}
  function setU64(view,off,n){view.setBigUint64(off,BigInt(n),false);}
  function getU64(view,off){const n=view.getBigUint64(off,false);if(n>BigInt(Number.MAX_SAFE_INTEGER))throw new Error('MKC5 length exceeds JS safe integer');return Number(n);}
  async function packHeader(method,original,payloadLen,{modelTag=new Uint8Array(16),blockSize=0,totalFrequency=0,seedLen=0}={}){
    const h=new Uint8Array(HEADER_BYTES),v=new DataView(h.buffer);h.set(new TextEncoder().encode(MAGIC),0);v.setUint8(4,VERSION);v.setUint8(5,method);v.setUint16(6,0,false);setU64(v,8,original.length);setU64(v,16,payloadLen);h.set(await sha256Bytes(original),24);h.set(modelTag.subarray(0,16),56);v.setUint32(72,blockSize,false);v.setUint32(76,totalFrequency,false);v.setUint32(80,seedLen,false);return h;
  }
  function parseHeader(archive){if(archive.length<HEADER_BYTES)throw new Error('truncated MKC5');const v=new DataView(archive.buffer,archive.byteOffset,archive.byteLength);const magic=new TextDecoder().decode(archive.subarray(0,4));if(magic!==MAGIC||v.getUint8(4)!==VERSION)throw new Error('unsupported MKC5');return {method:v.getUint8(5),originalLength:getU64(v,8),payloadLength:getU64(v,16),sha:archive.slice(24,56),modelTag:archive.slice(56,72),blockSize:v.getUint32(72,false),totalFrequency:v.getUint32(76,false),seedLen:v.getUint32(80,false)};}
  async function buildRawArchive(data){const header=await packHeader(METHOD_RAW,data,data.length);return concat([header,data]);}
  async function restoreRawArchive(archive){const h=parseHeader(archive);if(h.method!==METHOD_RAW)throw new Error('self-test expects raw');const payload=archive.subarray(HEADER_BYTES);if(payload.length!==h.payloadLength||payload.length!==h.originalLength)throw new Error('payload length mismatch');const got=await sha256Bytes(payload);for(let i=0;i<32;i++)if(got[i]!==h.sha[i])throw new Error('SHA-256 verification failed');return new Uint8Array(payload);}
  async function selfTest(){
    const src=new Uint8Array(2048);let s=0x19850212>>>0;for(let i=0;i<src.length;i++){s=(Math.imul(s^s>>>15,2246822519)+3266489917)>>>0;src[i]=(s>>>8)&255;}
    const cum=uniformCum(),enc=new ArithmeticEncoder();for(const b of src)enc.write(cum,b);const bits=enc.finish();const dec=new ArithmeticDecoder(bits),out=new Uint8Array(src.length);for(let i=0;i<out.length;i++)out[i]=dec.read(cum);for(let i=0;i<src.length;i++)if(src[i]!==out[i])throw new Error(`arithmetic round-trip failed @${i}`);
    const arc=await buildRawArchive(src),restored=await restoreRawArchive(arc);for(let i=0;i<src.length;i++)if(src[i]!==restored[i])throw new Error(`MKC5 raw round-trip failed @${i}`);
    return {ok:true,headerBytes:HEADER_BYTES,arithmeticBytes:bits.length,rawArchiveBytes:arc.length,sha256:[...await sha256Bytes(src)].map(x=>x.toString(16).padStart(2,'0')).join('')};
  }
  global.MKC5={MAGIC,VERSION,HEADER_BYTES,METHOD_RAW,METHOD_BROTLI,METHOD_ZSTD,METHOD_LZMA,METHOD_NEURAL,DEFAULT_TOTAL_FREQ,ArithmeticEncoder,ArithmeticDecoder,uniformCum,probsToCum,logitsToCum,packHeader,parseHeader,buildRawArchive,restoreRawArchive,selfTest};
})(globalThis);
