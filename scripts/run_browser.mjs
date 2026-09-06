// Runs one standalone test/profile in its own visible Chrome tab; saves results.
// Usage: node scripts/run_browser.mjs URL OUTPUT_JSON [CDP_PORT]
import {writeFile} from 'node:fs/promises';
const [url,output,port='9333']=process.argv.slice(2);
if(!url||!output)throw Error('Pass URL and output JSON');
const base=`http://127.0.0.1:${port}`;
const page=await(await fetch(`${base}/json/new?about:blank`,{method:'PUT'})).json();
const ws=new WebSocket(page.webSocketDebuggerUrl),pending=new Map();let id=0,success=false;
await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject;});
ws.onmessage=e=>{const r=JSON.parse(e.data),p=pending.get(r.id);if(!p)return;pending.delete(r.id);clearTimeout(p.timer);
  if(r.error||r.result?.exceptionDetails)p.reject(Error(JSON.stringify(r.error||r.result.exceptionDetails)));else p.resolve(r.result);};
const call=(method,params={})=>new Promise((resolve,reject)=>{const i=++id,timer=setTimeout(()=>{pending.delete(i);reject(Error('CDP timeout'));},30000);
  pending.set(i,{resolve,reject,timer});ws.send(JSON.stringify({id:i,method,params}));});
const evaluate=async expression=>(await call('Runtime.evaluate',{expression,returnByValue:true})).result.value;
try{
  await call('Page.enable');await call('Page.navigate',{url});await call('Page.bringToFront');
  console.log(JSON.stringify({target:page.id,url}));let started=false;
  const deadline=Date.now()+600000;
  while(Date.now()<deadline){
    const state=await evaluate('({ready:document.readyState==="complete"&&location.href!=="about:blank",status:document.body?.dataset.status})');
    if(state.ready&&!started){await evaluate('document.getElementById("start")?.click()');started=true;}
    if(['pass','fail'].includes(state.status)){
      const result=await evaluate('({status:document.body.dataset.status,log:document.getElementById("log")?.textContent,results:window.fftBenchmarkResults,userAgent:navigator.userAgent})');
      await writeFile(output,JSON.stringify(result,null,2));
      console.log(JSON.stringify({status:result.status,output,tail:result.log?.slice(-500)}));
      if(state.status==='fail')throw Error('Browser test failed');success=true;break;
    }
    await new Promise(resolve=>setTimeout(resolve,1000));
  }
  if(!success)throw Error('Browser test timed out');
}finally{if(!success)await call('Page.navigate',{url:'about:blank'}).catch(()=>{});ws.close();}
