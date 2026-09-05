// Optional Chrome DevTools test driver. Chrome must expose a debugging port.
const [command, target, expression] = process.argv.slice(2);
const port = process.env.CDP_PORT || '9333';
const base = `http://127.0.0.1:${port}`;
if (command === 'open') {
  console.log(await (await fetch(`${base}/json/new?${encodeURIComponent(target)}`, {method: 'PUT'})).text());
} else {
  const pages = await (await fetch(`${base}/json/list`)).json();
  const page = pages.find(page => page.id === target);
  if (!page) throw Error(`No Chrome target ${target}`);
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  const timeout = setTimeout(() => {console.error('CDP timeout');process.exit(2);}, 60000);
  ws.onopen = () => ws.send(JSON.stringify({id: 1, method: 'Runtime.evaluate',
    params: {expression, returnByValue: true, awaitPromise: true}}));
  ws.onmessage = event => {
    const reply = JSON.parse(event.data);
    if (reply.id !== 1) return;
    console.log(JSON.stringify(reply.result || reply.error));
    clearTimeout(timeout);ws.close();
  };
}
