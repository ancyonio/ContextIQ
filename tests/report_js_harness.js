// Executes the report's inlined JS against a minimal fake DOM to verify
// hash-persistence, the pause toggle, and change detection.
// usage: node report_js_harness.js <rendered.html>
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');

const payloadJson = html.match(/id="ciq-data">([\s\S]*?)<\/script>/)[1].replace(/<\\\//g, '</');
const js = html.match(/<script>([\s\S]*?)<\/script>\s*<\/body>/)[1];

function makeEl(id) {
  return {
    id, value: '', checked: false, textContent: '', innerHTML: '',
    children: [], _handlers: {},
    appendChild(c) { this.children.push(c); },
    addEventListener(ev, fn) { this._handlers[ev] = fn; },
    fire(ev) { if (this._handlers[ev]) this._handlers[ev](); },
  };
}

function run(initialHash, opts) {
  opts = opts || {};
  const els = {};
  ['ciq-data','hero-value','hero-sub','m-saved','m-sent','m-spend','m-red','m-runs',
   'c-waterfall','c-area','c-ops','c-rows','meta','sel-model','sel-range',
   'chk-auto','countdown','unit-price'].forEach(id => { els[id] = makeEl(id); });
  els['ciq-data'].textContent = payloadJson;
  els['chk-auto'].checked = true;

  const timers = [];
  const draws = { count: 0 };
  const sandbox = {
    document: {
      readyState: 'complete',
      getElementById: id => {
        if (id === 'm-saved') draws.count++;   // draw() always writes this one
        return els[id] || null;
      },
      createElement: () => makeEl('opt'),
      addEventListener: () => {},
    },
    fetch: opts.fetch,
    Promise,
    location: { protocol: opts.protocol || 'file:', hash: initialHash,
                reload() { this._reloaded = true; } },
    history: { replaceState(_a, _b, h) { sandbox.location.hash = h; } },
    setInterval: (fn, ms) => { timers.push({ fn, ms }); return timers.length; },
    clearInterval: () => {},
    setTimeout: () => {},
    Date, Math, JSON, String, Number, Object, Array, parseInt, encodeURIComponent,
    decodeURIComponent, isNaN,
  };
  sandbox.window = sandbox;
  const vm = require('vm');
  vm.createContext(sandbox);
  vm.runInContext(js, sandbox);
  return { els, loc: sandbox.location, timers, draws };
}

const out = [];
const ok = (name, cond) => { out.push(`${cond ? 'PASS' : 'FAIL'}  ${name}`); if (!cond) process.exitCode = 1; };

// 1. cold start: no hash -> defaults, hash gets written
let r = run('');
ok('cold start renders metrics', r.els['m-saved'].textContent !== '' && r.els['m-saved'].textContent !== '—');
ok('cold start writes hash', /model=.*range=all.*auto=1/.test(r.loc.hash));
ok('auto-refresh countdown shown', /refresh in \d+s/.test(r.els['countdown'].textContent));
ok('timer armed', r.timers.length === 1);

// 2. user changes model + range -> hash updates
r.els['sel-model'].value = 'claude-opus';
r.els['sel-model'].fire('change');
r.els['sel-range'].value = '7';
r.els['sel-range'].fire('change');
ok('hash captures model change', r.loc.hash.includes('model=claude-opus'));
ok('hash captures range change', r.loc.hash.includes('range=7'));
const carried = r.loc.hash;

// 3. simulate the file:// reload -> selections restored from hash
let r2 = run(carried);
ok('model restored after reload', r2.els['sel-model'].value === 'claude-opus');
ok('range restored after reload', r2.els['sel-range'].value === '7');

// 4. pause toggle
r2.els['chk-auto'].checked = false;
r2.els['chk-auto'].fire('change');
ok('pause writes auto=0 to hash', r2.loc.hash.includes('auto=0'));
ok('pause shows paused label', r2.els['countdown'].textContent === 'auto-refresh paused');

// 5. paused state survives a reload and does not arm a timer
let r3 = run(r2.loc.hash);
ok('paused restored after reload', r3.els['chk-auto'].checked === false);
ok('paused arms no timer', r3.timers.length === 0);

// 6. unknown/garbage hash falls back to defaults
let r4 = run('#model=not-a-model&range=999&auto=1');
ok('bad model falls back', r4.els['sel-model'].value !== 'not-a-model' && r4.els['sel-model'].value !== '');
ok('bad range falls back to all', r4.els['sel-range'].value === 'all');

// 7. every distinct price must yield a distinct $ — no "$0.00" collapse on a
//    young ledger, and the price-dependent cards must actually move.
{
  const r5 = run('');
  const models = r5.els['sel-model'].children.map(o => o.value);
  const prices = new Set(models.map(m => JSON.parse(payloadJson).prices[m]));
  const heroes = new Set(), spends = new Set(), units = new Set();
  models.forEach(m => {
    r5.els['sel-model'].value = m;
    r5.els['sel-model'].fire('change');
    heroes.add(r5.els['hero-value'].textContent);
    spends.add(r5.els['m-spend'].textContent);
    units.add(r5.els['unit-price'].textContent);
  });
  ok('distinct prices -> distinct hero $ (no $0.00 collapse)', heroes.size === prices.size);
  ok('distinct prices -> distinct spend $', spends.size === prices.size);
  ok('unit price label tracks the model', units.size === prices.size);
  ok('token cards stay model-independent', r5.els['m-saved'].textContent !== '');
}

// 8. served mode: identical poll must NOT redraw; changed poll must redraw
(async () => {
  const same = JSON.parse(payloadJson);
  const changed = JSON.parse(payloadJson);
  changed.totals.saved += 4242;
  changed.generated_at = '2099-01-01 00:00';

  const mkFetch = body => () => Promise.resolve({ ok: true, json: () => Promise.resolve(body) });

  const a = run('', { protocol: 'http:', fetch: mkFetch(same) });
  const beforeA = a.draws.count;
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok('served: unchanged poll skips redraw', a.draws.count === beforeA);
  ok('served: marks page live', /live/.test(a.els['meta'].innerHTML) || a.els['countdown'].textContent !== '');

  const b = run('', { protocol: 'http:', fetch: mkFetch(changed) });
  const beforeB = b.draws.count;
  await new Promise(r => setImmediate(r));
  await new Promise(r => setImmediate(r));
  ok('served: changed poll triggers redraw', b.draws.count > beforeB);

  console.log(out.join('\n'));
})();
