// Shared helpers for every demo page.  Usage: <script src="../shared/common.js"></script>
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/** api('friend-recommendation', '/recommend', {user_id:3}) -> POST JSON if body given, else GET. */
async function api(slug, path, body) {
  const opt = body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  const r = await fetch(`/api/${slug}${path}`, opt);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

/** Tabs: markup <div class="tabs"><div class="tab" data-pane="demo">..</div></div> + <div class="pane" id="demo"> */
function initTabs() {
  const tabs = $$('.tab'); if (!tabs.length) return;
  const show = id => { tabs.forEach(t => t.classList.toggle('active', t.dataset.pane === id)); $$('.pane').forEach(p => p.classList.toggle('active', p.id === id)); };
  tabs.forEach(t => t.onclick = () => { show(t.dataset.pane); history.replaceState(null, '', '#' + t.dataset.pane); });
  show((location.hash || '').slice(1) || tabs[0].dataset.pane);
}

/** Minimal markdown: #,##,###, **b**, `c`, ```fences```, lists, tables, > quotes, --- and
 *  <details><summary>Q</summary>A</details> passthrough (answers rendered as markdown). */
function md(src) {
  const inline = t => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/\*([^*]+)\*/g, '<i>$1</i>').replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  const lines = src.replace(/\r/g, '').split('\n'); let out = '', i = 0, list = null;
  const close = () => { if (list) { out += `</${list}>`; list = null; } };
  while (i < lines.length) {
    let l = lines[i];
    if (l.startsWith('```')) { close(); let code = []; i++; while (i < lines.length && !lines[i].startsWith('```')) code.push(lines[i++]); out += `<pre><code>${esc(code.join('\n'))}</code></pre>`; i++; continue; }
    if (/^<\/?details>|^<summary>/.test(l.trim())) { close(); out += l.trim().replace(/^<summary>(.*)<\/summary>$/, (m, q) => `<summary>${inline(q)}</summary>`); i++; continue; }
    if (/^\|.+\|$/.test(l) && /^\|[\s:|-]+\|$/.test(lines[i + 1] || '')) { close(); const row = r => r.slice(1, -1).split('|').map(c => c.trim()); let h = row(l); i += 2; let body = ''; while (i < lines.length && /^\|.+\|$/.test(lines[i])) body += '<tr>' + row(lines[i++]).map(c => `<td>${inline(c)}</td>`).join('') + '</tr>'; out += `<table><tr>${h.map(c => `<th>${inline(c)}</th>`).join('')}</tr>${body}</table>`; continue; }
    let m;
    if ((m = l.match(/^(#{1,3}) (.*)/))) { close(); out += `<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`; }
    else if ((m = l.match(/^\s*[-*] (.*)/))) { if (list !== 'ul') { close(); out += '<ul>'; list = 'ul'; } out += `<li>${inline(m[1])}</li>`; }
    else if ((m = l.match(/^\s*\d+\. (.*)/))) { if (list !== 'ol') { close(); out += '<ol>'; list = 'ol'; } out += `<li>${inline(m[1])}</li>`; }
    else if (l.startsWith('> ')) { close(); out += `<blockquote>${inline(l.slice(2))}</blockquote>`; }
    else if (l.trim() === '---') { close(); out += '<hr>'; }
    else if (l.trim() === '') close();
    else { close(); out += `<p>${inline(l)}</p>`; }
    i++;
  }
  close(); return out;
}

/** Load ../guides/<slug>.md via /api/guide/<slug> into element. */
async function loadGuide(slug, el) {
  // Backend first; fall back to the static markdown file so guides work on GitHub Pages.
  const get = async u => { const r = await fetch(u); if (!r.ok) throw new Error(r.status); return r.text(); };
  try {
    let t; try { t = await get(`/api/guide/${slug}`); } catch (e) { t = await get(`../../guides/${slug}.md`); }
    el.innerHTML = '<div class="md">' + md(t) + '</div>';
  } catch (e) { el.textContent = 'Guide failed to load: ' + e; }
}
// If there is no backend (static hosting), tell the reader how to run the demo.
document.addEventListener('DOMContentLoaded', async () => {
  const demo = document.getElementById('demo'); if (!demo) return;
  try { const r = await fetch('/api/modules'); if (!r.ok) throw 0; await r.json(); }
  catch (e) { demo.insertAdjacentHTML('afterbegin', '<div class="note" style="border-color:var(--warn)"><b>The live demo needs the Python backend.</b> The interview guide tab works here; to run the demo, clone the repo and run <code>cd ml-system-design &amp;&amp; ./run.sh</code>, then open <code>http://localhost:8000</code>.</div>'); }
});

// ---- tiny charts (canvas, no deps) ----
function lineChart(canvas, series, {xlabel = '', ylabel = '', h = 260} = {}) {
  // series: [{name, xs:[], ys:[], color?}]
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth || 600; canvas.width = W * dpr; canvas.height = h * dpr; canvas.style.height = h + 'px';
  const c = canvas.getContext('2d'); c.scale(dpr, dpr); c.clearRect(0, 0, W, h);
  const pad = {l: 44, r: 12, t: 12, b: 30}, cols = ['#6ea8fe', '#4cc38a', '#f0b429', '#ef6b73', '#b794f6'];
  const xs = series.flatMap(s => s.xs), ys = series.flatMap(s => s.ys);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const X = x => pad.l + (x - x0) / ((x1 - x0) || 1) * (W - pad.l - pad.r), Y = y => h - pad.b - (y - y0) / ((y1 - y0) || 1) * (h - pad.t - pad.b);
  c.strokeStyle = '#2a3144'; c.fillStyle = '#8d96ab'; c.font = '11px system-ui';
  for (let k = 0; k <= 4; k++) { const y = y0 + (y1 - y0) * k / 4; c.beginPath(); c.moveTo(pad.l, Y(y)); c.lineTo(W - pad.r, Y(y)); c.stroke(); c.fillText(y.toFixed(Math.abs(y) < 10 ? 2 : 0), 4, Y(y) + 4); }
  c.fillText(xlabel, W / 2 - 20, h - 6); const fx = x => Number.isInteger(x) ? String(x) : x.toFixed(Math.abs(x1) < 10 ? 2 : 0); c.fillText(fx(x0), pad.l, h - 16); c.fillText(fx(x1), W - pad.r - 26, h - 16);
  series.forEach((s, k) => { c.strokeStyle = s.color || cols[k % cols.length]; c.lineWidth = 2; c.beginPath(); s.xs.forEach((x, j) => j ? c.lineTo(X(x), Y(s.ys[j])) : c.moveTo(X(x), Y(s.ys[j]))); c.stroke(); c.fillStyle = c.strokeStyle; c.fillText(s.name || '', W - pad.r - 90, pad.t + 12 + k * 14); });
}
function barChart(el, labels, values, {max, fmt = v => v.toFixed(3)} = {}) {
  const m = max ?? Math.max(...values.map(Math.abs), 1e-9);
  el.innerHTML = labels.map((l, i) => `<div style="display:grid;grid-template-columns:140px 1fr 70px;gap:8px;align-items:center;margin:4px 0"><span class="mute">${esc(l)}</span><div class="bar"><i style="width:${Math.max(0, values[i]) / m * 100}%"></i></div><span>${fmt(values[i])}</span></div>`).join('');
}
function table(el, rows, cols) {
  cols = cols || Object.keys(rows[0] || {});
  el.innerHTML = `<table><tr>${cols.map(c => `<th>${esc(c)}</th>`).join('')}</tr>${rows.map(r => `<tr>${cols.map(c => `<td>${typeof r[c] === 'number' ? +r[c].toFixed(4) : esc(r[c] ?? '')}</td>`).join('')}</tr>`).join('')}</table>`;
}
function kpis(el, obj) { el.className = 'kpi'; el.innerHTML = Object.entries(obj).map(([k, v]) => `<div><b>${esc(typeof v === 'number' ? +v.toFixed(4) : v)}</b><span>${esc(k)}</span></div>`).join(''); }
document.addEventListener('DOMContentLoaded', initTabs);
