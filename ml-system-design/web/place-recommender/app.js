const SLUG = 'place-recommender'; loadGuide(SLUG, $('#guide'));
const CATC = {cafe: '#c98b5a', restaurant: '#ef6b73', fast_food: '#f0b429', bar: '#b794f6', park: '#4cc38a', museum: '#6ea8fe', bakery: '#f5a3c7', shopping: '#5fd3d3'};
const GC = {relevance: '#6ea8fe', distance: '#4cc38a', popularity: '#f0b429', context: '#b794f6'};
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const S = {places: [], city: 10, cell: 0.5, click: [5, 5], user: null, rec: null, hover: -1, onboard: [], isNew: false};

// ---------- small canvas helpers ----------
function prep(cv, h) { const dpr = devicePixelRatio || 1, W = cv.clientWidth || 500; cv.width = W * dpr; cv.height = h * dpr; cv.style.height = h + 'px'; const c = cv.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, W, h); c.font = '11px system-ui'; return [c, W]; }
/** grouped vertical bars. groups: labels on x; series: [{name,color,values}] */
function vbars(cv, groups, series, {h = 240, fmt = v => v.toFixed(2), max, line} = {}) {
  const [c, W] = prep(cv, h), pad = {l: 8, r: 8, t: 26, b: 44}, mx = max ?? Math.max(...series.flatMap(s => s.values), line ?? 0, 1e-9) * 1.12;
  const gw = (W - pad.l - pad.r) / groups.length, bw = Math.min(46, gw * 0.8 / series.length), Y = v => h - pad.b - Math.max(0, v) / mx * (h - pad.t - pad.b);
  c.strokeStyle = '#2a3144'; c.beginPath(); c.moveTo(pad.l, h - pad.b); c.lineTo(W - pad.r, h - pad.b); c.stroke();
  groups.forEach((g, i) => {
    const x0 = pad.l + gw * i + (gw - bw * series.length) / 2;
    series.forEach((s, k) => { const v = s.values[i]; if (v == null) return; const col = Array.isArray(s.color) ? s.color[i] : s.color; c.fillStyle = col; const y = Y(v); c.beginPath(); c.roundRect(x0 + bw * k + 1, y, bw - 2, h - pad.b - y, 3); c.fill(); c.fillStyle = '#e6e9f0'; c.textAlign = 'center'; c.fillText(fmt(v), x0 + bw * k + bw / 2, y - 4); });
    c.fillStyle = '#8d96ab'; c.textAlign = 'center'; const words = String(g).split(' '); let l1 = '', l2 = ''; words.forEach(w => { if ((l1 + ' ' + w).length <= Math.max(8, gw / 6.2)) l1 = (l1 + ' ' + w).trim(); else l2 = (l2 + ' ' + w).trim(); });
    c.fillText(l1, pad.l + gw * i + gw / 2, h - pad.b + 14); if (l2) c.fillText(l2.slice(0, Math.max(8, gw / 5.5)), pad.l + gw * i + gw / 2, h - pad.b + 27);
  });
  if (line != null) { c.setLineDash([5, 4]); c.strokeStyle = '#e6e9f0'; c.beginPath(); c.moveTo(pad.l, Y(line)); c.lineTo(W - pad.r, Y(line)); c.stroke(); c.setLineDash([]); }
  c.textAlign = 'left'; let lx = pad.l; series.forEach(s => { if (!s.name) return; c.fillStyle = Array.isArray(s.color) ? '#8d96ab' : s.color; c.fillRect(lx, 6, 10, 10); c.fillStyle = '#8d96ab'; c.fillText(s.name, lx + 14, 15); lx += 22 + c.measureText(s.name).width; });
}

// ---------- map ----------
function drawMap() {
  const cv = $('#map'), W = cv.clientWidth, [c] = prep(cv, W), k = W / S.city, X = x => x * k, Y = y => W - y * k, r = S.rec;
  c.strokeStyle = 'rgba(255,255,255,.035)'; c.lineWidth = 1;
  for (let g = 0; g <= S.city; g += S.cell) { c.beginPath(); c.moveTo(X(g), 0); c.lineTo(X(g), W); c.moveTo(0, Y(g)); c.lineTo(W, Y(g)); c.stroke(); }
  if (r) { c.fillStyle = 'rgba(110,168,254,.10)'; r.cells.forEach(([cx, cy]) => c.fillRect(X(cx * S.cell), Y((cy + 1) * S.cell), S.cell * k, S.cell * k));
    const [ux, uy] = r.location_used; c.strokeStyle = 'rgba(110,168,254,.7)'; c.setLineDash([6, 5]); c.beginPath(); c.arc(X(ux), Y(uy), +$('#rad').value * k, 0, 7); c.stroke(); c.setLineDash([]); }
  const inList = new Map(r ? r.items.map((it, i) => [it.id, i]) : []);
  S.places.forEach(p => { if (!p.live || inList.has(p.id)) return; const rad = 1.6 + Math.sqrt(p.visits) * 0.42; c.globalAlpha = 0.5; c.fillStyle = CATC[p.cat]; c.beginPath(); c.arc(X(p.x), Y(p.y), rad, 0, 7); c.fill(); c.globalAlpha = 1;
    if (p.new) { c.strokeStyle = '#f0b429'; c.lineWidth = 1.2; c.beginPath(); c.arc(X(p.x), Y(p.y), rad + 2.5, 0, 7); c.stroke(); } });
  const mark = (pt, ch) => { if (!pt) return; c.font = '15px system-ui'; c.fillStyle = '#e6e9f0'; c.textAlign = 'center'; c.fillText(ch, X(pt[0]), Y(pt[1]) + 5); };
  if (S.user && !S.isNew) { mark(S.user.home, '⌂'); mark(S.user.work, '▣'); }
  if (r) {
    const [ux, uy] = r.location_used;
    r.items.forEach((it, i) => { c.strokeStyle = i === S.hover ? '#fff' : 'rgba(230,233,240,.28)'; c.lineWidth = i === S.hover ? 2 : 1; c.beginPath(); c.moveTo(X(ux), Y(uy)); c.lineTo(X(it.x), Y(it.y)); c.stroke(); });
    r.items.forEach((it, i) => { const x = X(it.x), y = Y(it.y), R = i === S.hover ? 13 : 10; c.fillStyle = CATC[it.cat]; c.strokeStyle = it.cold_place ? '#f0b429' : '#0b0e14'; c.lineWidth = it.cold_place ? 3 : 2; c.beginPath(); c.arc(x, y, R, 0, 7); c.fill(); c.stroke(); c.fillStyle = '#0b1020'; c.font = 'bold 11px system-ui'; c.textAlign = 'center'; c.fillText(i + 1, x, y + 4); });
    if (Math.hypot(ux - S.click[0], uy - S.click[1]) > 0.05) { c.strokeStyle = '#8d96ab'; c.beginPath(); c.arc(X(S.click[0]), Y(S.click[1]), 5, 0, 7); c.stroke(); }
    c.fillStyle = '#fff'; c.strokeStyle = '#6ea8fe'; c.lineWidth = 3; c.beginPath(); c.arc(X(ux), Y(uy), 6, 0, 7); c.fill(); c.stroke();
  }
  c.textAlign = 'left';
}
function mapXY(e) { const cv = $('#map'), b = cv.getBoundingClientRect(), k = b.width / S.city; return [(e.clientX - b.left) / k, (b.bottom - e.clientY) / k]; }
$('#map').onclick = e => { S.click = mapXY(e); recommend(); };
$('#map').onmousemove = e => {
  const [x, y] = mapXY(e), tip = $('#tip'); let best = null, bd = 0.18;
  S.places.forEach(p => { if (!p.live) return; const d = Math.hypot(p.x - x, p.y - y); if (d < bd) { bd = d; best = p; } });
  if (!best) { tip.style.display = 'none'; return; }
  const b = $('#map').getBoundingClientRect();
  tip.innerHTML = `<b>${esc(best.name)}</b><br>${best.cat.replace('_', ' ')} · ${'$'.repeat(best.price)} · ★${best.rating} · ${best.visits} visits${best.new ? ' · <span class="warn">new</span>' : ''}`;
  tip.style.display = 'block'; tip.style.left = Math.min(e.clientX - b.left + 12, b.width - 200) + 'px'; tip.style.top = (e.clientY - b.top + 12) + 'px';
};
$('#map').onmouseleave = () => $('#tip').style.display = 'none';

// ---------- controls ----------
const segVal = id => $(`#${id} button.on`).dataset.v;
function seg(id, fn) { $$(`#${id} button`).forEach(b => b.onclick = () => { $$(`#${id} button`).forEach(x => x.classList.toggle('on', x === b)); fn(); }); }
function hourLabel() { const h = +$('#hr').value, e = h < 6 ? '🌙 night' : h < 11 ? '🌅 morning' : h < 15 ? '☀️ midday' : h < 18 ? '🌤 afternoon' : h < 22 ? '🌆 evening' : '🌙 night'; $('#hrv').textContent = `${String(h).padStart(2, '0')}:00  ${e}`; $('#skyM').style.left = (h / 23 * 100) + '%'; }

async function loadUser() {
  if (S.isNew) {
    $('#uInfo').innerHTML = `<label>No history. Ask what they like (onboarding)</label><div class="chips">${Object.keys(CATC).map(c => `<span class="chip ${S.onboard.includes(c) ? 'on' : ''}" data-c="${c}">${c.replace('_', ' ')}</span>`).join('')}</div>`;
    $$('#uInfo .chip').forEach(ch => ch.onclick = () => { const c = ch.dataset.c; S.onboard = S.onboard.includes(c) ? S.onboard.filter(x => x !== c) : [...S.onboard, c]; loadUser(); recommend(); });
    S.user = null; return;
  }
  const u = S.user = await api(SLUG, '/user/' + (+$('#uid').value));
  $('#uInfo').innerHTML = `<label>Taste, from ${u.check_ins} past check-ins (typical spend ${'$'.repeat(Math.round(u.avg_price))})</label><div class="taste">${u.category_share.map(s => `<i style="width:${s.share * 100}%;background:${CATC[s.cat]}" title="${s.cat} ${pct(s.share)}"></i>`).join('')}</div>
    <div class="sm mute">Mostly ${u.category_share.slice(0, 3).map(s => `<b style="color:${CATC[s.cat]}">${s.cat.replace('_', ' ')}</b> ${pct(s.share)}`).join(', ')}. <a href="#" id="goH">Stand at home</a> · <a href="#" id="goW">Stand at work</a></div>`;
  $('#goH').onclick = e => { e.preventDefault(); S.click = u.home; recommend(); }; $('#goW').onclick = e => { e.preventDefault(); S.click = u.work; recommend(); };
}

let tok = 0;
async function recommend() {
  const my = ++tok;
  const r = await api(SLUG, '/recommend', {user_id: S.isNew ? -1 : +$('#uid').value, onboard_cats: S.onboard, x: S.click[0], y: S.click[1], hour: +$('#hr').value, weekend: $('#wk').classList.contains('on'), rain: $('#rain').classList.contains('on'),
    budget: +segVal('bud'), radius: +$('#rad').value, k: 10, variant: segVal('var'), diversity: +$('#div').value, explore: segVal('exp'), explore_strength: 0.5, loc_precision_km: +$('#prec').value});
  if (my !== tok) return; S.rec = r; S.hover = -1; drawMap(); renderRec();
}

function renderRec() {
  const r = S.rec, mx = Math.max(...r.stages.map(s => s.out));
  $('#funnel').innerHTML = `<div class="f" style="width:100%">${r.stages[0].in} places in the city</div>` + r.stages.map(s => `<div class="t">▼ ${esc(s.stage)} · ${s.ms.toFixed(1)} ms</div><div class="f" style="width:${Math.max(9, Math.sqrt(s.out / r.stages[0].in) * 100)}%">${s.out}</div>`).join('') + `<div class="t">total ${r.total_ms.toFixed(0)} ms</div>`;
  const cm = Math.max(...r.items.map(it => Object.values(it.contrib).reduce((a, v) => a + Math.max(0, v), 0)), 1e-9);
  $('#res').innerHTML = r.items.map((it, i) => {
    const mv = it.rank_before_rerank - (i + 1), seg = Object.entries(it.contrib).map(([g, v]) => `<i style="width:${Math.max(0, v) / cm * 100}%;background:${GC[g]}" title="${g}: ${v.toFixed(2)}"></i>`).join('');
    return `<div class="r" data-i="${i}"><div class="n" style="background:${CATC[it.cat]}">${i + 1}</div>
      <div><div class="nm">${esc(it.name)}${it.cold_place ? '<span class="badge new">new</span>' : ''}${it.explored ? '<span class="badge new">explore</span>' : ''}${mv >= 3 ? `<span class="badge up">▲ ${mv} by re-rank</span>` : mv <= -3 ? `<span class="badge dn">▼ ${-mv}</span>` : ''}</div>
      <div class="meta">${it.cat.replace('_', ' ')} · ${'$'.repeat(it.price)} · ★${it.rating} · ${it.dist_km.toFixed(2)} km · ${it.visits} visits</div></div>
      <div><div class="stack">${seg}</div><div class="meta" style="margin-top:3px">P(visit) ${pct(it.p_visit)}</div></div></div>`; }).join('') || '<div class="mute">Nothing within this radius and budget. Widen the radius.</div>';
  $$('#res .r').forEach(el => { el.onmouseenter = () => { S.hover = +el.dataset.i; drawMap(); }; el.onmouseleave = () => { S.hover = -1; drawMap(); }; });
  const s = r.summary, k = {'categories in the list': s.categories_in_list, 'avg distance': s.mean_distance_km.toFixed(2) + ' km', 'avg popularity percentile': pct(s.mean_popularity_percentile), 'new places shown': `${s.new_places_in_list} of ${s.new_places_in_candidates} nearby`};
  if (!r.cold_user) k['true satisfaction vs best possible'] = pct(s.mean_true_satisfaction / s.oracle_best_satisfaction);
  kpis($('#sum'), k);
  if (r.cold_user || !r.items.length) { prep($('#satC'), 30); const c = $('#satC').getContext('2d'); c.fillStyle = '#8d96ab'; c.fillText(r.cold_user ? 'A brand-new user has no known tastes, so there is no hidden truth to compare against.' : '', 8, 18); }
  else vbars($('#satC'), r.items.map((_, i) => '#' + (i + 1)), [{values: r.items.map(it => it.true_satisfaction), color: r.items.map(it => CATC[it.cat])}], {h: 190, line: s.oracle_best_satisfaction, fmt: v => v.toFixed(1)});
}

// ---------- static sections ----------
let OV = null;
const MN = {popularity_global: 'Popular citywide', popularity_nearby: 'Popular nearby', mf: 'Taste embedding', mf_distance: 'Taste + distance', mf_geo_context_gbdt: 'Full ranker'};
function drawMetrics() {
  const sl = $('#slice').value, rows = Object.keys(MN).map(m => OV.metrics.slices.find(s => s.slice === sl && s.model === m)).filter(Boolean);
  vbars($('#mC'), rows.map(r => MN[r.model]), [{name: 'recall@10', color: '#6ea8fe', values: rows.map(r => r.recall)}, {name: 'NDCG@10', color: '#4cc38a', values: rows.map(r => r.ndcg)}, {name: 'retrieval recall (ceiling)', color: '#3a4257', values: rows.map(r => r.retrieval_recall)}], {h: 280, max: 1.1});
}
async function overview() {
  const [o, pl] = await Promise.all([api(SLUG, '/overview'), api(SLUG, '/places')]); OV = o; const d = o.dataset;
  S.places = pl.places; S.city = pl.city_km; S.cell = pl.cell_km;
  kpis($('#kpi'), {'places': d.places, 'people': d.users, 'check-ins': d.check_ins, 'opened during the log': d.places_opened_during_log, 'closed': d.places_closed, 'median trip to a check-in': d.median_checkin_distance_km.toFixed(1) + ' km', 'visits going to top 10% of places': pct(d.share_of_visits_to_top10pct_places)});
  $('#legend').innerHTML = Object.entries(CATC).map(([c, col]) => `<span><i style="background:${col}"></i>${c.replace('_', ' ')}</span>`).join('');
  const order = ['all', ...new Set(o.metrics.slices.map(s => s.slice).filter(s => s !== 'all'))];
  $('#slice').innerHTML = order.map(s => `<option>${esc(s)}</option>`).join(''); $('#slice').onchange = drawMetrics; drawMetrics();
  const grp = f => Object.entries(o.models.feature_groups).find(([, fs]) => fs.includes(f))[0], imp = o.importance.slice(0, 9);
  $('#impKey').innerHTML = Object.entries(GC).map(([g, c]) => `<span><i style="background:${c}"></i>${{relevance: 'taste match', distance: 'distance', popularity: 'popularity', context: 'time and weather'}[g]}</span>`).join('');
  vbars($('#iC'), imp.map(i => i.feature.replace(/_/g, ' ')), [{values: imp.map(i => Math.max(0, i.auc_drop)), color: imp.map(i => GC[grp(i.feature)])}], {h: 280, fmt: v => v.toFixed(3)});
}
async function distance() {
  const d = await api(SLUG, '/pitfall/distance_bias'), w = d.wrong, r = d.right, R = '#ef6b73', G = '#4cc38a';
  vbars($('#dH'), d.checkin_distance_hist.map(h => h.km), [{values: d.checkin_distance_hist.map(h => h.share), color: '#6ea8fe'}], {h: 220, fmt: v => pct(v)});
  vbars($('#dC'), ['closeness to home', 'true taste'], [{name: 'negatives from anywhere (wrong)', color: R, values: [w['corr(score, -distance_from_home)'], w['corr(score, true_taste_affinity)']]}, {name: 'negatives from nearby (right)', color: G, values: [r['corr(score, -distance_from_home)'], r['corr(score, true_taste_affinity)']]}], {h: 220});
  vbars($('#dR'), ['embedding only', 'embedding + distance'], [{name: 'wrong', color: R, values: [w['recall@10 MF (away from home)'], w['recall@10 MF+distance (away from home)']]}, {name: 'right', color: G, values: [r['recall@10 MF (away from home)'], r['recall@10 MF+distance (away from home)']]}], {h: 220});
}
async function position() {
  const p = await api(SLUG, '/pitfall/position_bias', {eta: +$('#eta').value, random_frac: +$('#rf').value});
  lineChart($('#pC'), [{name: 'production order', xs: p.positions, ys: p.ctr_by_position_logged.map(v => v * 100), color: '#ef6b73'}, {name: 'random order', xs: p.positions, ys: p.ctr_by_position_randomised.map(v => v * 100), color: '#4cc38a'}], {xlabel: 'position', h: 240});
  const u = p.most_underrated;
  vbars($('#pB'), u.map(x => x.place.replace(/ #\d+$/, '')), [{name: 'raw click rate', color: '#ef6b73', values: u.map(x => x.naive_ctr)}, {name: 'position-corrected', color: '#4cc38a', values: u.map(x => x.ipw_ctr)}, {name: 'true appeal', color: '#8d96ab', values: u.map(x => x.true_attractiveness)}], {h: 240, fmt: v => v.toFixed(1), max: 1.25});
  kpis($('#pK'), {'top-20 overlap with truly best, raw': pct(p.wrong.top20_overlap_with_truly_best), 'top-20 overlap, corrected': pct(p.right.top20_overlap_with_truly_best), 'average error, raw': p.wrong.mean_abs_error.toFixed(2), 'average error, corrected': p.right.mean_abs_error.toFixed(2), 'sessions shown in random order': p.randomised_sessions});
}
async function loop() {
  const body = {rounds: +$('#rd').value, pop_weight: +$('#pw').value};
  const [a, b] = await Promise.all([api(SLUG, '/pitfall/popularity_loop', {...body, fix: false}), api(SLUG, '/pitfall/popularity_loop', {...body, fix: true})]);
  const xs = a.series.map(s => s.round), ch = (id, f, m = 100) => lineChart($(id), [{name: 'raw counts', xs, ys: a.series.map(s => s[f] * m), color: '#ef6b73'}, {name: 'rates + explore', xs, ys: b.series.map(s => s[f] * m), color: '#4cc38a'}], {xlabel: 'round', h: 210});
  ch('#l1', 'head_share'); ch('#l2', 'coverage'); ch('#l3', 'mean_true_satisfaction_of_clicks', 1);
}
async function split() {
  const s = await api(SLUG, '/pitfall/split'), nm = {popularity_nearby: 'Popular nearby', mf: 'Taste embedding', mf_distance: 'Taste + distance'};
  vbars($('#sC'), s.rows.map(r => nm[r.model] || r.model), [{name: 'random split (wrong)', color: '#ef6b73', values: s.rows.map(r => r['random_split_recall@10'])}, {name: 'time split (right)', color: '#4cc38a', values: s.rows.map(r => r['time_split_recall@10'])}], {h: 230, fmt: v => v.toFixed(3)});
  const worst = s.rows.reduce((a, b) => b.relative_inflation_pct > a.relative_inflation_pct ? b : a);
  $('#sN').innerHTML = `<b>What to notice.</b> A random split lets the model train on visits that happen after the ones it is tested on. The ${nm[worst.model].toLowerCase()} looks ${worst.relative_inflation_pct.toFixed(0)}% better than it will be at launch, because it has memorised each person's future favourites. The popularity baseline barely moves, so the random split also exaggerates the gain of the learned model over the baseline. ${esc(s.explain)}`;
}

// ---------- wiring ----------
const live = (id, out, fmt, fn) => { $(id).oninput = e => { $(out).textContent = fmt(+e.target.value); }; $(id).onchange = fn; };
live('#rad', '#radv', v => v.toFixed(1), recommend); live('#div', '#divv', v => v.toFixed(2), recommend);
live('#prec', '#precv', v => v === 0 ? 'exact' : `within ${v} km`, recommend);
live('#eta', '#etav', v => v.toFixed(1), position); live('#rf', '#rfv', v => pct(v), position);
live('#pw', '#pwv', v => v.toFixed(1), loop); live('#rd', '#rdv', v => v, loop);
$('#hr').oninput = () => { hourLabel(); recommend(); };
['wk', 'rain'].forEach(id => $('#' + id).onclick = () => { $('#' + id).classList.toggle('on'); recommend(); });
['bud', 'var', 'exp'].forEach(id => seg(id, recommend));
$('#uid').onchange = async () => { S.isNew = false; $('#newU').classList.remove('on'); await loadUser(); recommend(); };
$('#rndU').onclick = () => { $('#uid').value = Math.floor(Math.random() * 800); $('#uid').onchange(); };
$('#newU').onclick = async () => { S.isNew = !S.isNew; $('#newU').classList.toggle('on', S.isNew); await loadUser(); recommend(); };
let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { drawMap(); if (S.rec) renderRec(); if (OV) drawMetrics(); }, 200); });
hourLabel();
overview().then(loadUser).then(recommend).catch(() => {});
distance(); position(); loop(); split();
