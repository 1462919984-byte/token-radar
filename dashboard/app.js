/* Token Radar v3 dashboard. Reads window.RADAR (written beside this page by
 * main.ts). Renders the ranked board, a filterable and sortable table, a
 * per-protocol detail drawer with the farm economics, and three views: the
 * launch radar, live tokens, and protocols set aside (unranked, not
 * candidates). No network is used at any point.
 *
 * The implied value shown in the table is recomputed in the page from a row's
 * own fields and the fitted coefficients so the code and the board's numbers
 * are visibly the same arithmetic:
 *
 *   compsValue(row)  = round(exp( compsBeta . compsFeature(row) ))
 *   tokenOdds(row)   = sigmoid( oddsBeta . oddsFeature(row) )
 *   impliedValue(row)= round( compsValue(row) * tokenOdds(row) )
 */
(function () {
  'use strict';
  var DATA = window.RADAR;
  if (!DATA) return;

  var M = DATA.model;
  var COMP_BETA = M.comps.beta;
  var COMP_CATS = M.comps.categories;
  var ODDS_BETA = M.odds.beta;
  var ODDS_CATS = M.odds.categories;
  var UNKNOWN_AGE = M.comps.unknown_age_days;

  // ---- formatting helpers -----------------------------------------------
  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function fmtPct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
  }
  function fmtAge(d) {
    if (d === null || d === undefined) return 'unknown';
    if (d >= 365) return (d / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(d)) + 'd';
  }
  function ln(x) { return Math.log(Math.max(x, 1)); }
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // ---- the model, recomputed in the page --------------------------------
  function featureComps(row) {
    var age = row.age_days === null || row.age_days === undefined ? UNKNOWN_AGE : row.age_days;
    var g = row.growth_pct === null || row.growth_pct === undefined ? 0 : row.growth_pct;
    var v = [1, ln(row.tvl_usd), ln(age + 1), g, ln(row.chains.length + 1)];
    for (var i = 0; i < COMP_CATS.length; i++) v.push(row.category === COMP_CATS[i] ? 1 : 0);
    return v;
  }
  function compsValueOf(row) {
    var v = featureComps(row);
    var s = 0;
    for (var j = 0; j < COMP_BETA.length; j++) s += COMP_BETA[j] * v[j];
    return Math.max(1, Math.round(Math.exp(s)));
  }
  function featureOdds(row) {
    var age = row.age_days === null || row.age_days === undefined ? UNKNOWN_AGE : row.age_days;
    var g = row.growth_pct === null || row.growth_pct === undefined ? 0 : row.growth_pct;
    var v = [1, ln(row.tvl_usd), ln(row.chains.length + 1), g / 100, age / 365];
    for (var i = 0; i < ODDS_CATS.length; i++) v.push(row.category === ODDS_CATS[i] ? 1 : 0);
    return v;
  }
  function oddsOf(row) {
    var v = featureOdds(row);
    var z = 0;
    for (var j = 0; j < ODDS_BETA.length; j++) z += ODDS_BETA[j] * v[j];
    return sigmoid(z);
  }
  function impliedValueOf(row) {
    return Math.max(1, Math.round(compsValueOf(row) * oddsOf(row)));
  }

  // ---- tiny DOM helper --------------------------------------------------
  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'class') node.className = attrs[k];
        else if (k === 'text') node.textContent = attrs[k];
        else if (k === 'html') node.innerHTML = attrs[k];
        else if (k.slice(0, 2) === 'on') node.addEventListener(k.slice(2), attrs[k]);
        else if (attrs[k] !== null && attrs[k] !== undefined) node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var child = arguments[i];
      if (child === null || child === undefined) continue;
      if (Array.isArray(child)) child.forEach(function (c) { if (c) node.appendChild(c); });
      else if (typeof child === 'string') node.appendChild(document.createTextNode(child));
      else node.appendChild(child);
    }
    return node;
  }

  // ---- state ------------------------------------------------------------
  var state = { tab: 'radar', q: '', cat: 'all', wl: false, sort: 'rank', dir: 1, selected: null };
  var app = document.getElementById('board');

  // ---- category options -------------------------------------------------
  var catSet = {};
  DATA.rows.forEach(function (r) { if (r.category) catSet[r.category] = true; });
  DATA.live.forEach(function (r) { if (r.category) catSet[r.category] = true; });
  DATA.products.forEach(function (r) { if (r.category) catSet[r.category] = true; });
  var cats = Object.keys(catSet).sort();
  var catSel = document.getElementById('cat');
  cats.forEach(function (c) { catSel.appendChild(el('option', { value: c, text: c })); });

  // ---- integrity strip ---------------------------------------------------
  function setR2() {
    // Display the honest computed R-squared with three decimal places, and as
    // a percentage with one decimal place for the headline.
    var r2 = M.comps.r2;
    document.getElementById('stR2').textContent = (r2 * 100).toFixed(1) + '%';
    document.getElementById('stR2').title = 'R\u00b2 = ' + r2.toFixed(4);
  }
  setR2();
  document.getElementById('stMae').textContent = M.comps.cv_mae_log.toFixed(2) + ' (\u00D7' + Math.exp(M.comps.cv_mae_log).toFixed(1) + ')';
  document.getElementById('stB1').textContent = COMP_BETA[1].toFixed(3);
  document.getElementById('stCov').textContent = DATA.counts.watchlist + ' / ' + DATA.counts.ranked;
  document.getElementById('metaDate').textContent = DATA.as_of;
  document.getElementById('metaWatch').textContent = DATA.counts.watchlist;
  document.getElementById('metaRanked').textContent = DATA.counts.ranked;
  document.getElementById('metaAside').textContent = DATA.counts.products;
  document.getElementById('cntRadar').textContent = DATA.counts.ranked;
  document.getElementById('cntLive').textContent = DATA.counts.live;
  document.getElementById('cntAside').textContent = DATA.counts.products;

  // ---- filters & sort ----------------------------------------------------
  function visibleRows() {
    var src = state.tab === 'radar' ? DATA.rows : state.tab === 'live' ? DATA.live : DATA.products;
    var q = state.q.trim().toLowerCase();
    var out = src.filter(function (r) {
      if (state.cat !== 'all' && r.category !== state.cat) return false;
      if (state.wl && state.tab === 'radar' && !r.is_watchlist) return false;
      if (q) {
        var hay = (r.name + ' ' + r.slug + ' ' + (r.category || '') + ' ' + (r.chains || []).join(' ')).toLowerCase();
        if (hay.indexOf(q) < 0) return false;
      }
      return true;
    });
    var dir = state.dir;
    var key = state.sort;
    out.sort(function (a, b) {
      var va = sortVal(a, key);
      var vb = sortVal(b, key);
      if (va < vb) return -1 * dir;
      if (va > vb) return 1 * dir;
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    });
    return out;
  }
  function sortVal(r, key) {
    switch (key) {
      case 'rank': return r.rank || 0;
      case 'iv': return r.implied_value_usd || 0;
      case 'tvl': return r.tvl_usd || 0;
      case 'mcap': return r.mcap_usd || 0;
      case 'change': return r.change_7d === null || r.change_7d === undefined ? -1e18 : r.change_7d;
      case 'age': return r.age_days === null || r.age_days === undefined ? -1 : r.age_days;
      case 'odds': return r.token_odds || 0;
      case 'net': return r.farm ? r.farm.net_value_usd || 0 : 0;
      case 'name': return r.name;
      case 'category': return r.category || '';
      default: return 0;
    }
  }

  // ---- render table -------------------------------------------------------
  function render() {
    var rows = visibleRows();
    document.getElementById('showing').textContent = rows.length + ' shown';
    var cols = columnsFor(state.tab);
    var wrap = el('div', { class: 'table-wrap' });
    var table = el('table');
    if (state.tab === 'radar') {
      table.appendChild(el('caption', { text: 'Pre-token launch candidates only, ranked by implied value. Derivatives, CEX, bridges, wrapped, staked and versioned pools are set aside unranked.' }));
    } else if (state.tab === 'aside') {
      table.appendChild(el('caption', { text: 'Set aside: not launch candidates and not ranked. Shown by TVL, with the market cap where a token already trades.' }));
    } else {
      table.appendChild(el('caption', { text: 'Live tokens: already trade, so their market cap is a fact, not a model. Never mixed into the launch ranking.' }));
    }
    var thead = el('thead');
    var tr = el('tr');
    cols.forEach(function (c) {
      var th = el('th', { class: c.sortable ? 'sortable' : '', text: c.label });
      if (c.sortable) {
        th.addEventListener('click', function () {
          if (state.sort === c.key) state.dir = -state.dir;
          else { state.sort = c.key; state.dir = c.key === 'rank' ? 1 : -1; }
          render();
        });
        if (state.sort === c.key) th.appendChild(el('span', { class: 'arrow', text: state.dir === 1 ? ' \u25B2' : ' \u25BC' }));
      }
      tr.appendChild(th);
    });
    thead.appendChild(tr);
    table.appendChild(thead);
    var tbody = el('tbody');
    if (rows.length === 0) {
      tbody.appendChild(el('tr', {}, el('td', { colspan: String(cols.length), class: 'empty', text: 'No protocols match these filters.' })));
    } else {
      rows.forEach(function (r) { tbody.appendChild(rowTr(r)); });
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
    app.textContent = '';
    app.appendChild(wrap);
  }

  function columnsFor(tab) {
    if (tab === 'radar') {
      return [
        { key: 'rank', label: '#', sortable: true },
        { key: 'name', label: 'Protocol', sortable: true },
        { key: 'category', label: 'Category', sortable: true },
        { key: 'tvl', label: 'TVL', sortable: true },
        { key: 'change', label: '7d change', sortable: true },
        { key: 'age', label: 'Age', sortable: true },
        { key: 'odds', label: 'Odds', sortable: true },
        { key: 'iv', label: 'Implied value', sortable: true },
        { key: 'net', label: 'Farm', sortable: true },
      ];
    }
    if (tab === 'live') {
      return [
        { key: 'name', label: 'Protocol', sortable: true },
        { key: 'category', label: 'Category', sortable: true },
        { key: 'tvl', label: 'TVL', sortable: true },
        { key: 'mcap', label: 'Market cap', sortable: true },
        { key: 'change', label: '7d change', sortable: true },
      ];
    }
    return [
      { key: 'name', label: 'Protocol', sortable: true },
      { key: 'category', label: 'Category', sortable: true },
      { key: 'tvl', label: 'TVL', sortable: true },
      { key: 'mcap', label: 'Market cap', sortable: true },
      { key: 'change', label: '7d change', sortable: true },
    ];
  }

  function rowTr(r) {
    var tab = state.tab;
    if (tab === 'radar') {
      var implied = impliedValueOf(r);
      var tr = el('tr', { onclick: function () { openDetail(r); } });
      tr.appendChild(el('td', { class: 'rank' + (r.rank <= 3 ? ' top3' : ''), text: String(r.rank) }));
      var nameTd = el('td', { class: 'name' }, el('span', { text: r.name }), el('span', { class: 'slug', text: r.slug }));
      if (r.is_watchlist) nameTd.appendChild(el('span', { class: 'badge wl', text: 'WL' }));
      tr.appendChild(nameTd);
      tr.appendChild(el('td', { class: 'cat', text: r.category || '\u2014' }));
      tr.appendChild(el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }));
      var ch = fmtPct(r.change_7d);
      tr.appendChild(el('td', { class: 'num pct' + (r.change_7d !== null && r.change_7d < 0 ? ' neg' : r.change_7d !== null && r.change_7d > 0 ? ' pos' : ''), text: ch }));
      tr.appendChild(el('td', { class: 'num', text: fmtAge(r.age_days) }));
      tr.appendChild(el('td', { class: 'num', text: (r.token_odds * 100).toFixed(1) + '%' }));
      tr.appendChild(el('td', { class: 'iv', text: fmtUsd(r.implied_value_usd) }));
      tr.appendChild(el('td', { class: 'num' }, el('span', { class: 'verdict ' + r.farm.verdict, text: r.farm.verdict })));
      return tr;
    }
    if (tab === 'live') {
      var tr2 = el('tr', { onclick: function () { openLive(r); } });
      tr2.appendChild(el('td', { class: 'name' }, el('span', { text: r.name }), el('span', { class: 'slug', text: r.slug })));
      tr2.appendChild(el('td', { class: 'cat', text: r.category || '\u2014' }));
      tr2.appendChild(el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }));
      tr2.appendChild(el('td', { class: 'num' }, el('span', { class: 'badge mcap', text: r.symbol || 'token' }), el('span', { text: ' ' + fmtUsd(r.mcap_usd) })));
      var ch2 = fmtPct(r.change_7d);
      tr2.appendChild(el('td', { class: 'num pct' + (r.change_7d !== null && r.change_7d < 0 ? ' neg' : r.change_7d !== null && r.change_7d > 0 ? ' pos' : ''), text: ch2 }));
      return tr2;
    }
    var tr3 = el('tr', { onclick: function () { openProduct(r); } });
    tr3.appendChild(el('td', { class: 'name' }, el('span', { text: r.name }), el('span', { class: 'slug', text: r.slug })));
    tr3.appendChild(el('td', { class: 'cat', text: r.category || '\u2014' }));
    tr3.appendChild(el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }));
    tr3.appendChild(el('td', { class: 'num', text: fmtUsd(r.mcap_usd) }));
    tr3.appendChild(el('td', { class: 'num', text: r.reason }));
    return tr3;
  }

  // ---- detail drawers ------------------------------------------------------
  var drawer = document.getElementById('drawer');
  function closeDrawer() { drawer.hidden = true; drawer.textContent = ''; }

  function openDetail(r) {
    var comps = compsValueOf(r);
    var odds = oddsOf(r);
    var implied = impliedValueOf(r);
    var scrim = el('div', { class: 'scrim', onclick: closeDrawer });
    var panel = el('div', { class: 'panel' });
    panel.appendChild(el('button', { class: 'close', text: '\u00D7', onclick: closeDrawer }));
    panel.appendChild(el('h2', { text: r.name }));
    panel.appendChild(el('div', { class: 'slugline', text: r.slug + (r.is_watchlist ? ' \u00B7 watchlist' : '') }));
    var grid = el('div', { class: 'detail-grid' });
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'TVL' }), el('div', { class: 'v', text: fmtUsd(r.tvl_usd) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Category' }), el('div', { class: 'v', text: r.category || '\u2014' })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: '7-day growth' }), el('div', { class: 'v' + (r.change_7d !== null && r.change_7d < 0 ? ' neg' : ''), text: fmtPct(r.change_7d) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Listed (age)' }), el('div', { class: 'v', text: (r.listed_at || 'unknown') + ' \u00B7 ' + fmtAge(r.age_days) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Comps value' }), el('div', { class: 'v', text: fmtUsd(comps) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Token odds' }), el('div', { class: 'v', text: (odds * 100).toFixed(1) + '%' })));
    panel.appendChild(grid);
    var chains = el('div', { class: 'chains' }, (r.chains || []).map(function (c) { return el('span', { class: 'chain-tag', text: c }); }));
    panel.appendChild(el('div', { class: 'detail-grid' }, el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Chains' }), chains)));

    // Value breakdown
    var bd = el('div', { class: 'farm-panel' });
    bd.appendChild(el('h3', { text: 'Implied value' }));
    bd.appendChild(el('div', { class: 'rows' },
      el('div', { class: 'row' }, el('span', { text: 'Comps value (fitted TVL\u2192mcap curve)' }), el('b', { text: fmtUsd(comps) })),
      el('div', { class: 'row' }, el('span', { text: 'Token odds (fitted issue chance)' }), el('b', { text: (odds * 100).toFixed(1) + '%' })),
      el('div', { class: 'row total' }, el('span', { text: 'Implied value = comps \u00D7 odds' }), el('b', { text: fmtUsd(implied) }))
    ));
    panel.appendChild(bd);

    // Farm economics
    var f = r.farm;
    var timeValue = Math.round(f.total_time_hours * f.dollars_per_hour);
    var fe = el('div', { class: 'farm-panel' });
    fe.appendChild(el('h3', { text: 'Farm economics' }));
    fe.appendChild(el('div', { class: 'rows' },
      el('div', { class: 'row' }, el('span', { text: 'Total farm cost' }), el('b', { text: fmtUsd(f.total_cost_usd) })),
      el('div', { class: 'row' }, el('span', { text: 'Total time' }), el('b', { text: f.total_time_hours + ' hours' })),
      el('div', { class: 'row' }, el('span', { text: 'Expected value (comps \u00D7 odds\u00B2)' }), el('b', { text: fmtUsd(f.expected_value_usd) })),
      el('div', { class: 'row' }, el('span', { text: 'Time value at $' + f.dollars_per_hour + '/h' }), el('b', { text: fmtUsd(timeValue) })),
      el('div', { class: 'row' }, el('span', { text: 'Net dollar value' }), el('b', { text: fmtUsd(f.net_value_usd) })),
      el('div', { class: 'row' }, el('span', { text: 'Verdict' }), el('b', { text: f.verdict })),
      el('div', { class: 'row total' }, el('span', { text: 'Break-even implied value' }), el('b', { text: fmtUsd(f.break_even_value_usd) }))
    ));
    panel.appendChild(fe);

    var farm = el('div', { class: 'farm' });
    farm.appendChild(el('h3', { text: 'How to farm it' }));
    var ol = el('ol');
    (r.farm_steps || []).forEach(function (s) { ol.appendChild(el('li', { text: s })); });
    farm.appendChild(ol);
    panel.appendChild(farm);

    var assume = el('div', { class: 'assumptions' });
    assume.textContent = DATA.farm.assumptions.join(' ');
    panel.appendChild(assume);

    drawer.textContent = '';
    drawer.appendChild(scrim);
    drawer.appendChild(panel);
    drawer.hidden = false;
  }

  function openLive(r) {
    var scrim = el('div', { class: 'scrim', onclick: closeDrawer });
    var panel = el('div', { class: 'panel' });
    panel.appendChild(el('button', { class: 'close', text: '\u00D7', onclick: closeDrawer }));
    panel.appendChild(el('h2', { text: r.name }));
    panel.appendChild(el('div', { class: 'slugline', text: r.slug + ' \u00B7 token already trades' }));
    var grid = el('div', { class: 'detail-grid' });
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Market cap' }), el('div', { class: 'v hl', text: fmtUsd(r.mcap_usd) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'TVL' }), el('div', { class: 'v', text: fmtUsd(r.tvl_usd) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Symbol' }), el('div', { class: 'v', text: r.symbol || '\u2014' })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Category' }), el('div', { class: 'v', text: r.category || '\u2014' })));
    panel.appendChild(grid);
    var note = el('div', { class: 'farm-panel' });
    note.appendChild(el('div', { class: 'rows' }, el('div', { class: 'row' }, el('span', { text: 'This protocol already records a token, so it is set aside from the launch radar and shown at the market\u2019s own value.' }))));
    panel.appendChild(note);
    drawer.textContent = '';
    drawer.appendChild(scrim);
    drawer.appendChild(panel);
    drawer.hidden = false;
  }

  function openProduct(r) {
    var scrim = el('div', { class: 'scrim', onclick: closeDrawer });
    var panel = el('div', { class: 'panel' });
    panel.appendChild(el('button', { class: 'close', text: '\u00D7', onclick: closeDrawer }));
    panel.appendChild(el('h2', { text: r.name }));
    panel.appendChild(el('div', { class: 'slugline', text: r.slug + ' \u00B7 set aside: ' + r.reason }));
    var grid = el('div', { class: 'detail-grid' });
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'TVL' }), el('div', { class: 'v', text: fmtUsd(r.tvl_usd) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Market cap' }), el('div', { class: 'v', text: fmtUsd(r.mcap_usd) })));
    grid.appendChild(el('div', { class: 'cell' }, el('div', { class: 'k', text: 'Category' }), el('div', { class: 'v', text: r.category || '\u2014' })));
    panel.appendChild(grid);
    var note = el('div', { class: 'farm-panel' });
    note.appendChild(el('div', { class: 'rows' }, el('div', { class: 'row' }, el('span', { text: 'This is a wrapped, staked, bridged, exchange or versioned product of a token that already trades, so it is not a launch candidate and is not ranked.' }))));
    panel.appendChild(note);
    drawer.textContent = '';
    drawer.appendChild(scrim);
    drawer.appendChild(panel);
    drawer.hidden = false;
  }

  // ---- wire up controls ----------------------------------------------------
  document.querySelectorAll('.tab').forEach(function (btn) {
    btn.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(function (b) { b.classList.remove('active'); });
      btn.classList.add('active');
      state.tab = btn.getAttribute('data-tab');
      if (state.tab !== 'radar') { state.sort = state.tab === 'live' ? 'mcap' : 'tvl'; state.dir = -1; }
      else { state.sort = 'rank'; state.dir = 1; }
      document.getElementById('wl').checked = false;
      state.wl = false;
      render();
    });
  });
  document.getElementById('q').addEventListener('input', function (e) { state.q = e.target.value; render(); });
  document.getElementById('cat').addEventListener('change', function (e) { state.cat = e.target.value; render(); });
  document.getElementById('wl').addEventListener('change', function (e) { state.wl = e.target.checked; render(); });

  // ---- deep links for the screenshots (board, detail, filtered) -----------
  function applyHash() {
    var h = (location.hash || '').replace(/^#/, '');
    if (!h) { render(); return; }
    var parts = h.split('&');
    var detailSlug = null;
    parts.forEach(function (part) {
      var kv = part.split('=');
      var k = decodeURIComponent(kv[0]);
      var v = kv.length > 1 ? decodeURIComponent(kv.slice(1).join('=')) : '';
      if (k === 'detail') detailSlug = v;
      else if (k === 'q') state.q = v;
      else if (k === 'cat') state.cat = v;
      else if (k === 'wl') state.wl = v === '1';
      else if (k === 'tab') state.tab = v;
    });
    render();
    if (detailSlug) {
      var row = DATA.rows.filter(function (r) { return r.slug === detailSlug; })[0];
      if (row) openDetail(row);
    }
  }
  window.addEventListener('hashchange', applyHash);

  // initial render
  applyHash();
})();
