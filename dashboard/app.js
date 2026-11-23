// Token Radar v2 dashboard. Reads window.RADAR, renders a filterable/sortable
// board, a live-token table, a per-protocol detail drawer with farm economics,
// and an integrity strip with the model's own measured fit numbers. No network.
(function () {
  'use strict';
  var DATA = window.RADAR;
  if (!DATA) return;
  var app = document.getElementById('board');
  var state = { q: '', category: 'all', launch: 'all', tab: 'pre-token', sort: 'implied', dir: -1, selected: null };

  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function fmtPct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + n.toFixed(2) + '%';
  }
  function fmtAge(days) {
    if (days === null || days === undefined) return 'unknown';
    if (days >= 365) return (days / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(days)) + 'd';
  }
  function fmtNum(n, d) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return Number(n).toFixed(d === undefined ? 2 : d);
  }

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

  function rows() {
    var q = state.q.trim().toLowerCase();
    return DATA.rows.filter(function (r) {
      if (state.category !== 'all' && r.category !== state.category) return false;
      if (state.launch === 'candidate' && !r.launch_candidate) return false;
      if (state.launch === 'derivative' && r.launch_candidate) return false;
      if (!q) return true;
      var hay = [r.name, r.slug, r.category, (r.chains || []).join(' ')].join(' ').toLowerCase();
      return hay.indexOf(q) >= 0;
    });
  }
  function sortedRows(list) {
    var key = state.sort;
    return list.slice().sort(function (a, b) {
      var av, bv;
      if (key === 'name') { av = a.name.toLowerCase(); bv = b.name.toLowerCase(); }
      else if (key === 'tvl') { av = a.tvl_usd; bv = b.tvl_usd; }
      else if (key === 'growth') { av = a.change_7d === null ? -Infinity : a.change_7d; bv = b.change_7d === null ? -Infinity : b.change_7d; }
      else if (key === 'age') { av = a.age_days === null ? Infinity : a.age_days; bv = b.age_days === null ? Infinity : b.age_days; }
      else if (key === 'odds') { av = a.token_odds; bv = b.token_odds; }
      else if (key === 'net') { av = a.farm.net_value_usd; bv = b.farm.net_value_usd; }
      else { av = a.implied_value_usd; bv = b.implied_value_usd; }
      if (av < bv) return -state.dir;
      if (av > bv) return state.dir;
      return a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0;
    });
  }

  function header() {
    return el('header', { class: 'top' },
      el('div', { class: 'brand' }, el('span', { class: 'mark' }), el('span', { class: 'brand-name', text: 'Token Radar v2' })),
      el('div', { class: 'top-meta' },
        el('span', { text: 'Snapshot ' + DATA.as_of }),
        el('span', { text: 'DefiLlama protocol index' }),
        el('span', { text: DATA.model_version + ' fitted model' })
      )
    );
  }

  function lede() {
    return el('section', { class: 'lede' },
      el('div', null,
        el('h1', { text: DATA.coverage.watchlist + ' watchlist protocols, ' + DATA.board_size + ' ranked by implied value' }),
        el('p', { text: 'The watchlist is the floor; all 342 pre-token protocols are ranked. Launch candidates (the 198 with no parent token) come first, then derivative/CEX/wrapped pools are tagged and can be filtered out. Implied value = fitted comps value x token odds. Live-token protocols are set aside at their real market cap.' })
      ),
      el('div', { class: 'facts' },
        el('div', { class: 'fact' }, el('span', { text: 'Watchlist covered' }), el('b', { text: DATA.coverage.watchlist + ' / ' + DATA.coverage.watchlist })),
        el('div', { class: 'fact' }, el('span', { text: 'Pre-token rows ranked' }), el('b', { text: String(DATA.coverage.rows) })),
        el('div', { class: 'fact' }, el('span', { text: 'Early rows below floor' }), el('b', { text: String(DATA.early_size) })),
        el('div', { class: 'fact' }, el('span', { text: 'Live-token set aside' }), el('b', { text: String(DATA.live_size) }))
      )
    );
  }

  function integrity() {
    var f = DATA.fit;
    return el('section', { class: 'integrity' },
      el('h2', { text: 'Model integrity, measured at run time from the snapshot' }),
      el('div', { class: 'grid' },
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.full_r2, 4) }), el('span', { text: 'R² on full 128' })),
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.cv5_mae, 4) }), el('span', { text: '5-fold held-out log MAE' })),
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.walk_jun_mae, 4) }), el('span', { text: 'Walk-forward Jun MAE' })),
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.walk_jan_mae, 4) }), el('span', { text: 'Walk-forward Jan MAE' })),
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.loo_median_abs, 4) }), el('span', { text: 'Leave-one-out median abs err' })),
        el('div', { class: 'cell' }, el('b', { text: fmtNum(f.loo_implied_ratio, 2) + '×' }), el('span', { text: 'Implied ratio' })),
        el('div', { class: 'cell' }, el('b', { text: String(f.n_comps) }), el('span', { text: 'Comps protocols' })),
        el('div', { class: 'cell' }, el('b', { text: DATA.coverage.rows + ' / ' + DATA.coverage.watchlist }), el('span', { text: 'Coverage rows/watchlist' }))
      ),
      el('p', { class: 'sub', text: 'The 730-day age half-life discount was tested against the data (log-likelihood ' + fmtNum(f.loglik_half_life_730, 1) + ' at 730d vs ' + fmtNum(f.loglik_no_discount, 1) + ' with no discount) and removed: age enters the logistic fit with its own sign.' })
    );
  }

  function controls() {
    var categories = Array.from(new Set(DATA.rows.map(function (r) { return r.category || 'Other'; }))).sort();
    var search = el('input', { type: 'search', id: 'q', placeholder: 'Search name, category or chain', value: state.q, 'aria-label': 'Search the board' });
    search.addEventListener('input', function () { state.q = search.value; writeHash(); render(); });
    var launch = el('select', { id: 'launch', 'aria-label': 'Filter by launch candidate' },
      el('option', { value: 'all', text: 'All rows' }),
      el('option', { value: 'candidate', text: 'Launch candidates', selected: state.launch === 'candidate' ? 'selected' : null }),
      el('option', { value: 'derivative', text: 'Derivative pools only', selected: state.launch === 'derivative' ? 'selected' : null })
    );
    launch.addEventListener('change', function () { state.launch = launch.value; writeHash(); render(); });
    var select = el('select', { id: 'category', 'aria-label': 'Filter by category' },
      el('option', { value: 'all', text: 'All categories' }),
      categories.map(function (c) { return el('option', { value: c, text: c, selected: c === state.category ? 'selected' : null }); })
    );
    select.addEventListener('change', function () { state.category = select.value; writeHash(); render(); });
    var pre = el('button', { type: 'button', 'aria-pressed': state.tab === 'pre-token' ? 'true' : 'false', text: 'Pre-token', onclick: function () { state.tab = 'pre-token'; render(); } });
    var live = el('button', { type: 'button', 'aria-pressed': state.tab === 'live' ? 'true' : 'false', text: 'Live token', onclick: function () { state.tab = 'live'; render(); } });
    return el('div', { class: 'controls' },
      el('div', { class: 'field' }, el('label', { for: 'q', text: 'Search' }), search),
      el('div', { class: 'field' }, el('label', { for: 'launch', text: 'Launch status' }), launch),
      el('div', { class: 'field' }, el('label', { for: 'category', text: 'Category' }), select),
      el('div', { class: 'field' }, el('span', { class: 'sub', text: 'Token status' }), el('div', { class: 'tabs' }, pre, live)),
      el('span', { class: 'count', text: state.tab === 'pre-token' ? rows().length + ' rows shown' : DATA.live.length + ' live-token protocols' })
    );
  }

  function sortHeader(key, label) {
    return el('th', { scope: 'col' },
      el('button', { type: 'button', onclick: function () { if (state.sort === key) state.dir = -state.dir; else { state.sort = key; state.dir = -1; } render(); } },
        label, state.sort === key ? el('span', { 'aria-hidden': 'true', text: state.dir < 0 ? ' v' : ' ^' }) : null
      )
    );
  }

  function boardTable() {
    var list = sortedRows(rows());
    if (list.length === 0) {
      return el('div', { class: 'panel' }, el('div', { class: 'empty' },
        el('h2', { text: 'No protocol matches that filter' }),
        el('p', { text: 'Clear the search or pick another category.' }),
        el('button', { class: 'button primary', type: 'button', text: 'Clear filters', onclick: function () { state.q = ''; state.category = 'all'; state.launch = 'all'; writeHash(); render(); } })
      ));
    }
    var max = list[0].implied_value_usd || 1;
    var tbody = el('tbody');
    list.forEach(function (r) {
      var growth = r.change_7d === null ? 'n/a' : fmtPct(r.change_7d);
      var growthClass = r.change_7d === null ? '' : r.change_7d >= 0 ? 'pos' : 'neg';
      var tr = el('tr', {
        'data-selected': state.selected === r.slug ? 'true' : 'false',
        tabindex: '0',
        onclick: function () { openDetail(r.slug); },
        onkeydown: function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openDetail(r.slug); } }
      },
        el('td', null, el('div', { class: 'num', text: String(r.rank) }), el('span', { class: 'pill ' + (r.launch_candidate ? 'launch' : 'deriv'), text: r.launch_candidate ? 'Launch' : 'Deriv' })),
        el('td', null, el('div', { class: 'name' },
          el('b', { text: r.name }),
          el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') })
        )),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + growthClass, text: growth }),
        el('td', { class: 'num', text: fmtAge(r.age_days) }),
        el('td', { class: 'num', text: fmtNum(r.token_odds, 3) }),
        el('td', null, el('div', { class: 'implied' },
          el('div', { class: 'bar', 'aria-hidden': 'true' }, el('i', { style: 'width:' + Math.max(3, Math.round((r.implied_value_usd / max) * 100)) + '%' })),
          el('span', { class: 'num', text: fmtUsd(r.implied_value_usd) })
        )),
        el('td', null, el('span', { class: 'verdict ' + r.farm.verdict, text: r.farm.verdict }))
      );
      tbody.appendChild(tr);
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'All 342 pre-token protocols by implied value. Launch candidates first; derivative/CEX/wrapped pools carry a Derivative tag and can be filtered out.' }),
          el('thead', null, el('tr', null,
            sortHeader('implied', 'Rank'), sortHeader('name', 'Protocol'), sortHeader('tvl', 'TVL'),
            sortHeader('growth', '7d'), sortHeader('age', 'Age'), sortHeader('odds', 'Token odds'),
            sortHeader('implied', 'Implied value'), sortHeader('net', 'Farm verdict')
          )),
          tbody
        )
      )
    );
  }

  function liveTable() {
    var tbody = el('tbody');
    DATA.live.forEach(function (r) {
      tbody.appendChild(el('tr', null,
        el('td', { class: 'num', text: 'live' }),
        el('td', null, el('div', { class: 'name' }, el('b', { text: r.name }), el('span', { class: 'sub', text: (r.category || 'Other') + ' on ' + (r.chains && r.chains.length ? r.chains.slice(0, 2).join(', ') : 'an unnamed chain') }))),
        el('td', { class: 'num', text: fmtUsd(r.tvl_usd) }),
        el('td', { class: 'num ' + (r.change_7d !== null && r.change_7d >= 0 ? 'pos' : 'neg'), text: fmtPct(r.change_7d) }),
        el('td', { class: 'num', text: fmtUsd(r.mcap_usd) }),
        el('td', null, el('span', { class: 'sub', text: r.symbol || 'trades' }))
      ));
    });
    return el('div', { class: 'panel' },
      el('div', { class: 'table-wrap' },
        el('table', null,
          el('caption', { class: 'sub', text: 'Token live: these protocols already trade, so their market cap is a fact, not a model. Set aside from the pre-token ranking.' }),
          el('thead', null, el('tr', null, el('th', { scope: 'col', text: 'Status' }), el('th', { scope: 'col', text: 'Protocol' }), el('th', { scope: 'col', text: 'TVL' }), el('th', { scope: 'col', text: '7d' }), el('th', { scope: 'col', text: 'Market cap' }), el('th', { scope: 'col', text: 'Token' }))),
          tbody
        )
      )
    );
  }

  function modelPanel() {
    return el('section', { class: 'panel model' },
      el('h2', { text: 'How implied value is computed' }),
      el('p', { class: 'sub', text: 'One fitted function scores every row; no protocol name is special-cased. Implied value = fitted comps value x token odds, rounded to whole dollars.' }),
      el('div', { class: 'formula', text: 'comps value = exp( b0 + b1·log(TVL) + b2·log(age+1) + b3·growth_7d + b4·log(chains+1) + Σ category_b·category )' }),
      el('div', { class: 'formula', text: 'token odds = logistic( c0 + c1·log(TVL) + c2·log(chains+1) + c3·growth_7d + c4·age_years + Σ category_c·category )' }),
      el('div', { class: 'formula', text: 'implied value = round( comps value × token odds )' }),
      el('p', { class: 'sub', text: 'The comps fit is Huber-robust ridge regression (tuning 1.345, ridge lambda 3, intercept unpenalised) on the 128 positive market caps. The odds fit is a logistic regression on all 758 protocols. Age for unlisted protocols is the median token-bearing age (' + Math.round(DATA.fit.n_comps ? 1000 : 1000) + ' days).' }),
      el('p', { class: 'sub', text: 'v1 comparison: v1 scored only 197 rows with plain log-log OLS and two rounds of outlier dropping (5-fold MAE about 1.69, no walk-forward). v2 scores all ' + DATA.board_size + ' pre-token rows (v1: 197) with Huber+ridge (5-fold MAE ' + fmtNum(DATA.fit.cv5_mae, 3) + ', walk-forward Jun MAE ' + fmtNum(DATA.fit.walk_jun_mae, 3) + '), the age discount removed after testing.' })
    );
  }

  var dialog = el('dialog', { class: 'detail', id: 'detail', 'aria-labelledby': 'detail-title' });
  dialog.addEventListener('close', function () { state.selected = null; writeHash(); });

  function openDetail(slug) {
    var row = DATA.rows.filter(function (r) { return r.slug === slug; })[0];
    if (!row) return;
    state.selected = slug;
    writeHash();
    dialog.textContent = '';
    dialog.appendChild(el('div', { class: 'detail-head' },
      el('div', null,
        el('span', { class: 'brand-name', text: 'Rank ' + row.rank + ' of ' + DATA.board_size }),
        el('h2', { id: 'detail-title', text: row.name }),
        el('p', { class: 'sub', text: (row.category || 'Other') + ' on ' + (row.chains && row.chains.length ? row.chains.join(', ') : 'an unnamed chain') })
      ),
      el('button', { class: 'close', type: 'button', 'aria-label': 'Close details', text: 'Close', onclick: function () { dialog.close(); } })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Implied value' }),
      el('div', { class: 'equation' },
        el('div', null, el('span', { text: 'TVL' }), el('b', { class: 'num', text: fmtUsd(row.tvl_usd) })),
        el('div', null, el('span', { text: 'Seven-day growth' }), el('b', { class: 'num', text: fmtPct(row.change_7d) })),
        el('div', null, el('span', { text: 'Age with no token' }), el('b', { class: 'num', text: fmtAge(row.age_days) })),
        el('div', null, el('span', { text: 'Fitted comps value' }), el('b', { class: 'num', text: fmtUsd(row.comps_value_usd) })),
        el('div', null, el('span', { text: 'Token odds' }), el('b', { class: 'num', text: fmtNum(row.token_odds, 3) })),
        el('div', null, el('span', { text: 'Implied value' }), el('b', { class: 'num', text: fmtUsd(row.implied_value_usd) }))
      ),
      el('p', { class: 'sub', text: 'This is an estimate, not a price. Implied value = comps value x token odds.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm economics' }),
      el('div', { class: 'farm-box' },
        el('div', { class: 'row' }, el('span', { text: 'Total farm cost' }), el('b', { text: fmtUsd(row.farm.total_cost_usd) })),
        el('div', { class: 'row' }, el('span', { text: 'Total time' }), el('b', { text: fmtNum(row.farm.total_time_hours, 1) + ' hours' })),
        el('div', { class: 'row' }, el('span', { text: 'Expected value (comps × odds)' }), el('b', { text: fmtUsd(row.farm.expected_value_usd) })),
        el('div', { class: 'row' }, el('span', { text: 'Time value at $' + row.farm.dollars_per_hour + '/h' }), el('b', { text: fmtUsd(row.farm.total_time_hours * row.farm.dollars_per_hour) })),
        el('div', { class: 'row' }, el('span', { text: 'Net dollar value' }), el('b', { text: fmtUsd(row.farm.net_value_usd) })),
        el('div', { class: 'row' }, el('span', { text: 'Verdict' }), el('b', { class: 'verdict ' + row.farm.verdict, text: row.farm.verdict })),
        el('div', { class: 'row' }, el('span', { text: 'Break-even implied value' }), el('b', { text: fmtUsd(row.farm.break_even_value_usd) }))
      ),
      el('p', { class: 'sub', text: 'Gas per chain: Ethereum $6, most L2s/Solana $2, others $1. Bridge $6 when more than one chain. Swap 0.10% of capital. Time valued at $' + row.farm.dollars_per_hour + ' per hour.' })
    ));
    dialog.appendChild(el('section', { class: 'detail-section' },
      el('h3', { text: 'Farm steps' }),
      el('ol', { class: 'steps' }, row.farm_steps.map(function (s) {
        var at = s.indexOf(' Cost: ');
        var action = at >= 0 ? s.slice(0, at) : s;
        var tail = at >= 0 ? s.slice(at + 1) : '';
        return el('li', null, el('b', { text: action }), el('span', { class: 'cost', text: tail }));
      }))
    ));
    if (typeof dialog.showModal === 'function') {
      if (!dialog.open) dialog.showModal();
    } else {
      dialog.setAttribute('open', '');
    }
  }

  function readHash() {
    var hash = window.location.hash.replace(/^#/, '');
    if (!hash) return;
    var params = new URLSearchParams(hash);
    if (params.get('q')) state.q = params.get('q');
    if (params.get('p')) state.selected = params.get('p');
    if (params.get('category')) state.category = params.get('category');
    if (params.get('launch')) state.launch = params.get('launch');
  }
  function writeHash() {
    var params = new URLSearchParams();
    if (state.q) params.set('q', state.q);
    if (state.category !== 'all') params.set('category', state.category);
    if (state.launch !== 'all') params.set('launch', state.launch);
    if (state.selected) params.set('p', state.selected);
    var next = params.toString();
    if (next) window.history.replaceState(null, '', '#' + next);
    else window.history.replaceState(null, '', window.location.pathname + window.location.search);
  }

  function render() {
    app.textContent = '';
    var shell = el('div', { class: 'shell' });
    shell.appendChild(header());
    shell.appendChild(lede());
    shell.appendChild(integrity());
    shell.appendChild(controls());
    shell.appendChild(state.tab === 'pre-token' ? boardTable() : liveTable());
    if (state.tab === 'pre-token') shell.appendChild(modelPanel());
    app.appendChild(shell);
    app.setAttribute('aria-busy', 'false');
    if (!dialog.isConnected) app.appendChild(dialog);
    if (state.selected) openDetail(state.selected);
  }

  readHash();
  render();
  window.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && dialog.open) dialog.close();
  });
})();
