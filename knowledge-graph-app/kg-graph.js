// ════════════════════════════════════════════════════
// kg-graph.js — D3 rendering, graph data, save/load, UI helpers, init
// ════════════════════════════════════════════════════

const LS_GRAPH       = 'kg_graph_data';
const NODE_R         = 22;
const COLOR_BY_DEPTH = ['#5b8dee','#a78bfa','#34d399','#f59e0b','#f87171','#e879f9'];

let svg, g, simulation, zoomBehavior;

// ── Graph Data ──────────────────────────────────────

function mergeGraph(newNodes, newEdges, parentDepth = -1) {
  const existingIds = new Set(state.nodes.map(n => n.id));
  newNodes.forEach(n => {
    if (!existingIds.has(n.id)) {
      state.nodes.push({
        id: n.id,
        label: n.label,
        summary: n.summary || '',
        explored: false,
        depth: parentDepth + 1,
        properties: {}
      });
      existingIds.add(n.id);
    }
  });
  const existingEdgeIds = new Set(state.edges.map(e => `${e.source}->${e.target}`));
  newEdges.forEach(e => {
    const key = `${e.source}->${e.target}`;
    if (!existingEdgeIds.has(key)) {
      state.edges.push({ id: key, source: e.source, target: e.target, relation: e.relation || '' });
      existingEdgeIds.add(key);
    }
  });
}

function clearGraph() {
  state.nodes = [];
  state.edges = [];
}

// ── D3 Rendering ────────────────────────────────────

function initGraph() {
  svg = d3.select('#graph-svg');
  svg.selectAll('*').remove();

  // Arrowhead marker
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrow')
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', NODE_R + 10)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', '#2e3245');

  g = svg.append('g');

  // Two fixed sub-layers: links always below nodes in DOM order
  g.append('g').attr('class', 'links-layer');
  g.append('g').attr('class', 'nodes-layer');

  zoomBehavior = d3.zoom()
    .scaleExtent([0.1, 4])
    .on('zoom', e => g.attr('transform', e.transform));
  svg.call(zoomBehavior);

  simulation = d3.forceSimulation()
    .force('link', d3.forceLink().id(d => d.id).distance(130).strength(0.6))
    .force('charge', d3.forceManyBody().strength(-420))
    .force('center', d3.forceCenter(svg.node().clientWidth / 2, svg.node().clientHeight / 2))
    .force('collision', d3.forceCollide(NODE_R + 20));

  // Dummy data so canvas is visible on first load
  state.nodes = [
    { id: 'root', label: 'Your Topic', summary: 'Enter a topic above to generate a real knowledge graph.', explored: false, depth: 0, properties: {} },
    { id: 'n1',   label: 'Concept A',  summary: '', explored: false, depth: 1, properties: {} },
    { id: 'n2',   label: 'Concept B',  summary: '', explored: false, depth: 1, properties: {} },
    { id: 'n3',   label: 'Concept C',  summary: '', explored: false, depth: 1, properties: {} },
    { id: 'n4',   label: 'Concept D',  summary: '', explored: false, depth: 2, properties: {} },
    { id: 'n5',   label: 'Concept E',  summary: '', explored: false, depth: 2, properties: {} },
  ];
  state.edges = [
    { id: 'e1', source: 'root', target: 'n1', relation: 'includes' },
    { id: 'e2', source: 'root', target: 'n2', relation: 'relates to' },
    { id: 'e3', source: 'root', target: 'n3', relation: 'uses' },
    { id: 'e4', source: 'n1',   target: 'n4', relation: 'leads to' },
    { id: 'e5', source: 'n2',   target: 'n5', relation: 'enables' },
  ];
  updateGraph();
}

function updateGraph() {
  const width  = svg.node().clientWidth;
  const height = svg.node().clientHeight;
  simulation.force('center', d3.forceCenter(width / 2, height / 2));

  // ── Links (always in links-layer, beneath nodes) ──
  const linksLayer = g.select('.links-layer');
  const linkGroup  = linksLayer.selectAll('.link-group').data(state.edges, d => d.id);
  const linkEnter  = linkGroup.enter().append('g').attr('class', 'link-group');
  linkEnter.append('line').attr('class', 'link').attr('marker-end', 'url(#arrow)');
  linkEnter.append('text').attr('class', 'link-label');
  linkGroup.exit().remove();
  const linkMerge = linkEnter.merge(linkGroup);
  linkMerge.select('.link-label').text(d => d.relation);

  // ── Nodes (always in nodes-layer, on top) ──
  const nodesLayer = g.select('.nodes-layer');
  const nodeGroup  = nodesLayer.selectAll('.node-group').data(state.nodes, d => d.id);
  const nodeEnter = nodeGroup.enter().append('g').attr('class', 'node-group')
    .call(d3.drag()
      .on('start', dragStarted)
      .on('drag',  dragged)
      .on('end',   dragEnded));

  nodeEnter.append('circle')
    .attr('class', 'node-circle')
    .attr('r', 0)
    .transition().duration(400)
    .attr('r', NODE_R);

  nodeEnter.append('text')
    .attr('class', 'node-label')
    .attr('y', NODE_R + 5);

  // Info badge — small circle + "i" text, anchored top-right of node
  const infoBadge = nodeEnter.append('g')
    .attr('class', 'node-info-badge')
    .attr('transform', `translate(${NODE_R - 7}, ${-NODE_R + 7})`);

  infoBadge.append('circle')
    .attr('class', 'node-info-bg')
    .attr('r', 7)
    .attr('fill', '#1a1d27')
    .attr('stroke', '#5b8dee')
    .attr('stroke-width', 1.5);

  infoBadge.append('text')
    .attr('class', 'node-info-icon')
    .attr('text-anchor', 'middle')
    .attr('dominant-baseline', 'central')
    .attr('x', 0)
    .attr('y', 0)
    .text('i');

  nodeGroup.exit().remove();
  const nodeMerge = nodeEnter.merge(nodeGroup);

  // Style circles by depth / explored state
  nodeMerge.select('.node-circle')
    .attr('fill',   d => d.explored ? '#1e2235' : COLOR_BY_DEPTH[Math.min(d.depth, COLOR_BY_DEPTH.length - 1)])
    .attr('stroke', d => d.explored ? '#4a5280' : COLOR_BY_DEPTH[Math.min(d.depth, COLOR_BY_DEPTH.length - 1)])
    .attr('stroke-width', d => d.explored ? 2.5 : 2)
    .attr('opacity', d => d.explored ? 0.7 : 1);

  // Truncate long labels
  nodeMerge.select('.node-label')
    .text(d => d.label.length > 16 ? d.label.slice(0, 14) + '…' : d.label);

  // Click handlers
  nodeMerge.select('.node-circle').on('click', (event, d) => {
    event.stopPropagation();
    expandNode(d);
  });
  nodeMerge.select('.node-info-badge').on('click', (event, d) => {
    event.stopPropagation();
    const rect = event.currentTarget.getBoundingClientRect();
    showSummaryPopup(d, rect.left + rect.width / 2, rect.top + rect.height / 2);
  });

  // Simulation tick
  simulation.nodes(state.nodes).on('tick', ticked);
  simulation.force('link').links(state.edges);
  simulation.alpha(0.4).restart();

  function ticked() {
    linkMerge.select('.link')
      .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
      .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
    linkMerge.select('.link-label')
      .attr('x', d => (d.source.x + d.target.x) / 2)
      .attr('y', d => (d.source.y + d.target.y) / 2);
    nodeMerge.attr('transform', d => `translate(${d.x},${d.y})`);
  }
}

function dragStarted(event, d) {
  if (!event.active) simulation.alphaTarget(0.3).restart();
  d.fx = d.x; d.fy = d.y;
}
function dragged(event, d)   { d.fx = event.x; d.fy = event.y; }
function dragEnded(event, d) {
  if (!event.active) simulation.alphaTarget(0);
  d.fx = null; d.fy = null;
}

function fitToView() {
  if (state.nodes.length === 0) return;
  const bounds = g.node().getBBox();
  const svgEl  = svg.node();
  const w = svgEl.clientWidth, h = svgEl.clientHeight;
  const scale = Math.min(0.9, 0.9 / Math.max(bounds.width / w, bounds.height / h));
  const tx = w / 2 - scale * (bounds.x + bounds.width / 2);
  const ty = h / 2 - scale * (bounds.y + bounds.height / 2);
  svg.transition().duration(600).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(tx, ty).scale(scale)
  );
}

// ── Generate & Expand ───────────────────────────────

async function generateGraph(topic) {
  showLoading(true);
  try {
    const raw    = await callLLM(topic);
    const parsed = parseGraphJSON(raw);
    clearGraph();
    mergeGraph(parsed.nodes || [], parsed.edges || [], -1);
    updateGraph();
    autoSave();
    setTimeout(fitToView, 800);
    showToast(`Graph generated for "${topic}"`, 'success');
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
    console.error(e);
  } finally {
    showLoading(false);
  }
}

async function expandNode(node) {
  if (node.explored) { showToast('Node already explored', ''); return; }
  showLoading(true, `Expanding "${node.label}"…`);
  try {
    const raw    = await callLLM(node.label);
    const parsed = parseGraphJSON(raw);
    mergeGraph(parsed.nodes || [], parsed.edges || [], node.depth);
    // Connect parent node to the root of the new subgraph
    const rootNode = (parsed.nodes || [])[0];
    if (rootNode) {
      const key = `${node.id}->${rootNode.id}`;
      if (!state.edges.find(e => e.id === key)) {
        state.edges.push({ id: key, source: node.id, target: rootNode.id, relation: 'expands to' });
      }
    }
    node.explored = true;
    updateGraph();
    autoSave();
    showToast(`Expanded "${node.label}"`, 'success');
  } catch(e) {
    showToast('Error: ' + e.message, 'error');
    console.error(e);
  } finally {
    showLoading(false);
  }
}

// ── Summary Popup ────────────────────────────────────

const popup      = document.getElementById('summary-popup');
const popTitle   = document.getElementById('popup-title');
const popSummary = document.getElementById('popup-summary');
const popLink    = document.getElementById('popup-link');

function showSummaryPopup(node, x, y) {
  popTitle.textContent   = node.label;
  popSummary.textContent = node.summary || 'No summary available. Click the node to expand it first.';
  const engine = state.settings.searchEngine === 'google'
    ? `https://www.google.com/search?q=${encodeURIComponent(node.label)}`
    : `https://duckduckgo.com/?q=${encodeURIComponent(node.label)}`;
  popLink.href = engine;

  popup.style.left = '-9999px';
  popup.classList.add('visible');
  const pw = popup.offsetWidth, ph = popup.offsetHeight;
  const vw = window.innerWidth;
  let left = x - pw / 2;
  let top  = y - ph - 12;
  if (left + pw > vw - 10) left = vw - pw - 10;
  if (left < 10) left = 10;
  if (top < 10)  top  = y + 30;
  popup.style.left = left + 'px';
  popup.style.top  = top  + 'px';
}

function hidePopup() { popup.classList.remove('visible'); }

document.addEventListener('click', e => {
  if (!popup.contains(e.target)) hidePopup();
});

// ── Save / Load ──────────────────────────────────────

function _serializeEdges() {
  return state.edges.map(e => ({
    id:       e.id,
    source:   typeof e.source === 'object' ? e.source.id : e.source,
    target:   typeof e.target === 'object' ? e.target.id : e.target,
    relation: e.relation
  }));
}

function saveGraph() {
  const data = { version: '1.0', created: new Date().toISOString(), nodes: state.nodes, edges: _serializeEdges() };
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = 'knowledge-graph.json'; a.click();
  URL.revokeObjectURL(url);
  showToast('Graph saved!', 'success');
}

function loadGraph(jsonString) {
  try {
    const data = JSON.parse(jsonString);
    if (!Array.isArray(data.nodes) || !Array.isArray(data.edges)) throw new Error('Invalid graph file');
    state.nodes = data.nodes;
    state.edges = data.edges;
    updateGraph();
    setTimeout(fitToView, 600);
    showToast('Graph loaded!', 'success');
  } catch(e) {
    showToast('Failed to load: ' + e.message, 'error');
  }
}

function autoSave() {
  const data = { version: '1.0', created: new Date().toISOString(), nodes: state.nodes, edges: _serializeEdges() };
  localStorage.setItem(LS_GRAPH, JSON.stringify(data));
}

// ── UI Helpers ───────────────────────────────────────

function showLoading(on, text = 'Querying LLM…') {
  document.getElementById('loading-overlay').classList.toggle('visible', on);
  setLoadingText(text);
}
function setLoadingText(t) { document.getElementById('loading-text').textContent = t; }

let toastTimer;
function showToast(msg, type = '') {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className   = 'visible ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.className = '', 3200);
}

// ── Toolbar event wiring ─────────────────────────────

document.getElementById('generate-btn').addEventListener('click', () => {
  const topic = document.getElementById('topic-input').value.trim();
  if (!topic) { showToast('Please enter a topic', 'error'); return; }
  generateGraph(topic);
});

document.getElementById('topic-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('generate-btn').click();
});

document.getElementById('save-btn').addEventListener('click', saveGraph);
document.getElementById('load-btn').addEventListener('click', () => {
  document.getElementById('file-input').click();
});
document.getElementById('file-input').addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => loadGraph(ev.target.result);
  reader.readAsText(file);
  e.target.value = '';
});

document.getElementById('fit-btn').addEventListener('click', fitToView);

document.getElementById('dismiss-welcome').addEventListener('click', () => {
  document.getElementById('welcome-banner').style.display = 'none';
  localStorage.setItem('kg_welcome_dismissed', '1');
});

window.addEventListener('resize', () => {
  if (simulation) {
    const w = svg.node().clientWidth, h = svg.node().clientHeight;
    simulation.force('center', d3.forceCenter(w / 2, h / 2));
    simulation.alpha(0.1).restart();
  }
});

// ── Init ─────────────────────────────────────────────

(function init() {
  loadSettings();
  if (localStorage.getItem('kg_welcome_dismissed') === '1') {
    document.getElementById('welcome-banner').style.display = 'none';
  }

  const saved = localStorage.getItem(LS_GRAPH);
  if (saved) {
    try {
      const data = JSON.parse(saved);
      if (data.nodes && data.nodes.length > 1) {
        const banner = document.createElement('div');
        banner.style.cssText = 'position:fixed;top:70px;right:16px;z-index:600;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-size:12px;color:var(--text);box-shadow:0 4px 16px rgba(0,0,0,.4);display:flex;gap:10px;align-items:center';
        banner.innerHTML = `<span>Resume last session? (${data.nodes.length} nodes)</span><button id="resume-yes" style="background:var(--accent);color:#fff;border:none;border-radius:5px;padding:4px 10px;cursor:pointer;font-size:12px">Resume</button><button id="resume-no" style="background:transparent;color:var(--muted);border:none;cursor:pointer;font-size:12px">Dismiss</button>`;
        document.body.appendChild(banner);
        document.getElementById('resume-yes').onclick = () => { loadGraph(saved); banner.remove(); };
        document.getElementById('resume-no').onclick  = () => banner.remove();
      }
    } catch(e) {}
  }

  initGraph();
})();
