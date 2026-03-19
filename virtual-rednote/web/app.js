// ─── 全局上下文 ───────────────────────────────────────
const ctx = {
  worldId: null,
  communityName: null,
  dimensions: [],
  contentText: null,
  contentVector: null,
  simId: null,
  seedContentId: null,
};

// ─── Wizard 导航 ──────────────────────────────────────
// step 0: 选世界  step 1: 世界详情/历史  step 2: 输入内容  step 3: 模拟主界面
function goStep(n) {
  if (n !== 3 && es) { es.close(); es = null; }
  document.getElementById('wizard').style.display = n <= 2 ? 'flex' : 'none';
  document.getElementById('main').style.display   = n === 3 ? 'flex' : 'none';
  document.getElementById('step0').style.display  = n === 0 ? 'flex' : 'none';
  document.getElementById('step1').style.display  = n === 1 ? 'flex' : 'none';
  document.getElementById('step2').style.display  = n === 2 ? 'flex' : 'none';
  document.getElementById('btn-back').style.display    = n === 3 ? 'inline-block' : 'none';
  document.getElementById('btn-start').style.display   = n === 3 ? 'inline-block' : 'none';
  document.getElementById('btn-save').style.display    = 'none';
  document.getElementById('sim-day').style.display     = n === 3 ? 'inline' : 'none';
  document.getElementById('sim-status').style.display  = n === 3 ? 'inline' : 'none';
  if (n === 2) {
    document.getElementById('content-text').value = '';
    const statusEl = document.getElementById('step2-status');
    if (statusEl) statusEl.style.display = 'none';
    const btn = document.getElementById('btn-analyze');
    if (btn) { btn.disabled = false; btn.textContent = '开始模拟 →'; }
  }
  if (n === 3 && !ctx.isSnapshot) {
    document.getElementById('log-list').innerHTML  = '';
    document.getElementById('feed-list').innerHTML = '<div style="color:#444;font-size:12px;padding:16px;text-align:center" data-placeholder>模拟开始后显示评论</div>';
    document.getElementById('seed-post').style.display = 'none';
    document.getElementById('chart-wrap').style.display = '';
    document.getElementById('sim-status').textContent = '';
    document.getElementById('sim-day').textContent = '';
    document.getElementById('btn-start').style.display = 'inline-block';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').textContent = '开始模拟';
  }
  if (n === 3) requestAnimationFrame(initGraph);
  if (n === 0) loadWorldList();
  if (n === 1 && ctx.worldId) loadWorldDetail(ctx.worldId);
}

// ─── Step 0：世界列表 ─────────────────────────────────
async function loadWorldList() {
  const listEl = document.getElementById('world-list');
  listEl.innerHTML = '<div class="world-empty">加载中...</div>';
  try {
    const res  = await fetch('/worlds');
    const data = await res.json();
    if (!data.worlds.length) {
      listEl.innerHTML = '<div class="world-empty">暂无社区，请运行脚本创建</div>';
      return;
    }
    listEl.innerHTML = data.worlds.map(w => {
      const dimNames = w.dimensions.map(d => d.name).join(' · ');
      const date = w.created_at
        ? new Date(w.created_at).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })
        : '';
      return `
        <div class="world-item" onclick="selectWorld('${w.world_id}', this)">
          <div class="world-info">
            <div class="world-name">${w.community_name}</div>
            <div class="world-time">${date}</div>
            <div class="world-dims">${dimNames}</div>
          </div>
        </div>`;
    }).join('');
  } catch (e) {
    listEl.innerHTML = '<div class="world-empty" style="color:#ff4444">加载失败</div>';
  }
}

async function selectWorld(worldId, el) {
  document.querySelectorAll('.world-item').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
  try {
    const res  = await fetch(`/worlds/${worldId}/load`, { method: 'POST' });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    ctx.worldId       = data.world_id;
    ctx.communityName = data.community_name;
    ctx.dimensions    = data.dimensions;
    document.getElementById('topbar-community').textContent =
      `${data.community_name}（${data.n_agents} agents）`;
    goStep(1);
  } catch (e) {
    alert('加载失败：' + e.message);
  }
}

// ─── Step 1：世界详情 + 历史模拟 ──────────────────────
async function loadWorldDetail(worldId) {
  document.getElementById('detail-name').textContent = ctx.communityName || '';
  document.getElementById('detail-dims').textContent =
    (ctx.dimensions || []).map(d => d.name).join(' · ');

  const listEl = document.getElementById('sim-history-list');
  listEl.innerHTML = '<div style="color:#444;font-size:12px;padding:8px 0">加载中...</div>';
  try {
    const res  = await fetch(`/worlds/${worldId}/simulations`);
    const data = await res.json();
    if (!data.simulations.length) {
      listEl.innerHTML = '<div style="color:#444;font-size:12px;padding:8px 0">暂无历史模拟</div>';
      return;
    }
    listEl.innerHTML = data.simulations.map(s => {
      const date = new Date(s.created_at).toLocaleString('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' });
      const m    = s.metrics || {};
      const preview = s.content_text.replace(/\n/g, ' ').slice(0, 40);
      return `
        <div class="sim-history-item" onclick="loadSnapshot('${s.sim_id}')">
          <div class="sim-h-date">${date}</div>
          <div class="sim-h-text">${preview}${s.content_text.length > 40 ? '...' : ''}</div>
          <div class="sim-h-stats">
            <span>触达 <b>${m.reach||0}</b></span>
            <span>转发 <b>${m.reposts||0}</b></span>
            <span>评论 <b>${m.comments||0}</b></span>
          </div>
          <button class="sim-h-del" onclick="event.stopPropagation();deleteSimulation('${s.sim_id}',this)">删除</button>
        </div>`;
    }).join('');
  } catch(e) {
    listEl.innerHTML = '<div style="color:#ff4444;font-size:12px;padding:8px 0">加载失败</div>';
  }
}

async function deleteSimulation(simId, btn) {
  btn.disabled = true;
  btn.textContent = '...';
  try {
    const res = await fetch(`/simulate/${simId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(await res.text());
    btn.closest('.sim-history-item').remove();
    const list = document.getElementById('sim-history-list');
    if (!list.children.length) {
      list.innerHTML = '<div style="color:#444;font-size:12px;padding:8px 0">暂无历史模拟</div>';
    }
  } catch(e) {
    btn.disabled = false;
    btn.textContent = '删除';
    alert('删除失败：' + e.message);
  }
}

async function loadSnapshot(simId) {
  try {
    const res  = await fetch(`/simulate/${simId}`);
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    ctx.simId       = simId;
    ctx.contentText = data.content_text;
    ctx.isSnapshot  = true;

    goStep(3);

    // 等 graph 初始化后渲染快照
    requestAnimationFrame(() => {
      initGraph();
      if (data.graph_init) {
        renderGraphInit(data.graph_init.nodes, data.graph_init.edges);
      }
      showSeedPost();
      updateMetrics(data.metrics);

      // 渲染评论 + 品牌回复
      const feedList = document.getElementById('feed-list');
      feedList.innerHTML = '';
      // 先建 agent_id -> 评论卡映射，方便品牌回复插入
      const commentCardMap = {};
      (data.comments || []).forEach(c => {
        const card = document.createElement('div');
        card.className = 'comment-card';
        card.dataset.agentId = c.agent_id;
        card.innerHTML = `
          <div class="author"><span class="tier-normal">${c.agent_id}</span></div>
          <div class="body comment-body">${c.text}</div>`;
        feedList.appendChild(card);
        commentCardMap[c.agent_id] = card;
      });
      // 品牌回复紧跟在对应评论后面
      (data.brand_replies || []).forEach(r => {
        const reply = document.createElement('div');
        reply.className = 'comment-card brand-reply-card';
        reply.innerHTML = `
          <div class="author"><span class="tier-brand">品牌方</span><span style="color:#555;font-size:10px"> 回复 ${r.agent_id}</span></div>
          <div class="body comment-body">${r.reply}</div>`;
        const target = commentCardMap[r.agent_id];
        if (target) target.after(reply);
        else feedList.appendChild(reply);
      });

      // 渲染日志
      const logList = document.getElementById('log-list');
      logList.innerHTML = '';
      (data.event_log || []).forEach(e => appendLog(e));

      document.getElementById('sim-status').textContent = '历史快照';
      document.getElementById('sim-day').textContent = '';
      document.getElementById('btn-start').style.display = 'none';
      document.getElementById('chart-wrap').style.display = 'none';
    });
  } catch(e) {
    alert('加载快照失败：' + e.message);
  }
}

// ─── Step 2：分析内容 ─────────────────────────────────
async function analyzeContent() {
  const text = document.getElementById('content-text').value.trim();
  if (!text) return;

  const btn      = document.getElementById('btn-analyze');
  const statusEl = document.getElementById('step2-status');

  btn.disabled    = true;
  btn.textContent = '分析中...';
  statusEl.style.display = 'flex';
  statusEl.innerHTML = '<div class="spinner"></div> 正在分析内容…';

  try {
    const res = await fetch('/worlds/analyze-content', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ world_id: ctx.worldId, content_text: text }),
    });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    ctx.contentText   = text;
    ctx.contentVector = data.vector;
    ctx.isSnapshot    = false;
    ctx.simId         = null;

    goStep(3);
    // 自动启动模拟
    launchSim();
  } catch (e) {
    statusEl.innerHTML = `<span class="status-err">✗ ${e.message}</span>`;
    btn.disabled    = false;
    btn.textContent = '开始模拟 →';
  }
}

// ─── Step 3：启动模拟 ─────────────────────────────────
async function launchSim() {
  const btn = document.getElementById('btn-start');
  btn.disabled = true;
  btn.textContent = '模拟中...';

  const res = await fetch('/simulate/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      world_id:       ctx.worldId,
      content_text:   ctx.contentText,
      content_vector: ctx.contentVector,
    }),
  });
  if (!res.ok) { alert('启动失败：' + await res.text()); return; }
  const data = await res.json();
  ctx.simId         = data.sim_id;
  ctx.seedContentId = data.seed_content_id;
  startSim(data.sim_id);
}

// ─── Chart.js ────────────────────────────────────────
const chartCtx   = document.getElementById('reach-chart').getContext('2d');
const reachChart = new Chart(chartCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: '触达', data: [], borderColor: '#ff2d55', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true, backgroundColor: 'rgba(255,45,85,0.08)' },
      { label: '转发', data: [], borderColor: '#ff9500', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
      { label: '评论', data: [], borderColor: '#5ac8fa', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
    ],
  },
  options: {
    animation: false, responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#666', font: { size: 10 }, boxWidth: 10 } } },
    scales: {
      x: { display: false },
      y: { ticks: { color: '#555', font: { size: 9 } }, grid: { color: '#1e1e1e' } },
    },
  },
});

function pushChart(reach, reposts, comments) {
  const MAX = 60;
  const d   = reachChart.data;
  d.labels.push('');
  d.datasets[0].data.push(reach);
  d.datasets[1].data.push(reposts);
  d.datasets[2].data.push(comments);
  if (d.labels.length > MAX) { d.labels.shift(); d.datasets.forEach(ds => ds.data.shift()); }
  reachChart.update('none');
}

// ─── D3 传播图 ───────────────────────────────────────
const svg = d3.select('#graph-svg');
let width = 0, height = 0;

const tierColor  = { kol: '#ff9500', koc: '#5ac8fa', normal: '#aaa', brand: '#ff2d55' };
const tierRadius = { kol: 9, koc: 6, brand: 12, normal: 3 };

// 高亮色：按 action 类型
const actionColor = { repost: '#ff9500', comment: '#5ac8fa', like: '#ff2d55' };

const gLinks = svg.append('g');
const gNodes = svg.append('g');

const simulation = d3.forceSimulation()
  .force('link',    d3.forceLink().id(d => d.id).distance(80).strength(0.3))
  .force('charge',  d3.forceManyBody().strength(d => d.tier === 'kol' ? -600 : d.tier === 'koc' ? -300 : -120))
  .force('collide', d3.forceCollide(d => (tierRadius[d.tier] || 3) + 6))
  .force('center',  d3.forceCenter())   // 保持整体质心在中心，防止漂移
  .alphaDecay(0.015)
  .velocityDecay(0.3)
  .on('tick', onTick);

const graphState = { nodeIndex: {}, linkIndex: {}, personaIndex: {} };
let es = null;

function initGraph() {
  const el = document.getElementById('panel-graph');
  width  = el.clientWidth;
  height = el.clientHeight - 30;
  if (!width || !height) return;
  svg.attr('viewBox', `0 0 ${width} ${height}`);
  simulation
    .force('center', d3.forceCenter(width / 2, height / 2))
    .force('x', null)
    .force('y', null);
}

// 一次性渲染所有节点和边（graph_init 事件）
function renderGraphInit(nodes, edges) {
  // 品牌节点固定在中心
  const nodeData = nodes.map(n => {
    const node = { ...n, x: width / 2 + (Math.random() - 0.5) * width * 0.8, y: height / 2 + (Math.random() - 0.5) * height * 0.8 };
    if (n.tier === 'brand') { node.fx = width / 2; node.fy = height / 2; }
    graphState.nodeIndex[n.id]   = node;
    graphState.personaIndex[n.id] = { tier: n.tier, persona: n.persona || '' };
    return node;
  });

  const edgeData = edges.map(e => {
    const key = `${e.source}__${e.target}`;
    graphState.linkIndex[key] = true;
    return { ...e };
  });

  gLinks.selectAll('line').data(edgeData, d => `${d.source}__${d.target}`)
    .enter().append('line')
      .attr('class', 'graph-edge')
      .attr('id', d => `edge-${d.source}__${d.target}`)
      .attr('stroke', '#555')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8);

  const tooltip = document.getElementById('agent-tooltip');

  gNodes.selectAll('circle').data(nodeData, d => d.id)
    .enter().append('circle')
      .attr('r',    d => tierRadius[d.tier] || 3)
      .attr('fill', d => tierColor[d.tier]  || '#aaa')
      .attr('opacity', 1.0)
      ;

  simulation.nodes(nodeData);
  simulation.force('link').links(edgeData);
  simulation.alpha(1).restart();
}

// 高亮某条边 + 目标节点，闪亮后淡回
function highlightEdge(sourceId, targetId, action) {
  const color   = actionColor[action] || '#fff';
  const key     = `${sourceId}__${targetId}`;
  const FADE_MS = 2000;

  const edgeEl = svg.select(`#edge-${key}`);
  edgeEl
    .raise()
    .attr('stroke', color)
    .attr('stroke-width', 5)
    .attr('opacity', 1)
    .transition().duration(FADE_MS).ease(d3.easeCubicOut)
      .attr('stroke', '#555')
      .attr('stroke-width', 1)
      .attr('opacity', 0.8);

  gNodes.selectAll('circle')
    .filter(d => d.id === targetId)
    .raise()
    .attr('fill', color)
    .attr('r',    d => (tierRadius[d.tier] || 3) * 2.5)
    .attr('opacity', 1)
    .transition().duration(FADE_MS).ease(d3.easeCubicOut)
      .attr('fill', d => tierColor[d.tier] || '#aaa')
      .attr('r',    d => tierRadius[d.tier] || 3)
      .attr('opacity', 1.0);
}

function onTick() {
  gLinks.selectAll('line')
    .attr('x1', d => clamp(d.source.x, 10, width  - 10))
    .attr('y1', d => clamp(d.source.y, 10, height - 10))
    .attr('x2', d => clamp(d.target.x, 10, width  - 10))
    .attr('y2', d => clamp(d.target.y, 10, height - 10));
  gNodes.selectAll('circle')
    .attr('cx', d => clamp(d.x, 10, width  - 10))
    .attr('cy', d => clamp(d.y, 10, height - 10));
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

// ─── Agent Tooltip ───────────────────────────────────
const tooltip = document.getElementById('agent-tooltip');
const TIER_LABEL_MAP = { kol: 'KOL', koc: 'KOC', normal: '普通用户', brand: '品牌' };

function showTooltip(event, agentId) {
  const info = graphState.personaIndex[agentId];
  if (!info) return;
  const tierLabel = TIER_LABEL_MAP[info.tier] || info.tier;
  tooltip.innerHTML = `
    <div class="tooltip-tier ${info.tier}">${tierLabel}</div>
    <div class="tooltip-id">${agentId}</div>
    ${info.persona ? `<div class="tooltip-persona">${info.persona}</div>` : ''}`;
  tooltip.style.display = 'block';
  moveTooltip(event);
}

function moveTooltip(event) {
  const x = event.clientX + 14;
  const y = event.clientY - 10;
  tooltip.style.left = Math.min(x, window.innerWidth - 240) + 'px';
  tooltip.style.top  = y + 'px';
}

function hideTooltip() {
  tooltip.style.display = 'none';
}

function makeAgentSpan(agentId, tierClass) {
  return `<span class="agent-link ${tierClass}"
    onmouseenter="showTooltip(event,'${agentId}')"
    onmousemove="moveTooltip(event)"
    onmouseleave="hideTooltip()">${agentId}</span>`;
}

// ─── 日志 ────────────────────────────────────────────
const ACTION_ICON = { like: '❤', comment: '💬', repost: '🔁' };
const LOG_MAX     = 200;

function appendLog(data) {
  const list = document.getElementById('log-list');
  const el   = document.createElement('div');
  const tier = data.agent_tier === 'kol' ? '🟠' : data.agent_tier === 'koc' ? '🔵' : '·';
  el.className = `log-item ${data.action}`;
  el.innerHTML = `<span class="t">Day${data.sim_time.toFixed(2)}</span>${tier} ${makeAgentSpan(data.agent_id, `tier-${data.agent_tier}`)} ${ACTION_ICON[data.action] || ''} ${data.action}`;
  list.appendChild(el);
  if (list.children.length > LOG_MAX) list.removeChild(list.firstChild);
  list.scrollTop = list.scrollHeight;
}

function appendBrandLog(data) {
  const list = document.getElementById('log-list');
  const el   = document.createElement('div');
  el.className = 'log-item brand-reply';
  el.innerHTML = `<span class="t">Day${data.sim_time.toFixed(2)}</span><span style="color:#ff2d55">品牌方</span> 💬 回复 ${makeAgentSpan(data.agent_id, 'tier-normal')}`;
  list.appendChild(el);
  if (list.children.length > LOG_MAX) list.removeChild(list.firstChild);
  list.scrollTop = list.scrollHeight;
}

// ─── 评论流 ──────────────────────────────────────────
const COMMENT_MAX = 200;

function appendCommentPlaceholder(data) {
  const list = document.getElementById('feed-list');
  const placeholder = list.querySelector('[data-placeholder]');
  if (placeholder) placeholder.remove();

  const tierClass = `tier-${data.agent_tier || 'normal'}`;
  const tierLabel = data.agent_tier === 'kol' ? ' [KOL]' : data.agent_tier === 'koc' ? ' [KOC]' : '';
  const card = document.createElement('div');
  card.className = 'comment-card';
  card.dataset.agentId = data.agent_id;
  card.innerHTML = `
    <div class="author">${makeAgentSpan(data.agent_id, tierClass)}${tierLabel ? `<span style="color:#555;font-size:10px"> ${tierLabel.trim()}</span>` : ''}</div>
    <div class="body comment-body" style="color:#444;font-style:italic">生成中...</div>`;
  list.appendChild(card);
  if (list.children.length > COMMENT_MAX) list.removeChild(list.firstChild);
  list.scrollTop = list.scrollHeight;
}

function fillComment(data) {
  const list = document.getElementById('feed-list');
  // 找到对应 agent 的最后一张占位卡（从后往前找，避免同一 agent 多条评论冲突）
  const cards = [...list.querySelectorAll(`.comment-card[data-agent-id="${data.agent_id}"]`)];
  const card = cards.reverse().find(c => c.querySelector('.comment-body')?.textContent === '生成中...');
  if (card) {
    const body = card.querySelector('.comment-body');
    body.textContent = data.text;
    body.style.color = '';
    body.style.fontStyle = '';
  }
  list.scrollTop = list.scrollHeight;
}

function appendBrandReply(data) {
  ctx.brandReplies = ctx.brandReplies || [];
  ctx.brandReplies.push(data);

  const list = document.getElementById('feed-list');
  // 找到被回复的 agent 评论卡，把品牌回复插在其后
  const cards = [...list.querySelectorAll(`.comment-card[data-agent-id="${data.agent_id}"]`)];
  const targetCard = cards[cards.length - 1];

  const reply = document.createElement('div');
  reply.className = 'comment-card brand-reply-card';
  reply.dataset.brandReply = data.agent_id;
  reply.innerHTML = `
    <div class="author"><span class="tier-brand">品牌方</span><span style="color:#555;font-size:10px"> 回复 ${data.agent_id}</span></div>
    <div class="body comment-body">${data.reply}</div>`;

  if (targetCard) {
    targetCard.after(reply);
  } else {
    list.appendChild(reply);
  }
  list.scrollTop = list.scrollHeight;
}

// ─── 指标 ────────────────────────────────────────────
function updateMetrics(m) {
  document.getElementById('m-reach').textContent       = m.reach;
  document.getElementById('m-reposts').textContent     = m.reposts;
  document.getElementById('m-repost-rate').textContent =
    ((m.reposts  / Math.max(m.reach, 1)) * 100).toFixed(1) + '%';
  document.getElementById('m-comment-rate').textContent =
    ((m.comments / Math.max(m.reach, 1)) * 100).toFixed(1) + '%';
  pushChart(m.reach, m.reposts, m.comments);
}

// ─── 原帖 ────────────────────────────────────────────
let seedStats = { likes: 0, comments: 0, reposts: 0 };

function showSeedPost() {
  document.getElementById('seed-post').style.display = 'block';
  document.getElementById('seed-text').textContent   = ctx.contentText || '';
  updateSeedStats();
}

function updateSeedStats() {
  document.getElementById('seed-stats').innerHTML =
    `<span>❤ ${seedStats.likes}</span><span>💬 ${seedStats.comments}</span><span>🔁 ${seedStats.reposts}</span>`;
}

// ─── SSE ─────────────────────────────────────────────
function setRunning(running) {
  document.getElementById('btn-stop').style.display  = running ? 'inline-block' : 'none';
  document.getElementById('btn-start').textContent   = running ? '重新开始' : '开始模拟';
  document.getElementById('sim-status').textContent  = running ? '模拟运行中...' : '已停止';
}

function stopSim() {
  if (es) { es.close(); es = null; }
  setRunning(false);
}

function startSim(simId) {
  if (es) { es.close(); es = null; }

  // 重置
  graphState.nodeIndex = {}; graphState.linkIndex = {};
  gLinks.selectAll('*').remove(); gNodes.selectAll('*').remove();
  simulation.stop();
  document.getElementById('log-list').innerHTML  = '';
  document.getElementById('feed-list').innerHTML = '<div style="color:#444;font-size:12px;padding:16px;text-align:center" data-placeholder>模拟开始后显示评论</div>';
  document.getElementById('seed-post').style.display = 'none';
  document.getElementById('btn-save').style.display = 'none';
  document.getElementById('chart-wrap').style.display = '';
  reachChart.data.labels = [];
  reachChart.data.datasets.forEach(d => d.data = []);
  reachChart.update('none');
  seedStats = { likes: 0, comments: 0, reposts: 0 };
  ctx.eventLog = [];      // 收集事件流用于保存
  ctx.brandReplies = [];  // 收集品牌回复用于保存

  initGraph();
  showSeedPost();
  setRunning(true);

  es = new EventSource(`/simulate/${simId}/stream`);

  es.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === 'graph_init') {
      ctx.graphInit = { nodes: data.nodes, edges: data.edges };
      renderGraphInit(data.nodes, data.edges);
    }

    if (data.type === 'agent_reacted') {
      ctx.eventLog.push(data);
      document.getElementById('sim-day').textContent = `Day ${data.sim_time.toFixed(2)}`;

      if (data.highlight_edge) {
        highlightEdge(data.highlight_edge.source, data.highlight_edge.target, data.action);
      }

      if (data.seed_stats) { seedStats = data.seed_stats; updateSeedStats(); }
      updateMetrics(data.metrics);
      appendLog(data);

      if (data.action === 'comment') appendCommentPlaceholder(data);
    }

    if (data.type === 'comment_generated') fillComment(data);
    if (data.type === 'brand_reply') { appendBrandReply(data); appendBrandLog(data); }
    if (data.type === 'metrics_update') updateMetrics(data.metrics);

    if (data.type === 'simulation_done') {
      ctx.doneMetrics = data.metrics;
      ctx.doneEvents  = data.total_events;
      setRunning(false);
      const btn = document.getElementById('btn-start');
      btn.disabled = false;
      btn.textContent = '重新开始';
      document.getElementById('btn-save').style.display = 'inline-block';
      es.close(); es = null;
    }
  };

  es.onerror = () => {
    document.getElementById('sim-status').textContent = '连接中断';
    document.getElementById('btn-stop').style.display = 'none';
  };
}

// ─── 保存模拟 ─────────────────────────────────────────
async function saveSim() {
  const btn = document.getElementById('btn-save');
  btn.disabled = true;
  btn.textContent = '保存中...';
  // 收集评论区已生成的评论文本
  const comments = [...document.querySelectorAll('.comment-card')].map(c => ({
    agent_id:   c.dataset.agentId,
    text:       c.querySelector('.comment-body')?.textContent || '',
  })).filter(c => c.text && c.text !== '生成中...');
  try {
    const res = await fetch(`/simulate/${ctx.simId}/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content_text:  ctx.contentText,
        metrics:       ctx.doneMetrics,
        total_events:  ctx.doneEvents,
        event_log:     ctx.eventLog,
        comments,
        brand_replies: ctx.brandReplies || [],
      }),
    });
    if (!res.ok) throw new Error(await res.text());
    btn.textContent = '已保存 ✓';
  } catch (e) {
    alert('保存失败：' + e.message);
    btn.disabled = false;
    btn.textContent = '保存模拟';
  }
}

// ─── 初始化 ──────────────────────────────────────────
window.addEventListener('load', () => {
  window.addEventListener('resize', initGraph);
  goStep(0);
});
