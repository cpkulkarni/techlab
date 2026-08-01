// ════════════════════════════════════════════════════
// kg-settings.js — LLM provider settings & Ollama connect
// ════════════════════════════════════════════════════

const LS_SETTINGS = 'kg_settings';

function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(LS_SETTINGS) || '{}');
    Object.assign(state.settings, s);
  } catch(e) {}
  applySettingsToForm();
}

function saveSettings() {
  const p = document.getElementById('set-provider').value;
  state.settings.provider     = p;
  state.settings.ollamaUrl    = safeOllamaUrl(document.getElementById('set-ollama-url').value.trim());
  state.settings.model        = getModelFromForm() || defaultModel(p);
  state.settings.apiKey       = document.getElementById('set-apikey').value.trim();
  state.settings.searchEngine = document.getElementById('set-search-engine').value;
  // Fix #1: never persist the API key to localStorage
  const { apiKey: _drop, ...toSave } = state.settings;
  localStorage.setItem(LS_SETTINGS, JSON.stringify(toSave));
  closeSettings();
  showToast('Settings saved', 'success');
}

function defaultModel(p) {
  return p === 'openai' ? 'gpt-4o-mini' : p === 'gemini' ? 'gemini-1.5-flash' : 'llama3';
}

function getModelFromForm() {
  const p = document.getElementById('set-provider').value;
  if (p === 'gemini') return document.getElementById('set-gemini-model').value;
  if (p === 'ollama') return document.getElementById('set-ollama-model').value || '';
  return document.getElementById('set-model').value.trim();
}

function applySettingsToForm() {
  const p = state.settings.provider;
  document.getElementById('set-provider').value      = p;
  document.getElementById('set-ollama-url').value    = state.settings.ollamaUrl;
  document.getElementById('set-model').value         = p === 'openai' ? state.settings.model : '';
  if (p === 'gemini' && state.settings.model) {
    const sel = document.getElementById('set-gemini-model');
    const opt = Array.from(sel.options).find(o => o.value === state.settings.model);
    if (opt) sel.value = state.settings.model;
  }
  // Fix #1: apiKey is never stored, so the field always starts empty — user re-enters each session
  document.getElementById('set-apikey').value        = '';
  document.getElementById('set-search-engine').value = state.settings.searchEngine;
  toggleProviderFields(p);
  if (p === 'ollama' && state.settings.ollamaUrl) {
    loadOllamaModels(state.settings.ollamaUrl, state.settings.model);
  }
}

function toggleProviderFields(p) {
  document.getElementById('set-group-ollama-url').style.display   = p === 'ollama' ? '' : 'none';
  document.getElementById('set-group-ollama-model').style.display = p === 'ollama' ? '' : 'none';
  document.getElementById('set-group-apikey').style.display        = p !== 'ollama' ? '' : 'none';
  document.getElementById('set-group-gemini-model').style.display  = p === 'gemini' ? '' : 'none';
  document.getElementById('set-group-model').style.display         = p === 'openai' ? '' : 'none';
}

// Fix #3: only allow http/https schemes for the Ollama base URL
function safeOllamaUrl(raw) {
  try {
    const u = new URL(raw || 'http://localhost:11434');
    if (u.protocol !== 'http:' && u.protocol !== 'https:') throw new Error();
    return u.origin;
  } catch {
    return 'http://localhost:11434';
  }
}

async function loadOllamaModels(baseUrl, selectValue) {
  const status = document.getElementById('ollama-connect-status');
  const sel    = document.getElementById('set-ollama-model');
  const grp    = document.getElementById('set-group-ollama-model');
  // Fix #3: validate URL before fetching
  const url    = safeOllamaUrl(baseUrl);
  status.style.color = 'var(--muted)';
  status.textContent = '⏳ Connecting…';
  try {
    const res = await fetch(`${url}/api/tags`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    const models = (data.models || []).map(m => m.name).sort();
    if (models.length === 0) throw new Error('No models found — pull a model first (e.g. ollama pull llama3)');
    // Fix #2: use DOM APIs instead of innerHTML to avoid XSS via model names
    sel.replaceChildren(...models.map(m => {
      const opt = document.createElement('option');
      opt.value       = m;
      opt.textContent = m;
      opt.selected    = m === selectValue;
      return opt;
    }));
    grp.style.display = '';
    status.style.color = 'var(--success)';
    status.textContent = `✓ Connected — ${models.length} model${models.length > 1 ? 's' : ''} available`;
  } catch(e) {
    grp.style.display = 'none';
    status.style.color = 'var(--danger)';
    const isNetwork = e instanceof TypeError && /fetch|network/i.test(e.message);
    status.textContent = isNetwork
      ? '✗ Cannot reach Ollama. Enable CORS by restarting Ollama with: OLLAMA_ORIGINS=* ollama serve'
      : `✗ ${e.message}`;
  }
}

function openSettings()  { document.getElementById('settings-backdrop').classList.add('visible'); }
function closeSettings() { document.getElementById('settings-backdrop').classList.remove('visible'); }

// ── Settings event wiring ──
document.getElementById('settings-btn').addEventListener('click', openSettings);
document.getElementById('settings-save-btn').addEventListener('click', saveSettings);
document.getElementById('settings-cancel-btn').addEventListener('click', closeSettings);
document.getElementById('settings-backdrop').addEventListener('click', e => {
  if (e.target === document.getElementById('settings-backdrop')) closeSettings();
});

document.getElementById('set-provider').addEventListener('change', e => {
  toggleProviderFields(e.target.value);
  if (e.target.value === 'ollama') {
    const url = document.getElementById('set-ollama-url').value.trim() || 'http://localhost:11434';
    loadOllamaModels(url, state.settings.model);
  }
});

document.getElementById('ollama-connect-btn').addEventListener('click', () => {
  // safeOllamaUrl is applied inside loadOllamaModels
  const url = document.getElementById('set-ollama-url').value.trim() || 'http://localhost:11434';
  loadOllamaModels(url, document.getElementById('set-ollama-model').value);
});

document.getElementById('set-ollama-url').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('ollama-connect-btn').click();
});
