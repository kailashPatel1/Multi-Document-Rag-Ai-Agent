// Multi-Doc RAG AI - Intelligent Knowledge Assistant Frontend Controller
// Ensure API_BASE always resolves to backend API correctly
const API_BASE = (window.location.protocol === 'file:' || !window.location.port || (window.location.port !== '8000' && !window.location.hostname.includes('localhost'))) 
  ? (window.location.port === '8000' ? '' : 'http://localhost:8000') 
  : '';

const state = {
  currentTab: 'dashboard',
  systemStatus: null,
  documents: [],
  sessions: [],
  currentSessionId: null,
  chatMessages: [],
  chatMode: 'agent',
  selectedDocument: null, // Scoped doc: { id: string, name: string }
  evalResults: null,
  isSending: false,
  isEvaluating: false,
  theme: 'dark'
};

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  setupNavigation();
  setupChatHandlers();
  setupUploadHandlers();
  setupComparisonHandlers();
  setupEvaluationHandlers();
  setupSettingsHandlers();
  setupThemeHandlers();

  await refreshAll();
  if (state.sessions.length > 0) {
    selectSession(state.sessions[0].id);
  } else {
    createNewSession();
  }
});

// --- Theme Management ---
function initTheme() {
  const savedTheme = localStorage.getItem('app_theme') || 'dark';
  setAppTheme(savedTheme, false);

  if (savedTheme === 'system') {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
      if (state.theme === 'system') {
        applyThemeClass(e.matches ? 'dark' : 'light');
      }
    });
  }
}

function setAppTheme(theme, showNotice = true) {
  state.theme = theme;
  localStorage.setItem('app_theme', theme);

  let effectiveTheme = theme;
  if (theme === 'system') {
    effectiveTheme = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }

  applyThemeClass(effectiveTheme);
  updateThemeUI(theme);

  const menu = document.getElementById('theme-dropdown-menu');
  if (menu) menu.style.display = 'none';

  const settingSelect = document.getElementById('setting-app-theme');
  if (settingSelect && settingSelect.value !== theme) {
    settingSelect.value = theme;
  }

  if (showNotice) {
    const label = theme.charAt(0).toUpperCase() + theme.slice(1);
    showToast(`Theme switched to ${label} Mode`, 'info');
  }
}

function applyThemeClass(theme) {
  document.documentElement.setAttribute('data-theme', theme);
}

function updateThemeUI(theme) {
  const icon = document.getElementById('theme-icon');
  const label = document.getElementById('theme-label');
  
  if (icon && label) {
    if (theme === 'dark') {
      icon.innerText = '\uD83C\uDF19';
      label.innerText = 'Dark';
    } else if (theme === 'light') {
      icon.innerText = '\u2600\uFE0F';
      label.innerText = 'Light';
    } else {
      icon.innerText = '\uD83D\uDCBB';
      label.innerText = 'System';
    }
  }

  // Update active state in menu
  document.querySelectorAll('.theme-option').forEach(opt => {
    opt.classList.remove('active');
    if (opt.innerText.toLowerCase().includes(theme)) {
      opt.classList.add('active');
    }
  });
}

function toggleThemeDropdown() {
  const menu = document.getElementById('theme-dropdown-menu');
  if (!menu) return;
  menu.style.display = menu.style.display === 'none' ? 'flex' : 'none';
}

function setupThemeHandlers() {
  document.addEventListener('click', (e) => {
    const dropdown = document.querySelector('.theme-dropdown-container');
    const menu = document.getElementById('theme-dropdown-menu');
    if (dropdown && menu && !dropdown.contains(e.target)) {
      menu.style.display = 'none';
    }
  });
}

// --- Navigation & Core Sync ---
async function refreshAll() {
  await Promise.all([
    fetchSystemStatus(),
    fetchDocuments(),
    fetchSessions()
  ]);
  renderDashboard();
  renderKnowledgeBase();
}

function setupNavigation() {
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      switchTab(tab);
    });
  });
}

function switchTab(tab) {
  state.currentTab = tab;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-view').forEach(v => v.style.display = 'none');
  
  const targetView = document.getElementById(`view-${tab}`);
  if (targetView) targetView.style.display = 'block';

  const titleMap = {
    dashboard: 'Knowledge Dashboard',
    chat: 'Multi-Doc AI Chat',
    comparison: 'Multi-Document Comparison',
    upload: 'Upload Documents',
    knowledge_base: 'Knowledge Base',
    evaluation: 'RAG Evaluation',
    settings: 'Settings & Configuration'
  };
  const pageTitle = document.getElementById('page-title');
  if (pageTitle) pageTitle.innerText = titleMap[tab] || 'Knowledge Assistant';

  if (tab === 'dashboard') renderDashboard();
  if (tab === 'chat') renderDocumentBanner();
  if (tab === 'knowledge_base') renderKnowledgeBase();
  if (tab === 'comparison') renderComparisonSelectors();
  if (tab === 'evaluation') fetchEvaluationResults();
  if (tab === 'settings') populateSettings();
}

async function fetchSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/system/status`);
    if (res.ok) {
      state.systemStatus = await res.json();
      updateTopBarStatus();
      if (state.currentTab === 'settings') {
        populateSettings();
      }
    }
  } catch (e) {
    console.error('Failed to fetch status:', e);
  }
}

async function fetchDocuments() {
  try {
    const res = await fetch(`${API_BASE}/api/documents`);
    if (res.ok) {
      state.documents = await res.json();
    }
  } catch (e) {
    console.error('Failed to fetch documents:', e);
  }
}

async function fetchSessions() {
  try {
    const res = await fetch(`${API_BASE}/api/chat/sessions`);
    if (res.ok) {
      state.sessions = await res.json();
      renderSessionsList();
    }
  } catch (e) {
    console.error('Failed to fetch sessions:', e);
  }
}

function updateTopBarStatus() {
  const statusElem = document.getElementById('llm-status-indicator');
  const chatBadge = document.getElementById('chat-llm-badge');

  if (state.systemStatus) {
    const s = state.systemStatus;
    let providerLabel = s.llm_provider.toUpperCase();
    if (s.llm_provider === 'groq') {
      providerLabel = s.has_groq_key ? 'Groq (Llama 3.3 70B)' : 'Groq (Key Required)';
    } else if (s.llm_provider === 'gemini') {
      providerLabel = s.has_gemini_key ? 'Gemini (1.5 Flash)' : 'Gemini (Key Required)';
    } else if (s.llm_provider === 'mock') {
      providerLabel = 'Local Extractive';
    }

    if (statusElem) {
      statusElem.innerHTML = `
        <span class="pulse-dot"></span>
        <span>${providerLabel}</span>
        <span style="color: var(--text-dim);">|</span>
        <span>${s.total_documents} Docs</span>
      `;
    }

    if (chatBadge) {
      if (state.selectedDocument) {
        chatBadge.innerHTML = `<span style="color: var(--accent-cyan);">\uD83D\uDCC4 Document Chat: ${escapeHtml(state.selectedDocument.name)}</span>`;
      } else {
        chatBadge.innerHTML = `<span>Active: ${providerLabel}</span>`;
      }
    }
  }
}

// --- Dashboard View ---
function renderDashboard() {
  const s = state.systemStatus || { total_documents: 0, total_chunks: 0, vector_chunks_indexed: 0, uptime_seconds: 0 };
  
  const dDocs = document.getElementById('dash-stat-docs');
  const dChunks = document.getElementById('dash-stat-chunks');
  const dVectors = document.getElementById('dash-stat-vectors');
  const dUptime = document.getElementById('dash-stat-uptime');

  if (dDocs) dDocs.innerText = s.total_documents;
  if (dChunks) dChunks.innerText = s.total_chunks;
  if (dVectors) dVectors.innerText = s.vector_chunks_indexed;
  if (dUptime) dUptime.innerText = `${Math.floor(s.uptime_seconds / 60)}m ${Math.floor(s.uptime_seconds % 60)}s`;

  const docList = document.getElementById('dash-recent-docs');
  if (docList) {
    if (state.documents.length === 0) {
      docList.innerHTML = `
        <div style="text-align: center; padding: 24px; color: var(--text-muted);">
          <p>No documents in knowledge base.</p>
          <button class="btn btn-primary btn-sm" style="margin-top: 10px;" onclick="switchTab('upload')">
            \uD83D\uDCE4 Upload Documents
          </button>
        </div>
      `;
    } else {
      docList.innerHTML = state.documents.slice(0, 5).map(d => `
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 0; border-bottom: 1px solid var(--border-color);">
          <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 20px;">\uD83D\uDCC4</span>
            <div>
              <div style="font-weight: 600; font-size: 13.5px; color: var(--text-main);">${escapeHtml(d.original_name)}</div>
              <div style="font-size: 12px; color: var(--text-dim);">${d.file_type.toUpperCase()} \u2022 ${d.total_pages} page(s) \u2022 ${d.total_chunks} chunks</div>
            </div>
          </div>
          <div style="display: flex; gap: 8px;">
            <button class="btn btn-primary btn-sm" onclick="startChatWithDoc('${d.id}', '${escapeHtml(d.original_name)}')">
              Ask Questions
            </button>
            <button class="btn btn-secondary btn-sm" onclick="viewDocumentChunks('${d.id}', '${escapeHtml(d.original_name)}')">
              Chunks
            </button>
          </div>
        </div>
      `).join('');
    }
  }
}

function launchRAGQuery(query) {
  state.selectedDocument = null;
  switchTab('chat');
  renderDocumentBanner();
  const input = document.getElementById('chat-input');
  if (input) {
    input.value = query;
    sendChatMessage();
  }
}

function startChatWithDoc(docId, docName) {
  state.selectedDocument = { id: docId, name: docName };
  switchTab('chat');
  renderDocumentBanner();
  createNewSessionForDoc(docName);
  
  const input = document.getElementById('chat-input');
  if (input) {
    input.placeholder = `Ask questions specifically about ${docName}...`;
    input.focus();
  }
  showToast(`Document Chat activated for '${docName}'`);
}

function renderDocumentBanner() {
  const banner = document.getElementById('doc-chat-banner');
  const bannerName = document.getElementById('doc-banner-name');
  const chatInput = document.getElementById('chat-input');
  const chatBadge = document.getElementById('chat-llm-badge');

  if (state.selectedDocument) {
    if (banner) banner.style.display = 'flex';
    if (bannerName) bannerName.innerText = state.selectedDocument.name;
    if (chatInput) chatInput.placeholder = `Ask questions specifically about ${state.selectedDocument.name}...`;
    if (chatBadge) {
      chatBadge.innerHTML = `<span style="color: var(--accent-cyan);">\uD83D\uDCC4 Scoped: ${escapeHtml(state.selectedDocument.name)}</span>`;
    }
  } else {
    if (banner) banner.style.display = 'none';
    if (chatInput) chatInput.placeholder = "Ask questions across uploaded documents (e.g. 'Summarize key findings', 'What is RAG?')...";
    updateTopBarStatus();
  }
}

function clearDocumentSelection() {
  state.selectedDocument = null;
  renderDocumentBanner();
  showToast('Switched to Multi-Document Chat mode', 'info');
}

async function createNewSessionForDoc(docName) {
  const newId = `session_${Date.now()}`;
  state.currentSessionId = newId;
  state.chatMessages = [];
  state.sessions.unshift({
    id: newId,
    title: `Chat: ${docName}`,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    message_count: 0
  });
  renderSessionsList();
  renderChatMessages();
}

// --- Chat View ---
function setupChatHandlers() {
  const sendBtn = document.getElementById('btn-send-chat');
  if (sendBtn) sendBtn.addEventListener('click', sendChatMessage);

  const chatInput = document.getElementById('chat-input');
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });
  }

  document.querySelectorAll('.mode-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.chatMode = btn.dataset.mode;
    });
  });

  const newSessBtn = document.getElementById('btn-new-session');
  if (newSessBtn) newSessBtn.addEventListener('click', createNewSession);

  const toggleSessionsBtn = document.getElementById('btn-toggle-sessions');
  if (toggleSessionsBtn) {
    toggleSessionsBtn.addEventListener('click', () => {
      const container = document.querySelector('.chat-container');
      if (container) {
        if (window.innerWidth <= 960) {
          container.classList.toggle('sessions-open');
        } else {
          container.classList.toggle('sessions-collapsed');
        }
      }
    });
  }
}

function renderSessionsList() {
  const list = document.getElementById('chat-sessions-list');
  if (!list) return;

  list.innerHTML = state.sessions.map(s => `
    <div class="session-item ${s.id === state.currentSessionId ? 'active' : ''}" onclick="selectSession('${s.id}')" title="${escapeHtml(s.title)}">
      <div class="session-item-title">${escapeHtml(s.title)}</div>
      <button class="btn-delete-session" title="Delete conversation" onclick="event.stopPropagation(); deleteSession('${s.id}')">\u2715</button>
    </div>
  `).join('');
}

async function selectSession(sessionId) {
  state.currentSessionId = sessionId;
  renderSessionsList();
  
  try {
    const res = await fetch(`${API_BASE}/api/chat/history/${sessionId}`);
    if (res.ok) {
      state.chatMessages = await res.json();
      renderChatMessages();
    }
  } catch (e) {
    console.error('Failed to load history:', e);
  }
}

async function createNewSession() {
  const newId = `session_${Date.now()}`;
  state.currentSessionId = newId;
  state.chatMessages = [];
  state.sessions.unshift({
    id: newId,
    title: 'New Conversation',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    message_count: 0
  });
  renderSessionsList();
  renderChatMessages();
}

async function deleteSession(sessionId) {
  try {
    await fetch(`${API_BASE}/api/chat/sessions/${sessionId}`, { method: 'DELETE' });
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
    if (state.currentSessionId === sessionId) {
      if (state.sessions.length > 0) {
        selectSession(state.sessions[0].id);
      } else {
        createNewSession();
      }
    } else {
      renderSessionsList();
    }
  } catch (e) {
    console.error('Delete session error:', e);
  }
}

function renderChatMessages() {
  const container = document.getElementById('chat-messages-container');
  if (!container) return;

  if (state.chatMessages.length === 0) {
    container.innerHTML = `
      <div style="text-align: center; margin: auto; max-width: 500px; color: var(--text-muted);">
        <div style="font-size: 38px; margin-bottom: 12px;">\uD83E\uDD16</div>
        <h3 style="color: var(--text-main); margin-bottom: 8px; font-family: var(--font-display);">Multi-Doc Knowledge Assistant</h3>
        <p style="font-size: 13.5px; line-height: 1.5; margin-bottom: 20px;">
          Ask natural-language questions over your uploaded documents. Answers are strictly grounded in retrieved document facts.
        </p>
        <div class="scenario-pills" style="justify-content: center;">
          <div class="scenario-pill" onclick="launchRAGQuery('What information is available in my documents?')">\uD83D\uDCC4 "What information is available?"</div>
          <div class="scenario-pill" onclick="launchRAGQuery('Summarize my documents')">\uD83D\uDCD1 "Summarize my documents"</div>
          <div class="scenario-pill" onclick="launchRAGQuery('Find key topics in my documents')">\uD83D\uDD0D "Find key topics"</div>
          <div class="scenario-pill" onclick="launchRAGQuery('What is RAG?')">\uD83D\uDCD6 "What is RAG?"</div>
        </div>
      </div>
    `;
    return;
  }

  container.innerHTML = state.chatMessages.map(msg => {
    const isUser = msg.role === 'user';
    const parsedContent = typeof marked !== 'undefined' ? marked.parse(msg.content) : escapeHtml(msg.content);
    
    let citationsHtml = '';
    if (!isUser && msg.citations && msg.citations.length > 0) {
      const docMap = {};
      msg.citations.forEach((c, idx) => {
        const docKey = c.document_name || 'Document';
        if (!docMap[docKey]) docMap[docKey] = [];
        docMap[docKey].push({ citation: c, idx: idx });
      });

      citationsHtml = `
        <div class="citations-container">
          <div class="citations-header">
            <span>\uD83D\uDCC4 Grounded Sources:</span>
          </div>
          <div class="citations-list">
            ${Object.keys(docMap).map(docName => {
              const pages = docMap[docName];
              if (pages.length === 1) {
                const item = pages[0];
                const c = item.citation;
                return `
                  <div class="citation-chip" onclick="showCitationSnippet(${item.idx}, '${msg.id}')">
                    <span>${escapeHtml(docName)}</span>
                    <span class="page-tag">Page ${c.page_number}</span>
                    <span class="score-tag">${c.relevance_grade}</span>
                  </div>
                `;
              } else {
                return `
                  <div class="citation-group-chip">
                    <span class="doc-group-title">\uD83D\uDCC4 ${escapeHtml(docName)}</span>
                    <div class="doc-group-pages">
                      ${pages.map(item => {
                        const c = item.citation;
                        return `
                          <div class="citation-chip sub-page-chip" onclick="showCitationSnippet(${item.idx}, '${msg.id}')">
                            <span class="page-tag">Page ${c.page_number}</span>
                            <span class="score-tag">${c.relevance_grade}</span>
                          </div>
                        `;
                      }).join('')}
                    </div>
                  </div>
                `;
              }
            }).join('')}
          </div>
        </div>
      `;
    }

    let devTraceHtml = '';
    if (!isUser && msg.tool_traces && msg.tool_traces.length > 0) {
      const t = msg.tool_traces[0];
      devTraceHtml = `
        <details style="margin-top: 10px; font-size: 11.5px; color: var(--text-dim); cursor: pointer;">
          <summary style="outline: none; color: var(--accent-cyan);">\uD83D\uDD0D View Developer Retrieval Trace (${t.tool_name} \u2022 ${t.execution_time_ms}ms)</summary>
          <div style="background: rgba(0,0,0,0.4); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; margin-top: 6px; font-family: monospace; font-size: 11px;">
            <div><strong>Query:</strong> ${escapeHtml(JSON.stringify(t.input_params))}</div>
            <div style="margin-top: 4px;"><strong>Selected Tool:</strong> ${t.tool_name}</div>
            <div style="margin-top: 4px;"><strong>Execution Time:</strong> ${t.execution_time_ms} ms</div>
            <div style="margin-top: 4px;"><strong>Grounded Citations:</strong> ${msg.citations ? msg.citations.length : 0} sources</div>
          </div>
        </details>
      `;
    }

    return `
      <div class="chat-message-row ${isUser ? 'user' : 'agent'}">
        <div class="chat-avatar ${isUser ? 'user' : 'agent'}">
          ${isUser ? 'U' : 'AI'}
        </div>
        <div class="chat-bubble markdown-body">
          ${parsedContent}
          ${citationsHtml}
          ${devTraceHtml}
        </div>
      </div>
    `;
  }).join('');

  container.scrollTop = container.scrollHeight;
}

async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  if (!input) return;
  const message = input.value.trim();
  if (!message || state.isSending) return;

  state.isSending = true;
  input.value = '';
  const btn = document.getElementById('btn-send-chat');
  if (btn) btn.disabled = true;

  const tempUserMsg = {
    id: `temp_${Date.now()}`,
    session_id: state.currentSessionId,
    role: 'user',
    content: message,
    created_at: new Date().toISOString()
  };
  state.chatMessages.push(tempUserMsg);
  renderChatMessages();

  try {
    const payload = {
      message: message,
      session_id: state.currentSessionId,
      mode: state.chatMode
    };
    if (state.selectedDocument && state.selectedDocument.id) {
      payload.document_ids = [state.selectedDocument.id];
    }

    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      const data = await res.json();
      state.chatMessages.push({
        id: `ai_${Date.now()}`,
        session_id: data.session_id,
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        tool_traces: data.tool_traces,
        created_at: new Date().toISOString()
      });
      renderChatMessages();
      fetchSessions();
    } else {
      const err = await res.json();
      showToast(err.detail || 'Error communicating with agent', 'error');
    }
  } catch (e) {
    showToast(`Network error: ${e.message}`, 'error');
  } finally {
    state.isSending = false;
    if (btn) btn.disabled = false;
  }
}

function showCitationSnippet(citationIdx, msgId) {
  const msg = state.chatMessages.find(m => m.id === msgId);
  if (!msg || !msg.citations || !msg.citations[citationIdx]) return;

  const c = msg.citations[citationIdx];
  openModal(`Source: ${c.document_name} (Page ${c.page_number})`, `
    <div style="margin-bottom: 12px; display: flex; gap: 8px;">
      <span class="status-badge-inline">Relevance: ${c.relevance_grade} (${Math.round(c.similarity_score * 100)}%)</span>
    </div>
    <h4 style="color: var(--text-main); margin-bottom: 8px;">Retrieved Passage:</h4>
    <div style="background: rgba(0,0,0,0.35); padding: 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-color); font-size: 13.5px; line-height: 1.6; white-space: pre-wrap;">${escapeHtml(c.snippet)}</div>
  `);
}

// --- Upload View ---
function setupUploadHandlers() {
  const dropzone = document.getElementById('upload-dropzone');
  const fileInput = document.getElementById('file-upload-input');
  if (!dropzone || !fileInput) return;

  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFiles(e.dataTransfer.files);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFiles(fileInput.files);
    }
  });
}

async function handleFiles(fileList) {
  const statusContainer = document.getElementById('upload-status-box');
  if (statusContainer) {
    statusContainer.style.display = 'block';
    statusContainer.innerHTML = `<div style="display: flex; align-items: center; gap: 8px;"><div class="pulse-dot"></div><span>Uploading and indexing ${fileList.length} file(s)...</span></div>`;
  }

  const formData = new FormData();
  for (let i = 0; i < fileList.length; i++) {
    formData.append('files', fileList[i]);
  }

  const chunkSizeElem = document.getElementById('upload-chunk-size');
  const chunkOverlapElem = document.getElementById('upload-chunk-overlap');
  if (chunkSizeElem && chunkSizeElem.value) formData.append('chunk_size', chunkSizeElem.value);
  if (chunkOverlapElem && chunkOverlapElem.value) formData.append('chunk_overlap', chunkOverlapElem.value);

  try {
    const res = await fetch(`${API_BASE}/api/documents/upload`, {
      method: 'POST',
      body: formData
    });

    if (res.ok) {
      const docs = await res.json();
      if (statusContainer) {
        statusContainer.innerHTML = `
          <div style="color: var(--accent-emerald); font-weight: 600; margin-bottom: 8px;">
            \u2713 Successfully indexed ${docs.length} document(s)!
          </div>
          <div style="font-size: 12px; color: var(--text-muted);">
            ${docs.map(d => `<div>\u2022 <strong>${escapeHtml(d.original_name)}</strong> - ${d.total_pages} page(s), ${d.total_chunks} chunks</div>`).join('')}
          </div>
        `;
      }
      await refreshAll();
      showToast(`Uploaded ${docs.length} document(s)`);
    } else {
      const err = await res.json();
      if (statusContainer) statusContainer.innerHTML = `<div style="color: var(--accent-rose);">\u274C Upload failed: ${escapeHtml(err.detail)}</div>`;
    }
  } catch (e) {
    if (statusContainer) statusContainer.innerHTML = `<div style="color: var(--accent-rose);">\u274C Network error: ${escapeHtml(e.message)}</div>`;
  }
}

// --- Knowledge Base View ---
function renderKnowledgeBase() {
  const tbody = document.getElementById('kb-documents-tbody');
  if (!tbody) return;

  if (state.documents.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="7" style="text-align: center; padding: 24px; color: var(--text-muted);">
          No documents in knowledge base.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = state.documents.map(d => `
    <tr>
      <td style="font-weight: 600;">
        <span style="margin-right: 6px;">\uD83D\uDCC4</span>${escapeHtml(d.original_name)}
      </td>
      <td><span class="status-badge-inline" style="margin: 0; font-size: 11px;">${d.file_type.toUpperCase()}</span></td>
      <td>${(d.file_size / 1024).toFixed(1)} KB</td>
      <td>${d.total_pages}</td>
      <td><strong style="color: var(--primary);">${d.total_chunks}</strong></td>
      <td>${new Date(d.created_at).toLocaleDateString()}</td>
      <td>
        <div style="display: flex; gap: 6px;">
          <button class="btn btn-primary btn-sm" onclick="startChatWithDoc('${d.id}', '${escapeHtml(d.original_name)}')">
            Ask Questions
          </button>
          <button class="btn btn-secondary btn-sm" onclick="viewDocumentChunks('${d.id}', '${escapeHtml(d.original_name)}')">
            Chunks
          </button>
          <button class="btn btn-danger btn-sm" onclick="deleteDocument('${d.id}')">
            Delete
          </button>
        </div>
      </td>
    </tr>
  `).join('');
}

async function viewDocumentChunks(docId, docName) {
  try {
    const res = await fetch(`${API_BASE}/api/documents/${docId}/chunks`);
    if (res.ok) {
      const chunks = await res.json();
      openModal(`Indexed Chunks: ${docName} (${chunks.length} Chunks)`, `
        <div style="display: flex; flex-direction: column; gap: 14px;">
          ${chunks.map(c => `
            <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: var(--radius-sm); padding: 12px;">
              <div style="display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 12px; color: var(--accent-cyan);">
                <span>Chunk ID: <strong>${c.chunk_id}</strong> (Page ${c.page_number})</span>
                <span>${c.char_count} Chars</span>
              </div>
              <div style="font-size: 13px; line-height: 1.5; color: var(--text-main); white-space: pre-wrap;">${escapeHtml(c.content)}</div>
            </div>
          `).join('')}
        </div>
      `);
    }
  } catch (e) {
    showToast(`Failed to load chunks: ${e.message}`, 'error');
  }
}

async function deleteDocument(docId) {
  if (!confirm('Are you sure you want to delete this document from the knowledge base?')) return;
  try {
    const res = await fetch(`${API_BASE}/api/documents/${docId}`, { method: 'DELETE' });
    if (res.ok) {
      showToast('Document deleted successfully');
      if (state.selectedDocument && state.selectedDocument.id === docId) {
        clearDocumentSelection();
      }
      await refreshAll();
    } else {
      showToast('Failed to delete document', 'error');
    }
  } catch (e) {
    showToast(`Network error: ${e.message}`, 'error');
  }
}

// --- Multi-Document Comparison View ---
function setupComparisonHandlers() {
  const compareBtn = document.getElementById('btn-run-comparison');
  if (compareBtn) compareBtn.addEventListener('click', runDocumentComparison);
}

function renderComparisonSelectors() {
  const selectA = document.getElementById('compare-doc-a');
  const selectB = document.getElementById('compare-doc-b');
  if (!selectA || !selectB) return;

  const seenNames = new Set();
  const uniqueDocs = [];
  for (const d of state.documents) {
    if (!seenNames.has(d.original_name)) {
      seenNames.add(d.original_name);
      uniqueDocs.push(d);
    }
  }

  const options = uniqueDocs.map(d => `<option value="${d.id}">${escapeHtml(d.original_name)}</option>`).join('');
  selectA.innerHTML = options || '<option disabled>No documents available</option>';
  selectB.innerHTML = options || '<option disabled>No documents available</option>';

  if (uniqueDocs.length >= 2) {
    selectB.selectedIndex = 1;
  }
}

async function runDocumentComparison() {
  const docAId = document.getElementById('compare-doc-a')?.value;
  const docBId = document.getElementById('compare-doc-b')?.value;
  const aspects = document.getElementById('compare-aspects')?.value.trim();
  const resultBox = document.getElementById('comparison-result-box');
  const outputContent = document.getElementById('comparison-output-content');
  const btn = document.getElementById('btn-run-comparison');

  if (!docAId || !docBId) {
    showToast('Please select two documents to compare', 'error');
    return;
  }

  if (btn) {
    btn.disabled = true;
    btn.innerText = 'Analyzing Documents...';
  }
  if (resultBox) resultBox.style.display = 'block';
  if (outputContent) outputContent.innerHTML = '<div style="display: flex; align-items: center; gap: 8px;"><div class="pulse-dot"></div><span>Synthesizing comparative cross-document insights...</span></div>';

  try {
    const res = await fetch(`${API_BASE}/api/tools/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        document_ids: [docAId, docBId],
        query: aspects || "Compare key concepts, differences, and complementary aspects across these documents."
      })
    });

    if (res.ok) {
      const data = await res.json();
      const docAName = data.doc_a_name || 'Document A';
      const docBName = data.doc_b_name || 'Document B';
      const chunksA = data.doc_a_chunks || [];
      const chunksB = data.doc_b_chunks || [];

      const comparisonMd = `### \uD83D\uDD0D Multi-Document Comparison: ${docAName} vs ${docBName}

**Key Findings from ${docAName}:**
${chunksA.length > 0 ? chunksA.map(c => `- ${c}`).join('\n') : '- No specific matching sections found.'}

**Key Findings from ${docBName}:**
${chunksB.length > 0 ? chunksB.map(c => `- ${c}`).join('\n') : '- No specific matching sections found.'}
`;
      if (outputContent) {
        outputContent.innerHTML = typeof marked !== 'undefined' ? marked.parse(comparisonMd) : escapeHtml(comparisonMd);
      }
    } else {
      const err = await res.json();
      if (outputContent) outputContent.innerHTML = `<div style="color: var(--accent-rose);">\u274C Comparison error: ${escapeHtml(err.detail)}</div>`;
    }
  } catch (e) {
    if (outputContent) outputContent.innerHTML = `<div style="color: var(--accent-rose);">\u274C Network error: ${escapeHtml(e.message)}</div>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '\uD83D\uDE80 Run Comparative Analysis';
    }
  }
}

// --- Evaluation Benchmark View ---
function setupEvaluationHandlers() {
  const evalBtn = document.getElementById('btn-run-eval');
  if (evalBtn) evalBtn.addEventListener('click', triggerEvaluation);
}

async function fetchEvaluationResults() {
  try {
    const res = await fetch(`${API_BASE}/api/evaluation/results`);
    if (res.ok) {
      const data = await res.json();
      state.evalResults = data;
      renderEvaluationMetrics(data);
    }
  } catch (e) {
    console.error('Failed to fetch benchmark results:', e);
  }
}

async function triggerEvaluation() {
  const btn = document.getElementById('btn-run-eval');
  if (btn) {
    btn.disabled = true;
    btn.innerText = '\u26A1 Evaluating RAG Pipeline...';
  }

  try {
    const res = await fetch(`${API_BASE}/api/evaluation/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ top_k: 5 })
    });
    if (res.ok) {
      const data = await res.json();
      state.evalResults = data;
      renderEvaluationMetrics(data);
      showToast('Benchmark evaluation completed!');
    }
  } catch (e) {
    showToast(`Evaluation failed: ${e.message}`, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerText = '\u26A1 Run Live Evaluation';
    }
  }
}

function renderEvaluationMetrics(data) {
  if (!data) return;
  const m = data.metrics || data;
  const pElem = document.getElementById('eval-precision');
  const rElem = document.getElementById('eval-recall');
  const revElem = document.getElementById('eval-relevance');
  const fElem = document.getElementById('eval-faithfulness');
  const cElem = document.getElementById('eval-citation-acc');
  const lElem = document.getElementById('eval-latency');

  if (pElem && m.retrieval_precision !== undefined) pElem.innerText = `${Math.round(m.retrieval_precision * 100)}%`;
  if (rElem && m.retrieval_recall !== undefined) rElem.innerText = `${Math.round(m.retrieval_recall * 100)}%`;
  if (revElem && m.context_relevance !== undefined) revElem.innerText = `${Math.round(m.context_relevance * 100)}%`;
  if (fElem && m.answer_faithfulness !== undefined) fElem.innerText = `${Math.round(m.answer_faithfulness * 100)}%`;
  if (cElem && m.citation_accuracy !== undefined) cElem.innerText = `${Math.round(m.citation_accuracy * 100)}%`;
  if (lElem && m.average_latency_ms !== undefined) lElem.innerText = `${Math.round(m.average_latency_ms)} ms`;

  const tbody = document.getElementById('eval-queries-tbody');
  if (!tbody) return;

  const items = data.item_results || data.detailed_evaluations || [];
  tbody.innerHTML = items.map(e => `
    <tr>
      <td style="font-weight: 500;">${escapeHtml(e.query || e.question || '')}</td>
      <td><code>${escapeHtml(e.expected_document || e.relevant_document || '')}</code></td>
      <td>${(e.retrieved_documents || []).map(d => `<span class="status-badge-inline" style="margin: 2px;">${escapeHtml(d)}</span>`).join('')}</td>
      <td><strong style="color: ${(e.retrieval_precision || 0) >= 0.5 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${Math.round((e.retrieval_precision || 0) * 100)}%</strong></td>
      <td>${Math.round(e.latency_ms || 0)} ms</td>
    </tr>
  `).join('');
}

// --- Settings View ---
function setupSettingsHandlers() {
  const saveBtn = document.getElementById('btn-save-settings');
  if (saveBtn) saveBtn.addEventListener('click', saveSettings);

  const resetBtn = document.getElementById('btn-system-reset');
  if (resetBtn) resetBtn.addEventListener('click', resetDatabase);
}

function populateSettings() {
  if (state.systemStatus) {
    const s = state.systemStatus;
    const provSelect = document.getElementById('setting-llm-provider');
    if (provSelect) provSelect.value = s.llm_provider;

    const groqBadge = document.getElementById('groq-key-status');
    if (groqBadge) {
      groqBadge.innerText = s.has_groq_key ? 'Configured \u2713' : 'Missing Key';
      groqBadge.style.color = s.has_groq_key ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }

    const geminiBadge = document.getElementById('gemini-key-status');
    if (geminiBadge) {
      geminiBadge.innerText = s.has_gemini_key ? 'Configured \u2713' : 'Missing Key';
      geminiBadge.style.color = s.has_gemini_key ? 'var(--accent-emerald)' : 'var(--accent-amber)';
    }
  }

  const themeSelect = document.getElementById('setting-app-theme');
  if (themeSelect) {
    themeSelect.value = state.theme || 'dark';
  }
}

async function saveSettings() {
  const provElem = document.getElementById('setting-llm-provider');
  const groqElem = document.getElementById('setting-groq-key');
  const geminiElem = document.getElementById('setting-gemini-key');
  const chunkElem = document.getElementById('setting-chunk-size');
  const overlapElem = document.getElementById('setting-chunk-overlap');

  const provider = provElem ? provElem.value : 'groq';
  const groqKey = groqElem ? groqElem.value.trim() : '';
  const geminiKey = geminiElem ? geminiElem.value.trim() : '';
  const chunkSize = chunkElem ? parseInt(chunkElem.value, 10) : 600;
  const chunkOverlap = overlapElem ? parseInt(overlapElem.value, 10) : 100;

  const payload = {
    llm_provider: provider,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap
  };
  if (groqKey) payload.groq_api_key = groqKey;
  if (geminiKey) payload.gemini_api_key = geminiKey;

  try {
    const res = await fetch(`${API_BASE}/api/system/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (res.ok) {
      if (groqElem) groqElem.value = '';
      if (geminiElem) geminiElem.value = '';
      showToast('Settings saved successfully!');
      await fetchSystemStatus();
    } else {
      showToast('Failed to save settings', 'error');
    }
  } catch (e) {
    showToast(`Network error: ${e.message}`, 'error');
  }
}

async function resetDatabase() {
  if (!confirm('WARNING: This will purge all indexed documents, ChromaDB vectors, and chat history. Continue?')) return;
  try {
    const res = await fetch(`${API_BASE}/api/system/reset`, { method: 'POST' });
    if (res.ok) {
      showToast('System database reset successfully');
      await refreshAll();
      createNewSession();
    }
  } catch (e) {
    showToast(`Reset error: ${e.message}`, 'error');
  }
}

// --- UI Utilities ---
function openModal(title, bodyHtml) {
  const modalTitle = document.getElementById('modal-title');
  const modalBody = document.getElementById('modal-body');
  const modal = document.getElementById('global-modal');

  if (modalTitle) modalTitle.innerText = title;
  if (modalBody) modalBody.innerHTML = bodyHtml;
  if (modal) modal.style.display = 'flex';
}

function closeModal() {
  const modal = document.getElementById('global-modal');
  if (modal) modal.style.display = 'none';
}

function showToast(message, type = 'success') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerText = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Explicitly bind globally for inline HTML onclick handlers
window.switchTab = switchTab;
window.launchRAGQuery = launchRAGQuery;
window.startChatWithDoc = startChatWithDoc;
window.viewDocumentChunks = viewDocumentChunks;
window.deleteDocument = deleteDocument;
window.clearDocumentSelection = clearDocumentSelection;
window.showCitationSnippet = showCitationSnippet;
window.selectSession = selectSession;
window.deleteSession = deleteSession;
window.createNewSession = createNewSession;
window.toggleThemeDropdown = toggleThemeDropdown;
window.setAppTheme = setAppTheme;
window.openModal = openModal;
window.closeModal = closeModal;
window.saveSettings = saveSettings;
window.resetDatabase = resetDatabase;
window.sendChatMessage = sendChatMessage;
window.runDocumentComparison = runDocumentComparison;
window.triggerEvaluation = triggerEvaluation;
