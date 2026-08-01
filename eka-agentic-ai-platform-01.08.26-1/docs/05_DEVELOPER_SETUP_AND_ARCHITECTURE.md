# 💻 Developer Architecture, Setup & Installation Guide

This document provides a comprehensive technical overview for software engineers, contributors, and system architects. It details how code is organized, installation steps, default configurations, output generation mechanisms, and data storage/logging rules.

---

## 📂 1. Directory Structure & Codebase Organization

The project follows a modular, type-safe full-stack layout combining **Express 4/5**, **Vite 5**, **React 18**, **TypeScript**, and **Tailwind CSS**.

```
.
├── .env                          # Local environment variables & secrets (ignored by git)
├── .env.example                  # Environment variable blueprint
├── package.json                  # Dependencies, scripts (dev, build, start, lint)
├── server.ts                     # Primary Express server entry point & SSE API router
├── vite.config.ts                # Vite dev server configuration & plugin setup
├── tsconfig.json                 # TypeScript compiler configuration
├── metadata.json                 # Application name, description & capabilities metadata
│
├── docs/                         # Comprehensive System Documentation & Research Vault
│   ├── INDEX.md                  # Documentation Master Index & Architecture Map
│   ├── 01_USER_GUIDE_APPLICATION.md # End-User Guide, UI Wireframes & Interactive Scenarios
│   ├── 02_LOCAL_LLM_AND_SERVICES_GUIDE.md # Local LLM Frameworks & Python Mail Server
│   ├── 03_MULTI_AGENT_MCP_A2A_GUIDE.md    # Multi-Agent Architecture, MCP Tools & A2A Bus
│   ├── 04_PIPELINE_WORKFLOW_BUILDER.md    # Visual Workflow Pipeline Builder Guide
│   ├── 05_DEVELOPER_SETUP_AND_ARCHITECTURE.md # (This File) Developer Setup & Code Map
│   ├── 06_API_REFERENCE_AND_ENDPOINTS.md  # Complete REST & SSE API Reference
│   └── research/                 # Generated Research Summaries & Web Grounding Briefs
│
├── src/                          # Frontend React Source & Backend Shared API Handlers
│   ├── main.tsx                  # React Application DOM Entry Point
│   ├── App.tsx                   # Main React Application Container & Workspace Layout
│   ├── index.css                 # Tailwind CSS Global Imports & Styles
│   ├── types.ts                  # Shared TypeScript Types, Enums & Default Configs
│   │
│   ├── api/                      # Backend Route Handlers & Integration Drivers
│   │   ├── routes/
│   │   │   ├── agent.ts          # AI Agent prompt execution & surgical code editing
│   │   │   ├── chat.ts           # Conversational chat handler & SSE stream manager
│   │   │   ├── documentation.ts  # System documentation generator service
│   │   │   ├── mailserver.ts     # Local Python SMTP mail server process manager
│   │   │   ├── models.ts         # LLM model connectivity & health check router
│   │   │   ├── multiAgent.ts     # Multi-Agent orchestrator, MCP & A2A message bus
│   │   │   ├── research.ts       # Deep technical research & web search aggregator
│   │   │   ├── testing.ts        # Unit test suite generation & test runner
│   │   │   ├── workflow.ts       # Visual pipeline workflow execution engine
│   │   │   └── workspace.ts      # Workspace file tree, file view/edit endpoints
│   │   └── shared/
│   │       ├── llm.ts            # Unified LLM Driver (Gemini, Local LLM, OpenAI)
│   │       └── search.ts         # Multi-provider web search assist driver
│   │
│   ├── components/               # Modular React UI Components
│   │   ├── ChatPanel.tsx         # Conversational Chat UI & Model Controls
│   │   ├── CodeViewer.tsx        # File Code Viewer, Diff Inspector & Editor
│   │   ├── DirectoryViewer.tsx   # Workspace File Tree Explorer
│   │   ├── ModelSelector.tsx     # Settings Sidebar (#model-selector) & Services Panel
│   │   ├── Sidebar.tsx           # Navigation Drawer & Quick Action Controls
│   │   ├── WorkflowBuilder.tsx   # Drag-and-Drop Pipeline Step Builder
│   │   ├── WorkspaceTabs.tsx     # Tab Switcher Header (Chat, Docs, Tests, Multi-Agent)
│   │   └── multi_agent/
│   │       ├── AgentNetworkVisualizer.tsx # Agent Topology Node Graph & A2A Log Stream
│   │       ├── MultiAgentWorkspace.tsx    # Multi-Agent Container & HITL Verifier Panel
│   │       └── MCPToolRegistryViewer.tsx  # MCP Tools & Resource Inspector
│   │
│   └── multi_agent/              # Multi-Agent System Core Logic
│       ├── types.ts              # Agent, MCP Tool, and A2A Protocol Type Definitions
│       ├── registry.ts           # MCP Tool Definitions & Execution Sandboxes
│       └── orchestrator.ts       # Multi-Agent Coordinator Engine & A2A Bus
```

---

## 🛠️ 2. System Requirements & Installation

### Prerequisites
- **Node.js**: v18.0.0 or higher
- **npm**: v9.0.0 or higher
- **Python**: 3.8+ (Required for local Python SMTP Mail Server on Port 1025)
- **Local LLM Engine** (Optional, if using local AI): Ollama, vLLM, LM Studio, or llama.cpp

### Installation Steps

1. **Clone the Repository & Install Dependencies**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   npm install
   ```

2. **Configure Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

3. **Start Development Server**:
   ```bash
   # Option A: Combined Full-Stack Server (Frontend + Express API on Port 3000)
   npm run dev

   # Option B: Separate Standalone Backend API (Port 3001) & Standalone Frontend (Port 3000)
   # Terminal 1 - Backend Server:
   npm run dev:backend

   # Terminal 2 - Frontend Vite Server:
   npm run dev:frontend
   ```
   The application boots on `http://localhost:3000` (with API proxying to `http://localhost:3001` in standalone mode).

4. **Production Build & Execution**:
   ```bash
   # Build Vite static assets & compile server.ts to dist/server.cjs via esbuild
   npm run build

   # Option A: Combined Production Server (Frontend + Express API on Port 3000)
   npm run start

   # Option B: Standalone Production Backend API (Port 3001)
   npm run start:backend
   ```

---

## ⚙️ 3. Default Configurations & Environment Variables

### Environment Variables (`.env`)

| Variable Name | Required | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `PORT` | Optional | `3000` | Application server port. Hardcoded ingress routing targets Port 3000. |
| `NODE_ENV` | Optional | `development` | Runtime environment (`development` or `production`). |
| `GEMINI_API_KEY` | Optional | `""` | Google Gemini API key for server-side Gemini 2.5 Flash calls. |
| `ENABLE_MULTI_AGENT` | Optional | `true` | Feature flag enabling Multi-Agent Workspace, MCP, and A2A bus. |

### Default Model Server Configuration (`ModelServerConfig`)

```typescript
export const DEFAULT_MODEL_CONFIG: ModelServerConfig = {
  type: 'gemini',
  baseUrl: '',
  apiKey: '',
  selectedModel: 'gemini-3.6-flash',
  isOnline: true,
  availableModels: ['gemini-3.6-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-thinking'],
  activeLocalProvider: 'ollama',
  localConfigs: DEFAULT_LOCAL_CONFIGS, // Stored isolated per provider
  searchEngine: 'duckduckgo',
  searchEntryCount: 5,
};
```

---

## ⚙️ 4. How Output is Generated for Each Mode

### 4.1 Unified LLM Dispatcher (`src/api/shared/llm.ts`)
All text and code generation requests pass through `generateText()`:

1. **Gemini Engine (`type === 'gemini'`)**:
   - Uses `@google/genai` SDK with `GEMINI_API_KEY`.
   - Supports native web search grounding or fallback DuckDuckGo grounding.
2. **Local LLM Engine (`type === 'local_llm'` or `'ollama'`)**:
   - Inspects `customConfig.baseUrl` (e.g. `http://localhost:8000/v1`).
   - Automatically formats requests to OpenAI-compliant `/chat/completions` or native Ollama `/api/generate`.
3. **OpenAI Cloud (`type === 'openai'`)**:
   - Sends standard OpenAI POST requests to `https://api.openai.com/v1/chat/completions`.

### 4.2 Real-Time Event Streaming (SSE)
- Long-running multi-agent tasks, pipeline executions, and conversational streaming use Express Server-Sent Events (`res.setHeader('Content-Type', 'text/event-stream')`).
- Chunked JSON payloads are pushed over the stream (`data: { ... }\n\n`).
- Client controllers consume the stream using `EventSource` or `fetch` stream readers.

### 4.3 Code Generation & Surgical File Edits
- When the AI Coder Agent generates code:
  - It receives the current file content via `read_file`.
  - It generates targeted code edits or replacement blocks.
  - The backend applies the edits to disk using Node `fs/promises`.
  - The frontend re-fetches updated file nodes and updates the live preview canvas.

---

## 🗄️ 5. Storage of Logs, Outputs & Workspace Data

| Data Category | Storage Location | Persistence Format | Lifecycle |
| :--- | :--- | :--- | :--- |
| **Workspace Source Files** | `/src`, `/server.ts`, `/public` | Standard Disk Files (`.ts`, `.tsx`, `.json`) | Persistent across sessions |
| **System Documentation** | `/docs/` | Markdown Files (`.md`) | Persistent across sessions |
| **Research Summaries** | `/docs/research/` | Markdown Briefs (`.md`) | Persistent across sessions |
| **Unit Test Suites** | `/src/*.test.ts` or `tests/` | TypeScript Vitest Files | Persistent across sessions |
| **A2A Protocol Logs** | Express Server Memory / SSE Stream | In-Memory Circular Buffer (`A2AMessage[]`) | Active Session / Refreshable |
| **Workflow Executions** | Express Server Memory Store | Execution History Dictionary | Active Session |
| **Mail Server Logs** | Express Process Terminal Output | Console Log Stream | Active Process |
