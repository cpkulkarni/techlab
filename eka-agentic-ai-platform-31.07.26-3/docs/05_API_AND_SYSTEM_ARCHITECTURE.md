# API Reference & System Architecture

The application runs on a unified Express + Vite hybrid architecture listening on port `3000`.

---

## 🛰️ Express Backend API Endpoints

### Multi-Agent System Routes

| HTTP Method | Route Endpoint | Description | Query/Body Params |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/multi-agent/agents` | Fetches list of registered agents, current status, and metrics. | None |
| `GET` | `/api/multi-agent/mcp-tools` | Returns all available Model Context Protocol (MCP) tool definitions. | None |
| `POST` | `/api/multi-agent/mcp-execute` | Directly executes an MCP tool with provided JSON arguments. | `{ "tool": "search_web", "args": { "query": "..." } }` |
| `POST` | `/api/multi-agent/a2a-message` | Injects a manual Agent-to-Agent protocol message into the system bus. | `{ "senderId": "...", "receiverId": "...", "intent": "...", "payload": {} }` |
| `GET` | `/api/multi-agent/stream` | **Server-Sent Events (SSE)** endpoint streaming live agent tasks, A2A messages, and metrics. | None |

### Agent & Workspace Routes

| HTTP Method | Route Endpoint | Description | Query/Body Params |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/agent/research` | Triggers deep technical research on a query topic. | `{ "query": "..." }` |
| `POST` | `/api/agent/generate-docs` | Triggers documentation generation task across workspace files. | `{ "scope": "full" }` |
| `POST` | `/api/agent/generate-tests` | Triggers test suite generation for target modules. | `{ "targetFiles": ["src/utils.ts"] }` |

---

## ⚙️ Environment Variables & Feature Flags

Declare system environment variables in `.env`:

```env
# Server Port Configuration (Default: 3000)
PORT=3000

# Feature Flags
ENABLE_MULTI_AGENT=true

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 🏗️ Production Build & Deployment Command

To compile the application for production deployment:
```bash
# 1. Build frontend Vite assets and compile Express server to dist/server.cjs
npm run build

# 2. Start CommonJS production server
npm run start
```
