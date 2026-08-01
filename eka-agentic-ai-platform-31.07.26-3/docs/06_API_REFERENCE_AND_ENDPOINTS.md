# 🛰️ Complete REST & SSE API Reference Specification

This document provides complete API reference documentation for all Express backend routes, Server-Sent Event (SSE) endpoints, Model Context Protocol (MCP) handlers, and local service routes.

---

## 📋 Endpoint Summary Table

> **Note on Direct API Access & CORS**: All `/api/*` endpoints are CORS-enabled (`Access-Control-Allow-Origin: *`) with support for standard HTTP methods (`GET`, `POST`, `PUT`, `DELETE`, `OPTIONS`, `PATCH`). External programs, mobile apps, Python scripts, and third-party frontends can call these endpoints directly on the standalone backend server (default `http://localhost:3001` or `http://localhost:3000`).

| Category | HTTP Method | Route Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **API Discovery** | `GET` | `/api` | Service metadata & list of all available backend endpoints |
| **System Health** | `GET` | `/api/health` | Health status, uptime & ISO timestamp |
| **Model Connectivity** | `POST` | `/api/models/check` | Test model connection, check health & list models |
| **Conversational Chat** | `POST` | `/api/chat` | Send prompt & stream back conversational responses |
| **Agent Engineering** | `POST` | `/api/agent/prompt` | Execute code generation or file editing prompt |
| **Research Engine** | `POST` | `/api/agent/research` | Trigger deep web research & save markdown brief |
| **Docs Generator** | `POST` | `/api/agent/generate-docs` | Scan codebase & generate markdown system documentation |
| **Test Suite Generator**| `POST` | `/api/agent/generate-tests` | Generate unit tests & execute test runner |
| **Multi-Agent System** | `GET` | `/api/multi-agent/agents` | Fetch active agent topology & status metrics |
| **Multi-Agent System** | `GET` | `/api/multi-agent/mcp-tools` | Get registered MCP tool definitions |
| **Multi-Agent System** | `POST` | `/api/multi-agent/mcp-execute` | Execute an MCP tool directly |
| **Multi-Agent System** | `POST` | `/api/multi-agent/a2a/send` | Ingest an Agent-to-Agent protocol message |
| **Multi-Agent System** | `GET` | `/api/multi-agent/stream` | SSE stream for real-time agent tasks & A2A logs |
| **Workflow Pipeline** | `POST` | `/api/workflow/execute` | Boot visual pipeline workflow execution |
| **Workflow Pipeline** | `GET` | `/api/workflow/stream/:id` | SSE stream for pipeline step execution updates |
| **Local Mail Server** | `GET` | `/api/mailserver/status` | Get Python SMTP mail server running state |
| **Local Mail Server** | `POST` | `/api/mailserver/start` | Launch local Python SMTP mail server (Port 1025) |
| **Local Mail Server** | `POST` | `/api/mailserver/stop` | Stop local Python SMTP mail server |
| **Workspace Files** | `GET` | `/api/workspace/files` | Get workspace file tree hierarchy |
| **Workspace Files** | `POST` | `/api/workspace/file` | Read single file contents |
| **Workspace Files** | `PUT` | `/api/workspace/file` | Save / overwrite file contents |

---

## 🔍 Detailed Endpoint Specifications

### 1. Model Check Endpoint (`POST /api/models/check`)

Checks connectivity and fetches available models for Gemini, Local LLMs (Ollama, vLLM, LM Studio, llama.cpp), or OpenAI.

#### Request Body Schema
```json
{
  "type": "local_llm",
  "provider": "vllm",
  "baseUrl": "http://localhost:8000/v1",
  "apiKey": ""
}
```

#### Response JSON Schema (Online)
```json
{
  "isOnline": true,
  "availableModels": [
    "meta-llama/Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen2.5-7B-Instruct"
  ]
}
```

#### Response JSON Schema (Offline Error)
```json
{
  "isOnline": false,
  "error": "Could not connect to VLLM at http://localhost:8000/v1: Connection refused",
  "availableModels": [
    "meta-llama/Llama-3-8B-Instruct"
  ]
}
```

#### cURL Sample
```bash
curl -X POST http://localhost:3000/api/models/check \
  -H "Content-Type: application/json" \
  -d '{
    "type": "local_llm",
    "provider": "vllm",
    "baseUrl": "http://localhost:8000/v1"
  }'
```

---

### 2. Multi-Agent A2A Message Ingestion (`POST /api/multi-agent/a2a/send`)

Injects an A2A message or HITL approval/rejection into the system message bus.

#### Request Body Schema
```json
{
  "sender_id": "human_operator",
  "recipient_id": "agent_coordinator",
  "message_type": "hitl_approval",
  "channel": "hitl",
  "payload": {
    "task": "HITL Verification APPROVED by Human Operator",
    "approved": true,
    "modified_output": "export function Header() { return <header>Updated</header>; }",
    "human_feedback": "Approved with minor style cleanups"
  }
}
```

#### Response JSON Schema
```json
{
  "success": true,
  "messageId": "msg_1722168000_a2a_09b",
  "timestamp": "2026-07-28T09:20:00.000Z"
}
```

#### cURL Sample
```bash
curl -X POST http://localhost:3000/api/multi-agent/a2a/send \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "human_operator",
    "recipient_id": "agent_coder",
    "message_type": "human_direct",
    "payload": { "task": "Refactor Button component to use Tailwind indigo theme" }
  }'
```

---

### 3. Execute MCP Tool (`POST /api/multi-agent/mcp-execute`)

Executes a registered Model Context Protocol (MCP) tool.

#### Request Body Schema
```json
{
  "tool": "read_file",
  "args": {
    "path": "src/types.ts"
  }
}
```

#### Response JSON Schema
```json
{
  "success": true,
  "tool": "read_file",
  "output": "export type ServerType = 'gemini' | 'local_llm' | 'ollama' | 'openai';...",
  "executionTimeMs": 12
}
```

---

### 4. Local Python SMTP Mail Server Controls (`POST /api/mailserver/start`)

Triggers the backend process manager to start the local Python SMTP server bound to Port 1025.

#### Response JSON Schema
```json
{
  "success": true,
  "running": true,
  "port": 1025,
  "message": "Python SMTP mail server running on 127.0.0.1:1025"
}
```

#### cURL Sample
```bash
# Check Status
curl http://localhost:3000/api/mailserver/status

# Start Mail Server
curl -X POST http://localhost:3000/api/mailserver/start

# Stop Mail Server
curl -X POST http://localhost:3000/api/mailserver/stop
```

---

## 📡 Server-Sent Events (SSE) Stream Specification

### Multi-Agent SSE Stream (`GET /api/multi-agent/stream`)

Streams real-time agent state changes, A2A protocol messages, and task metrics.

#### Headers
```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
```

#### SSE Stream Events
```http
data: {"type": "agent_state_update", "agentId": "agent_coder", "status": "EXECUTING"}

data: {"type": "a2a_message", "message": {"id": "msg_123", "sender_id": "agent_coordinator", "recipient_id": "agent_coder", "message_type": "task_delegation", "payload": {"task": "Implement Auth Endpoint"}}}

data: {"type": "hitl_checkpoint", "messageId": "msg_123", "output": "export const auth = ..."}
```
