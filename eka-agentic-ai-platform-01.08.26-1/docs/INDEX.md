# 📘 Autonomous AI Studio & Multi-Agent Development Platform
## Comprehensive User & Developer Documentation

Welcome to the official technical and operational documentation for the **Autonomous AI Studio & Multi-Agent Development Platform**. This workspace is a full-stack engineering hub integrating **Gemini AI**, **Local LLM engines (Ollama, vLLM, LM Studio, llama.cpp, Custom OpenAI-compatible servers)**, **Model Context Protocol (MCP)**, **Agent-to-Agent (A2A) message channels**, **Human-in-the-Loop (HITL) verifiers**, **Visual Workflow Pipelines**, and a **Python SMTP Mail Server**.

---

## 🗺️ Documentation System Architecture

```
                                  +-------------------------------------------------------+
                                  |                 MASTER DOCUMENTATION                   |
                                  |                      INDEX.md                         |
                                  +---------------------------+---------------------------+
                                                              |
            +-------------------------------------------------+-------------------------------------------------+
            |                                                 |                                                 |
+-----------v-------------------------+   +-------------------v-------------------+   +-------------------------v-----------+
| 01. USER GUIDE & UI SCENARIOS       |   | 02. LOCAL LLMs & SERVICES INTEGRATION |   | 03. MULTI-AGENT & MCP / A2A ENGINE  |
| 01_USER_GUIDE_APPLICATION.md        |   | 02_LOCAL_LLM_AND_SERVICES_GUIDE.md    |   | 03_MULTI_AGENT_MCP_A2A_GUIDE.md     |
| - Interactive UI Tab Overview       |   | - Ollama, vLLM, LM Studio, llama.cpp  |   | - Agent Topology Graph              |
| - Visual Diagrams & UI Screens      |   | - Per-Provider Isolated Configurations |   | - MCP Tool & Resource Registry      |
| - End-to-End User Usage Scenarios   |   | - Local Python Mail Server (Port 1025)|   | - A2A Channels & HITL Gatekeeper    |
+-------------------------------------+   +---------------------------------------+   +-------------------------------------+
            |                                                 |                                                 |
            +-------------------------------------------------+-------------------------------------------------+
                                                              |
            +-------------------------------------------------+-------------------------------------------------+
            |                                                 |                                                 |
+-----------v-------------------------+   +-------------------v-------------------+   +-------------------------v-----------+
| 04. PIPELINE WORKFLOW BUILDER       |   | 05. DEVELOPER SETUP & ARCHITECTURE    |   | 06. COMPLETE API & LOGGING REF      |
| 04_PIPELINE_WORKFLOW_BUILDER.md     |   | 05_DEVELOPER_SETUP_AND_ARCHITECTURE.md|   | 06_API_REFERENCE_AND_ENDPOINTS.md   |
| - Drag-and-Drop Workflow Engine     |   | - Directory Tree & Code Structure     |   | - REST API Schemas & SSE Streams    |
| - Step Types & Approval Controls    |   | - Installation, Build & Env Vars      |   | - Log Persistence & File Storage    |
+-------------------------------------+   +---------------------------------------+   +-------------------------------------+
```

---

## 📚 Master Table of Contents

### 1. 📖 [User Guide & Application Scenarios](./01_USER_GUIDE_APPLICATION.md)
   - **Interactive Layout Architecture**: Overview of Workspace Tabs, Navigation, Settings Sidebar, Terminal & Preview Canvas.
   - **Visual Diagrams & Layout Illustrations**: UI wireframes and visual layouts for all primary app sections.
   - **Usage Scenario A**: Natural Language AI Coding & Interactive Code Edits.
   - **Usage Scenario B**: Deep Technical Research & Web Document Synthesis.
   - **Usage Scenario C**: Automatic Workspace Documentation Generation.
   - **Usage Scenario D**: Automated Unit & Integration Test Suite Generation.
   - **Usage Scenario E**: Visual Workflow Pipeline Design & Execution.
   - **Usage Scenario F**: Multi-Agent Orchestration with MCP Tools, A2A Channels, and HITL Verification.
   - **Usage Scenario G**: Local Python SMTP Mail Server Operations (Port 1025).
   - **Usage Scenario H**: Internet Search Engine Assists (DuckDuckGo, Google CSE, Bing, Brave, Serper).

### 2. 🦙 [Local LLM Engines & Local Services Integration](./02_LOCAL_LLM_AND_SERVICES_GUIDE.md)
   - **Supported Local LLM Frameworks**:
     - **Ollama** (`http://localhost:11434`)
     - **vLLM** (`http://localhost:8000/v1`)
     - **LM Studio** (`http://localhost:1234/v1`)
     - **llama.cpp / llama-server** (`http://localhost:8080/v1`)
     - **Custom OpenAI-Compatible Servers** (TextGenWebUI, Jan.ai, LocalAI, KoboldCPP)
   - **Isolated Provider Configuration Storage**: How base URLs, API keys, online health checks, and active models are saved separately per provider.
   - **Local Python SMTP Mail Server**: Starting, stopping, and testing the built-in mail server on Port 1025.

### 3. 🤖 [Multi-Agent System, MCP & A2A Protocol Engine](./03_MULTI_AGENT_MCP_A2A_GUIDE.md)
   - **Core Agent Architecture**: Coordinator, Code Synthesis, Technical Researcher, QA & Test Engineer, HITL Verifier, Custom User Agents.
   - **Model Context Protocol (MCP)**: Registered tools (`read_file`, `write_file`, `edit_file`, `execute_command`, `web_search`, `run_tests`), security scopes, and JSON schema arguments.
   - **Agent-to-Agent (A2A) Bus**: Channel routing (`human_agent`, `agent_to_agent`, `hitl`), message format, and real-time inspector.
   - **Human-in-the-Loop (HITL) Gatekeeper**: Output review, inline modification, approval/rejection workflows, and direct instruction messaging.

### 4. 🎛️ [Pipeline & Workflow Builder](./04_PIPELINE_WORKFLOW_BUILDER.md)
   - **Visual Workflow Node Construction**: Sequential and branching step sequences.
   - **Step Types**: Research, Code Generation, Testing, System Documentation, Command Execution.
   - **Execution Engine**: Server-Sent Events (SSE) streaming, status indicators, pause for human approval (`requireApproval`), and retry controls.

### 5. 💻 [Developer Architecture, Setup & Installation](./05_DEVELOPER_SETUP_AND_ARCHITECTURE.md)
   - **Code Base Organization**: Complete codebase file tree (`/src`, `/src/api`, `/src/components`, `/src/multi_agent`, `/docs`).
   - **System Requirements & Installation**: Prerequisites (Node.js 18+, npm, Python 3, Local LLM binaries).
   - **Environment Variables & Configuration**: `.env.example`, default values, feature flags (`ENABLE_MULTI_AGENT`), fallback rules.
   - **Output Generation Pipeline**: Text & Code generation drivers, streaming mechanics, markdown rendering.
   - **Storage & Log Persistence**: Disk persistence locations (`/docs`, `/docs/research`, `/src`, execution memory stores, message logs).

### 6. 🛰️ [Complete API Specification & Event Streaming](./06_API_REFERENCE_AND_ENDPOINTS.md)
   - **REST API Reference**: Request/Response schemas for `/api/chat`, `/api/agent/*`, `/api/models/*`, `/api/multi-agent/*`, `/api/mailserver/*`.
   - **Server-Sent Events (SSE)**: Event format, payload structures, streaming cancellation tokens.
   - **cURL Examples**: Code samples for testing all backend routes.

---

## ⚡ Quick Developer Execution Commands

```bash
# 1. Install project dependencies
npm install

# 2. Configure environment variables (.env)
cp .env.example .env

# 3. Launch full-stack development server (Port 3000)
npm run dev

# 4. Compile production build
npm run build

# 5. Start production Node server
npm run start
```
