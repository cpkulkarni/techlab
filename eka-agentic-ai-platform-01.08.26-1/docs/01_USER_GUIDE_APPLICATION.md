# 📖 End-User Guide & Interactive Application Scenarios

Welcome to the **End-User Guide** for the Autonomous AI Studio & Multi-Agent Development Platform. This document explains every interactive visual tab, header control, sidebar setting, and usage scenario available to users.

---

## 🎨 1. Application Layout & Component Structure

The user interface is structured into four primary visual zones:

```
+---------------------------------------------------------------------------------------------------------+
| [A] TOP NAVIGATION HEADER & WORKSPACE TAB BAR                                                           |
| [Chat / Code]  [Pipeline Builder]  [Multi-Agent Workspace]  [Generate Docs]  [Generate Tests]  [Settings] |
+------------------------------------+--------------------------------------------------------------------+
|                                    |                                                                    |
| [B] LEFT CONTROL SIDEBAR           | [C] PRIMARY ACTIVE WORKSPACE VIEW                                  |
| - Model Selector & Status          |   - Interactive Chat Canvas / Code View / Diff Inspector           |
| - Local LLM Engine Selector        |   - Multi-Agent Topology Visualizer & A2A Stream Inspector         |
| - Local Mail Server Toggle         |   - Visual Pipeline Step Cards & Workflow Controls                 |
| - Internet Search Assist Engine    |   - Documentation Viewer / Unit Test Generator                      |
| - File Tree Explorer               |                                                                    |
|                                    |                                                                    |
+------------------------------------+--------------------------------------------------------------------+
| [D] BOTTOM STATUS & LOG TERMINAL PANEL                                                                  |
| [Diagnostics] [Terminal Output] [Mail Logs] [SSE Stream Status] [Stop Generation / Abort Task]          |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🖼️ 2. Detailed Visual Wireframes & Screen Mockups

### 2.1 Workspace Top Header & Navigation Bar
```
+-------------------------------------------------------------------------------------------------------------------------------+
| ⚡ Autonomous AI Studio | 💬 Workspace | 🔀 Pipeline Builder | 🌐 Multi-Agent System | 📑 Docs | 🧪 Tests | ⚙️ Settings | [Stop 🛑] |
+-------------------------------------------------------------------------------------------------------------------------------+
```

### 2.2 Model & Local Services Settings Panel (`#model-selector`)
```
+--------------------------------------------------------------------------+
| 🖥️ Settings                                              [● Online]     |
+--------------------------------------------------------------------------+
|  Server Type:                                                            |
|  [ Gemini ]           [ 💻 Local LLM ]            [ OpenAI ]             |
+--------------------------------------------------------------------------+
|  Local LLM Service Engine:                                               |
|  [ Ollama :11434 ] [ vLLM :8000 ] [ LM Studio :1234 ] [ llama.cpp :8080 ] |
|                                                                          |
|  Base URL: [ http://localhost:8000/v1                     ] [🔄 Reset]   |
|  API Key:  [ •••••••••••••••••••••                          ] 🛡️         |
|  [ 🔄 Test vLLM Connection ]                                             |
|  Active Model: [ meta-llama/Llama-3-8B-Instruct                     ▼ ]  |
+--------------------------------------------------------------------------+
|  📧 Local Python Mail Server (Port 1025)                      [ Stopped ]|
|  [ ▶️ Start Mail Server ]                                                 |
+--------------------------------------------------------------------------+
|  🌐 Internet Assist Engine                                               |
|  Search Provider: [ DuckDuckGo (Free & Privacy-focused)             ▼ ]  |
|  Max Results: [ 5 ] entries                                              |
+--------------------------------------------------------------------------+
```

### 2.3 Multi-Agent Workspace & A2A Stream Visualizer
```
+---------------------------------------------------------------------------------------------------------+
|  🌐 Multi-Agent Network Visualizer                                            Feature Flag: [ ENABLED ] |
+---------------------------------------------------------------------------------------------------------+
|  +---------------------+       +---------------------+       +---------------------+                   |
|  | 🎯 Agent Coordinator| ----> | 💻 Code Synthesis   | ----> | 🛡️ HITL Verifier    |                   |
|  | Status: DELEGATING  |       | Status: EXECUTING   |       | Status: AWAITING    |                   |
|  +---------------------+       +---------------------+       +---------------------+                   |
|             |                             |                             |                               |
|             v                             v                             v                               |
|  +---------------------+       +---------------------+       +---------------------+                   |
|  | 🔍 Research Agent   |       | 🧪 QA & Test Agent  |       | 👤 Custom User Agent|                   |
|  | Status: IDLE        |       | Status: THINKING    |       | Status: IDLE        |                   |
|  +---------------------+       +---------------------+       +---------------------+                   |
+---------------------------------------------------------------------------------------------------------+
|  💬 Real-Time A2A Stream & Protocol Inspector                                                           |
|  Filter: [ All Logs (14) ] [ Human ↔ Agent ] [ Agent ↔ Agent ] [ HITL Checkpoints ]                     |
|  +---------------------------------------------------------------------------------------------------+  |
|  | [10:14:02] [agent_coder -> agent_hitl] (HITL_REQUEST) "Subtask 2 Code Implementation Complete"     |  |
|  | [10:14:05] [human_operator -> agent_coordinator] (HITL_APPROVAL) "Approved with minor edits"     |  |
|  +---------------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------------+
```

---

## 🎯 3. End-to-End User Usage Scenarios

### Scenario A: Everyday AI Coding & Component Editing
1. **Goal**: Create or modify a React component or Express server route using natural language.
2. **Steps**:
   - Navigate to the **Workspace / Chat** tab.
   - Type your request in the chat prompt area (e.g. *"Add a user profile avatar dropdown with dark mode toggle"*).
   - Use `@` in the prompt to attach specific workspace files for exact context (e.g. `@src/components/Sidebar.tsx`).
   - Click **Send** or press `Enter`.
   - The AI Coder Agent analyzes the request, generates the modified TypeScript/JSX code, and applies surgical edits to the target file.
   - Inspect the live preview canvas or view diffs in the Code Viewer.

### Scenario B: Deep Technical Web & Documentation Research
1. **Goal**: Research an external API, library, or architectural pattern and save a research synthesis to disk.
2. **Steps**:
   - Toggle **Research Mode** in the chat interface or run a Research step in the Pipeline Builder.
   - Enter your research topic (e.g., *"Compare vLLM vs LM Studio performance for local code generation"*).
   - The agent executes internet searches using your selected search engine (DuckDuckGo, Google CSE, Bing, Brave, or Serper).
   - The agent fetches search result contents, synthesizes key findings with cited URLs, and saves the generated Markdown file to `/docs/research/research_summary.md`.

### Scenario C: Automatic System Documentation Generation
1. **Goal**: Generate up-to-date documentation for all codebase components and API routes.
2. **Steps**:
   - Click **Generate System Docs** in the top action bar.
   - The System Documentation Specialist Agent scans `src/`, `server.ts`, and `/docs`.
   - It drafts structured markdown documentation covering system architecture, component hierarchies, and API endpoints.
   - The newly generated markdown automatically opens in the **Documentation Viewer** tab for review.

### Scenario D: Automated Unit & Integration Test Suite Generation
1. **Goal**: Create comprehensive unit tests for modules lacking test coverage.
2. **Steps**:
   - Click **Generate Test Suite** in the top navigation bar.
   - The QA Agent scans the codebase for untested functions, hooks, and endpoints.
   - It writes TypeScript test files (`.test.ts` / `.test.tsx`) with assertions.
   - The agent invokes the built-in MCP `run_tests` tool to run the test suite and verify test pass rates.

### Scenario E: Visual Workflow Pipeline Execution
1. **Goal**: Run a automated multi-step development pipeline.
2. **Steps**:
   - Open the **Pipeline Builder** tab.
   - Click **Load Template** and select *Full Feature Pipeline* (Research -> Code -> Test -> Docs).
   - Configure step parameters and toggle **Require Human Approval** on sensitive steps.
   - Click **Execute Pipeline**.
   - Monitor real-time progress as each step transitions through `Pending` -> `In Progress` -> `Success`.
   - If a step requires approval, review the intermediate output and click **Approve & Resume**.

### Scenario F: Multi-Agent Orchestration with MCP Tools & HITL Verification
1. **Goal**: Leverage specialized agents operating via Model Context Protocol (MCP) and Agent-to-Agent (A2A) message channels.
2. **Steps**:
   - Open the **Multi-Agent Workspace** tab.
   - Verify that the Multi-Agent feature flag shows `ENABLED`.
   - Submit a complex request to the **Agent Coordinator**.
   - Watch the Agent Coordinator decompose the task and emit A2A delegation messages to Coder and Researcher agents.
   - When the Coder agent finishes, the output is intercepted by the **Human-in-the-Loop (HITL) Verifier**.
   - Review the proposed code in the inline editor, make optional manual edits, and click **Approve Output** or **Reject & Request Re-run**.

### Scenario G: Local Python SMTP Mail Server Operations (Port 1025)
1. **Goal**: Test email notifications locally without external SMTP services.
2. **Steps**:
   - Open **Settings** sidebar (`#model-selector`).
   - Locate the **Local Python Mail Server (Port 1025)** section.
   - Click **Start Mail Server**.
   - The backend launches a Python SMTP server bound to `127.0.0.1:1025`.
   - Applications sending mail to localhost:1025 will log incoming emails to the Mail Terminal log panel.

### Scenario H: Internet Search Assist Engine Setup
1. **Goal**: Configure web search assist for real-time web grounding.
2. **Steps**:
   - Open **Settings** sidebar.
   - Under **Internet Assist Engine**, choose your search provider:
     - **DuckDuckGo**: Zero setup required; privacy-focused free search.
     - **Google Custom Search (CSE)**: Provide Google API Key and Search Engine ID (CX).
     - **Bing**: Provide Bing Cognitive Services Key.
     - **Brave**: Provide Brave Search API Key.
     - **Serper**: Provide Serper API Key.
   - Set **Search Entry Count** (e.g. 5 results).
   - All research and agent web grounding calls will now route through your chosen provider.
