# Multi-Agent System (MCP & A2A Protocols)

The Multi-Agent System provides an enterprise-grade agent orchestration framework leveraging two key protocol standards:
1. **Model Context Protocol (MCP)**: Standardized protocol for exposing tools, database schemas, prompt templates, and filesystem resources to LLMs.
2. **Agent-to-Agent (A2A) Protocol**: Structured inter-agent communication standard supporting agent handoffs, request-response delegation, consensus negotiation, and telemetry broadcasts.

---

## 🤖 Registered Agent Roles

The multi-agent network consists of specialized autonomous agents:

| Agent Name | Role | Primary Responsibilities | MCP Tools Access |
| :--- | :--- | :--- | :--- |
| **Coordinator Agent** | Coordinator / Orchestrator | Decomposes high-level requirements into subtasks, delegates work via A2A protocol, and synthesizes final responses. | `read_file`, `list_files`, `run_workflow` |
| **Code Synthesis Agent** | Coder / Engineer | Implements features, refactors files, updates code blocks, enforces TypeScript safety. | `read_file`, `write_file`, `edit_file`, `execute_command` |
| **Web & Docs Research Agent** | Technical Researcher | Searches web/docs, synthesizes competitive benchmarks, evaluates APIs & packages. | `web_search`, `read_file`, `list_files` |
| **QA & Test Engineer Agent** | Quality Assurance Engineer | Generates unit tests, executes test suites, identifies regression failures and boundary errors. | `run_tests`, `read_file`, `execute_command` |
| **Human-in-the-Loop Verifier Agent** | HITL Verifier / Gatekeeper | Intercepts sub-task outputs, presents proposed changes for human review, records human approvals/edits, and directs subtask re-execution. | `read_file`, `list_files`, `read_workspace_logs` |

---

## 🌐 Agent Network Visualizer & Feature Flag Handling

The `AgentNetworkVisualizer` component (`src/components/multi_agent/AgentNetworkVisualizer.tsx`) provides a visual graph of active agents and their communication channels:

- **Feature Flag Validation**: Automatically queries system configuration (`ENABLE_MULTI_AGENT`). If disabled, a clear banner alerts the user and disables multi-agent dispatching.
- **Node Topology Graph**: Displays active agent cards, status indicators (`IDLE`, `THINKING`, `DELEGATING`, `EXECUTING`, `AWAITING_HITL`), role colors, and pulse pings when messages traverse the network.
- **Dynamic Agent Registration**: Allows users to dynamically define new custom agent roles, system prompts, and allowed MCP tools.

---

## 💬 Human ↔ Agent & Agent ↔ Agent Communication Logs

All inter-agent and human-agent communications are logged in real time:

- **Channel Classification**:
  - `Human ↔ Agent`: Direct human prompts, HITL verification feedback, and operator commands.
  - `Agent ↔ Agent`: A2A protocol handoffs, delegations, tool invocations, and subtask summaries.
  - `HITL Checkpoints`: Intercepted outputs awaiting human review or confirmation.
- **Log Stream Filters**: Filter log entries by `All Logs`, `Human ↔ Agent`, `Agent ↔ Agent`, or `HITL Checkpoints`.
- **Payload Inspector**: Expand any log entry to view raw JSON payloads, conversation IDs, timestamps, and message types.

---

## 🛑 Human-in-the-Loop (HITL) Gatekeeper Controls

The platform includes a Human-in-the-Loop checkpoint system:

1. **HITL Interception**: When enabled, subtask outputs produced by specialized agents (e.g. Coder, Researcher) are routed to the `agent_hitl_verifier`.
2. **Interactive Output Editing**: Human operators can review proposed code or text outputs in an inline editor and modify them prior to downstream agent consumption.
3. **Approve & Resume / Reject & Re-run**: Operators can approve the verified output or request an agent re-run with custom feedback notes.
4. **Direct Messaging**: Human operators can send direct messages or instructions to any selected agent in the network via the Direct Instruction panel.

---

## 📐 Agent-to-Agent (A2A) Communication Protocol

### Message Structure
Every A2A payload passed through the internal event bus strictly complies with the following JSON schema:

```json
{
  "id": "msg_9841203912",
  "senderId": "agent-architect",
  "receiverId": "agent-coder",
  "intent": "DELEGATE_TASK",
  "timestamp": 1785231267000,
  "payload": {
    "taskId": "task-01",
    "description": "Implement authentication middleware",
    "requiredFiles": ["src/middleware/auth.ts"]
  },
  "status": "DELIVERED"
}
```

### Supported Intent Enums
- `DELEGATE_TASK`: Primary agent hands off a specific task unit to a secondary agent.
- `REQUEST_DATA`: Agent requests specific file/tool outputs from another specialized agent.
- `RESPONSE_DATA`: Agent returns requested context or computed artifacts.
- `TASK_COMPLETED`: Sub-agent reports successful execution along with status metadata.
- `TASK_FAILED`: Sub-agent reports an unrecoverable failure with detailed stack traces.
- `BROADCAST_METRIC`: Agent broadcasts telemetry, resource usage, or model token consumption.

---

## 🛠️ Model Context Protocol (MCP) Integration

The platform includes a built-in MCP Server & Client implementation.

### Built-in MCP Tools

1. **`search_web`**: Executes real-time web searches for technical documentation or API reference guides.
2. **`write_code`**: Safely writes generated source code to specified relative target paths with syntax validation.
3. **`edit_file`**: Applies surgical block-level replacements to existing files.
4. **`run_test_suite`**: Executes local test commands and returns pass/fail statistics and error outputs.
5. **`read_codebase`**: Scans the project repository tree and returns structural file maps and dependencies.
6. **`fetch_url_content`**: Downloads and extracts clean markdown/text content from web URLs.
7. **`generate_markdown_docs`**: Formats structured technical docs and updates the `/docs` directory.

---

## 💻 Using the Multi-Agent Workspace UI

To open the Multi-Agent Workspace:
1. Click the **Multi-Agent System** tab in the main workspace header.
2. Choose from the four workspace tabs:
   - **Agent Network Topology**: Interactive node-graph showing connected agent nodes, active status badges, and communication links.
   - **A2A Message Inspector**: Real-time log of all transmitted A2A messages, filtered by sender/receiver or intent. Allows manual JSON payload injection to simulate inter-agent triggers.
   - **MCP Context Manager**: Interactive tool sandbox to inspect loaded MCP tool definitions, test direct tool invocation with mock JSON inputs, and view raw JSON responses.
   - **Multi-Agent Task Trace**: Step-by-step visual timeline tracking orchestrations, multi-agent handoffs, and execution latencies.
