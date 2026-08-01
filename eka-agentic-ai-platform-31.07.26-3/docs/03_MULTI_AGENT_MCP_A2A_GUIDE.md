# 🤖 Multi-Agent System, MCP Tools & A2A Protocol Engine

The **Multi-Agent System** enables specialized autonomous agents to collaborate, exchange subtasks, invoke external tools via Model Context Protocol (MCP), and submit work outputs for Human-in-the-Loop (HITL) verification.

---

## 🏛️ 1. Multi-Agent System Architecture

```
                                    +------------------------------+
                                    |     HUMAN OPERATOR / USER    |
                                    +--------------+---------------+
                                                   |
                                                   v  (human_to_agent)
                                    +------------------------------+
                                    |    🎯 COORDINATOR AGENT      |
                                    | (Agent Network Orchestrator) |
                                    +--------------+---------------+
                                                   |
                     +-----------------------------+-----------------------------+
                     |  A2A Delegation Bus         | A2A Research Task           | A2A Testing Request
                     v                             v                             v
        +--------------------------+  +--------------------------+  +--------------------------+
        |  💻 CODE SYNTHESIS AGENT |  | 🔍 RESEARCHER AGENT     |  | 🧪 QA & TEST ENGINEER    |
        |  MCP Tools: write_file,  |  |  MCP Tools: web_search,  |  |  MCP Tools: run_tests,   |
        |  edit_file, execute_cmd  |  |  read_file, list_files   |  |  read_file, execute_cmd  |
        +------------+-------------+  +------------+-------------+  +------------+-------------+
                     |                             |                             |
                     +-----------------------------+-----------------------------+
                                                   | (Subtask Outputs)
                                                   v
                                    +------------------------------+
                                    |  🛡️ HITL VERIFIER GATEKEEPER |
                                    |  (Interactive Output Review) |
                                    +--------------+---------------+
                                                   |
                                    +--------------+---------------+
                                    | [ Approve ]    [ Modify/Edit]|
                                    +--------------+---------------+
                                                   |
                                                   v
                                     Final Task State Update / Disk Sync
```

---

## 👥 2. Active Agent Roles & Capabilities

| Agent ID | Display Name | Primary Role | Assigned MCP Tools |
| :--- | :--- | :--- | :--- |
| `agent_coordinator` | **Agent Coordinator** | Decomposes complex prompt requirements into subtasks, delegates work via A2A protocol, synthesizes final system responses. | `read_file`, `list_files`, `run_workflow` |
| `agent_coder` | **Code Synthesis Agent** | Generates, refactors, and edits TypeScript/React components and backend endpoints. | `read_file`, `write_file`, `edit_file`, `execute_command` |
| `agent_researcher` | **Web & Docs Research Agent** | Executes multi-hop internet searches, extracts web documentation, synthesizes technical briefs. | `web_search`, `read_file`, `list_files` |
| `agent_qa` | **QA & Test Engineer Agent** | Generates unit tests, executes test suites, identifies syntax/runtime errors. | `run_tests`, `read_file`, `execute_command` |
| `agent_hitl_verifier` | **Human-in-the-Loop Verifier** | Gatekeeper intercepting sub-task outputs for operator review and modification prior to disk commit. | `read_file`, `list_files`, `read_workspace_logs` |

---

## 🧰 3. Model Context Protocol (MCP) Tool Registry

The platform exposes standardized **MCP Tools** accessible to agents during task execution:

```typescript
export interface MCPToolDefinition {
  name: string;
  description: string;
  category: 'code' | 'files' | 'search' | 'terminal' | 'testing' | 'workflow';
  inputSchema: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
  securityScope: 'workspace_read' | 'workspace_write' | 'external_network' | 'system_exec';
}
```

### Registered MCP Tools
1. **`read_file`**: Read file contents from workspace (`path`).
2. **`write_file`**: Create a new file in workspace (`path`, `content`).
3. **`edit_file`**: Apply surgical replacement chunk to an existing file (`path`, `targetContent`, `replacementContent`).
4. **`list_files`**: List directory entries recursively (`directoryPath`).
5. **`web_search`**: Search the web using configured search engine (`query`, `maxResults`).
6. **`execute_command`**: Execute background shell commands (`command`, `cwd`).
7. **`run_tests`**: Execute Vitest/Jest test suites (`testFilePath`).

---

## 💬 4. Agent-to-Agent (A2A) Message Channels & Protocol

Agents communicate over structured **A2A Message Channels**:

```typescript
export interface A2AMessage {
  id: string;
  sender_id: string;
  recipient_id: string | 'broadcast';
  conversation_id: string;
  message_type: 'task_delegation' | 'subtask_result' | 'tool_invocation' | 'hitl_request' | 'hitl_approval' | 'hitl_rejection' | 'human_direct';
  payload: {
    task?: string;
    subtasks?: string[];
    output?: string;
    tool_name?: string;
    tool_args?: any;
    approved?: boolean;
    modified_output?: string;
    human_feedback?: string;
  };
  timestamp: string;
  channel?: 'human_agent' | 'human_to_agent' | 'agent_to_human' | 'agent_to_agent' | 'hitl';
}
```

### Channel Classification
- **`human_agent`**: Communications between human operator and agents (e.g. direct instruction, status requests).
- **`agent_to_agent`**: Inter-agent task delegations, tool execution results, and peer feedback.
- **`hitl`**: Intercepted subtask outputs awaiting human review or confirmation.

---

## 🛡️ 5. Human-in-the-Loop (HITL) Gatekeeper Controls

The HITL Verifier ensures human operators retain control over agent actions:

1. **Interception**: When a subtask output is generated (e.g. Coder Agent edits code or Researcher drafts a document), a `hitl_request` message is emitted.
2. **Review Panel**: The UI highlights the pending output in the **HITL Verification Panel**.
3. **Inline Output Editor**: The operator can directly edit the proposed code or text inside a live Monaco/textarea editor.
4. **Action**:
   - **Approve Output**: Submits `hitl_approval` message with modified output and resumes downstream workflows.
   - **Reject & Request Re-run**: Submits `hitl_rejection` message with feedback notes to prompt agent re-execution.
5. **Direct Instruction**: Operators can send direct instructions to any target agent via the **Direct Agent Instruction** panel.
