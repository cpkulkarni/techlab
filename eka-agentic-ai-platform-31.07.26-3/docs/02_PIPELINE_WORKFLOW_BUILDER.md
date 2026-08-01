# Pipeline & Workflow Builder

The **Pipeline & Workflow Builder** enables developers to visually design, order, configure, and execute multi-stage AI agent automation workflows.

---

## 🏗️ Core Pipeline Concepts

A **Pipeline** is a directed sequential or branching workflow composed of individual **Pipeline Steps**. Each step delegates execution to an AI agent or system tool.

### Supported Step Types

1. **Research Step (`RESEARCH`)**
   - **Agent**: Technical Researcher Agent
   - **Function**: Performs web searches and gathers API references or code context.
   - **Inputs**: Query topic, domain filters.

2. **Code Generation Step (`CODE`)**
   - **Agent**: Software Engineer Agent
   - **Function**: Writes or edits TypeScript/React component files according to specifications.
   - **Inputs**: Target file path, prompt description, reference files.

3. **Testing Step (`TEST`)**
   - **Agent**: QA Engineer Agent
   - **Function**: Formulates unit tests using Vitest/Jest and verifies code correctness.
   - **Inputs**: Source files to test, assertion criteria.

4. **Documentation Step (`DOCS`)**
   - **Agent**: Documentation Specialist Agent
   - **Function**: Produces comprehensive Markdown documentation files in `/docs`.
   - **Inputs**: Topic name, file list, style guide.

5. **Command Execution Step (`COMMAND`)**
   - **Agent**: System Runner
   - **Function**: Executes shell scripts or build checks (`npm run lint`, `npm run build`).

---

## 🎛️ Pipeline Configuration & Human-in-the-Loop

### Step Approval Settings
Each step can be configured with **Human Approval Required (`requireApproval: true`)**:
- When enabled, the workflow pauses upon reaching this step.
- The UI highlights the step with a yellow **Awaiting Approval** state.
- Developers can inspect intermediate step outputs, edit configuration inputs, and click **Approve & Resume** or **Reject Step**.

### Reordering & Adding Steps
- Click **+ Add Step** in the Pipeline Builder toolbar.
- Drag and drop step cards to adjust the execution sequence.
- Toggle steps between Enabled/Disabled states without deleting configuration parameters.

---

## 🚀 Running a Pipeline

1. Select **Pipeline Builder** in the top workspace tabs.
2. Load a template (e.g., *Full Feature Pipeline*, *Documentation Sync*, *Test Coverage Suite*) or build a custom step sequence.
3. Click **Execute Pipeline**.
4. Monitor active progress indicators on each step card:
   - 🟡 **Pending**: Step in queue.
   - 🔵 **In Progress**: Step currently running via SSE execution engine.
   - 🟢 **Success**: Step complete with output logs.
   - 🔴 **Failed**: Execution error with stack trace view.
   - 🟠 **Paused**: Awaiting human developer approval.
