# AI Agent Modes & Interactive Tools

The platform provides dedicated task-focused AI modes accessible via the left sidebar or quick action toolbars.

---

## 💬 1. Conversational Chat Mode

The **Conversational Chat** interface allows natural language interaction with Gemini-powered coding agents.

### Features
- **Code Generation & Editing**: Instruct the agent to build new React components, backend Express endpoints, or CSS styling updates.
- **Context Injection**: Use `@filename` in prompts to inject specific workspace files into the agent's context window.
- **Model Selector**: Switch between Gemini 2.5 Flash, Gemini 2.5 Pro, or Gemini Flash Thinking depending on task complexity.

---

## 🔍 2. Deep Technical Research Mode

Located under the **Research Mode** toggle:
- Performs multi-hop web searches and extracts content from documentation sites.
- Generates structured research syntheses with cited source URLs.
- Automatically saves research briefs into the `/docs/research` directory for immediate developer review.

---

## 📑 3. Automated System Documentation Generator

Click **Generate System Docs** in the primary action bar:
- Analyzes the workspace file structure, Express route definitions, and React component hierarchies.
- Produces markdown documentation files organized by feature module.
- Auto-opens generated markdown files inside the **Documentation Viewer** tab.

---

## 🧪 4. Unit & Integration Test Suite Generator

Click **Generate Test Suite** in the top workspace controls:
- Scans `src/` modules for utility functions, hooks, and Express handlers lacking test coverage.
- Formulates TypeScript test files (`.test.ts` / `.test.tsx`) using standard assertions.
- Integrates directly with the built-in MCP `run_test_suite` tool to verify test pass rates.

---

## 🛑 5. Interrupt & Stop Controls

When an agent execution or pipeline stream is active:
- A red **Stop Generation / Abort Task** button appears in the bottom status bar and chat input header.
- Clicking **Stop** immediately signals the abort controller on the backend, terminating server-sent event (SSE) streams and releasing system resources safely.
