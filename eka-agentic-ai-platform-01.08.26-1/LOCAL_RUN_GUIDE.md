# Eka Agentic AI Platform - Local Running & User Guide

Welcome to the **Eka Agentic AI Platform**, a production-grade, advanced AI-powered development workspace environment built with **React, Vite, Tailwind CSS, Express, and the modern @google/genai SDK**. 

This application allows you to explore full-cycle agentic workflows—including automated planning, interactive directory management, step-by-step execution, and self-correcting loop processes—using either state-of-the-art cloud LLMs (Google Gemini) or local offline model services (Ollama, OpenAI-compatible APIs).

---

## 🌟 Application Core Capabilities

The architecture is built on a custom Express backend serving as an agent compiler/runner, paired with a React single-page frontend.

1. **Virtual Workspace File System**
   * Live folder/file tree visualization with grid and list views.
   * Real-time file CRUD operations mapped directly to a local sandboxed directory (`/workspace`).
   * Clean, syntax-highlighted code editor with saving features.

2. **Multi-Model Support & Integration**
   * **Google Gemini API**: Direct native SDK connection using `@google/genai` (supporting `gemini-3.5-flash`, `gemini-3.1-pro-preview`, etc.).
   * **Ollama (Local Offline)**: Integration with local instances running on `http://localhost:11434` (supporting models like `llama3`, `mistral`, `codegemma`).
   * **OpenAI-Compatible APIs**: Support for OpenAI or third-party compatible API routes.

3. **Grounded Agentic Chat**
   * Natural language interface with interactive mode switching (Chat, Code, Research, Documentation, Testing).
   * Real-time **Google Search Grounding** toggle for web-retrieved source citations.

4. **Structured Multi-Step Agent Execution**
   * Automatically parses high-level prompts into sequential step-by-step development plans.
   * Prompts user for approval prior to irreversible actions (e.g., file deletions or terminal commands).
   * Handles multi-step execution visually on a pipeline flow chart.
   * **Self-Correction Loop**: If a test or step fails, the agent auto-triggers a secondary debugging prompt to self-correct code and re-evaluate.

---

## 🛠️ Prerequisites

To run this application locally, ensure your system has:
* **Node.js** (v18.0.0 or higher is recommended)
* **npm** (comes with Node), **yarn**, or **bun**
* *Optional but highly recommended*: A Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
* *Optional for local offline LLMs*: [Ollama](https://ollama.com/) installed and running locally.

---

## 📂 Installation

1. **Download or Clone the codebase** into a local directory:
   ```bash
   git clone <your-repo-link> ai-coding-agent-studio
   cd ai-coding-agent-studio
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

---

## 🔒 Environment Configuration

Create a `.env` file in the root of your project. You can copy the template from `.env.example`:

```bash
cp .env.example .env
```

Open `.env` and fill in the values:

```env
# GEMINI_API_KEY: Paste your Gemini API key from Google AI Studio.
# Required for default Gemini cloud models and structured schema workflows.
GEMINI_API_KEY="AIzaSyYourActualAPIKeyHere"

# APP_URL: The hosting url. For local development, this is typically http://localhost:3000
APP_URL="http://localhost:3000"
```

---

## 🚀 How to Run Locally

Eka Agentic AI Platform includes three pre-configured scripts in `package.json` for running, compiling, and testing the application.

### 1. Run in Development Mode
Runs the local Express API backend with live-reloaded Vite frontend middleware serving on port `3000`.

```bash
npm run dev
```
* Once running, open your web browser and navigate to **`http://localhost:3000`**.

### 2. Build for Production
Compiles the React frontend using Vite, and bundles the Express server into a standalone self-contained CJS bundle in `dist/server.cjs` using esbuild. This ensures rapid cold starts and bypasses node module path mismatches.

```bash
npm run build
```

### 3. Run Production Build
Launches the high-performance compiled server in production mode.

```bash
npm start
```

---

## 📖 User & Workflow Guide

### Step 1: Set Up Your Model Provider
When you first open the app, look at the **LLM Server Connection Panel** in the sidebar:
1. Choose your provider: **Gemini**, **Ollama (Local)**, or **OpenAI**.
2. For **Gemini**, verify your API key is populated.
3. For **Ollama**, verify your Ollama daemon is running locally (`ollama serve`) and press **Check Connection** to discover your locally downloaded models.
4. Select your preferred target model from the dropdown.

### Step 2: Explore the Workspace & Directory Viewer
The middle section of the screen displays either your current active code file or the interactive **Directory Viewer**.
* **Folder Navigation**: Select any directory in the sidebar tree or click folders in the directory explorer to browse contents.
* **Layout Switcher**: Switch between **List View** (for deep details) and **Grid View** (for visual asset browsing) using the grid/list icons on the directory toolbar.
* **File Operations**: Click `+ File` or `+ Folder` to instantly create new resources in the virtual sandbox workspace, or press the trashcan icon to delete them.
* **Editing**: Click any file to view and edit its code in real-time. Make modifications and click **Save File** to persist back to disk.

### Step 3: Launch an Agent Workflow
1. Select an **Agent Mode** from the chat header panel (e.g., *Code* for development, *Testing* for building suites).
2. Type a high-level goal in the chat box. (e.g., *"Create an analytical statistics library in Python under src/math_stats.py, then write and run comprehensive unit tests under tests/test_stats.py."*)
3. Toggle the **Google Search** grounding button if your task requires web knowledge (like external API specifications).
4. Press **Submit**.

### Step 4: The Execution Pipeline & Prior Approval
1. **Planning Phase**: The agent uses Gemini's structured output schema to create an interactive checklist pipeline.
2. **Prior Approval Gate**: If a step is deemed sensitive (e.g. running high-risk commands or modifying primary structures), the workflow pauses and renders a prominent approval dialogue. Click **Approve & Execute** to allow the step, or **Deny** to request revision.
3. **Checklist Progress**: Watch steps turn green as they complete. Live server-side run logs are displayed in terminal drawers inside each step.
4. **Self-Correction**: If a test step fails, the agent automatically captures the error stack, feeds it into Gemini, applies a bugfix patch, and retries the test.

---

## 🛠️ Project Structure Details

```
├── .env.example              # Configuration template for environments
├── package.json              # App dependencies & run scripts
├── server.ts                 # Full Express + Vite dev server entry point
├── vite.config.ts            # Vite compiler configuration
├── workspace/                # Virtual sandbox where the Agent writes code
├── src/
│   ├── main.tsx              # React mounting entry point
│   ├── App.tsx               # Primary UI hub and State coordinator
│   ├── types.ts              # Shared TypeScript definitions
│   ├── index.css             # Tailwind 4.0 global styles
│   └── components/
│       ├── Sidebar.tsx       # File tree and connection configurations
│       ├── DirectoryViewer.tsx # Visual workspace file browser
│       ├── AgentFlowGraph.tsx# Live execution checklist & pipeline
│       ├── CodeViewer.tsx    # Live editor and code visualizer
│       └── ChatPanel.tsx     # Chat logs and workflow dispatcher
```
