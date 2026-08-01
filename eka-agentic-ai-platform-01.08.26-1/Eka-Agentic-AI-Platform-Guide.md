# **Eka Agentic AI Platform — Technical User Guide**

Welcome to the technical guide for the **Eka Agentic AI Platform**. This document provides an exhaustive overview of the platform's architecture, key capabilities, visual workflow connectors, setup procedures, and developer guidelines to help you run, test, and further enhance the application.

## **1\. Executive Summary & Core Platform Vision**

The **Eka Agentic AI Platform** is an advanced, production-ready AI development workspace environment. Built with React 19, Vite, Tailwind CSS, Express, and the official Google GenAI SDK (@google/genai), the platform bridges autonomous LLM reasoning with interactive developer tooling. It empowers engineers to design, build, test, and execute complex agentic workflows, multi-node RAG pipelines, and automated software tasks either in online cloud modes (powered by Google Gemini) or fully offline/local environments (via Ollama or OpenAI-compatible local model servers).

## **2\. Comprehensive Feature Breakdown**

### **2.1 Interactive Chat Panel & Autonomous Planner**

> * **Multi-Agent Operating Modes:** Switch seamlessly between specialized agent personas (e.g., General Chat, Autonomous Planner, Code Architect, Research Analyst, Search Grounded).  
> * **Human-in-the-Loop (HITL) Controls:** Review and approve proposed multi-step execution plans before any system modification or tool action is taken. Individual steps can be selectively approved, rejected, or re-queued.  
> * **Google Search Grounding:** Dynamic web search integration providing real-time data retrieval for up-to-date analysis and fact-checking.

### **2.2 Agent Flow Graph Visualizer**

> * **Interactive DAG Visualization:** Real-time graphical execution graph tracking pending, active, approved, completed, or failed plan steps.  
> * **Execution State Monitoring:** Offers full transparency into intermediate reasoning steps, tool call parameters, and self-correcting loop iterations.

### **2.3 Pipeline Builder (Visual Node Connector Studio)**

A LangFlow-inspired drag-and-drop workflow canvas allowing users to visually compose complex agentic and data pipelines across a wide spectrum of node connectors:

| Node Category | Connector / Node Type | Technical Functionality   |
| :---- | :---- | :---- |
| **Core Logic** | Input / Output | Entry points for raw prompt payloads and structured output sinks. |
| **Control Flow** | Loop / HumanIntervention | Iterative logic execution loops and manual approval gates. |
| **Data Access** | DB Read / DB Write | Relational database connections for querying and persisting data. |
| **Integrations** | API Call | HTTP client REST node for fetching or posting data to external services. |
| **Compute** | Code Execution / Test Runner | In-sandbox code evaluation and automated unit test execution. |
| **RAG & Knowledge** | Vector DB / Elastic Search / Local Files / Search Engine | Knowledge retrieval nodes enabling semantic vector search, full-text index queries, and local file reading. |
| **AI Reasoning** | LLM Node | Generative AI prompt processing node supporting customizable model parameters and temperature. |
| **Communication** | Email Send / Email Receive | Automated outbound email dispatch and inbound message parsing. |

### **2.4 Code Viewer & In-Browser Editor**

> * **Full Syntax Highlighting & Navigation:** Clean line-numbering, indent guide indicators, and direct jump-to-line navigation.  
> * **In-Place Disk Persistence:** Toggle edit mode to modify files directly in the browser and save updates back to the underlying file system.  
> * **Code Actions:** Quick copy-to-clipboard, raw inspection, and side-by-side edit previews.

### **2.5 Documents & Research Viewer**

> * **Rendered Output Inspection:** Built-in viewer for structured research documents, generated Markdown notes, research reports, and pipeline output files.  
> * **Clean HTML/Markdown Formatting:** Supports embedded headers, tables, code fences, and formatted lists without manual export.

### **2.6 Directory Viewer & Workspace Manager**

> * **Flexible Workspace Navigation:** Grid and List view modes for exploring files and subdirectories.  
> * **File Management:** Direct file/folder creation, deletion, tree navigation, and server root directory switching.  
> * **Workspace Export:** One-click ZIP archiving for downloading entire project directories.

### **2.7 Model Selector & Infrastructure Controls**

> * **Flexible Model Provider Switching:** Supports Google Gemini API (Gemini 2.5/3.0), local Ollama endpoints, or custom OpenAI-compatible server URLs.  
> * **Embedded SMTP Mail Server Control:** In-app start/stop toggles for the background Python SMTP mail server running on port 1025\.

## **3\. System Architecture**

The application follows a clean monorepo architecture combining a high-performance Express backend with a modern React SPA client:  
`┌────────────────────────────────────────────────────────┐`  
`│                      Web Browser                       │`  
`│  (React 19 SPA + Vite + Tailwind CSS v4 + Framer)      │`  
`└───────────────────────────┬────────────────────────────┘`  
                            `│ REST / JSON API`  
                            `▼`  
`┌────────────────────────────────────────────────────────┐`  
`│                   Express Backend                      │`  
`│ (Workspace REST APIs, File System Sync, Exec Engine)  │`  
`└──────────────┬──────────────────────────┬──────────────┘`  
               `│                          │`  
               `▼                          ▼`  
`┌──────────────────────────────┐ ┌───────────────────────┐`  
`│ @google/genai & Model SDKs  │ │ Python SMTP Server    │`  
`│ (Cloud & Local LLM Services) │ │ (Background Port 1025)│`  
`└──────────────────────────────┘ └───────────────────────┘`

## **4\. Dependencies & Tech Stack**

| Package Name | Version | Role / Purpose   |
| :---- | :---- | :---- |
| @google/genai | ^2.4.0 | Official Google GenAI SDK for Gemini models & web search grounding. |
| react / react-dom | ^19.0.1 | Core UI component library for building modern reactive interfaces. |
| vite | ^6.2.3 | Next-generation frontend build tool and dev server. |
| @tailwindcss/vite | ^4.1.14 | Utility-first CSS styling engine with zero configuration setup. |
| express | ^4.21.2 | Node.js web framework handling local file systems and API routes. |
| lucide-react | ^0.546.0 | Comprehensive iconography set across all workspace components. |
| motion | ^12.23.24 | High-performance animation engine for fluid tab transitions and UI. |
| nodemailer | ^6.9.14 | Node.js email sending library configured for local SMTP testing. |
| tsx | ^4.21.0 | TypeScript Execute CLI for seamless Node server development. |

## **5\. Installation & Local Setup Guide**

> 1. **Clone the Repository:**  
>    `git clone <repository-url>`  
>    `cd eka-agentic-ai-platform`  
> 2. **Install Dependencies:**  
>    `npm install`  
> 3. **Configure Environment Variables:**  
>    Copy .env.example to .env and insert your API credentials:  
>    `GEMINI_API_KEY=your_google_gemini_api_key_here`  
>    `PORT=3000`  
> 4. **Start Development Server:**  
>    `npm run dev`  
>    This starts both the Express API backend and Vite client concurrently at http://localhost:3000.

## **6\. External Services & Mail Integration**

The platform includes an embedded standalone Python SMTP server (smtp\_server.py) designed for offline testing of automated email dispatch and reception within pipeline workflows without requiring third-party credentials. The server listens locally on 127.0.0.1:1025 and captures all outbound messages into app-output/emails/.

## **7\. Testing & Development Guidelines**

> * **Excluded Directories:** Ignore generated output directories starting with app-\*, build artifacts, logs, and node\_modules during application commits.  
> * **Maintenance:** When adding new visual workflow nodes in src/components/WorkflowBuilder.tsx, update corresponding interfaces in src/types.ts to maintain strict type safety across the application.