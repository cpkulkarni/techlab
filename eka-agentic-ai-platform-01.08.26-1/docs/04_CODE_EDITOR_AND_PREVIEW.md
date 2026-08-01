# Workspace Code Editor & File Synchronization

The platform features an integrated code workspace with live syntax highlighting, file tree navigation, and native file system synchronization capabilities.

---

## 📂 File Explorer & Code Viewer

- **File Tree**: Displays all workspace directories (`src/`, `server.ts`, `docs/`, `public/`).
- **Syntax Highlighting**: Supports TypeScript, JSX/TSX, JSON, CSS, Markdown, and YAML.
- **Diff Viewer**: Displays side-by-side or inline git diffs whenever an AI agent proposes code edits.

---

## 🔄 Native Local File System API Sync

The editor includes native support for the W3C File System Access API:
1. Click **Connect Local Folder** in the top file toolbar.
2. Grant read/write permissions to a directory on your local machine.
3. Edits made by the AI agent in the web workspace are synchronized directly back to your local files on disk in real time.

---

## 🩺 Real-Time Diagnostics & Terminal Logs

### Diagnostics Tab
- **TypeScript Errors**: Monitors real-time compilation errors across the workspace.
- **Linter Warnings**: Displays ESLint and formatting issues.
- **Click-to-Fix**: Click any diagnostic item to jump directly to the target line in the code editor or submit an auto-fix prompt to the AI Coder Agent.

### Terminal & Server Logs
- Displays Express server console output, HTTP request logs, and background worker task statuses.
