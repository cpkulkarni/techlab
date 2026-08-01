# Security Architecture & Operational Security Guide

This document outlines the security architecture, allowed capabilities, operational boundaries, misuse vectors, and defense mechanisms implemented in the application workspace.

---

## 1. What Is Allowed (Permitted System Capabilities)

The platform is designed to serve as a local/developer workflow automation and multi-agent development environment.

- **Workspace File Management**: Reading, creating, editing, and managing files and directories **strictly inside** the designated workspace directory.
- **Git Control**: Checking repository status, staging, creating commits (with explicit user approval), and managing git branches within the workspace.
- **Controlled Command & Code Execution**:
  - Running developer commands (such as npm builds, tests, or scripts) within the workspace boundary or in an isolated Docker sandbox container.
  - Executing user-configured agent workflows and background scheduled tasks targeting workspace assets.
- **LLM Code Audits & Reviews**: Automated static code review and vulnerability scanning via Gemini API integrations.
- **System Documentation**: Generating system documentation and architecture guides inside the `/docs` workspace folder.

---

## 2. Implemented Security Controls & Safeguards

The platform implements multi-layered security controls to protect the underlying host and environment:

### A. Workspace Boundary & Path Traversal Prevention
- **Strict Boundary Assertion**: All file system operations use `assertWorkspaceBoundary(...)` (`/src/api/shared/workspaceGuard.ts`).
- **Canonicalization**: Paths are normalized and resolved against `getWorkspaceDir()`. Any attempt to access paths outside the workspace boundary (e.g. `../../etc/passwd` or system directories) throws a `Security Violation` exception with HTTP `403`.

### B. Core Source Code Locking
- **Virtual Folder Lock**: Platform core code (`src`, `server.ts`, `package.json`, `metadata.json`, etc.) is tagged as `locked` under the `source-code` group (`/src/api/shared/workspace.ts`).
- **Modification Protections**: API endpoints (`/api/workspace/file`, `/api/workspace/folder`) explicitly reject file/folder edits or deletions targeting locked source files.

### C. Explicit Confirmation & Human-in-the-Loop Gates
- **Sensitive Command Execution**: Sandbox executions requiring elevated permissions (`requiresExplicitApproval`) demand explicit `approved: true` flags before spawning sub-processes (`/src/api/routes/sandbox.ts`).
- **Git Commit Approvals**: Creating commits via the API requires explicit confirmation flags (`/src/api/routes/git.ts`).

### D. Isolated Sandbox Execution Options
- **Docker Container Isolation**: Commands can be routed through an isolated node/alpine Docker container (`docker run --rm -v "${workspaceRoot}:/workspace" ...`) to prevent state pollution or host degradation (`/src/api/routes/sandbox.ts`).

---

## 3. Potential Misuse Vectors & How To Prevent Them

| Misuse Vector | Potential Impact | Prevention & Defense Strategy |
| :--- | :--- | :--- |
| **Directory Traversal** | Accessing sensitive host files outside workspace (`/etc/passwd`, system environment) | Enforced by `assertWorkspaceBoundary` on all API path params. Never disable path normalization. |
| **Arbitrary Code Execution** | Running untrusted third-party scripts or malicious payloads | Always run untrusted scripts inside the Docker sandbox mode. Require explicit user confirmation for high-risk operations. |
| **Core Source Tampering** | Modifying platform routing or bypassing safety checks | Enforced by `isSourceCodePath(...)` check on write/delete APIs. Ensure source files remain locked. |
| **Credential & API Key Exposure** | Leaking server secrets or API keys to client browsers | Keep all secret API keys (`GEMINI_API_KEY`) strictly server-side. Never prefix backend secrets with `VITE_`. |

---

## 4. Security Best Practices for Developers & Users

1. **Verify Workspace Roots**: Always ensure the workspace directory points to an isolated directory intended for project development.
2. **Review Scheduled Tasks**: Regularly check the Scheduler panel to ensure background jobs only execute intended workspace scripts.
3. **Use Containerized Mode for External Code**: Enable Docker sandbox mode whenever executing unverified third-party libraries or scripts.
4. **Maintain Environmental Isolation**: Keep sensitive keys declared in `.env.example` without committing actual credentials or secrets to version control.
