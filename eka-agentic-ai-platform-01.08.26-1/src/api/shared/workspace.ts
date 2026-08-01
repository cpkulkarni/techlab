/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';
import fs from 'fs/promises';

// Mutable workspace root — modified via the root-path API
let WORKSPACE_DIR = process.cwd();

export function getWorkspaceDir(): string {
  return WORKSPACE_DIR;
}

export function setWorkspaceDir(newPath: string): void {
  WORKSPACE_DIR = newPath;
}

// ── Source-code virtual folder ────────────────────────────────────────────────
// These are the platform's own source files. They are grouped under a virtual
// "source-code" folder in the workspace tree and marked locked so the UI
// prevents accidental edits or deletions. Nothing moves on disk.
export const SOURCE_CODE_VIRTUAL_DIR = 'source-code';

// File/dir names at the workspace root that belong to the locked group
const SOURCE_CODE_MEMBERS = new Set([
  'src', 'server.ts', 'smtp_server.py',
  'vite.config.ts', 'tsconfig.json',
  'package.json', 'index.html',
  'metadata.json', 'bun.lock', 'package-lock.json',
]);

// Returns true if a workspace-relative path is inside the locked source-code group.
export function isSourceCodePath(relativePath: string): boolean {
  const normalized = relativePath.replace(/\\/g, '/');
  if (normalized === SOURCE_CODE_VIRTUAL_DIR) return true;
  if (normalized.startsWith(SOURCE_CODE_VIRTUAL_DIR + '/')) return true;
  return false;
}

// Recursive directory tree builder (used by workspace and agent routes)
export async function buildTree(dirPath: string, rootPath?: string, _isTopLevel = true, _isLocked = false): Promise<any[]> {
  const root = rootPath ?? WORKSPACE_DIR;
  const items = await fs.readdir(dirPath, { withFileTypes: true });
  const result: any[] = [];
  const sourceCodeChildren: any[] = [];

  for (const item of items) {
    if (
      item.name === 'node_modules' ||
      item.name === 'dist' ||
      item.name === '.git' ||
      item.name === '.npm' ||
      item.name === '.cache' ||
      item.name === 'workspace-bundle' ||
      item.name.startsWith('.')
    ) continue;

    const fullPath = path.join(dirPath, item.name);
    const relativePath = path.relative(root, fullPath);

    // At the top level, collect source-code members into the virtual locked folder
    if (_isTopLevel && SOURCE_CODE_MEMBERS.has(item.name)) {
      if (item.isDirectory()) {
        const children = await buildTree(fullPath, root, false, true);
        sourceCodeChildren.push({
          name: item.name,
          path: `${SOURCE_CODE_VIRTUAL_DIR}/${item.name}`,
          type: 'directory',
          locked: true,
          children
        });
      } else {
        const stats = await fs.stat(fullPath);
        sourceCodeChildren.push({
          name: item.name,
          path: `${SOURCE_CODE_VIRTUAL_DIR}/${item.name}`,
          type: 'file',
          locked: true,
          size: stats.size
        });
      }
      continue; // skip adding it as a top-level node
    }

    if (item.isDirectory()) {
      const children = await buildTree(fullPath, root, false, _isLocked);
      const node: any = {
        name: item.name,
        path: _isLocked ? `${SOURCE_CODE_VIRTUAL_DIR}/${relativePath}` : relativePath,
        type: 'directory',
        children
      };
      if (_isLocked) node.locked = true;
      result.push(node);
    } else {
      const stats = await fs.stat(fullPath);
      const node: any = {
        name: item.name,
        path: _isLocked ? `${SOURCE_CODE_VIRTUAL_DIR}/${relativePath}` : relativePath,
        type: 'file',
        size: stats.size
      };
      if (_isLocked) node.locked = true;
      result.push(node);
    }
  }

  // Inject the virtual source-code folder at the top level
  if (_isTopLevel && sourceCodeChildren.length > 0) {
    result.push({
      name: 'source-code',
      path: SOURCE_CODE_VIRTUAL_DIR,
      type: 'directory',
      locked: true,
      children: sourceCodeChildren.sort((a, b) => {
        if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
        return a.name.localeCompare(b.name);
      })
    });
  }

  return result.sort((a, b) => {
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}

// Recursively collect all readable text files in a directory
export async function collectDirFiles(dirPath: string, rootPath: string): Promise<{ path: string; content: string }[]> {
  const results: { path: string; content: string }[] = [];
  const skipDirs = new Set(['node_modules', '.git', 'dist', '.npm', '.cache', '.next', 'build', 'out']);
  const textExts = new Set(['ts', 'tsx', 'js', 'jsx', 'py', 'css', 'html', 'json', 'md', 'txt', 'yaml', 'yml', 'sh', 'env', 'toml', 'ini', 'cfg', 'xml', 'sql']);

  async function walk(dir: string) {
    let entries: any[];
    try { entries = await fs.readdir(dir, { withFileTypes: true }); } catch { return; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!skipDirs.has(entry.name) && !entry.name.startsWith('.')) await walk(full);
      } else {
        const ext = entry.name.split('.').pop()?.toLowerCase() || '';
        try {
          const stat = await fs.stat(full);
          if (stat.size > 500 * 1024) continue; // skip files > 500 KB
          if (textExts.has(ext)) {
            const content = await fs.readFile(full, 'utf8');
            results.push({ path: path.relative(rootPath, full), content });
          }
        } catch { /* skip unreadable files */ }
      }
    }
  }

  await walk(dirPath);
  return results;
}
