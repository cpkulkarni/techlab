/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';

/**
 * Validates that a target file path resolves strictly within the workspace directory.
 * Prevents directory traversal attacks (e.g. `../../etc/passwd` or absolute system paths).
 */
export function validateWorkspacePath(targetPath: string, workspaceRoot?: string): { safePath: string; isWithin: boolean } {
  const root = path.resolve(workspaceRoot || process.cwd());
  
  // Normalize and resolve full absolute path
  let absoluteTarget = targetPath;
  if (!path.isAbsolute(targetPath)) {
    absoluteTarget = path.resolve(root, targetPath);
  } else {
    absoluteTarget = path.resolve(targetPath);
  }

  // Ensure resolved path starts with the root directory path
  const isWithin = absoluteTarget === root || absoluteTarget.startsWith(root + path.sep);

  return {
    safePath: absoluteTarget,
    isWithin,
  };
}

/**
 * Throws a Security Error if the path attempts to escape the workspace boundary.
 */
export function assertWorkspaceBoundary(targetPath: string, workspaceRoot?: string): string {
  const { safePath, isWithin } = validateWorkspacePath(targetPath, workspaceRoot);
  if (!isWithin) {
    throw new Error(`Security Violation: Access denied for path outside workspace boundary (${targetPath}).`);
  }
  return safePath;
}
