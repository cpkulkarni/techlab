/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import { exec, execSync } from 'child_process';
import path from 'path';
import { getWorkspaceDir } from '../shared/workspace.js';
import { assertWorkspaceBoundary } from '../shared/workspaceGuard.js';

const router = Router();

// GET /api/sandbox/docker-status - Check if local Docker daemon is running
router.get('/docker-status', async (req, res) => {
  try {
    const versionOutput = execSync('docker --version', { encoding: 'utf8', timeout: 3000 }).trim();
    execSync('docker info', { encoding: 'utf8', timeout: 3000 });
    return res.json({
      success: true,
      isDockerAvailable: true,
      dockerVersion: versionOutput,
      containerEngine: 'Local Docker Daemon'
    });
  } catch (err: any) {
    return res.json({
      success: true,
      isDockerAvailable: false,
      containerEngine: 'Workspace Restricted Execution Engine',
      reason: 'Local Docker daemon not running or docker CLI not installed.'
    });
  }
});

// POST /api/sandbox/execute - Execute command in local Docker sandbox or workspace sandbox
router.post('/execute', async (req, res) => {
  const { command, cwd, useDocker, requiresExplicitApproval } = req.body;

  if (!command || typeof command !== 'string') {
    return res.status(400).json({ success: false, error: 'command string is required' });
  }

  const workspaceRoot = getWorkspaceDir();
  const targetCwd = cwd ? assertWorkspaceBoundary(path.resolve(workspaceRoot, cwd), workspaceRoot) : workspaceRoot;

  let isDockerAvailable = false;
  let dockerVersion = '';
  try {
    dockerVersion = execSync('docker --version', { encoding: 'utf8', timeout: 2000 }).trim();
    execSync('docker info', { encoding: 'utf8', timeout: 2000 });
    isDockerAvailable = true;
  } catch (e) {
    isDockerAvailable = false;
  }

  // If command requires explicit approval and user hasn't explicitly confirmed
  if (requiresExplicitApproval && req.body.approved !== true) {
    return res.status(403).json({
      success: false,
      requiresApproval: true,
      message: `Execution Paused: Command "${command}" requires explicit user confirmation before execution.`,
      command,
      targetCwd
    });
  }

  const startTime = Date.now();

  if (useDocker && isDockerAvailable) {
    // Run in isolated local Docker container mounting the workspace directory
    const dockerCmd = `docker run --rm -v "${workspaceRoot}:/workspace" -w /workspace node:18-alpine sh -c ${JSON.stringify(command)}`;
    
    exec(dockerCmd, { timeout: 30000, maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      const durationMs = Date.now() - startTime;
      if (error) {
        return res.json({
          success: false,
          sandboxed: true,
          engine: 'docker',
          stdout,
          stderr: stderr || error.message,
          exitCode: error.code || 1,
          durationMs
        });
      }
      return res.json({
        success: true,
        sandboxed: true,
        engine: 'docker',
        stdout,
        stderr,
        exitCode: 0,
        durationMs
      });
    });
  } else {
    // Run directly inside protected workspace directory
    exec(command, { cwd: targetCwd, timeout: 30000, maxBuffer: 10 * 1024 * 1024 }, (error, stdout, stderr) => {
      const durationMs = Date.now() - startTime;
      if (error) {
        return res.json({
          success: false,
          sandboxed: false,
          engine: 'workspace_guarded',
          stdout,
          stderr: stderr || error.message,
          exitCode: error.code || 1,
          durationMs
        });
      }
      return res.json({
        success: true,
        sandboxed: false,
        engine: 'workspace_guarded',
        stdout,
        stderr,
        exitCode: 0,
        durationMs
      });
    });
  }
});

export default router;
