/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import { exec, execSync } from 'child_process';
import path from 'path';
import { getWorkspaceDir } from '../shared/workspace.js';
import { generateText } from '../shared/llm.js';

const router = Router();

function runGit(cmd: string, cwd: string): Promise<{ stdout: string; stderr: string; exitCode: number }> {
  return new Promise((resolve) => {
    exec(`git ${cmd}`, { cwd, timeout: 15000, maxBuffer: 5 * 1024 * 1024 }, (err, stdout, stderr) => {
      resolve({
        stdout: stdout || '',
        stderr: stderr || (err ? err.message : ''),
        exitCode: err ? (err.code || 1) : 0
      });
    });
  });
}

// GET /api/git/status
router.get('/status', async (req, res) => {
  const cwd = getWorkspaceDir();
  try {
    const isRepo = (await runGit('rev-parse --is-inside-work-tree', cwd)).exitCode === 0;
    if (!isRepo) {
      return res.json({ success: true, isRepo: false, branch: '', files: [], commits: [] });
    }

    const branchRes = await runGit('branch --show-current', cwd);
    const branch = branchRes.stdout.trim() || 'main';

    const statusRes = await runGit('status --porcelain', cwd);
    const lines = statusRes.stdout.split('\n').filter(Boolean);
    const files = lines.map(line => {
      const indexStatus = line.substring(0, 1);
      const workStatus = line.substring(1, 2);
      const filePath = line.substring(3).trim();
      let status: 'staged' | 'modified' | 'untracked' | 'deleted' = 'modified';
      if (indexStatus === 'D' || workStatus === 'D') status = 'deleted';
      else if (indexStatus === '?' || workStatus === '?') status = 'untracked';
      else if (indexStatus !== ' ') status = 'staged';

      return { path: filePath, status, raw: line };
    });

    const logRes = await runGit('log -n 5 --oneline', cwd);
    const commits = logRes.stdout.split('\n').filter(Boolean).map(line => {
      const parts = line.split(' ');
      return { hash: parts[0], message: parts.slice(1).join(' ') };
    });

    return res.json({
      success: true,
      isRepo: true,
      branch,
      files,
      commits
    });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/git/init
router.post('/init', async (req, res) => {
  const cwd = getWorkspaceDir();
  const resInit = await runGit('init', cwd);
  if (resInit.exitCode === 0) {
    return res.json({ success: true, message: 'Git repository initialized successfully.' });
  } else {
    return res.status(500).json({ success: false, error: resInit.stderr });
  }
});

// POST /api/git/stage
router.post('/stage', async (req, res) => {
  const cwd = getWorkspaceDir();
  const { files } = req.body; // string[] or '.'
  const target = Array.isArray(files) && files.length > 0 ? files.map(f => `"${f}"`).join(' ') : '.';
  const resStage = await runGit(`add ${target}`, cwd);
  if (resStage.exitCode === 0) {
    return res.json({ success: true, message: `Staged target: ${target}` });
  } else {
    return res.status(500).json({ success: false, error: resStage.stderr });
  }
});

// POST /api/git/commit - REQUIRES EXPLICIT USER CONFIRMATION
router.post('/commit', async (req, res) => {
  const { message, confirmed } = req.body;
  if (!confirmed) {
    return res.status(403).json({
      success: false,
      requiresConfirmation: true,
      message: 'Explicit user confirmation required before creating a git commit.'
    });
  }

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ success: false, error: 'Commit message is required.' });
  }

  const cwd = getWorkspaceDir();
  const safeMessage = message.replace(/"/g, '\\"');
  const resCommit = await runGit(`commit -m "${safeMessage}"`, cwd);
  if (resCommit.exitCode === 0) {
    return res.json({ success: true, stdout: resCommit.stdout });
  } else {
    return res.status(500).json({ success: false, error: resCommit.stderr || resCommit.stdout });
  }
});

// POST /api/git/branch
router.post('/branch', async (req, res) => {
  const { branchName } = req.body;
  if (!branchName) return res.status(400).json({ success: false, error: 'branchName is required' });
  const cwd = getWorkspaceDir();
  const resBranch = await runGit(`checkout -b "${branchName}"`, cwd);
  if (resBranch.exitCode === 0) {
    return res.json({ success: true, message: `Switched to new branch '${branchName}'` });
  } else {
    return res.status(500).json({ success: false, error: resBranch.stderr });
  }
});

// POST /api/git/pr-summary - Generates Pull Request summary using LLM diff analysis
router.post('/pr-summary', async (req, res) => {
  const { modelConfig } = req.body;
  const cwd = getWorkspaceDir();
  try {
    const diffRes = await runGit('diff HEAD', cwd);
    let diffText = diffRes.stdout;
    if (!diffText.trim()) {
      const stagedDiff = await runGit('diff --staged', cwd);
      diffText = stagedDiff.stdout;
    }

    if (!diffText.trim()) {
      return res.json({
        success: true,
        title: 'Update Application Components',
        description: 'No uncommitted or staged git diff found. Please make changes and stage them to generate an automated PR summary.'
      });
    }

    const truncatedDiff = diffText.length > 8000 ? diffText.substring(0, 8000) + '\n... (diff truncated)' : diffText;

    const prompt = `Analyze the following git diff and generate a professional Pull Request summary.

GIT DIFF:
\`\`\`diff
${truncatedDiff}
\`\`\`

Return a JSON object with:
1. "title": concise PR title (e.g. "feat: add user authentication flow")
2. "description": structured markdown PR description with Key Changes, Visual Modifications, and Verification Steps.`;

    const responseResult = await generateText({
      customConfig: modelConfig,
      prompt,
      systemInstruction: 'You are an expert lead engineer generating clean Pull Request descriptions.'
    });
    const responseText = responseResult.text || '';

    let prData = { title: 'Pull Request Summary', description: responseText };
    try {
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        prData = JSON.parse(jsonMatch[0]);
      }
    } catch (e) {
      // fallback
    }

    return res.json({ success: true, ...prData });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
