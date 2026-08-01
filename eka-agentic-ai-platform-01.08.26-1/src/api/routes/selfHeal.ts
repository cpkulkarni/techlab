/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { exec } from 'child_process';
import { getWorkspaceDir } from '../shared/workspace.js';
import { assertWorkspaceBoundary } from '../shared/workspaceGuard.js';
import { generateText } from '../shared/llm.js';

const router = Router();

function runVerification(cmd: string, cwd: string): Promise<{ success: boolean; output: string }> {
  return new Promise((resolve) => {
    exec(cmd, { cwd, timeout: 20000, maxBuffer: 5 * 1024 * 1024 }, (err, stdout, stderr) => {
      const output = (stdout + '\n' + stderr).trim();
      resolve({
        success: !err,
        output
      });
    });
  });
}

// POST /api/agent/self-heal - Automated Autonomous Self-Healing Debug Loop
router.post('/self-heal', async (req, res) => {
  const {
    errorOutput,
    verificationCommand = 'npx tsc --noEmit',
    targetFilePath,
    maxAttempts = 3,
    modelConfig
  } = req.body;

  if (!errorOutput) {
    return res.status(400).json({ success: false, error: 'errorOutput string is required' });
  }

  const workspaceRoot = getWorkspaceDir();
  const attemptLogs: Array<{ attempt: number; diagnosis: string; fixApplied?: string; verificationResult: string; success: boolean }> = [];

  let currentError = errorOutput;
  let attempt = 0;
  let healed = false;

  while (attempt < maxAttempts && !healed) {
    attempt++;
    let fileContent = '';
    let resolvedPath = '';

    if (targetFilePath) {
      try {
        resolvedPath = assertWorkspaceBoundary(path.resolve(workspaceRoot, targetFilePath), workspaceRoot);
        fileContent = await fs.readFile(resolvedPath, 'utf8');
      } catch (e) {
        // file unreadable
      }
    }

    const systemPrompt = `You are an expert self-healing autonomous AI debugging agent.
Analyze the provided stack trace or error log and generate a surgical fix.

WORKSPACE ROOT: ${workspaceRoot}
TARGET FILE: ${targetFilePath || 'Autodetect from stack trace'}

FILE CONTENT:
\`\`\`
${fileContent.substring(0, 4000)}
\`\`\`

CURRENT FAILURE:
\`\`\`
${currentError.substring(0, 3000)}
\`\`\`

Respond ONLY with a JSON object:
{
  "diagnosis": "Short explanation of the root cause",
  "targetFile": "relative/path/to/file.ext",
  "targetContent": "exact code block to replace",
  "replacementContent": "fixed replacement code block"
}`;

    try {
      const genResult = await generateText({
        customConfig: modelConfig,
        prompt: 'Fix the reported runtime/type error.',
        systemInstruction: systemPrompt
      });
      const responseText = genResult.text || '';

      let fixData: any = null;
      try {
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);
        if (jsonMatch) fixData = JSON.parse(jsonMatch[0]);
      } catch (e) {
        // fallback parse
      }

      if (!fixData || !fixData.targetFile || !fixData.targetContent) {
        attemptLogs.push({
          attempt,
          diagnosis: 'Failed to extract valid JSON fix from LLM response.',
          verificationResult: 'Skipped - no edit applied.',
          success: false
        });
        break;
      }

      // Apply the fix strictly within workspace
      const editFilePath = assertWorkspaceBoundary(path.resolve(workspaceRoot, fixData.targetFile), workspaceRoot);
      const existingCode = await fs.readFile(editFilePath, 'utf8');

      if (existingCode.includes(fixData.targetContent)) {
        const newCode = existingCode.replace(fixData.targetContent, fixData.replacementContent);
        await fs.writeFile(editFilePath, newCode, 'utf8');

        // Verify fix
        const verifyRes = await runVerification(verificationCommand, workspaceRoot);
        attemptLogs.push({
          attempt,
          diagnosis: fixData.diagnosis,
          fixApplied: `Replaced snippet in ${fixData.targetFile}`,
          verificationResult: verifyRes.output,
          success: verifyRes.success
        });

        if (verifyRes.success) {
          healed = true;
          break;
        } else {
          currentError = verifyRes.output;
        }
      } else {
        attemptLogs.push({
          attempt,
          diagnosis: `Target snippet not found in ${fixData.targetFile}.`,
          verificationResult: 'Target content match failed.',
          success: false
        });
      }
    } catch (err: any) {
      attemptLogs.push({
        attempt,
        diagnosis: `Self-heal execution error: ${err.message}`,
        verificationResult: 'Aborted.',
        success: false
      });
      break;
    }
  }

  return res.json({
    success: healed,
    healed,
    totalAttempts: attempt,
    maxAttempts,
    logs: attemptLogs
  });
});

export default router;
