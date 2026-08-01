/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';
import { execSync } from 'child_process';
import { Type } from '@google/genai';
import { generateText } from '../shared/llm.js';
import { getWorkspaceDir, setWorkspaceDir, buildTree, collectDirFiles, isSourceCodePath } from '../shared/workspace.js';
import { getGeminiClient } from '../shared/llm.js';
import { generateKnowledgeGraph } from '../shared/knowledgeGraph.js';

const router = Router();

// GET /api/workspace/root-path
router.get('/root-path', (req, res) => {
  res.json({ success: true, rootPath: getWorkspaceDir() });
});

// POST /api/workspace/root-path
router.post('/root-path', async (req, res) => {
  const { rootPath } = req.body;
  if (!rootPath) return res.status(400).json({ success: false, error: 'rootPath is required' });
  try {
    const absolutePath = path.resolve(rootPath);
    if (!existsSync(absolutePath)) await fs.mkdir(absolutePath, { recursive: true });
    setWorkspaceDir(absolutePath);
    console.log(`Workspace root path changed to: ${absolutePath}`);
    res.json({ success: true, rootPath: absolutePath });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace/list-server-dirs
router.get('/list-server-dirs', async (req, res) => {
  const queryPath = (req.query.path as string) || process.cwd();
  try {
    const targetPath = path.resolve(queryPath);
    const stats = await fs.stat(targetPath);
    if (!stats.isDirectory()) return res.status(400).json({ success: false, error: 'Path is not a directory' });
    const items = await fs.readdir(targetPath, { withFileTypes: true });
    const subdirs = items
      .filter(item => item.isDirectory() && !item.name.startsWith('.'))
      .map(item => item.name);
    res.json({ success: true, currentPath: targetPath, parentPath: path.dirname(targetPath), subdirs: subdirs.sort() });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace/knowledge-graph
router.get('/knowledge-graph', async (req, res) => {
  try {
    const kgPath = path.join(process.cwd(), 'eka_src_code_knowledge_graph.json');
    if (existsSync(kgPath)) {
      const content = await fs.readFile(kgPath, 'utf8');
      return res.json(JSON.parse(content));
    } else {
      const kg = await generateKnowledgeGraph({ saveToFile: true });
      return res.json(kg);
    }
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workspace/sync
router.post('/sync', async (req, res) => {
  const { files: syncFiles } = req.body;
  if (!Array.isArray(syncFiles)) return res.status(400).json({ success: false, error: 'files must be an array' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    for (const f of syncFiles) {
      if (!f.path) continue;
      const fullPath = path.join(WORKSPACE_DIR, f.path);
      if (!fullPath.startsWith(WORKSPACE_DIR)) continue;
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      await fs.writeFile(fullPath, f.content ?? '', 'utf8');
    }
    // Auto-update Knowledge Graph after file changes
    generateKnowledgeGraph().catch(err => console.error('Auto KG update error:', err));
    res.json({ success: true, count: syncFiles.length });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace
router.get('/', async (req, res) => {
  try {
    const tree = await buildTree(getWorkspaceDir());
    res.json({ success: true, files: tree });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace/file
router.get('/file', async (req, res) => {
  const relativePath = req.query.path as string;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Path query parameter is required.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    const content = await fs.readFile(fullPath, 'utf8');
    res.json({ success: true, content });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workspace/file
router.post('/file', async (req, res) => {
  const { path: relativePath, content } = req.body;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Path is required.' });
  if (isSourceCodePath(relativePath)) return res.status(403).json({ success: false, error: 'This file is part of the locked source-code group and cannot be modified.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    await fs.mkdir(path.dirname(fullPath), { recursive: true });
    await fs.writeFile(fullPath, content ?? '', 'utf8');
    generateKnowledgeGraph().catch(err => console.error('Auto KG update error:', err));
    res.json({ success: true, message: 'File saved successfully.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workspace/review
router.post('/review', async (req, res) => {
  const { path: relativePath, customConfig } = req.body;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Path is required.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    const content = await fs.readFile(fullPath, 'utf8');
    if (!content.trim()) {
      return res.json({
        success: true,
        score: 100,
        metrics: { performance: 'N/A', security: 'N/A', style: 'N/A', complexity: 'N/A' },
        summary: 'This file is empty. No review suggestions needed.',
        suggestions: []
      });
    }

    const systemPrompt = `You are an elite Senior Software Architect and automated code reviewer (like SonarQube combined with a premium technical leader).
Analyze the code in the file provided and generate a thorough review.
Identify security flaws, memory leaks, performance bottlenecks, readability improvements, or typescript best-practices.
For each suggestion, provide:
1. "line": The approximate line number.
2. "title": A short catchy title for the suggestion.
3. "description": Why it should be changed.
4. "severity": One of: "low", "medium", "high".
5. "targetContent": The EXACT code block in the original file that should be replaced (must match exactly to let the editor apply it).
6. "replacementContent": The corrected replacement code block.

If no changes are necessary or if it's perfectly written, return empty suggestions.
Return your response as a JSON object matching this schema:
{
  "score": number (0 to 100),
  "metrics": {
    "performance": string (e.g. "Excellent", "Good", "Needs Improvement"),
    "security": string,
    "style": string,
    "complexity": string
  },
  "summary": string (overall high level feedback),
  "suggestions": [
    {
      "id": string (unique identifier like "s1"),
      "line": number,
      "title": string,
      "description": string,
      "severity": "low" | "medium" | "high",
      "targetContent": string,
      "replacementContent": string
    }
  ]
}
Return only the raw JSON. No markdown blocks outside the JSON.`;

    const schema = {
      type: Type.OBJECT,
      properties: {
        score: { type: Type.INTEGER },
        metrics: {
          type: Type.OBJECT,
          properties: {
            performance: { type: Type.STRING },
            security: { type: Type.STRING },
            style: { type: Type.STRING },
            complexity: { type: Type.STRING }
          },
          required: ['performance', 'security', 'style', 'complexity']
        },
        summary: { type: Type.STRING },
        suggestions: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              id: { type: Type.STRING },
              line: { type: Type.INTEGER },
              title: { type: Type.STRING },
              description: { type: Type.STRING },
              severity: { type: Type.STRING, description: 'Must be exactly low, medium, or high' },
              targetContent: { type: Type.STRING },
              replacementContent: { type: Type.STRING }
            },
            required: ['id', 'line', 'title', 'description', 'severity', 'targetContent', 'replacementContent']
          }
        }
      },
      required: ['score', 'metrics', 'summary', 'suggestions']
    };

    const result = await generateText({
      customConfig,
      prompt: `File Name: ${relativePath}\nFile Contents:\n\`\`\`\n${content}\n\`\`\``,
      systemInstruction: systemPrompt,
      responseMimeType: 'application/json',
      responseSchema: schema,
      logType: 'Code Audit'
    });

    let rawText = (result.text || '{}').trim();
    if (rawText.startsWith('```')) {
      rawText = rawText.replace(/^```[a-z]*\n?/i, '').replace(/\n?```$/, '').trim();
    }
    const jsonMatch = rawText.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      rawText = jsonMatch[0];
    }
    const reviewData = JSON.parse(rawText);
    res.json({ success: true, ...reviewData });
  } catch (error: any) {
    console.error('Code review error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

// DELETE /api/workspace/file
router.delete('/file', async (req, res) => {
  const relativePath = req.query.path as string;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Path query is required.' });
  if (isSourceCodePath(relativePath)) return res.status(403).json({ success: false, error: 'This file is part of the locked source-code group and cannot be deleted.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    await fs.unlink(fullPath);
    generateKnowledgeGraph().catch(err => console.error('Auto KG update error:', err));
    res.json({ success: true, message: 'File deleted.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workspace/folder
router.post('/folder', async (req, res) => {
  const { path: relativePath } = req.body;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Folder path is required.' });
  if (isSourceCodePath(relativePath)) return res.status(403).json({ success: false, error: 'This folder is part of the locked source-code group and cannot be modified.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    await fs.mkdir(fullPath, { recursive: true });
    res.json({ success: true, message: 'Folder created.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// DELETE /api/workspace/folder
router.delete('/folder', async (req, res) => {
  const relativePath = req.query.path as string;
  if (!relativePath) return res.status(400).json({ success: false, error: 'Folder path is required.' });
  if (isSourceCodePath(relativePath)) return res.status(403).json({ success: false, error: 'The source-code folder and its contents are locked and cannot be deleted.' });
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const fullPath = path.join(WORKSPACE_DIR, relativePath);
    if (!fullPath.startsWith(WORKSPACE_DIR)) return res.status(403).json({ success: false, error: 'Access denied: outside workspace bounds.' });
    await fs.rm(fullPath, { recursive: true, force: true });
    res.json({ success: true, message: 'Folder deleted.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace/export-zip
router.get('/export-zip', async (req, res) => {
  const WORKSPACE_DIR = getWorkspaceDir();
  try {
    const files = await collectDirFiles(WORKSPACE_DIR, WORKSPACE_DIR);
    const zipName = `workspace-export-${Date.now()}.zip`;
    const tmpZipPath = path.join(process.cwd(), zipName);

    try {
      const zipCmd = process.platform === 'win32'
        ? `powershell -Command "Compress-Archive -Path '${WORKSPACE_DIR}\\*' -DestinationPath '${tmpZipPath}' -Force"`
        : `cd "${WORKSPACE_DIR}" && zip -r "${tmpZipPath}" . --exclude "*/node_modules/*" --exclude "*/.git/*" --exclude "*/dist/*"`;
      execSync(zipCmd, { timeout: 30000 });
      res.setHeader('Content-Type', 'application/zip');
      res.setHeader('Content-Disposition', `attachment; filename="${zipName}"`);
      const zipBuffer = await fs.readFile(tmpZipPath);
      res.send(zipBuffer);
      fs.unlink(tmpZipPath).catch(() => {});
    } catch (_zipErr) {
      // Fallback: send as JSON bundle
      const bundle: Record<string, string> = {};
      for (const f of files) bundle[f.path] = f.content;
      res.setHeader('Content-Type', 'application/json');
      res.setHeader('Content-Disposition', `attachment; filename="workspace-export-${Date.now()}.json"`);
      res.send(JSON.stringify(bundle, null, 2));
    }
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
