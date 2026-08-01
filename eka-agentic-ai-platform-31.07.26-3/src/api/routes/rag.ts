/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { getWorkspaceDir, collectDirFiles } from '../shared/workspace.js';
import { generateText } from '../shared/llm.js';

const router = Router();

interface VectorChunk {
  id: string;
  filePath: string;
  codeSnippet: string;
  startLine: number;
  endLine: number;
  tokens: Set<string>;
}

// In-memory Vector Index store
let vectorIndexStore: VectorChunk[] = [];
let lastIndexedAt: string | null = null;

function tokenize(text: string): Set<string> {
  const words = text.toLowerCase().replace(/[^a-z0-9_]/g, ' ').split(/\s+/).filter(w => w.length > 2);
  return new Set(words);
}

function computeSimilarity(queryTokens: Set<string>, chunkTokens: Set<string>): number {
  if (queryTokens.size === 0 || chunkTokens.size === 0) return 0;
  let matches = 0;
  for (const token of queryTokens) {
    if (chunkTokens.has(token)) matches++;
  }
  return matches / Math.sqrt(queryTokens.size * chunkTokens.size);
}

// POST /api/multi-agent/rag/index - Index workspace files into vector store
router.post('/index', async (req, res) => {
  try {
    const workspaceDir = getWorkspaceDir();
    const files = await collectDirFiles(workspaceDir, workspaceDir);
    const newChunks: VectorChunk[] = [];

    for (const fileObj of files) {
      const relPath = fileObj.path;
      if (relPath.includes('node_modules') || relPath.includes('dist') || relPath.includes('.git')) continue;
      try {
        const content = fileObj.content;
        const lines = content.split('\n');
        const chunkSize = 25; // lines per chunk
        
        for (let i = 0; i < lines.length; i += chunkSize) {
          const chunkLines = lines.slice(i, i + chunkSize);
          const snippet = chunkLines.join('\n');
          if (!snippet.trim()) continue;

          newChunks.push({
            id: `${relPath}:${i + 1}`,
            filePath: relPath,
            codeSnippet: snippet,
            startLine: i + 1,
            endLine: Math.min(i + chunkSize, lines.length),
            tokens: tokenize(snippet + ' ' + relPath)
          });
        }
      } catch (e) {
        // skip unreadable
      }
    }

    vectorIndexStore = newChunks;
    lastIndexedAt = new Date().toISOString();

    return res.json({
      success: true,
      indexedFilesCount: files.length,
      chunksCount: vectorIndexStore.length,
      lastIndexedAt
    });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/multi-agent/rag/search - Perform semantic vector search over index
router.post('/search', async (req, res) => {
  const { query, topK = 5 } = req.body;
  if (!query || typeof query !== 'string') {
    return res.status(400).json({ success: false, error: 'query string is required' });
  }

  if (vectorIndexStore.length === 0) {
    // Auto-trigger indexing if empty
    try {
      const workspaceDir = getWorkspaceDir();
      const files = await collectDirFiles(workspaceDir, workspaceDir);
      for (const fileObj of files) {
        const relPath = fileObj.path;
        if (relPath.includes('node_modules') || relPath.includes('dist') || relPath.includes('.git')) continue;
        try {
          const content = fileObj.content;
          const lines = content.split('\n');
          const chunkSize = 25;
          for (let i = 0; i < lines.length; i += chunkSize) {
            const chunkLines = lines.slice(i, i + chunkSize);
            const snippet = chunkLines.join('\n');
            if (!snippet.trim()) continue;
            vectorIndexStore.push({
              id: `${relPath}:${i + 1}`,
              filePath: relPath,
              codeSnippet: snippet,
              startLine: i + 1,
              endLine: Math.min(i + chunkSize, lines.length),
              tokens: tokenize(snippet + ' ' + relPath)
            });
          }
        } catch (e) {}
      }
      lastIndexedAt = new Date().toISOString();
    } catch (e) {}
  }

  const queryTokens = tokenize(query);
  const scoredResults = vectorIndexStore.map(chunk => ({
    chunk,
    score: computeSimilarity(queryTokens, chunk.tokens)
  }))
  .filter(item => item.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, Number(topK))
  .map(item => ({
    filePath: item.chunk.filePath,
    startLine: item.chunk.startLine,
    endLine: item.chunk.endLine,
    snippet: item.chunk.codeSnippet,
    score: Number(item.score.toFixed(3))
  }));

  return res.json({
    success: true,
    query,
    totalIndexedChunks: vectorIndexStore.length,
    results: scoredResults
  });
});

export default router;
