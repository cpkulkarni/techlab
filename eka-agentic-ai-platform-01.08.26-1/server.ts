/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import cors from 'cors';
import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import dotenv from 'dotenv';

import { loadTodayLogs } from './src/api/shared/logs.js';
import { generateKnowledgeGraph } from './src/api/shared/knowledgeGraph.js';

// Route modules — one file per domain area
import mailserverRouter from './src/api/routes/mailserver.js';
import configRouter from './src/api/routes/config.js';
import modelsRouter from './src/api/routes/models.js';
import logsRouter from './src/api/routes/logs.js';
import chatRouter from './src/api/routes/chat.js';
import workspaceRouter from './src/api/routes/workspace.js';
import agentRouter from './src/api/routes/agent.js';
import researchRouter from './src/api/routes/research.js';
import documentationRouter from './src/api/routes/documentation.js';
import testingRouter from './src/api/routes/testing.js';
import workflowRouter from './src/api/routes/workflow.js';
import multiAgentRouter from './src/api/routes/multiAgent.js';
import sandboxRouter from './src/api/routes/sandbox.js';
import gitRouter from './src/api/routes/git.js';
import ragRouter from './src/api/routes/rag.js';
import selfHealRouter from './src/api/routes/selfHeal.js';
import snapshotsRouter from './src/api/routes/snapshots.js';
import knowledgeRouter from './src/api/routes/knowledge.js';
import schedulerRouter from './src/api/routes/scheduler.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

// Enable CORS for direct API access from external programs, scripts, and frontends
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin']
}));

app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// Core API Health & Metadata Endpoints for direct API callers
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    uptime: process.uptime(),
    timestamp: new Date().toISOString(),
    service: 'Eka Agentic AI Studio API'
  });
});

app.get('/api', (req, res) => {
  res.json({
    name: 'Eka Agentic AI Studio API',
    version: '1.0.0',
    endpoints: [
      '/api/health',
      '/api/workspace/knowledge-graph',
      '/api/chat',
      '/api/config',
      '/api/models',
      '/api/interaction-logs',
      '/api/workspace',
      '/api/sandbox',
      '/api/git',
      '/api/agent',
      '/api/workflow',
      '/api/multi-agent'
    ]
  });
});

// Mount all API routers
app.use('/api/mailserver',       mailserverRouter);
app.use('/api/config',           configRouter);
app.use('/api/models',           modelsRouter);
app.use('/api/interaction-logs', logsRouter);
app.use('/api/chat',             chatRouter);
app.use('/api/workspace',        workspaceRouter);
app.use('/api/workspace/snapshots', snapshotsRouter);
app.use('/api/sandbox',          sandboxRouter);
app.use('/api/git',              gitRouter);
app.use('/api/multi-agent/rag',  ragRouter);
app.use('/api/multi-agent/knowledge', knowledgeRouter);
app.use('/api/agent/self-heal',  selfHealRouter);
// Mount specific agent sub-routes BEFORE the generic /api/agent mount
app.use('/api/agent/research',         researchRouter);
app.use('/api/agent/document',         documentationRouter);
app.use('/api/agent/test-suite',       testingRouter);
// Generic agent routes: plan, execute-step, auto-correct
app.use('/api/agent',                  agentRouter);
app.use('/api/workflow',         workflowRouter);
app.use('/api/scheduler',        schedulerRouter);
app.use('/api/multi-agent',      multiAgentRouter);

async function startServer() {
  await loadTodayLogs();
  generateKnowledgeGraph().catch(err => console.error('Failed initial Knowledge Graph generation:', err));

  const isStandaloneBackend = process.env.STANDALONE_BACKEND === 'true' || process.env.SERVE_FRONTEND === 'false';

  if (isStandaloneBackend) {
    console.log('[Server] Operating in Standalone API Backend mode (Frontend serving disabled).');
  } else if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on port ${PORT} (${isStandaloneBackend ? 'Standalone API Backend' : 'Combined Frontend & Backend'})`);
  });
}

startServer();
