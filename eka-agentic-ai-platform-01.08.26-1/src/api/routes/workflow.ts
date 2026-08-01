/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';
import { executeNode } from './workflowNodes.js';

const router = Router();

// ── Directory constants ──────────────────────────────────────
const WORKFLOWS_DIR = () => path.join(process.cwd(), 'app-config', 'workflow');

async function ensureWorkflowsDir() {
  const dir = WORKFLOWS_DIR();
  if (!existsSync(dir)) await fs.mkdir(dir, { recursive: true });
}

function wfConfigFilename(name: string, id: string): string {
  const safeName = (name || 'untitled').replace(/[^a-z0-9_\-]/gi, '_').toLowerCase().slice(0, 60);
  return `${safeName}-${id}.config`;
}

async function findWfConfigPath(id: string): Promise<string | null> {
  const dir = WORKFLOWS_DIR();
  if (!existsSync(dir)) return null;
  const files = await fs.readdir(dir);
  const match = files.find(f => f.endsWith(`-${id}.config`));
  return match ? path.join(dir, match) : null;
}

// ── Human gate pause/resume map ──────────────────────────────
interface GatePending {
  resolve: (action: { action: 'approve' | 'reject'; input?: string }) => void;
  timer?: ReturnType<typeof setTimeout>;
}
const pendingGates = new Map<string, GatePending>();

// GET /api/workflow/list
router.get('/list', async (req, res) => {
  try {
    await ensureWorkflowsDir();
    const files = await fs.readdir(WORKFLOWS_DIR());
    const workflows = [];
    for (const f of files.filter(f => f.endsWith('.config'))) {
      try {
        const raw = await fs.readFile(path.join(WORKFLOWS_DIR(), f), 'utf8');
        workflows.push(JSON.parse(raw));
      } catch {}
    }
    workflows.sort((a: any, b: any) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    res.json({ success: true, workflows });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workflow/save
router.post('/save', async (req, res) => {
  try {
    await ensureWorkflowsDir();
    const wf = req.body;
    if (!wf?.id) return res.status(400).json({ success: false, error: 'Workflow id is required.' });
    const filename = wfConfigFilename(wf.name || 'untitled', wf.id);
    const oldPath = await findWfConfigPath(wf.id);
    if (oldPath && path.basename(oldPath) !== filename) await fs.unlink(oldPath).catch(() => {});
    await fs.writeFile(path.join(WORKFLOWS_DIR(), filename), JSON.stringify(wf, null, 2), 'utf8');
    res.json({ success: true, id: wf.id, filename });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// DELETE /api/workflow/:id
router.delete('/:id', async (req, res) => {
  try {
    const filePath = await findWfConfigPath(req.params.id);
    if (filePath && existsSync(filePath)) await fs.unlink(filePath);
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/workflow/gate-resolve
router.post('/gate-resolve', (req, res) => {
  const { gateId, action, input } = req.body;
  const gate = pendingGates.get(gateId);
  if (!gate) return res.status(404).json({ success: false, error: 'Gate not found or already resolved.' });
  if (gate.timer) clearTimeout(gate.timer);
  pendingGates.delete(gateId);
  gate.resolve({ action: action === 'reject' ? 'reject' : 'approve', input });
  res.json({ success: true });
});

// POST /api/workflow/execute — SSE streaming pipeline execution
//
// SSE event types (data field is JSON):
//   log      – { line: string }
//   node     – { id, status, preview }
//   gate     – { gateId, prompt, mode, context }  → frontend shows modal
//   done     – { success, logPath }
//   error    – { message }
router.post('/execute', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  let isAborted = false;
  req.on('close', () => { isAborted = true; });

  const send = (event: string, data: object) => {
    if (!res.writableEnded && !isAborted) res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  try {
    const { nodes, edges, modelConfig: customConfig, workflowName } = req.body;

    if (!Array.isArray(nodes) || nodes.length === 0) {
      send('error', { message: 'No nodes provided.' });
      res.end(); return;
    }

    const nodeResults: Record<string, { status: string; preview: string }> = {};
    const allLogs: string[] = [];
    const log = (line: string) => { allLogs.push(line); send('log', { line }); };

    // Per-run log directory setup
    const safeName = (workflowName || 'pipeline').replace(/[^a-z0-9_\-]/gi, '_');
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const runLogDir = path.join(process.cwd(), 'workflow-logs', safeName, ts);
    try { await fs.mkdir(runLogDir, { recursive: true }); } catch {}

    const appendNodeLog = async (nodeId: string, nodeLabel: string, nodeType: string, input: string, output: string, status: string) => {
      try {
        const entry = `=== [${nodeType}] ${nodeLabel} ===\nTimestamp: ${new Date().toISOString()}\nStatus: ${status}\n\n--- INPUT ---\n${input}\n\n--- OUTPUT ---\n${output}\n\n`;
        await fs.appendFile(path.join(runLogDir, `${nodeLabel.replace(/[^a-z0-9_\-]/gi, '_')}_${nodeId.slice(-6)}.log`), entry, 'utf8');
      } catch {}
    };

    log('▶ Pipeline execution started');

    // Build topological order
    const inDegree: Record<string, number> = {};
    const adj: Record<string, string[]> = {};
    for (const n of nodes) { inDegree[n.id] = 0; adj[n.id] = []; }
    for (const e of edges) {
      adj[e.sourceId]?.push(e.targetId);
      if (inDegree[e.targetId] !== undefined) inDegree[e.targetId]++;
    }
    const topoOrder: string[] = [];
    const remaining = { ...inDegree };
    const q = nodes.filter(n => inDegree[n.id] === 0).map(n => n.id);
    while (q.length > 0) {
      const id = q.shift()!;
      topoOrder.push(id);
      for (const nxt of (adj[id] || [])) {
        remaining[nxt]--;
        if (remaining[nxt] === 0) q.push(nxt);
      }
    }

    const nodeMap: Record<string, any> = Object.fromEntries(nodes.map((n: any) => [n.id, n]));
    const contextByNode: Record<string, string> = {};

    for (const nodeId of topoOrder) {
      if (isAborted) {
        log('🛑 Execution halted by user request.');
        break;
      }
      const node = nodeMap[nodeId];
      if (!node) continue;
      log(`⚙ Executing: [${node.type}] ${node.label}`);
      send('node', { id: nodeId, status: 'running', preview: '' });

      const upstreamEdges = edges.filter((e: any) => e.targetId === nodeId);
      const upstreamContext = upstreamEdges
        .map((e: any) => contextByNode[e.sourceId] || '')
        .filter(Boolean)
        .join('\n\n---\n\n');

      try {
        let output = '';

        // human_intervention is handled inline here because it needs pendingGates
        if (node.type === 'human_intervention') {
          const cfg = node.config;
          const mode = cfg.mode || 'confirm';
          const timeoutSec = cfg.timeoutSeconds ?? 0;
          const defaultAction = cfg.defaultAction || 'approve';
          const gateId = `gate_${nodeId}_${Date.now()}`;
          const context = upstreamContext || '';

          log(`  ⏸ Human gate paused — waiting for response (${mode} mode)`);

          const gateResult = await new Promise<{ action: 'approve' | 'reject'; input?: string }>((resolve) => {
            const entry: GatePending = { resolve };
            if (timeoutSec > 0) {
              entry.timer = setTimeout(() => {
                pendingGates.delete(gateId);
                resolve({ action: defaultAction });
                log(`  ⏱ Human gate timed out after ${timeoutSec}s → auto-${defaultAction}`);
              }, timeoutSec * 1000);
            }
            pendingGates.set(gateId, entry);
            send('gate', { gateId, prompt: cfg.prompt, mode, context: context.slice(0, 2000) });
          });

          if (gateResult.action === 'reject') {
            log(`  ⛔ Human gate REJECTED — pipeline aborted at this node.`);
            nodeResults[nodeId] = { status: 'failed', preview: 'Gate rejected by user' };
            send('node', { id: nodeId, status: 'failed', preview: 'Gate rejected by user' });
            contextByNode[nodeId] = '';
            continue;
          }

          if (mode === 'input' && gateResult.input) {
            output = `${context}\n\n[Human Input]\n${gateResult.input}`;
            log(`  ✅ Human gate approved with input (${gateResult.input.length} chars)`);
          } else {
            output = context;
            log(`  ✅ Human gate approved — upstream output passed through.`);
          }
        } else {
          // Delegate all other node types to workflowNodes.ts
          output = await executeNode({
            node, nodes, edges, nodeMap, contextByNode,
            customConfig, workflowName, log
          });
        }

        contextByNode[nodeId] = output;
        nodeResults[nodeId] = { status: 'completed', preview: output.slice(0, 200) };
        send('node', { id: nodeId, status: 'completed', preview: output.slice(0, 200) });
        await appendNodeLog(nodeId, node.label, node.type, upstreamContext.slice(0, 500), output.slice(0, 1000), 'completed');

      } catch (err: any) {
        const errMsg = `❌ Node [${node.label}] failed: ${err.message}`;
        log(errMsg);
        nodeResults[nodeId] = { status: 'failed', preview: errMsg };
        send('node', { id: nodeId, status: 'failed', preview: errMsg });
        contextByNode[nodeId] = '';
        await appendNodeLog(nodeId, node.label, node.type, upstreamContext.slice(0, 500), errMsg, 'failed');
      }
    }

    // Write run summary
    const logPath = `workflow-logs/${safeName}/${ts}/`;
    try {
      const summary = `WORKFLOW RUN SUMMARY\nName: ${workflowName || 'pipeline'}\nTimestamp: ${ts}\nNodes: ${nodes.length}\nEdges: ${edges.length}\nStatus: completed\n\nLogs:\n${allLogs.join('\n')}\n`;
      await fs.writeFile(path.join(runLogDir, '_summary.log'), summary, 'utf8');
    } catch {}

    log(`✅ Pipeline execution complete · logs saved to ${logPath}`);
    send('done', { success: true, logPath });
    res.end();
  } catch (fatalErr: any) {
    console.error('[workflow/execute] fatal:', fatalErr);
    send('error', { message: fatalErr.message || 'Internal server error during pipeline execution.' });
    res.end();
  }
});

export default router;
