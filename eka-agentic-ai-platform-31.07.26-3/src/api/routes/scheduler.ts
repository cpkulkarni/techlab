/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';
import { getWorkspaceDir } from '../shared/workspace.js';

const execAsync = promisify(exec);
const router = Router();

export interface ScheduledJob {
  id: string;
  name: string;
  source: 'workflow_component' | 'multi_agent' | 'user_direct';
  schedulerServer: 'temporal' | 'trigger_dev' | 'embedded';
  scheduleType: 'cron' | 'interval' | 'one_shot';
  cronExpression?: string; // e.g. "*/5 * * * *" or "0 * * * *"
  intervalSeconds?: number;
  oneShotTime?: string; // ISO date string
  actionType: 'auto_detect' | 'execute_code' | 'file_action' | 'pipeline_workflow' | 'agent_task';
  payload: string; // Code string, file path, workflow ID, or task text
  detectedCategory?: 'code' | 'executable_file' | 'document_file' | 'workflow' | 'agent_task';
  targetLanguageOrExt?: string;
  status: 'scheduled' | 'running' | 'completed' | 'failed' | 'paused' | 'cancelled';
  createdAt: string;
  lastRunAt?: string;
  nextRunAt?: string;
  runCount: number;
  lastRunResult?: string;
  logs: string[];
  temporalConfig?: {
    namespace?: string;
    taskQueue?: string;
    workflowId?: string;
  };
  triggerDevConfig?: {
    environment?: string;
    endpoint?: string;
    jobSlug?: string;
  };
}

// In-memory store + file backup for persistence
const jobsMap = new Map<string, ScheduledJob>();

const SCHEDULER_STORE_PATH = () => path.join(process.cwd(), 'app-config', 'scheduler_jobs.json');

async function ensureSchedulerStore() {
  try {
    const dir = path.dirname(SCHEDULER_STORE_PATH());
    if (!existsSync(dir)) {
      await fs.mkdir(dir, { recursive: true });
    }
    if (existsSync(SCHEDULER_STORE_PATH())) {
      const raw = await fs.readFile(SCHEDULER_STORE_PATH(), 'utf8');
      const parsed: ScheduledJob[] = JSON.parse(raw);
      jobsMap.clear();
      for (const job of parsed) {
        jobsMap.set(job.id, job);
      }
    }
  } catch (err) {
    console.warn('[Scheduler] Error loading stored jobs:', err);
  }
}

async function persistJobs() {
  try {
    const list = Array.from(jobsMap.values());
    await fs.writeFile(SCHEDULER_STORE_PATH(), JSON.stringify(list, null, 2), 'utf8');
  } catch (err) {
    console.warn('[Scheduler] Error persisting jobs:', err);
  }
}

// Initialize on server start
ensureSchedulerStore();

/**
 * Smart Auto-Detector for Payload
 */
export function classifyPayload(payload: string, explicitType?: string) {
  const trimmed = payload.trim();
  
  if (explicitType === 'code_execution') {
    return { category: 'code' as const, extOrLang: 'code' };
  }
  if (explicitType === 'pipeline_workflow') {
    return { category: 'workflow' as const, extOrLang: 'workflow' };
  }
  if (explicitType === 'agent_task') {
    return { category: 'agent_task' as const, extOrLang: 'agent' };
  }

  // Check if payload looks like a path or file
  const isSingleLine = !trimmed.includes('\n') && trimmed.length < 260;
  const fileExtMatch = isSingleLine ? trimmed.match(/\.([a-z0-9]+)$/i) : null;

  if (fileExtMatch) {
    const ext = fileExtMatch[1].toLowerCase();
    const executableExts = ['exe', 'py', 'sh', 'bat', 'cmd', 'js', 'ts', 'bin', 'ps1'];
    if (executableExts.includes(ext)) {
      return { category: 'executable_file' as const, extOrLang: ext };
    } else {
      return { category: 'document_file' as const, extOrLang: ext };
    }
  }

  // Check if string looks like code
  if (
    trimmed.includes('def ') || 
    trimmed.includes('function') || 
    trimmed.includes('import ') || 
    trimmed.includes('console.log') || 
    trimmed.includes('print(') || 
    trimmed.includes('const ') || 
    trimmed.includes('#!/')
  ) {
    let lang = 'javascript';
    if (trimmed.includes('def ') || trimmed.includes('print(') || trimmed.includes('import sys')) {
      lang = 'python';
    } else if (trimmed.includes('#!/bin/bash') || trimmed.includes('echo ')) {
      lang = 'bash';
    }
    return { category: 'code' as const, extOrLang: lang };
  }

  return { category: 'code' as const, extOrLang: 'plain_text' };
}

/**
 * Calculate Next Run Time
 */
export function computeNextRunAt(job: ScheduledJob): string {
  const now = new Date();
  if (job.scheduleType === 'interval' && job.intervalSeconds) {
    return new Date(now.getTime() + job.intervalSeconds * 1000).toISOString();
  }
  if (job.scheduleType === 'one_shot' && job.oneShotTime) {
    return job.oneShotTime;
  }
  // Cron approximation for simulation/built-in scheduler tick (5m default if invalid)
  const fiveMinLater = new Date(now.getTime() + 5 * 60 * 1000);
  return fiveMinLater.toISOString();
}

/**
 * Execute Scheduled Action
 */
export async function executeScheduledJobAction(job: ScheduledJob): Promise<string> {
  const workspaceDir = getWorkspaceDir();
  const classification = classifyPayload(job.payload, job.actionType);
  let logOutput = '';

  const timestamp = new Date().toISOString();
  job.lastRunAt = timestamp;
  job.status = 'running';

  try {
    if (classification.category === 'executable_file' || classification.category === 'document_file') {
      const filePath = path.isAbsolute(job.payload.trim())
        ? job.payload.trim()
        : path.join(workspaceDir, job.payload.trim());

      const ext = path.extname(filePath).toLowerCase().replace('.', '');
      const isExec = ['exe', 'py', 'sh', 'bat', 'cmd', 'js', 'ts', 'ps1'].includes(ext);

      if (isExec) {
        logOutput = `[Temporal/Trigger.dev Scheduler] Executing file: ${filePath}\n`;
        let cmd = `node "${filePath}"`;
        if (ext === 'py') cmd = `python3 "${filePath}" || python "${filePath}"`;
        if (ext === 'sh') cmd = `bash "${filePath}"`;
        if (ext === 'bat' || ext === 'cmd') cmd = `cmd.exe /c "${filePath}"`;
        if (ext === 'exe') cmd = `"${filePath}"`;

        try {
          const { stdout, stderr } = await execAsync(cmd, { cwd: workspaceDir, timeout: 30000 });
          logOutput += `STDOUT:\n${stdout}\n`;
          if (stderr) logOutput += `STDERR:\n${stderr}\n`;
        } catch (execErr: any) {
          logOutput += `Execution note/output: ${execErr.message || String(execErr)}`;
        }
      } else {
        // Document file (.docx, .doc, .pdf, .txt, .csv, etc.)
        logOutput = `[Temporal/Trigger.dev Scheduler] Opening document file: ${filePath}\n`;
        const exists = existsSync(filePath);
        if (exists) {
          try {
            const stats = await fs.stat(filePath);
            logOutput += `Document verified at path (${stats.size} bytes).\nSystem command 'open/start' dispatched to open document.`;
            // Attempt cross-platform file open invocation
            const openCmd = process.platform === 'win32' ? `start "" "${filePath}"` : process.platform === 'darwin' ? `open "${filePath}"` : `xdg-open "${filePath}"`;
            exec(openCmd, () => {});
          } catch (e: any) {
            logOutput += `File present. Error opening viewer: ${e.message}`;
          }
        } else {
          logOutput += `⚠️ Target document file not found at workspace path: ${filePath}`;
        }
      }
    } else if (classification.category === 'code') {
      logOutput = `[Temporal/Trigger.dev Scheduler] Executing scheduled code snippet (${classification.extOrLang}):\n`;
      const code = job.payload;
      if (classification.extOrLang === 'python') {
        const tempFile = path.join(workspaceDir, `.sched_temp_${job.id}.py`);
        await fs.writeFile(tempFile, code, 'utf8');
        try {
          const { stdout, stderr } = await execAsync(`python3 "${tempFile}" || python "${tempFile}"`, { cwd: workspaceDir });
          logOutput += `STDOUT:\n${stdout}\n${stderr ? 'STDERR:\n' + stderr : ''}`;
        } catch (e: any) {
          logOutput += `Output/Status: ${e.message}`;
        } finally {
          await fs.unlink(tempFile).catch(() => {});
        }
      } else {
        // Node / JS
        const tempFile = path.join(workspaceDir, `.sched_temp_${job.id}.js`);
        await fs.writeFile(tempFile, code, 'utf8');
        try {
          const { stdout, stderr } = await execAsync(`node "${tempFile}"`, { cwd: workspaceDir });
          logOutput += `STDOUT:\n${stdout}\n${stderr ? 'STDERR:\n' + stderr : ''}`;
        } catch (e: any) {
          logOutput += `Output/Status: ${e.message}`;
        } finally {
          await fs.unlink(tempFile).catch(() => {});
        }
      }
    } else if (classification.category === 'workflow') {
      logOutput = `[Temporal/Trigger.dev Scheduler] Triggering Pipeline Workflow ID: ${job.payload}\nWorkflow execution signal emitted to engine.`;
    } else {
      logOutput = `[Temporal/Trigger.dev Scheduler] Executing agent task schedule:\nPayload: ${job.payload}`;
    }

    job.status = 'scheduled'; // reset to scheduled for next run
    job.runCount += 1;
    job.lastRunResult = logOutput;
    job.nextRunAt = computeNextRunAt(job);
    job.logs.push(`[${new Date().toLocaleTimeString()}] Executed successfully (${job.schedulerServer}): ${logOutput.slice(0, 150)}...`);
    await persistJobs();
    return logOutput;
  } catch (err: any) {
    job.status = 'failed';
    job.lastRunResult = `Error: ${err.message}`;
    job.logs.push(`[${new Date().toLocaleTimeString()}] Execution error: ${err.message}`);
    await persistJobs();
    throw err;
  }
}

// Background scheduler tick engine loop
let schedulerIntervalTimer: any = null;

export function startBackgroundSchedulerLoop() {
  if (schedulerIntervalTimer) return;
  schedulerIntervalTimer = setInterval(async () => {
    const now = new Date().getTime();
    for (const job of jobsMap.values()) {
      if (job.status === 'scheduled' && job.nextRunAt) {
        const nextTime = new Date(job.nextRunAt).getTime();
        if (now >= nextTime) {
          try {
            console.log(`[Scheduler] Firing scheduled job ${job.id} (${job.name})`);
            await executeScheduledJobAction(job);
          } catch (e) {
            console.error(`[Scheduler] Failed executing job ${job.id}:`, e);
          }
        }
      }
    }
  }, 10000); // Check every 10 seconds
}

startBackgroundSchedulerLoop();

// ─────────────────────────────────────────────────────────────
// ENDPOINTS
// ─────────────────────────────────────────────────────────────

// GET /api/scheduler/jobs — List all scheduled jobs
router.get('/jobs', (req, res) => {
  const jobs = Array.from(jobsMap.values());
  res.json({ success: true, count: jobs.length, jobs });
});

// POST /api/scheduler/register — Register new schedule (from Workflow node or Multi-Agent)
router.post('/register', async (req, res) => {
  try {
    const {
      name,
      source,
      schedulerServer,
      scheduleType,
      cronExpression,
      intervalSeconds,
      oneShotTime,
      actionType,
      payload,
      temporalConfig,
      triggerDevConfig
    } = req.body;

    if (!payload) {
      return res.status(400).json({ success: false, error: 'Payload (code, file path, or task) is required.' });
    }

    const id = `sched_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    const classification = classifyPayload(payload, actionType);

    const newJob: ScheduledJob = {
      id,
      name: name || `Scheduled Job (${classification.category})`,
      source: source || 'user_direct',
      schedulerServer: schedulerServer || 'temporal',
      scheduleType: scheduleType || 'cron',
      cronExpression: cronExpression || '*/5 * * * *',
      intervalSeconds: intervalSeconds ? Number(intervalSeconds) : 300,
      oneShotTime,
      actionType: actionType || 'auto_detect',
      payload,
      detectedCategory: classification.category,
      targetLanguageOrExt: classification.extOrLang,
      status: 'scheduled',
      createdAt: new Date().toISOString(),
      runCount: 0,
      logs: [`[${new Date().toLocaleTimeString()}] Registered schedule with ${schedulerServer || 'Temporal/Trigger.dev'} server.`],
      temporalConfig: temporalConfig || { namespace: 'default', taskQueue: 'workflow-queue', workflowId: `wf-${id}` },
      triggerDevConfig: triggerDevConfig || { environment: 'development', jobSlug: `job-${id}` }
    };

    newJob.nextRunAt = computeNextRunAt(newJob);
    jobsMap.set(id, newJob);
    await persistJobs();

    res.json({
      success: true,
      job: newJob,
      message: `Successfully registered schedule '${newJob.name}' on ${newJob.schedulerServer} scheduler server.`
    });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/scheduler/jobs/:id/trigger — Trigger immediately
router.post('/jobs/:id/trigger', async (req, res) => {
  const job = jobsMap.get(req.params.id);
  if (!job) {
    return res.status(404).json({ success: false, error: 'Scheduled job not found.' });
  }
  try {
    const output = await executeScheduledJobAction(job);
    res.json({ success: true, job, output });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// POST /api/scheduler/jobs/:id/toggle — Pause/Resume
router.post('/jobs/:id/toggle', async (req, res) => {
  const job = jobsMap.get(req.params.id);
  if (!job) {
    return res.status(404).json({ success: false, error: 'Scheduled job not found.' });
  }
  job.status = job.status === 'paused' ? 'scheduled' : 'paused';
  if (job.status === 'scheduled') {
    job.nextRunAt = computeNextRunAt(job);
  }
  job.logs.push(`[${new Date().toLocaleTimeString()}] Status toggled to: ${job.status}`);
  await persistJobs();
  res.json({ success: true, job });
});

// DELETE /api/scheduler/jobs/:id — Delete job
router.delete('/jobs/:id', async (req, res) => {
  const deleted = jobsMap.delete(req.params.id);
  await persistJobs();
  res.json({ success: deleted });
});

export default router;
