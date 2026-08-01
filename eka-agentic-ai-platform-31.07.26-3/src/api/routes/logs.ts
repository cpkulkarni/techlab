/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';
import { interactionLogs } from '../shared/logs.js';

const router = Router();

// GET /api/interaction-logs
router.get('/', async (req, res) => {
  try {
    res.json({ success: true, logs: [...interactionLogs].reverse() });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// DELETE /api/interaction-logs
router.delete('/', async (req, res) => {
  try {
    interactionLogs.length = 0;

    // Clear old legacy .logs file if exists
    const oldLogFilePath = path.join(process.cwd(), '.logs', 'interaction_logs.jsonl');
    if (existsSync(oldLogFilePath)) await fs.unlink(oldLogFilePath);

    // Clear old pre-migration root-level Eka-Agentic-AI-platform-logs-* dirs
    const rootItems = await fs.readdir(process.cwd(), { withFileTypes: true });
    for (const item of rootItems) {
      if (item.isDirectory() && item.name.startsWith('Eka-Agentic-AI-platform-logs-')) {
        const p = path.join(process.cwd(), item.name, 'interaction_logs.jsonl');
        if (existsSync(p)) await fs.unlink(p);
      }
    }

    // Clear new app-log/<date>/interaction_logs.jsonl
    const appLogDir = path.join(process.cwd(), 'app-log');
    if (existsSync(appLogDir)) {
      const dateItems = await fs.readdir(appLogDir, { withFileTypes: true });
      for (const item of dateItems) {
        if (item.isDirectory() && item.name.startsWith('Eka-Agentic-AI-platform-logs-')) {
          const p = path.join(appLogDir, item.name, 'interaction_logs.jsonl');
          if (existsSync(p)) await fs.unlink(p);
        }
      }
    }

    res.json({ success: true, message: 'Logs cleared successfully.' });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
