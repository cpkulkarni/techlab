/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import { spawn, execSync } from 'child_process';

const router = Router();

let mailServerProcess: any = null;

// GET /api/mailserver/status
router.get('/status', (req, res) => {
  res.json({
    success: true,
    running: mailServerProcess !== null && !mailServerProcess.killed,
    port: 1025
  });
});

// POST /api/mailserver/start
router.post('/start', (req, res) => {
  if (mailServerProcess && !mailServerProcess.killed) {
    return res.json({ success: true, message: 'Mail server is already running.', running: true });
  }

  try {
    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
    mailServerProcess = spawn(pythonCmd, ['smtp_server.py'], { cwd: process.cwd(), detached: false });

    mailServerProcess.stdout.on('data', (data: any) => {
      console.log(`[Python Mail Server]: ${data.toString().trim()}`);
    });
    mailServerProcess.stderr.on('data', (data: any) => {
      console.error(`[Python Mail Server Error]: ${data.toString().trim()}`);
    });
    mailServerProcess.on('close', (code: any) => {
      console.log(`[Python Mail Server] Exited with code ${code}`);
      mailServerProcess = null;
    });

    setTimeout(() => {
      res.json({
        success: mailServerProcess !== null && !mailServerProcess.killed,
        running: mailServerProcess !== null && !mailServerProcess.killed,
        message: mailServerProcess !== null && !mailServerProcess.killed
          ? 'Mail server started successfully.'
          : 'Mail server failed to start.'
      });
    }, 1000);
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/mailserver/stop
router.post('/stop', (req, res) => {
  if (!mailServerProcess || mailServerProcess.killed) {
    return res.json({ success: true, message: 'Mail server is not running.', running: false });
  }

  try {
    if (process.platform === 'win32') {
      execSync(`taskkill /pid ${mailServerProcess.pid} /t /f`);
    } else {
      mailServerProcess.kill('SIGINT');
    }
    mailServerProcess = null;
    res.json({ success: true, message: 'Mail server stopped successfully.', running: false });
  } catch (error: any) {
    mailServerProcess = null;
    res.json({ success: true, message: 'Mail server stopped.', running: false });
  }
});

export default router;
