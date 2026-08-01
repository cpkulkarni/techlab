/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';

const router = Router();

const APP_CONFIG_DIR = () => path.join(process.cwd(), 'app-config');
const APP_CONFIG_FILE = () => path.join(APP_CONFIG_DIR(), 'app-config.json');
const LLM_CONFIG_DIR = () => path.join(process.cwd(), 'app-config', 'llm');
const LLM_CONFIG_FILE = () => path.join(LLM_CONFIG_DIR(), 'llm.config');

// GET /api/config — return full persisted app configuration from /app-config/app-config.json
router.get('/', async (req, res) => {
  try {
    const mainFile = APP_CONFIG_FILE();
    if (existsSync(mainFile)) {
      const raw = await fs.readFile(mainFile, 'utf8');
      return res.json({ success: true, config: JSON.parse(raw) });
    }

    const llmFile = LLM_CONFIG_FILE();
    if (existsSync(llmFile)) {
      const raw = await fs.readFile(llmFile, 'utf8');
      return res.json({ success: true, config: { modelConfig: JSON.parse(raw) } });
    }

    return res.json({ success: true, config: null });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/config — persist full app configuration to /app-config/app-config.json
router.post('/', async (req, res) => {
  try {
    const dir = APP_CONFIG_DIR();
    if (!existsSync(dir)) await fs.mkdir(dir, { recursive: true });

    const configData = req.body || {};
    await fs.writeFile(APP_CONFIG_FILE(), JSON.stringify(configData, null, 2), 'utf8');

    // Also sync llm.config if modelConfig is provided
    if (configData.modelConfig) {
      const llmDir = LLM_CONFIG_DIR();
      if (!existsSync(llmDir)) await fs.mkdir(llmDir, { recursive: true });
      await fs.writeFile(LLM_CONFIG_FILE(), JSON.stringify(configData.modelConfig, null, 2), 'utf8');
    }

    return res.json({ success: true, path: APP_CONFIG_FILE() });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/config/llm — return persisted LLM config
router.get('/llm', async (req, res) => {
  try {
    const file = LLM_CONFIG_FILE();
    if (existsSync(file)) {
      const raw = await fs.readFile(file, 'utf8');
      res.json({ success: true, config: JSON.parse(raw) });
    } else {
      res.json({ success: true, config: null });
    }
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// POST /api/config/llm — persist LLM config
router.post('/llm', async (req, res) => {
  try {
    const dir = LLM_CONFIG_DIR();
    if (!existsSync(dir)) await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(LLM_CONFIG_FILE(), JSON.stringify(req.body, null, 2), 'utf8');
    res.json({ success: true });
  } catch (error: any) {
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;

