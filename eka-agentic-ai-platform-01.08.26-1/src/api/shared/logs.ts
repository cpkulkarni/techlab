/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import path from 'path';
import fs from 'fs/promises';
import { existsSync } from 'fs';

export interface InteractionLog {
  id: string;
  timestamp: string;
  type: string;
  prompt: string;
  systemInstruction?: string;
  response: string;
  model: string;
  citations?: Array<{ title: string; url: string }>;
}

// In-memory log store (populated from file on startup)
export const interactionLogs: InteractionLog[] = [];

export async function writeLogToEkaDirectory(logEntry: InteractionLog) {
  try {
    const d = new Date();
    const dd = String(d.getDate()).padStart(2, '0');
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const yyyy = d.getFullYear();
    const childDirName = `Eka-Agentic-AI-platform-logs-${dd}.${mm}.${yyyy}`;
    const logDir = path.join(process.cwd(), 'app-log', childDirName);
    if (!existsSync(logDir)) {
      await fs.mkdir(logDir, { recursive: true });
    }
    const logFilePath = path.join(logDir, 'interaction_logs.jsonl');
    await fs.appendFile(logFilePath, JSON.stringify(logEntry) + '\n', 'utf8');
  } catch (e) {
    console.error('Error saving persistent log:', e);
  }
}

export async function loadTodayLogs() {
  try {
    const d = new Date();
    const dd = String(d.getDate()).padStart(2, '0');
    const mm = String(d.getMonth() + 1).padStart(2, '0');
    const yyyy = d.getFullYear();
    const childDirName = `Eka-Agentic-AI-platform-logs-${dd}.${mm}.${yyyy}`;
    const logFilePath = path.join(process.cwd(), 'app-log', childDirName, 'interaction_logs.jsonl');
    if (existsSync(logFilePath)) {
      const content = await fs.readFile(logFilePath, 'utf8');
      const lines = content.trim().split('\n');
      for (const line of lines) {
        if (line.trim()) {
          try {
            interactionLogs.push(JSON.parse(line));
          } catch (pe) {
            console.error('Error parsing line in logs:', pe);
          }
        }
      }
      console.log(`Loaded ${interactionLogs.length} previous logs from today's Eka log directory.`);
    }
  } catch (e) {
    console.error("Error reading today's persistent logs:", e);
  }
}
