/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import fs from 'fs/promises';
import path from 'path';
import { getWorkspaceDir, collectDirFiles } from '../shared/workspace.js';
import { assertWorkspaceBoundary } from '../shared/workspaceGuard.js';

const router = Router();

export interface WorkspaceSnapshot {
  id: string;
  timestamp: string;
  description: string;
  files: Record<string, string>; // relativePath -> content
}

// In-memory snapshots array
const snapshotsStore: WorkspaceSnapshot[] = [];

// POST /api/workspace/snapshot - Create a new time-travel snapshot
router.post('/snapshot', async (req, res) => {
  const { description = 'Automated Turn Snapshot' } = req.body;
  const workspaceDir = getWorkspaceDir();

  try {
    const fileList = await collectDirFiles(workspaceDir, workspaceDir);
    const fileSnapshots: Record<string, string> = {};

    for (const f of fileList) {
      if (f.path.includes('node_modules') || f.path.includes('dist') || f.path.includes('.git')) continue;
      fileSnapshots[f.path] = f.content;
    }

    const snapshot: WorkspaceSnapshot = {
      id: `snap_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
      timestamp: new Date().toISOString(),
      description,
      files: fileSnapshots
    };

    snapshotsStore.unshift(snapshot);
    // Keep max 20 snapshots
    if (snapshotsStore.length > 20) snapshotsStore.pop();

    return res.json({
      success: true,
      snapshot: {
        id: snapshot.id,
        timestamp: snapshot.timestamp,
        description: snapshot.description,
        fileCount: Object.keys(snapshot.files).length
      }
    });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

// GET /api/workspace/snapshots - List all available snapshots
router.get('/snapshots', (req, res) => {
  const list = snapshotsStore.map(s => ({
    id: s.id,
    timestamp: s.timestamp,
    description: s.description,
    fileCount: Object.keys(s.files).length
  }));
  return res.json({ success: true, snapshots: list });
});

// POST /api/workspace/restore - Rollback workspace files to selected snapshot
router.post('/restore', async (req, res) => {
  const { snapshotId } = req.body;
  if (!snapshotId) return res.status(400).json({ success: false, error: 'snapshotId is required' });

  const targetSnapshot = snapshotsStore.find(s => s.id === snapshotId);
  if (!targetSnapshot) return res.status(404).json({ success: false, error: 'Snapshot not found' });

  const workspaceDir = getWorkspaceDir();
  try {
    let restoredCount = 0;
    for (const [relPath, content] of Object.entries(targetSnapshot.files)) {
      const fullPath = assertWorkspaceBoundary(path.join(workspaceDir, relPath), workspaceDir);
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      await fs.writeFile(fullPath, content, 'utf8');
      restoredCount++;
    }

    return res.json({
      success: true,
      message: `Successfully rolled back workspace to snapshot ${snapshotId}`,
      restoredFilesCount: restoredCount,
      timestamp: targetSnapshot.timestamp
    });
  } catch (error: any) {
    return res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
