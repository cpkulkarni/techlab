/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { History, RotateCcw, Camera, RefreshCw } from 'lucide-react';

export default function TimeTravelBar({ theme = 'dark', onRestored }: { theme?: string; onRestored?: () => void }) {
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [description, setDescription] = useState('');
  const [feedback, setFeedback] = useState<string | null>(null);

  const isDark = theme === 'dark';

  const fetchSnapshots = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/workspace/snapshots/snapshots');
      const data = await res.json();
      if (data.success) {
        setSnapshots(data.snapshots);
      }
    } catch (e) {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSnapshots();
  }, []);

  const handleCreateSnapshot = async () => {
    try {
      const res = await fetch('/api/workspace/snapshots/snapshot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ description: description || 'Manual Time-Travel Checkpoint' }),
      });
      const data = await res.json();
      if (data.success) {
        setDescription('');
        setFeedback('Snapshot saved.');
        fetchSnapshots();
        setTimeout(() => setFeedback(null), 3000);
      }
    } catch (e: any) {
      setFeedback(`Error: ${e.message}`);
    }
  };

  const handleRestore = async (id: string) => {
    if (!confirm('Are you sure you want to rollback workspace files to this snapshot?')) return;
    setRestoringId(id);
    try {
      const res = await fetch('/api/workspace/snapshots/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ snapshotId: id }),
      });
      const data = await res.json();
      if (data.success) {
        setFeedback('Workspace restored to selected snapshot!');
        if (onRestored) onRestored();
        setTimeout(() => setFeedback(null), 4000);
      }
    } catch (e: any) {
      setFeedback(`Restore failed: ${e.message}`);
    } finally {
      setRestoringId(null);
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-3`}>
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <History className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-sm">Conversation & Code Time-Travel Rollback</h3>
        </div>
        <button onClick={fetchSnapshots} className="p-1 text-slate-400 hover:text-slate-200">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {feedback && (
        <div className="p-2 bg-indigo-500/10 text-indigo-300 text-xs rounded border border-indigo-500/20">
          {feedback}
        </div>
      )}

      {/* Save Checkpoint Control */}
      <div className="flex space-x-2">
        <input
          type="text"
          placeholder="Checkpoint label (e.g. Before refactoring auth)..."
          value={description}
          onChange={e => setDescription(e.target.value)}
          className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none focus:border-indigo-500"
        />
        <button
          onClick={handleCreateSnapshot}
          className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg shadow transition flex items-center space-x-1"
        >
          <Camera className="w-3.5 h-3.5" />
          <span>Save Checkpoint</span>
        </button>
      </div>

      {/* Snapshots Timeline List */}
      <div className="space-y-1.5 text-xs max-h-48 overflow-y-auto pt-1">
        {snapshots.length === 0 ? (
          <p className="text-slate-500 italic text-center py-2">No saved snapshots yet. Click "Save Checkpoint" to record state.</p>
        ) : (
          snapshots.map((snap) => (
            <div key={snap.id} className="flex items-center justify-between p-2 bg-slate-950 rounded-lg border border-slate-800/80 hover:border-slate-700 transition">
              <div className="truncate pr-2">
                <span className="font-medium text-slate-200 block truncate">{snap.description}</span>
                <span className="text-[10px] text-slate-500 font-mono">
                  {new Date(snap.timestamp).toLocaleTimeString()} ({snap.fileCount} files)
                </span>
              </div>
              <button
                onClick={() => handleRestore(snap.id)}
                disabled={restoringId === snap.id}
                className="px-2.5 py-1 bg-slate-800 hover:bg-indigo-600 text-slate-200 text-[11px] rounded transition flex items-center space-x-1 shrink-0 font-medium"
              >
                <RotateCcw className={`w-3 h-3 ${restoringId === snap.id ? 'animate-spin' : ''}`} />
                <span>Restore</span>
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
