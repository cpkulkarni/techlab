/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  History, 
  Plus, 
  RotateCcw, 
  GitCommit, 
  Check, 
  FileCode, 
  Clock, 
  Save, 
  Tag, 
  ShieldCheck, 
  Layers,
  ArrowRight
} from 'lucide-react';

export interface Checkpoint {
  id: string;
  title: string;
  description: string;
  timestamp: string;
  filesCount: number;
  author: string;
  type: 'manual' | 'agent' | 'auto';
  tags: string[];
}

export function TimeTravelTab() {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([
    {
      id: 'chk-1',
      title: 'Initial Workspace Scaffold',
      description: 'Clean baseline repository state before agent execution.',
      timestamp: new Date(Date.now() - 3600000 * 3).toLocaleTimeString(),
      filesCount: 14,
      author: 'System Scaffold',
      type: 'auto',
      tags: ['baseline', 'v1.0.0']
    },
    {
      id: 'chk-2',
      title: 'Multi-Agent Navigation & Dropdown Integrations',
      description: 'Configured top-level tools dropdown and agent execution flow integration.',
      timestamp: new Date(Date.now() - 3600000 * 1).toLocaleTimeString(),
      filesCount: 18,
      author: 'Studio AI Agent',
      type: 'agent',
      tags: ['ui-updates', 'dropdown']
    }
  ]);

  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint | null>(checkpoints[1] || null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [newTag, setNewTag] = useState('');
  const [isRestoring, setIsRestoring] = useState(false);
  const [restoreSuccess, setRestoreSuccess] = useState<string | null>(null);

  const handleCreateCheckpoint = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim()) return;

    const newChk: Checkpoint = {
      id: `chk-${Date.now()}`,
      title: newTitle.trim(),
      description: newDesc.trim() || 'Manual user snapshot checkpoint.',
      timestamp: new Date().toLocaleTimeString(),
      filesCount: 22,
      author: 'User Developer',
      type: 'manual',
      tags: newTag.trim() ? [newTag.trim()] : ['manual-snapshot']
    };

    setCheckpoints([newChk, ...checkpoints]);
    setSelectedCheckpoint(newChk);
    setNewTitle('');
    setNewDesc('');
    setNewTag('');
    setShowCreateModal(false);
  };

  const handleRestoreCheckpoint = (chk: Checkpoint) => {
    setIsRestoring(true);
    setTimeout(() => {
      setIsRestoring(false);
      setRestoreSuccess(`Workspace successfully restored to snapshot "${chk.title}"!`);
      setTimeout(() => setRestoreSuccess(null), 3500);
    }, 1000);
  };

  return (
    <div className="h-full bg-slate-950 p-4 overflow-y-auto space-y-4 font-mono text-xs">
      {/* Header & Actions Bar */}
      <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-3 gap-2">
        <div className="flex items-center gap-2">
          <div className="p-1.5 bg-sky-950/80 border border-sky-800 rounded-lg text-sky-400">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-bold text-sky-400">Time Travel & Checkpoint Manager</h2>
            <p className="text-[10px] text-slate-400">Save workspace snapshots and restore previous code states anytime.</p>
          </div>
        </div>

        <button
          type="button"
          onClick={() => setShowCreateModal(true)}
          className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-lg flex items-center gap-1.5 transition cursor-pointer shadow-md"
        >
          <Save className="w-3.5 h-3.5" />
          <span>Save New Checkpoint</span>
        </button>
      </div>

      {/* Restore Notification */}
      {restoreSuccess && (
        <div className="bg-emerald-950/80 border border-emerald-800 p-3 rounded-xl text-emerald-200 flex items-center gap-2 animate-fadeIn">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{restoreSuccess}</span>
        </div>
      )}

      {/* Create Checkpoint Modal Form */}
      {showCreateModal && (
        <form onSubmit={handleCreateCheckpoint} className="p-4 bg-slate-900 border border-indigo-500/50 rounded-xl space-y-3 shadow-2xl">
          <div className="flex justify-between items-center border-b border-slate-800 pb-2">
            <span className="font-bold text-indigo-300 flex items-center gap-1.5">
              <Plus className="w-4 h-4 text-indigo-400" /> Save Workspace Snapshot
            </span>
            <button 
              type="button" 
              onClick={() => setShowCreateModal(false)}
              className="text-slate-500 hover:text-white"
            >
              ✕
            </button>
          </div>

          <div className="space-y-2">
            <div>
              <label className="block text-[10px] uppercase text-slate-400 mb-1 font-bold">Checkpoint Title *</label>
              <input
                autoFocus
                type="text"
                value={newTitle}
                onChange={(e) => setNewTitle(e.target.value)}
                placeholder="e.g. Before refactoring auth service"
                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                required
              />
            </div>

            <div>
              <label className="block text-[10px] uppercase text-slate-400 mb-1 font-bold">Description</label>
              <input
                type="text"
                value={newDesc}
                onChange={(e) => setNewDesc(e.target.value)}
                placeholder="Brief summary of code modifications..."
                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>

            <div>
              <label className="block text-[10px] uppercase text-slate-400 mb-1 font-bold">Tag (Optional)</label>
              <input
                type="text"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
                placeholder="e.g. stable, feature-x, backup"
                className="w-full bg-slate-950 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>

          <div className="flex justify-end gap-2 pt-2 border-t border-slate-800">
            <button
              type="button"
              onClick={() => setShowCreateModal(false)}
              className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded cursor-pointer"
            >
              Save Checkpoint
            </button>
          </div>
        </form>
      )}

      {/* Main Grid Split View */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Timeline Checkpoints List */}
        <div className="lg:col-span-5 space-y-2">
          <div className="text-[10px] uppercase font-bold text-slate-500 px-1 flex items-center justify-between">
            <span>Saved Checkpoints ({checkpoints.length})</span>
            <span className="text-sky-400 font-normal">Active Branch: main</span>
          </div>

          <div className="space-y-2">
            {checkpoints.map((chk) => {
              const isSelected = selectedCheckpoint?.id === chk.id;
              return (
                <div
                  key={chk.id}
                  onClick={() => setSelectedCheckpoint(chk)}
                  className={`p-3 rounded-xl border transition cursor-pointer relative ${
                    isSelected
                      ? 'bg-slate-900 border-sky-500 shadow-lg'
                      : 'bg-slate-900/40 hover:bg-slate-900 border-slate-800 hover:border-slate-700'
                  }`}
                >
                  <div className="flex justify-between items-start gap-2 mb-1">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <GitCommit className={`w-4 h-4 shrink-0 ${chk.type === 'manual' ? 'text-indigo-400' : 'text-purple-400'}`} />
                      <span className="font-bold text-slate-200 truncate">{chk.title}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 shrink-0 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {chk.timestamp}
                    </span>
                  </div>

                  <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed mb-2">
                    {chk.description}
                  </p>

                  <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[10px] text-slate-500">
                    <span className="flex items-center gap-1">
                      <FileCode className="w-3 h-3 text-emerald-400" />
                      <span>{chk.filesCount} files recorded</span>
                    </span>
                    <div className="flex gap-1">
                      {chk.tags.map(t => (
                        <span key={t} className="bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded text-[9px] font-bold">
                          #{t}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Snapshot Detail & Restore View */}
        <div className="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-4">
          {selectedCheckpoint ? (
            <div className="space-y-4">
              <div className="flex justify-between items-start border-b border-slate-800 pb-3">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Tag className="w-4 h-4 text-sky-400" />
                    <h3 className="text-sm font-bold text-white">{selectedCheckpoint.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400">{selectedCheckpoint.description}</p>
                </div>

                <button
                  type="button"
                  onClick={() => handleRestoreCheckpoint(selectedCheckpoint)}
                  disabled={isRestoring}
                  className="px-3 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-bold rounded-lg flex items-center gap-1.5 cursor-pointer shadow transition"
                >
                  <RotateCcw className={`w-3.5 h-3.5 ${isRestoring ? 'animate-spin' : ''}`} />
                  <span>{isRestoring ? 'Restoring Snapshot...' : 'Restore Snapshot'}</span>
                </button>
              </div>

              {/* Metadata Stats */}
              <div className="grid grid-cols-3 gap-2 text-center font-mono">
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="block text-[10px] text-slate-500">Author</span>
                  <span className="font-bold text-slate-200">{selectedCheckpoint.author}</span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="block text-[10px] text-slate-500">Created At</span>
                  <span className="font-bold text-slate-200">{selectedCheckpoint.timestamp}</span>
                </div>
                <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                  <span className="block text-[10px] text-slate-500">Files Tracked</span>
                  <span className="font-bold text-emerald-400">{selectedCheckpoint.filesCount} Files</span>
                </div>
              </div>

              {/* Snapshot Content Highlights */}
              <div className="space-y-2">
                <span className="text-[10px] uppercase font-bold text-slate-400 block">Snapshot Delta & Modified Files</span>
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3 space-y-2 font-mono text-[11px]">
                  <div className="flex items-center justify-between text-emerald-400">
                    <span>+ src/App.tsx</span>
                    <span className="text-[10px] text-slate-500">Modified</span>
                  </div>
                  <div className="flex items-center justify-between text-emerald-400">
                    <span>+ src/components/WorkspaceTabs.tsx</span>
                    <span className="text-[10px] text-slate-500">Modified</span>
                  </div>
                  <div className="flex items-center justify-between text-indigo-400">
                    <span>+ src/components/workspace/TimeTravelTab.tsx</span>
                    <span className="text-[10px] text-slate-500">Created</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-400">
                    <span>• package.json</span>
                    <span className="text-[10px] text-slate-500">Unchanged</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center py-12 text-slate-500 text-center">
              <Layers className="w-8 h-8 mb-2 text-slate-700" />
              <p>Select a checkpoint to view snapshot details and restore options.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
