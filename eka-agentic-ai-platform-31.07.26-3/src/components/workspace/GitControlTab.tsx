/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { GitBranch, GitCommit, RefreshCw, CheckCircle } from 'lucide-react';

export function GitControlTab() {
  const [branch, setBranch] = useState('main');
  const [statusText, setStatusText] = useState('Clean working tree. All changes committed.');
  const [commitMsg, setCommitMsg] = useState('');
  const [commits, setCommits] = useState<{ hash: string; msg: string; date: string }[]>([
    { hash: 'a1b2c3d', msg: 'refactor: split files > 500 lines into modular components', date: new Date().toLocaleTimeString() },
    { hash: 'e5f6g7h', msg: 'feat: add decision node to workflow builder', date: '10 mins ago' },
  ]);

  const handleCommit = () => {
    if (!commitMsg.trim()) return;
    const newCommit = {
      hash: Math.random().toString(16).slice(2, 9),
      msg: commitMsg,
      date: new Date().toLocaleTimeString(),
    };
    setCommits([newCommit, ...commits]);
    setCommitMsg('');
    setStatusText('Commit created successfully.');
  };

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-orange-400 flex items-center gap-1.5">
          <GitBranch className="w-4 h-4" /> Git Version Control Panel
        </span>
        <span className="text-[10px] font-mono text-slate-300 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded">
          Branch: <strong className="text-orange-400">{branch}</strong>
        </span>
      </div>

      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <div>
          <label className="text-xs font-mono text-slate-300 block mb-1">Commit Message:</label>
          <input
            type="text"
            value={commitMsg}
            onChange={(e) => setCommitMsg(e.target.value)}
            placeholder="e.g. feat: add decision branching node"
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-orange-500"
          />
        </div>

        <button
          type="button"
          onClick={handleCommit}
          disabled={!commitMsg.trim()}
          className="bg-orange-600 hover:bg-orange-500 disabled:opacity-40 text-white font-mono text-xs font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow-md"
        >
          <GitCommit className="w-4 h-4" /> Create Local Snapshot Commit
        </button>
      </div>

      {/* Commit History */}
      <div className="space-y-2">
        <span className="text-xs font-mono font-bold text-slate-300 block">Commit History Log:</span>
        <div className="space-y-1.5">
          {commits.map(c => (
            <div key={c.hash} className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg flex items-center justify-between text-xs font-mono">
              <div className="flex items-center gap-2">
                <span className="text-orange-400 font-bold">{c.hash}</span>
                <span className="text-slate-200">{c.msg}</span>
              </div>
              <span className="text-[10px] text-slate-400">{c.date}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
