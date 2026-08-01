/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { GitBranch, GitCommit, GitPullRequest, RefreshCw, CheckCircle, Plus, ShieldAlert, Sparkles, Terminal } from 'lucide-react';

export default function GitControlPanel({ theme = 'dark' }: { theme?: string }) {
  const [loading, setLoading] = useState(false);
  const [gitStatus, setGitStatus] = useState<any>(null);
  const [commitMsg, setCommitMsg] = useState('');
  const [branchName, setBranchName] = useState('');
  const [showCommitModal, setShowCommitModal] = useState(false);
  const [prSummary, setPrSummary] = useState<any>(null);
  const [generatingPr, setGeneratingPr] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const isDark = theme === 'dark';

  const fetchStatus = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/git/status');
      const data = await res.json();
      if (data.success) {
        setGitStatus(data);
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: 'Failed to fetch git status' });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const handleInitRepo = async () => {
    try {
      const res = await fetch('/api/git/init', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setFeedback({ type: 'success', text: data.message });
        fetchStatus();
      } else {
        setFeedback({ type: 'error', text: data.error });
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: e.message });
    }
  };

  const handleStageAll = async () => {
    try {
      const res = await fetch('/api/git/stage', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: ['.'] }),
      });
      const data = await res.json();
      if (data.success) {
        setFeedback({ type: 'success', text: 'All changes staged.' });
        fetchStatus();
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: e.message });
    }
  };

  const handleExecuteCommit = async () => {
    if (!commitMsg.trim()) return;
    try {
      const res = await fetch('/api/git/commit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: commitMsg, confirmed: true }),
      });
      const data = await res.json();
      if (data.success) {
        setFeedback({ type: 'success', text: 'Commit created successfully!' });
        setCommitMsg('');
        setShowCommitModal(false);
        fetchStatus();
      } else {
        setFeedback({ type: 'error', text: data.error });
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: e.message });
    }
  };

  const handleCreateBranch = async () => {
    if (!branchName.trim()) return;
    try {
      const res = await fetch('/api/git/branch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ branchName }),
      });
      const data = await res.json();
      if (data.success) {
        setFeedback({ type: 'success', text: data.message });
        setBranchName('');
        fetchStatus();
      } else {
        setFeedback({ type: 'error', text: data.error });
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: e.message });
    }
  };

  const handleGeneratePr = async () => {
    setGeneratingPr(true);
    try {
      const res = await fetch('/api/git/pr-summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      if (data.success) {
        setPrSummary(data);
      }
    } catch (e: any) {
      setFeedback({ type: 'error', text: e.message });
    } finally {
      setGeneratingPr(false);
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-4`}>
      <div className="flex items-center justify-between pb-3 border-b border-slate-700/50">
        <div className="flex items-center space-x-2">
          <GitBranch className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-base">Native Git Control & PR Generator</h3>
        </div>
        <button
          onClick={fetchStatus}
          disabled={loading}
          className="p-1.5 rounded-lg hover:bg-slate-800 transition text-slate-400 hover:text-slate-200"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {feedback && (
        <div className={`p-2.5 rounded-lg text-xs flex items-center justify-between ${feedback.type === 'success' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'}`}>
          <span>{feedback.text}</span>
          <button onClick={() => setFeedback(null)} className="hover:opacity-80">✕</button>
        </div>
      )}

      {!gitStatus?.isRepo ? (
        <div className="p-6 text-center space-y-3">
          <p className="text-sm text-slate-400">Workspace is not a Git repository yet.</p>
          <button
            onClick={handleInitRepo}
            className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium rounded-lg shadow transition"
          >
            Initialize Git Repository
          </button>
        </div>
      ) : (
        <div className="space-y-4 text-xs">
          {/* Branch & Actions Header */}
          <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50">
            <div className="flex items-center space-x-2">
              <span className="px-2 py-1 bg-indigo-500/20 text-indigo-300 font-mono rounded text-[11px] font-semibold">
                Branch: {gitStatus.branch}
              </span>
              <span className="text-slate-400">({gitStatus.files?.length || 0} modified files)</span>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleStageAll}
                className="px-2.5 py-1 bg-slate-700 hover:bg-slate-600 text-slate-200 rounded transition flex items-center space-x-1"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Stage All</span>
              </button>
              <button
                onClick={() => setShowCommitModal(true)}
                className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white rounded transition flex items-center space-x-1 font-medium"
              >
                <GitCommit className="w-3.5 h-3.5" />
                <span>Commit...</span>
              </button>
            </div>
          </div>

          {/* Changed Files List */}
          <div className="space-y-1">
            <h4 className="font-semibold text-slate-400 text-[11px] uppercase tracking-wider">Working Directory Status</h4>
            {gitStatus.files?.length === 0 ? (
              <p className="text-slate-500 py-2 italic text-center">Working tree clean. No uncommitted changes.</p>
            ) : (
              <div className="max-h-40 overflow-y-auto space-y-1 font-mono text-[11px] border border-slate-800 rounded p-2 bg-slate-950/40">
                {gitStatus.files.map((f: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between py-0.5 px-1 hover:bg-slate-800/40 rounded">
                    <span className="truncate">{f.path}</span>
                    <span className={`text-[10px] uppercase font-bold px-1.5 py-0.2 rounded ${f.status === 'staged' ? 'text-emerald-400 bg-emerald-950/60' : f.status === 'untracked' ? 'text-amber-400 bg-amber-950/60' : 'text-sky-400 bg-sky-950/60'}`}>
                      {f.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Branch Switcher & PR Generation */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-2">
            <div className="p-3 bg-slate-800/30 rounded-lg border border-slate-800 space-y-2">
              <h4 className="font-semibold text-slate-300">Create New Branch</h4>
              <div className="flex space-x-2">
                <input
                  type="text"
                  placeholder="feature/new-agent-mode"
                  value={branchName}
                  onChange={e => setBranchName(e.target.value)}
                  className="flex-1 bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 outline-none focus:border-indigo-500"
                />
                <button
                  onClick={handleCreateBranch}
                  className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded transition"
                >
                  Create
                </button>
              </div>
            </div>

            <div className="p-3 bg-slate-800/30 rounded-lg border border-slate-800 space-y-2">
              <h4 className="font-semibold text-slate-300 flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
                <span>AI Pull Request Summary</span>
              </h4>
              <button
                onClick={handleGeneratePr}
                disabled={generatingPr}
                className="w-full py-1.5 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-200 border border-indigo-500/30 rounded transition flex items-center justify-center space-x-1 font-medium"
              >
                <GitPullRequest className="w-3.5 h-3.5" />
                <span>{generatingPr ? 'Analyzing Diff...' : 'Generate PR Summary'}</span>
              </button>
            </div>
          </div>

          {/* Generated PR Card */}
          {prSummary && (
            <div className="p-3 bg-indigo-950/30 border border-indigo-500/30 rounded-lg space-y-2">
              <div className="flex items-center justify-between border-b border-indigo-500/20 pb-2">
                <h5 className="font-bold text-indigo-300 text-xs">{prSummary.title}</h5>
                <button onClick={() => setPrSummary(null)} className="text-slate-400 hover:text-slate-200">✕</button>
              </div>
              <div className="text-[11px] text-slate-300 whitespace-pre-wrap max-h-48 overflow-y-auto font-sans leading-relaxed">
                {prSummary.description}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Explicit User Confirm Commit Modal */}
      {showCommitModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-700 rounded-xl max-w-md w-full p-5 space-y-4 shadow-2xl">
            <div className="flex items-center space-x-2 text-amber-400">
              <ShieldAlert className="w-5 h-5" />
              <h4 className="font-bold text-slate-100 text-sm">Explicit Git Commit Confirmation</h4>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">
              You are about to execute a native Git commit. Please write a clear commit message and confirm:
            </p>
            <textarea
              rows={3}
              placeholder="feat: implement multi-agent workspace boundaries"
              value={commitMsg}
              onChange={e => setCommitMsg(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg p-2.5 text-xs text-slate-100 outline-none focus:border-indigo-500 font-mono"
            />
            <div className="flex items-center justify-end space-x-2 pt-2">
              <button
                onClick={() => setShowCommitModal(false)}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs rounded-lg transition"
              >
                Cancel
              </button>
              <button
                onClick={handleExecuteCommit}
                disabled={!commitMsg.trim()}
                className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg transition shadow"
              >
                Confirm & Commit
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
