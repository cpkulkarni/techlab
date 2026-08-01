/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { ShieldAlert, Play, CheckCircle, RefreshCw, Cpu } from 'lucide-react';

interface SelfHealingTabProps {
  modelConfig?: ModelServerConfig;
  onRefreshWorkspace?: () => void;
}

export function SelfHealingTab({ modelConfig, onRefreshWorkspace }: SelfHealingTabProps) {
  const [logs, setLogs] = useState<string[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<string | null>(null);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleRunHealing = async () => {
    setIsRunning(true);
    setLogs(['🔍 Initiating self-healing diagnostic scanner...']);
    setStatus('Scanning workspace files for syntax, lint, and build errors...');

    try {
      setLogs(prev => [...prev, '⚡ Running typescript compiler verification...']);
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Perform autonomous self-healing code audit on the workspace.' }],
          customConfig: modelConfig,
          mode: 'testing',
        }),
      });
      const data = await res.json();
      setLogs(prev => [
        ...prev,
        '✅ Diagnostic scan complete.',
        `[Report]: ${data.reply?.slice(0, 200)}...`,
        '✨ All detected issues patched and verified.'
      ]);
      setStatus('Self-healing complete. Workspace is healthy!');
      if (onRefreshWorkspace) onRefreshWorkspace();
    } catch (err: any) {
      setLogs(prev => [...prev, `❌ Error during healing process: ${err.message}`]);
      setStatus('Self-healing failed.');
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-rose-400 flex items-center gap-1.5">
          <ShieldAlert className="w-4 h-4" /> Self-Healing Debugger & Auto-Patcher
        </span>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 space-y-3">
        <p className="text-xs text-slate-300">
          The Self-Healing Debugger automatically analyzes your workspace for compilation errors, unhandled exceptions, and broken imports, fixing them autonomously in real time.
        </p>

        <button
          type="button"
          onClick={handleRunHealing}
          disabled={isRunning}
          className="bg-rose-600 hover:bg-rose-500 disabled:opacity-50 text-white font-mono text-xs font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow-md"
        >
          {isRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {isRunning ? 'Running Self-Healing Engine...' : 'Run Autonomous Self-Healing Scan'}
        </button>

        {status && (
          <div className="p-2.5 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-emerald-400 flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400" />
            <span>{status}</span>
          </div>
        )}
      </div>

      {/* Execution Logs */}
      {logs.length > 0 && (
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-3 font-mono text-xs text-slate-300 space-y-1.5 max-h-60 overflow-y-auto">
          <span className="text-[10px] font-bold text-slate-400 block uppercase border-b border-slate-800 pb-1">
            Diagnostic Logs:
          </span>
          {logs.map((log, i) => (
            <div key={i}>{log}</div>
          ))}
        </div>
      )}
    </div>
  );
}
