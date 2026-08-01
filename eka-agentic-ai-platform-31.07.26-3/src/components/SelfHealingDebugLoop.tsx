/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Wrench, Play, CheckCircle2, AlertOctagon, RefreshCw, Cpu, Bug } from 'lucide-react';

export default function SelfHealingDebugLoop({ theme = 'dark' }: { theme?: string }) {
  const [errorOutput, setErrorOutput] = useState('');
  const [targetFilePath, setTargetFilePath] = useState('');
  const [verificationCommand, setVerificationCommand] = useState('npx tsc --noEmit');
  const [maxAttempts, setMaxAttempts] = useState(3);
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<any>(null);

  const isDark = theme === 'dark';

  const handleRunSelfHeal = async () => {
    if (!errorOutput.trim()) return;
    setRunning(true);
    setResults(null);
    try {
      const res = await fetch('/api/agent/self-heal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          errorOutput,
          targetFilePath: targetFilePath || undefined,
          verificationCommand,
          maxAttempts,
        }),
      });
      const data = await res.json();
      setResults(data);
    } catch (e: any) {
      setResults({ success: false, error: e.message });
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-4`}>
      <div className="flex items-center space-x-2 pb-3 border-b border-slate-800">
        <Bug className="w-5 h-5 text-emerald-400" />
        <div>
          <h3 className="font-semibold text-sm">Autonomous "Self-Healing" Debug Loop</h3>
          <p className="text-[11px] text-slate-400">Captures build/test failures and applies automated LLM repair cycles</p>
        </div>
      </div>

      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-slate-400 font-medium mb-1">Stack Trace or Build Error Output</label>
          <textarea
            rows={4}
            placeholder="Paste stack trace or build error log (e.g. TypeError: Cannot read properties of undefined in App.tsx:42)..."
            value={errorOutput}
            onChange={e => setErrorOutput(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-200 font-mono text-[11px] outline-none focus:border-indigo-500"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="block text-slate-400 font-medium mb-1">Target File (Optional)</label>
            <input
              type="text"
              placeholder="src/App.tsx"
              value={targetFilePath}
              onChange={e => setTargetFilePath(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 font-medium mb-1">Verification Command</label>
            <input
              type="text"
              value={verificationCommand}
              onChange={e => setVerificationCommand(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500 font-mono text-[11px]"
            />
          </div>
        </div>

        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center space-x-2">
            <span className="text-slate-400">Max Retry Attempts:</span>
            <select
              value={maxAttempts}
              onChange={e => setMaxAttempts(Number(e.target.value))}
              className="bg-slate-950 border border-slate-800 rounded px-2 py-1 text-slate-200 outline-none"
            >
              <option value={1}>1 Attempt</option>
              <option value={3}>3 Attempts</option>
              <option value={5}>5 Attempts</option>
            </select>
          </div>

          <button
            onClick={handleRunSelfHeal}
            disabled={running || !errorOutput.trim()}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium rounded-lg shadow transition flex items-center space-x-1.5"
          >
            {running ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Wrench className="w-4 h-4" />}
            <span>{running ? 'Repairing Code...' : 'Start Self-Healing Repair'}</span>
          </button>
        </div>

        {/* Results Log Display */}
        {results && (
          <div className="mt-4 p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <span className={`font-bold flex items-center space-x-1.5 ${results.healed ? 'text-emerald-400' : 'text-rose-400'}`}>
                {results.healed ? <CheckCircle2 className="w-4 h-4" /> : <AlertOctagon className="w-4 h-4" />}
                <span>{results.healed ? `Self-Healed Successfully in ${results.totalAttempts} Attempt(s)!` : 'Repair Failed After Max Attempts'}</span>
              </span>
            </div>

            {results.logs && results.logs.length > 0 && (
              <div className="space-y-2">
                {results.logs.map((log: any, idx: number) => (
                  <div key={idx} className="p-2.5 bg-slate-900 rounded border border-slate-800 text-[11px] space-y-1">
                    <div className="flex items-center justify-between font-semibold">
                      <span className="text-indigo-400">Attempt #{log.attempt}</span>
                      <span className={log.success ? 'text-emerald-400' : 'text-rose-400'}>
                        {log.success ? 'PASSED' : 'FAILED'}
                      </span>
                    </div>
                    <p className="text-slate-300 font-sans">{log.diagnosis}</p>
                    {log.fixApplied && (
                      <p className="text-slate-400 font-mono text-[10px]">Applied: {log.fixApplied}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
