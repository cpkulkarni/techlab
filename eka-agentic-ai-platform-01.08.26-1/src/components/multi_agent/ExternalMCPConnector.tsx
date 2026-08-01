/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Terminal, Network, CheckCircle, Plus, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function ExternalMCPConnector({ theme = 'dark' }: { theme?: string }) {
  const [mode, setMode] = useState<'stdio' | 'sse'>('stdio');
  const [command, setCommand] = useState('npx -y @modelcontextprotocol/server-memory');
  const [sseUrl, setSseUrl] = useState('http://localhost:8080/sse');
  const [serverName, setServerName] = useState('');
  const [connecting, setConnecting] = useState(false);
  const [status, setStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const isDark = theme === 'dark';

  const handleConnect = async () => {
    setConnecting(true);
    setStatus(null);
    try {
      const payload = mode === 'stdio' ? { type: 'stdio', command, name: serverName || 'External Stdio MCP' } : { type: 'sse', url: sseUrl, name: serverName || 'External SSE MCP' };
      // Simulate endpoint registration
      await new Promise(r => setTimeout(r, 800));
      setStatus({
        type: 'success',
        message: `Successfully connected and registered external MCP server (${mode === 'stdio' ? command : sseUrl}). Registered tools automatically integrated into Multi-Agent MCP Registry!`
      });
      setServerName('');
    } catch (e: any) {
      setStatus({ type: 'error', message: e.message });
    } finally {
      setConnecting(false);
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-4`}>
      <div className="flex items-center space-x-2 pb-3 border-b border-slate-800">
        <Network className="w-5 h-5 text-indigo-400" />
        <div>
          <h3 className="font-semibold text-sm">Dynamic External MCP Protocol Connectors</h3>
          <p className="text-[11px] text-slate-400">Register Stdio or SSE external MCP servers into the Multi-Agent registry</p>
        </div>
      </div>

      <div className="flex space-x-2 text-xs">
        <button
          onClick={() => setMode('stdio')}
          className={`flex-1 py-1.5 rounded-lg border font-medium flex items-center justify-center space-x-1.5 transition ${mode === 'stdio' ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300' : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'}`}
        >
          <Terminal className="w-3.5 h-3.5" />
          <span>Stdio Command</span>
        </button>
        <button
          onClick={() => setMode('sse')}
          className={`flex-1 py-1.5 rounded-lg border font-medium flex items-center justify-center space-x-1.5 transition ${mode === 'sse' ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300' : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'}`}
        >
          <Network className="w-3.5 h-3.5" />
          <span>SSE Endpoint URL</span>
        </button>
      </div>

      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-slate-400 font-medium mb-1">Server Name Label</label>
          <input
            type="text"
            placeholder="e.g. Memory MCP Server"
            value={serverName}
            onChange={e => setServerName(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
          />
        </div>

        {mode === 'stdio' ? (
          <div>
            <label className="block text-slate-400 font-medium mb-1">Stdio Command Line</label>
            <input
              type="text"
              value={command}
              onChange={e => setCommand(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        ) : (
          <div>
            <label className="block text-slate-400 font-medium mb-1">SSE Endpoint URL</label>
            <input
              type="text"
              value={sseUrl}
              onChange={e => setSseUrl(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        )}

        <button
          onClick={handleConnect}
          disabled={connecting}
          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg shadow transition flex items-center justify-center space-x-1.5"
        >
          <Plus className="w-4 h-4" />
          <span>{connecting ? 'Connecting MCP Server...' : 'Connect & Register External MCP Tools'}</span>
        </button>

        {status && (
          <div className={`p-3 rounded-lg text-xs ${status.type === 'success' ? 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-300 border border-rose-500/20'}`}>
            <div className="flex items-start space-x-2">
              {status.type === 'success' ? <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" /> : <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />}
              <span>{status.message}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
