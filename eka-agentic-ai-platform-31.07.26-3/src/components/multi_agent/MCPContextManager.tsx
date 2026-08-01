/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  Server,
  Wrench,
  Database,
  FileText,
  Play,
  CheckCircle2,
  XCircle,
  ToggleLeft,
  ToggleRight,
  Code2,
  Search,
  Terminal,
  Layers,
  ChevronRight,
  Sparkles
} from 'lucide-react';
import { MCPTool, MCPResource, MCPServerInfo } from '../../multi_agent/types';

interface MCPContextManagerProps {
  tools: MCPTool[];
  resources: MCPResource[];
  servers: MCPServerInfo[];
  theme: 'white' | 'light-grey' | 'dark';
  onToggleTool?: (name: string, enabled: boolean) => void;
  onInvokeTool?: (name: string, args: Record<string, any>) => Promise<any>;
}

export default function MCPContextManager({
  tools,
  resources,
  servers,
  theme,
  onToggleTool,
  onInvokeTool,
}: MCPContextManagerProps) {
  const [activeSubTab, setActiveSubTab] = useState<'tools' | 'resources' | 'servers'>('tools');
  const [selectedToolName, setSelectedToolName] = useState<string | null>('web_search');
  const [toolArgsText, setToolArgsText] = useState<string>('{\n  "query": "React 19 features"\n}');
  const [executionResult, setExecutionResult] = useState<any>(null);
  const [isExecutingTool, setIsExecutingTool] = useState(false);

  const isDark = theme === 'dark';
  const selectedTool = tools.find(t => t.name === selectedToolName) || tools[0];

  const handleRunTool = async () => {
    if (!selectedTool) return;
    setIsExecutingTool(true);
    setExecutionResult(null);
    try {
      const parsedArgs = JSON.parse(toolArgsText || '{}');
      const res = await onInvokeTool?.(selectedTool.name, parsedArgs);
      setExecutionResult(res);
    } catch (err: any) {
      setExecutionResult({ error: err.message });
    } finally {
      setIsExecutingTool(false);
    }
  };

  const getToolCategoryIcon = (category: string) => {
    switch (category) {
      case 'file': return FileText;
      case 'search': return Search;
      case 'command': return Terminal;
      case 'test': return Wrench;
      default: return Layers;
    }
  };

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Top Banner & Sub-Tabs */}
      <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <h3 className={`text-sm font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Model Context Protocol (MCP) Explorer
            </h3>
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Standardized tools, shared resources, and server connectivity for multi-agent network
            </p>
          </div>
        </div>

        {/* Sub Navigation */}
        <div className={`flex items-center gap-1 p-1 rounded-lg border ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
          <button
            type="button"
            onClick={() => setActiveSubTab('tools')}
            className={`px-3 py-1 rounded text-xs font-mono font-bold transition-all cursor-pointer ${
              activeSubTab === 'tools'
                ? 'bg-emerald-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            MCP Tools ({tools.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveSubTab('resources')}
            className={`px-3 py-1 rounded text-xs font-mono font-bold transition-all cursor-pointer ${
              activeSubTab === 'resources'
                ? 'bg-emerald-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Resources ({resources.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveSubTab('servers')}
            className={`px-3 py-1 rounded text-xs font-mono font-bold transition-all cursor-pointer ${
              activeSubTab === 'servers'
                ? 'bg-emerald-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            Servers ({servers.length})
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {activeSubTab === 'tools' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-[420px]">
          {/* Tools List Column */}
          <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
            <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
              <span className={`text-xs font-mono font-bold uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                Registered Tools
              </span>
              <span className="text-[10px] font-mono text-emerald-400">MCP Schema Ready</span>
            </div>

            <div className="space-y-2 overflow-y-auto max-h-[460px] pr-1 flex-1">
              {tools.map(tool => {
                const Icon = getToolCategoryIcon(tool.category);
                const isSelected = selectedToolName === tool.name;
                return (
                  <div
                    key={tool.name}
                    onClick={() => {
                      setSelectedToolName(tool.name);
                      setToolArgsText('{\n  "query": "example"\n}');
                      setExecutionResult(null);
                    }}
                    className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                      isSelected
                        ? 'bg-emerald-500/10 border-emerald-500 text-white shadow'
                        : isDark
                        ? 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                        : 'bg-white border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center gap-2.5">
                      <div className="p-2 rounded bg-emerald-500/10 text-emerald-400">
                        <Icon className="w-4 h-4" />
                      </div>
                      <div>
                        <h4 className={`text-xs font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                          {tool.name}
                        </h4>
                        <span className="text-[9px] font-mono uppercase text-slate-500">Category: {tool.category}</span>
                      </div>
                    </div>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        onToggleTool?.(tool.name, !tool.enabled);
                      }}
                      className="text-slate-400 hover:text-emerald-400 transition-colors"
                      title={tool.enabled ? 'Disable MCP tool' : 'Enable MCP tool'}
                    >
                      {tool.enabled ? (
                        <ToggleRight className="w-6 h-6 text-emerald-400" />
                      ) : (
                        <ToggleLeft className="w-6 h-6 text-slate-600" />
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Interactive Tool Details & Execution Sandbox */}
          <div className={`lg:col-span-2 p-5 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
            {selectedTool ? (
              <div className="space-y-4 flex-1 flex flex-col">
                <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                  <div>
                    <h4 className={`text-sm font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                      MCP Tool: {selectedTool.name}
                    </h4>
                    <p className="text-xs text-slate-400 mt-0.5">{selectedTool.description}</p>
                  </div>
                  <span className={`px-2 py-0.5 text-[10px] font-mono font-bold rounded ${selectedTool.enabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                    {selectedTool.enabled ? 'ENABLED' : 'DISABLED'}
                  </span>
                </div>

                {/* Test Invocation Sandbox */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-1">
                  <div>
                    <label className="text-[10px] font-mono uppercase text-slate-500 block mb-1">
                      Invocation Arguments (JSON)
                    </label>
                    <textarea
                      value={toolArgsText}
                      onChange={e => setToolArgsText(e.target.value)}
                      rows={8}
                      className={`w-full p-3 rounded-lg border text-xs font-mono ${isDark ? 'bg-slate-950 border-slate-800 text-emerald-300' : 'bg-slate-900 border-slate-800 text-emerald-300'}`}
                    />
                    <button
                      type="button"
                      onClick={handleRunTool}
                      disabled={isExecutingTool || !selectedTool.enabled}
                      className="mt-2 w-full py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-mono font-bold text-xs rounded-lg transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
                    >
                      <Play className="w-3.5 h-3.5 fill-current" />
                      {isExecutingTool ? 'Invoking MCP Tool...' : 'Test Invoke Tool'}
                    </button>
                  </div>

                  <div>
                    <label className="text-[10px] font-mono uppercase text-slate-500 block mb-1">
                      Tool Execution Result
                    </label>
                    <pre className={`p-3 rounded-lg border text-xs font-mono overflow-auto max-h-[220px] ${isDark ? 'bg-slate-950 border-slate-800 text-slate-300' : 'bg-slate-900 border-slate-800 text-slate-300'}`}>
                      {executionResult ? JSON.stringify(executionResult, null, 2) : '// Result will appear here after execution'}
                    </pre>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-xs text-slate-500 font-mono">Select a tool to inspect and test.</p>
            )}
          </div>
        </div>
      )}

      {/* Resources Sub-Tab */}
      {activeSubTab === 'resources' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {resources.map(res => (
            <div key={res.uri} className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
              <div className="flex items-center gap-3 mb-2">
                <Database className="w-4 h-4 text-emerald-400" />
                <h4 className={`text-xs font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                  {res.name}
                </h4>
              </div>
              <p className="text-xs text-slate-400 mb-3">{res.description}</p>
              <div className="flex items-center justify-between text-[10px] font-mono text-slate-500">
                <span>URI: {res.uri}</span>
                <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">{res.mimeType}</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Servers Sub-Tab */}
      {activeSubTab === 'servers' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {servers.map(s => (
            <div key={s.id} className={`p-4 rounded-xl border flex items-center justify-between ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
              <div className="flex items-center gap-3">
                <Server className="w-5 h-5 text-emerald-400" />
                <div>
                  <h4 className={`text-xs font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    {s.name} (v{s.version})
                  </h4>
                  <p className="text-[10px] font-mono text-slate-500">Endpoint: {s.endpoint}</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3" /> CONNECTED
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
