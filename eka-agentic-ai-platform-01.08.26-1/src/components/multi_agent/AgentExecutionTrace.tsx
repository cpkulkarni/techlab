/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  Activity,
  CheckCircle2,
  XCircle,
  Clock,
  ChevronDown,
  ChevronRight,
  Code2,
  Wrench,
  Radio,
  Cpu,
  Globe,
  CheckSquare,
  Bot,
  Sparkles
} from 'lucide-react';
import { MultiAgentTaskExecution, MultiAgentTaskTraceStep } from '../../multi_agent/types';

interface AgentExecutionTraceProps {
  executions: MultiAgentTaskExecution[];
  theme: 'white' | 'light-grey' | 'dark';
}

export default function AgentExecutionTrace({
  executions,
  theme,
}: AgentExecutionTraceProps) {
  const [selectedExecutionId, setSelectedExecutionId] = useState<string | null>(null);
  const [expandedStepId, setExpandedStepId] = useState<string | null>(null);

  const isDark = theme === 'dark';
  const selectedExecution = executions.find(e => e.id === selectedExecutionId) || executions[executions.length - 1] || null;

  const getAgentRoleIcon = (role: string) => {
    switch (role) {
      case 'coordinator': return Cpu;
      case 'researcher': return Globe;
      case 'coder': return Code2;
      case 'tester': return CheckSquare;
      default: return Bot;
    }
  };

  const getActionBadge = (action: MultiAgentTaskTraceStep['action']) => {
    switch (action) {
      case 'delegating':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">DELEGATING</span>;
      case 'mcp_tool_execution':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">MCP TOOL EXECUTION</span>;
      case 'completed':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">COMPLETED</span>;
      case 'failed':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">FAILED</span>;
      default:
        return <span className="px-2 py-0.5 text-[9px] font-mono font-medium rounded bg-slate-500/20 text-slate-400 border border-slate-500/30">RECEIVED</span>;
    }
  };

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Top Banner */}
      <div className={`p-4 rounded-xl border flex items-center justify-between ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className={`text-sm font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Multi-Agent Execution Timeline & Trace
            </h3>
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Step-by-step breakdown of user requests, coordinator delegations, MCP tool calls, and final aggregations
            </p>
          </div>
        </div>

        <span className="text-xs font-mono font-bold text-slate-400">
          Total Recorded Tasks: {executions.length}
        </span>
      </div>

      {/* Grid: Executions List + Step Timeline */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-[420px]">
        {/* Execution Tasks List */}
        <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
            <span className={`text-xs font-mono font-bold uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Task Runs ({executions.length})
            </span>
          </div>

          <div className="space-y-2 overflow-y-auto max-h-[460px] pr-1 flex-1">
            {executions.length === 0 ? (
              <div className="py-12 text-center text-slate-500 font-mono text-xs">
                No multi-agent executions launched yet.
              </div>
            ) : (
              executions.slice().reverse().map(exec => {
                const isSelected = selectedExecution?.id === exec.id;
                return (
                  <div
                    key={exec.id}
                    onClick={() => setSelectedExecutionId(exec.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-indigo-500/10 border-indigo-500 text-white shadow'
                        : isDark
                        ? 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                        : 'bg-white border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono gap-2 mb-1">
                      <span className="font-bold text-indigo-400">{exec.id}</span>
                      <span className={`px-1.5 py-0.5 rounded font-bold uppercase ${
                        exec.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' : exec.status === 'running' ? 'bg-amber-500/20 text-amber-400 animate-pulse' : 'bg-rose-500/20 text-rose-400'
                      }`}>
                        {exec.status}
                      </span>
                    </div>

                    <p className={`text-xs font-mono line-clamp-2 ${isDark ? 'text-slate-300' : 'text-slate-800'}`}>
                      {exec.taskPrompt}
                    </p>

                    <div className="mt-2 flex items-center justify-between text-[9px] font-mono text-slate-500">
                      <span>Steps: {exec.steps.length}</span>
                      <span>{new Date(exec.startTime).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Selected Task Step Timeline */}
        <div className={`lg:col-span-2 p-5 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
          {selectedExecution ? (
            <div className="space-y-4 flex-1 flex flex-col">
              {/* Task Header */}
              <div className="pb-3 border-b border-slate-800">
                <div className="flex items-center justify-between mb-1">
                  <h4 className={`text-sm font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    Task Execution Trace
                  </h4>
                  <span className="text-xs font-mono text-slate-400">ID: {selectedExecution.id}</span>
                </div>
                <p className={`text-xs ${isDark ? 'text-indigo-300' : 'text-indigo-900'} font-mono`}>
                  "{selectedExecution.taskPrompt}"
                </p>
              </div>

              {/* Timeline Steps */}
              <div className="space-y-3 overflow-y-auto max-h-[440px] pr-1 flex-1">
                {selectedExecution.steps.map((step) => {
                  const Icon = getAgentRoleIcon(step.agentRole);
                  const isExpanded = expandedStepId === step.id;

                  return (
                    <div
                      key={step.id}
                      className={`p-3.5 rounded-xl border transition-all ${
                        isDark ? 'bg-slate-950/90 border-slate-800' : 'bg-slate-50 border-slate-200'
                      }`}
                    >
                      <div
                        onClick={() => setExpandedStepId(isExpanded ? null : step.id)}
                        className="flex items-start justify-between gap-3 cursor-pointer"
                      >
                        <div className="flex items-start gap-3">
                          <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mt-0.5">
                            <Icon className="w-4 h-4" />
                          </div>
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-[10px] font-mono font-bold text-slate-500">#{step.stepNumber}</span>
                              <h5 className={`text-xs font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                                {step.title}
                              </h5>
                              <span className="text-[10px] font-mono text-slate-400">({step.agentName})</span>
                            </div>
                            <p className="text-xs text-slate-400 mt-1 line-clamp-2">{step.description}</p>
                          </div>
                        </div>

                        <div className="flex items-center gap-2">
                          {getActionBadge(step.action)}
                          {step.durationMs !== undefined && (
                            <span className="text-[10px] font-mono text-slate-400 flex items-center gap-1">
                              <Clock className="w-3 h-3 text-indigo-400" /> {step.durationMs}ms
                            </span>
                          )}
                          {isExpanded ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronRight className="w-4 h-4 text-slate-400" />}
                        </div>
                      </div>

                      {/* Expandable Details */}
                      {isExpanded && (
                        <div className="mt-3 pt-3 border-t border-slate-800 font-mono text-xs space-y-2">
                          {step.mcpToolCalled && (
                            <div className="p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-emerald-300">
                              <span className="font-bold">MCP Tool Called:</span> {step.mcpToolCalled}
                            </div>
                          )}
                          {step.output && (
                            <div>
                              <span className="text-[10px] text-slate-500 uppercase block mb-1">Step Output / Payload</span>
                              <pre className={`p-2.5 rounded border text-[11px] overflow-x-auto ${isDark ? 'bg-slate-900 border-slate-800 text-slate-300' : 'bg-slate-100 border-slate-200 text-slate-800'}`}>
                                {step.output}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center flex-1 text-slate-500 text-xs font-mono">
              Select a task execution to view step trace timeline.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
