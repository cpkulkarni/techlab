/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { AgentWorkflow, PlanStep } from '../types';
import { 
  ClipboardList, 
  Cpu, 
  CheckCircle2, 
  AlertCircle, 
  ArrowRight,
  RefreshCw,
  Clock,
  Play,
  Wrench,
  Check,
  X,
  Terminal,
  FilePlus,
  FileText,
  AlertTriangle,
  ChevronRight,
  Square
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface AgentFlowGraphProps {
  workflow: AgentWorkflow | null;
  onApproveStep?: (stepId: string, approved: boolean) => void;
  onApprovePlan?: () => void;
  onRejectPlan?: () => void;
  onStopWorkflow?: () => void;
  onStopStep?: (stepId: string) => void;
  isExecuting?: boolean;
  theme?: 'white' | 'light-grey' | 'dark';
}

export default function AgentFlowGraph({ workflow, onApproveStep, onApprovePlan, onRejectPlan, onStopWorkflow, onStopStep, isExecuting, theme = 'dark' }: AgentFlowGraphProps) {
  const isDark = theme === 'dark';
  const isGray = theme === 'light-grey';

  if (!workflow) {
    const emptyBg = isDark ? 'bg-slate-900 border-slate-800' : isGray ? 'bg-zinc-100 border-zinc-200' : 'bg-slate-50 border-slate-200';
    const emptyText = isDark ? 'text-slate-400' : 'text-slate-500';
    const emptyHeading = isDark ? 'text-white' : 'text-slate-800';
    const emptyMuted = isDark ? 'text-slate-500' : 'text-slate-500';

    return (
      <div className={`${emptyBg} rounded-lg border p-6 text-center ${emptyText} space-y-2 h-full flex flex-col justify-center items-center`} id="empty-flow-graph">
        <Cpu className={`w-8 h-8 ${isDark ? 'text-slate-600' : 'text-slate-300'} animate-pulse`} />
        <p className={`font-display font-medium text-xs ${emptyHeading}`}>No active coding task</p>
        <p className={`text-[11px] ${emptyMuted} max-w-xs leading-relaxed`}>
          Select a mode and enter an instruction in the chat workspace to launch the AI Coding Agent cycle.
        </p>
      </div>
    );
  }

  const status = workflow.status;
  const planSteps = workflow.plan || [];
  const currentStep = planSteps[workflow.currentStepIndex];

  // Helper to get step type icon
  const getStepIcon = (type: string, statusStr: string) => {
    if (statusStr === 'completed') return CheckCircle2;
    if (statusStr === 'failed') return AlertCircle;
    
    switch (type) {
      case 'create':
        return FilePlus;
      case 'edit':
        return FileText;
      case 'delete':
        return X;
      case 'command':
        return Terminal;
      case 'test':
        return RefreshCw;
      default:
        return Clock;
    }
  };

  const containerBg = isDark ? 'bg-slate-900 border-slate-800' : isGray ? 'bg-zinc-50 border-zinc-200 shadow-sm' : 'bg-white border-slate-200 shadow-md';
  const headingText = isDark ? 'text-white' : 'text-slate-900';
  const normalText = isDark ? 'text-slate-300' : 'text-slate-700';
  const mutedText = isDark ? 'text-slate-500' : 'text-slate-500';
  const subContainerBg = isDark ? 'bg-slate-950 border-slate-800/80' : isGray ? 'bg-zinc-100 border-zinc-200' : 'bg-slate-50 border-slate-200';

  return (
    <div className={`${containerBg} rounded-lg border p-4 space-y-4`} id="agent-flow-graph">
      
      {/* Header section */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className={`font-display font-semibold ${headingText} text-xs`}>Agent Execution Flow</h3>
          <p className="text-[10px] text-slate-500 font-mono">Task ID: {workflow.taskId.slice(0, 8)}...</p>
        </div>
        <div className="flex items-center gap-2">
          {(status === 'planning' || status === 'executing' || status === 'correcting' || status === 'waiting_plan_approval' || status === 'waiting_approval' || isExecuting) && onStopWorkflow && (
            <button
              type="button"
              onClick={onStopWorkflow}
              className="flex items-center gap-1 px-2.5 py-1 bg-rose-600 hover:bg-rose-500 text-white font-mono font-bold text-[10px] rounded shadow-sm transition-all cursor-pointer animate-pulse shrink-0"
              title="Stop entire workflow execution"
            >
              <Square className="w-3 h-3 fill-current" />
              Stop Execution
            </button>
          )}
          <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono font-medium border ${
            status === 'completed' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' :
            status === 'failed' ? 'bg-rose-500/10 text-rose-400 border-rose-500/20' :
            status === 'waiting_approval' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20 animate-pulse' :
            'bg-indigo-500/10 text-indigo-400 border-indigo-500/20 animate-pulse'
          }`}>
            {status.toUpperCase().replace('_', ' ')}
          </span>
        </div>
      </div>

      {/* Graphical Flow Pipeline: Dynamic from the generated plan steps */}
      <div className={`p-2 ${subContainerBg} rounded-lg border`}>
        {planSteps.length > 0 ? (
          <div className="flex flex-row items-center gap-1.5 overflow-x-auto py-1 scrollbar-thin scrollbar-thumb-slate-800">
            {planSteps.map((step, idx) => {
              const StepIcon = getStepIcon(step.type, step.status);
              const isActive = idx === workflow.currentStepIndex;
              const isCompleted = step.status === 'completed';
              const isWaiting = status === 'waiting_approval' && isActive;

              return (
                <React.Fragment key={step.id}>
                  {idx > 0 && (
                    <ChevronRight className={`w-3 h-3 shrink-0 ${
                      isCompleted ? 'text-emerald-500' : 'text-slate-800'
                    }`} />
                  )}
                  <div 
                    className={`flex items-center gap-1.5 p-1.5 rounded border transition-all shrink-0 max-w-[150px] ${
                      isWaiting
                        ? 'bg-amber-500/10 border-amber-500/40 text-amber-500 dark:text-amber-300 animate-pulse ring-1 ring-amber-500/25'
                        : isActive 
                          ? 'bg-indigo-500/10 border-indigo-500/40 text-indigo-700 dark:text-white shadow-sm scale-[1.01] shadow-[0_0_8px_rgba(99,102,241,0.2)]' 
                          : isCompleted 
                            ? 'bg-emerald-500/5 border-emerald-500/10 text-emerald-600 dark:text-emerald-400'
                            : isDark
                              ? 'bg-slate-900 border-slate-800/60 text-slate-500 opacity-60'
                              : 'bg-zinc-200 border-zinc-300/60 text-zinc-500 opacity-80'
                    }`}
                    title={`${step.title} (${step.status})`}
                  >
                    <div className={`p-1 rounded shrink-0 ${
                      isWaiting
                        ? 'bg-amber-500/20 text-amber-500 dark:text-amber-400'
                        : isActive 
                          ? 'bg-indigo-500/25 text-indigo-600 dark:text-indigo-300' 
                          : isCompleted 
                            ? 'bg-emerald-500/10 text-emerald-600 dark:text-emerald-400' 
                            : isDark
                              ? 'bg-slate-950 text-slate-600'
                              : 'bg-zinc-100 text-zinc-500'
                    }`}>
                      <StepIcon className={`w-3 h-3 ${isActive && step.status === 'running' ? 'animate-spin' : ''}`} />
                    </div>
                    <div className="min-w-0">
                      <p className={`text-[9px] font-bold truncate leading-none mb-0.5 ${isDark ? 'text-slate-200' : 'text-slate-800'}`}>{step.title}</p>
                      <p className={`text-[8px] opacity-75 truncate leading-none ${isDark ? 'text-slate-450' : 'text-slate-500'}`}>{step.type.toUpperCase()}</p>
                    </div>
                  </div>
                </React.Fragment>
              );
            })}
          </div>
        ) : (
          /* Fallback static stage pipeline during initial warm up */
          <div className="flex flex-row items-center justify-between gap-1">
            {['Planning', 'Executing', 'Verifying', 'Finalizing'].map((stage, idx) => (
              <div key={stage} className={`flex-1 flex items-center gap-2 p-1.5 rounded border text-slate-500 opacity-65 ${isDark ? 'border-slate-800/60 bg-slate-900' : 'border-zinc-200 bg-zinc-100'}`}>
                <Clock className="w-3.5 h-3.5 text-slate-400 dark:text-slate-600" />
                <span className="text-[9px] font-bold">{stage}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Plan-Level Overall Approval Gate */}
      <AnimatePresence>
        {status === 'waiting_plan_approval' && onApprovePlan && onRejectPlan && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`border rounded p-3.5 space-y-3 overflow-hidden shadow-lg ${
              isDark 
                ? 'bg-indigo-950/25 border-indigo-900/40' 
                : 'bg-indigo-50/60 border-indigo-150'
            }`}
          >
            <div className="flex gap-2.5">
              <div className="p-1.5 bg-indigo-500/10 text-indigo-550 dark:text-indigo-400 rounded shrink-0 h-7 w-7 flex items-center justify-center">
                <ClipboardList className="w-4 h-4 animate-pulse" />
              </div>
              <div className="flex-1 min-w-0">
                <h4 className="text-xs font-bold text-indigo-600 dark:text-indigo-400 font-display">Plan Review Required</h4>
                <p className={`text-[10px] mt-0.5 leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-650'}`}>
                  The AI Agent has formulated a development plan with <strong>{planSteps.length} steps</strong>. Please review the checklist below. You must authorize the plan before any execution can begin.
                </p>
              </div>
            </div>

            <div className="flex gap-1.5 justify-end">
              <button
                type="button"
                onClick={onRejectPlan}
                className={`px-2.5 py-1.5 border hover:bg-rose-50 dark:hover:bg-rose-950/40 hover:text-rose-650 hover:border-rose-300 text-[10px] font-mono font-medium rounded transition-colors cursor-pointer flex items-center gap-1 ${
                  isDark ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-white border-zinc-200 text-slate-600'
                }`}
              >
                <X className="w-3 h-3" />
                Reject Plan
              </button>
              <button
                type="button"
                onClick={onApprovePlan}
                className="px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-[10px] font-mono font-medium rounded shadow-sm transition-colors cursor-pointer flex items-center gap-1 animate-pulse"
              >
                <Check className="w-3 h-3" />
                Approve & Run Plan
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Prior Approval Panel embedded directly within the workflow for full visibility */}
      <AnimatePresence>
        {status === 'waiting_approval' && currentStep && onApproveStep && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`border rounded p-3 space-y-2.5 overflow-hidden shadow-lg ${
              isDark 
                ? 'bg-amber-950/25 border-amber-900/40' 
                : 'bg-amber-50/60 border-amber-200'
            }`}
          >
            <div className="flex gap-2.5">
              <div className="p-1.5 bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded shrink-0 h-7 w-7 flex items-center justify-center">
                <Wrench className="w-4 h-4 animate-pulse" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className="text-[9px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-wider font-mono">Requires Prior Action Approval</span>
                  <span className={`text-[8px] font-mono px-1 py-0.2 rounded border uppercase ${
                    isDark ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-amber-100 border-amber-200 text-amber-800'
                  }`}>
                    {currentStep.type}
                  </span>
                </div>
                <p className={`text-[11px] font-bold mt-1 font-mono ${isDark ? 'text-slate-100' : 'text-slate-800'}`}>{currentStep.title}</p>
                <p className={`text-[10px] mt-0.5 leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-650'}`}>{currentStep.description}</p>
                
                {currentStep.target && (
                  <div className={`mt-2 p-1.5 rounded border font-mono text-[9px] truncate ${
                    isDark ? 'bg-slate-950 border-slate-850 text-slate-300' : 'bg-amber-100/40 border-amber-150 text-slate-705'
                  }`}>
                    <span className="text-slate-505 font-semibold">Target:</span> {currentStep.target}
                  </div>
                )}
              </div>
            </div>

            <div className="flex gap-1.5 justify-end">
              <button
                type="button"
                onClick={() => onApproveStep(currentStep.id, false)}
                className={`px-2.5 py-1 border hover:bg-rose-50 dark:hover:bg-rose-950/40 hover:text-rose-650 hover:border-rose-300 text-[10px] font-mono font-medium rounded transition-colors cursor-pointer flex items-center gap-1 ${
                  isDark ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-white border-zinc-200 text-slate-600'
                }`}
              >
                <X className="w-3 h-3" />
                Deny
              </button>
              <button
                type="button"
                onClick={() => onApproveStep(currentStep.id, true)}
                className="px-2.5 py-1 bg-amber-600 hover:bg-amber-500 text-white text-[10px] font-mono font-medium rounded shadow-sm transition-colors cursor-pointer flex items-center gap-1 animate-pulse"
              >
                <Check className="w-3 h-3" />
                Approve & Execute
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Interactive Step-by-Step Task Checklist */}
      <div className="space-y-2">
        <p className={`text-[10px] font-bold uppercase tracking-widest ${isDark ? 'text-slate-550' : 'text-slate-400'}`}>Plan & Steps Checklist</p>
        <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 font-sans">
          {planSteps.map((step, idx) => {
            const isActive = idx === workflow.currentStepIndex;
            return (
              <motion.div
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                key={step.id}
                className={`flex items-start gap-2.5 p-2.5 rounded text-xs transition-colors border ${
                  isActive 
                    ? 'bg-indigo-500/10 border-indigo-500/30 ring-1 ring-indigo-500/10' 
                    : step.status === 'completed' 
                      ? 'bg-emerald-500/5 border-emerald-500/10'
                      : step.status === 'failed'
                        ? 'bg-rose-500/5 border-rose-500/10'
                        : isDark
                          ? 'bg-slate-900/50 border-slate-800/60'
                          : 'bg-zinc-100 border-zinc-200/80'
                }`}
              >
                {/* Status indicator icon */}
                <div className="shrink-0 mt-0.5">
                  {step.status === 'completed' ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  ) : step.status === 'failed' ? (
                    <AlertCircle className="w-3.5 h-3.5 text-rose-500 animate-pulse" />
                  ) : step.status === 'running' ? (
                    <RefreshCw className="w-3.5 h-3.5 text-indigo-500 dark:text-indigo-400 animate-spin" />
                  ) : step.status === 'approved' ? (
                    <Play className="w-3.5 h-3.5 text-amber-500 animate-pulse" />
                  ) : (
                    <Clock className="w-3.5 h-3.5 text-slate-450" />
                  )}
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <p className={`font-semibold font-mono text-[11px] ${
                      step.status === 'completed' 
                        ? 'text-slate-400 dark:text-slate-550 line-through' 
                        : isDark 
                          ? 'text-slate-200' 
                          : 'text-slate-800'
                    }`}>
                      {step.title}
                    </p>
                    <div className="flex items-center gap-1.5 shrink-0">
                      {onStopStep && step.status !== 'completed' && step.status !== 'failed' && (
                        <button
                          type="button"
                          onClick={(e) => { e.stopPropagation(); onStopStep(step.id); }}
                          className="px-1.5 py-0.5 bg-rose-600/20 hover:bg-rose-600 text-rose-400 hover:text-white border border-rose-500/30 rounded text-[9px] font-mono font-bold transition-all cursor-pointer flex items-center gap-1 shrink-0"
                          title={`Stop/Cancel step component "${step.title}"`}
                        >
                          <Square className="w-2.5 h-2.5 fill-current" />
                          Stop
                        </button>
                      )}
                      <span className={`text-[9px] font-mono px-1 py-0.2 rounded border ${
                        isDark 
                          ? 'bg-slate-950 border-slate-800 text-slate-400' 
                          : 'bg-slate-100 border-zinc-250 text-slate-600'
                      }`}>
                        {step.type.toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <p className={`text-[11px] mt-0.5 leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-650'}`}>{step.description}</p>
                  
                  {step.target && (
                    <p className={`text-[10px] font-mono mt-1 truncate ${isDark ? 'text-slate-500' : 'text-slate-450'}`}>
                      Target: <span className={isDark ? 'text-slate-300' : 'text-slate-700'}>{step.target}</span>
                    </p>
                  )}

                  {/* Logs if active or has run */}
                  {step.logs && (
                    <div className="mt-2 p-2 bg-black text-emerald-400 font-mono text-[9px] rounded border border-slate-800 overflow-x-auto max-h-24 whitespace-pre">
                      {step.logs}
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
