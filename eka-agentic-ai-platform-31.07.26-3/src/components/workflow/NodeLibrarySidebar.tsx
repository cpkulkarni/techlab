/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { WFNodeType } from '../../types';
import { 
  LogIn, Code, CheckSquare, Database, Bot, LogOut, RotateCcw, 
  UserCheck, Split, Search, Clock
} from 'lucide-react';

interface NodeLibrarySidebarProps {
  onAddNode: (type: WFNodeType) => void;
}

export const NODE_TYPES_INFO: { type: WFNodeType; label: string; desc: string; icon: React.ReactNode; color: string }[] = [
  { type: 'input', label: 'Input Prompt', desc: 'Starting prompt or context source', icon: <LogIn className="w-4 h-4" />, color: 'border-blue-500/40 bg-blue-950/30 text-blue-300' },
  { type: 'llm', label: 'AI Model Generator', desc: 'Execute AI inference with selected LLM', icon: <Bot className="w-4 h-4" />, color: 'border-purple-500/40 bg-purple-950/30 text-purple-300' },
  { type: 'scheduler', label: 'Temporal / Trigger Scheduler', desc: 'Schedule cron jobs, execute code or open documents', icon: <Clock className="w-4 h-4" />, color: 'border-violet-500/40 bg-violet-950/30 text-violet-300' },
  { type: 'decision', label: 'Decision Node (Branching)', desc: 'Yes/No conditional evaluation & branching', icon: <Split className="w-4 h-4" />, color: 'border-amber-500/40 bg-amber-950/30 text-amber-300' },
  { type: 'code_execution', label: 'Code Execution', desc: 'Run script or project file', icon: <Code className="w-4 h-4" />, color: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-300' },
  { type: 'human_intervention', label: 'Human Gate Approval', desc: 'Pause execution for human confirmation', icon: <UserCheck className="w-4 h-4" />, color: 'border-rose-500/40 bg-rose-950/30 text-rose-300' },
  { type: 'test_runner', label: 'Test Suite Runner', desc: 'Execute unit and integration tests', icon: <CheckSquare className="w-4 h-4" />, color: 'border-cyan-500/40 bg-cyan-950/30 text-cyan-300' },
  { type: 'rag_vector_db', label: 'RAG Vector Search', desc: 'Query vector database embeddings', icon: <Database className="w-4 h-4" />, color: 'border-indigo-500/40 bg-indigo-950/30 text-indigo-300' },
  { type: 'rag_local_files', label: 'Local Files Search', desc: 'Extract context from workspace files', icon: <Search className="w-4 h-4" />, color: 'border-teal-500/40 bg-teal-950/30 text-teal-300' },
  { type: 'loop', label: 'Iterative Loop', desc: 'Iterate execution over items or count', icon: <RotateCcw className="w-4 h-4" />, color: 'border-sky-500/40 bg-sky-950/30 text-sky-300' },
  { type: 'output', label: 'File Output Writer', desc: 'Write output to file or storage', icon: <LogOut className="w-4 h-4" />, color: 'border-slate-500/40 bg-slate-900/60 text-slate-300' },
];

export function NodeLibrarySidebar({ onAddNode }: NodeLibrarySidebarProps) {
  return (
    <aside className="w-64 border-r border-slate-800 bg-slate-900/80 p-3 flex flex-col gap-3 overflow-y-auto shrink-0">
      <div className="text-xs font-mono font-bold text-slate-300 border-b border-slate-800 pb-2">
        🧩 Pipeline Node Palette
      </div>

      <div className="space-y-2">
        {NODE_TYPES_INFO.map(node => (
          <button
            key={node.type}
            type="button"
            onClick={() => onAddNode(node.type)}
            className={`w-full text-left p-2.5 rounded-xl border transition flex items-center gap-2.5 cursor-pointer hover:scale-[1.02] ${node.color}`}
          >
            <div className="shrink-0">{node.icon}</div>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-bold font-mono truncate">{node.label}</div>
              <div className="text-[10px] text-slate-400 truncate">{node.desc}</div>
            </div>
          </button>
        ))}
      </div>
    </aside>
  );
}
