/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { WFNode, WFEdge } from '../../types';
import { Settings, Trash2, Split, Bot, Code, LogIn, CheckSquare, Database, LogOut, UserCheck } from 'lucide-react';

interface WorkflowCanvasProps {
  nodes: WFNode[];
  edges: WFEdge[];
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
  onDeleteNode: (id: string) => void;
  onOpenConfig: (node: WFNode) => void;
  onConnectNodes: (sourceId: string, targetId: string) => void;
  isExecuting: boolean;
  nodeStatuses: Record<string, 'idle' | 'running' | 'success' | 'failed'>;
}

export function WorkflowCanvas({
  nodes,
  edges,
  selectedNodeId,
  onSelectNode,
  onDeleteNode,
  onOpenConfig,
  nodeStatuses,
}: WorkflowCanvasProps) {

  const getNodeIcon = (type: string) => {
    switch (type) {
      case 'input': return <LogIn className="w-4 h-4 text-blue-400" />;
      case 'llm': return <Bot className="w-4 h-4 text-purple-400" />;
      case 'decision': return <Split className="w-4 h-4 text-amber-400" />;
      case 'code_execution': return <Code className="w-4 h-4 text-emerald-400" />;
      case 'human_intervention': return <UserCheck className="w-4 h-4 text-rose-400" />;
      case 'test_runner': return <CheckSquare className="w-4 h-4 text-cyan-400" />;
      case 'rag_vector_db': return <Database className="w-4 h-4 text-indigo-400" />;
      default: return <LogOut className="w-4 h-4 text-slate-400" />;
    }
  };

  return (
    <div className="flex-1 bg-slate-950 relative overflow-auto p-8 border-r border-slate-800">
      <div className="min-w-[800px] min-h-[600px] relative">
        {/* Render Connection Edges */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none z-0">
          {edges.map(edge => {
            const src = nodes.find(n => n.id === edge.sourceId);
            const tgt = nodes.find(n => n.id === edge.targetId);
            if (!src || !tgt) return null;

            const x1 = src.position.x + 100;
            const y1 = src.position.y + 40;
            const x2 = tgt.position.x + 100;
            const y2 = tgt.position.y + 40;

            return (
              <g key={edge.id}>
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke="#6366f1"
                  strokeWidth="2"
                  strokeDasharray="4"
                  className="animate-pulse"
                />
              </g>
            );
          })}
        </svg>

        {/* Render Canvas Nodes */}
        {nodes.map(node => {
          const isSelected = node.id === selectedNodeId;
          const status = nodeStatuses[node.id] || 'idle';

          return (
            <div
              key={node.id}
              onClick={() => onSelectNode(node.id)}
              style={{ left: `${node.position.x}px`, top: `${node.position.y}px` }}
              className={`absolute w-52 p-3 rounded-2xl border-2 bg-slate-900/90 shadow-2xl z-10 cursor-pointer transition-all ${
                isSelected ? 'border-indigo-500 ring-4 ring-indigo-500/20' : 'border-slate-800 hover:border-slate-700'
              } ${status === 'running' ? 'border-amber-400 ring-4 ring-amber-400/20' : ''}`}
            >
              {/* Node Header */}
              <div className="flex items-center justify-between border-b border-slate-800/80 pb-2 mb-2">
                <div className="flex items-center gap-2">
                  {getNodeIcon(node.type)}
                  <span className="text-xs font-bold font-mono text-slate-200 truncate max-w-[100px]">
                    {node.label}
                  </span>
                </div>

                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onOpenConfig(node); }}
                    className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition cursor-pointer"
                    title="Configure Node"
                  >
                    <Settings className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); onDeleteNode(node.id); }}
                    className="p-1 text-slate-400 hover:text-rose-400 rounded hover:bg-slate-800 transition cursor-pointer"
                    title="Delete Node"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* Node Content Preview */}
              <div className="text-[10px] font-mono text-slate-400 space-y-1">
                {node.type === 'decision' && (
                  <div className="p-1.5 rounded bg-amber-950/40 border border-amber-800/50 text-amber-300">
                    Question: {(node.config as any)?.questionPrompt || 'Is condition valid?'}
                  </div>
                )}
                {node.type === 'llm' && (
                  <div className="truncate">Prompt: {(node.config as any)?.prompt || '{{input}}'}</div>
                )}
                {node.type === 'input' && (
                  <div className="truncate">Text: {(node.config as any)?.inputText || '(empty)'}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
