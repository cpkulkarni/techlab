/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig, WFNode, WFEdge, WFNodeType } from '../types';
import { Play, Square, Plus, Save, Workflow, Clock } from 'lucide-react';
import { NodeLibrarySidebar } from './workflow/NodeLibrarySidebar';
import { WorkflowCanvas } from './workflow/WorkflowCanvas';
import { NodeConfigModal } from './workflow/NodeConfigModal';
import { SchedulerStatusPanel } from './workflow/SchedulerStatusPanel';

interface WorkflowBuilderProps {
  modelConfig?: ModelServerConfig;
  files?: any[];
  onRefreshWorkspace?: () => void;
}

export default function WorkflowBuilder({
  modelConfig,
}: WorkflowBuilderProps) {
  const [nodes, setNodes] = useState<WFNode[]>([
    { id: 'node-1', type: 'input', label: 'Start Prompt', position: { x: 50, y: 100 }, config: { inputText: 'Build a secure REST API' } },
    { id: 'node-2', type: 'llm', label: 'AI Code Spec', position: { x: 300, y: 100 }, config: { prompt: 'Generate architecture for {{input}}' } },
    { id: 'node-3', type: 'decision', label: 'Decision Node', position: { x: 550, y: 100 }, config: { questionPrompt: 'Does the architecture pass security review?', evalType: 'llm_boolean' } },
  ]);

  const [edges, setEdges] = useState<WFEdge[]>([
    { id: 'edge-1', sourceId: 'node-1', targetId: 'node-2' },
    { id: 'edge-2', sourceId: 'node-2', targetId: 'node-3' },
  ]);

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>('node-3');
  const [configuringNode, setConfiguringNode] = useState<WFNode | null>(null);
  const [showSchedulerModal, setShowSchedulerModal] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [nodeStatuses, setNodeStatuses] = useState<Record<string, 'idle' | 'running' | 'success' | 'failed'>>({});
  const [logs, setLogs] = useState<string[]>([]);

  const handleAddNode = (type: WFNodeType) => {
    const newNode: WFNode = {
      id: `node-${Date.now()}`,
      type,
      label: `${type.toUpperCase()} Node`,
      position: { x: 100 + nodes.length * 30, y: 150 + (nodes.length % 3) * 50 },
      config: type === 'decision' ? { questionPrompt: 'Is input output valid?', evalType: 'llm_boolean' } : {},
    };
    setNodes([...nodes, newNode]);

    // Auto-connect to previous node
    if (nodes.length > 0) {
      const prevNode = nodes[nodes.length - 1];
      setEdges(prev => [...prev, { id: `edge-${Date.now()}`, sourceId: prevNode.id, targetId: newNode.id }]);
    }
  };

  const handleDeleteNode = (id: string) => {
    setNodes(prev => prev.filter(n => n.id !== id));
    setEdges(prev => prev.filter(e => e.sourceId !== id && e.targetId !== id));
    if (selectedNodeId === id) setSelectedNodeId(null);
  };

  const handleRunWorkflow = async () => {
    setIsExecuting(true);
    setLogs(['▶ Executing Multi-Agent Pipeline...']);

    try {
      const res = await fetch('/api/workflow/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          nodes,
          edges,
          modelConfig,
          workflowName: 'Eka_Pipeline',
        }),
      });

      if (!res.body) return;
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.replace('data: ', ''));
              if (data.line) setLogs(prev => [...prev, data.line]);
            } catch (e) {}
          }
        }
      }
    } catch (err: any) {
      setLogs(prev => [...prev, `❌ Workflow Execution Error: ${err.message}`]);
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="h-full bg-slate-950 flex flex-col overflow-hidden">
      {/* Top Controls Bar */}
      <div className="h-12 px-4 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Workflow className="w-4 h-4 text-purple-400" />
          <span className="text-xs font-bold font-mono text-slate-200">
            Multi-Agent Workflow & Decision Canvas
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowSchedulerModal(true)}
            className="bg-slate-800 hover:bg-slate-700 text-violet-300 border border-violet-500/30 font-mono text-xs font-semibold px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 cursor-pointer shadow"
          >
            <Clock className="w-3.5 h-3.5 text-violet-400" /> Temporal / Trigger Scheduler
          </button>

          {isExecuting ? (
            <button
              type="button"
              onClick={() => setIsExecuting(false)}
              className="bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs font-bold px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 cursor-pointer shadow"
            >
              <Square className="w-3.5 h-3.5" /> Stop Workflow Execution
            </button>
          ) : (
            <button
              type="button"
              onClick={handleRunWorkflow}
              className="bg-purple-600 hover:bg-purple-500 text-white font-mono text-xs font-bold px-3 py-1.5 rounded-lg transition flex items-center gap-1.5 cursor-pointer shadow"
            >
              <Play className="w-3.5 h-3.5" /> Run Workflow
            </button>
          )}
        </div>
      </div>

      {/* Main Builder Grid */}
      <div className="flex-1 flex overflow-hidden">
        <NodeLibrarySidebar onAddNode={handleAddNode} />

        <WorkflowCanvas
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedNodeId}
          onSelectNode={setSelectedNodeId}
          onDeleteNode={handleDeleteNode}
          onOpenConfig={setConfiguringNode}
          onConnectNodes={(s, t) => setEdges([...edges, { id: `e-${Date.now()}`, sourceId: s, targetId: t }])}
          isExecuting={isExecuting}
          nodeStatuses={nodeStatuses}
        />
      </div>

      {/* Execution Logs Drawer */}
      {logs.length > 0 && (
        <div className="h-36 bg-slate-950 border-t border-slate-800 p-3 font-mono text-xs text-slate-300 overflow-y-auto space-y-1">
          <span className="text-[10px] font-bold text-purple-400 uppercase block mb-1">
            Pipeline Execution Stream Logs:
          </span>
          {logs.map((log, i) => (
            <div key={i}>{log}</div>
          ))}
        </div>
      )}

      {/* Modal Config overlay */}
      {configuringNode && (
        <NodeConfigModal
          node={configuringNode}
          onUpdateConfig={(updated) => {
            setNodes(prev => prev.map(n => n.id === configuringNode.id ? { ...n, config: updated, label: updated.nodeLabel || n.label } : n));
            setConfiguringNode(prev => prev ? { ...prev, config: updated, label: updated.nodeLabel || prev.label } : null);
          }}
          onClose={() => setConfiguringNode(null)}
        />
      )}

      {/* Scheduler Monitor Modal */}
      {showSchedulerModal && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="w-full max-w-5xl h-[85vh]">
            <SchedulerStatusPanel onClose={() => setShowSchedulerModal(false)} />
          </div>
        </div>
      )}
    </div>
  );
}
