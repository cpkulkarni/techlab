/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Globe,
  Code2,
  CheckSquare,
  Bot,
  Zap,
  Radio,
  Plus,
  Shield,
  Activity,
  Layers,
  ChevronRight,
  Sparkles,
  ShieldCheck,
  UserCheck,
  Send,
  MessageSquare,
  Filter,
  CheckCircle2,
  XCircle,
  Edit3,
  Sliders,
  AlertTriangle,
  RefreshCw,
  Eye
} from 'lucide-react';
import { AgentDefinition, AgentStatus, A2AMessage, A2AMessageType } from '../../multi_agent/types';

interface AgentNetworkVisualizerProps {
  agents: AgentDefinition[];
  recentMessages: A2AMessage[];
  theme: 'white' | 'light-grey' | 'dark';
  onRegisterAgent?: (agent: Partial<AgentDefinition>) => void;
  onSendDirectMessage?: (msg: { recipient_id: string; message: string }) => void;
  onHitlAction?: (action: 'approve' | 'reject', messageId: string, modifiedOutput?: string, feedback?: string) => void;
}

export default function AgentNetworkVisualizer({
  agents,
  recentMessages,
  theme,
  onRegisterAgent,
  onSendDirectMessage,
  onHitlAction,
}: AgentNetworkVisualizerProps) {
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>('agent_coordinator');
  const [showAddModal, setShowAddModal] = useState(false);
  const [isFeatureEnabled, setIsFeatureEnabled] = useState<boolean | null>(null);

  // New Agent Form State
  const [newAgentName, setNewAgentName] = useState('');
  const [newAgentRole, setNewAgentRole] = useState<'coordinator' | 'researcher' | 'coder' | 'tester' | 'hitl_verifier' | 'specialist'>('specialist');
  const [newAgentPrompt, setNewAgentPrompt] = useState('');

  // Log View Filter
  const [logFilter, setLogFilter] = useState<'all' | 'human_agent' | 'agent_agent' | 'hitl'>('all');
  const [selectedLogMessage, setSelectedLogMessage] = useState<A2AMessage | null>(null);

  // Human Direct Message Input
  const [directMsgText, setDirectMsgText] = useState('');
  const [targetAgentId, setTargetAgentId] = useState<string>('agent_coordinator');

  // HITL Interactive Verification State
  const [hitlGatekeeperActive, setHitlGatekeeperActive] = useState<boolean>(true);
  const [editingOutput, setEditingOutput] = useState<string>('');
  const [humanFeedback, setHumanFeedback] = useState<string>('');

  const isDark = theme === 'dark';

  // Check feature flag ENABLE_MULTI_AGENT on mount
  useEffect(() => {
    fetch('/api/multi-agent/config')
      .then(res => res.json())
      .then(data => setIsFeatureEnabled(data.enabled !== false))
      .catch(() => setIsFeatureEnabled(true));
  }, []);

  const selectedAgent = agents.find(a => a.id === selectedAgentId) || agents[0];

  const getAgentIcon = (role?: string) => {
    switch (role) {
      case 'coordinator': return Cpu;
      case 'researcher': return Globe;
      case 'coder': return Code2;
      case 'tester': return CheckSquare;
      case 'hitl_verifier': return ShieldCheck;
      default: return Bot;
    }
  };

  const getStatusBadge = (status: AgentStatus) => {
    switch (status) {
      case 'thinking':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 animate-pulse flex items-center gap-1"><Zap className="w-2.5 h-2.5" /> THINKING</span>;
      case 'delegating':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 animate-pulse flex items-center gap-1"><Radio className="w-2.5 h-2.5" /> DELEGATING</span>;
      case 'executing':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 animate-pulse flex items-center gap-1"><Activity className="w-2.5 h-2.5" /> EXECUTING</span>;
      case 'awaiting_hitl':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-pink-500/20 text-pink-400 border border-pink-500/30 animate-pulse flex items-center gap-1"><UserCheck className="w-2.5 h-2.5" /> AWAITING HITL</span>;
      case 'error':
        return <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">ERROR</span>;
      default:
        return <span className="px-2 py-0.5 text-[9px] font-mono font-medium rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">IDLE</span>;
    }
  };

  // Classify message channels
  const classifiedMessages = recentMessages.map(msg => {
    const isHumanSender = msg.sender_id === 'user_system' || msg.sender_id === 'human_operator';
    const isHumanRecipient = msg.recipient_id === 'human_operator' || msg.recipient_id === 'user_system';
    const isHitl = msg.message_type === 'hitl_request' || msg.message_type === 'hitl_approval' || msg.message_type === 'hitl_rejection';

    let channel: 'human_agent' | 'agent_to_agent' | 'hitl' = 'agent_to_agent';
    if (isHitl) channel = 'hitl';
    else if (isHumanSender || isHumanRecipient) channel = 'human_agent';

    return { ...msg, channel };
  });

  const filteredLogs = classifiedMessages.filter(msg => {
    if (logFilter === 'human_agent') return msg.channel === 'human_agent';
    if (logFilter === 'agent_agent') return msg.channel === 'agent_to_agent';
    if (logFilter === 'hitl') return msg.channel === 'hitl';
    return true;
  });

  // Active HITL verification requests requiring human approval
  const pendingHitlRequests = recentMessages.filter(
    m => m.message_type === 'hitl_request' && !recentMessages.some(
      reply => (reply.message_type === 'hitl_approval' || reply.message_type === 'hitl_rejection') && reply.conversation_id === m.conversation_id
    )
  );

  const handleCreateAgent = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAgentName.trim()) return;
    const id = `agent_${Date.now()}`;
    onRegisterAgent?.({
      id,
      name: newAgentName.trim(),
      role: newAgentRole,
      description: `Custom ${newAgentRole} agent created dynamically.`,
      systemPrompt: newAgentPrompt.trim() || `You are an autonomous AI Agent with role ${newAgentRole}.`,
      capabilities: ['custom_task'],
      mcpToolsAllowed: ['read_file', 'list_files'],
      status: 'idle',
      avatarIcon: newAgentRole === 'hitl_verifier' ? 'ShieldCheck' : 'Bot',
      color: newAgentRole === 'hitl_verifier' ? '#ec4899' : '#8b5cf6',
    });
    setNewAgentName('');
    setNewAgentPrompt('');
    setShowAddModal(false);
  };

  const handleSendDirect = (e: React.FormEvent) => {
    e.preventDefault();
    if (!directMsgText.trim()) return;
    onSendDirectMessage?.({
      recipient_id: targetAgentId,
      message: directMsgText.trim(),
    });
    setDirectMsgText('');
  };

  if (isFeatureEnabled === false) {
    return (
      <div className={`p-8 rounded-xl border text-center font-mono ${isDark ? 'bg-slate-900 border-slate-800 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>
        <AlertTriangle className="w-10 h-10 text-amber-500 mx-auto mb-3" />
        <h3 className="text-sm font-bold text-white mb-1">Multi-Agent System Disabled</h3>
        <p className="text-xs">
          The `ENABLE_MULTI_AGENT` feature flag is currently disabled in system configuration.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Top Banner & Control Bar */}
      <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className={`text-sm font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                Agent Network Visualizer & Communication Hub
              </h3>
              <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                ENABLE_MULTI_AGENT = TRUE
              </span>
            </div>
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Real-time topology graph, Human ↔ Agent log monitoring, and HITL gatekeeper verifications
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* HITL Gatekeeper Toggle */}
          <button
            type="button"
            onClick={() => setHitlGatekeeperActive(!hitlGatekeeperActive)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 border cursor-pointer ${
              hitlGatekeeperActive
                ? 'bg-pink-500/20 border-pink-500/40 text-pink-400 hover:bg-pink-500/30'
                : isDark ? 'bg-slate-800 border-slate-700 text-slate-400' : 'bg-slate-100 border-slate-200 text-slate-600'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5 text-pink-400" />
            HITL Gatekeeper: {hitlGatekeeperActive ? 'ENABLED' : 'BYPASS'}
          </button>

          <button
            type="button"
            onClick={() => setShowAddModal(true)}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-mono font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-md cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" />
            Register Agent
          </button>
        </div>
      </div>

      {/* Main Grid: Visual Topology Graph + Selected Agent Properties */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Topology Node Graph Visualizer */}
        <div className={`lg:col-span-2 p-6 rounded-xl border relative overflow-hidden flex flex-col justify-between min-h-[380px] ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
          {/* Grid canvas background */}
          <div className="absolute inset-0 opacity-[0.03] bg-[radial-gradient(#6366f1_1px,transparent_1px)] [background-size:16px_16px] pointer-events-none" />

          {/* Topology Header */}
          <div className="flex items-center justify-between z-10 mb-4">
            <div className="flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span className={`text-xs font-mono font-bold uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
                Active Network Nodes ({agents.length})
              </span>
            </div>
            <div className="flex items-center gap-3 text-[10px] font-mono text-slate-500">
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> Live A2A Broker</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-pink-500" /> Human Operator</span>
            </div>
          </div>

          {/* Graph Nodes Grid */}
          <div className="relative z-10 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 my-2">
            {agents.map((agent) => {
              const Icon = getAgentIcon(agent.role);
              const isSelected = agent.id === selectedAgentId;
              const isCoordinator = agent.role === 'coordinator';
              const isHitlVerifier = agent.role === 'hitl_verifier';

              return (
                <div
                  key={agent.id}
                  onClick={() => setSelectedAgentId(agent.id)}
                  className={`p-3.5 rounded-xl border cursor-pointer transition-all relative ${
                    isSelected
                      ? 'border-indigo-500 bg-indigo-500/10 shadow-lg scale-[1.02]'
                      : isDark
                      ? 'bg-slate-900/90 border-slate-800 hover:border-slate-700 hover:bg-slate-900'
                      : 'bg-white border-slate-200 hover:border-indigo-300 hover:bg-slate-50 shadow-sm'
                  } ${isCoordinator ? 'border-dashed border-indigo-500/60' : ''} ${isHitlVerifier ? 'border-pink-500/40' : ''}`}
                >
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2.5">
                      <div
                        className="p-2 rounded-lg text-white font-bold shadow-sm shrink-0"
                        style={{ backgroundColor: agent.color || '#6366f1' }}
                      >
                        <Icon className="w-4 h-4" />
                      </div>
                      <div className="min-w-0">
                        <h4 className={`text-xs font-bold font-mono truncate ${isDark ? 'text-white' : 'text-slate-900'}`}>
                          {agent.name}
                        </h4>
                        <p className={`text-[9px] font-mono uppercase ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                          Role: {agent.role}
                        </p>
                      </div>
                    </div>
                    {getStatusBadge(agent.status)}
                  </div>

                  {/* Capabilities Tags */}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {agent.capabilities.slice(0, 3).map((cap) => (
                      <span
                        key={cap}
                        className={`text-[8px] font-mono px-1.5 py-0.5 rounded border ${
                          isDark ? 'bg-slate-950 border-slate-800 text-slate-400' : 'bg-slate-100 border-slate-200 text-slate-600'
                        }`}
                      >
                        {cap}
                      </span>
                    ))}
                  </div>

                  {/* Communication Ping Badge */}
                  {recentMessages.some(m => m.sender_id === agent.id || m.recipient_id === agent.id) && (
                    <div className="absolute -top-1 -right-1 w-3 h-3 bg-indigo-500 rounded-full animate-ping" />
                  )}
                </div>
              );
            })}
          </div>

          <div className={`pt-3 border-t flex items-center justify-between text-[10px] font-mono ${isDark ? 'border-slate-800 text-slate-400' : 'border-slate-200 text-slate-600'}`}>
            <span>Protocol: A2A Event Bus + MCP Client</span>
            <span>Total Logged Messages: {recentMessages.length}</span>
          </div>
        </div>

        {/* Selected Agent Details Panel */}
        <div className={`p-5 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
          {selectedAgent ? (
            <div className="space-y-3 text-xs font-mono">
              <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2.5">
                  <div
                    className="p-2 rounded-lg text-white font-bold"
                    style={{ backgroundColor: selectedAgent.color || '#6366f1' }}
                  >
                    {React.createElement(getAgentIcon(selectedAgent.role), { className: 'w-4 h-4' })}
                  </div>
                  <div>
                    <h4 className={`text-xs font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>
                      {selectedAgent.name}
                    </h4>
                    <span className="text-[10px] text-slate-500">ID: {selectedAgent.id}</span>
                  </div>
                </div>
                {getStatusBadge(selectedAgent.status)}
              </div>

              <div>
                <label className="text-[10px] uppercase text-slate-500 block mb-1">Description</label>
                <p className={`${isDark ? 'text-slate-300' : 'text-slate-700'}`}>
                  {selectedAgent.description}
                </p>
              </div>

              <div>
                <label className="text-[10px] uppercase text-slate-500 block mb-1">System Prompt</label>
                <div className={`p-2 rounded border text-[10px] max-h-28 overflow-y-auto ${isDark ? 'bg-slate-950 border-slate-800 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-800'}`}>
                  {selectedAgent.systemPrompt}
                </div>
              </div>

              <div>
                <label className="text-[10px] uppercase text-slate-500 block mb-1 flex items-center gap-1">
                  <Shield className="w-3 h-3 text-indigo-400" /> Allowed MCP Tools ({selectedAgent.mcpToolsAllowed.length})
                </label>
                <div className="flex flex-wrap gap-1">
                  {selectedAgent.mcpToolsAllowed.map(toolName => (
                    <span
                      key={toolName}
                      className="px-2 py-0.5 rounded text-[9px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20"
                    >
                      {toolName}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-xs text-slate-500 font-mono">Select an agent node to inspect properties.</p>
          )}

          <div className={`mt-3 pt-3 border-t text-[10px] font-mono text-slate-500 flex items-center gap-1 ${isDark ? 'border-slate-800' : 'border-slate-200'}`}>
            <ChevronRight className="w-3 h-3 text-indigo-400" />
            Configured in defaultAgents.ts
          </div>
        </div>
      </div>

      {/* Human-in-the-Loop Verification Queue (If pending approval) */}
      {hitlGatekeeperActive && pendingHitlRequests.length > 0 && (
        <div className={`p-4 rounded-xl border border-pink-500/50 ${isDark ? 'bg-pink-950/20' : 'bg-pink-50'}`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-pink-400" />
              <h4 className="text-xs font-mono font-bold text-pink-400 uppercase tracking-wider">
                Human-in-the-Loop Review Required ({pendingHitlRequests.length})
              </h4>
            </div>
            <span className="text-[10px] font-mono text-pink-300/80">Subtask execution paused awaiting human operator verification</span>
          </div>

          {pendingHitlRequests.map(req => (
            <div key={req.id} className={`p-3 rounded-lg border mb-2 text-xs font-mono ${isDark ? 'bg-slate-900/90 border-slate-800' : 'bg-white border-slate-200'}`}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-indigo-400 font-bold">Sender: {req.sender_id}</span>
                <span className="text-slate-500 text-[10px]">{new Date(req.timestamp).toLocaleTimeString()}</span>
              </div>
              <p className="text-slate-300 mb-2 font-bold">{req.payload.task}</p>
              
              {/* Proposed Output Box */}
              <div className="mb-3">
                <label className="text-[10px] uppercase text-slate-500 block mb-1">Proposed Output (Inspect & Edit):</label>
                <textarea
                  value={editingOutput || req.payload.result || ''}
                  onChange={e => setEditingOutput(e.target.value)}
                  rows={3}
                  className={`w-full p-2 rounded border font-mono text-xs ${isDark ? 'bg-slate-950 border-slate-800 text-slate-200' : 'bg-slate-50 border-slate-200'}`}
                />
              </div>

              {/* Feedback Input */}
              <div className="mb-3">
                <label className="text-[10px] uppercase text-slate-500 block mb-1">Human Guidance / Feedback Notes:</label>
                <input
                  type="text"
                  value={humanFeedback}
                  onChange={e => setHumanFeedback(e.target.value)}
                  placeholder="Optional guidance or adjustment notes..."
                  className={`w-full p-2 rounded border font-mono text-xs ${isDark ? 'bg-slate-950 border-slate-800 text-slate-200' : 'bg-slate-50 border-slate-200'}`}
                />
              </div>

              <div className="flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => {
                    onHitlAction?.('reject', req.id, editingOutput, humanFeedback);
                    setEditingOutput('');
                    setHumanFeedback('');
                  }}
                  className="px-3 py-1.5 rounded bg-rose-600 hover:bg-rose-500 text-white font-bold flex items-center gap-1 cursor-pointer"
                >
                  <XCircle className="w-3.5 h-3.5" />
                  Reject & Re-run
                </button>
                <button
                  type="button"
                  onClick={() => {
                    onHitlAction?.('approve', req.id, editingOutput, humanFeedback);
                    setEditingOutput('');
                    setHumanFeedback('');
                  }}
                  className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-bold flex items-center gap-1 cursor-pointer"
                >
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Approve & Resume
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Human Direct Message Dispatcher */}
      <form onSubmit={handleSendDirect} className={`p-3 rounded-xl border flex items-center gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <UserCheck className="w-5 h-5 text-pink-400 shrink-0" />
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-slate-400 shrink-0">Send Direct Instruction To:</span>
          <select
            value={targetAgentId}
            onChange={e => setTargetAgentId(e.target.value)}
            className={`px-2 py-1 rounded border text-xs font-mono ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-100 border-slate-200'}`}
          >
            {agents.map(a => (
              <option key={a.id} value={a.id}>{a.name} ({a.role})</option>
            ))}
          </select>
        </div>
        <input
          type="text"
          value={directMsgText}
          onChange={e => setDirectMsgText(e.target.value)}
          placeholder="Type human prompt or feedback directly to agent..."
          className={`flex-1 bg-transparent text-xs font-mono focus:outline-none ${isDark ? 'text-white placeholder-slate-500' : 'text-slate-900 placeholder-slate-400'}`}
        />
        <button
          type="submit"
          disabled={!directMsgText.trim()}
          className="px-3 py-1.5 bg-pink-600 hover:bg-pink-500 disabled:opacity-50 text-white text-xs font-mono font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-md cursor-pointer shrink-0"
        >
          <Send className="w-3.5 h-3.5" />
          Send to Agent
        </button>
      </form>

      {/* Comprehensive Communication Logs Stream (Human ↔ Agent & Agent ↔ Agent) */}
      <div className={`p-5 rounded-xl border flex flex-col gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-indigo-400" />
            <h4 className={`text-xs font-mono font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Communication Logs (Human ↔ Agent & Agent ↔ Agent)
            </h4>
          </div>

          {/* Log Filter Pills */}
          <div className={`flex items-center gap-1 p-1 rounded-lg border text-[10px] font-mono ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
            <button
              type="button"
              onClick={() => setLogFilter('all')}
              className={`px-2.5 py-1 rounded font-bold transition-all cursor-pointer ${
                logFilter === 'all' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              All Logs ({classifiedMessages.length})
            </button>
            <button
              type="button"
              onClick={() => setLogFilter('human_agent')}
              className={`px-2.5 py-1 rounded font-bold transition-all cursor-pointer ${
                logFilter === 'human_agent' ? 'bg-pink-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Human ↔ Agent ({classifiedMessages.filter(m => m.channel === 'human_agent').length})
            </button>
            <button
              type="button"
              onClick={() => setLogFilter('agent_agent')}
              className={`px-2.5 py-1 rounded font-bold transition-all cursor-pointer ${
                logFilter === 'agent_agent' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              Agent ↔ Agent ({classifiedMessages.filter(m => m.channel === 'agent_to_agent').length})
            </button>
            <button
              type="button"
              onClick={() => setLogFilter('hitl')}
              className={`px-2.5 py-1 rounded font-bold transition-all cursor-pointer ${
                logFilter === 'hitl' ? 'bg-amber-600 text-white' : 'text-slate-400 hover:text-white'
              }`}
            >
              HITL Checkpoints ({classifiedMessages.filter(m => m.channel === 'hitl').length})
            </button>
          </div>
        </div>

        {/* Log Entries List */}
        <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
          {filteredLogs.length === 0 ? (
            <p className="text-xs text-slate-500 font-mono py-4 text-center">No communication logs recorded yet for this filter.</p>
          ) : (
            filteredLogs.map((msg) => {
              const isHuman = msg.channel === 'human_agent';
              const isHitl = msg.channel === 'hitl';

              return (
                <div
                  key={msg.id}
                  onClick={() => setSelectedLogMessage(selectedLogMessage?.id === msg.id ? null : msg)}
                  className={`p-3 rounded-lg border text-xs font-mono cursor-pointer transition-all ${
                    isHitl
                      ? 'bg-pink-500/10 border-pink-500/30 hover:border-pink-500/50'
                      : isHuman
                      ? 'bg-purple-500/10 border-purple-500/30 hover:border-purple-500/50'
                      : isDark
                      ? 'bg-slate-950 border-slate-800 hover:border-slate-700'
                      : 'bg-slate-50 border-slate-200 hover:border-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      <span className={`px-1.5 py-0.5 text-[9px] font-bold rounded uppercase ${
                        isHitl ? 'bg-pink-500/20 text-pink-400' : isHuman ? 'bg-purple-500/20 text-purple-400' : 'bg-indigo-500/20 text-indigo-400'
                      }`}>
                        {msg.channel === 'hitl' ? 'HITL GATEKEEPER' : msg.channel === 'human_agent' ? 'HUMAN ↔ AGENT' : 'A2A AGENT ↔ AGENT'}
                      </span>
                      <span className="font-bold text-slate-300">{msg.sender_id}</span>
                      <ChevronRight className="w-3 h-3 text-slate-500" />
                      <span className="font-bold text-indigo-400">{msg.recipient_id}</span>
                    </div>
                    <span className="text-[10px] text-slate-500">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                  </div>

                  <p className="text-slate-300 font-medium truncate mb-1">
                    {msg.payload.task || msg.payload.result || JSON.stringify(msg.payload)}
                  </p>

                  {/* Expandable Payload View */}
                  {selectedLogMessage?.id === msg.id && (
                    <div className={`mt-2 pt-2 border-t text-[11px] space-y-1 ${isDark ? 'border-slate-800 text-slate-400' : 'border-slate-200 text-slate-600'}`}>
                      <div><strong className="text-indigo-400">Message ID:</strong> {msg.id}</div>
                      <div><strong className="text-indigo-400">Message Type:</strong> {msg.message_type}</div>
                      <div><strong className="text-indigo-400">Conversation ID:</strong> {msg.conversation_id}</div>
                      <div>
                        <strong className="text-indigo-400 block mb-1">Full Payload:</strong>
                        <pre className="p-2 rounded bg-black/40 text-[10px] overflow-x-auto text-emerald-400">
                          {JSON.stringify(msg.payload, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Add Agent Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className={`w-full max-w-md p-6 rounded-xl border shadow-2xl ${isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
            <h3 className={`text-sm font-bold font-mono mb-4 ${isDark ? 'text-white' : 'text-slate-900'}`}>
              Register New Agent
            </h3>
            <form onSubmit={handleCreateAgent} className="space-y-4 text-xs font-mono">
              <div>
                <label className="block text-slate-400 mb-1">Agent Name</label>
                <input
                  type="text"
                  value={newAgentName}
                  onChange={e => setNewAgentName(e.target.value)}
                  placeholder="e.g. Code Review & Security Agent"
                  className={`w-full p-2.5 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200'}`}
                  required
                />
              </div>
              <div>
                <label className="block text-slate-400 mb-1">Role</label>
                <select
                  value={newAgentRole}
                  onChange={e => setNewAgentRole(e.target.value as any)}
                  className={`w-full p-2.5 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200'}`}
                >
                  <option value="specialist">Specialist</option>
                  <option value="researcher">Researcher</option>
                  <option value="coder">Coder</option>
                  <option value="tester">Tester</option>
                  <option value="hitl_verifier">HITL Verifier</option>
                  <option value="coordinator">Coordinator</option>
                </select>
              </div>
              <div>
                <label className="block text-slate-400 mb-1">System Prompt</label>
                <textarea
                  value={newAgentPrompt}
                  onChange={e => setNewAgentPrompt(e.target.value)}
                  placeholder="Specify system prompt instructions..."
                  rows={4}
                  className={`w-full p-2.5 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200'}`}
                />
              </div>
              <div className="flex items-center justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowAddModal(false)}
                  className={`px-3 py-1.5 rounded border text-xs font-bold ${isDark ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-slate-100 border-slate-200 text-slate-700'}`}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-bold"
                >
                  Register Agent
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
