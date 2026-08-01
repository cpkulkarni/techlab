/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import {
  Layers,
  MessageSquare,
  Database,
  Activity,
  Play,
  Zap,
  Radio,
  Sparkles,
  ShieldAlert,
  Bot
} from 'lucide-react';
import AgentNetworkVisualizer from './AgentNetworkVisualizer';
import A2AMessageInspector from './A2AMessageInspector';
import MCPContextManager from './MCPContextManager';
import AgentExecutionTrace from './AgentExecutionTrace';
import ExternalMCPConnector from './ExternalMCPConnector';
import KnowledgeGraphViewer from './KnowledgeGraphViewer';
import { Network, BookOpen } from 'lucide-react';
import {
  AgentDefinition,
  A2AMessage,
  MCPTool,
  MCPResource,
  MCPServerInfo,
  MultiAgentTaskExecution,
  A2AMessageType,
} from '../../multi_agent/types';
import { ModelServerConfig } from '../../types';

interface MultiAgentWorkspaceProps {
  modelConfig: ModelServerConfig;
  theme: 'white' | 'light-grey' | 'dark';
}

export default function MultiAgentWorkspace({
  modelConfig,
  theme,
}: MultiAgentWorkspaceProps) {
  const [activeTab, setActiveTab] = useState<'topology' | 'a2a' | 'mcp' | 'external_mcp' | 'knowledge' | 'traces'>('topology');
  const [isEnabled, setIsEnabled] = useState(true);
  const [agents, setAgents] = useState<AgentDefinition[]>([]);
  const [messages, setMessages] = useState<A2AMessage[]>([]);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [resources, setResources] = useState<MCPResource[]>([]);
  const [servers, setServers] = useState<MCPServerInfo[]>([]);
  const [executions, setExecutions] = useState<MultiAgentTaskExecution[]>([]);

  // Task Launcher state
  const [taskPrompt, setTaskPrompt] = useState('');
  const [isLaunching, setIsLaunching] = useState(false);

  const isDark = theme === 'dark';

  // Load initial multi-agent state
  useEffect(() => {
    fetchConfig();
    fetchAgents();
    fetchMessages();
    fetchMCP();
    fetchExecutions();

    // Setup SSE live stream listener
    const eventSource = new EventSource('/api/multi-agent/stream');

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'a2a_message' && data.message) {
          setMessages(prev => [...prev, data.message]);
        } else if (data.type === 'agent_status_change') {
          setAgents(prev => prev.map(a => a.id === data.agentId ? { ...a, status: data.status } : a));
        } else if (data.type === 'agent_registered') {
          setAgents(prev => [...prev, data.agent]);
        } else if (data.type === 'execution_step' || data.type === 'execution_completed') {
          fetchExecutions();
        }
      } catch (err) {
        console.warn('Error parsing SSE event:', err);
      }
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const fetchConfig = async () => {
    try {
      const res = await fetch('/api/multi-agent/config');
      const data = await res.json();
      setIsEnabled(data.enabled !== false);
    } catch {}
  };

  const fetchAgents = async () => {
    try {
      const res = await fetch('/api/multi-agent/agents');
      const data = await res.json();
      if (data.agents) setAgents(data.agents);
    } catch {}
  };

  const fetchMessages = async () => {
    try {
      const res = await fetch('/api/multi-agent/a2a/messages');
      const data = await res.json();
      if (data.messages) setMessages(data.messages);
    } catch {}
  };

  const fetchMCP = async () => {
    try {
      const [tRes, rRes, sRes] = await Promise.all([
        fetch('/api/multi-agent/mcp/tools'),
        fetch('/api/multi-agent/mcp/resources'),
        fetch('/api/multi-agent/mcp/servers'),
      ]);
      const tData = await tRes.json();
      const rData = await rRes.json();
      const sData = await sRes.json();
      if (tData.tools) setTools(tData.tools);
      if (rData.resources) setResources(rData.resources);
      if (sData.servers) setServers(sData.servers);
    } catch {}
  };

  const fetchExecutions = async () => {
    try {
      const res = await fetch('/api/multi-agent/executions');
      const data = await res.json();
      if (data.executions) setExecutions(data.executions);
    } catch {}
  };

  const handleLaunchTask = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!taskPrompt.trim() || isLaunching) return;

    setIsLaunching(true);
    try {
      await fetch('/api/multi-agent/orchestrate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: taskPrompt.trim(), modelConfig }),
      });
      setTaskPrompt('');
      setActiveTab('traces');
      fetchExecutions();
      fetchMessages();
    } catch (err: any) {
      console.error('Task launch failed:', err);
    } finally {
      setIsLaunching(false);
    }
  };

  const handleToggleTool = async (name: string, enabled: boolean) => {
    try {
      await fetch('/api/multi-agent/mcp/tools/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, enabled }),
      });
      fetchMCP();
    } catch {}
  };

  const handleInvokeTool = async (name: string, args: Record<string, any>) => {
    const res = await fetch('/api/multi-agent/mcp/invoke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, args }),
    });
    return await res.json();
  };

  const handleSendA2AMessage = async (msg: {
    sender_id: string;
    recipient_id: string;
    message_type: A2AMessageType;
    task: string;
  }) => {
    try {
      await fetch('/api/multi-agent/a2a/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_id: msg.sender_id,
          recipient_id: msg.recipient_id,
          message_type: msg.message_type,
          payload: { task: msg.task },
        }),
      });
      fetchMessages();
    } catch {}
  };

  const handleRegisterAgent = async (newAgent: Partial<AgentDefinition>) => {
    try {
      await fetch('/api/multi-agent/agents/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newAgent),
      });
      fetchAgents();
    } catch {}
  };

  const handleSendDirectMessage = async (msg: { recipient_id: string; message: string }) => {
    try {
      await fetch('/api/multi-agent/a2a/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_id: 'human_operator',
          recipient_id: msg.recipient_id,
          message_type: 'human_direct',
          payload: { task: msg.message },
          channel: 'human_to_agent',
        }),
      });
      fetchMessages();
    } catch {}
  };

  const handleHitlAction = async (action: 'approve' | 'reject', messageId: string, modifiedOutput?: string, feedback?: string) => {
    try {
      await fetch('/api/multi-agent/a2a/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sender_id: 'human_operator',
          recipient_id: 'agent_coordinator',
          message_type: action === 'approve' ? 'hitl_approval' : 'hitl_rejection',
          payload: {
            task: `HITL Verification ${action.toUpperCase()}D by Human Operator`,
            approved: action === 'approve',
            modified_output: modifiedOutput,
            human_feedback: feedback,
          },
          channel: 'human_to_agent',
        }),
      });
      fetchMessages();
      fetchExecutions();
    } catch {}
  };

  if (!isEnabled) {
    return (
      <div className={`p-8 rounded-xl border text-center font-mono ${isDark ? 'bg-slate-900 border-slate-800 text-slate-400' : 'bg-slate-50 border-slate-200 text-slate-600'}`}>
        <ShieldAlert className="w-12 h-12 text-amber-500 mx-auto mb-3" />
        <h3 className="text-base font-bold mb-1">Multi-Agent System Disabled</h3>
        <p className="text-xs">
          Multi-agent collaboration features are turned off by feature flag (ENABLE_MULTI_AGENT=false).
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full gap-4 p-4">
      {/* Header Toolbar & Task Launcher Input */}
      <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-4 ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 text-white shadow-md">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className={`text-base font-bold font-mono ${isDark ? 'text-white' : 'text-slate-900'}`}>
                Multi-Agent System (MCP & A2A)
              </h2>
              <span className="px-2 py-0.5 text-[9px] font-mono font-bold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
                ACTIVE ORCHESTRATOR
              </span>
            </div>
            <p className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Autonomous multi-agent task delegation, Model Context Protocol tools, and A2A event stream
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className={`flex items-center gap-1 p-1 rounded-xl border ${isDark ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'}`}>
          <button
            type="button"
            onClick={() => setActiveTab('topology')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'topology'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" /> Topology Graph
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('a2a')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'a2a'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" /> A2A Inspector ({messages.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('mcp')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'mcp'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Database className="w-3.5 h-3.5" /> MCP Explorer ({tools.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('external_mcp')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'external_mcp'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Network className="w-3.5 h-3.5" /> External MCP
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('knowledge')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'knowledge'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" /> Knowledge Vault
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('traces')}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono font-bold transition-all flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'traces'
                ? 'bg-indigo-600 text-white shadow'
                : isDark ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <Activity className="w-3.5 h-3.5" /> Task Traces ({executions.length})
          </button>
        </div>
      </div>

      {/* Task Launcher Bar */}
      <form onSubmit={handleLaunchTask} className={`p-3 rounded-xl border flex items-center gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <Bot className="w-5 h-5 text-indigo-400 shrink-0" />
        <input
          type="text"
          value={taskPrompt}
          onChange={e => setTaskPrompt(e.target.value)}
          placeholder="Launch Multi-Agent Task (e.g. Audit workspace files, research best practices, and verify with test suite)..."
          className={`flex-1 bg-transparent text-xs font-mono focus:outline-none ${isDark ? 'text-white placeholder-slate-500' : 'text-slate-900 placeholder-slate-400'}`}
        />
        <button
          type="submit"
          disabled={isLaunching || !taskPrompt.trim()}
          className="px-4 py-2 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 text-white font-mono font-bold text-xs rounded-lg transition-all flex items-center gap-2 shadow-md cursor-pointer shrink-0"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          {isLaunching ? 'Orchestrating...' : 'Launch Multi-Agent Task'}
        </button>
      </form>

      {/* Tab Panels */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'topology' && (
          <AgentNetworkVisualizer
            agents={agents}
            recentMessages={messages}
            theme={theme}
            onRegisterAgent={handleRegisterAgent}
            onSendDirectMessage={handleSendDirectMessage}
            onHitlAction={handleHitlAction}
          />
        )}

        {activeTab === 'a2a' && (
          <A2AMessageInspector
            messages={messages}
            theme={theme}
            onSendA2AMessage={handleSendA2AMessage}
            onClearHistory={() => setMessages([])}
          />
        )}

        {activeTab === 'mcp' && (
          <MCPContextManager
            tools={tools}
            resources={resources}
            servers={servers}
            theme={theme}
            onToggleTool={handleToggleTool}
            onInvokeTool={handleInvokeTool}
          />
        )}

        {activeTab === 'external_mcp' && (
          <ExternalMCPConnector theme={theme} />
        )}

        {activeTab === 'knowledge' && (
          <KnowledgeGraphViewer theme={theme} />
        )}

        {activeTab === 'traces' && (
          <AgentExecutionTrace
            executions={executions}
            theme={theme}
          />
        )}
      </div>
    </div>
  );
}
