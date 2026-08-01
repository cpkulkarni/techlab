/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import {
  MessageSquare,
  Search,
  Copy,
  Check,
  Send,
  Trash2,
  Code2,
  ChevronRight,
  Filter
} from 'lucide-react';
import { A2AMessage, A2AMessageType } from '../../multi_agent/types';

interface A2AMessageInspectorProps {
  messages: A2AMessage[];
  theme: 'white' | 'light-grey' | 'dark';
  onSendA2AMessage?: (msg: {
    sender_id: string;
    recipient_id: string;
    message_type: A2AMessageType;
    task: string;
  }) => void;
  onClearHistory?: () => void;
}

export default function A2AMessageInspector({
  messages,
  theme,
  onSendA2AMessage,
  onClearHistory,
}: A2AMessageInspectorProps) {
  const [selectedMessageId, setSelectedMessageId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMessageType, setFilterMessageType] = useState<string>('all');
  const [filterAgentId, setFilterAgentId] = useState<string>('all');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Manual dispatch form state
  const [showSendForm, setShowSendForm] = useState(false);
  const [sendSenderId, setSendSenderId] = useState('agent_coordinator');
  const [sendRecipientId, setSendRecipientId] = useState('agent_researcher');
  const [sendMessageType, setSendMessageType] = useState<A2AMessageType>('delegate');
  const [sendTask, setSendTask] = useState('');

  const isDark = theme === 'dark';

  const filteredMessages = messages.filter(msg => {
    if (filterMessageType !== 'all' && msg.message_type !== filterMessageType) return false;
    if (filterAgentId !== 'all' && msg.sender_id !== filterAgentId && msg.recipient_id !== filterAgentId) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const content = JSON.stringify(msg).toLowerCase();
      return content.includes(q);
    }
    return true;
  });

  const selectedMessage = messages.find(m => m.id === selectedMessageId) || filteredMessages[filteredMessages.length - 1] || null;

  const handleCopyPayload = (msg: A2AMessage) => {
    navigator.clipboard.writeText(JSON.stringify(msg, null, 2));
    setCopiedId(msg.id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleManualSend = (e: React.FormEvent) => {
    e.preventDefault();
    if (!sendTask.trim()) return;
    onSendA2AMessage?.({
      sender_id: sendSenderId,
      recipient_id: sendRecipientId,
      message_type: sendMessageType,
      task: sendTask.trim(),
    });
    setSendTask('');
    setShowSendForm(false);
  };

  const getTypeBadge = (type: A2AMessageType) => {
    switch (type) {
      case 'delegate':
        return <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">DELEGATE</span>;
      case 'request':
        return <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">REQUEST</span>;
      case 'response':
        return <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">RESPONSE</span>;
      case 'error':
        return <span className="px-1.5 py-0.5 text-[9px] font-mono font-bold rounded bg-rose-500/20 text-rose-400 border border-rose-500/30">ERROR</span>;
      default:
        return <span className="px-1.5 py-0.5 text-[9px] font-mono font-medium rounded bg-slate-500/20 text-slate-400 border border-slate-500/30">HEARTBEAT</span>;
    }
  };

  return (
    <div className="flex flex-col h-full gap-4">
      {/* Top Filter & Toolbar */}
      <div className={`p-4 rounded-xl border flex flex-wrap items-center justify-between gap-3 ${isDark ? 'bg-slate-900/60 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
        <div className="flex items-center gap-2 flex-1 min-w-[240px]">
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 absolute left-3 top-3 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search A2A JSON payload stream..."
              className={`w-full pl-9 pr-3 py-1.5 rounded-lg border text-xs font-mono ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200 text-slate-800'}`}
            />
          </div>

          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={filterMessageType}
              onChange={e => setFilterMessageType(e.target.value)}
              className={`p-1.5 rounded-lg border text-xs font-mono ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200 text-slate-800'}`}
            >
              <option value="all">All Types</option>
              <option value="request">Request</option>
              <option value="delegate">Delegate</option>
              <option value="response">Response</option>
              <option value="error">Error</option>
            </select>

            <select
              value={filterAgentId}
              onChange={e => setFilterAgentId(e.target.value)}
              className={`p-1.5 rounded-lg border text-xs font-mono ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-slate-50 border-slate-200 text-slate-800'}`}
            >
              <option value="all">All Agents</option>
              <option value="agent_coordinator">Coordinator</option>
              <option value="agent_researcher">Researcher</option>
              <option value="agent_coder">Coder</option>
              <option value="agent_tester">Tester</option>
            </select>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowSendForm(!showSendForm)}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-mono font-bold rounded-lg transition-all flex items-center gap-1.5 shadow-md cursor-pointer"
          >
            <Send className="w-3.5 h-3.5" />
            Manual Dispatch
          </button>
          {onClearHistory && (
            <button
              type="button"
              onClick={onClearHistory}
              className={`p-1.5 rounded-lg border transition-colors cursor-pointer ${isDark ? 'border-slate-800 text-slate-400 hover:text-rose-400' : 'border-slate-200 text-slate-600 hover:text-rose-600'}`}
              title="Clear A2A message log history"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Manual Dispatch Modal Form */}
      {showSendForm && (
        <div className={`p-4 rounded-xl border font-mono text-xs ${isDark ? 'bg-slate-900 border-indigo-500/40' : 'bg-indigo-50/50 border-indigo-200'}`}>
          <h4 className={`font-bold mb-3 ${isDark ? 'text-indigo-300' : 'text-indigo-900'}`}>
            Dispatch Manual A2A Protocol Message
          </h4>
          <form onSubmit={handleManualSend} className="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-1">Sender</label>
              <select
                value={sendSenderId}
                onChange={e => setSendSenderId(e.target.value)}
                className={`w-full p-2 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-white border-slate-200'}`}
              >
                <option value="agent_coordinator">Coordinator</option>
                <option value="agent_researcher">Researcher</option>
                <option value="agent_coder">Coder</option>
                <option value="agent_tester">Tester</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-1">Recipient</label>
              <select
                value={sendRecipientId}
                onChange={e => setSendRecipientId(e.target.value)}
                className={`w-full p-2 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-white border-slate-200'}`}
              >
                <option value="agent_researcher">Researcher</option>
                <option value="agent_coder">Coder</option>
                <option value="agent_tester">Tester</option>
                <option value="broadcast">Broadcast (All)</option>
              </select>
            </div>

            <div>
              <label className="block text-[10px] text-slate-500 uppercase mb-1">Message Type</label>
              <select
                value={sendMessageType}
                onChange={e => setSendMessageType(e.target.value as A2AMessageType)}
                className={`w-full p-2 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-white border-slate-200'}`}
              >
                <option value="delegate">delegate</option>
                <option value="request">request</option>
                <option value="response">response</option>
                <option value="error">error</option>
              </select>
            </div>

            <div className="md:col-span-4">
              <label className="block text-[10px] text-slate-500 uppercase mb-1">Task / Payload Content</label>
              <input
                type="text"
                value={sendTask}
                onChange={e => setSendTask(e.target.value)}
                placeholder="e.g. Inspect workspace files and generate test assertions"
                className={`w-full p-2 rounded border ${isDark ? 'bg-slate-950 border-slate-800 text-white' : 'bg-white border-slate-200'}`}
                required
              />
            </div>

            <div className="md:col-span-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setShowSendForm(false)}
                className="px-3 py-1.5 rounded border text-slate-400 hover:bg-slate-800"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-3 py-1.5 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-bold"
              >
                Dispatch Message
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Two Column Inspector: Message List + JSON Payload Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-[400px]">
        {/* Stream Message List */}
        <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-950/80 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
          <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-3">
            <span className={`text-xs font-mono font-bold uppercase tracking-wider ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              A2A Event Stream ({filteredMessages.length})
            </span>
            <span className="text-[10px] font-mono text-emerald-400 flex items-center gap-1">
              <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-ping" /> Real-Time SSE
            </span>
          </div>

          <div className="space-y-2 overflow-y-auto max-h-[480px] pr-1 flex-1">
            {filteredMessages.length === 0 ? (
              <div className="py-12 text-center text-slate-500 font-mono text-xs">
                No A2A protocol messages recorded yet.
              </div>
            ) : (
              filteredMessages.slice().reverse().map(msg => {
                const isSelected = selectedMessage?.id === msg.id;
                return (
                  <div
                    key={msg.id}
                    onClick={() => setSelectedMessageId(msg.id)}
                    className={`p-3 rounded-lg border cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-indigo-500/10 border-indigo-500 text-white shadow'
                        : isDark
                        ? 'bg-slate-900/80 border-slate-800 hover:border-slate-700'
                        : 'bg-white border-slate-200 hover:border-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[10px] font-mono gap-2 mb-1">
                      <div className="flex items-center gap-1.5">
                        <span className="font-bold text-indigo-400">{msg.sender_id}</span>
                        <ChevronRight className="w-3 h-3 text-slate-500" />
                        <span className="font-bold text-cyan-400">{msg.recipient_id}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        {getTypeBadge(msg.message_type)}
                      </div>
                    </div>

                    <p className={`text-xs font-mono line-clamp-2 ${isDark ? 'text-slate-300' : 'text-slate-800'}`}>
                      {msg.payload.task}
                    </p>

                    <div className="mt-2 flex items-center justify-between text-[9px] font-mono text-slate-500">
                      <span>Conv: {msg.conversation_id.slice(0, 12)}...</span>
                      <span>{new Date(msg.timestamp).toLocaleTimeString()}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* JSON Inspector & Syntax Highlighted View */}
        <div className={`p-4 rounded-xl border flex flex-col justify-between ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white border-slate-200 shadow-sm'}`}>
          {selectedMessage ? (
            <div className="space-y-3 flex-1 flex flex-col">
              <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                <div className="flex items-center gap-2">
                  <Code2 className="w-4 h-4 text-indigo-400" />
                  <span className={`text-xs font-mono font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>
                    A2A Payload JSON Schema
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => handleCopyPayload(selectedMessage)}
                  className="flex items-center gap-1 px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] font-mono cursor-pointer"
                >
                  {copiedId === selectedMessage.id ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  {copiedId === selectedMessage.id ? 'Copied' : 'Copy JSON'}
                </button>
              </div>

              {/* JSON Code Viewer */}
              <pre className={`p-4 rounded-lg border font-mono text-xs overflow-x-auto flex-1 max-h-[460px] ${isDark ? 'bg-slate-950 border-slate-800 text-indigo-300' : 'bg-slate-900 border-slate-800 text-indigo-300'}`}>
                {JSON.stringify(selectedMessage, null, 2)}
              </pre>
            </div>
          ) : (
            <div className="flex items-center justify-center flex-1 text-slate-500 text-xs font-mono">
              Select an A2A message from the stream to view full protocol JSON.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
