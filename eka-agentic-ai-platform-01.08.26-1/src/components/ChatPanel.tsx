/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useRef, useEffect } from 'react';
import { AgentMode, ChatMessage, AgentWorkflow, FileAttachment, ModelServerConfig } from '../types';
import MultimodalStudio from './MultimodalStudio';
import { Trash2, StopCircle, RefreshCw, PanelRightClose } from 'lucide-react';
import { ChatMessageItem } from './chat/ChatMessageItem';
import { ChatInputArea } from './chat/ChatInputArea';

interface ChatPanelProps {
  activeMode: AgentMode;
  messages: ChatMessage[];
  workflow: AgentWorkflow | null;
  isExecuting: boolean;
  onSendMessage: (text: string, mode: AgentMode, search: boolean, attachments?: FileAttachment[]) => void;
  onStopExecution: (targetMode?: AgentMode) => void;
  onStopStep?: (stepId: string) => void;
  onClearMemory: (mode: AgentMode) => void;
  searchEnabled: boolean;
  setSearchEnabled: (val: boolean) => void;
  modelConfig?: ModelServerConfig;
  multimodalFeature?: string;
  width?: number;
  leftMaximized?: boolean;
  onToggleMinimize?: () => void;
}

export default function ChatPanel({
  activeMode,
  messages,
  workflow,
  isExecuting,
  onSendMessage,
  onStopExecution,
  onClearMemory,
  searchEnabled,
  setSearchEnabled,
  modelConfig,
  multimodalFeature = 'text_to_image',
  onToggleMinimize,
}: ChatPanelProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, workflow]);

  const handleRetry = (promptText: string) => {
    onSendMessage(promptText, activeMode, searchEnabled);
  };

  if (activeMode === 'multimodal') {
    return (
      <div className="h-full bg-slate-950 border-l border-slate-800 flex flex-col">
        <MultimodalStudio
          initialFeature={multimodalFeature as any}
          modelConfig={modelConfig}
        />
      </div>
    );
  }

  return (
    <div className="h-full bg-slate-950 border-l border-slate-800 flex flex-col justify-between overflow-hidden">
      {/* Header Bar */}
      <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/80 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-bold font-mono text-slate-200 capitalize">
            {activeMode} Assistant
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          {isExecuting && (
            <button
              type="button"
              onClick={() => onStopExecution(activeMode)}
              className="flex items-center gap-1 text-[10px] font-mono bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-800 px-2 py-0.5 rounded cursor-pointer transition"
            >
              <StopCircle className="w-3 h-3 text-rose-400" /> Stop Process
            </button>
          )}

          <button
            type="button"
            onClick={() => onClearMemory(activeMode)}
            className="p-1 rounded text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition cursor-pointer"
            title="Clear mode conversation memory"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>

          {onToggleMinimize && (
            <button
              type="button"
              onClick={onToggleMinimize}
              className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer ml-0.5"
              title="Minimize Chat Panel"
            >
              <PanelRightClose className="w-3.5 h-3.5 theme-accent-text" />
            </button>
          )}
        </div>
      </div>

      {/* Messages Scroll View */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-4 text-slate-400 space-y-2">
            <div className="w-10 h-10 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center text-indigo-400 font-bold font-mono">
              E
            </div>
            <p className="text-xs font-medium text-slate-300">
              Welcome to the {activeMode.toUpperCase()} Workspace
            </p>
            <p className="text-[11px] text-slate-400 max-w-xs">
              Type your prompt below to start generating code, architecture docs, research reports, or running tests.
            </p>
          </div>
        ) : (
          messages.map(msg => (
            <ChatMessageItem
              key={msg.id}
              message={msg}
              onRetry={handleRetry}
            />
          ))
        )}

        {/* Active execution status indicator */}
        {isExecuting && (
          <div className="flex items-center gap-2 p-2.5 rounded-xl bg-indigo-950/40 border border-indigo-800/60 text-indigo-300 text-xs font-mono">
            <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
            <span>AI model processing request...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input Form */}
      <ChatInputArea
        onSendMessage={(text, search, atts) => onSendMessage(text, activeMode, search, atts)}
        onStopExecution={() => onStopExecution(activeMode)}
        isExecuting={isExecuting}
        searchEnabled={searchEnabled}
        setSearchEnabled={setSearchEnabled}
      />
    </div>
  );
}
