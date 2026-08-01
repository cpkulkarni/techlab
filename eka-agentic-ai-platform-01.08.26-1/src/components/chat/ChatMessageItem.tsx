/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ChatMessage } from '../../types';
import { User, Bot, RefreshCw, ExternalLink, Paperclip } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ChatMessageItemProps {
  message: ChatMessage;
  onRetry?: (messageText: string) => void;
}

export function ChatMessageItem({ message, onRetry }: ChatMessageItemProps) {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 text-xs ${isUser ? 'flex-row-reverse' : 'flex-row'} group`}>
      {/* Avatar */}
      <div
        className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 font-bold ${
          isUser
            ? 'bg-indigo-600 text-white'
            : 'bg-slate-800 text-indigo-400 border border-slate-700'
        }`}
      >
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message Content Bubble */}
      <div
        className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 space-y-2 border ${
          isUser
            ? 'bg-indigo-600/90 text-white border-indigo-500 rounded-tr-none'
            : 'bg-slate-900/90 text-slate-200 border-slate-800 rounded-tl-none shadow-md'
        }`}
      >
        {/* Timestamp */}
        <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono mb-1">
          <span>{isUser ? 'You' : 'Eka Assistant'}</span>
          <span>{message.timestamp}</span>
        </div>

        {/* Attachments if any */}
        {message.attachments && message.attachments.length > 0 && (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {message.attachments.map(att => (
              <div
                key={att.id}
                className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-800/80 border border-slate-700 text-[10px] font-mono text-slate-300"
              >
                <Paperclip className="w-3 h-3 text-indigo-400" />
                <span className="truncate max-w-[120px]">{att.name}</span>
              </div>
            ))}
          </div>
        )}

        {/* Markdown content */}
        <div className="prose prose-invert max-w-none text-xs leading-relaxed break-words">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>

        {/* Citations if any */}
        {message.citations && message.citations.length > 0 && (
          <div className="pt-2 border-t border-slate-800/80 space-y-1">
            <span className="text-[10px] font-mono text-slate-400 block font-bold">Sources & Citations:</span>
            <div className="flex flex-wrap gap-1.5">
              {message.citations.map((c, i) => (
                <a
                  key={i}
                  href={c.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 text-[10px] text-indigo-400 hover:underline bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700"
                >
                  <ExternalLink className="w-2.5 h-2.5" />
                  <span className="truncate max-w-[150px]">{c.title}</span>
                </a>
              ))}
            </div>
          </div>
        )}

        {/* Retry prompt button */}
        {isUser && onRetry && (
          <div className="pt-1 flex justify-end opacity-0 group-hover:opacity-100 transition">
            <button
              type="button"
              onClick={() => onRetry(message.content)}
              className="flex items-center gap-1 text-[10px] font-mono text-indigo-200 hover:text-white bg-indigo-800/60 hover:bg-indigo-700 px-2 py-0.5 rounded cursor-pointer transition"
              title="Resubmit this prompt"
            >
              <RefreshCw className="w-3 h-3" /> Retry Prompt
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
