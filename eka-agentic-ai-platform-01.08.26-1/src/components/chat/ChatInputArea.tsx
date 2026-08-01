/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Send, Globe, Paperclip, Square } from 'lucide-react';
import { FileAttachment } from '../../types';

interface ChatInputAreaProps {
  onSendMessage: (text: string, search: boolean, attachments: FileAttachment[]) => void;
  onStopExecution: () => void;
  isExecuting: boolean;
  searchEnabled: boolean;
  setSearchEnabled: (s: boolean) => void;
}

export function ChatInputArea({
  onSendMessage,
  onStopExecution,
  isExecuting,
  searchEnabled,
  setSearchEnabled,
}: ChatInputAreaProps) {
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if ((!text.trim() && attachments.length === 0) || isExecuting) return;
    onSendMessage(text, searchEnabled, attachments);
    setText('');
    setAttachments([]);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();
      reader.onload = () => {
        const result = reader.result as string;
        const newAtt: FileAttachment = {
          id: `att-${Date.now()}-${Math.random()}`,
          name: file.name,
          size: file.size,
          type: file.type,
          content: file.type.startsWith('text/') || file.name.endsWith('.ts') || file.name.endsWith('.tsx') || file.name.endsWith('.js') || file.name.endsWith('.json') || file.name.endsWith('.md') ? result : undefined,
          dataUrl: result.startsWith('data:') ? result : undefined,
        };
        setAttachments(prev => [...prev, newAtt]);
      };
      if (file.type.startsWith('image/') || file.type.startsWith('audio/')) {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    }
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-slate-800 p-3 bg-slate-900/90 space-y-2">
      {/* File attachments badge strip */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pb-1">
          {attachments.map(att => (
            <div
              key={att.id}
              className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-300"
            >
              <Paperclip className="w-3 h-3 text-indigo-400" />
              <span className="truncate max-w-[120px]">{att.name}</span>
              <button
                type="button"
                onClick={() => setAttachments(prev => prev.filter(a => a.id !== att.id))}
                className="text-slate-400 hover:text-rose-400 ml-1 cursor-pointer font-bold"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Main Text Input & Actions */}
      <div className="relative flex items-center">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              handleSubmit(e);
            }
          }}
          placeholder="Ask a question or request a task..."
          className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-3 pr-24 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 resize-none min-h-[44px] max-h-32"
          rows={1}
        />

        <div className="absolute right-2 flex items-center gap-1">
          {/* Web Search Toggle */}
          <button
            type="button"
            onClick={() => setSearchEnabled(!searchEnabled)}
            className={`p-1.5 rounded-lg border transition cursor-pointer ${
              searchEnabled
                ? 'theme-accent-badge'
                : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
            title={searchEnabled ? 'Internet Assist / Web Search ENABLED' : 'Internet Assist / Web Search DISABLED'}
          >
            <Globe className="w-3.5 h-3.5" />
          </button>

          {/* Attach file button */}
          <label className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 hover:text-slate-200 transition cursor-pointer">
            <Paperclip className="w-3.5 h-3.5" />
            <input type="file" multiple onChange={handleFileUpload} className="hidden" />
          </label>

          {/* Send or Stop button */}
          {isExecuting ? (
            <button
              type="button"
              onClick={onStopExecution}
              className="p-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white transition cursor-pointer font-bold shadow"
              title="Stop execution immediately"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!text.trim() && attachments.length === 0}
              className="p-1.5 rounded-lg theme-accent-bg disabled:opacity-40 text-white transition cursor-pointer shadow"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </form>
  );
}
