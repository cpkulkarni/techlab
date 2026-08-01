/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { Layers, Sparkles, RefreshCw, Cpu, Code } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface WireframeToCodeTabProps {
  modelConfig?: ModelServerConfig;
  onRefreshWorkspace?: () => void;
}

export function WireframeToCodeTab({ modelConfig, onRefreshWorkspace }: WireframeToCodeTabProps) {
  const [specPrompt, setSpecPrompt] = useState('Create a full React component with Tailwind CSS for a high-performance Analytics Dashboard card grid.');
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleGenerate = async () => {
    if (!specPrompt.trim() || loading) return;
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: `Convert spec/diagram into clean React component code: "${specPrompt}"` }],
          customConfig: modelConfig,
          mode: 'code',
        }),
      });
      const data = await res.json();
      setGeneratedCode(data.reply || 'No code generated.');
      if (onRefreshWorkspace) onRefreshWorkspace();
    } catch (err: any) {
      setGeneratedCode(`Error generating code: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-indigo-400 flex items-center gap-1.5">
          <Layers className="w-4 h-4" /> Spec & Diagram-to-Code Synthesis Engine
        </span>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs font-mono text-slate-300 block mb-1">Architecture Diagram or UI Spec Description:</label>
          <textarea
            value={specPrompt}
            onChange={(e) => setSpecPrompt(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 h-28 resize-none"
          />
        </div>

        <button
          type="button"
          onClick={handleGenerate}
          disabled={!specPrompt.trim() || loading}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow-md"
        >
          {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
          {loading ? 'Synthesizing Code...' : 'Generate React Component Code'}
        </button>
      </div>

      {generatedCode && (
        <div className="border border-slate-800 rounded-xl bg-slate-900/80 p-4 space-y-2 max-h-96 overflow-y-auto">
          <div className="flex items-center gap-2 text-xs font-mono text-emerald-400 font-bold border-b border-slate-800 pb-2">
            <Code className="w-4 h-4" /> Generated Production Code:
          </div>
          <div className="prose prose-invert max-w-none text-xs leading-relaxed">
            <ReactMarkdown>{generatedCode}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
