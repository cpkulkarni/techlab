/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { Video, Box, Globe, Cpu, RefreshCw, Sparkles } from 'lucide-react';

interface Video3DPanelProps {
  modelConfig?: ModelServerConfig;
  feature: 'text_to_video' | 'text_to_3d' | 'translation';
}

export function Video3DPanel({ modelConfig, feature }: Video3DPanelProps) {
  const [prompt, setPrompt] = useState('');
  const [resultText, setResultText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleRun = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: `Perform ${feature} generation for prompt: "${prompt}"` }],
          customConfig: modelConfig,
          mode: 'multimodal',
        }),
      });
      const data = await res.json();
      setResultText(data.reply || 'Process finished.');
    } catch (err: any) {
      setResultText(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const titles = {
    text_to_video: { label: 'Generative Text-to-Video Engine', icon: <Video className="w-4 h-4 text-violet-400" /> },
    text_to_3d: { label: 'Text-to-3D Mesh Asset Generator', icon: <Box className="w-4 h-4 text-amber-500" /> },
    translation: { label: 'Multilingual Neural Translation', icon: <Globe className="w-4 h-4 text-teal-400" /> },
  };

  const currentInfo = titles[feature];

  return (
    <div className="p-4 space-y-4 max-w-2xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-slate-200 flex items-center gap-1.5">
          {currentInfo.icon}
          {currentInfo.label}
        </span>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs font-mono text-slate-300 block mb-1">
            {feature === 'translation' ? 'Text to translate:' : 'Synthesis Prompt:'}
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder={`Enter input for ${feature}...`}
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 h-28 resize-none"
          />
        </div>

        <button
          type="button"
          onClick={handleRun}
          disabled={!prompt.trim() || loading}
          className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow"
        >
          {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
          {loading ? 'Processing Multimodal Request...' : 'Synthesize Output'}
        </button>
      </div>

      {resultText && (
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl text-xs font-mono text-slate-200">
          <span className="text-[10px] font-bold text-slate-400 block mb-1">Result:</span>
          <p className="whitespace-pre-wrap">{resultText}</p>
        </div>
      )}
    </div>
  );
}
