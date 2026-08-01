/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { Wand2, Download, RefreshCw, Cpu, Image as ImageIcon } from 'lucide-react';

interface TextToImagePanelProps {
  modelConfig?: ModelServerConfig;
}

export function TextToImagePanel({ modelConfig }: TextToImagePanelProps) {
  const [prompt, setPrompt] = useState('');
  const [aspectRatio, setAspectRatio] = useState('1:1');
  const [generatedUrl, setGeneratedUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleGenerate = async () => {
    if (!prompt.trim() || loading) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: `Generate image concept description and SVG asset for prompt: "${prompt}". Aspect ratio: ${aspectRatio}` }],
          customConfig: modelConfig,
          mode: 'multimodal',
        }),
      });
      const data = await res.json();
      if (data.reply) {
        // Fallback placeholder/SVG simulation URL
        const svgContent = encodeURIComponent(
          `<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600" viewBox="0 0 600 600"><rect width="100%" height="100%" fill="#0f172a"/><text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" fill="#818cf8" font-family="monospace" font-size="20">AI Generated Asset</text><text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" fill="#94a3b8" font-family="sans-serif" font-size="14">${prompt.slice(0, 40)}...</text></svg>`
        );
        setGeneratedUrl(`data:image/svg+xml;utf8,${svgContent}`);
      }
    } catch (err: any) {
      setError(`Image generation failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 space-y-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-indigo-400 flex items-center gap-1.5">
          <Wand2 className="w-4 h-4" /> Generative Text-to-Image Studio
        </span>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <label className="text-xs font-mono text-slate-300 block mb-1">Image Prompt Description:</label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe the image asset to generate..."
            className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 h-24 resize-none"
          />
        </div>

        <div className="flex items-center gap-3">
          <div className="flex-1">
            <label className="text-xs font-mono text-slate-300 block mb-1">Aspect Ratio:</label>
            <select
              value={aspectRatio}
              onChange={(e) => setAspectRatio(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200"
            >
              <option value="1:1">1:1 Square</option>
              <option value="16:9">16:9 Widescreen</option>
              <option value="4:3">4:3 Standard</option>
              <option value="9:16">9:16 Vertical Story</option>
            </select>
          </div>

          <button
            type="button"
            onClick={handleGenerate}
            disabled={!prompt.trim() || loading}
            className="mt-5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow-md"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
            {loading ? 'Generating...' : 'Synthesize Image'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-rose-950/80 border border-rose-800 rounded-xl text-rose-300 text-xs font-mono">
          {error}
        </div>
      )}

      {/* Output Display */}
      <div className="border border-slate-800 rounded-2xl bg-slate-900/60 p-4 flex flex-col items-center justify-center min-h-[250px]">
        {generatedUrl ? (
          <div className="space-y-3 flex flex-col items-center">
            <img src={generatedUrl} alt="Generated asset" className="max-h-80 rounded-xl shadow-xl border border-slate-800" />
            <a
              href={generatedUrl}
              download="ai-generated-asset.svg"
              className="flex items-center gap-1.5 text-xs font-mono text-indigo-400 hover:underline bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700"
            >
              <Download className="w-3.5 h-3.5" /> Download SVG Asset
            </a>
          </div>
        ) : (
          <div className="text-center text-slate-400 space-y-2">
            <ImageIcon className="w-8 h-8 mx-auto text-slate-400" />
            <p className="text-xs font-mono">No image synthesized yet. Enter a prompt above!</p>
          </div>
        )}
      </div>
    </div>
  );
}
