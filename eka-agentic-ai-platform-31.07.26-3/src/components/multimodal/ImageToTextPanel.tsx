/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { FileSearch, Upload, RefreshCw, Cpu, CheckCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ImageToTextPanelProps {
  modelConfig?: ModelServerConfig;
}

export function ImageToTextPanel({ modelConfig }: ImageToTextPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('Extract all text, OCR content, and describe key visual elements in detail.');
  const [analysisText, setAnalysisText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || loading) return;
    setLoading(true);

    try {
      const reader = new FileReader();
      reader.onload = async () => {
        const dataUrl = reader.result as string;
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [{ role: 'user', content: prompt }],
            attachments: [{ name: selectedFile.name, type: selectedFile.type, size: selectedFile.size, dataUrl }],
            customConfig: modelConfig,
            mode: 'multimodal',
          }),
        });
        const data = await res.json();
        setAnalysisText(data.reply || 'No analysis returned.');
        setLoading(false);
      };
      reader.readAsDataURL(selectedFile);
    } catch (err: any) {
      setAnalysisText(`Error analyzing image: ${err.message}`);
      setLoading(false);
    }
  };

  return (
    <div className="p-4 space-y-4 max-w-3xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-cyan-400 flex items-center gap-1.5">
          <FileSearch className="w-4 h-4" /> Vision OCR & Image-to-Text Analytics Engine
        </span>
        <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* File Upload Box */}
        <div className="space-y-3">
          <label className="text-xs font-mono text-slate-300 block mb-1">Select Image File:</label>
          <div className="border-2 border-dashed border-slate-800 rounded-xl p-4 text-center bg-slate-900/40 hover:bg-slate-900/80 transition relative">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="absolute inset-0 opacity-0 cursor-pointer w-full h-full"
            />
            {previewUrl ? (
              <img src={previewUrl} alt="Preview" className="max-h-48 mx-auto rounded-lg shadow" />
            ) : (
              <div className="py-6 space-y-2 text-slate-400">
                <Upload className="w-8 h-8 mx-auto text-cyan-400" />
                <p className="text-xs font-mono">Click or drop image file here</p>
                <p className="text-[10px] text-slate-400 font-mono">PNG, JPG, WEBP, GIF up to 10MB</p>
              </div>
            )}
          </div>

          <div>
            <label className="text-xs font-mono text-slate-300 block mb-1">Analysis Prompt:</label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
            />
          </div>

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!selectedFile || loading}
            className="w-full bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-white text-xs font-mono font-bold py-2 rounded-lg transition flex items-center justify-center gap-2 cursor-pointer shadow-md"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <CheckCircle className="w-4 h-4" />}
            {loading ? 'Analyzing Vision Content...' : 'Run Vision Analysis'}
          </button>
        </div>

        {/* Output Area */}
        <div className="border border-slate-800 rounded-xl bg-slate-900/80 p-3.5 min-h-[220px] max-h-[400px] overflow-y-auto">
          <span className="text-[10px] font-mono text-slate-400 font-bold uppercase block mb-2">Vision Analysis Output:</span>
          {analysisText ? (
            <div className="prose prose-invert max-w-none text-xs leading-relaxed">
              <ReactMarkdown>{analysisText}</ReactMarkdown>
            </div>
          ) : (
            <div className="text-slate-400 text-xs font-mono pt-10 text-center">
              Upload an image and run analysis to view extracted text or visual insights.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
