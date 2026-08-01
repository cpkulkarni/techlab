/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { Globe, ArrowLeftRight, Sparkles, RefreshCw, Cpu, Copy, Check } from 'lucide-react';

interface TranslationPanelProps {
  modelConfig?: ModelServerConfig;
}

const LANGUAGES = [
  { code: 'auto', name: 'Auto-Detect' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'hi', name: 'Hindi' },
  { code: 'mr', name: 'Marathi' },
  { code: 'gu', name: 'Gujarati' },
  { code: 'bn', name: 'Bengali' },
  { code: 'ta', name: 'Tamil' },
  { code: 'te', name: 'Telugu' },
  { code: 'ar', name: 'Arabic' },
  { code: 'ru', name: 'Russian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'it', name: 'Italian' },
  { code: 'ko', name: 'Korean' },
  { code: 'nl', name: 'Dutch' },
  { code: 'tr', name: 'Turkish' },
];

export function TranslationPanel({ modelConfig }: TranslationPanelProps) {
  const [sourceLang, setSourceLang] = useState('auto');
  const [targetLang, setTargetLang] = useState('en');
  const [inputText, setInputText] = useState('');
  const [translatedText, setTranslatedText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleSwapLanguages = () => {
    if (sourceLang === 'auto') {
      setSourceLang(targetLang);
      setTargetLang('en');
    } else {
      const temp = sourceLang;
      setSourceLang(targetLang);
      setTargetLang(temp);
    }
  };

  const handleTranslate = async () => {
    if (!inputText.trim() || loading) return;
    setLoading(true);
    setTranslatedText(null);

    const sourceLabel = LANGUAGES.find(l => l.code === sourceLang)?.name || sourceLang;
    const targetLabel = LANGUAGES.find(l => l.code === targetLang)?.name || targetLang;

    const promptText = `Translate the following text from ${sourceLabel} to ${targetLabel}:\n\n"${inputText}"\n\nProvide only the accurate, natural translation without additional commentary.`;

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: promptText }],
          customConfig: modelConfig,
          mode: 'multimodal',
        }),
      });
      const data = await res.json();
      setTranslatedText(data.reply || 'Translation failed.');
    } catch (err: any) {
      setTranslatedText(`Error during translation: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!translatedText) return;
    navigator.clipboard.writeText(translatedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-4 space-y-4 max-w-2xl mx-auto font-sans">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-slate-200 flex items-center gap-1.5">
          <Globe className="w-4 h-4 text-teal-400" />
          Neural Language Translator Studio
        </span>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/60 border border-indigo-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      {/* Language Selector Toolbar */}
      <div className="flex items-center gap-2 bg-slate-900 p-2 rounded-xl border border-slate-800 text-xs font-mono">
        {/* Source Language */}
        <div className="flex-1 space-y-1">
          <label className="text-[10px] text-slate-400 font-bold block uppercase tracking-wider">Source Language:</label>
          <select
            value={sourceLang}
            onChange={(e) => setSourceLang(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-500 font-medium"
          >
            {LANGUAGES.map(lang => (
              <option key={`src-${lang.code}`} value={lang.code}>{lang.name}</option>
            ))}
          </select>
        </div>

        {/* Swap Button */}
        <div className="pt-4 shrink-0">
          <button
            type="button"
            onClick={handleSwapLanguages}
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition cursor-pointer"
            title="Swap Source & Target Languages"
          >
            <ArrowLeftRight className="w-4 h-4 text-teal-400" />
          </button>
        </div>

        {/* Target Language */}
        <div className="flex-1 space-y-1">
          <label className="text-[10px] text-slate-400 font-bold block uppercase tracking-wider">Target Language:</label>
          <select
            value={targetLang}
            onChange={(e) => setTargetLang(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-500 font-medium"
          >
            {LANGUAGES.filter(l => l.code !== 'auto').map(lang => (
              <option key={`tgt-${lang.code}`} value={lang.code}>{lang.name}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Input Text Area */}
      <div className="space-y-1.5">
        <label className="text-xs font-mono text-slate-300 block">
          Enter Text to Translate:
        </label>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Type or paste sentences to translate..."
          className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 h-32 resize-none leading-relaxed"
        />
      </div>

      {/* Translate Action Button */}
      <button
        type="button"
        onClick={handleTranslate}
        disabled={!inputText.trim() || loading}
        className="w-full bg-teal-600 hover:bg-teal-500 disabled:opacity-40 text-white text-xs font-mono font-bold py-2 rounded-xl transition flex items-center justify-center gap-2 cursor-pointer shadow"
      >
        {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
        {loading ? 'Translating Content...' : 'Translate Text'}
      </button>

      {/* Translated Result Output Box */}
      {translatedText && (
        <div className="p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-2 font-mono text-xs text-slate-200 animate-fadeIn">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="text-[10px] font-bold uppercase text-teal-400 tracking-wider">
              Translation Result ({LANGUAGES.find(l => l.code === targetLang)?.name}):
            </span>
            <button
              type="button"
              onClick={handleCopy}
              className="text-slate-400 hover:text-white flex items-center gap-1 text-[10px] cursor-pointer"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? 'Copied' : 'Copy'}</span>
            </button>
          </div>
          <p className="whitespace-pre-wrap leading-relaxed font-sans text-sm">{translatedText}</p>
        </div>
      )}
    </div>
  );
}
