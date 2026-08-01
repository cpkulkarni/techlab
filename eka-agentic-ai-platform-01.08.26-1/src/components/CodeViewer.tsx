/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { 
  FileCode, 
  Search, 
  Copy, 
  Check, 
  Download, 
  Wand2, 
  AlignLeft, 
  WrapText, 
  Hash, 
  X, 
  Sparkles,
  Bug,
  HelpCircle,
  FileCheck
} from 'lucide-react';
import { CodeEditorHeader } from './code_viewer/CodeEditorHeader';
import { ModelServerConfig } from '../types';

interface CodeViewerProps {
  selectedFile: string | null;
  selectedFileContent: string;
  onSaveFile: (path: string, content: string) => void;
  onCloseFile?: () => void;
  modelConfig?: ModelServerConfig;
}

export default function CodeViewer({
  selectedFile,
  selectedFileContent,
  onSaveFile,
  onCloseFile,
  modelConfig,
}: CodeViewerProps) {
  const [content, setContent] = useState(selectedFileContent);
  const [isSaving, setIsSaving] = useState(false);
  const [copied, setCopied] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [replaceTerm, setReplaceTerm] = useState('');
  const [wordWrap, setWordWrap] = useState(false);
  const [showLineNumbers, setShowLineNumbers] = useState(true);
  const [cursorPos, setCursorPos] = useState({ line: 1, col: 1 });
  const [aiNotice, setAiNotice] = useState<string | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setContent(selectedFileContent);
  }, [selectedFileContent, selectedFile]);

  const isDirty = content !== selectedFileContent;

  const handleSave = async () => {
    if (!selectedFile || !isDirty) return;
    setIsSaving(true);
    await onSaveFile(selectedFile, content);
    setIsSaving(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    if (!selectedFile) return;
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile.split('/').pop() || 'file.txt';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleFormatCode = () => {
    try {
      if (selectedFile?.endsWith('.json')) {
        const formatted = JSON.stringify(JSON.parse(content), null, 2);
        setContent(formatted);
      } else {
        // Basic code indentation cleanup
        const lines = content.split('\n');
        const cleaned = lines.map(line => line.trimEnd()).join('\n');
        setContent(cleaned);
      }
      setAiNotice('Code formatted successfully');
      setTimeout(() => setAiNotice(null), 2500);
    } catch {
      setAiNotice('Formatting failed (syntax error)');
      setTimeout(() => setAiNotice(null), 2500);
    }
  };

  const handleReplaceAll = () => {
    if (!searchTerm) return;
    const newContent = content.replaceAll(searchTerm, replaceTerm);
    setContent(newContent);
    setAiNotice(`Replaced occurrences of "${searchTerm}"`);
    setTimeout(() => setAiNotice(null), 2500);
  };

  const handleAiAction = async (action: string) => {
    const fileName = selectedFile?.split('/').pop() || 'file';
    const activeModel = modelConfig?.selectedModel || 'selected model';
    setAiNotice(`[Model: ${activeModel}] Analyzing ${fileName} for ${action}...`);

    let prompt = '';
    if (action === 'Fix Bugs') {
      prompt = `Analyze the following code for syntax bugs, edge cases, or runtime errors, and explain fixes:\n\`\`\`\n${content}\n\`\`\``;
    } else if (action === 'Explain Code') {
      prompt = `Provide a high-level summary and breakdown of what this code file does:\n\`\`\`\n${content}\n\`\`\``;
    } else if (action === 'Add Comments') {
      prompt = `Add clean JSDoc documentation header and inline comments to key functions in this code file:\n\`\`\`\n${content}\n\`\`\``;
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: prompt }],
          customConfig: modelConfig,
          mode: 'code',
        }),
      });
      const data = await res.json();
      if (data.reply) {
        setAiNotice(`[${activeModel}]: ${data.reply.slice(0, 140)}...`);
      } else {
        setAiNotice(`Analysis completed using model ${activeModel}`);
      }
    } catch (err: any) {
      setAiNotice(`Error using selected model (${activeModel}): ${err.message}`);
    }
    setTimeout(() => setAiNotice(null), 6000);
  };

  const updateCursorPosition = () => {
    if (!textareaRef.current) return;
    const selStart = textareaRef.current.selectionStart;
    const textBefore = content.substring(0, selStart);
    const lines = textBefore.split('\n');
    const currentLine = lines.length;
    const currentCol = lines[lines.length - 1].length + 1;
    setCursorPos({ line: currentLine, col: currentCol });
  };

  const lines = content.split('\n');
  const lineCount = lines.length;
  const matchCount = searchTerm ? (content.match(new RegExp(searchTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g')) || []).length : 0;
  const fileExt = selectedFile ? selectedFile.split('.').pop()?.toUpperCase() || 'TXT' : 'TXT';

  if (!selectedFile) {
    return (
      <div className="h-full bg-slate-950 flex flex-col items-center justify-center text-slate-400 p-6 text-center space-y-3">
        <div className="w-12 h-12 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center text-emerald-400">
          <FileCode className="w-6 h-6" />
        </div>
        <div className="space-y-1">
          <p className="text-xs font-mono font-bold text-slate-300">No File Open</p>
          <p className="text-[11px] text-slate-400 max-w-xs font-mono">
            Double-click or select a file from the workspace explorer to view and edit its code.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-slate-950 flex flex-col overflow-hidden font-mono text-xs">
      {/* Editor Header */}
      <CodeEditorHeader
        selectedFile={selectedFile}
        isDirty={isDirty}
        isSaving={isSaving}
        onSave={handleSave}
        onClose={onCloseFile}
      />

      {/* Code Editor Toolbar */}
      <div className="h-8 bg-slate-900 border-b border-slate-800 px-3 flex items-center justify-between shrink-0 text-[11px] text-slate-400 select-none">
        <div className="flex items-center gap-2 overflow-x-auto scrollbar-none py-1">
          <button
            type="button"
            onClick={() => setShowSearch(!showSearch)}
            className={`px-2 py-0.5 rounded flex items-center gap-1 transition cursor-pointer ${
              showSearch ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/50' : 'hover:bg-slate-800 hover:text-slate-200'
            }`}
            title="Search & Replace (Ctrl+F)"
          >
            <Search className="w-3 h-3" />
            <span>Find</span>
          </button>

          <button
            type="button"
            onClick={handleFormatCode}
            className="px-2 py-0.5 rounded flex items-center gap-1 hover:bg-slate-800 hover:text-slate-200 transition cursor-pointer"
            title="Format Code Indentation"
          >
            <AlignLeft className="w-3 h-3 text-emerald-400" />
            <span>Format</span>
          </button>

          <button
            type="button"
            onClick={() => setWordWrap(!wordWrap)}
            className={`px-2 py-0.5 rounded flex items-center gap-1 transition cursor-pointer ${
              wordWrap ? 'bg-indigo-600/30 text-indigo-300' : 'hover:bg-slate-800 hover:text-slate-200'
            }`}
            title="Toggle Word Wrap"
          >
            <WrapText className="w-3 h-3" />
            <span>Wrap</span>
          </button>

          <button
            type="button"
            onClick={() => setShowLineNumbers(!showLineNumbers)}
            className={`px-2 py-0.5 rounded flex items-center gap-1 transition cursor-pointer ${
              showLineNumbers ? 'bg-indigo-600/30 text-indigo-300' : 'hover:bg-slate-800 hover:text-slate-200'
            }`}
            title="Toggle Line Numbers"
          >
            <Hash className="w-3 h-3" />
            <span>Lines</span>
          </button>

          <span className="w-px h-3 bg-slate-800 my-auto" />

          {/* AI Helper Quick Actions */}
          <button
            type="button"
            onClick={() => handleAiAction('Fix Bugs')}
            className="px-2 py-0.5 rounded flex items-center gap-1 text-rose-300 hover:bg-rose-950/40 transition cursor-pointer"
            title="AI Diagnostics Check"
          >
            <Bug className="w-3 h-3 text-rose-400" />
            <span>AI Fix</span>
          </button>

          <button
            type="button"
            onClick={() => handleAiAction('Explain Code')}
            className="px-2 py-0.5 rounded flex items-center gap-1 text-amber-300 hover:bg-amber-950/40 transition cursor-pointer"
            title="AI Code Explanation"
          >
            <HelpCircle className="w-3 h-3 text-amber-400" />
            <span>Explain</span>
          </button>

          <button
            type="button"
            onClick={() => handleAiAction('Add Comments')}
            className="px-2 py-0.5 rounded flex items-center gap-1 text-purple-300 hover:bg-purple-950/40 transition cursor-pointer"
            title="AI JSDoc Comment Generator"
          >
            <Sparkles className="w-3 h-3 text-purple-400" />
            <span>DocGen</span>
          </button>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            type="button"
            onClick={handleCopy}
            className="px-2 py-0.5 rounded flex items-center gap-1 hover:bg-slate-800 text-slate-300 hover:text-white transition cursor-pointer"
            title="Copy all code to clipboard"
          >
            {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          <button
            type="button"
            onClick={handleDownload}
            className="px-2 py-0.5 rounded flex items-center gap-1 hover:bg-slate-800 text-slate-300 hover:text-white transition cursor-pointer"
            title="Download file"
          >
            <Download className="w-3 h-3 text-sky-400" />
            <span>Download</span>
          </button>
        </div>
      </div>

      {/* AI Notice Toast */}
      {aiNotice && (
        <div className="bg-indigo-950/90 border-b border-indigo-800/80 px-3 py-1 text-[11px] text-indigo-200 flex items-center gap-2 shrink-0 animate-fadeIn">
          <Wand2 className="w-3.5 h-3.5 text-indigo-400 animate-spin" />
          <span>{aiNotice}</span>
        </div>
      )}

      {/* Find & Replace Bar */}
      {showSearch && (
        <div className="bg-slate-900 border-b border-slate-800 p-2 flex flex-wrap items-center gap-2 shrink-0 text-[11px]">
          <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded px-2 py-1">
            <Search className="w-3 h-3 text-slate-500" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Find..."
              className="bg-transparent border-none text-slate-200 focus:outline-none w-28 text-xs"
            />
            {searchTerm && (
              <span className="text-[10px] text-indigo-400 font-bold">{matchCount} matches</span>
            )}
          </div>

          <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 rounded px-2 py-1">
            <input
              type="text"
              value={replaceTerm}
              onChange={(e) => setReplaceTerm(e.target.value)}
              placeholder="Replace with..."
              className="bg-transparent border-none text-slate-200 focus:outline-none w-28 text-xs"
            />
          </div>

          <button
            type="button"
            onClick={handleReplaceAll}
            disabled={!searchTerm}
            className="px-2 py-1 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white rounded font-bold cursor-pointer transition"
          >
            Replace All
          </button>

          <button
            type="button"
            onClick={() => setShowSearch(false)}
            className="p-1 text-slate-400 hover:text-white rounded ml-auto"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* Code Editor Body with Line Numbers */}
      <div className="flex-1 flex relative overflow-hidden bg-slate-950">
        {/* Line Numbers Gutter */}
        {showLineNumbers && (
          <div className="w-11 bg-slate-900/60 border-r border-slate-800 py-4 text-right pr-2 text-slate-600 select-none font-mono text-xs overflow-hidden shrink-0">
            {Array.from({ length: lineCount }).map((_, idx) => (
              <div 
                key={idx} 
                className={`leading-relaxed ${cursorPos.line === idx + 1 ? 'text-indigo-400 font-bold' : ''}`}
              >
                {idx + 1}
              </div>
            ))}
          </div>
        )}

        {/* Text Area */}
        <textarea
          ref={textareaRef}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onClick={updateCursorPosition}
          onKeyUp={updateCursorPosition}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
              e.preventDefault();
              handleSave();
            } else if ((e.ctrlKey || e.metaKey) && e.key === 'f') {
              e.preventDefault();
              setShowSearch(true);
            }
          }}
          className={`flex-1 p-4 bg-slate-950 font-mono text-xs text-slate-200 focus:outline-none resize-none leading-relaxed border-none ${
            wordWrap ? 'whitespace-pre-wrap break-words' : 'whitespace-pre overflow-x-auto'
          }`}
          spellCheck={false}
        />
      </div>

      {/* Editor Status Bar */}
      <div className="h-6 bg-slate-900 border-t border-slate-800 px-3 flex items-center justify-between shrink-0 text-[10px] text-slate-500 font-mono select-none">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1 text-indigo-400">
            <FileCheck className="w-3 h-3" />
            <span>Ln {cursorPos.line}, Col {cursorPos.col}</span>
          </span>
          <span>|</span>
          <span>{lineCount} Lines</span>
          <span>|</span>
          <span>{content.length} Chars ({((content.length)/1024).toFixed(1)} KB)</span>
        </div>

        <div className="flex items-center gap-3">
          <span className="bg-slate-800 text-slate-300 px-1.5 py-0.5 rounded font-bold uppercase">{fileExt}</span>
          <span>UTF-8</span>
          <span>Spaces: 2</span>
        </div>
      </div>
    </div>
  );
}
