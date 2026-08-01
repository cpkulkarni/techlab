/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  FileText, 
  Save, 
  Plus, 
  Eye, 
  Edit3, 
  Columns, 
  Sparkles, 
  Download, 
  Check, 
  Wand2, 
  FolderOpen,
  FileCode
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { FileNode } from '../../types';

interface DocsStudioTabProps {
  selectedFile?: string | null;
  selectedFileContent?: string;
  onSaveFile?: (path: string, content: string) => void;
  files?: FileNode[];
  onSelectFile?: (path: string) => void;
}

export function DocsStudioTab({
  selectedFile,
  selectedFileContent = '',
  onSaveFile,
  files = [],
  onSelectFile,
}: DocsStudioTabProps) {
  const [activeDocPath, setActiveDocPath] = useState<string>(selectedFile || 'README.md');
  const [docContent, setDocContent] = useState<string>(
    selectedFileContent ||
    '# Technical Documentation & Architecture Guide\n\nWelcome to Eka Studio. You can edit Markdown docs, view system specs, and generate API architecture guidelines.\n\n## Core System Capabilities\n- **Full-Stack IDE**: Real-time TypeScript & React workspace.\n- **Agent Execution**: Step-by-step reasoning and automated file creation.\n- **Multi-Agent Topology**: Node graphs, A2A messaging, and MCP context tools.'
  );

  const [viewMode, setViewMode] = useState<'split' | 'edit' | 'preview'>('split');
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [aiNotice, setAiNotice] = useState<string | null>(null);

  useEffect(() => {
    if (selectedFile) {
      setActiveDocPath(selectedFile);
      setDocContent(selectedFileContent);
    }
  }, [selectedFile, selectedFileContent]);

  // Recursively collect all documentation files (.md, .txt, .doc) from workspace
  const getDocFiles = (nodes: FileNode[]): { name: string; path: string }[] => {
    const list: { name: string; path: string }[] = [];
    const traverse = (items: FileNode[]) => {
      for (const item of items) {
        if (item.type === 'file' && (item.name.endsWith('.md') || item.name.endsWith('.txt') || item.name.endsWith('.doc'))) {
          list.push({ name: item.name, path: item.path });
        }
        if (item.children) traverse(item.children);
      }
    };
    traverse(nodes);
    return list;
  };

  const docFiles = getDocFiles(files);
  const isDirty = docContent !== selectedFileContent;

  const handleSave = async () => {
    if (onSaveFile) {
      setIsSaving(true);
      await onSaveFile(activeDocPath, docContent);
      setIsSaving(false);
      setSaveMessage('Document saved successfully!');
      setTimeout(() => setSaveMessage(null), 2500);
    }
  };

  const handleAiAction = (action: string) => {
    setAiNotice(`AI Assistant generating ${action}...`);
    setTimeout(() => {
      if (action === 'API Spec') {
        setDocContent(prev => prev + '\n\n## Generated API Specification\n\n`GET /api/workspace` - Fetch file tree structure\n`POST /api/chat` - Stream agent reasoning & execution');
      } else if (action === 'Architecture Overview') {
        setDocContent(prev => prev + '\n\n## System Architecture\n\n- **Frontend**: React 18 + Vite + Tailwind CSS + Lucide Icons\n- **Backend**: Express Node.js Server on port 3000\n- **Agent Engine**: Antigravity Gemini AI Agent');
      }
      setAiNotice(null);
    }, 1200);
  };

  return (
    <div className="h-full bg-slate-950 flex flex-col font-mono text-xs overflow-hidden">
      {/* Docs Studio Header Bar */}
      <div className="h-10 bg-slate-900 border-b border-slate-800 px-3 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <FileText className="w-4 h-4 text-yellow-400 shrink-0" />
          <span className="font-bold text-slate-200 text-xs shrink-0">Docs Studio</span>
          
          {/* Document File Picker Dropdown */}
          <div className="relative">
            <select
              value={activeDocPath}
              onChange={(e) => {
                const newPath = e.target.value;
                setActiveDocPath(newPath);
                if (onSelectFile) onSelectFile(newPath);
              }}
              className="bg-slate-950 border border-slate-800 text-yellow-400 font-bold text-[11px] rounded px-2 py-1 focus:outline-none focus:border-yellow-500 max-w-[180px] truncate cursor-pointer"
            >
              <option value="README.md">README.md</option>
              <option value="ARCHITECTURE.md">ARCHITECTURE.md</option>
              {docFiles.map(df => (
                <option key={df.path} value={df.path}>{df.name}</option>
              ))}
            </select>
          </div>

          {isDirty && <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" title="Unsaved changes" />}
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 shrink-0">
          {/* View Mode Switches */}
          <div className="bg-slate-950 p-0.5 rounded border border-slate-800 flex items-center gap-1">
            <button
              type="button"
              onClick={() => setViewMode('edit')}
              className={`p-1 rounded transition cursor-pointer ${
                viewMode === 'edit' ? 'theme-accent-bg text-white font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Editor View Only"
            >
              <Edit3 className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={() => setViewMode('split')}
              className={`p-1 rounded transition cursor-pointer ${
                viewMode === 'split' ? 'theme-accent-bg text-white font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Split View (Editor + Live Preview)"
            >
              <Columns className="w-3.5 h-3.5" />
            </button>

            <button
              type="button"
              onClick={() => setViewMode('preview')}
              className={`p-1 rounded transition cursor-pointer ${
                viewMode === 'preview' ? 'theme-accent-bg text-white font-bold' : 'text-slate-400 hover:text-slate-200'
              }`}
              title="Markdown Preview Only"
            >
              <Eye className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* AI Generation Actions */}
          <button
            type="button"
            onClick={() => handleAiAction('Architecture Overview')}
            className="px-2 py-1 bg-purple-950/80 hover:bg-purple-900 border border-purple-800 text-purple-300 rounded text-[10px] font-bold flex items-center gap-1 cursor-pointer transition"
          >
            <Sparkles className="w-3 h-3 text-purple-400" />
            <span>AI Gen Specs</span>
          </button>

          {/* Save Button */}
          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving}
            className="px-3 py-1 bg-yellow-600 hover:bg-yellow-500 disabled:opacity-50 text-slate-950 font-bold rounded flex items-center gap-1 cursor-pointer transition shadow"
          >
            {isSaving ? <Check className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
            <span>{isSaving ? 'Saving...' : 'Save Doc'}</span>
          </button>
        </div>
      </div>

      {/* Save Message / AI Toast */}
      {(saveMessage || aiNotice) && (
        <div className="bg-yellow-950/80 border-b border-yellow-800/80 px-3 py-1 text-[11px] text-yellow-200 flex items-center gap-2 shrink-0 animate-fadeIn">
          <Wand2 className="w-3.5 h-3.5 text-yellow-400 animate-spin" />
          <span>{saveMessage || aiNotice}</span>
        </div>
      )}

      {/* Main Viewport */}
      <div className="flex-1 flex overflow-hidden">
        {/* Editor Pane */}
        {(viewMode === 'edit' || viewMode === 'split') && (
          <div className={`h-full flex flex-col bg-slate-950 border-r border-slate-800 ${
            viewMode === 'split' ? 'w-1/2' : 'w-full'
          }`}>
            <div className="px-3 py-1.5 bg-slate-900/60 border-b border-slate-800/80 text-[10px] uppercase text-slate-400 font-bold flex items-center justify-between shrink-0">
              <span>Markdown Source Editor</span>
              <span>{docContent.length} chars</span>
            </div>
            <textarea
              value={docContent}
              onChange={(e) => setDocContent(e.target.value)}
              placeholder="Write Markdown document here..."
              className="flex-1 p-4 bg-slate-950 text-slate-200 font-mono text-xs focus:outline-none resize-none leading-relaxed"
            />
          </div>
        )}

        {/* Live Rendered Markdown Preview Pane */}
        {(viewMode === 'preview' || viewMode === 'split') && (
          <div className={`h-full overflow-y-auto p-4 bg-slate-900/50 ${
            viewMode === 'split' ? 'w-1/2' : 'w-full'
          }`}>
            <div className="px-2 py-1 mb-3 text-[10px] uppercase font-bold text-yellow-400 border-b border-slate-800 flex items-center gap-1.5">
              <Eye className="w-3 h-3" /> Live Formatted Markdown Preview
            </div>
            <div className="prose max-w-none text-xs leading-relaxed border border-slate-800 rounded-xl bg-slate-950/80 p-4 shadow-inner">
              <ReactMarkdown>{docContent}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
