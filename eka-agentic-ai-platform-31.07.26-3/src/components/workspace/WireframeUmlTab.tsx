/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { ModelServerConfig, FileNode } from '../../types';
import { 
  Layers, 
  Upload, 
  FolderPlus, 
  FileText, 
  Image as ImageIcon, 
  FileCode, 
  Sparkles, 
  RefreshCw, 
  Cpu, 
  Code, 
  Eye, 
  Trash2, 
  Download, 
  Save, 
  HardDrive, 
  CheckCircle2, 
  Box, 
  Workflow, 
  FileCheck
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface WireframeUmlTabProps {
  modelConfig?: ModelServerConfig;
  files?: FileNode[];
  onRefreshWorkspace?: () => void;
}

export function WireframeUmlTab({ modelConfig, files = [], onRefreshWorkspace }: WireframeUmlTabProps) {
  const [selectedFilePath, setSelectedFilePath] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [isImageFile, setIsImageFile] = useState<boolean>(false);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [specType, setSpecType] = useState<'mermaid' | 'plantuml' | 'json_wireframe' | 'ascii' | 'custom'>('mermaid');
  
  const [diagramTitle, setDiagramTitle] = useState('architecture_diagram.mermaid');
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [loadingAi, setLoadingAi] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Dedicated directory for uploaded diagrams & wireframes
  const UPLOAD_DIR = 'uploaded_diagrams';

  // Extract diagram and wireframe files from workspace tree
  const flattenFiles = (nodes: FileNode[]): FileNode[] => {
    let result: FileNode[] = [];
    for (const node of nodes) {
      if (node.type === 'file') {
        result.push(node);
      }
      if (node.children) {
        result = result.concat(flattenFiles(node.children));
      }
    }
    return result;
  };

  const allWorkspaceFiles = flattenFiles(files);
  const diagramFiles = allWorkspaceFiles.filter(f => {
    const name = f.name.toLowerCase();
    return (
      f.path.startsWith(UPLOAD_DIR) ||
      name.endsWith('.mermaid') ||
      name.endsWith('.puml') ||
      name.endsWith('.plantuml') ||
      name.endsWith('.drawio') ||
      name.endsWith('.uml') ||
      name.endsWith('.spec') ||
      name.endsWith('.png') ||
      name.endsWith('.jpg') ||
      name.endsWith('.jpeg') ||
      name.endsWith('.svg') ||
      name.endsWith('.json')
    );
  });

  // Default initial diagram spec if empty
  const defaultMermaid = `graph TD
    User([End User]) --> API[API Gateway]
    API --> Auth[Auth Service]
    API --> DB[(Cloud Database)]
    API --> AI[AI Agent Engine]`;

  useEffect(() => {
    if (!selectedFilePath && diagramFiles.length > 0) {
      handleSelectFile(diagramFiles[0].path);
    } else if (!selectedFilePath) {
      setFileContent(defaultMermaid);
    }
  }, [files]);

  const handleSelectFile = async (path: string) => {
    setSelectedFilePath(path);
    setAiAnalysis(null);
    const lower = path.toLowerCase();
    
    if (lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg') || lower.endsWith('.svg')) {
      setIsImageFile(true);
      setImageUrl(`/api/workspace/file?path=${encodeURIComponent(path)}`);
      setFileContent(`[Image File: ${path}]`);
    } else {
      setIsImageFile(false);
      setImageUrl(null);
      try {
        const res = await fetch(`/api/workspace/file?path=${encodeURIComponent(path)}`);
        const data = await res.json();
        if (data.success) {
          setFileContent(data.content || '');
          if (path.endsWith('.mermaid')) setSpecType('mermaid');
          else if (path.endsWith('.puml') || path.endsWith('.plantuml')) setSpecType('plantuml');
          else if (path.endsWith('.json')) setSpecType('json_wireframe');
        }
      } catch (err) {
        setFileContent('// Error loading file content');
      }
    }
  };

  const handleSaveDiagram = async () => {
    const filename = selectedFilePath || `${UPLOAD_DIR}/${diagramTitle}`;
    setIsSaving(true);
    setUploadStatus(null);
    try {
      // Ensure uploaded_diagrams directory exists
      await fetch('/api/workspace/folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: UPLOAD_DIR }),
      }).catch(() => {});

      const res = await fetch('/api/workspace/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: filename, content: fileContent }),
      });
      const data = await res.json();
      if (data.success) {
        setUploadStatus(`Saved successfully to workspace: ${filename}`);
        if (onRefreshWorkspace) onRefreshWorkspace();
        setSelectedFilePath(filename);
      }
    } catch (err: any) {
      setUploadStatus(`Error saving: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = e.target.files;
    if (!uploadedFiles || uploadedFiles.length === 0) return;

    setUploadStatus('Uploading file to workspace directory...');

    try {
      // Ensure uploaded_diagrams folder exists
      await fetch('/api/workspace/folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: UPLOAD_DIR }),
      }).catch(() => {});

      for (let i = 0; i < uploadedFiles.length; i++) {
        const file = uploadedFiles[i];
        const destPath = `${UPLOAD_DIR}/${file.name}`;

        if (file.type.startsWith('image/')) {
          const reader = new FileReader();
          reader.onload = async () => {
            const base64Content = reader.result as string;
            await fetch('/api/workspace/file', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ path: destPath, content: base64Content }),
            });
            if (onRefreshWorkspace) onRefreshWorkspace();
            handleSelectFile(destPath);
          };
          reader.readAsDataURL(file);
        } else {
          const text = await file.text();
          await fetch('/api/workspace/file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: destPath, content: text }),
          });
          if (onRefreshWorkspace) onRefreshWorkspace();
          handleSelectFile(destPath);
        }
      }
      setUploadStatus('File(s) uploaded successfully!');
    } catch (err: any) {
      setUploadStatus(`Upload error: ${err.message}`);
    }
  };

  const handleAiAction = async (action: 'analyze' | 'generate_doc' | 'compile_code') => {
    if (!fileContent.trim() || loadingAi) return;
    setLoadingAi(true);
    setAiAnalysis(null);

    let prompt = '';
    if (action === 'analyze') {
      prompt = `Analyze this wireframe / UML diagram spec in detail. Explain the architecture, data flow, component hierarchy, and potential edge cases:\n\n\`\`\`${fileContent}\`\`\``;
    } else if (action === 'generate_doc') {
      prompt = `Generate technical documentation & API spec based on this diagram:\n\n\`\`\`${fileContent}\`\`\``;
    } else {
      prompt = `Compile this wireframe / UML specification into clean React + Tailwind CSS code:\n\n\`\`\`${fileContent}\`\`\``;
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
      setAiAnalysis(data.reply || 'No output generated.');
    } catch (err: any) {
      setAiAnalysis(`Error: ${err.message}`);
    } finally {
      setLoadingAi(false);
    }
  };

  return (
    <div className="h-full bg-slate-950 text-slate-100 p-4 flex flex-col gap-4 overflow-y-auto">
      {/* Header bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3 shrink-0">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          <div>
            <h3 className="font-bold text-sm font-mono text-white">Wireframes & UML Diagram Studio</h3>
            <p className="text-[11px] text-slate-400">
              Upload, view, and analyze UML diagrams, wireframes, and architecture specs directly from workspace directory <code className="text-indigo-300">/uploaded_diagrams</code>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            className="hidden"
            accept=".mermaid,.puml,.plantuml,.drawio,.uml,.spec,.png,.jpg,.jpeg,.svg,.json,.txt"
            multiple
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white font-mono font-bold text-xs rounded-lg transition flex items-center gap-1.5 cursor-pointer shadow"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Diagram / File</span>
          </button>
        </div>
      </div>

      {uploadStatus && (
        <div className="p-2.5 rounded-xl bg-slate-900 border border-indigo-900/60 text-indigo-300 text-xs font-mono flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>{uploadStatus}</span>
        </div>
      )}

      {/* Main Grid: Left Workspace File Browser & Right Preview/Editor */}
      <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-4 min-h-[500px]">
        {/* Column 1: Workspace Uploads Directory & Diagram Files */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <div className="flex items-center gap-2 font-mono text-xs font-bold text-slate-200">
              <HardDrive className="w-4 h-4 text-indigo-400" />
              <span>Workspace Diagrams ({diagramFiles.length})</span>
            </div>
            <span className="text-[10px] font-mono text-slate-500 uppercase bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
              {UPLOAD_DIR}/
            </span>
          </div>

          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 font-mono text-xs">
            {diagramFiles.length === 0 ? (
              <div className="p-4 text-center text-slate-500 text-[11px] space-y-2">
                <Box className="w-8 h-8 text-slate-700 mx-auto" />
                <p>No uploaded diagrams or UML files found in workspace.</p>
                <p className="text-[10px] text-slate-600">Click Upload or create a new diagram spec on the right.</p>
              </div>
            ) : (
              diagramFiles.map(file => {
                const isSelected = selectedFilePath === file.path;
                const isImg = file.name.match(/\.(png|jpg|jpeg|svg)$/i);
                return (
                  <button
                    key={file.path}
                    type="button"
                    onClick={() => handleSelectFile(file.path)}
                    className={`w-full text-left p-2 rounded-xl border flex items-center gap-2.5 transition cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-950/80 border-indigo-500 text-white font-bold shadow-sm'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                    }`}
                  >
                    {isImg ? (
                      <ImageIcon className="w-4 h-4 text-pink-400 shrink-0" />
                    ) : (
                      <FileCode className="w-4 h-4 text-indigo-400 shrink-0" />
                    )}
                    <div className="flex-1 min-w-0">
                      <div className="truncate font-semibold">{file.name}</div>
                      <div className="text-[10px] text-slate-500 truncate font-normal">{file.path}</div>
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Quick Create New Spec */}
          <div className="pt-2 border-t border-slate-800 space-y-2">
            <div className="text-[11px] font-bold text-slate-300 font-mono">Create New Diagram File:</div>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={diagramTitle}
                onChange={e => setDiagramTitle(e.target.value)}
                placeholder="filename.mermaid"
                className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1 text-xs font-mono text-slate-200 focus:outline-none focus:border-indigo-500"
              />
              <button
                type="button"
                onClick={() => {
                  setSelectedFilePath(`${UPLOAD_DIR}/${diagramTitle}`);
                  setFileContent(defaultMermaid);
                  setIsImageFile(false);
                  setImageUrl(null);
                }}
                className="p-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 cursor-pointer"
                title="Create Spec"
              >
                <FolderPlus className="w-4 h-4 text-indigo-400" />
              </button>
            </div>
          </div>
        </div>

        {/* Columns 2 & 3: Selected File Editor / Visual Preview & AI Actions */}
        <div className="md:col-span-2 bg-slate-900/80 border border-slate-800 rounded-2xl p-3 flex flex-col gap-3">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2 font-mono text-xs">
            <div className="flex items-center gap-2 text-indigo-400 font-bold">
              <Eye className="w-4 h-4" />
              <span>{selectedFilePath || 'New Unsaved Diagram'}</span>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleSaveDiagram}
                disabled={isSaving}
                className="px-2.5 py-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-lg transition flex items-center gap-1 cursor-pointer"
              >
                <Save className="w-3.5 h-3.5" />
                <span>{isSaving ? 'Saving...' : 'Save File'}</span>
              </button>
            </div>
          </div>

          {/* Spec Type Selectors (if not image) */}
          {!isImageFile && (
            <div className="flex items-center gap-1.5 font-mono text-xs overflow-x-auto pb-1">
              {(['mermaid', 'plantuml', 'json_wireframe', 'ascii', 'custom'] as const).map(type => (
                <button
                  key={type}
                  type="button"
                  onClick={() => setSpecType(type)}
                  className={`px-2.5 py-1 rounded-lg border font-bold capitalize transition cursor-pointer ${
                    specType === type
                      ? 'bg-indigo-950 border-indigo-500 text-indigo-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800'
                  }`}
                >
                  {type.replace('_', ' ')}
                </button>
              ))}
            </div>
          )}

          {/* Main Editor / Viewer Box */}
          <div className="flex-1 min-h-[220px] bg-slate-950 border border-slate-800 rounded-xl p-3 relative flex flex-col">
            {isImageFile && imageUrl ? (
              <div className="flex-1 flex flex-col items-center justify-center p-4 space-y-2">
                <img
                  src={imageUrl}
                  alt="Wireframe diagram asset"
                  className="max-h-72 max-w-full object-contain rounded-lg border border-slate-800 shadow-lg"
                />
                <span className="text-[10px] font-mono text-slate-500">{selectedFilePath}</span>
              </div>
            ) : (
              <textarea
                value={fileContent}
                onChange={e => setFileContent(e.target.value)}
                className="w-full flex-1 bg-transparent font-mono text-xs text-emerald-400 outline-none resize-none leading-relaxed"
                placeholder="Enter Mermaid, PlantUML, or Wireframe specification code here..."
              />
            )}
          </div>

          {/* AI Diagram Analysis & Action Controls */}
          <div className="p-3 bg-slate-950/90 border border-slate-800 rounded-xl space-y-3 font-mono text-xs">
            <div className="flex items-center justify-between">
              <span className="font-bold text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-indigo-400" /> AI Diagram Intelligence Tools:
              </span>
              {loadingAi && <RefreshCw className="w-4 h-4 text-indigo-400 animate-spin" />}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <button
                type="button"
                onClick={() => handleAiAction('analyze')}
                disabled={loadingAi}
                className="py-2 px-3 bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 rounded-xl font-bold transition flex items-center justify-center gap-2 cursor-pointer"
              >
                <Workflow className="w-3.5 h-3.5 text-blue-400" />
                <span>Analyze Architecture</span>
              </button>

              <button
                type="button"
                onClick={() => handleAiAction('generate_doc')}
                disabled={loadingAi}
                className="py-2 px-3 bg-slate-900 hover:bg-slate-800 text-slate-200 border border-slate-700 rounded-xl font-bold transition flex items-center justify-center gap-2 cursor-pointer"
              >
                <FileText className="w-3.5 h-3.5 text-amber-400" />
                <span>Generate Spec Doc</span>
              </button>

              <button
                type="button"
                onClick={() => handleAiAction('compile_code')}
                disabled={loadingAi}
                className="py-2 px-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl font-bold transition flex items-center justify-center gap-2 cursor-pointer shadow"
              >
                <Code className="w-3.5 h-3.5 text-indigo-200" />
                <span>Synthesize Code</span>
              </button>
            </div>

            {/* AI Analysis Output */}
            {aiAnalysis && (
              <div className="mt-3 p-3 bg-slate-900 border border-slate-800 rounded-xl max-h-60 overflow-y-auto space-y-2">
                <div className="text-[11px] font-bold text-emerald-400 flex items-center gap-1.5 border-b border-slate-800 pb-1.5">
                  <FileCheck className="w-4 h-4" /> AI Output:
                </div>
                <div className="prose prose-invert max-w-none text-xs leading-relaxed font-sans">
                  <ReactMarkdown>{aiAnalysis}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
