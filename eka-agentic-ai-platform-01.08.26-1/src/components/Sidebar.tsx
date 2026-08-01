/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { FileNode } from '../types';
import {
  Folder,
  File,
  Plus,
  FolderPlus,
  Trash2,
  ChevronDown,
  ChevronRight,
  Database,
  RefreshCw,
  FolderOpen,
  Settings,
  Archive
} from 'lucide-react';
import { motion } from 'motion/react';

interface SidebarProps {
  files: FileNode[];
  selectedFile: string | null;
  selectedFolder: string | null;
  onSelectFile: (path: string) => void;
  onSelectFolder: (path: string) => void;
  onCreateFile: (parentPath: string, name: string) => void;
  onCreateFolder: (parentPath: string, name: string) => void;
  onDeleteFile: (path: string) => void;
  onDeleteFolder: (path: string) => void;
  activeBaseFolder: string;
  onSelectBaseFolder: (path: string) => void;
  onRefreshWorkspace: () => void;
  onExportZip?: () => void;
  theme?: 'white' | 'light-grey' | 'dark';
  
  // New props for Local & Server directory picking
  serverRootPath?: string;
  onSetServerRootPath?: (path: string) => Promise<boolean>;
  onMountLocalFolder?: () => Promise<void>;
  localDirName?: string;
}

export default function Sidebar({
  files,
  selectedFile,
  selectedFolder,
  onSelectFile,
  onSelectFolder,
  onCreateFile,
  onCreateFolder,
  onDeleteFile,
  onDeleteFolder,
  activeBaseFolder,
  onSelectBaseFolder,
  onRefreshWorkspace,
  onExportZip,
  theme = 'dark',
  serverRootPath = '',
  onSetServerRootPath,
  onMountLocalFolder,
  localDirName = '',
}: SidebarProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    'src': true,
    'tests': true,
  });
  const [newInput, setNewInput] = useState<{ parentPath: string; type: 'file' | 'folder' } | null>(null);
  const [newItemName, setNewItemName] = useState('');
  const [showMountPanel, setShowMountPanel] = useState(false);
  const [customPathInput, setCustomPathInput] = useState(serverRootPath);

  const [serverBrowserPath, setServerBrowserPath] = useState(serverRootPath || '');
  const [serverSubdirs, setServerSubdirs] = useState<string[]>([]);
  const [serverParentPath, setServerParentPath] = useState<string | null>(null);
  const [isBrowserLoading, setIsBrowserLoading] = useState(false);
  const [browserError, setBrowserError] = useState<string | null>(null);

  // Sync customPathInput and serverBrowserPath when serverRootPath loads or changes
  useEffect(() => {
    if (serverRootPath) {
      setCustomPathInput(serverRootPath);
      setServerBrowserPath(serverRootPath);
    }
  }, [serverRootPath]);

  const fetchServerDirs = async (pathStr: string) => {
    setIsBrowserLoading(true);
    setBrowserError(null);
    try {
      const response = await fetch(`/api/workspace/list-server-dirs?path=${encodeURIComponent(pathStr)}`);
      const data = await response.json();
      if (data.success) {
        setServerBrowserPath(data.currentPath);
        setServerSubdirs(data.subdirs || []);
        setServerParentPath(data.parentPath || null);
        setCustomPathInput(data.currentPath);
      } else {
        setBrowserError(data.error || 'Failed to list directories');
      }
    } catch (err: any) {
      setBrowserError(err.message || 'Error loading directory');
    } finally {
      setIsBrowserLoading(false);
    }
  };

  useEffect(() => {
    if (showMountPanel) {
      fetchServerDirs(serverBrowserPath);
    }
  }, [showMountPanel, serverBrowserPath]);

  // Helper to extract all folder paths recursively
  const getAllDirectories = (nodes: FileNode[]): string[] => {
    let dirs: string[] = [];
    const traverse = (list: FileNode[]) => {
      for (const node of list) {
        if (node.type === 'directory') {
          dirs.push(node.path);
          if (node.children) traverse(node.children);
        }
      }
    };
    traverse(nodes);
    return dirs.sort();
  };

  // Find a specific node by its relative path
  const findNodeByPath = (nodes: FileNode[], targetPath: string): FileNode | null => {
    for (const node of nodes) {
      if (node.path === targetPath) return node;
      if (node.children) {
        const found = findNodeByPath(node.children, targetPath);
        if (found) return found;
      }
    }
    return null;
  };

  const allDirs = getAllDirectories(files);
  const activeBaseNode = activeBaseFolder && activeBaseFolder !== 'Root' 
    ? findNodeByPath(files, activeBaseFolder) 
    : null;
  
  // Fix A: The visual tree always displays all files in the hierarchy, 
  // preventing it from collapsing/disappearing when navigating to a folder!
  const nodesToRender = files;

  const toggleExpand = (path: string) => {
    setExpanded(prev => ({ ...prev, [path]: !prev[path] }));
  };

  const handleCreateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newItemName.trim()) return;

    if (newInput) {
      if (newInput.type === 'file') {
        onCreateFile(newInput.parentPath, newItemName.trim());
      } else {
        onCreateFolder(newInput.parentPath, newItemName.trim());
      }
    }
    setNewInput(null);
    setNewItemName('');
  };

  const renderNode = (node: FileNode, depth: number = 0) => {
    const isDir = node.type === 'directory';
    const isExpanded = expanded[node.path];
    const isSelected = (!isDir && selectedFile === node.path) || (isDir && selectedFolder === node.path);
    const isActiveBase = activeBaseFolder === node.path || (node.path === '' && activeBaseFolder === 'Root');

    return (
      <div key={node.path} className="space-y-0.5">
        <div
          className={`group flex items-center justify-between px-2 py-1 rounded transition-colors hover:bg-slate-800/80 cursor-pointer ${
            isSelected 
              ? 'bg-indigo-500/10 text-indigo-400 font-bold border-l-2 border-indigo-500 rounded-l-none' 
              : 'text-slate-400 hover:text-slate-200'
          }`}
          style={{ paddingLeft: `${depth * 10 + 8}px` }}
        >
          <div 
            className="flex items-center gap-1.5 flex-1 min-w-0" 
            onClick={() => {
              if (isDir) {
                toggleExpand(node.path);
                onSelectFolder(node.path);
              } else {
                onSelectFile(node.path);
              }
            }}
            onDoubleClick={() => {
              if (!isDir) {
                onSelectFile(node.path);
              }
            }}
          >
            {isDir ? (
              <>
                {isExpanded ? (
                  <ChevronDown className="w-3 h-3 shrink-0 text-slate-500" />
                ) : (
                  <ChevronRight className="w-3 h-3 shrink-0 text-slate-500" />
                )}
                <Folder className="w-3.5 h-3.5 text-amber-500 shrink-0" />
              </>
            ) : (
              <>
                <span className="w-3" />
                <File className="w-3.5 h-3.5 text-indigo-400/80 shrink-0" />
              </>
            )}
            <span className="truncate text-[11px] font-mono">{node.name}</span>
          </div>

          <div className="opacity-0 group-hover:opacity-100 flex items-center gap-0.5 shrink-0">
            {isDir && (
              <>
                <button
                  type="button"
                  title="New File"
                  onClick={(e) => {
                    e.stopPropagation();
                    setNewInput({ parentPath: node.path, type: 'file' });
                  }}
                  className="p-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
                >
                  <Plus className="w-3 h-3" />
                </button>
                <button
                  type="button"
                  title="New Folder"
                  onClick={(e) => {
                    e.stopPropagation();
                    setNewInput({ parentPath: node.path, type: 'folder' });
                  }}
                  className="p-0.5 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
                >
                  <FolderPlus className="w-3 h-3" />
                </button>
                {node.path && (
                  <button
                    type="button"
                    title="Set as Workspace Base"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectBaseFolder(node.path);
                    }}
                    className={`p-0.5 rounded text-slate-400 hover:text-white ${
                      isActiveBase ? 'bg-indigo-500/20 text-indigo-300' : ''
                    }`}
                  >
                    <FolderOpen className="w-3 h-3" />
                  </button>
                )}
              </>
            )}
            {node.path && (
              <button
                type="button"
                title="Delete"
                onClick={(e) => {
                  e.stopPropagation();
                  if (confirm(`Are you sure you want to delete ${node.name}?`)) {
                    if (isDir) {
                      onDeleteFolder(node.path);
                    } else {
                      onDeleteFile(node.path);
                    }
                  }
                }}
                className="p-0.5 hover:bg-rose-500/20 rounded text-slate-400 hover:text-rose-400"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {newInput && newInput.parentPath === node.path && (
          <form
            onSubmit={handleCreateSubmit}
            className="flex items-center gap-1 py-0.5"
            style={{ paddingLeft: `${(depth + 1) * 10 + 8}px` }}
            onClick={(e) => e.stopPropagation()}
          >
            {newInput.type === 'file' ? (
              <File className="w-3 h-3 text-indigo-400/80" />
            ) : (
              <Folder className="w-3 h-3 text-amber-500" />
            )}
            <input
              autoFocus
              type="text"
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              placeholder={`New ${newInput.type}...`}
              className="bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-200 rounded px-1.5 py-0.5 w-28 focus:outline-none focus:border-indigo-500"
              onBlur={() => setNewInput(null)}
            />
          </form>
        )}

        {isDir && isExpanded && node.children && (
          <div className="space-y-0.5">
            {node.children.map(child => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const isDark = theme === 'dark';
  const isGray = theme === 'light-grey';
  const sidebarBg = isDark ? 'bg-slate-900 border-slate-800' : isGray ? 'bg-zinc-100 border-zinc-300' : 'bg-white border-slate-200';
  const headerBg = isDark ? 'bg-slate-900 border-slate-800' : isGray ? 'bg-zinc-200 border-zinc-300' : 'bg-slate-50 border-slate-100';
  const headerText = isDark ? 'text-white' : 'text-slate-900';
  const dropdownBg = isDark ? 'bg-slate-950 border-slate-800 text-slate-300' : isGray ? 'bg-zinc-50 border-zinc-300 text-zinc-800' : 'bg-white border-slate-200 text-slate-700';

  return (
    <div className={`flex flex-col h-full rounded-lg overflow-hidden shadow-lg border ${sidebarBg}`} id="workspace-sidebar">
      {/* Unified Sidebar Header */}
      <div className={`p-3 border-b flex items-center justify-between ${headerBg}`}>
        <div className="flex items-center gap-2">
          <FolderOpen className="w-3.5 h-3.5 text-indigo-500 shrink-0" />
          <span className={`font-display font-semibold text-xs tracking-tight ${headerText}`}>Workspace Files</span>
        </div>
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowMountPanel(!showMountPanel)}
            className={`p-1 rounded transition-colors ${showMountPanel ? 'bg-indigo-500/20 text-indigo-400' : isDark ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200'}`}
            title="Configure Workspace Root / Mount Folder"
          >
            <Settings className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              const creationPath = selectedFolder || '';
              setNewInput({ parentPath: creationPath, type: 'file' });
            }}
            className={`p-1 rounded transition-colors ${isDark ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200'}`}
            title="Create File in Current Highlighted Directory"
          >
            <Plus className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => {
              const creationPath = selectedFolder || '';
              setNewInput({ parentPath: creationPath, type: 'folder' });
            }}
            className={`p-1 rounded transition-colors ${isDark ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200'}`}
            title="Create Folder in Current Highlighted Directory"
          >
            <FolderPlus className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={onRefreshWorkspace}
            className={`p-1 rounded transition-colors ${isDark ? 'text-slate-400 hover:text-white hover:bg-slate-800' : 'text-slate-500 hover:text-slate-800 hover:bg-slate-200'}`}
            title="Refresh Files"
          >
            <RefreshCw className="w-3 h-3" />
          </button>
          {onExportZip && (
            <button
              type="button"
              onClick={onExportZip}
              className={`p-1 rounded transition-colors ${isDark ? 'text-slate-400 hover:text-emerald-400 hover:bg-emerald-500/10' : 'text-slate-500 hover:text-emerald-700 hover:bg-emerald-100'}`}
              title="Export workspace as ZIP"
            >
              <Archive className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Dynamic Filesystem Mount & Path Selection Panel */}
      {showMountPanel && (
        <div className={`p-3 border-b space-y-3.5 text-xs transition-all ${isDark ? 'bg-slate-950/40 border-slate-850' : 'bg-slate-50 border-slate-200'}`}>
          <div className="space-y-1.5">
            <span className={`block font-semibold ${isDark ? 'text-slate-300' : 'text-slate-700'} flex items-center gap-1.5`}>
              📂 Server OS Directory Browser
            </span>
            <span className={`block text-[10px] leading-relaxed ${isDark ? 'text-slate-400' : 'text-slate-600'}`}>
              Browse and select any folder directly on the server's filesystem. No browser uploads required.
            </span>
            
            <div className="flex gap-1">
              <input
                type="text"
                value={serverBrowserPath}
                onChange={(e) => setServerBrowserPath(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    fetchServerDirs(serverBrowserPath);
                  }
                }}
                className={`flex-1 text-[11px] font-mono rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500 ${isDark ? 'bg-slate-950 border-slate-850 text-slate-200' : 'bg-white border-slate-300 text-slate-850'}`}
                placeholder="Absolute server folder path"
              />
              <button
                type="button"
                onClick={() => fetchServerDirs(serverBrowserPath)}
                className={`py-1 px-2.5 rounded text-[10px] font-medium transition-colors cursor-pointer border ${isDark ? 'bg-slate-800 hover:bg-slate-750 text-slate-200 border-slate-750' : 'bg-slate-100 hover:bg-slate-200 text-slate-800 border-slate-300'}`}
              >
                Go
              </button>
            </div>

            {browserError && (
              <div className="text-[10px] text-rose-500 font-mono bg-rose-500/10 p-2 rounded border border-rose-500/20">
                ⚠️ {browserError}
              </div>
            )}

            {/* Folder Navigation Tree */}
            <div className={`p-2 rounded border space-y-1.5 ${isDark ? 'bg-slate-950/60 border-slate-850' : 'bg-white border-zinc-200'}`}>
              {/* Parent folder link */}
              {serverParentPath && serverParentPath !== serverBrowserPath && (
                <button
                  type="button"
                  onClick={() => {
                    if (serverParentPath) {
                      setServerBrowserPath(serverParentPath);
                    }
                  }}
                  className={`w-full text-left py-0.5 px-1.5 rounded text-[10px] font-medium flex items-center gap-1.5 transition-colors ${
                    isDark ? 'hover:bg-slate-800 text-indigo-400' : 'hover:bg-zinc-100 text-indigo-600'
                  }`}
                >
                  <span>⬆️</span>
                  <span>[Go Up] Parent Directory</span>
                </button>
              )}

              {isBrowserLoading ? (
                <div className="text-center py-4 text-[10px] text-slate-400 animate-pulse">
                  Loading subdirectories...
                </div>
              ) : serverSubdirs.length === 0 ? (
                <div className="text-center py-2 text-[10px] text-slate-500 italic">
                  No subdirectories found
                </div>
              ) : (
                <div className="max-h-[120px] overflow-y-auto space-y-0.5 font-mono text-[10px]">
                  {serverSubdirs.map((subdir) => {
                    const isWindows = serverBrowserPath.includes('\\');
                    const sep = isWindows ? '\\' : '/';
                    const fullSubdirPath = serverBrowserPath.endsWith(sep)
                      ? serverBrowserPath + subdir
                      : serverBrowserPath + sep + subdir;
                    return (
                      <button
                        type="button"
                        key={subdir}
                        onClick={() => {
                          setServerBrowserPath(fullSubdirPath);
                        }}
                        className={`w-full text-left py-1 px-1.5 rounded flex items-center gap-1.5 transition-colors truncate ${
                          isDark ? 'hover:bg-slate-800 text-slate-300' : 'hover:bg-zinc-100 text-slate-700'
                        }`}
                        title={fullSubdirPath}
                      >
                        <span className="text-amber-500">📁</span>
                        <span className="truncate">{subdir}</span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            <button
              type="button"
              onClick={async () => {
                if (onSetServerRootPath) {
                  const success = await onSetServerRootPath(serverBrowserPath);
                  if (success) {
                    alert(`Successfully set workspace to server folder: ${serverBrowserPath}`);
                  }
                }
              }}
              disabled={isBrowserLoading}
              className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded bg-indigo-600 hover:bg-indigo-500 text-white font-medium transition-colors cursor-pointer text-[11px] disabled:opacity-50"
            >
              <FolderOpen className="w-3.5 h-3.5" />
              <span>Mount Selected Folder</span>
            </button>
          </div>

          <div className="border-t border-slate-800/60 dark:border-slate-800 pt-2.5 space-y-1">
            <span className={`block text-[10px] leading-tight ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
              Mounted path: <code className="font-mono text-[9px] text-indigo-400 break-all">{serverRootPath || '/'}</code>
            </span>
          </div>
        </div>
      )}

      {/* Selected Directory Context Bar */}
      <div className={`p-2 border-b flex flex-col gap-0.5 ${isDark ? 'bg-slate-900/45 border-slate-850' : 'bg-slate-50/50 border-slate-150'}`}>
        <span className={`text-[9px] uppercase font-bold tracking-wider font-mono ${isDark ? 'text-slate-500' : 'text-slate-400'}`}>
          Active Folder Context:
        </span>
        <div className={`text-[11px] font-mono px-2 py-1 rounded truncate flex items-center gap-1.5 ${isDark ? 'bg-slate-950 text-slate-300' : 'bg-zinc-100 text-slate-700'}`}>
          <Folder className="w-3.5 h-3.5 text-amber-500 shrink-0" />
          <span>/{selectedFolder || '(Root)'}</span>
        </div>
      </div>

      {/* File Tree List */}
      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {newInput && (newInput.parentPath === '' || newInput.parentPath === 'Root') && (
          <form
            onSubmit={handleCreateSubmit}
            className="flex items-center gap-1 py-0.5 px-2"
            onClick={(e) => e.stopPropagation()}
          >
            {newInput.type === 'file' ? (
              <File className="w-3.5 h-3.5 text-indigo-400/80" />
            ) : (
              <Folder className="w-3.5 h-3.5 text-amber-500" />
            )}
            <input
              autoFocus
              type="text"
              value={newItemName}
              onChange={(e) => setNewItemName(e.target.value)}
              placeholder={`New ${newInput.type}...`}
              className={`border text-[11px] font-mono rounded px-1.5 py-0.5 w-full focus:outline-none focus:border-indigo-500 ${isDark ? 'bg-slate-950 border-slate-800 text-slate-200' : 'bg-white border-slate-300 text-slate-800'}`}
              onBlur={() => setNewInput(null)}
            />
          </form>
        )}

        {nodesToRender.length > 0 ? (
          nodesToRender.map(node => renderNode(node, 0))
        ) : (
          <div className="text-center py-8 text-xs text-slate-500 font-mono">
            No files found in workspace root
          </div>
        )}
      </div>
    </div>
  );
}
