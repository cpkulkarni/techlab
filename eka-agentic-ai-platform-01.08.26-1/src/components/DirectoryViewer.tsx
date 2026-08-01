/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { FileNode } from '../types';
import {
  Folder,
  File,
  Plus,
  FolderPlus,
  Trash2,
  ChevronRight,
  FolderOpen,
  ArrowUp,
  FileCode,
  HardDrive,
  Grid,
  List,
  Calendar,
  Layers,
  Sparkles,
  Lock,
  PanelLeftClose,
  Settings,
  CornerDownRight,
  Search,
  X,
  Check,
  FolderTree,
  CheckCircle2
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

interface DirectoryViewerProps {
  folderPath: string;
  files: FileNode[];
  serverRootPath?: string;
  onSetServerRootPath?: (newPath: string) => Promise<boolean>;
  onSelectFile: (path: string) => void;
  onSelectFolder: (path: string) => void;
  onCreateFile: (parentPath: string, name: string) => void;
  onCreateFolder: (parentPath: string, name: string) => void;
  onDeleteFile: (path: string) => void;
  onDeleteFolder: (path: string) => void;
  onToggleMinimize?: () => void;
}

export default function DirectoryViewer({
  folderPath,
  files,
  serverRootPath,
  onSetServerRootPath,
  onSelectFile,
  onSelectFolder,
  onCreateFile,
  onCreateFolder,
  onDeleteFile,
  onDeleteFolder,
  onToggleMinimize,
}: DirectoryViewerProps) {
  const [viewMode, setViewMode] = useState<'list' | 'grid'>('list');
  const [showCreateForm, setShowCreateForm] = useState<'file' | 'folder' | null>(null);
  const [newItemName, setNewItemName] = useState('');
  const [showChangeDir, setShowChangeDir] = useState(false);
  const [customPathInput, setCustomPathInput] = useState('');

  // Local Disk Directory Picker Modal state
  const [showFolderSelectorModal, setShowFolderSelectorModal] = useState(false);
  const [targetDiskPathInput, setTargetDiskPathInput] = useState('');
  const [currentBrowseDiskPath, setCurrentBrowseDiskPath] = useState('');
  const [parentDiskPath, setParentDiskPath] = useState('');
  const [diskSubdirs, setDiskSubdirs] = useState<string[]>([]);
  const [isLoadingDiskDirs, setIsLoadingDiskDirs] = useState(false);
  const [diskDirError, setDiskDirError] = useState<string | null>(null);
  const [isApplyingRootChange, setIsApplyingRootChange] = useState(false);

  const fetchDiskDirs = async (pathQuery?: string) => {
    setIsLoadingDiskDirs(true);
    setDiskDirError(null);
    try {
      const url = pathQuery 
        ? `/api/workspace/list-server-dirs?path=${encodeURIComponent(pathQuery)}`
        : '/api/workspace/list-server-dirs';
      const res = await fetch(url);
      const data = await res.json();
      if (data.success) {
        setCurrentBrowseDiskPath(data.currentPath);
        setParentDiskPath(data.parentPath || '');
        setDiskSubdirs(data.subdirs || []);
        setTargetDiskPathInput(data.currentPath);
      } else {
        setDiskDirError(data.error || 'Failed to list directory contents');
      }
    } catch (err: any) {
      setDiskDirError(err.message || 'Error connecting to local file system');
    } finally {
      setIsLoadingDiskDirs(false);
    }
  };

  const handleOpenFolderSelector = () => {
    setShowFolderSelectorModal(true);
    fetchDiskDirs(serverRootPath || '');
  };

  const handleApplyRootWorkspacePath = async (pathToSend?: string) => {
    const finalPath = (pathToSend || targetDiskPathInput).trim();
    if (!finalPath) return;
    setIsApplyingRootChange(true);
    try {
      if (onSetServerRootPath) {
        const success = await onSetServerRootPath(finalPath);
        if (success) {
          setShowFolderSelectorModal(false);
        } else {
          setDiskDirError(`Failed to set directory "${finalPath}" as workspace root.`);
        }
      }
    } catch (err: any) {
      setDiskDirError(err.message || 'Failed to update workspace root path.');
    } finally {
      setIsApplyingRootChange(false);
    }
  };

  const handleNativeOSFolderPicker = async () => {
    if ('showDirectoryPicker' in window) {
      try {
        const dirHandle = await (window as any).showDirectoryPicker();
        if (dirHandle && dirHandle.name) {
          const newPath = currentBrowseDiskPath 
            ? `${currentBrowseDiskPath}/${dirHandle.name}` 
            : dirHandle.name;
          setTargetDiskPathInput(newPath);
          await handleApplyRootWorkspacePath(newPath);
        }
      } catch (err: any) {
        if (err.name === 'AbortError') return;
        console.warn('showDirectoryPicker cancelled or error:', err);
      }
    }
  };

  // Recursive search for the current directory node
  const findNodeByPath = (nodes: FileNode[], path: string): FileNode | null => {
    if (path === '' || path === 'Root') {
      return { name: 'Root', path: '', type: 'directory', children: nodes };
    }
    for (const node of nodes) {
      if (node.path === path) return node;
      if (node.children) {
        const found = findNodeByPath(node.children, path);
        if (found) return found;
      }
    }
    return null;
  };

  const currentNode = findNodeByPath(files, folderPath) || { name: 'Root', path: '', type: 'directory' as const, children: files };
  const items = currentNode.children || [];

  const handleCreateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newItemName.trim()) return;

    if (showCreateForm === 'file') {
      onCreateFile(currentNode.path, newItemName.trim());
    } else {
      onCreateFolder(currentNode.path, newItemName.trim());
    }

    setNewItemName('');
    setShowCreateForm(null);
  };

  // Breadcrumbs generator
  const getBreadcrumbs = () => {
    const parts = folderPath.split('/').filter(Boolean);
    const crumbs = [{ name: 'Root', path: '' }];
    let currentPath = '';
    
    parts.forEach((part) => {
      currentPath = currentPath ? `${currentPath}/${part}` : part;
      crumbs.push({ name: part, path: currentPath });
    });
    
    return crumbs;
  };

  const crumbs = getBreadcrumbs();

  // Statistics
  const fileCount = items.filter(i => i.type === 'file').length;
  const folderCount = items.filter(i => i.type === 'directory').length;

  return (
    <div className="bg-slate-950 text-slate-200 rounded-lg border border-slate-800 shadow-xl overflow-hidden flex flex-col h-full" id="directory-viewer">
      
      {/* Breadcrumb Header Toolbar */}
      <div className="bg-slate-900 border-b border-slate-800 px-3 py-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none py-1">
          <HardDrive className="w-3.5 h-3.5 text-indigo-400 shrink-0 mr-1" />
          {crumbs.map((crumb, idx) => (
            <React.Fragment key={crumb.path}>
              {idx > 0 && <ChevronRight className="w-3 h-3 text-slate-600 shrink-0" />}
              <button
                type="button"
                onClick={() => onSelectFolder(crumb.path)}
                className={`text-[11px] font-mono hover:text-white transition-colors shrink-0 ${
                  idx === crumbs.length - 1 ? 'text-indigo-400 font-bold' : 'text-slate-400'
                }`}
              >
                {crumb.name}
              </button>
            </React.Fragment>
          ))}
        </div>

        {/* Layout controls */}
        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setViewMode('list')}
            className={`p-1 rounded transition-all cursor-pointer ${
              viewMode === 'list' ? 'bg-indigo-600/20 text-indigo-400' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="List View"
          >
            <List className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setViewMode('grid')}
            className={`p-1 rounded transition-all cursor-pointer ${
              viewMode === 'grid' ? 'bg-indigo-600/20 text-indigo-400' : 'text-slate-500 hover:text-slate-300'
            }`}
            title="Grid View"
          >
            <Grid className="w-3.5 h-3.5" />
          </button>

          {onToggleMinimize && (
            <button
              type="button"
              onClick={onToggleMinimize}
              className="p-1 text-slate-400 hover:text-white rounded hover:bg-slate-800 transition cursor-pointer ml-1"
              title="Minimize File Panel"
            >
              <PanelLeftClose className="w-3.5 h-3.5 theme-accent-text" />
            </button>
          )}
        </div>
      </div>

      {/* Directory Statistics & Action bar */}
      <div className="px-4 py-2 bg-slate-900/40 border-b border-slate-850 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-3 text-[10px] text-slate-500 font-mono">
          <span className="flex items-center gap-1">
            <Layers className="w-3 h-3 text-indigo-500" />
            <span>{folderCount} Folders</span>
          </span>
          <span>|</span>
          <span className="flex items-center gap-1">
            <FileCode className="w-3 h-3 text-emerald-500" />
            <span>{fileCount} Files</span>
          </span>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleOpenFolderSelector}
            className="px-2 py-0.5 theme-accent-badge text-[10px] font-mono font-medium rounded flex items-center gap-1 transition-all cursor-pointer shadow-sm"
            title="Open workspace folder selector modal"
          >
            <FolderOpen className="w-3 h-3 theme-accent-text" />
            <span>Change Dir</span>
          </button>
          <button
            type="button"
            onClick={() => setShowCreateForm('file')}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-750 text-slate-300 text-[10px] font-mono font-medium rounded flex items-center gap-1 transition-all cursor-pointer"
          >
            <Plus className="w-3 h-3 text-indigo-400" />
            <span>+ File</span>
          </button>
          <button
            type="button"
            onClick={() => setShowCreateForm('folder')}
            className="px-2 py-0.5 bg-slate-800 hover:bg-slate-750 text-slate-300 text-[10px] font-mono font-medium rounded flex items-center gap-1 transition-all cursor-pointer"
          >
            <FolderPlus className="w-3 h-3 text-amber-500" />
            <span>+ Folder</span>
          </button>
        </div>
      </div>

      {/* Create form modal overlay */}
      {showCreateForm && (
        <form 
          onSubmit={handleCreateSubmit}
          className="p-3 bg-slate-900/90 border-b border-slate-800 flex items-center gap-2 shrink-0 animate-fadeIn"
        >
          {showCreateForm === 'file' ? (
            <File className="w-4 h-4 text-indigo-400" />
          ) : (
            <Folder className="w-4 h-4 text-amber-500" />
          )}
          <input
            autoFocus
            type="text"
            value={newItemName}
            onChange={(e) => setNewItemName(e.target.value)}
            placeholder={`Enter new ${showCreateForm} name...`}
            className="flex-1 bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-200 rounded px-2.5 py-1 focus:outline-none focus:border-indigo-500"
          />
          <div className="flex gap-1">
            <button
              type="button"
              onClick={() => {
                setShowCreateForm(null);
                setNewItemName('');
              }}
              className="px-2 py-1 bg-slate-800 text-slate-400 hover:text-white font-mono text-[10px] rounded"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-2.5 py-1 bg-indigo-600 hover:bg-indigo-550 text-white font-mono text-[10px] rounded font-medium"
            >
              Create
            </button>
          </div>
        </form>
      )}

      {/* Directory Browser list/grid */}
      <div className="flex-1 overflow-y-auto p-4">
        {Boolean(folderPath) && (
          <button
            type="button"
            onClick={() => {
              const parts = folderPath.split('/').filter(Boolean);
              parts.pop();
              onSelectFolder(parts.join('/'));
            }}
            className="w-full flex items-center gap-2 px-3 py-2 mb-3 bg-slate-900/80 hover:bg-slate-800 border border-slate-800 text-indigo-300 hover:text-white rounded text-xs font-mono font-medium transition-all cursor-pointer shadow-sm"
          >
            <ArrowUp className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <span>.. (Go Up One Level)</span>
          </button>
        )}

        {items.length === 0 ? (
          <div className="h-full py-16 flex flex-col items-center justify-center text-center text-slate-600 border border-dashed border-slate-850 rounded-lg">
            <FolderOpen className="w-10 h-10 text-slate-850 animate-pulse mb-2" />
            <p className="text-[11px] font-mono">This directory is empty</p>
            <p className="text-[10px] text-slate-600 mt-1">Create files or folders to populate the workspace.</p>
          </div>
        ) : viewMode === 'list' ? (
          <div className="space-y-1">
            {items.map((item) => (
              <motion.div
                initial={{ opacity: 0, y: 3 }}
                animate={{ opacity: 1, y: 0 }}
                key={item.path}
                onClick={() => {
                  if (item.type === 'directory') {
                    onSelectFolder(item.path);
                  } else {
                    onSelectFile(item.path);
                  }
                }}
                onDoubleClick={() => {
                  if (item.type !== 'directory') {
                    onSelectFile(item.path);
                  }
                }}
                className={`group flex items-center justify-between px-3 py-2 border rounded transition-all cursor-pointer ${
                  item.locked
                    ? 'bg-slate-900/20 border-slate-800/40 opacity-80'
                    : 'bg-slate-900/30 hover:bg-slate-900/80 border-slate-900 hover:border-slate-800/80'
                }`}
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  {item.locked ? (
                    item.type === 'directory'
                      ? <Folder className="w-4 h-4 text-slate-500 shrink-0" />
                      : <File className="w-4 h-4 text-slate-500 shrink-0" />
                  ) : item.type === 'directory' ? (
                    <Folder className="w-4 h-4 text-amber-500 shrink-0" />
                  ) : (
                    <File className="w-4 h-4 text-indigo-400/80 shrink-0" />
                  )}
                  <div className="min-w-0">
                    <p className={`text-[11.5px] font-mono font-medium truncate transition-colors ${
                      item.locked ? 'text-slate-400' : 'text-slate-200 group-hover:text-indigo-400'
                    }`}>
                      {item.name}
                    </p>
                    <p className="text-[9px] text-slate-500 font-mono">
                      {item.locked ? 'locked · read-only' : item.type === 'directory' ? 'Directory' : 'File'}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <span className="text-[10px] text-slate-600 font-mono">
                    {item.type === 'directory' ? '--' : `${((item.content?.length || 0) / 1024).toFixed(1)} KB`}
                  </span>
                  {item.locked ? (
                    <span title="Locked — part of platform source code"><Lock className="w-3 h-3 text-slate-600 shrink-0" /></span>
                  ) : (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Are you sure you want to delete ${item.name}?`)) {
                          if (item.type === 'directory') {
                            onDeleteFolder(item.path);
                          } else {
                            onDeleteFile(item.path);
                          }
                        }
                      }}
                      className="p-1 opacity-0 group-hover:opacity-100 bg-rose-500/10 hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 rounded transition-all"
                      title="Delete Item"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {items.map((item) => (
              <motion.div
                initial={{ opacity: 0, scale: 0.97 }}
                animate={{ opacity: 1, scale: 1 }}
                key={item.path}
                onClick={() => {
                  if (item.type === 'directory') {
                    onSelectFolder(item.path);
                  } else {
                    onSelectFile(item.path);
                  }
                }}
                onDoubleClick={() => {
                  if (item.type !== 'directory') {
                    onSelectFile(item.path);
                  }
                }}
                className={`group p-3 border rounded-lg flex flex-col justify-between h-24 transition-all cursor-pointer relative overflow-hidden ${
                  item.locked
                    ? 'bg-slate-900/20 border-slate-800/40 opacity-80'
                    : 'bg-slate-900/30 hover:bg-slate-900/80 border-slate-900 hover:border-slate-800/80'
                }`}
              >
                <div className="absolute right-1 top-1">
                  {item.locked ? (
                    <span title="Locked — part of platform source code"><Lock className="w-3 h-3 text-slate-600 m-1" /></span>
                  ) : (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (confirm(`Are you sure you want to delete ${item.name}?`)) {
                          if (item.type === 'directory') {
                            onDeleteFolder(item.path);
                          } else {
                            onDeleteFile(item.path);
                          }
                        }
                      }}
                      className="p-1 opacity-0 group-hover:opacity-100 bg-rose-500/10 hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 rounded transition-all"
                      title="Delete Item"
                    >
                      <Trash2 className="w-3 h-3" />
                    </button>
                  )}
                </div>

                {item.locked ? (
                  item.type === 'directory'
                    ? <Folder className="w-6 h-6 text-slate-500 mb-2" />
                    : <File className="w-6 h-6 text-slate-500 mb-2" />
                ) : item.type === 'directory' ? (
                  <Folder className="w-6 h-6 text-amber-500 mb-2" />
                ) : (
                  <File className="w-6 h-6 text-indigo-400/80 mb-2" />
                )}

                <div className="min-w-0">
                  <p className={`text-[11px] font-mono font-medium truncate transition-colors ${
                    item.locked ? 'text-slate-400' : 'text-slate-200 group-hover:text-indigo-400'
                  }`}>
                    {item.name}
                  </p>
                  <div className="flex justify-between items-center mt-0.5">
                    <span className="text-[9px] text-slate-500 font-mono">
                      {item.locked ? 'locked' : item.type === 'directory' ? 'folder' : 'file'}
                    </span>
                    <span className="text-[9px] text-slate-600 font-mono">
                      {item.type === 'directory' ? '' : `${((item.content?.length || 0) / 1024).toFixed(1)} KB`}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Local Disk Directory Selector Modal */}
      <AnimatePresence>
        {showFolderSelectorModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm">
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 10 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 10 }}
              className="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-xl shadow-2xl overflow-hidden flex flex-col max-h-[88vh]"
            >
              {/* Modal Header */}
              <div className="px-5 py-4 theme-panel-header border-b border-slate-800 flex items-center justify-between shrink-0">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg border border-indigo-500/20">
                    <HardDrive className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
                      Set Local Drive Directory as Root
                    </h3>
                    <p className="text-xs text-slate-400 mt-0.5">
                      Type any local drive path or browse folders on your drive to set as workspace root
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowFolderSelectorModal(false)}
                  className="p-1.5 text-slate-400 hover:text-white bg-slate-800/60 hover:bg-slate-800 rounded-lg transition-colors cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Direct Path Input Bar & OS Picker Button */}
              <div className="p-4 border-b border-slate-800 bg-slate-950/70 space-y-3 shrink-0">
                <label className="block text-[11px] uppercase font-bold text-slate-400 tracking-wider">
                  Target Disk / Directory Path:
                </label>
                <div className="flex items-center gap-2">
                  <div className="relative flex-1">
                    <FolderOpen className="w-4 h-4 text-indigo-400 absolute left-3 top-2.5" />
                    <input
                      type="text"
                      value={targetDiskPathInput}
                      onChange={(e) => setTargetDiskPathInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleApplyRootWorkspacePath();
                      }}
                      placeholder="e.g. /home/user/my-project or C:\Projects\my-app"
                      className="w-full pl-9 pr-3 py-2 bg-slate-900 border border-slate-700/80 rounded-lg text-xs font-mono text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 shadow-inner"
                    />
                  </div>
                  <button
                    type="button"
                    onClick={() => fetchDiskDirs(targetDiskPathInput)}
                    className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors cursor-pointer shrink-0 border border-slate-700"
                    title="Inspect directory contents at target path"
                  >
                    Inspect Path
                  </button>
                </div>

                {/* Quick Presets & OS Folder Picker Button */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-1 text-[11px] font-mono">
                  <div className="flex items-center gap-1.5 overflow-x-auto scrollbar-none">
                    <span className="text-slate-500 text-[10px] uppercase font-semibold shrink-0">Quick Shortcuts:</span>
                    <button
                      type="button"
                      onClick={() => fetchDiskDirs('')}
                      className="px-2 py-0.5 bg-slate-800/80 hover:bg-slate-800 text-slate-300 rounded border border-slate-750 cursor-pointer"
                    >
                      Project Root
                    </button>
                    <button
                      type="button"
                      onClick={() => fetchDiskDirs('/')}
                      className="px-2 py-0.5 bg-slate-800/80 hover:bg-slate-800 text-slate-300 rounded border border-slate-750 cursor-pointer"
                    >
                      Root ( / )
                    </button>
                    <button
                      type="button"
                      onClick={() => fetchDiskDirs('/tmp')}
                      className="px-2 py-0.5 bg-slate-800/80 hover:bg-slate-800 text-slate-300 rounded border border-slate-750 cursor-pointer"
                    >
                      /tmp
                    </button>
                  </div>

                  {'showDirectoryPicker' in window && (
                    <button
                      type="button"
                      onClick={handleNativeOSFolderPicker}
                      className="px-2.5 py-1 bg-indigo-950/60 hover:bg-indigo-900/60 text-indigo-300 text-[11px] font-semibold rounded-md border border-indigo-500/40 flex items-center gap-1.5 transition-colors cursor-pointer"
                      title="Open native platform directory selector dialog"
                    >
                      <FolderTree className="w-3.5 h-3.5 text-indigo-400" />
                      <span>OS Dialog Picker</span>
                    </button>
                  )}
                </div>
              </div>

              {/* Error Banner */}
              {diskDirError && (
                <div className="px-4 py-2 bg-rose-500/10 border-b border-rose-500/20 text-rose-300 text-xs font-mono flex items-center justify-between">
                  <span>⚠️ {diskDirError}</span>
                  <button type="button" onClick={() => setDiskDirError(null)} className="text-rose-400 hover:text-white">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}

              {/* Directory Browser Subfolders */}
              <div className="p-4 overflow-y-auto flex-1 space-y-2">
                <div className="flex items-center justify-between text-xs font-mono pb-2 border-b border-slate-800">
                  <div className="flex items-center gap-2 truncate text-slate-300">
                    <span className="text-slate-500 text-[10px] uppercase font-bold shrink-0">Browsing:</span>
                    <span className="font-semibold text-indigo-300 truncate bg-slate-950 px-2 py-0.5 rounded border border-slate-800">
                      {currentBrowseDiskPath || '/'}
                    </span>
                  </div>

                  {parentDiskPath && parentDiskPath !== currentBrowseDiskPath && (
                    <button
                      type="button"
                      onClick={() => fetchDiskDirs(parentDiskPath)}
                      className="px-2.5 py-1 bg-slate-800 hover:bg-slate-750 text-indigo-300 text-xs rounded border border-slate-700 flex items-center gap-1 cursor-pointer shrink-0 transition-colors"
                    >
                      <ArrowUp className="w-3.5 h-3.5" />
                      <span>.. Up One Level</span>
                    </button>
                  )}
                </div>

                {isLoadingDiskDirs ? (
                  <div className="py-12 text-center text-slate-500 text-xs font-mono flex items-center justify-center gap-2">
                    <div className="w-4 h-4 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                    <span>Loading local disk directories...</span>
                  </div>
                ) : diskSubdirs.length === 0 ? (
                  <div className="py-10 text-center text-slate-500 text-xs font-mono">
                    No subdirectories found in this folder.
                  </div>
                ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {diskSubdirs.map((dirName) => {
                      const fullSubPath = currentBrowseDiskPath === '/' ? `/${dirName}` : `${currentBrowseDiskPath}/${dirName}`;
                      return (
                        <div
                          key={dirName}
                          onClick={() => fetchDiskDirs(fullSubPath)}
                          className="p-2.5 bg-slate-950/60 hover:bg-slate-850/80 border border-slate-800 hover:border-indigo-500/50 rounded-lg flex items-center justify-between cursor-pointer transition-all group"
                        >
                          <div className="flex items-center gap-2 min-w-0">
                            <Folder className="w-4 h-4 text-amber-400 group-hover:text-amber-300 shrink-0" />
                            <span className="text-xs font-mono text-slate-200 group-hover:text-indigo-200 truncate">
                              {dirName}
                            </span>
                          </div>
                          <ChevronRight className="w-3.5 h-3.5 text-slate-600 group-hover:text-indigo-400 shrink-0" />
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Modal Footer */}
              <div className="p-4 bg-slate-950/80 border-t border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3 shrink-0">
                <div className="text-[11px] text-slate-400 font-mono flex items-center gap-1.5 min-w-0">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                  <span className="truncate">Active Root will be updated to: <strong className="text-indigo-300">{targetDiskPathInput}</strong></span>
                </div>

                <div className="flex items-center gap-2 justify-end shrink-0">
                  <button
                    type="button"
                    onClick={() => setShowFolderSelectorModal(false)}
                    className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-800 hover:bg-slate-750 rounded-lg transition-all cursor-pointer font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    disabled={isApplyingRootChange || !targetDiskPathInput.trim()}
                    onClick={() => handleApplyRootWorkspacePath()}
                    className="px-4 py-1.5 text-xs text-white bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-lg shadow-md transition-all cursor-pointer font-semibold flex items-center gap-1.5"
                  >
                    {isApplyingRootChange ? (
                      <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Check className="w-3.5 h-3.5" />
                    )}
                    <span>Set as Workspace Root</span>
                  </button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
