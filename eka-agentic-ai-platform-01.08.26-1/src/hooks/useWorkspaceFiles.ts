/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useCallback } from 'react';
import { FileNode } from '../types';

export function useWorkspaceFiles() {
  const [files, setFiles] = useState<FileNode[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [selectedFolder, setSelectedFolder] = useState<string | null>('');
  const [selectedFileContent, setSelectedFileContent] = useState<string>('');
  const [isSaving, setIsSaving] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [serverRootPath, setServerRootPath] = useState<string>('');
  const [localDirName, setLocalDirName] = useState<string>('');
  const [localDirHandle, setLocalDirHandle] = useState<any>(null);

  const refreshWorkspace = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const response = await fetch('/api/workspace');
      const data = await response.json();
      if (data.success) {
        setFiles(data.files || []);
      }
    } catch (err) {
      console.error('Error refreshing files:', err);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  const loadFileContent = useCallback(async (path: string) => {
    try {
      const response = await fetch(`/api/workspace/file?path=${encodeURIComponent(path)}`);
      const data = await response.json();
      if (data.success) {
        setSelectedFileContent(data.content);
        setSelectedFile(path);
        setSelectedFolder(null);
        return data.content;
      }
    } catch (err) {
      console.error('Error loading file content:', err);
    }
    return '';
  }, []);

  const handleSaveFile = async (path: string, content: string) => {
    setIsSaving(true);
    try {
      const response = await fetch('/api/workspace/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, content }),
      });
      const data = await response.json();
      if (data.success) {
        setSelectedFileContent(content);
      }
    } catch (err) {
      console.error('Error saving file:', err);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCreateFile = async (parentPath: string, name: string) => {
    const fullPath = parentPath ? `${parentPath}/${name}` : name;
    try {
      await fetch('/api/workspace/file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: fullPath, content: '# Start coding here...' }),
      });
      await refreshWorkspace();
      await loadFileContent(fullPath);
    } catch (err) {
      console.error('Error creating file:', err);
    }
  };

  const handleCreateFolder = async (parentPath: string, name: string) => {
    const fullPath = parentPath ? `${parentPath}/${name}` : name;
    try {
      await fetch('/api/workspace/folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: fullPath }),
      });
      await refreshWorkspace();
    } catch (err) {
      console.error('Error creating folder:', err);
    }
  };

  const handleDeleteFile = async (path: string) => {
    if (path === 'source-code' || path.startsWith('source-code/')) {
      alert('This item is part of the locked source-code group and cannot be deleted.');
      return;
    }
    try {
      await fetch(`/api/workspace/file?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
      if (selectedFile === path) {
        setSelectedFile(null);
        setSelectedFileContent('');
      }
      await refreshWorkspace();
    } catch (err) {
      console.error('Error deleting file:', err);
    }
  };

  const handleDeleteFolder = async (path: string) => {
    if (path === 'source-code' || path.startsWith('source-code/')) {
      alert('This folder is part of the locked source-code group and cannot be deleted.');
      return;
    }
    try {
      await fetch(`/api/workspace/folder?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
      if (selectedFolder === path || (selectedFolder && selectedFolder.startsWith(path + '/'))) {
        setSelectedFolder('');
      }
      await refreshWorkspace();
    } catch (err) {
      console.error('Error deleting folder:', err);
    }
  };

  const handleSetServerRootPath = async (newPath: string) => {
    try {
      const response = await fetch('/api/workspace/root-path', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rootPath: newPath }),
      });
      const data = await response.json();
      if (data.success) {
        setServerRootPath(data.rootPath);
        setLocalDirHandle(null);
        setLocalDirName('');
        await refreshWorkspace();
        return true;
      }
    } catch (err) {
      console.error('Error setting server root path:', err);
    }
    return false;
  };

  useEffect(() => {
    refreshWorkspace();
    fetch('/api/workspace/root-path')
      .then(r => r.json())
      .then(d => { if (d.success) setServerRootPath(d.rootPath); })
      .catch(() => {});
  }, [refreshWorkspace]);

  return {
    files,
    selectedFile,
    setSelectedFile,
    selectedFolder,
    setSelectedFolder,
    selectedFileContent,
    setSelectedFileContent,
    isSaving,
    isRefreshing,
    serverRootPath,
    localDirName,
    localDirHandle,
    refreshWorkspace,
    loadFileContent,
    handleSaveFile,
    handleCreateFile,
    handleCreateFolder,
    handleDeleteFile,
    handleDeleteFolder,
    handleSetServerRootPath,
  };
}
