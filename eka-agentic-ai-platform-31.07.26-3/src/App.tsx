/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { AgentMode, ChatMessage, FileAttachment, AgentWorkflow } from './types';
import WorkspaceHeaderBar, { WORKSPACE_OPTIONS, WorkspaceOption } from './components/WorkspaceHeaderBar';
import WorkspaceTabs from './components/WorkspaceTabs';
import ChatPanel from './components/ChatPanel';
import ModelSelector from './components/ModelSelector';
import DirectoryViewer from './components/DirectoryViewer';
import { useAppConfig } from './hooks/useAppConfig';
import { useWorkspaceFiles } from './hooks/useWorkspaceFiles';
import { Folder, Bot, PanelLeftOpen, PanelRightOpen, HardDrive, MessageSquare } from 'lucide-react';

export function App() {
  const { theme, setTheme, accentColor, setAccentColor, modelConfig, setModelConfig } = useAppConfig();
  const workspace = useWorkspaceFiles();

  const [activeWorkspaceOption, setActiveWorkspaceOption] = useState<string>('code');
  const [activeTab, setActiveTab] = useState<'flow' | 'editor' | 'diagnostics' | 'docs' | 'pipeline' | 'multiagent' | 'git' | 'self_heal' | 'time_travel' | 'wireframe'>('editor');
  const [activeMode, setActiveMode] = useState<AgentMode>('code');
  const [multimodalFeature, setMultimodalFeature] = useState<string>('text_to_image');

  const [isLeftPanelCollapsed, setIsLeftPanelCollapsed] = useState(false);
  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState(false);
  const [leftPanelWidth, setLeftPanelWidth] = useState(240);
  const [rightPanelWidth, setRightPanelWidth] = useState(380);
  const [isDraggingLeft, setIsDraggingLeft] = useState(false);
  const [isDraggingRight, setIsDraggingRight] = useState(false);

  const [workflow, setWorkflow] = useState<AgentWorkflow | null>(null);

  const handleLeftMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingLeft(true);
  };

  const handleRightMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDraggingRight(true);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDraggingLeft) {
        const newWidth = Math.max(160, Math.min(500, e.clientX));
        setLeftPanelWidth(newWidth);
      }
      if (isDraggingRight) {
        const newWidth = Math.max(260, Math.min(700, window.innerWidth - e.clientX));
        setRightPanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      setIsDraggingLeft(false);
      setIsDraggingRight(false);
    };

    if (isDraggingLeft || isDraggingRight) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingLeft, isDraggingRight]);

  const [messagesByMode, setMessagesByMode] = useState<Record<AgentMode, ChatMessage[]>>({
    chat: [], code: [], research: [], multimodal: [], documentation: [], testing: [],
  });

  const [isExecuting, setIsExecuting] = useState(false);
  const [searchEnabled, setSearchEnabled] = useState(true);
  const [isModelSelectorOpen, setIsModelSelectorOpen] = useState(false);

  // Single Workspace Navigation Handler
  const handleSelectWorkspaceOption = (opt: WorkspaceOption) => {
    setActiveWorkspaceOption(opt.id);
    setActiveMode(opt.mode);
    if (opt.workspaceTab) setActiveTab(opt.workspaceTab);
    if (opt.multimodalFeature) setMultimodalFeature(opt.multimodalFeature);
  };

  const handleOpenFile = async (path: string) => {
    await workspace.loadFileContent(path);
    if (path.endsWith('.md') || path.endsWith('.txt') || path.endsWith('.doc')) {
      setActiveTab('docs');
    } else {
      setActiveTab('editor');
    }
  };

  const handleSendMessage = async (text: string, mode: AgentMode, search: boolean, attachments: FileAttachment[] = []) => {
    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString(),
      attachments,
    };

    setMessagesByMode(prev => ({
      ...prev,
      [mode]: [...(prev[mode] || []), userMsg],
    }));

    setIsExecuting(true);

    if (mode === 'code') {
      setWorkflow({
        taskId: `task-${Date.now()}`,
        prompt: text,
        mode: 'code',
        status: 'executing',
        plan: [
          { id: 'step-1', title: 'Analyze Request & Requirements', description: text, status: 'completed', type: 'edit', target: 'Workspace', approvalRequired: false },
          { id: 'step-2', title: 'Synthesize & Apply Code Changes', description: 'Generating and updating workspace files', status: 'running', type: 'edit', target: 'Source Code', approvalRequired: false },
          { id: 'step-3', title: 'Verify Compilation & Diagnostics', description: 'Running diagnostics check', status: 'pending', type: 'test', target: 'Linter', approvalRequired: false },
        ],
        currentStepIndex: 1,
        logs: ['Task initialized', 'Analyzing prompt requirements...'],
        thinkingLines: ['Parsing user instructions', 'Connecting to model server...'],
      });
    }

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...(messagesByMode[mode] || []), userMsg],
          attachments,
          searchEnabled: search,
          customConfig: modelConfig,
          mode,
        }),
      });

      const data = await res.json();
      const assistantMsg: ChatMessage = {
        id: `msg-${Date.now() + 1}`,
        role: 'assistant',
        content: data.reply || 'No output received.',
        timestamp: new Date().toLocaleTimeString(),
        citations: data.citations || [],
      };

      setMessagesByMode(prev => ({
        ...prev,
        [mode]: [...(prev[mode] || []), assistantMsg],
      }));

      if (mode === 'code') {
        setWorkflow(prev => prev ? {
          ...prev,
          status: 'completed',
          currentStepIndex: 2,
          plan: prev.plan.map(s => ({ ...s, status: 'completed' as const })),
        } : null);
      }
    } catch (err: any) {
      const errorMsg: ChatMessage = {
        id: `msg-${Date.now() + 2}`,
        role: 'assistant',
        content: `⚠️ Error during execution: ${err.message}`,
        timestamp: new Date().toLocaleTimeString(),
      };
      setMessagesByMode(prev => ({
        ...prev,
        [mode]: [...(prev[mode] || []), errorMsg],
      }));

      if (mode === 'code') {
        setWorkflow(prev => prev ? { ...prev, status: 'failed' } : null);
      }
    } finally {
      setIsExecuting(false);
    }
  };

  const handleApproveStep = (stepId: string, approved: boolean) => {
    if (!workflow) return;
    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        status: approved ? 'executing' : 'failed',
        plan: prev.plan.map(s => s.id === stepId ? { ...s, status: approved ? 'completed' : 'failed' } : s),
      };
    });
  };

  const handleApprovePlan = () => {
    if (!workflow) return;
    setWorkflow(prev => prev ? { ...prev, status: 'executing' } : null);
  };

  const handleRejectPlan = () => {
    if (!workflow) return;
    setWorkflow(prev => prev ? { ...prev, status: 'failed' } : null);
  };

  const handleStopWorkflow = () => {
    setIsExecuting(false);
    setWorkflow(prev => prev ? { ...prev, status: 'failed' } : null);
  };

  const handleStopStep = (stepId: string) => {
    setWorkflow(prev => {
      if (!prev) return null;
      return {
        ...prev,
        plan: prev.plan.map(s => s.id === stepId ? { ...s, status: 'failed' } : s),
      };
    });
  };

  const handleStopExecution = () => {
    setIsExecuting(false);
  };

  const handleClearMemory = (mode: AgentMode) => {
    setMessagesByMode(prev => ({ ...prev, [mode]: [] }));
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    document.documentElement.setAttribute('data-accent', accentColor);
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme, accentColor]);

  const containerThemeClass = theme === 'white' 
    ? 'bg-slate-100 text-slate-900' 
    : theme === 'light-grey' 
      ? 'bg-zinc-200 text-zinc-900' 
      : 'bg-slate-950 text-slate-100';

  return (
    <div className={`h-screen w-screen flex flex-col font-sans overflow-hidden ${containerThemeClass}`}>
      {/* Master Workspace Selector Bar */}
      <WorkspaceHeaderBar
        activeOptionId={activeWorkspaceOption}
        onSelectOption={handleSelectWorkspaceOption}
        modelConfig={modelConfig}
        theme={theme}
        setTheme={setTheme}
        accentColor={accentColor}
        setAccentColor={setAccentColor}
        onOpenModelSelector={() => setIsModelSelectorOpen(true)}
        isLeftPanelCollapsed={isLeftPanelCollapsed}
        onToggleLeftPanel={() => setIsLeftPanelCollapsed(prev => !prev)}
        isRightPanelCollapsed={isRightPanelCollapsed}
        onToggleRightPanel={() => setIsRightPanelCollapsed(prev => !prev)}
      />

      {/* Main Studio Viewport */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Left File Explorer Sidebar */}
        {isLeftPanelCollapsed ? (
          <aside className="w-11 border-r border-slate-800 bg-slate-900/90 shrink-0 flex flex-col items-center py-3 gap-4">
            <button
              type="button"
              onClick={() => setIsLeftPanelCollapsed(false)}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition cursor-pointer"
              title="Expand File Explorer Sidebar"
            >
              <PanelLeftOpen className="w-4 h-4 theme-accent-text" />
            </button>
            <div className="w-px h-4 bg-slate-800" />
            <button
              type="button"
              onClick={() => setIsLeftPanelCollapsed(false)}
              className="p-1.5 text-slate-500 hover:text-slate-300 transition cursor-pointer flex flex-col items-center gap-1"
              title="Click to view workspace files"
            >
              <HardDrive className="w-4 h-4 theme-accent-text" />
              <span className="text-[9px] font-mono text-slate-500 uppercase rotate-90 origin-center mt-6 tracking-wider">
                Files
              </span>
            </button>
          </aside>
        ) : (
          <aside
            style={{ width: `${leftPanelWidth}px` }}
            className="border-r border-slate-800 bg-slate-900/80 shrink-0 flex flex-col h-full overflow-hidden"
          >
            <DirectoryViewer
              folderPath={workspace.selectedFolder || ''}
              files={workspace.files}
              serverRootPath={workspace.serverRootPath}
              onSetServerRootPath={workspace.handleSetServerRootPath}
              onSelectFile={handleOpenFile}
              onSelectFolder={workspace.setSelectedFolder}
              onCreateFile={workspace.handleCreateFile}
              onCreateFolder={workspace.handleCreateFolder}
              onDeleteFile={workspace.handleDeleteFile}
              onDeleteFolder={workspace.handleDeleteFolder}
              onToggleMinimize={() => setIsLeftPanelCollapsed(true)}
            />
          </aside>
        )}

        {/* Left Panel Resizer Handle */}
        {!isLeftPanelCollapsed && (
          <div
            onMouseDown={handleLeftMouseDown}
            className={`w-1 hover:w-1.5 cursor-col-resize bg-slate-800 hover:bg-indigo-500 active:bg-indigo-600 transition-colors z-30 shrink-0 select-none ${
              isDraggingLeft ? 'bg-indigo-500' : ''
            }`}
            title="Drag to resize Left File Panel"
          />
        )}

        {/* Center Active Workspace View */}
        <main className="flex-1 min-w-0 bg-slate-950 flex flex-col h-full overflow-hidden">
          <WorkspaceTabs
            activeTab={activeTab}
            setActiveTab={setActiveTab}
            selectedFile={workspace.selectedFile}
            selectedFileContent={workspace.selectedFileContent}
            onSaveFile={workspace.handleSaveFile}
            onCloseFile={() => workspace.setSelectedFile(null)}
            modelConfig={modelConfig}
            files={workspace.files}
            onRefreshWorkspace={workspace.refreshWorkspace}
            workflow={workflow}
            onApproveStep={handleApproveStep}
            onApprovePlan={handleApprovePlan}
            onRejectPlan={handleRejectPlan}
            onStopWorkflow={handleStopWorkflow}
            onStopStep={handleStopStep}
            isExecuting={isExecuting}
          />
        </main>

        {/* Right Panel Resizer Handle */}
        {!isRightPanelCollapsed && (
          <div
            onMouseDown={handleRightMouseDown}
            className={`w-1 hover:w-1.5 cursor-col-resize bg-slate-800 hover:bg-indigo-500 active:bg-indigo-600 transition-colors z-30 shrink-0 select-none ${
              isDraggingRight ? 'bg-indigo-500' : ''
            }`}
            title="Drag to resize Right Chat Panel"
          />
        )}

        {/* Right Active AI Chat / Multimodal Panel */}
        {isRightPanelCollapsed ? (
          <section className="w-11 border-l border-slate-800 bg-slate-950 shrink-0 flex flex-col items-center py-3 gap-4">
            <button
              type="button"
              onClick={() => setIsRightPanelCollapsed(false)}
              className="p-1.5 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition cursor-pointer"
              title="Expand AI Assistant Panel"
            >
              <PanelRightOpen className="w-4 h-4 theme-accent-text" />
            </button>
            <div className="w-px h-4 bg-slate-800" />
            <button
              type="button"
              onClick={() => setIsRightPanelCollapsed(false)}
              className="p-1.5 text-slate-500 hover:text-slate-300 transition cursor-pointer flex flex-col items-center gap-1"
              title="Click to view AI Chat"
            >
              <Bot className="w-4 h-4 theme-accent-text" />
              <span className="text-[9px] font-mono text-slate-500 uppercase -rotate-90 origin-center mt-6 tracking-wider">
                Assistant
              </span>
            </button>
          </section>
        ) : (
          <section
            style={{ width: `${rightPanelWidth}px` }}
            className="border-l border-slate-800 bg-slate-950 shrink-0 flex flex-col h-full overflow-hidden"
          >
            <ChatPanel
              activeMode={activeMode}
              messages={messagesByMode[activeMode] || []}
              workflow={workflow}
              isExecuting={isExecuting}
              onSendMessage={handleSendMessage}
              onStopExecution={handleStopExecution}
              onStopStep={handleStopStep}
              onClearMemory={handleClearMemory}
              searchEnabled={searchEnabled}
              setSearchEnabled={setSearchEnabled}
              modelConfig={modelConfig}
              multimodalFeature={multimodalFeature}
              onToggleMinimize={() => setIsRightPanelCollapsed(true)}
            />
          </section>
        )}
      </div>

      {/* Settings Modal */}
      {isModelSelectorOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl p-2">
            <button
              type="button"
              onClick={() => setIsModelSelectorOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white text-xs font-mono bg-slate-800 px-2.5 py-1 rounded-lg border border-slate-700 cursor-pointer"
            >
              ✕ Close Settings
            </button>
            <ModelSelector config={modelConfig} onChange={setModelConfig} />
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
