/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { ModelServerConfig, AgentWorkflow, FileNode } from '../types';
import CodeViewer from './CodeViewer';
import WorkflowBuilder from './WorkflowBuilder';
import AgentFlowGraph from './AgentFlowGraph';
import { SelfHealingTab } from './workspace/SelfHealingTab';
import { WireframeUmlTab } from './workspace/WireframeUmlTab';
import { GitControlTab } from './workspace/GitControlTab';
import { TimeTravelTab } from './workspace/TimeTravelTab';
import { DocsStudioTab } from './workspace/DocsStudioTab';
import { DiagnosticsTab } from './workspace/DiagnosticsTab';
import MultiAgentWorkspace from './multi_agent/MultiAgentWorkspace';
import { 
  Code, 
  Workflow, 
  Layers, 
  FileText, 
  CheckCircle, 
  GitBranch, 
  ShieldAlert, 
  History,
  Box,
  ChevronDown,
  Wrench,
  Bot
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export type WorkspaceTabId = 'flow' | 'editor' | 'diagnostics' | 'docs' | 'pipeline' | 'multiagent' | 'git' | 'self_heal' | 'time_travel' | 'wireframe';

interface WorkspaceTabsProps {
  activeTab: WorkspaceTabId;
  setActiveTab: (tab: WorkspaceTabId) => void;
  selectedFile: string | null;
  selectedFileContent: string;
  onSaveFile: (path: string, content: string) => void;
  onCloseFile?: () => void;
  modelConfig?: ModelServerConfig;
  files: FileNode[];
  onRefreshWorkspace?: () => void;
  workflow?: AgentWorkflow | null;
  onApproveStep?: (stepId: string, approved: boolean) => void;
  onApprovePlan?: () => void;
  onRejectPlan?: () => void;
  onStopWorkflow?: () => void;
  onStopStep?: (stepId: string) => void;
  isExecuting?: boolean;
}

export default function WorkspaceTabs({
  activeTab,
  setActiveTab,
  selectedFile,
  selectedFileContent,
  onSaveFile,
  onCloseFile,
  modelConfig,
  files,
  onRefreshWorkspace,
  workflow = null,
  onApproveStep,
  onApprovePlan,
  onRejectPlan,
  onStopWorkflow,
  onStopStep,
  isExecuting = false,
}: WorkspaceTabsProps) {
  const [docContent] = useState<string>(
    '# Technical Documentation & Architecture Guide\n\nWelcome to Eka Studio. You can generate API specs, system diagrams, and architectural documentation.'
  );

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Primary tabs pinned on the tab bar
  const primaryTabs = [
    { id: 'editor', label: 'Code Editor', icon: <Code className="w-3.5 h-3.5 text-emerald-400" /> },
    { id: 'flow', label: 'Agent Flow', icon: <Workflow className="w-3.5 h-3.5 text-purple-400" /> },
    { id: 'docs', label: 'Docs Studio', icon: <FileText className="w-3.5 h-3.5 text-yellow-400" /> },
  ] as const;

  // Secondary tools available in top dropdown menu
  const secondaryTools = [
    { id: 'multiagent', label: 'Multi-Agent System', icon: <Bot className="w-3.5 h-3.5 text-cyan-400" /> },
    { id: 'wireframe', label: 'Wireframes & UML', icon: <Layers className="w-3.5 h-3.5 text-indigo-400" /> },
    { id: 'diagnostics', label: 'Diagnostics', icon: <CheckCircle className="w-3.5 h-3.5 text-green-400" /> },
    { id: 'pipeline', label: 'Workflow Builder', icon: <Box className="w-3.5 h-3.5 text-amber-400" /> },
    { id: 'git', label: 'Git Control', icon: <GitBranch className="w-3.5 h-3.5 text-orange-400" /> },
    { id: 'self_heal', label: 'Self-Healing', icon: <ShieldAlert className="w-3.5 h-3.5 text-rose-400" /> },
    { id: 'time_travel', label: 'Time Travel', icon: <History className="w-3.5 h-3.5 text-sky-400" /> },
  ] as const;

  const activeSecondaryTool = secondaryTools.find(t => t.id === activeTab);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-950 overflow-hidden">
      {/* Top Workspace Tab Strip */}
      <div className="h-9 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between px-2 gap-2 shrink-0 font-mono text-xs select-none relative z-30">
        {/* Primary Tabs (Horizontally Scrollable if needed) */}
        <div className="flex items-center gap-1 overflow-x-auto min-w-0 flex-1 scrollbar-none py-1">
          {primaryTabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id as WorkspaceTabId)}
                className={`px-2.5 sm:px-3 py-1 rounded-md flex items-center gap-1.5 transition text-[11px] whitespace-nowrap cursor-pointer shrink-0 ${
                  isActive
                    ? 'bg-slate-800 text-white font-bold border border-slate-700 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                {tab.icon}
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Studio Tools Dropdown (Pinned to Right, Overflow Visible) */}
        <div className="relative shrink-0 z-50" ref={dropdownRef}>
          <button
            type="button"
            onClick={() => setIsDropdownOpen(!isDropdownOpen)}
            className={`px-2.5 py-1 rounded-md flex items-center gap-1.5 transition text-[11px] font-mono cursor-pointer border ${
              activeSecondaryTool
                ? 'bg-indigo-950/80 text-indigo-300 border-indigo-500 font-bold shadow-sm'
                : 'bg-slate-800/60 text-slate-300 border-slate-700 hover:bg-slate-800 hover:text-white'
            }`}
          >
            {activeSecondaryTool ? (
              <>
                {activeSecondaryTool.icon}
                <span className="max-w-[120px] truncate">{activeSecondaryTool.label}</span>
              </>
            ) : (
              <>
                <Wrench className="w-3.5 h-3.5 text-slate-400" />
                <span>More Tools</span>
              </>
            )}
            <ChevronDown className={`w-3 h-3 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
          </button>

          {/* Dropdown Menu */}
          {isDropdownOpen && (
            <div className="absolute right-0 top-full mt-1 w-56 bg-slate-900 border border-slate-800 rounded-xl shadow-2xl z-50 py-1.5 font-mono text-xs overflow-hidden backdrop-blur-lg">
              <div className="px-3 py-1 text-[10px] uppercase font-bold text-slate-500 border-b border-slate-800/80 mb-1">
                Workspace Utilities
              </div>
              {secondaryTools.map((tool) => {
                const isSelected = activeTab === tool.id || (activeTab === 'multiagent' && tool.id === 'pipeline');
                return (
                  <button
                    key={tool.id}
                    type="button"
                    onClick={() => {
                      setActiveTab(tool.id as WorkspaceTabId);
                      setIsDropdownOpen(false);
                    }}
                    className={`w-full text-left px-3 py-1.5 flex items-center gap-2 transition text-[11px] cursor-pointer ${
                      isSelected
                        ? 'bg-indigo-950/80 text-indigo-200 font-bold border-l-2 border-indigo-500'
                        : 'text-slate-300 hover:bg-slate-800 hover:text-white'
                    }`}
                  >
                    {tool.icon}
                    <span>{tool.label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Active Tab Viewport */}
      <div className="flex-1 overflow-hidden relative">
        {activeTab === 'flow' && (
          <div className="h-full p-4 overflow-y-auto">
            <AgentFlowGraph
              workflow={workflow}
              onApproveStep={onApproveStep}
              onApprovePlan={onApprovePlan}
              onRejectPlan={onRejectPlan}
              onStopWorkflow={onStopWorkflow}
              onStopStep={onStopStep}
              isExecuting={isExecuting}
              theme="dark"
            />
          </div>
        )}

        {activeTab === 'editor' && (
          <CodeViewer
            selectedFile={selectedFile}
            selectedFileContent={selectedFileContent}
            onSaveFile={onSaveFile}
            onCloseFile={onCloseFile}
            modelConfig={modelConfig}
          />
        )}

        {activeTab === 'wireframe' && (
          <WireframeUmlTab
            modelConfig={modelConfig}
            files={files}
            onRefreshWorkspace={onRefreshWorkspace}
          />
        )}

        {activeTab === 'multiagent' && (
          <MultiAgentWorkspace
            modelConfig={modelConfig || {
              type: 'gemini',
              baseUrl: '',
              apiKey: '',
              selectedModel: 'gemini-2.5-flash',
              isOnline: true,
              availableModels: ['gemini-2.5-flash']
            }}
            theme="dark"
          />
        )}

        {activeTab === 'pipeline' && (
          <WorkflowBuilder
            modelConfig={modelConfig}
            files={files}
            onRefreshWorkspace={onRefreshWorkspace}
          />
        )}

        {activeTab === 'self_heal' && (
          <SelfHealingTab
            modelConfig={modelConfig}
            onRefreshWorkspace={onRefreshWorkspace}
          />
        )}

        {activeTab === 'git' && <GitControlTab />}

        {activeTab === 'docs' && (
          <DocsStudioTab
            selectedFile={selectedFile}
            selectedFileContent={selectedFileContent}
            onSaveFile={onSaveFile}
            files={files}
            onSelectFile={(path) => {
              if (path) {
                // If opening in docs, keep active tab as docs
              }
            }}
          />
        )}

        {activeTab === 'diagnostics' && <DiagnosticsTab />}

        {activeTab === 'time_travel' && <TimeTravelTab />}
      </div>
    </div>
  );
}
