/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { WFNode } from '../../types';
import { X, Check } from 'lucide-react';

interface NodeConfigModalProps {
  node: WFNode;
  onUpdateConfig: (updatedConfig: any) => void;
  onClose: () => void;
}

export function NodeConfigModal({ node, onUpdateConfig, onClose }: NodeConfigModalProps) {
  const cfg: any = node.config || {};

  const handleChange = (key: string, value: any) => {
    onUpdateConfig({ ...cfg, [key]: value });
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg shadow-2xl p-4 space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="text-xs font-bold font-mono text-slate-200">
            Configure Node: <span className="text-indigo-400">{node.label}</span> ({node.type})
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white rounded cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          {/* Decision Node specific configuration */}
          {node.type === 'decision' && (
            <div className="space-y-3">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Question / Decision Condition Prompt:</label>
                <textarea
                  value={cfg.questionPrompt || ''}
                  onChange={(e) => handleChange('questionPrompt', e.target.value)}
                  placeholder="e.g. Does the previous step code pass all test assertions?"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 focus:outline-none focus:border-amber-500 font-mono h-20 resize-none"
                />
              </div>

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Evaluation Method:</label>
                <select
                  value={cfg.evalType || 'llm_boolean'}
                  onChange={(e) => handleChange('evalType', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                >
                  <option value="llm_boolean">LLM Evaluator (Ask selected model YES/NO)</option>
                  <option value="contains_text">Contains Keyword Text Match</option>
                  <option value="js_expression font-mono">JavaScript Condition Expression</option>
                </select>
              </div>

              {cfg.evalType === 'contains_text' && (
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Expected Match Value:</label>
                  <input
                    type="text"
                    value={cfg.expectedValue || 'PASS'}
                    onChange={(e) => handleChange('expectedValue', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  />
                </div>
              )}

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs font-mono text-emerald-400 block mb-1">YES Branch Label:</label>
                  <input
                    type="text"
                    value={cfg.yesLabel || 'Continue Pipeline'}
                    onChange={(e) => handleChange('yesLabel', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  />
                </div>
                <div>
                  <label className="text-xs font-mono text-rose-400 block mb-1">NO Branch Label:</label>
                  <input
                    type="text"
                    value={cfg.noLabel || 'Stop / Reject'}
                    onChange={(e) => handleChange('noLabel', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Scheduler Node Config */}
          {node.type === 'scheduler' && (
            <div className="space-y-3">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Schedule Name / Job Title:</label>
                <input
                  type="text"
                  value={cfg.jobName || ''}
                  onChange={(e) => handleChange('jobName', e.target.value)}
                  placeholder="e.g. Daily Code Executor & Doc Launcher"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                />
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Scheduler Server:</label>
                  <select
                    value={cfg.schedulerServer || 'temporal'}
                    onChange={(e) => handleChange('schedulerServer', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  >
                    <option value="temporal">Temporal.io Server</option>
                    <option value="trigger_dev">Trigger.dev Cloud/Self-Hosted</option>
                    <option value="embedded">Built-in Scheduler Engine</option>
                  </select>
                </div>

                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Schedule Type:</label>
                  <select
                    value={cfg.scheduleType || 'cron'}
                    onChange={(e) => handleChange('scheduleType', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                  >
                    <option value="cron">Cron Expression Syntax</option>
                    <option value="interval">Repeat Interval (Seconds)</option>
                    <option value="one_shot">One-Shot Delayed Execution</option>
                  </select>
                </div>
              </div>

              {(cfg.scheduleType || 'cron') === 'cron' ? (
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Cron Expression (e.g. */5 * * * * or 0 9 * * 1-5):</label>
                  <input
                    type="text"
                    value={cfg.cronExpression || '*/5 * * * *'}
                    onChange={(e) => handleChange('cronExpression', e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-indigo-300 font-mono"
                  />
                  <p className="text-[10px] text-slate-500 mt-1 font-mono">
                    Standard 5-field cron: min hour day month weekday
                  </p>
                </div>
              ) : (
                <div>
                  <label className="text-xs font-mono text-slate-300 block mb-1">Interval Seconds:</label>
                  <input
                    type="number"
                    value={cfg.intervalSeconds || 300}
                    onChange={(e) => handleChange('intervalSeconds', Number(e.target.value))}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-indigo-300 font-mono"
                  />
                </div>
              )}

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Action Classifier & Handler:</label>
                <select
                  value={cfg.actionType || 'auto_detect'}
                  onChange={(e) => handleChange('actionType', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                >
                  <option value="auto_detect">Auto-Detect (Code &rarr; Exec; .exe/.py &rarr; Exec; .docx/.pdf &rarr; Open File)</option>
                  <option value="code_execution">Force Code Execution (Python / Node / Bash)</option>
                  <option value="file_action">Force File Action (Execute process or Open document)</option>
                  <option value="pipeline_workflow">Trigger Pipeline Workflow</option>
                </select>
              </div>

              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Target Payload / Code / Local File Path (Leave blank to inherit from upstream node):</label>
                <textarea
                  value={cfg.targetPayload || ''}
                  onChange={(e) => handleChange('targetPayload', e.target.value)}
                  placeholder="e.g. print('Hello Temporal!') OR /path/to/script.py OR report.docx"
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono h-24 resize-none"
                />
              </div>
            </div>
          )}

          {/* Input Node Config */}
          {node.type === 'input' && (
            <div>
              <label className="text-xs font-mono text-slate-300 block mb-1">Input Text Content:</label>
              <textarea
                value={cfg.inputText || ''}
                onChange={(e) => handleChange('inputText', e.target.value)}
                placeholder="Enter baseline prompt text..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono h-24 resize-none"
              />
            </div>
          )}

          {/* LLM Node Config */}
          {node.type === 'llm' && (
            <div className="space-y-3">
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">Prompt Template (Use {"{{input}}"} or {"{{context}}"}):</label>
                <textarea
                  value={cfg.prompt || '{{input}}'}
                  onChange={(e) => handleChange('prompt', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono h-24 resize-none"
                />
              </div>
              <div>
                <label className="text-xs font-mono text-slate-300 block mb-1">System Instruction:</label>
                <input
                  type="text"
                  value={cfg.systemInstruction || ''}
                  onChange={(e) => handleChange('systemInstruction', e.target.value)}
                  placeholder="e.g. You are a senior QA engineer."
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
                />
              </div>
            </div>
          )}

          {/* Default Label Input for all nodes */}
          <div>
            <label className="text-xs font-mono text-slate-300 block mb-1">Custom Node Label:</label>
            <input
              type="text"
              value={node.label}
              onChange={(e) => handleChange('nodeLabel', e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 font-mono"
            />
          </div>
        </div>

        <div className="pt-2 border-t border-slate-800 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="bg-indigo-600 hover:bg-indigo-500 text-white font-mono text-xs font-bold px-4 py-2 rounded-lg transition flex items-center gap-1.5 cursor-pointer shadow-md"
          >
            <Check className="w-4 h-4" /> Save Node Configuration
          </button>
        </div>
      </div>
    </div>
  );
}
