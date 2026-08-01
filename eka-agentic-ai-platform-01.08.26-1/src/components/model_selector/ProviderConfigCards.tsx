/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ModelServerConfig, ServerType, LocalLLMProvider, LocalLLMProviderConfig } from '../../types';
import { Server, Shield, Globe, Cpu } from 'lucide-react';

interface ProviderConfigCardsProps {
  config: ModelServerConfig;
  localConfigs: Record<LocalLLMProvider, LocalLLMProviderConfig>;
  activeLocalProvider: LocalLLMProvider;
  onSelectType: (type: ServerType, provider?: LocalLLMProvider) => void;
  onUpdateConfig: (updated: Partial<ModelServerConfig>) => void;
  onUpdateLocalConfig: (provider: LocalLLMProvider, updated: Partial<LocalLLMProviderConfig>) => void;
  onTestConnection: (type: ServerType, url: string, key: string, provider?: LocalLLMProvider) => void;
  loading: boolean;
}

export function ProviderConfigCards({
  config,
  localConfigs,
  activeLocalProvider,
  onSelectType,
  onUpdateConfig,
  onUpdateLocalConfig,
  onTestConnection,
  loading,
}: ProviderConfigCardsProps) {
  const currentLocal = localConfigs[activeLocalProvider];

  return (
    <div className="space-y-4">
      {/* Provider Selector Badges */}
      <div>
        <label className="text-xs font-mono font-bold text-slate-300 block mb-2">
          Select LLM Inference Server:
        </label>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {/* Gemini Cloud */}
          <button
            type="button"
            onClick={() => onSelectType('gemini')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'gemini'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-indigo-400">
              <Globe className="w-4 h-4" /> Google Gemini
            </div>
            <span className="text-[10px] text-slate-400 mt-1">Direct Gemini 3.6 Flash / Pro API</span>
          </button>

          {/* Ollama Local */}
          <button
            type="button"
            onClick={() => onSelectType('local_llm', 'ollama')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'local_llm' && activeLocalProvider === 'ollama'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-emerald-400">
              <Cpu className="w-4 h-4" /> Local Ollama
            </div>
            <span className="text-[10px] text-slate-400 mt-1">Native Ollama API server</span>
          </button>

          {/* vLLM Local */}
          <button
            type="button"
            onClick={() => onSelectType('local_llm', 'vllm')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'local_llm' && activeLocalProvider === 'vllm'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-purple-400">
              <Server className="w-4 h-4" /> vLLM Server
            </div>
            <span className="text-[10px] text-slate-400 mt-1">High-throughput OpenAI API</span>
          </button>

          {/* LM Studio Local */}
          <button
            type="button"
            onClick={() => onSelectType('local_llm', 'lmstudio')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'local_llm' && activeLocalProvider === 'lmstudio'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-amber-400">
              <Cpu className="w-4 h-4" /> LM Studio
            </div>
            <span className="text-[10px] text-slate-400 mt-1">Local LM Studio instance</span>
          </button>

          {/* OpenAI Official */}
          <button
            type="button"
            onClick={() => onSelectType('openai')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'openai'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-cyan-400">
              <Shield className="w-4 h-4" /> OpenAI Platform
            </div>
            <span className="text-[10px] text-slate-400 mt-1">GPT-4o & GPT-3.5 API</span>
          </button>

          {/* Custom Local */}
          <button
            type="button"
            onClick={() => onSelectType('local_llm', 'custom')}
            className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
              config.type === 'local_llm' && activeLocalProvider === 'custom'
                ? 'bg-indigo-950/60 border-indigo-500 text-white shadow-lg'
                : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:bg-slate-800/80'
            }`}
          >
            <div className="flex items-center gap-2 font-semibold text-xs text-rose-400">
              <Server className="w-4 h-4" /> Custom Local / v1
            </div>
            <span className="text-[10px] text-slate-400 mt-1">OpenAI format compatible</span>
          </button>
        </div>
      </div>

      {/* Inputs for selected provider */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3.5 space-y-3">
        {config.type === 'gemini' ? (
          <div>
            <label className="text-xs font-mono text-slate-300 block mb-1">Gemini API Key (Optional Override):</label>
            <input
              type="password"
              value={config.apiKey || ''}
              onChange={(e) => onUpdateConfig({ apiKey: e.target.value })}
              placeholder="Leave blank to use system GEMINI_API_KEY secret"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
            />
          </div>
        ) : (
          <>
            <div>
              <label className="text-xs font-mono text-slate-300 block mb-1">Base URL:</label>
              <input
                type="text"
                value={config.type === 'local_llm' ? currentLocal?.baseUrl || '' : config.baseUrl || ''}
                onChange={(e) => {
                  const val = e.target.value;
                  if (config.type === 'local_llm') {
                    onUpdateLocalConfig(activeLocalProvider, { baseUrl: val });
                  } else {
                    onUpdateConfig({ baseUrl: val });
                  }
                }}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>

            <div>
              <label className="text-xs font-mono text-slate-300 block mb-1">API Key (if required):</label>
              <input
                type="password"
                value={config.type === 'local_llm' ? currentLocal?.apiKey || '' : config.apiKey || ''}
                onChange={(e) => {
                  const val = e.target.value;
                  if (config.type === 'local_llm') {
                    onUpdateLocalConfig(activeLocalProvider, { apiKey: val });
                  } else {
                    onUpdateConfig({ apiKey: val });
                  }
                }}
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
              />
            </div>
          </>
        )}

        {/* Selected Model dropdown */}
        <div>
          <label className="text-xs font-mono text-slate-300 block mb-1">Active Model Name:</label>
          {((config.type === 'local_llm' ? currentLocal?.availableModels : config.availableModels) || []).length > 0 ? (
            <select
              value={config.type === 'local_llm' ? (currentLocal?.selectedModel || config.selectedModel) : config.selectedModel}
              onChange={(e) => {
                const val = e.target.value;
                if (config.type === 'local_llm') {
                  onUpdateLocalConfig(activeLocalProvider, { selectedModel: val });
                }
                onUpdateConfig({ selectedModel: val });
              }}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
            >
              {(config.type === 'local_llm' ? (currentLocal?.availableModels || []) : (config.availableModels || [])).map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={config.type === 'local_llm' ? (currentLocal?.selectedModel || config.selectedModel || '') : (config.selectedModel || '')}
              onChange={(e) => {
                const val = e.target.value;
                if (config.type === 'local_llm') {
                  onUpdateLocalConfig(activeLocalProvider, { selectedModel: val });
                }
                onUpdateConfig({ selectedModel: val });
              }}
              placeholder="e.g. llama3, mistral, gpt-4o"
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
            />
          )}
        </div>

        <button
          type="button"
          onClick={() => {
            const url = config.type === 'local_llm' ? currentLocal.baseUrl : config.baseUrl;
            const key = config.type === 'local_llm' ? currentLocal.apiKey : config.apiKey;
            onTestConnection(config.type, url, key, activeLocalProvider);
          }}
          disabled={loading}
          className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-mono text-xs py-1.5 rounded-lg transition cursor-pointer font-bold shadow-md"
        >
          {loading ? 'Checking Endpoint Connection...' : 'Test Endpoint & Load Available Models'}
        </button>
      </div>
    </div>
  );
}
