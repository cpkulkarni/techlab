/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig, ServerType, LocalLLMProvider, LocalLLMProviderConfig, DEFAULT_LOCAL_CONFIGS } from '../types';
import { CheckCircle, AlertCircle, Cpu, Globe, Mail } from 'lucide-react';
import { ProviderConfigCards } from './model_selector/ProviderConfigCards';
import { SearchAssistSettings } from './model_selector/SearchAssistSettings';
import { LocalMailServerSettings } from './model_selector/LocalMailServerSettings';

interface ModelSelectorProps {
  config: ModelServerConfig;
  onChange: (config: ModelServerConfig) => void;
  theme?: 'white' | 'light-grey' | 'dark';
  accentColor?: string;
}

export default function ModelSelector({ config, onChange }: ModelSelectorProps) {
  const [activeTab, setActiveTab] = useState<'models' | 'search' | 'mail'>('models');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showSuccess, setShowSuccess] = useState(false);

  const localConfigs: Record<LocalLLMProvider, LocalLLMProviderConfig> = {
    ...DEFAULT_LOCAL_CONFIGS,
    ...(config.localConfigs || {})
  };

  const activeLocalProvider: LocalLLMProvider = config.activeLocalProvider || 'ollama';

  const handleSelectType = (type: ServerType, provider?: LocalLLMProvider) => {
    const newLocalProv = provider || activeLocalProvider;
    const localConf = localConfigs[newLocalProv] || DEFAULT_LOCAL_CONFIGS[newLocalProv];

    onChange({
      ...config,
      type,
      activeLocalProvider: newLocalProv,
      baseUrl: type === 'local_llm' ? localConf.baseUrl : config.baseUrl,
      apiKey: type === 'local_llm' ? localConf.apiKey : config.apiKey,
      // Retain existing model name
      selectedModel: type === 'local_llm' ? (localConf.selectedModel || config.selectedModel) : config.selectedModel,
    });
  };

  const handleUpdateConfig = (updated: Partial<ModelServerConfig>) => {
    onChange({ ...config, ...updated });
  };

  const handleUpdateLocalConfig = (prov: LocalLLMProvider, updated: Partial<LocalLLMProviderConfig>) => {
    const updatedLocal = { ...localConfigs[prov], ...updated };
    const newLocalConfigs = { ...localConfigs, [prov]: updatedLocal };
    onChange({
      ...config,
      localConfigs: newLocalConfigs,
      ...(config.activeLocalProvider === prov ? {
        baseUrl: updatedLocal.baseUrl,
        apiKey: updatedLocal.apiKey,
        selectedModel: updatedLocal.selectedModel || config.selectedModel,
      } : {})
    });
  };

  const handleTestConnection = async (type: ServerType, url: string, key: string, provider?: LocalLLMProvider) => {
    setLoading(true);
    setErrorMsg(null);
    setShowSuccess(false);

    const currentProv = provider || activeLocalProvider;
    try {
      const res = await fetch('/api/models/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: (type === 'ollama' || type === 'local_llm') ? 'local_llm' : type,
          baseUrl: url,
          apiKey: key,
          provider: currentProv
        })
      });
      const data = await res.json();

      if (data.isOnline) {
        setShowSuccess(true);
        setTimeout(() => setShowSuccess(false), 4000);
      } else {
        setErrorMsg(data.error || 'Endpoint did not respond as expected.');
      }

      const availableModels = data.availableModels || [];
      // Keep existing selected model if valid
      const existingModel = config.selectedModel || localConfigs[currentProv]?.selectedModel;
      const keepModel = (availableModels.length > 0 && availableModels.includes(existingModel))
        ? existingModel
        : (availableModels[0] || existingModel || 'default');

      if (type === 'local_llm' || type === 'ollama') {
        const updatedLocalConfig: LocalLLMProviderConfig = {
          ...localConfigs[currentProv],
          baseUrl: url,
          apiKey: key,
          isOnline: data.isOnline,
          availableModels,
          selectedModel: keepModel,
        };
        onChange({
          ...config,
          type: 'local_llm',
          activeLocalProvider: currentProv,
          localConfigs: { ...localConfigs, [currentProv]: updatedLocalConfig },
          baseUrl: url,
          apiKey: key,
          isOnline: data.isOnline,
          availableModels,
          selectedModel: keepModel,
        });
      } else {
        onChange({
          ...config,
          type,
          baseUrl: url,
          apiKey: key,
          isOnline: data.isOnline,
          availableModels,
          selectedModel: keepModel,
        });
      }
    } catch (err: any) {
      setErrorMsg(`Failed to connect: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 bg-slate-950 border border-slate-800 rounded-2xl max-w-2xl mx-auto space-y-4">
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <h3 className="text-sm font-bold font-mono text-white flex items-center gap-2">
          ⚙️ Workspace & Engine Settings
        </h3>
        <span className="text-[10px] font-mono text-indigo-400 bg-indigo-950/80 border border-indigo-800 px-2 py-0.5 rounded">
          Saved & Synchronized
        </span>
      </div>

      {/* Tabs Bar */}
      <div className="flex items-center gap-2 bg-slate-900 p-1 rounded-xl border border-slate-800 font-mono text-xs">
        <button
          type="button"
          onClick={() => setActiveTab('models')}
          className={`flex-1 py-1.5 px-3 rounded-lg flex items-center justify-center gap-2 font-bold transition cursor-pointer ${
            activeTab === 'models'
              ? 'bg-indigo-600 text-white shadow'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <Cpu className="w-3.5 h-3.5" />
          <span>AI Models & Inference</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('search')}
          className={`flex-1 py-1.5 px-3 rounded-lg flex items-center justify-center gap-2 font-bold transition cursor-pointer ${
            activeTab === 'search'
              ? 'bg-indigo-600 text-white shadow'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <Globe className="w-3.5 h-3.5" />
          <span>Internet Assist</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('mail')}
          className={`flex-1 py-1.5 px-3 rounded-lg flex items-center justify-center gap-2 font-bold transition cursor-pointer ${
            activeTab === 'mail'
              ? 'bg-indigo-600 text-white shadow'
              : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
          }`}
        >
          <Mail className="w-3.5 h-3.5" />
          <span>Python Mail Server</span>
        </button>
      </div>

      {showSuccess && (
        <div className="p-2.5 rounded-lg bg-emerald-950/80 border border-emerald-700/80 text-emerald-300 text-xs font-mono flex items-center gap-2">
          <CheckCircle className="w-4 h-4 text-emerald-400" />
          <span>Endpoint connection successful! Models synchronized.</span>
        </div>
      )}

      {errorMsg && (
        <div className="p-2.5 rounded-lg bg-rose-950/80 border border-rose-700/80 text-rose-300 text-xs font-mono flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {activeTab === 'models' && (
        <ProviderConfigCards
          config={config}
          localConfigs={localConfigs}
          activeLocalProvider={activeLocalProvider}
          onSelectType={handleSelectType}
          onUpdateConfig={handleUpdateConfig}
          onUpdateLocalConfig={handleUpdateLocalConfig}
          onTestConnection={handleTestConnection}
          loading={loading}
        />
      )}

      {activeTab === 'search' && (
        <SearchAssistSettings
          config={config}
          onUpdateConfig={handleUpdateConfig}
        />
      )}

      {activeTab === 'mail' && (
        <LocalMailServerSettings />
      )}
    </div>
  );
}
