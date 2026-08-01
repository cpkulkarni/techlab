/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { ModelServerConfig, DEFAULT_LOCAL_CONFIGS } from '../types';
import { ThemeMode, AccentColor } from '../utils/theme';

const LOCAL_STORAGE_KEY = 'eka_model_config_v2';

export function useAppConfig() {
  const [theme, setTheme] = useState<ThemeMode>('dark');
  const [accentColor, setAccentColor] = useState<AccentColor>('indigo');
  const [modelConfig, setModelConfigState] = useState<ModelServerConfig>(() => {
    try {
      const saved = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && typeof parsed === 'object') return parsed;
      }
    } catch (e) {}
    return {
      type: 'gemini',
      baseUrl: '',
      apiKey: '',
      selectedModel: 'gemini-3.6-flash',
      isOnline: false,
      availableModels: [],
      activeLocalProvider: 'ollama',
      localConfigs: DEFAULT_LOCAL_CONFIGS,
      searchEngine: 'duckduckgo',
    };
  });

  const llmSaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const llmConfigLoadedRef = useRef(false);

  // Set modelConfig and save to localStorage
  const updateModelConfig = (newConfig: ModelServerConfig) => {
    setModelConfigState(newConfig);
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(newConfig));
    } catch (e) {}
  };

  // Load backend config on mount
  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(data => {
        if (data.success && data.config) {
          if (data.config.modelConfig) {
            setModelConfigState(prev => ({
              ...prev,
              ...data.config.modelConfig,
              // Keep selectedModel if explicitly set
              selectedModel: data.config.modelConfig.selectedModel || prev.selectedModel,
            }));
          }
          if (data.config.theme) setTheme(data.config.theme);
          if (data.config.accentColor) setAccentColor(data.config.accentColor);
        }
        llmConfigLoadedRef.current = true;
      })
      .catch(() => {
        llmConfigLoadedRef.current = true;
      });
  }, []);

  // Sync to server config API debounced
  useEffect(() => {
    if (!llmConfigLoadedRef.current) return;
    if (llmSaveTimerRef.current) clearTimeout(llmSaveTimerRef.current);
    llmSaveTimerRef.current = setTimeout(() => {
      fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modelConfig,
          theme,
          accentColor,
        }),
      }).catch(() => {});
    }, 800);

    return () => {
      if (llmSaveTimerRef.current) clearTimeout(llmSaveTimerRef.current);
    };
  }, [modelConfig, theme, accentColor]);

  return {
    theme,
    setTheme,
    accentColor,
    setAccentColor,
    modelConfig,
    setModelConfig: updateModelConfig,
  };
}
