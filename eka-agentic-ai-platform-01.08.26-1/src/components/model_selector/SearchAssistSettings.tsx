/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { ModelServerConfig } from '../../types';
import { Globe, Search, Key, Hash, CheckCircle2 } from 'lucide-react';

interface SearchAssistSettingsProps {
  config: ModelServerConfig;
  onUpdateConfig: (updated: Partial<ModelServerConfig>) => void;
}

export function SearchAssistSettings({ config, onUpdateConfig }: SearchAssistSettingsProps) {
  const engine = config.searchEngine || 'duckduckgo';
  const entryCount = config.searchEntryCount || 5;

  return (
    <div className="space-y-4 font-mono text-xs">
      <div className="p-3 bg-slate-900/80 border border-slate-800 rounded-xl space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2 text-indigo-400 font-bold">
            <Globe className="w-4 h-4" />
            <span>Internet Search Assist Grounding</span>
          </div>
          <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3" /> Active
          </span>
        </div>

        <p className="text-[11px] text-slate-400 font-sans">
          Configure search engine integration to enable live web retrieval for up-to-date facts, documentation, and research queries.
        </p>

        {/* Search Engine Selector */}
        <div>
          <label className="text-slate-300 block mb-1.5 font-bold">Search Engine Provider:</label>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {[
              { id: 'duckduckgo', name: 'DuckDuckGo', desc: 'No API Key required' },
              { id: 'google_cse', name: 'Google CSE', desc: 'Google Custom Search API' },
              { id: 'bing', name: 'Bing Web API', desc: 'Azure Bing Search' },
              { id: 'brave', name: 'Brave Search', desc: 'Brave API Key' },
              { id: 'serper', name: 'Serper.dev', desc: 'Google Serper API' },
            ].map(item => (
              <button
                key={item.id}
                type="button"
                onClick={() => onUpdateConfig({ searchEngine: item.id })}
                className={`p-2.5 rounded-xl border text-left flex flex-col justify-between transition cursor-pointer ${
                  engine === item.id
                    ? 'bg-indigo-950/60 border-indigo-500 text-white'
                    : 'bg-slate-950 border-slate-800 text-slate-400 hover:bg-slate-800/60'
                }`}
              >
                <div className="font-bold text-xs text-slate-200">{item.name}</div>
                <div className="text-[10px] text-slate-400 mt-1">{item.desc}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Search Result Count Slider */}
        <div className="pt-2 border-t border-slate-800">
          <div className="flex items-center justify-between mb-1">
            <label className="text-slate-300 font-bold flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-indigo-400" /> Max Search Results Per Query:
            </label>
            <span className="text-indigo-400 font-bold">{entryCount} results</span>
          </div>
          <input
            type="range"
            min="1"
            max="20"
            value={entryCount}
            onChange={(e) => onUpdateConfig({ searchEntryCount: parseInt(e.target.value, 10) })}
            className="w-full accent-indigo-500 cursor-pointer"
          />
        </div>

        {/* Credentials for Google CSE */}
        {engine === 'google_cse' && (
          <div className="space-y-2 pt-2 border-t border-slate-800">
            <div>
              <label className="text-slate-300 block mb-1">Google CSE API Key:</label>
              <input
                type="password"
                value={config.googleCseApiKey || ''}
                onChange={(e) => onUpdateConfig({ googleCseApiKey: e.target.value })}
                placeholder="AIzaSy..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <div>
              <label className="text-slate-300 block mb-1">Search Engine ID (CX):</label>
              <input
                type="text"
                value={config.googleCseCx || ''}
                onChange={(e) => onUpdateConfig({ googleCseCx: e.target.value })}
                placeholder="0123456789..."
                className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
              />
            </div>
          </div>
        )}

        {/* Credentials for Bing */}
        {engine === 'bing' && (
          <div className="pt-2 border-t border-slate-800">
            <label className="text-slate-300 block mb-1">Bing API Subscription Key:</label>
            <input
              type="password"
              value={config.bingApiKey || ''}
              onChange={(e) => onUpdateConfig({ bingApiKey: e.target.value })}
              placeholder="Bing Subscription Key..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        )}

        {/* Credentials for Brave */}
        {engine === 'brave' && (
          <div className="pt-2 border-t border-slate-800">
            <label className="text-slate-300 block mb-1">Brave Search API Key:</label>
            <input
              type="password"
              value={config.braveApiKey || ''}
              onChange={(e) => onUpdateConfig({ braveApiKey: e.target.value })}
              placeholder="BSA..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        )}

        {/* Credentials for Serper */}
        {engine === 'serper' && (
          <div className="pt-2 border-t border-slate-800">
            <label className="text-slate-300 block mb-1">Serper.dev API Key:</label>
            <input
              type="password"
              value={config.serperApiKey || ''}
              onChange={(e) => onUpdateConfig({ serperApiKey: e.target.value })}
              placeholder="API Key..."
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500"
            />
          </div>
        )}
      </div>
    </div>
  );
}
