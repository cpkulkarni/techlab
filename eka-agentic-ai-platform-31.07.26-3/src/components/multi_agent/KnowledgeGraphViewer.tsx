/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { BookOpen, Network, FileText, Plus, RefreshCw, Hash } from 'lucide-react';

export default function KnowledgeGraphViewer({ theme = 'dark' }: { theme?: string }) {
  const [entities, setEntities] = useState<any[]>([]);
  const [documents, setDocuments] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [newEntityName, setNewEntityName] = useState('');
  const [newEntityDesc, setNewEntityDesc] = useState('');

  const isDark = theme === 'dark';

  const fetchKnowledge = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/multi-agent/knowledge');
      const data = await res.json();
      if (data.success) {
        setEntities(data.entities || []);
        setDocuments(data.documents || []);
      }
    } catch (e) {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchKnowledge();
  }, []);

  const handleAddEntity = async () => {
    if (!newEntityName.trim() || !newEntityDesc.trim()) return;
    try {
      const res = await fetch('/api/multi-agent/knowledge/entity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newEntityName,
          type: 'vocabulary_term',
          description: newEntityDesc,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setNewEntityName('');
        setNewEntityDesc('');
        fetchKnowledge();
      }
    } catch (e) {
      // ignore
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-4`}>
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <Network className="w-5 h-5 text-indigo-400" />
          <div>
            <h3 className="font-semibold text-sm">Vocabulary, Knowledge Graph & Vault</h3>
            <p className="text-[11px] text-slate-400">Structured domain terms and research vault injected into Multi-Agent prompts</p>
          </div>
        </div>
        <button onClick={fetchKnowledge} className="p-1 text-slate-400 hover:text-slate-200">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Add Domain Term Form */}
      <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2 text-xs">
        <h4 className="font-semibold text-slate-300">Add Vocabulary Term or Concept Entity</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          <input
            type="text"
            placeholder="Term name (e.g. Rate Limiter Middleware)..."
            value={newEntityName}
            onChange={e => setNewEntityName(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
          />
          <input
            type="text"
            placeholder="Definition / Description..."
            value={newEntityDesc}
            onChange={e => setNewEntityDesc(e.target.value)}
            className="bg-slate-900 border border-slate-800 rounded px-2.5 py-1.5 text-slate-200 outline-none focus:border-indigo-500"
          />
        </div>
        <button
          onClick={handleAddEntity}
          className="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium transition flex items-center space-x-1"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Add Term</span>
        </button>
      </div>

      {/* Entities Grid */}
      <div className="space-y-2 text-xs">
        <h4 className="font-semibold text-slate-400 text-[11px] uppercase tracking-wider flex items-center space-x-1">
          <Hash className="w-3.5 h-3.5 text-indigo-400" />
          <span>Knowledge Graph Terms ({entities.length})</span>
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-40 overflow-y-auto">
          {entities.map(e => (
            <div key={e.id} className="p-2.5 bg-slate-950 border border-slate-800/80 rounded-lg space-y-1">
              <span className="font-bold text-indigo-300 block">{e.name}</span>
              <p className="text-slate-400 text-[11px] leading-tight">{e.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Knowledge Documents */}
      <div className="space-y-2 text-xs pt-1">
        <h4 className="font-semibold text-slate-400 text-[11px] uppercase tracking-wider flex items-center space-x-1">
          <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
          <span>Knowledge Vault Documents ({documents.length})</span>
        </h4>
        <div className="space-y-1.5 max-h-36 overflow-y-auto">
          {documents.map(d => (
            <div key={d.id} className="p-2.5 bg-slate-950 border border-slate-800/80 rounded-lg space-y-1">
              <div className="flex items-center justify-between">
                <span className="font-bold text-slate-200">{d.title}</span>
                <span className="text-[10px] px-1.5 py-0.2 bg-indigo-500/20 text-indigo-300 rounded">{d.category}</span>
              </div>
              <p className="text-slate-400 text-[11px] line-clamp-2">{d.content}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
