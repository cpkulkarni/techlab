/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { FileCode, Play, Sparkles, Layers, CheckCircle2 } from 'lucide-react';

export default function DiagramSpecToCode({ theme = 'dark', onCodeGenerated }: { theme?: string; onCodeGenerated?: (code: string) => void }) {
  const [specType, setSpecType] = useState<'mermaid' | 'plantuml' | 'json_wireframe' | 'ascii'>('mermaid');
  const [specText, setSpecText] = useState(`graph TD
    A[User Dashboard] --> B[Analytics Panel]
    A --> C[Settings Modal]
    B --> D[Export CSV Button]`);
  const [generating, setGenerating] = useState(false);
  const [generatedCode, setGeneratedCode] = useState<string | null>(null);

  const isDark = theme === 'dark';

  const handleCompileToCode = async () => {
    if (!specText.trim()) return;
    setGenerating(true);
    setGeneratedCode(null);
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{
            role: 'user',
            content: `Compile the following ${specType.toUpperCase()} architectural wireframe / diagram spec directly into a fully functional, beautiful React + Tailwind component:\n\n\`\`\`${specType}\n${specText}\n\`\`\``
          }],
          modelConfig: { type: 'gemini', selectedModel: 'gemini-3.6-flash' }
        }),
      });
      const data = await res.json();
      if (data.reply) {
        setGeneratedCode(data.reply);
        if (onCodeGenerated) onCodeGenerated(data.reply);
      }
    } catch (e: any) {
      // ignore
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className={`p-4 rounded-xl border ${isDark ? 'bg-slate-900/90 border-slate-800 text-slate-100' : 'bg-white border-slate-200 text-slate-800'} space-y-4`}>
      <div className="flex items-center space-x-2 pb-3 border-b border-slate-800">
        <Layers className="w-5 h-5 text-indigo-400" />
        <div>
          <h3 className="font-semibold text-sm">Industry Standard Wireframe & UML Spec-to-Code Compiler</h3>
          <p className="text-[11px] text-slate-400">Render Mermaid, PlantUML, ASCII wireframes or JSON UI specs and compile them directly to React code</p>
        </div>
      </div>

      <div className="flex space-x-2 text-xs">
        {(['mermaid', 'plantuml', 'json_wireframe', 'ascii'] as const).map((type) => (
          <button
            key={type}
            onClick={() => setSpecType(type)}
            className={`px-3 py-1.5 rounded-lg border font-medium capitalize transition ${specType === type ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300' : 'bg-slate-950 border-slate-800 text-slate-400'}`}
          >
            {type.replace('_', ' ')}
          </button>
        ))}
      </div>

      <div className="space-y-3 text-xs">
        <div>
          <label className="block text-slate-400 font-medium mb-1">Diagram Specification Input</label>
          <textarea
            rows={5}
            value={specText}
            onChange={e => setSpecText(e.target.value)}
            className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-slate-200 font-mono text-[11px] outline-none focus:border-indigo-500"
          />
        </div>

        <button
          onClick={handleCompileToCode}
          disabled={generating}
          className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white font-medium rounded-lg shadow transition flex items-center justify-center space-x-1.5"
        >
          <Sparkles className="w-4 h-4 text-indigo-200" />
          <span>{generating ? 'Compiling Diagram Spec to React Component...' : 'Compile Diagram Spec to React Code'}</span>
        </button>

        {generatedCode && (
          <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg space-y-2">
            <div className="flex items-center justify-between text-emerald-400 font-semibold">
              <span className="flex items-center space-x-1">
                <CheckCircle2 className="w-4 h-4" />
                <span>Generated Code Component</span>
              </span>
            </div>
            <pre className="p-2 bg-slate-900 rounded font-mono text-[11px] max-h-60 overflow-y-auto text-slate-300 whitespace-pre-wrap">
              {generatedCode}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
