/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Save, X, FileCode, Check } from 'lucide-react';

interface CodeEditorHeaderProps {
  selectedFile: string | null;
  isDirty: boolean;
  isSaving: boolean;
  onSave: () => void;
  onClose?: () => void;
}

export function CodeEditorHeader({
  selectedFile,
  isDirty,
  isSaving,
  onSave,
  onClose,
}: CodeEditorHeaderProps) {
  if (!selectedFile) return null;

  const fileName = selectedFile.split('/').pop() || selectedFile;

  return (
    <div className="h-10 px-3 border-b border-slate-800 bg-slate-900/90 flex items-center justify-between shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <FileCode className="w-4 h-4 text-emerald-400 shrink-0" />
        <span className="text-xs font-mono text-slate-200 truncate font-semibold">
          {fileName}
        </span>
        {isDirty && (
          <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" title="Unsaved changes" />
        )}
      </div>

      <div className="flex items-center gap-1.5 shrink-0">
        <button
          type="button"
          onClick={onSave}
          disabled={!isDirty || isSaving}
          className="flex items-center gap-1 text-[11px] font-mono bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-2.5 py-1 rounded transition cursor-pointer font-bold shadow"
        >
          {isSaving ? <Check className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" />}
          <span>{isSaving ? 'Saving...' : 'Save'}</span>
        </button>

        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-white hover:bg-slate-800 rounded transition cursor-pointer"
            title="Close active file"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
