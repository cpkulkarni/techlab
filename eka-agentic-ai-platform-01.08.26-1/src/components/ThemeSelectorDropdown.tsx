/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect } from 'react';
import { Sun, Moon, Palette, ChevronDown, Check, Sparkles } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { ThemeMode, AccentColor, ACCENT_COLORS, getSelectThemeClass } from '../utils/theme';

interface ThemeSelectorDropdownProps {
  theme: ThemeMode;
  onThemeChange: (theme: ThemeMode) => void;
  accentColor: AccentColor;
  onAccentChange: (accent: AccentColor) => void;
}

export default function ThemeSelectorDropdown({
  theme,
  onThemeChange,
  accentColor,
  onAccentChange
}: ThemeSelectorDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const activeAccent = ACCENT_COLORS[accentColor] || ACCENT_COLORS.indigo;

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const triggerBg = theme === 'white'
    ? 'bg-white border-slate-300 text-slate-800 hover:bg-slate-50'
    : theme === 'light-grey'
      ? 'bg-zinc-100 border-zinc-300 text-zinc-800 hover:bg-zinc-200'
      : 'bg-slate-900 border-slate-800 text-slate-200 hover:bg-slate-800';

  const menuBg = theme === 'white'
    ? 'bg-white border-slate-200 text-slate-800 shadow-xl'
    : theme === 'light-grey'
      ? 'bg-zinc-100 border-zinc-300 text-zinc-800 shadow-xl'
      : 'bg-slate-900 border-slate-800 text-slate-200 shadow-2xl';

  const sectionHeader = theme === 'dark' ? 'text-slate-400' : 'text-slate-500';

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`px-2.5 py-1.5 rounded-md border text-xs font-mono font-medium flex items-center gap-2 cursor-pointer transition-all shadow-sm ${triggerBg}`}
        title="Customize Theme Mode & Accent Color Palette"
      >
        <span className="flex items-center gap-1.5">
          {theme === 'white' && <Sun className="w-3.5 h-3.5 text-amber-500" />}
          {theme === 'light-grey' && <Palette className="w-3.5 h-3.5 text-indigo-500" />}
          {theme === 'dark' && <Moon className="w-3.5 h-3.5 text-indigo-400" />}
          <span className="capitalize font-semibold">{theme === 'light-grey' ? 'Gray' : theme}</span>
        </span>

        {/* Active Accent Color Dot */}
        <span
          className="w-3 h-3 rounded-full border border-black/20 shadow-inner shrink-0"
          style={{ backgroundColor: activeAccent.colorHex }}
        />

        <ChevronDown className={`w-3 h-3 transition-transform duration-200 opacity-70 ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Popover Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 6, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className={`absolute right-0 top-full mt-1.5 w-64 p-3 rounded-lg border z-50 select-none ${menuBg}`}
          >
            {/* Header */}
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-slate-500/20">
              <span className="text-[11px] font-mono font-bold uppercase tracking-wider flex items-center gap-1.5">
                <Sparkles className="w-3 h-3 text-indigo-400" />
                Theme & Color Palette
              </span>
            </div>

            {/* Mode Selection */}
            <div className="mb-3">
              <span className={`block text-[10px] font-mono font-semibold uppercase tracking-wider mb-1.5 ${sectionHeader}`}>
                Canvas Background
              </span>
              <div className="grid grid-cols-3 gap-1.5">
                <button
                  type="button"
                  onClick={() => onThemeChange('white')}
                  className={`px-2 py-1.5 rounded text-[10px] font-mono font-medium flex items-center justify-center gap-1 cursor-pointer transition-all border ${
                    theme === 'white'
                      ? `${activeAccent.bgClass} ${activeAccent.contrastText} ${activeAccent.borderClass} font-bold shadow-sm`
                      : theme === 'dark'
                        ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                        : 'bg-white border-slate-300 text-slate-700 hover:bg-slate-100'
                  }`}
                >
                  <Sun className="w-3 h-3 shrink-0" />
                  <span>White</span>
                </button>

                <button
                  type="button"
                  onClick={() => onThemeChange('light-grey')}
                  className={`px-2 py-1.5 rounded text-[10px] font-mono font-medium flex items-center justify-center gap-1 cursor-pointer transition-all border ${
                    theme === 'light-grey'
                      ? `${activeAccent.bgClass} ${activeAccent.contrastText} ${activeAccent.borderClass} font-bold shadow-sm`
                      : theme === 'dark'
                        ? 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700'
                        : 'bg-zinc-200 border-zinc-300 text-zinc-800 hover:bg-zinc-300'
                  }`}
                >
                  <Palette className="w-3 h-3 shrink-0" />
                  <span>Gray</span>
                </button>

                <button
                  type="button"
                  onClick={() => onThemeChange('dark')}
                  className={`px-2 py-1.5 rounded text-[10px] font-mono font-medium flex items-center justify-center gap-1 cursor-pointer transition-all border ${
                    theme === 'dark'
                      ? `${activeAccent.bgClass} ${activeAccent.contrastText} ${activeAccent.borderClass} font-bold shadow-sm`
                      : 'bg-slate-900 border-slate-800 text-slate-300 hover:bg-slate-800'
                  }`}
                >
                  <Moon className="w-3 h-3 shrink-0" />
                  <span>Dark</span>
                </button>
              </div>
            </div>

            {/* Accent Color Palette Selection */}
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className={`text-[10px] font-mono font-semibold uppercase tracking-wider ${sectionHeader}`}>
                  Accent Color
                </span>
                <span className="text-[10px] font-mono font-bold capitalize text-indigo-400">
                  {activeAccent.name}
                </span>
              </div>

              <div className="grid grid-cols-7 gap-1.5">
                {(Object.keys(ACCENT_COLORS) as AccentColor[]).map((key) => {
                  const item = ACCENT_COLORS[key];
                  const isSelected = accentColor === key;
                  return (
                    <button
                      key={key}
                      type="button"
                      onClick={() => onAccentChange(key)}
                      className={`w-7 h-7 rounded-full flex items-center justify-center transition-transform cursor-pointer relative ${
                        isSelected ? 'scale-110 ring-2 ring-indigo-400 ring-offset-2 ring-offset-slate-900' : 'hover:scale-105 opacity-80 hover:opacity-100'
                      }`}
                      style={{ backgroundColor: item.colorHex }}
                      title={`${item.name} Accent`}
                    >
                      {isSelected && <Check className={`w-3.5 h-3.5 ${item.contrastText}`} />}
                    </button>
                  );
                })}
              </div>

              {/* Live Preview Bar */}
              <div className={`mt-3 p-2 rounded border flex items-center justify-between text-[10px] font-mono ${activeAccent.bgClass} ${activeAccent.contrastText}`}>
                <span className="font-bold">Active Accent Preview</span>
                <span className="px-1.5 py-0.5 rounded bg-black/20 text-[9px] font-bold">Contrast OK</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
