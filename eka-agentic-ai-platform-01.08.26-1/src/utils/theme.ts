/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type ThemeMode = 'white' | 'light-grey' | 'dark';
export type AccentColor = 'indigo' | 'emerald' | 'violet' | 'rose' | 'amber' | 'cyan' | 'blue';

export interface AccentConfig {
  id: AccentColor;
  name: string;
  colorHex: string;
  // Primary filled button / active badge style
  bgClass: string;
  bgHoverClass: string;
  // Text colors
  textClass: string;
  lightTextClass: string;
  // Borders and Focus Rings
  borderClass: string;
  ringClass: string;
  // Opposite high-contrast text on accent fill
  contrastText: string;
  // Light tint backgrounds for active tabs/cards
  badgeBg: string;
  badgeText: string;
}

export const ACCENT_COLORS: Record<AccentColor, AccentConfig> = {
  indigo: {
    id: 'indigo',
    name: 'Indigo',
    colorHex: '#6366f1',
    bgClass: 'bg-indigo-600',
    bgHoverClass: 'hover:bg-indigo-500',
    textClass: 'text-indigo-600 dark:text-indigo-400',
    lightTextClass: 'text-indigo-400',
    borderClass: 'border-indigo-500',
    ringClass: 'focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500',
    contrastText: 'text-white',
    badgeBg: 'bg-indigo-500/15',
    badgeText: 'text-indigo-600 dark:text-indigo-300',
  },
  emerald: {
    id: 'emerald',
    name: 'Emerald',
    colorHex: '#10b981',
    bgClass: 'bg-emerald-600',
    bgHoverClass: 'hover:bg-emerald-500',
    textClass: 'text-emerald-600 dark:text-emerald-400',
    lightTextClass: 'text-emerald-400',
    borderClass: 'border-emerald-500',
    ringClass: 'focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500',
    contrastText: 'text-white',
    badgeBg: 'bg-emerald-500/15',
    badgeText: 'text-emerald-600 dark:text-emerald-300',
  },
  violet: {
    id: 'violet',
    name: 'Violet',
    colorHex: '#8b5cf6',
    bgClass: 'bg-violet-600',
    bgHoverClass: 'hover:bg-violet-500',
    textClass: 'text-violet-600 dark:text-violet-400',
    lightTextClass: 'text-violet-400',
    borderClass: 'border-violet-500',
    ringClass: 'focus:border-violet-500 focus:ring-1 focus:ring-violet-500',
    contrastText: 'text-white',
    badgeBg: 'bg-violet-500/15',
    badgeText: 'text-violet-600 dark:text-violet-300',
  },
  rose: {
    id: 'rose',
    name: 'Rose',
    colorHex: '#f43f5e',
    bgClass: 'bg-rose-600',
    bgHoverClass: 'hover:bg-rose-500',
    textClass: 'text-rose-600 dark:text-rose-400',
    lightTextClass: 'text-rose-400',
    borderClass: 'border-rose-500',
    ringClass: 'focus:border-rose-500 focus:ring-1 focus:ring-rose-500',
    contrastText: 'text-white',
    badgeBg: 'bg-rose-500/15',
    badgeText: 'text-rose-600 dark:text-rose-300',
  },
  amber: {
    id: 'amber',
    name: 'Amber',
    colorHex: '#f59e0b',
    bgClass: 'bg-amber-500',
    bgHoverClass: 'hover:bg-amber-400',
    textClass: 'text-amber-600 dark:text-amber-400',
    lightTextClass: 'text-amber-400',
    borderClass: 'border-amber-500',
    ringClass: 'focus:border-amber-500 focus:ring-1 focus:ring-amber-500',
    contrastText: 'text-slate-950 font-bold',
    badgeBg: 'bg-amber-500/15',
    badgeText: 'text-amber-700 dark:text-amber-300',
  },
  cyan: {
    id: 'cyan',
    name: 'Cyan',
    colorHex: '#06b6d4',
    bgClass: 'bg-cyan-600',
    bgHoverClass: 'hover:bg-cyan-500',
    textClass: 'text-cyan-600 dark:text-cyan-400',
    lightTextClass: 'text-cyan-400',
    borderClass: 'border-cyan-500',
    ringClass: 'focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500',
    contrastText: 'text-white',
    badgeBg: 'bg-cyan-500/15',
    badgeText: 'text-cyan-600 dark:text-cyan-300',
  },
  blue: {
    id: 'blue',
    name: 'Blue',
    colorHex: '#3b82f6',
    bgClass: 'bg-blue-600',
    bgHoverClass: 'hover:bg-blue-500',
    textClass: 'text-blue-600 dark:text-blue-400',
    lightTextClass: 'text-blue-400',
    borderClass: 'border-blue-500',
    ringClass: 'focus:border-blue-500 focus:ring-1 focus:ring-blue-500',
    contrastText: 'text-white',
    badgeBg: 'bg-blue-500/15',
    badgeText: 'text-blue-600 dark:text-blue-300',
  },
};

/**
 * Returns clean classes for form <select> dropdowns based on active theme & accent.
 */
export function getSelectThemeClass(theme: ThemeMode, accent: AccentColor = 'indigo'): string {
  const acc = ACCENT_COLORS[accent] || ACCENT_COLORS.indigo;
  if (theme === 'white') {
    return `bg-white text-slate-800 border-slate-300 hover:border-slate-400 shadow-sm ${acc.ringClass}`;
  }
  if (theme === 'light-grey') {
    return `bg-zinc-100 text-zinc-800 border-zinc-300 hover:border-zinc-400 shadow-sm ${acc.ringClass}`;
  }
  return `bg-slate-900 text-slate-200 border-slate-800 hover:border-slate-700 ${acc.ringClass}`;
}

/**
 * Returns container card/panel classes matching the theme.
 */
export function getCardThemeClass(theme: ThemeMode): string {
  if (theme === 'white') {
    return 'bg-white border-slate-200 text-slate-900 shadow-sm';
  }
  if (theme === 'light-grey') {
    return 'bg-zinc-100 border-zinc-300 text-zinc-900 shadow-sm';
  }
  return 'bg-slate-900 border-slate-800 text-slate-200';
}
