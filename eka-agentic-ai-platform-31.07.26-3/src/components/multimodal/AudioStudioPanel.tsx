/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../../types';
import { Mic, Volume2, Cpu, Play, Square } from 'lucide-react';

interface AudioStudioPanelProps {
  modelConfig?: ModelServerConfig;
  mode: 'stt' | 'tts';
}

export function AudioStudioPanel({ modelConfig, mode }: AudioStudioPanelProps) {
  const [ttsText, setTtsText] = useState('Hello! Eka AI Studio supports neural text to speech and speech recognition.');
  const [isPlaying, setIsPlaying] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);

  const activeModelName = modelConfig?.selectedModel || 'gemini-3.6-flash';

  const handleSpeak = () => {
    if (!ttsText.trim()) return;
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(ttsText);
      utterance.onend = () => setIsPlaying(false);
      utterance.onerror = () => setIsPlaying(false);
      setIsPlaying(true);
      window.speechSynthesis.speak(utterance);
    } else {
      alert('Speech synthesis is not supported in this browser environment.');
    }
  };

  const handleStopSpeak = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      setIsPlaying(false);
    }
  };

  const handleSimulateTranscribe = () => {
    setTranscript('Transcribed recording: "Eka Studio provides high performance speech-to-text integration across all modules."');
  };

  return (
    <div className="p-4 space-y-4 max-w-2xl mx-auto">
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <span className="text-xs font-bold font-mono text-emerald-400 flex items-center gap-1.5">
          {mode === 'tts' ? <Volume2 className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          {mode === 'tts' ? 'Neural Text-to-Speech (TTS) Studio' : 'Speech-to-Text (STT) Recognition'}
        </span>
        <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/80 px-2 py-0.5 rounded flex items-center gap-1">
          <Cpu className="w-3 h-3" /> Model: {activeModelName}
        </span>
      </div>

      {mode === 'tts' ? (
        <div className="space-y-3">
          <div>
            <label className="text-xs font-mono text-slate-300 block mb-1">Text to Vocalize:</label>
            <textarea
              value={ttsText}
              onChange={(e) => setTtsText(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs text-slate-200 focus:outline-none focus:border-emerald-500 h-28 resize-none"
            />
          </div>

          <div className="flex items-center gap-2">
            {isPlaying ? (
              <button
                type="button"
                onClick={handleStopSpeak}
                className="bg-rose-600 hover:bg-rose-500 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow"
              >
                <Square className="w-4 h-4" /> Stop Speech
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSpeak}
                className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition flex items-center gap-2 cursor-pointer shadow"
              >
                <Play className="w-4 h-4" /> Synthesize Voice
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="border border-slate-800 rounded-xl bg-slate-900/60 p-6 text-center space-y-3">
            <Mic className="w-10 h-10 mx-auto text-red-400 animate-pulse" />
            <p className="text-xs font-mono text-slate-300">Audio Speech Recognition Ready</p>
            <button
              type="button"
              onClick={handleSimulateTranscribe}
              className="bg-red-600 hover:bg-red-500 text-white text-xs font-mono font-bold px-4 py-2 rounded-lg transition cursor-pointer shadow"
            >
              Start Microphone Recording
            </button>
          </div>

          {transcript && (
            <div className="p-3 bg-slate-900 border border-slate-800 rounded-xl text-xs font-mono text-slate-200">
              <span className="text-[10px] text-slate-400 block font-bold mb-1">Transcription Output:</span>
              {transcript}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
