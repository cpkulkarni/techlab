/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ModelServerConfig } from '../types';
import { Image, Eye, Mic, Volume2, Video, Box, Globe } from 'lucide-react';
import { TextToImagePanel } from './multimodal/TextToImagePanel';
import { ImageToTextPanel } from './multimodal/ImageToTextPanel';
import { AudioStudioPanel } from './multimodal/AudioStudioPanel';
import { Video3DPanel } from './multimodal/Video3DPanel';
import { TranslationPanel } from './multimodal/TranslationPanel';

export type MultimodalFeature =
  | 'text_to_image'
  | 'image_to_text'
  | 'stt'
  | 'tts'
  | 'text_to_video'
  | 'text_to_3d'
  | 'translation';

interface MultimodalStudioProps {
  initialFeature?: MultimodalFeature;
  modelConfig?: ModelServerConfig;
}

export default function MultimodalStudio({
  initialFeature = 'text_to_image',
  modelConfig,
}: MultimodalStudioProps) {
  const [activeFeature, setActiveFeature] = useState<MultimodalFeature>(initialFeature);

  const tabs: { id: MultimodalFeature; label: string; icon: React.ReactNode }[] = [
    { id: 'text_to_image', label: 'Text-to-Image', icon: <Image className="w-3.5 h-3.5 text-pink-400" /> },
    { id: 'image_to_text', label: 'Image-to-Text OCR', icon: <Eye className="w-3.5 h-3.5 text-cyan-400" /> },
    { id: 'stt', label: 'Speech-to-Text', icon: <Mic className="w-3.5 h-3.5 text-red-400" /> },
    { id: 'tts', label: 'Text-to-Speech', icon: <Volume2 className="w-3.5 h-3.5 text-emerald-400" /> },
    { id: 'text_to_video', label: 'Text-to-Video', icon: <Video className="w-3.5 h-3.5 text-violet-400" /> },
    { id: 'text_to_3d', label: 'Text-to-3D', icon: <Box className="w-3.5 h-3.5 text-amber-500" /> },
    { id: 'translation', label: 'Translation', icon: <Globe className="w-3.5 h-3.5 text-teal-400" /> },
  ];

  return (
    <div className="h-full bg-slate-950 flex flex-col overflow-hidden">
      {/* Tab Strip */}
      <div className="px-3 py-2 border-b border-slate-800 bg-slate-900/90 flex items-center gap-1.5 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveFeature(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-lg text-xs font-mono font-medium transition cursor-pointer shrink-0 ${
              activeFeature === tab.id
                ? 'bg-indigo-950 border border-indigo-500/50 text-indigo-300 font-bold shadow'
                : 'bg-slate-900/60 border border-slate-800 text-slate-400 hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Feature Panel View */}
      <div className="flex-1 overflow-y-auto">
        {activeFeature === 'text_to_image' && <TextToImagePanel modelConfig={modelConfig} />}
        {activeFeature === 'image_to_text' && <ImageToTextPanel modelConfig={modelConfig} />}
        {(activeFeature === 'stt' || activeFeature === 'tts') && (
          <AudioStudioPanel modelConfig={modelConfig} mode={activeFeature} />
        )}
        {(activeFeature === 'text_to_video' || activeFeature === 'text_to_3d') && (
          <Video3DPanel modelConfig={modelConfig} feature={activeFeature} />
        )}
        {activeFeature === 'translation' && (
          <TranslationPanel modelConfig={modelConfig} />
        )}
      </div>
    </div>
  );
}
