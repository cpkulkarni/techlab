/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { performWebSearch } from './search.js';
import { interactionLogs, writeLogToEkaDirectory, InteractionLog } from './logs.js';

export type { InteractionLog };

// ── Current date/time context injected into every system instruction ──────
export function nowContext(): string {
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
  const timeStr = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', timeZoneName: 'short' });
  return `CURRENT DATE & TIME: ${dateStr}, ${timeStr}. Use this as the definitive "today" for any date-related questions.`;
}

// Lazy singleton Gemini client
let geminiClient: GoogleGenAI | null = null;

export function getGeminiClient(customApiKey?: string): GoogleGenAI {
  const key = customApiKey || process.env.GEMINI_API_KEY;
  if (!key) throw new Error('GEMINI_API_KEY environment variable is required. Please set it in the Secrets panel.');
  if (!geminiClient || customApiKey) {
    geminiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: { headers: { 'User-Agent': 'aistudio-build' } },
    });
  }
  return geminiClient;
}

export interface GenerationResult {
  text: string;
  citations?: Array<{ title: string; url: string }>;
}

export async function generateText({
  model,
  systemInstruction,
  prompt,
  contents,
  customConfig,
  responseMimeType,
  responseSchema,
  temperature,
  searchEnabled,
  logType = 'General AI Call',
  signal,
}: {
  model?: string;
  systemInstruction?: string;
  prompt: string;
  contents?: any;
  customConfig?: any;
  responseMimeType?: string;
  responseSchema?: any;
  temperature?: number;
  searchEnabled?: boolean;
  logType?: string;
  signal?: AbortSignal;
}): Promise<GenerationResult> {
  const type = customConfig?.type || 'gemini';
  let selectedModel = model || customConfig?.selectedModel;
  if (type === 'local_llm' || type === 'ollama') {
    const activeProv = customConfig?.activeLocalProvider || 'ollama';
    const localConf = customConfig?.localConfigs?.[activeProv];
    if (!selectedModel && localConf?.selectedModel) {
      selectedModel = localConf.selectedModel;
    }
  }
  if (!selectedModel) {
    selectedModel = type === 'gemini' ? 'gemini-3.6-flash' : 'default';
  }
  const apiKey = customConfig?.apiKey || process.env.GEMINI_API_KEY;

  let resultText = '';
  let resultCitations: Array<{ title: string; url: string }> = [];
  let searchContext = '';

  // Check early if signal is already aborted
  if (signal?.aborted) {
    throw new Error('Request aborted by user before execution started.');
  }

  // 1. Optionally perform a web search to augment the prompt
  if (searchEnabled) {
    try {
      const searchResults = await performWebSearch(prompt, customConfig);
      if (searchResults && searchResults.length > 0) {
        searchContext = `\n\n[INTERNET SEARCH RESULTS — retrieved ${new Date().toLocaleString('en-US')}]\n` +
          searchResults.map((r, i) => `Source [${i + 1}]: ${r.title}\nURL: ${r.url}\nSnippet: ${r.snippet}`).join('\n\n') +
          `\n\nUsing the search results above, answer the user's question with accurate, up-to-date information. Cite sources with bracketed numbers [1], [2], etc.`;
        resultCitations = searchResults.map(r => ({ title: r.title, url: r.url }));
      }
    } catch (searchError: any) {
      console.error('Web search failed:', searchError);
      searchContext = `\n\n[Warning: Internet search failed: ${searchError.message}. Answer based on your best knowledge but note the search failure.]`;
    }
  }

  const augmentedPrompt = searchContext ? `${prompt}${searchContext}` : prompt;
  const activeSystemInstruction = systemInstruction || '';

  try {
    if (type === 'gemini') {
      const ai = getGeminiClient(apiKey);
      const config: any = {};
      if (activeSystemInstruction) config.systemInstruction = activeSystemInstruction;
      if (responseMimeType) config.responseMimeType = responseMimeType;
      if (responseSchema) config.responseSchema = responseSchema;
      if (temperature !== undefined) config.temperature = temperature;

      let geminiContents = contents || augmentedPrompt;
      if (contents && searchContext) {
        geminiContents = [
          ...contents.slice(0, -1),
          {
            ...contents[contents.length - 1],
            parts: [...(contents[contents.length - 1].parts || []), { text: searchContext }],
          },
        ];
      }

      const response = await ai.models.generateContent({ model: selectedModel, contents: geminiContents, config });
      resultText = response.text || '';

      if (resultCitations.length === 0) {
        const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
        resultCitations = chunks?.map((chunk: any) => ({
          title: chunk.web?.title || 'Grounding Source',
          url: chunk.web?.uri || ''
        })).filter((c: any) => c.url) || [];
      }
    } else if (type === 'ollama' && !customConfig?.baseUrl?.includes('/v1')) {
      const baseUrl = customConfig?.baseUrl || 'http://localhost:11434';
      const messages: any[] = [];
      if (activeSystemInstruction) messages.push({ role: 'system', content: activeSystemInstruction });
      messages.push({ role: 'user', content: augmentedPrompt });

      const body: any = { model: selectedModel, messages, stream: false };
      if (temperature !== undefined) body.options = { temperature };
      if (responseMimeType === 'application/json') body.format = 'json';

      const res = await fetch(`${baseUrl.replace(/\/+$/, '')}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`Ollama Error [HTTP ${res.status}] Endpoint: ${baseUrl}/api/chat | Model: ${selectedModel} | Details: ${errText}`);
      }
      const data = await res.json() as any;
      resultText = data.message?.content || '';
    } else if (
      type === 'openai' ||
      type === 'local_llm' ||
      type === 'vllm' ||
      type === 'lmstudio' ||
      type === 'llamacpp' ||
      type === 'custom' ||
      type === 'ollama'
    ) {
      let defaultUrl = 'https://api.openai.com/v1';
      if (type === 'local_llm' || type === 'vllm') defaultUrl = 'http://localhost:8000/v1';
      else if (type === 'lmstudio') defaultUrl = 'http://localhost:1234/v1';
      else if (type === 'llamacpp') defaultUrl = 'http://localhost:8080/v1';
      else if (type === 'ollama') defaultUrl = 'http://localhost:11434/v1';
      else if (type === 'custom') defaultUrl = 'http://localhost:5000/v1';

      const baseUrl = (customConfig?.baseUrl || defaultUrl).replace(/\/+$/, '');
      let chatUrl = baseUrl;
      if (chatUrl.endsWith('/chat/completions')) {
        // already complete
      } else if (chatUrl.endsWith('/v1')) {
        chatUrl = `${chatUrl}/chat/completions`;
      } else {
        chatUrl = `${chatUrl}/v1/chat/completions`;
      }

      const messages: any[] = [];
      if (activeSystemInstruction) messages.push({ role: 'system', content: activeSystemInstruction });
      messages.push({ role: 'user', content: augmentedPrompt });

      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (customConfig?.apiKey) headers['Authorization'] = `Bearer ${customConfig.apiKey}`;

      const body: any = { model: selectedModel, messages };
      if (temperature !== undefined) body.temperature = temperature;
      if (responseMimeType === 'application/json') body.response_format = { type: 'json_object' };

      const res = await fetch(chatUrl, { method: 'POST', headers, body: JSON.stringify(body), signal });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`${type.toUpperCase()} Server Error [HTTP ${res.status}] Endpoint: ${chatUrl} | Model: ${selectedModel} | Details: ${errText}`);
      }
      const data = await res.json() as any;
      resultText = data.choices?.[0]?.message?.content || '';
    } else {
      throw new Error(`Unsupported LLM server engine type: ${type}`);
    }

    const logEntry: InteractionLog = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      type: logType,
      prompt,
      systemInstruction,
      response: resultText,
      model: selectedModel,
      citations: resultCitations
    };
    interactionLogs.push(logEntry);
    await writeLogToEkaDirectory(logEntry);

    return { text: resultText, citations: resultCitations };
  } catch (error: any) {
    const logEntry: InteractionLog = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      type: `${logType} (FAILED)`,
      prompt,
      systemInstruction,
      response: `ERROR: ${error.message}`,
      model: selectedModel,
      citations: []
    };
    interactionLogs.push(logEntry);
    await writeLogToEkaDirectory(logEntry);
    throw error;
  }
}
