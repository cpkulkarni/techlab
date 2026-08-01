/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import { GoogleGenAI } from '@google/genai';

const router = Router();

// POST /api/models/check — verify connectivity and list available models
router.post('/check', async (req, res) => {
  const { type, baseUrl, apiKey, provider } = req.body;

  if (type === 'gemini') {
    try {
      const activeKey = apiKey || process.env.GEMINI_API_KEY;
      if (!activeKey) {
        return res.json({
          isOnline: false,
          error: 'Gemini API Key is missing. Add it to Settings > Secrets or the server configuration.',
          availableModels: []
        });
      }
      const tempClient = new GoogleGenAI({ apiKey: activeKey, httpOptions: { headers: { 'User-Agent': 'aistudio-build' } } });
      await tempClient.models.generateContent({ model: 'gemini-3.6-flash', contents: 'Ping' });
      return res.json({
        isOnline: true,
        availableModels: [
          'gemini-3.6-flash',
          'gemini-3.5-flash',
          'gemini-3.1-pro-preview',
          'gemini-3.1-flash-lite',
          'gemini-3.1-flash-image'
        ]
      });
    } catch (error: any) {
      return res.json({ isOnline: false, error: `Gemini verification failed: ${error.message}`, availableModels: [] });
    }
  }

  const activeProvider = provider || (type === 'ollama' ? 'ollama' : 'custom');

  if (type === 'ollama' || activeProvider === 'ollama') {
    const url = (baseUrl || 'http://localhost:11434').replace(/\/+$/, '');
    try {
      // First attempt native Ollama tags endpoint
      let response = await fetch(`${url}/api/tags`, { signal: AbortSignal.timeout(3000) });
      if (response.ok) {
        const data = (await response.json()) as any;
        const models = data.models?.map((m: any) => m.name) || [];
        return res.json({ isOnline: true, availableModels: models.length > 0 ? models : ['llama3', 'mistral', 'codegemma'] });
      }

      // Fallback to OpenAI-compatible endpoint on Ollama
      const modelsUrl = url.endsWith('/v1') ? `${url}/models` : `${url}/v1/models`;
      response = await fetch(modelsUrl, { signal: AbortSignal.timeout(3000) });
      if (response.ok) {
        const data = (await response.json()) as any;
        const models = data.data?.map((m: any) => m.id || m.name) || [];
        return res.json({ isOnline: true, availableModels: models.length > 0 ? models : ['llama3', 'mistral'] });
      }

      throw new Error(`Ollama returned HTTP status ${response.status}`);
    } catch (error: any) {
      return res.json({
        isOnline: false,
        error: `Could not connect to Ollama: ${error.message}. Ensure Ollama is running locally (e.g. 'ollama serve').`,
        availableModels: ['llama3', 'mistral', 'codegemma']
      });
    }
  }

  if (type === 'local_llm' || type === 'vllm' || type === 'lmstudio' || type === 'llamacpp' || type === 'custom' || type === 'openai') {
    let defaultUrl = 'http://localhost:8000/v1';
    if (activeProvider === 'vllm') defaultUrl = 'http://localhost:8000/v1';
    else if (activeProvider === 'lmstudio') defaultUrl = 'http://localhost:1234/v1';
    else if (activeProvider === 'llamacpp') defaultUrl = 'http://localhost:8080/v1';
    else if (activeProvider === 'custom') defaultUrl = 'http://localhost:5000/v1';
    else if (type === 'openai') defaultUrl = 'https://api.openai.com/v1';

    const cleanBaseUrl = (baseUrl || defaultUrl).replace(/\/+$/, '');
    const activeKey = apiKey || '';
    const headers: Record<string, string> = {};
    if (activeKey) headers['Authorization'] = `Bearer ${activeKey}`;

    // Candidate URL paths to check
    const candidateUrls = [
      cleanBaseUrl.endsWith('/models') ? cleanBaseUrl : `${cleanBaseUrl}/models`,
      cleanBaseUrl.endsWith('/v1') ? `${cleanBaseUrl}/models` : `${cleanBaseUrl}/v1/models`,
    ];

    let lastError = '';
    for (const testUrl of candidateUrls) {
      try {
        const response = await fetch(testUrl, { headers, signal: AbortSignal.timeout(3500) });
        if (response.ok) {
          const data = (await response.json()) as any;
          const rawList = data.data || data.models || data.items || [];
          const models = Array.isArray(rawList)
            ? rawList.map((m: any) => (typeof m === 'string' ? m : m.id || m.name)).filter(Boolean)
            : [];
          return res.json({
            isOnline: true,
            availableModels: models.length > 0 ? models : getFallbackModels(activeProvider)
          });
        } else {
          lastError = `Server returned status ${response.status} (${response.statusText})`;
        }
      } catch (err: any) {
        lastError = err.message;
      }
    }

    return res.json({
      isOnline: false,
      error: `Could not connect to ${activeProvider.toUpperCase()} at ${cleanBaseUrl}: ${lastError}`,
      availableModels: getFallbackModels(activeProvider)
    });
  }

  res.status(400).json({ success: false, error: 'Invalid server type' });
});

function getFallbackModels(provider: string): string[] {
  switch (provider) {
    case 'vllm':
      return ['meta-llama/Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.2', 'Qwen/Qwen2.5-7B-Instruct'];
    case 'lmstudio':
      return ['local-model', 'qwen2.5-7b-instruct', 'gemma-2-9b-it', 'llama-3.2-3b-instruct'];
    case 'llamacpp':
      return ['default', 'llama-2-7b-chat', 'codellama-7b-instruct', 'mistral-7b-v0.1'];
    case 'openai':
      return ['gpt-4o', 'gpt-4o-mini', 'o1-mini'];
    default:
      return ['default', 'llama3', 'mistral'];
  }
}

export default router;
