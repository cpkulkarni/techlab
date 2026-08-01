/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import * as cheerio from 'cheerio';

export interface SearchResultItem {
  title: string;
  url: string;
  snippet: string;
}

export function prepareSearchQuery(prompt: string): string {
  let q = prompt.trim();

  // Remove trailing punctuation
  q = q.replace(/[?.,!]+$/, '');

  // Strip common conversational phrases
  const patterns = [
    /^(can you\s+)?search\s+(for\s+)?/i,
    /^(can you\s+)?find\s+(me\s+)?(details\s+on\s+|info\s+on\s+|information\s+on\s+)?/i,
    /^(can you\s+)?tell\s+me\s+(about\s+|what\s+is\s+|who\s+is\s+)?/i,
    /^what\s+(is|are)\s+the\s+/i,
    /^what\s+is\s+/i,
    /^who\s+is\s+/i,
    /^check\s+the\s+latest\s+(on\s+|about\s+)?/i,
  ];

  for (const pattern of patterns) {
    if (pattern.test(q)) {
      q = q.replace(pattern, '');
      break;
    }
  }

  // If still too long, take the first 15 words
  const words = q.split(/\s+/);
  if (words.length > 15) {
    q = words.slice(0, 15).join(' ');
  }

  return q.trim();
}

export async function performWebSearch(prompt: string, customConfig: any): Promise<SearchResultItem[]> {
  const query = prepareSearchQuery(prompt);
  const engine = customConfig?.searchEngine || 'duckduckgo';
  const maxResults = customConfig?.searchEntryCount ? Math.max(1, Math.min(20, Number(customConfig.searchEntryCount))) : 5;
  console.log(`[Web Search] Query: "${query}" using Engine: "${engine}" (Max Entries: ${maxResults})`);

  if (!query) return [];

  try {
    if (engine === 'duckduckgo') {
      const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
      const response = await fetch(searchUrl, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
          'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
          'Accept-Language': 'en-US,en;q=0.5'
        },
        signal: AbortSignal.timeout(8000)
      });
      if (!response.ok) throw new Error(`DuckDuckGo returned status ${response.status}`);
      const html = await response.text();
      const $ = cheerio.load(html);
      const results: SearchResultItem[] = [];

      $('.result').each((i, el) => {
        if (results.length >= maxResults) return;
        const isAd = $(el).hasClass('result--ad') ||
          $(el).find('.result__type').text().toLowerCase().includes('ad') ||
          $(el).find('.result__snippet').text().toLowerCase().includes('sponsored');
        if (isAd) return;
        const titleEl = $(el).find('.result__a');
        const title = titleEl.text().trim();
        const urlRaw = titleEl.attr('href') || '';
        const snippet = $(el).find('.result__snippet').text().trim();
        let url = urlRaw;
        if (url.includes('uddg=')) {
          try { url = decodeURIComponent(url.split('uddg=')[1].split('&')[0]); } catch (e) {}
        }
        if (title && url && url.startsWith('http') && !url.includes('duckduckgo.com/')) {
          results.push({ title, url, snippet });
        }
      });
      return results;
    }

    if (engine === 'google_cse') {
      const apiKey = customConfig?.googleCseApiKey;
      const cx = customConfig?.googleCseCx;
      if (!apiKey || !cx) throw new Error('Missing Google CSE API Key or Search Engine ID (CX).');
      const searchUrl = `https://www.googleapis.com/customsearch/v1?key=${encodeURIComponent(apiKey)}&cx=${encodeURIComponent(cx)}&q=${encodeURIComponent(query)}&num=${maxResults}`;
      const response = await fetch(searchUrl, { signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw new Error(`Google CSE error (${response.status}): ${await response.text()}`);
      const data = await response.json() as any;
      return (data.items || []).slice(0, maxResults).map((item: any) => ({ title: item.title || '', url: item.link || '', snippet: item.snippet || '' }));
    }

    if (engine === 'bing') {
      const apiKey = customConfig?.bingApiKey;
      if (!apiKey) throw new Error('Missing Bing Search API Key.');
      const searchUrl = `https://api.bing.microsoft.com/v7.0/search?q=${encodeURIComponent(query)}&count=${maxResults}`;
      const response = await fetch(searchUrl, { headers: { 'Ocp-Apim-Subscription-Key': apiKey }, signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw new Error(`Bing Search error (${response.status}): ${await response.text()}`);
      const data = await response.json() as any;
      return (data.webPages?.value || []).slice(0, maxResults).map((item: any) => ({ title: item.name || '', url: item.url || '', snippet: item.snippet || '' }));
    }

    if (engine === 'brave') {
      const apiKey = customConfig?.braveApiKey;
      if (!apiKey) throw new Error('Missing Brave Search API Key.');
      const searchUrl = `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${maxResults}`;
      const response = await fetch(searchUrl, { headers: { 'X-Subscription-Token': apiKey, 'Accept': 'application/json' }, signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw new Error(`Brave Search error (${response.status}): ${await response.text()}`);
      const data = await response.json() as any;
      return (data.web?.results || []).slice(0, maxResults).map((item: any) => ({ title: item.title || '', url: item.url || '', snippet: item.description || '' }));
    }

    if (engine === 'serper') {
      const apiKey = customConfig?.serperApiKey;
      if (!apiKey) throw new Error('Missing Serper API Key.');
      const response = await fetch('https://google.serper.dev/search', {
        method: 'POST',
        headers: { 'X-API-KEY': apiKey, 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: query, num: maxResults }),
        signal: AbortSignal.timeout(8000)
      });
      if (!response.ok) throw new Error(`Serper API error (${response.status}): ${await response.text()}`);
      const data = await response.json() as any;
      return (data.organic || []).slice(0, maxResults).map((item: any) => ({ title: item.title || '', url: item.link || '', snippet: item.snippet || '' }));
    }

    throw new Error(`Unsupported search engine: ${engine}`);
  } catch (err: any) {
    console.error(`[Web Search Error] ${err.message}`);
    throw err;
  }
}
