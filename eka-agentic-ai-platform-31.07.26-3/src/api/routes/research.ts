/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import path from 'path';
import fs from 'fs/promises';
import { generateText, nowContext } from '../shared/llm.js';
import { performWebSearch, SearchResultItem } from '../shared/search.js';

const router = Router();

function cleanAndParseJSON(text: string): any {
  if (!text) return null;
  try {
    const cleaned = text.replace(/```json/gi, '').replace(/```/g, '').trim();
    return JSON.parse(cleaned);
  } catch (e) {
    const match = text.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
    if (match) {
      try {
        return JSON.parse(match[1]);
      } catch (e2) {}
    }
    return null;
  }
}

// POST /api/agent/research
router.post('/', async (req, res) => {
  const { prompt, customConfig, searchEnabled } = req.body;
  try {
    const isSearchOn = searchEnabled !== false;
    const researchDate = new Date().toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });

    console.log(`[Research Workflow] Starting multi-step research pipeline for prompt: "${prompt}"`);

    // 1. Fetch initial search results
    let initialSearchResults: SearchResultItem[] = [];
    if (isSearchOn) {
      try {
        initialSearchResults = await performWebSearch(prompt, customConfig);
        console.log(`[Research Workflow] Step 1: Initial search fetched ${initialSearchResults.length} results.`);
      } catch (err: any) {
        console.warn(`[Research Workflow] Step 1 initial search warning: ${err.message}`);
      }
    }

    // 2. Prepare a vocabulary of key concepts from initial search results and model's own knowledge
    const vocabPrompt = `You are an expert Research Knowledge Architect.
Topic: "${prompt}"
Initial Search Results:
${
  initialSearchResults.length > 0
    ? JSON.stringify(initialSearchResults, null, 2)
    : 'No initial search results available; use your comprehensive domain knowledge.'
}

Task: Identify and prepare a vocabulary of 4 to 6 key concepts essential to understanding this topic deeply.
For each key concept, provide:
1. "term": The technical concept name
2. "searchQuery": A targeted search query to perform a deeper search for this concept
3. "definition": A clear initial definition combining search findings and domain knowledge

Return STRICTLY a valid JSON object in this exact schema:
{
  "topic": "${prompt}",
  "concepts": [
    {
      "term": "Term Name",
      "searchQuery": "Search Query for Deep Search",
      "definition": "Initial concept definition"
    }
  ]
}`;

    const vocabGen = await generateText({
      customConfig,
      prompt: vocabPrompt,
      systemInstruction: 'You are a JSON-only API. Respond strictly with valid JSON.',
      searchEnabled: false,
      responseMimeType: 'application/json',
      logType: 'Research Vocabulary Extraction'
    });

    let vocabularyData = cleanAndParseJSON(vocabGen.text);
    if (!vocabularyData || !Array.isArray(vocabularyData.concepts) || vocabularyData.concepts.length === 0) {
      vocabularyData = {
        topic: prompt,
        concepts: [
          {
            term: prompt,
            searchQuery: prompt,
            definition: `Primary concept and core architectural domain for ${prompt}`
          }
        ]
      };
    }

    // 3. Use the same vocabulary to search again for deeper search for each key concept
    const deepSearchResults: Record<string, SearchResultItem[]> = {};
    const allCitationsMap = new Map<string, { title: string; url: string }>();

    for (const item of initialSearchResults) {
      if (item.url) allCitationsMap.set(item.url, { title: item.title, url: item.url });
    }

    if (isSearchOn) {
      console.log(`[Research Workflow] Step 3: Conducting deeper searches for ${vocabularyData.concepts.length} key concepts...`);
      for (const concept of vocabularyData.concepts) {
        const query = concept.searchQuery || concept.term;
        try {
          const results = await performWebSearch(query, customConfig);
          deepSearchResults[concept.term] = results;
          for (const r of results) {
            if (r.url) allCitationsMap.set(r.url, { title: r.title, url: r.url });
          }
        } catch (deepErr: any) {
          console.warn(`[Research Workflow] Deep search failed for concept "${concept.term}": ${deepErr.message}`);
          deepSearchResults[concept.term] = [];
        }
      }
    }

    // 4. Extract search results and merge all earlier and newly received content into a single large Knowledge MD file
    let knowledgeBaseContent = `# Knowledge Base: ${prompt}\n\n`;
    knowledgeBaseContent += `> **Generated Date:** ${researchDate}\n`;
    knowledgeBaseContent += `> **Topic:** ${prompt}\n\n`;
    knowledgeBaseContent += `---\n\n`;

    knowledgeBaseContent += `## 1. Initial Research & Context\n\n`;
    if (initialSearchResults.length > 0) {
      initialSearchResults.forEach((r, idx) => {
        knowledgeBaseContent += `### [Initial Result ${idx + 1}] ${r.title}\n`;
        knowledgeBaseContent += `- **URL:** ${r.url}\n`;
        knowledgeBaseContent += `- **Snippet:** ${r.snippet}\n\n`;
      });
    } else {
      knowledgeBaseContent += `*Initial context synthesized from domain knowledge and model intelligence.*\n\n`;
    }

    knowledgeBaseContent += `## 2. Key Concepts & Vocabulary\n\n`;
    for (const c of vocabularyData.concepts) {
      knowledgeBaseContent += `### ${c.term}\n`;
      knowledgeBaseContent += `- **Search Query Used:** \`${c.searchQuery}\`
`;
      knowledgeBaseContent += `- **Definition & Overview:** ${c.definition}\n\n`;
    }

    knowledgeBaseContent += `## 3. Deep Research Findings by Concept\n\n`;
    for (const c of vocabularyData.concepts) {
      const results = deepSearchResults[c.term] || [];
      knowledgeBaseContent += `### Deep Dive: ${c.term}\n\n`;
      if (results.length > 0) {
        results.forEach((r, idx) => {
          knowledgeBaseContent += `#### Source [${idx + 1}]: ${r.title}\n`;
          knowledgeBaseContent += `- **Link:** ${r.url}\n`;
          knowledgeBaseContent += `- **Content Extract:** ${r.snippet}\n\n`;
        });
      } else {
        knowledgeBaseContent += `*Deep conceptual analysis generated via domain model synthesis.*\n\n`;
      }
    }

    // 5. Using the knowledge MD file, generate a Knowledge Graph
    const kgPrompt = `You are an expert Data Visualization and Knowledge Graph Architect.
Analyze the following Knowledge Base Markdown document and generate a structured Knowledge Graph of key entities, relationships, and concepts.

Knowledge Base Document:
${knowledgeBaseContent.slice(0, 12000)}

Task: Generate a knowledge graph with:
1. "nodes": Array of objects with "id", "label", "category" (e.g. Core Concept, Tool, Architecture, Method), and "description"
2. "edges": Array of objects with "source", "target", and "relationship" (e.g. "implements", "uses", "enables", "competes with")
3. "mermaid": A valid Mermaid.js graph string (e.g. \`graph TD\\n  A[Label A] -->|relationship| B[Label B]\`)

Return STRICTLY valid JSON.`;

    const kgGen = await generateText({
      customConfig,
      prompt: kgPrompt,
      systemInstruction: 'You are a JSON-only API. Respond strictly with valid JSON.',
      searchEnabled: false,
      responseMimeType: 'application/json',
      logType: 'Knowledge Graph Generation'
    });

    let kgData = cleanAndParseJSON(kgGen.text);
    if (!kgData || !Array.isArray(kgData.nodes)) {
      kgData = {
        nodes: vocabularyData.concepts.map((c: any, i: number) => ({
          id: `node-${i + 1}`,
          label: c.term,
          category: 'Key Concept',
          description: c.definition
        })),
        edges: vocabularyData.concepts.slice(1).map((c: any, i: number) => ({
          source: `node-1`,
          target: `node-${i + 2}`,
          relationship: 'relates to'
        })),
        mermaid:
          `graph TD\n` +
          vocabularyData.concepts.map((c: any, i: number) => `  N${i + 1}[${c.term}]`).join('\n')
      };
    }

    // Format Knowledge Graph as Markdown as well
    let kgMdContent = `# Knowledge Graph: ${prompt}\n\n`;
    kgMdContent += `> **Topic:** ${prompt}\n\n`;
    if (kgData.mermaid) {
      kgMdContent += `\`\`\`mermaid\n${kgData.mermaid}\n\`\`\`\n\n`;
    }
    kgMdContent += `## Graph Nodes\n\n`;
    for (const n of kgData.nodes || []) {
      kgMdContent += `- **${n.label}** (${n.category}): ${n.description || ''}\n`;
    }
    kgMdContent += `\n## Graph Edges\n\n`;
    for (const e of kgData.edges || []) {
      kgMdContent += `- \`${e.source}\` ➔ *${e.relationship}* ➔ \`${e.target}\`\n`;
    }

    // 6. Generate the Final Research Report using Vocabulary, Knowledge Graph, and Knowledge Base MD
    const reportSystemInstruction = `${nowContext()}

You are a Senior Technical Research Analyst with deep expertise in software engineering, APIs, frameworks, and industry trends.
Using the provided Vocabulary, Knowledge Base, and Knowledge Graph, produce a comprehensive, publication-ready technical research report in Markdown.

The report MUST follow this structure:
# [Topic Title]

> **Research Date:** ${researchDate}
> **Research Scope:** Comprehensive Multi-Step Deep Research Pipeline

---

## Executive Summary
[2-3 paragraph overview]

## Conceptual Vocabulary & Core Framework
[Synthesize key vocabulary terms, definitions, and domain scope]

## Knowledge Graph & Ecosystem Architecture
[Include Mermaid graph diagram and narrative breakdown of node/edge relationships]

## Technical Deep Dive & Architecture
[In-depth technical breakdown based on knowledge base, architecture, code examples, best practices]

## Comparison, Ecosystem & Tooling
[Comparison table or analysis of alternatives and tools]

## Strategic Recommendations
[Actionable guidance, do's and don'ts]

## References & Research Citations
[List all cited sources and URLs]

---
*Report generated by Eka Agentic AI Platform*`;

    const reportPrompt = `Topic: "${prompt}"

Vocabulary Summary:
${JSON.stringify(vocabularyData, null, 2)}

Knowledge Graph Summary:
${JSON.stringify({ nodesCount: kgData.nodes?.length, edgesCount: kgData.edges?.length, mermaid: kgData.mermaid })}

Knowledge Base Content Preview:
${knowledgeBaseContent.slice(0, 8000)}

Please generate the final comprehensive research report now.`;

    const reportGen = await generateText({
      customConfig,
      prompt: reportPrompt,
      systemInstruction: reportSystemInstruction,
      searchEnabled: false,
      logType: 'Final Research Report Generation'
    });

    const reportMdContent = reportGen.text || '';

    // 7. Save Vocabulary, Knowledge Graph, Knowledge MD, and Research Report to app-output/research (and app-output/reasarch)
    const safeTitle =
      prompt
        .toLowerCase()
        .replace(/[^a-z0-9\s-]/g, '')
        .trim()
        .replace(/\s+/g, '-')
        .slice(0, 60) || 'research';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const reportFileName = `${safeTitle}-${timestamp}.md`;

    const primaryDir = path.join(process.cwd(), 'app-output', 'research');
    const altDir = path.join(process.cwd(), 'app-output', 'reasarch');

    await fs.mkdir(primaryDir, { recursive: true });
    await fs.mkdir(altDir, { recursive: true });

    const filesToWrite = [
      {
        p1: path.join(primaryDir, 'vocabulary.json'),
        p2: path.join(altDir, 'vocabulary.json'),
        content: JSON.stringify(vocabularyData, null, 2)
      },
      {
        p1: path.join(primaryDir, 'vocabulary.md'),
        p2: path.join(altDir, 'vocabulary.md'),
        content:
          `# Research Vocabulary\n\n` +
          vocabularyData.concepts
            .map(
              (c: any) => `### ${c.term}\n- **Query:** \`${c.searchQuery}\`
- **Definition:** ${c.definition}\n`
            )
            .join('\n')
      },
      {
        p1: path.join(primaryDir, 'knowledge_base.md'),
        p2: path.join(altDir, 'knowledge_base.md'),
        content: knowledgeBaseContent
      },
      {
        p1: path.join(primaryDir, `knowledge_base_${safeTitle}.md`),
        p2: path.join(altDir, `knowledge_base_${safeTitle}.md`),
        content: knowledgeBaseContent
      },
      {
        p1: path.join(primaryDir, 'knowledge_graph.json'),
        p2: path.join(altDir, 'knowledge_graph.json'),
        content: JSON.stringify(kgData, null, 2)
      },
      {
        p1: path.join(primaryDir, 'knowledge_graph.md'),
        p2: path.join(altDir, 'knowledge_graph.md'),
        content: kgMdContent
      },
      {
        p1: path.join(primaryDir, reportFileName),
        p2: path.join(altDir, reportFileName),
        content: reportMdContent
      },
      {
        p1: path.join(primaryDir, 'research_report.md'),
        p2: path.join(altDir, 'research_report.md'),
        content: reportMdContent
      }
    ];

    for (const f of filesToWrite) {
      await fs.writeFile(f.p1, f.content, 'utf8');
      await fs.writeFile(f.p2, f.content, 'utf8');
    }

    const aggregatedCitations = Array.from(allCitationsMap.values());

    res.json({
      success: true,
      filePath: `app-output/research/${reportFileName}`,
      citations: aggregatedCitations,
      vocabularyPath: `app-output/research/vocabulary.json`,
      knowledgeBasePath: `app-output/research/knowledge_base.md`,
      knowledgeGraphPath: `app-output/research/knowledge_graph.json`,
      preview: reportMdContent.slice(0, 400)
    });
  } catch (error: any) {
    console.error('Research workflow error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
});

export default router;
