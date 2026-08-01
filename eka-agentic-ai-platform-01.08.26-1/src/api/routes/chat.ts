/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';
import { generateText, nowContext } from '../shared/llm.js';

const router = Router();

// POST /api/chat
router.post('/', async (req, res) => {
  const abortController = new AbortController();
  res.on('close', () => {
    if (!res.writableEnded) {
      abortController.abort();
    }
  });

  const { messages, attachments, searchEnabled, customConfig, mode = 'chat' } = req.body;

  try {
    const lastMsgObj = messages[messages.length - 1] || {};
    let lastMessage = lastMsgObj.content || '';
    
    // Attach text file attachments directly into the user message context
    const allAtts = [...(attachments || []), ...(lastMsgObj.attachments || [])];
    if (allAtts.length > 0) {
      let attachmentTextContext = '\n\n[ATTACHED FILES & CONTENT SHARED WITH MODEL]:\n';
      for (const att of allAtts) {
        if (att.content) {
          attachmentTextContext += `\n📄 File: ${att.name}\n\`\`\`\n${att.content}\n\`\`\`\n`;
        } else if (att.dataUrl) {
          attachmentTextContext += `\n📎 Media Attachment: ${att.name} (Type: ${att.type})\n`;
        }
      }
      lastMessage += attachmentTextContext;
    }

    const historyPrompt = messages.map((m: any) => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`).join('\n\n');

    let baseSystemInstruction = nowContext() + '\n\n';
    if (mode === 'chat') {
      baseSystemInstruction += 'You are a helpful, professional AI Systems Architect. Focus on systems design, architectural guidelines, structural patterns, and conceptual clarifications. Keep responses concise, clean, and highly educational.';
    } else if (mode === 'code') {
      baseSystemInstruction += 'You are an elite Software Engineer. Focus on providing clean, optimized, production-ready code blocks. Always specify code block languages and follow SOLID design principles.';
    } else if (mode === 'research') {
      baseSystemInstruction += 'You are a Senior Technical Research Analyst. Focus on analyzing APIs, library features, tech stack comparisons, and up-to-date documentation. Rely heavily on current, factual technical references.';
    } else if (mode === 'multimodal') {
      baseSystemInstruction += 'You are an expert AI Multimodal Media Studio Agent. You assist with Speech-to-Text, Text-to-Speech, Text-to-Video, Video-to-Text, 3D Assets, and Language Translation. Provide clear, helpful, well-structured answers.';
    } else if (mode === 'documentation') {
      baseSystemInstruction += 'You are an expert Technical Writer. Focus on writing exceptionally clear, comprehensive markdown documents, user manuals, setup guides, and detailed inline comments.';
    } else if (mode === 'testing') {
      baseSystemInstruction += 'You are a Senior QA Automation Engineer. Focus on writing robust unit tests, integration test specifications, finding boundary conditions, and explaining failure stack traces.';
    } else {
      baseSystemInstruction += 'You are a professional AI Coding Agent with high expertise. Format responses in beautiful markdown.';
    }

    if (searchEnabled) {
      baseSystemInstruction += '\n\nIMPORTANT: Internet Assist is ENABLED. You MUST actively search the live internet/web for any relevant, factual, current details, library changes, or API specifications before formulating your answer.';
    }

    let contents: any[] | undefined = undefined;
    if (!customConfig || customConfig.type === 'gemini') {
      contents = messages.map((m: any, idx: number) => {
        const isLast = idx === messages.length - 1;
        const parts: any[] = [{ text: isLast ? lastMessage : m.content }];
        
        // Include inline binary data for Gemini if dataUrl attachments exist
        if (isLast && allAtts.length > 0) {
          for (const att of allAtts) {
            if (att.dataUrl && att.dataUrl.includes(';base64,')) {
              const [header, base64] = att.dataUrl.split(';base64,');
              const mimeType = header.replace('data:', '') || att.type || 'image/png';
              parts.push({
                inlineData: {
                  mimeType,
                  data: base64
                }
              });
            }
          }
        }

        return {
          role: m.role === 'user' ? 'user' : 'model',
          parts
        };
      });
    }

    const systemInstruction = (customConfig?.type === 'gemini' || !customConfig?.type)
      ? baseSystemInstruction
      : baseSystemInstruction + `\n\nContext of chat so far:\n${historyPrompt}`;

    const result = await generateText({
      customConfig,
      prompt: lastMessage,
      contents,
      systemInstruction,
      searchEnabled,
      logType: `${mode.toUpperCase()} Chat`,
      signal: abortController.signal,
    });

    res.json({ success: true, reply: result.text || 'No response generated.', citations: result.citations || [] });
  } catch (error: any) {
    res.status(500).json({ success: true, reply: `⚠️ Error during model generation: ${error.message}` });
  }
});

export default router;
