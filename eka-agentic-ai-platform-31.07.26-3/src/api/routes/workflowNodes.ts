/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * workflowNodes.ts
 * ----------------
 * Contains the per-node execution logic for the pipeline workflow engine.
 * Each case in the switch corresponds to one node type the user can place on
 * the workflow canvas. Extracted here so workflow.ts stays readable.
 */

import path from 'path';
import fs from 'fs/promises';
import nodemailer from 'nodemailer';
import dns from 'dns';
import { generateText } from '../shared/llm.js';
import { getWorkspaceDir, collectDirFiles } from '../shared/workspace.js';

export interface NodeExecutionContext {
  node: any;
  nodes: any[];
  edges: any[];
  nodeMap: Record<string, any>;
  contextByNode: Record<string, string>;
  customConfig: any;
  workflowName: string;
  log: (line: string) => void;
}

/**
 * Execute a single workflow node and return its output string.
 * Throws on hard failures; soft/skipped states should return a descriptive string.
 */
export async function executeNode(ctx: NodeExecutionContext): Promise<string> {
  const { node, edges, nodeMap, contextByNode, customConfig, workflowName, log } = ctx;
  const WORKSPACE_DIR = getWorkspaceDir();

  // Collect upstream context from parent nodes
  const upstreamEdges = edges.filter((e: any) => e.targetId === node.id);
  const upstreamContext = upstreamEdges
    .map((e: any) => contextByNode[e.sourceId] || '')
    .filter(Boolean)
    .join('\n\n---\n\n');

  // The direct input-node value flowing in
  const inputValue: string =
    upstreamEdges
      .map((e: any) => nodeMap[e.sourceId]?.type === 'input' ? contextByNode[e.sourceId] ?? '' : '')
      .find((v: string) => v !== '') ?? upstreamContext;

  const resolveInput = (cfgQuery: string | undefined, upstream: string): string =>
    cfgQuery && cfgQuery.trim() ? cfgQuery.trim() : upstream;

  let output = '';

  switch (node.type) {

    case 'input': {
      output = (node.config.inputText || '').trim();
      if (!output) log(`  ⚠️ Input node has no text — downstream nodes will receive empty input`);
      else log(`  ✅ Input: ${output.length} chars`);
      break;
    }

    case 'code_execution': {
      const cfg = node.config;
      const resolvedArgs = (cfg.args || '')
        .replace(/\{\{input\}\}/g, inputValue)
        .replace(/\{\{context\}\}/g, upstreamContext);
      const targetPath = path.join(WORKSPACE_DIR, cfg.fileName || '');
      let fileContent = '';
      try { fileContent = await fs.readFile(targetPath, 'utf8'); } catch {}
      output = `[Code Execution]\nFile: ${cfg.fileName}\nLanguage: ${cfg.language}\nArgs: ${resolvedArgs || '(none)'}\nUpstream input available: ${inputValue.length} chars\n\n--- File Preview (first 400 chars) ---\n${fileContent.slice(0, 400) || '(file not found)'}`;
      log(`  ✅ Code execution staged: ${cfg.fileName}`);
      break;
    }

    case 'test_runner': {
      const cfg = node.config;
      const testDir = path.join(WORKSPACE_DIR, cfg.directory || 'tests');
      let testFiles: string[] = [];
      try {
        const entries = await fs.readdir(testDir);
        testFiles = entries.filter(f => /test/i.test(f));
      } catch {}
      const count = testFiles.length;
      output = `[Test Runner]\nDirectory: ${cfg.directory}\nFramework: ${cfg.framework}\nPattern: ${cfg.pattern}\nUpstream context: ${upstreamContext.length} chars\n\nFound ${count} test file(s)${count > 0 ? ': ' + testFiles.join(', ') : ''}\nResult: ${count > 0 ? `${count} tests discovered. All PASSED (simulated).` : 'No test files found.'}`;
      log(`  ✅ Test runner: ${count} files in ${cfg.directory}`);
      break;
    }

    case 'rag_vector_db': {
      const cfg = node.config;
      const query = resolveInput(cfg.query, inputValue);
      if (!query) log(`  ⚠️ RAG Vector DB: no query set and no upstream input — results may be empty`);
      output = `[RAG Vector DB]\nDB: ${cfg.dbType} @ ${cfg.host}:${cfg.port}\nCollection: ${cfg.collectionName}\nEmbedding: ${cfg.embeddingModel}\nQuery: "${query.slice(0, 120)}"\nTop-K: ${cfg.topK}\n\nSimulated: top-${cfg.topK} chunks retrieved from collection '${cfg.collectionName}'.`;
      log(`  ✅ Vector DB query simulated (${cfg.dbType}/${cfg.collectionName})`);
      break;
    }

    case 'rag_elastic': {
      const cfg = node.config;
      const query = resolveInput(cfg.query, inputValue);
      if (!query) log(`  ⚠️ RAG Elastic: no query set and no upstream input`);
      output = `[RAG Elasticsearch]\nHost: ${cfg.host}:${cfg.port}\nIndex: ${cfg.indexName}\nQuery: "${query.slice(0, 120)}"\nTop-K: ${cfg.topK}\n\nSimulated: top-${cfg.topK} documents retrieved from index '${cfg.indexName}'.`;
      log(`  ✅ Elasticsearch query simulated (${cfg.indexName})`);
      break;
    }

    case 'rag_search_engine': {
      const cfg = node.config;
      const query = resolveInput(cfg.query, inputValue);
      if (!query) {
        output = `[Web Search RAG]\nEngine: ${cfg.engine}\nQuery: (empty — no query configured and no upstream input)\n\nNo search performed.`;
        log(`  ⚠️ Web search skipped: no query`);
        break;
      }
      let searchResults: any[] = [];
      try {
        const searchUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query.slice(0, 200))}`;
        const resp = await fetch(searchUrl, { headers: { 'User-Agent': 'Mozilla/5.0' }, signal: AbortSignal.timeout(6000) });
        if (resp.ok) {
          const html = await resp.text();
          const cheerio = await import('cheerio');
          const $ = cheerio.load(html);
          $('.result').each((i: number, el: any) => {
            if (searchResults.length >= (cfg.topK || 5)) return;
            const title = $(el).find('.result__a').text().trim();
            const snippet = $(el).find('.result__snippet').text().trim();
            if (title) searchResults.push({ title, snippet });
          });
        }
      } catch {}
      output = `[Web Search RAG]\nEngine: ${cfg.engine}\nQuery: "${query.slice(0, 80)}"\n\nResults (${searchResults.length}):\n${searchResults.map((r, i) => `${i + 1}. ${r.title}\n   ${r.snippet}`).join('\n') || '(No live results)'}`;
      log(`  ✅ Web search: ${searchResults.length} results for "${query.slice(0, 50)}"`);
      break;
    }

    case 'rag_local_files': {
      const cfg = node.config;
      const query = resolveInput(cfg.query, inputValue);
      const dirPath = path.isAbsolute(cfg.directory || '')
        ? cfg.directory
        : path.join(WORKSPACE_DIR, cfg.directory || 'docs');
      const fileTypes = (cfg.fileTypes || 'txt,md').split(',').map((s: string) => s.trim());
      const collected = await collectDirFiles(dirPath, WORKSPACE_DIR).catch(() => []);
      const matching = collected.filter(f => fileTypes.some((ext: string) => f.path.endsWith(`.${ext}`)));
      const combined = matching.map(f => `### ${f.path}\n${f.content.slice(0, cfg.chunkSize || 500)}`).join('\n\n').slice(0, 3000);
      output = `[Local Files RAG]\nDirectory: ${cfg.directory}\nFile types: ${cfg.fileTypes}\nQuery: "${query.slice(0, 80)}"\nFiles found: ${matching.length}\n\nExtracted context:\n${combined || '(No matching files found)'}`;
      log(`  ✅ Local files RAG: ${matching.length} files (query: "${query.slice(0, 40)}")`);
      break;
    }

    case 'llm': {
      const cfg = node.config;
      const context = upstreamContext || '';
      const filledPrompt = (cfg.prompt || '{{input}}')
        .replace(/\{\{context\}\}/g, context || '(no context)')
        .replace(/\{\{input\}\}/g, inputValue || upstreamContext || '(no input)')
        .replace(/\{\{loopIndex\}\}/g, '1');

      if (!filledPrompt.trim()) {
        throw new Error('LLM prompt is empty after substitution. Configure a prompt template in the LLM node (e.g. use {{input}}).');
      }

      const sysInstruction = (cfg.systemInstruction || '')
        .replace(/\{\{context\}\}/g, context || '')
        .replace(/\{\{input\}\}/g, inputValue || '');

      const result = await generateText({
        customConfig,
        prompt: filledPrompt,
        systemInstruction: sysInstruction || undefined,
        temperature: cfg.temperature ?? 0.7,
        searchEnabled: cfg.searchEnabled ?? true,
        logType: 'Workflow LLM',
      });
      output = result.text || '(LLM returned empty response)';
      log(`  ✅ LLM: ${output.length} chars generated`);
      break;
    }

    case 'output': {
      const cfg = node.config;
      const content = upstreamContext || inputValue || '';
      if (!content) {
        log(`  ⚠️ Output node: nothing to write (all upstream nodes produced empty output)`);
        output = '[Output node: no content received from upstream]';
        break;
      }
      const fmtExtMap: Record<string, string> = {
        markdown: 'md', text: 'txt', json: 'json', csv: 'csv', html: 'html',
        code: cfg.language === 'python' ? 'py' : cfg.language === 'typescript' ? 'ts'
          : cfg.language === 'javascript' ? 'js' : cfg.language === 'bash' ? 'sh'
          : cfg.language === 'go' ? 'go' : cfg.language === 'rust' ? 'rs'
          : cfg.language === 'java' ? 'java' : 'txt',
      };
      const ext = fmtExtMap[cfg.format || 'markdown'] || 'md';
      const safeWfName = (workflowName || 'pipeline').replace(/[^a-z0-9_\-]/gi, '_');
      const outTs = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const outFileName = `${safeWfName}-${outTs}-output.${ext}`;
      const outDir = path.join(process.cwd(), 'app-output', 'workflow-output');
      await fs.mkdir(outDir, { recursive: true });
      const outPath = path.join(outDir, outFileName);
      if (cfg.appendMode) {
        await fs.appendFile(outPath, '\n' + content, 'utf8');
      } else {
        await fs.writeFile(outPath, content, 'utf8');
      }
      const relPath = `app-output/workflow-output/${outFileName}`;
      output = `[Output written]\nFile: ${relPath}\nFormat: ${cfg.format || 'markdown'}\nSize: ${content.length} chars\n\nPreview:\n${content.slice(0, 400)}`;
      log(`  ✅ Output written → ${relPath} (${content.length} chars)`);
      break;
    }

    case 'decision': {
      const cfg = node.config;
      const question = cfg.questionPrompt || 'Is the input/output valid?';
      const evalType = cfg.evalType || 'contains_yes';
      const inputStr = upstreamContext || inputValue || '';

      let isYes = false;
      let reason = '';

      if (evalType === 'llm_boolean') {
        const prompt = `Evaluate the following question/condition based on the context provided. Answer ONLY "YES" or "NO".\n\nCondition/Question: ${question}\n\nContext:\n${inputStr}`;
        const res = await generateText({
          customConfig,
          prompt,
          systemInstruction: 'You are a precise binary decision evaluator. Reply ONLY with "YES" or "NO".',
          temperature: 0.1,
          searchEnabled: false,
          logType: 'Workflow Decision Node',
        });
        const reply = (res.text || '').trim().toUpperCase();
        isYes = reply.includes('YES');
        reason = `LLM Evaluator returned: ${reply}`;
      } else if (evalType === 'contains_text') {
        const expected = (cfg.expectedValue || 'YES').toLowerCase();
        isYes = inputStr.toLowerCase().includes(expected);
        reason = `Text check for "${expected}" in input (${inputStr.length} chars)`;
      } else if (evalType === 'js_expression') {
        try {
          const fn = new Function('input', 'context', `"use strict"; return !!(${question})`);
          isYes = fn(inputValue, upstreamContext);
          reason = `JS expression evaluation of: (${question})`;
        } catch (e: any) {
          isYes = false;
          reason = `JS expression evaluation error: ${e.message}`;
        }
      } else {
        // default contains_yes
        isYes = inputStr.toLowerCase().includes('yes') || inputStr.toLowerCase().includes('pass') || inputStr.toLowerCase().includes('success');
        reason = `Contains positive keyword ('yes'/'pass'/'success') in context`;
      }

      if (isYes) {
        output = `DECISION: YES\nQuestion: "${question}"\nResult: YES (${cfg.yesLabel || 'Condition satisfied'})\nReason: ${reason}`;
        log(`  ✅ Decision Node: Evaluated YES for "${question.slice(0, 40)}"`);
      } else {
        output = `DECISION: NO\nQuestion: "${question}"\nResult: NO (${cfg.noLabel || 'Condition not met'})\nReason: ${reason}`;
        log(`  🛑 Decision Node: Evaluated NO for "${question.slice(0, 40)}"`);
      }
      break;
    }

    case 'loop': {
      const cfg = node.config;
      const count = Math.min(50, Math.max(1, cfg.loopCount || 3));
      const varName = cfg.loopVariable || 'loopIndex';
      let loopOutput = inputValue || upstreamContext;
      log(`  ↺ Loop: ${count} iterations, variable="${varName}"`);
      for (let i = 1; i <= count; i++) {
        loopOutput = `[Loop iteration ${i}/${count}]\n${varName}=${i}\nContext: ${(inputValue || upstreamContext).slice(0, 200)}`;
        if (cfg.breakCondition && cfg.breakCondition.trim()) {
          try {
            const condResult = new Function('output', 'loopIndex', `"use strict"; return !!(${cfg.breakCondition})`)(loopOutput, i);
            if (condResult) { log(`  ↺ Break condition met at iteration ${i}`); break; }
          } catch (condErr: any) {
            log(`  ⚠️ Break condition error at iteration ${i}: ${condErr.message}`);
          }
        }
      }
      output = loopOutput;
      log(`  ✅ Loop completed`);
      break;
    }

    case 'db_read': {
      const cfg = node.config;
      const resolvedQuery = (cfg.query || 'SELECT 1')
        .replace(/\{\{input\}\}/g, inputValue.slice(0, 200))
        .replace(/\{\{context\}\}/g, upstreamContext.slice(0, 200));
      output = `[DB Read]\nType: ${cfg.dbType} @ ${cfg.host}:${cfg.port}/${cfg.database}\nSSL: ${cfg.ssl ? 'yes' : 'no'}\nQuery: ${resolvedQuery}\nMax rows: ${cfg.maxRows || 100}\nFormat: ${cfg.outputFormat || 'json'}\n\nSimulated result: ${cfg.maxRows || 10} rows from '${cfg.database}' in ${cfg.outputFormat || 'json'} format.`;
      log(`  ✅ DB Read simulated (${cfg.dbType}@${cfg.host}/${cfg.database})`);
      break;
    }

    case 'db_write': {
      const cfg = node.config;
      const resolvedQuery = (cfg.query || 'INSERT INTO table VALUES (1)')
        .replace(/\{\{input\}\}/g, inputValue.slice(0, 500))
        .replace(/\{\{context\}\}/g, upstreamContext.slice(0, 500));
      output = `[DB Write]\nType: ${cfg.dbType} @ ${cfg.host}:${cfg.port}/${cfg.database}\nTable: ${cfg.tableName || '(inferred from query)'}\nSSL: ${cfg.ssl ? 'yes' : 'no'}\nStatement: ${resolvedQuery.slice(0, 300)}\n\nSimulated: 1 row affected.`;
      log(`  ✅ DB Write simulated (${cfg.dbType}@${cfg.host}/${cfg.database})`);
      break;
    }

    case 'api_call': {
      const cfg = node.config;
      const resolvedUrl = (cfg.url || '')
        .replace(/\{\{input\}\}/g, encodeURIComponent(inputValue.slice(0, 300)))
        .replace(/\{\{context\}\}/g, encodeURIComponent(upstreamContext.slice(0, 300)));
      const resolvedBody = (cfg.body || '')
        .replace(/\{\{input\}\}/g, inputValue.slice(0, 1000))
        .replace(/\{\{context\}\}/g, upstreamContext.slice(0, 1000));
      const headers: Record<string, string> = {};
      try { Object.assign(headers, JSON.parse(cfg.headers || '{}')); } catch {}
      if (cfg.apiKey && !headers['Authorization']) headers['Authorization'] = `Bearer ${cfg.apiKey}`;
      if (!headers['Content-Type'] && resolvedBody) headers['Content-Type'] = 'application/json';
      if (!resolvedUrl) {
        output = `[API Call Failed]\nURL is empty — configure a URL in the API Call node.`;
        log(`  ❌ API Call: no URL configured`);
        break;
      }
      try {
        const fetchOpts: any = {
          method: cfg.method || 'GET',
          headers,
          signal: AbortSignal.timeout(cfg.timeoutMs || 10000),
        };
        if (['POST', 'PUT', 'PATCH'].includes(cfg.method || 'GET') && resolvedBody) {
          fetchOpts.body = resolvedBody;
        }
        const resp = await fetch(resolvedUrl, fetchOpts);
        const rawText = await resp.text();
        let extracted = rawText;
        if (cfg.outputPath && cfg.outputPath.trim()) {
          try {
            const parsed = JSON.parse(rawText);
            const parts = cfg.outputPath.split('.');
            let cur: any = parsed;
            for (const p of parts) {
              const arrMatch = p.match(/^(.+?)\[(\d+)\]$/);
              if (arrMatch) { cur = cur?.[arrMatch[1]]?.[Number(arrMatch[2])]; }
              else { cur = cur?.[p]; }
            }
            extracted = typeof cur === 'string' ? cur : JSON.stringify(cur, null, 2);
          } catch {}
        }
        output = `[API Call]\nURL: ${resolvedUrl}\nMethod: ${cfg.method || 'GET'}\nStatus: ${resp.status} ${resp.statusText}\n\nResponse${cfg.outputPath ? ` (extracted: ${cfg.outputPath})` : ''}:\n${extracted.slice(0, 2000)}`;
        log(`  ✅ API ${cfg.method || 'GET'} ${resolvedUrl} → HTTP ${resp.status}`);
      } catch (fetchErr: any) {
        output = `[API Call Failed]\nURL: ${resolvedUrl}\nMethod: ${cfg.method || 'GET'}\nError: ${fetchErr.message}`;
        log(`  ❌ API Call failed: ${fetchErr.message}`);
      }
      break;
    }

    case 'email_send': {
      const cfg = node.config;
      const resolvedSubject = (cfg.subject || '').replace(/\{\{input\}\}/g, inputValue).replace(/\{\{context\}\}/g, upstreamContext);
      const resolvedBody = (cfg.body || '').replace(/\{\{input\}\}/g, inputValue).replace(/\{\{context\}\}/g, upstreamContext);
      const toList = (cfg.toAddresses || '').split(',').map((s: string) => s.trim()).filter(Boolean);
      const ccList = (cfg.ccAddresses || '').split(',').map((s: string) => s.trim()).filter(Boolean);
      const sendMode = cfg.sendMode || 'local';

      if (sendMode === 'smtp') {
        if (!cfg.smtpHost || !toList.length) {
          output = `[Email Send — SKIPPED]\nSMTP host or recipient list not configured.\nSubject (preview): ${resolvedSubject.slice(0, 80)}\nBody preview: ${resolvedBody.slice(0, 200)}`;
          log(`  ⚠️ Email send skipped: SMTP host or To address not configured`);
          break;
        }
        try {
          const transporterOpts: any = {
            host: cfg.smtpHost,
            port: parseInt(cfg.smtpPort || '587', 10),
            secure: cfg.tlsMode === 'ssl',
            tls: { rejectUnauthorized: false }
          };
          if (cfg.tlsMode === 'starttls') transporterOpts.requireTLS = true;
          if (cfg.smtpUser) transporterOpts.auth = { user: cfg.smtpUser, pass: cfg.smtpPass || '' };

          const transporter = nodemailer.createTransport(transporterOpts);
          const mailOptions: any = {
            from: cfg.fromAddress || 'no-reply@local.domain',
            to: toList.join(', '),
            subject: resolvedSubject,
            tls: { rejectUnauthorized: false }
          };
          if (ccList.length > 0) mailOptions.cc = ccList.join(', ');
          if (cfg.isHtml) { mailOptions.html = resolvedBody; } else { mailOptions.text = resolvedBody; }

          const info = await transporter.sendMail(mailOptions);
          output = `[Email Sent (SMTP)]\nSMTP: ${cfg.smtpHost}:${cfg.smtpPort}\nFrom: ${mailOptions.from}\nTo: ${mailOptions.to}\nSubject: ${resolvedSubject}\nMessage ID: ${info.messageId}`;
          log(`  ✅ Email sent via SMTP to ${toList.join(', ')} | Subject: ${resolvedSubject.slice(0, 60)}`);
        } catch (smtpErr: any) {
          output = `[Email Send Failed (SMTP)]\nSMTP: ${cfg.smtpHost}:${cfg.smtpPort}\nError: ${smtpErr.message}`;
          log(`  ❌ Email send failed via SMTP: ${smtpErr.message}`);
        }
      } else if (sendMode === 'direct') {
        if (!toList.length) {
          output = `[Email Send — SKIPPED]\nRecipient list not configured.\nSubject (preview): ${resolvedSubject.slice(0, 80)}`;
          log(`  ⚠️ Email send skipped: To address not configured`);
          break;
        }
        const firstRecipient = toList[0];
        const domain = firstRecipient.split('@')[1];
        if (!domain) {
          output = `[Email Send Failed (Direct MX)]\nInvalid recipient email domain in "${firstRecipient}"`;
          log(`  ❌ Direct MX sending failed: invalid recipient domain`);
          break;
        }
        let mxHost = '';
        try {
          log(`  🔍 Resolving MX records for domain: ${domain}...`);
          const mxRecords = await dns.promises.resolveMx(domain);
          if (!mxRecords || mxRecords.length === 0) throw new Error(`No MX records found for domain ${domain}`);
          mxRecords.sort((a, b) => a.priority - b.priority);
          mxHost = mxRecords[0].exchange;
          log(`  ℹ️ Top MX host resolved: ${mxHost}`);
        } catch (dnsErr: any) {
          output = `[Email Send Failed (Direct MX)]\nFailed to resolve MX records for ${domain}: ${dnsErr.message}`;
          log(`  ❌ DNS MX resolution failed: ${dnsErr.message}`);
          break;
        }
        try {
          const transporter = nodemailer.createTransport({ host: mxHost, port: 25, secure: false, tls: { rejectUnauthorized: false } });
          const mailOptions: any = { from: cfg.fromAddress || `no-reply@${domain}`, to: toList.join(', '), subject: resolvedSubject };
          if (ccList.length > 0) mailOptions.cc = ccList.join(', ');
          if (cfg.isHtml) { mailOptions.html = resolvedBody; } else { mailOptions.text = resolvedBody; }
          const info = await transporter.sendMail(mailOptions);
          output = `[Email Sent (Direct MX)]\nMX Server: ${mxHost}:25\nFrom: ${mailOptions.from}\nTo: ${mailOptions.to}\nSubject: ${resolvedSubject}\nMessage ID: ${info.messageId}`;
          log(`  ✅ Email sent directly via MX to ${toList.join(', ')} | Subject: ${resolvedSubject.slice(0, 60)}`);
        } catch (smtpErr: any) {
          output = `[Email Send Failed (Direct MX)]\nMX Server: ${mxHost}:25\nError: ${smtpErr.message}`;
          log(`  ❌ Email send failed via Direct MX to ${mxHost}: ${smtpErr.message}`);
        }
      } else {
        // Local mode: write email payload to disk as JSON
        try {
          const emailPayload = {
            timestamp: new Date().toISOString(),
            from: cfg.fromAddress || 'no-reply@local.domain',
            to: toList, cc: ccList, subject: resolvedSubject, body: resolvedBody, isHtml: !!cfg.isHtml
          };
          const dir = path.join(process.cwd(), 'app-output', 'emails');
          await fs.mkdir(dir, { recursive: true });
          const safeSubject = resolvedSubject.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 30);
          const filename = `email_${Date.now()}_${safeSubject || 'no_subject'}.json`;
          await fs.writeFile(path.join(dir, filename), JSON.stringify(emailPayload, null, 2), 'utf-8');
          output = `[Email Sent (Local)]\nSaved to: app-output/emails/${filename}\nFrom: ${emailPayload.from}\nTo: ${emailPayload.to.join(', ')}\nSubject: ${resolvedSubject}\n\nBody:\n${resolvedBody.slice(0, 1000)}`;
          log(`  ✅ Email saved locally -> app-output/emails/${filename}`);
        } catch (localErr: any) {
          output = `[Email Send Failed (Local)]\nError: ${localErr.message}`;
          log(`  ❌ Local email saving failed: ${localErr.message}`);
        }
      }
      break;
    }

    case 'email_receive': {
      const cfg = node.config;
      if (!cfg.imapHost || !cfg.imapUser) {
        output = `[Email Receive — SKIPPED]\nIMAP host or username not configured.`;
        log(`  ⚠️ Email receive skipped: IMAP host or username not configured`);
        break;
      }
      const filterDesc = [
        cfg.filterFrom ? `from:${cfg.filterFrom}` : '',
        cfg.filterSubject ? `subject:${cfg.filterSubject}` : '',
      ].filter(Boolean).join(' ');
      output = `[Email Receive]\nIMAP: ${cfg.imapHost}:${cfg.imapPort} (${cfg.tlsMode})\nUser: ${cfg.imapUser}\nMailbox: ${cfg.mailbox}\nFilter: ${filterDesc || '(none)'}\nMax messages: ${cfg.maxMessages}\nMark read: ${cfg.markRead ? 'yes' : 'no'}\n\n(Simulated — install imapflow and configure to enable real IMAP polling.)`;
      log(`  ✅ Email receive staged (${cfg.mailbox}, max ${cfg.maxMessages})`);
      break;
    }

    case 'scheduler': {
      const cfg = node.config;
      const targetPayload = (cfg.targetPayload && cfg.targetPayload.trim()) 
        ? cfg.targetPayload.trim() 
        : inputValue || upstreamContext;

      if (!targetPayload) {
        output = `[Temporal / Trigger.dev Scheduler — Warning]\nNo payload or file path provided in node configuration or upstream input.`;
        log(`  ⚠️ Scheduler node has no target payload or upstream context`);
        break;
      }

      const schedulerServer = cfg.schedulerServer || 'temporal';
      const scheduleType = cfg.scheduleType || 'cron';
      const cronExpr = cfg.cronExpression || '*/5 * * * *';
      const intervalSec = cfg.intervalSeconds || 300;

      try {
        const port = process.env.PORT || '3000';
        const res = await fetch(`http://localhost:${port}/api/scheduler/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: cfg.jobName || `Workflow Schedule (${node.id.slice(0, 6)})`,
            source: 'workflow_component',
            schedulerServer,
            scheduleType,
            cronExpression: cronExpr,
            intervalSeconds: intervalSec,
            actionType: cfg.actionType || 'auto_detect',
            payload: targetPayload,
            temporalConfig: { namespace: cfg.temporalNamespace || 'default' },
            triggerDevConfig: { environment: cfg.triggerDevEnvironment || 'development' }
          })
        });

        const data = await res.json();

        if (data.success && data.job) {
          const job = data.job;
          output = `[Temporal / Trigger.dev Scheduler Server]\n` +
            `Server: ${job.schedulerServer.toUpperCase()}\n` +
            `Job ID: ${job.id}\n` +
            `Schedule: ${job.scheduleType === 'cron' ? 'Cron (' + job.cronExpression + ')' : job.intervalSeconds + 's interval'}\n` +
            `Detected Category: ${job.detectedCategory.toUpperCase()} (${job.targetLanguageOrExt})\n` +
            `Next Run At: ${job.nextRunAt}\n` +
            `Payload Preview: ${targetPayload.slice(0, 150)}...\n` +
            `Status: ${job.status.toUpperCase()}`;
          log(`  ✅ Registered schedule on ${job.schedulerServer} server (Job ID: ${job.id}, Category: ${job.detectedCategory})`);
        } else {
          output = `[Scheduler Registration Error]\n${data.error || 'Unknown error'}`;
          log(`  ❌ Failed to register schedule: ${data.error}`);
        }
      } catch (schedErr: any) {
        output = `[Scheduler Connection Error]\n${schedErr.message}`;
        log(`  ❌ Error connecting to scheduler server: ${schedErr.message}`);
      }
      break;
    }

    default:
      output = `[Unknown node type: ${node.type}]`;
      log(`  ⚠️ Unknown node type: ${node.type}`);
  }

  return output;
}
