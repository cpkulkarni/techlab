/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router, Request, Response } from 'express';
import { MultiAgentOrchestrator } from '../../multi_agent/a2a/orchestrator.js';
import { A2ABroker } from '../../multi_agent/a2a/broker.js';
import { MCPRegistry } from '../../multi_agent/mcp/registry.js';
import { A2AMessageType, A2APayload } from '../../multi_agent/types.js';

const router = Router();
const orchestrator = MultiAgentOrchestrator.getInstance();
const broker = A2ABroker.getInstance();
const mcp = MCPRegistry.getInstance();

// Feature Flag Check
const isMultiAgentEnabled = () => process.env.ENABLE_MULTI_AGENT !== 'false';

// 1. Config & Status
router.get('/config', (req: Request, res: Response) => {
  res.json({
    enabled: isMultiAgentEnabled(),
    mode: 'multi_agent_a2a_mcp',
    activeAgentsCount: orchestrator.getAgents().length,
    mcpToolsCount: mcp.getTools().length,
    mcpResourcesCount: mcp.getResources().length,
  });
});

// 2. Agents List & Registration
router.get('/agents', (req: Request, res: Response) => {
  res.json({
    enabled: isMultiAgentEnabled(),
    agents: orchestrator.getAgents(),
  });
});

router.post('/agents/register', (req: Request, res: Response) => {
  const { id, name, role, description, systemPrompt, capabilities, mcpToolsAllowed } = req.body;
  if (!id || !name || !role) {
    return res.status(400).json({ error: 'Missing required agent properties: id, name, role' });
  }

  const newAgent = {
    id,
    name,
    role: role || 'specialist',
    description: description || '',
    systemPrompt: systemPrompt || 'You are an autonomous AI Agent.',
    capabilities: capabilities || ['custom_task'],
    mcpToolsAllowed: mcpToolsAllowed || ['read_file', 'list_files'],
    status: 'idle' as const,
    avatarIcon: 'Bot',
    color: '#8b5cf6',
  };

  orchestrator.registerAgent(newAgent);
  res.json({ success: true, agent: newAgent });
});

// 3. A2A Messages & Direct Dispatch
router.get('/a2a/messages', (req: Request, res: Response) => {
  const { conversationId, agentId, messageType, limit } = req.query;
  const history = broker.getHistory({
    conversationId: conversationId as string,
    agentId: agentId as string,
    messageType: messageType as A2AMessageType,
    limit: limit ? parseInt(limit as string, 10) : 100,
  });
  res.json({ messages: history });
});

router.post('/a2a/send', (req: Request, res: Response) => {
  const { sender_id, recipient_id, conversation_id, message_type, payload } = req.body;
  if (!sender_id || !recipient_id || !message_type || !payload) {
    return res.status(400).json({ error: 'Invalid A2A payload structure' });
  }

  const msg = broker.publish(
    sender_id,
    recipient_id,
    conversation_id || `conv_${Date.now()}`,
    message_type as A2AMessageType,
    payload as A2APayload
  );

  res.json({ success: true, message: msg });
});

// 4. MCP Tools & Execution
router.get('/mcp/tools', (req: Request, res: Response) => {
  res.json({ tools: mcp.getTools() });
});

router.post('/mcp/tools/toggle', (req: Request, res: Response) => {
  const { name, enabled } = req.body;
  if (!name || typeof enabled !== 'boolean') {
    return res.status(400).json({ error: 'Missing name or enabled parameter' });
  }
  const success = mcp.toggleTool(name, enabled);
  res.json({ success, name, enabled });
});

router.post('/mcp/invoke', async (req: Request, res: Response) => {
  const { name, args } = req.body;
  if (!name) {
    return res.status(400).json({ error: 'Tool name is required' });
  }
  try {
    const result = await mcp.invokeTool(name, args || {});
    res.json({ success: true, tool: name, result });
  } catch (err: any) {
    res.status(500).json({ success: false, tool: name, error: err.message });
  }
});

// 5. MCP Resources & Servers
router.get('/mcp/resources', (req: Request, res: Response) => {
  res.json({ resources: mcp.getResources() });
});

router.get('/mcp/resources/read', async (req: Request, res: Response) => {
  const uri = req.query.uri as string;
  if (!uri) {
    return res.status(400).json({ error: 'Resource URI required' });
  }
  try {
    const data = await mcp.readResource(uri);
    res.json({ success: true, uri, data });
  } catch (err: any) {
    res.status(404).json({ success: false, error: err.message });
  }
});

router.get('/mcp/servers', (req: Request, res: Response) => {
  res.json({ servers: mcp.getServers() });
});

// 6. Multi-Agent Task Orchestration
router.post('/orchestrate', async (req: Request, res: Response) => {
  const { prompt, modelConfig } = req.body;
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  try {
    const execution = await orchestrator.executeMultiAgentTask(prompt, modelConfig);
    res.json({ success: true, execution });
  } catch (err: any) {
    res.status(500).json({ success: false, error: err.message });
  }
});

router.get('/executions', (req: Request, res: Response) => {
  res.json({ executions: orchestrator.getExecutions() });
});

// 7. Real-Time SSE Event Stream
router.get('/stream', (req: Request, res: Response) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  // Send initial handshake
  res.write(`data: ${JSON.stringify({ type: 'connected', timestamp: new Date().toISOString() })}\n\n`);

  const sendFn = (data: string) => {
    res.write(data);
  };

  const unregister = broker.registerSSEClient(sendFn);

  req.on('close', () => {
    unregister();
  });
});

export default router;
