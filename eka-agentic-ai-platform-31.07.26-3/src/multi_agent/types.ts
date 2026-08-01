/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// ============================================================================
// MCP (MODEL CONTEXT PROTOCOL) TYPES
// ============================================================================

export interface MCPToolSchema {
  type?: string;
  properties?: Record<string, {
    type: string;
    description: string;
    enum?: string[];
  }>;
  required?: string[];
}

export interface MCPTool {
  name: string;
  description: string;
  category: 'file' | 'search' | 'command' | 'test' | 'workflow' | 'system' | 'code';
  schema: MCPToolSchema;
  enabled: boolean;
  handler: (args: Record<string, any>) => Promise<any>;
}

export interface MCPResource {
  uri: string;
  name: string;
  description: string;
  mimeType: string;
  category: 'workspace' | 'system' | 'state' | 'logs';
  readHandler: () => Promise<{ contents: Array<{ uri: string; mimeType: string; text: string }> }>;
}

export interface MCPPrompt {
  name: string;
  description: string;
  arguments: Array<{ name: string; description: string; required?: boolean }>;
  template: string;
}

export interface MCPServerInfo {
  id: string;
  name: string;
  version: string;
  status: 'connected' | 'syncing' | 'disconnected';
  endpoint?: string;
  toolsCount: number;
  resourcesCount: number;
}

// ============================================================================
// A2A (AGENT-TO-AGENT) PROTOCOL TYPES
// ============================================================================

export type A2AMessageType = 
  | 'request' 
  | 'response' 
  | 'delegate' 
  | 'error' 
  | 'heartbeat' 
  | 'hitl_request' 
  | 'hitl_approval' 
  | 'hitl_rejection' 
  | 'human_direct';

export interface A2APayload {
  task: string;
  data?: Record<string, any>;
  mcp_context?: {
    tools_used?: string[];
    resources_accessed?: string[];
    prompt_used?: string;
  };
  result?: any;
  error?: string;
  human_feedback?: string;
  modified_output?: string;
  approved?: boolean;
}

export interface A2AMessage {
  id: string;
  sender_id: string;
  recipient_id: string | 'broadcast';
  conversation_id: string;
  message_type: A2AMessageType;
  payload: A2APayload;
  timestamp: string; // ISO-8601
  channel?: 'human_agent' | 'human_to_agent' | 'agent_to_human' | 'agent_to_agent' | 'hitl';
}

// ============================================================================
// AGENT CONFIGURATION & ORCHESTRATION TYPES
// ============================================================================

export type AgentStatus = 'idle' | 'thinking' | 'delegating' | 'executing' | 'awaiting_hitl' | 'error';

export interface AgentCapability {
  id: string;
  name: string;
  description: string;
}

export interface AgentDefinition {
  id: string;
  name: string;
  role: 'coordinator' | 'researcher' | 'coder' | 'tester' | 'hitl_verifier' | 'specialist';
  description: string;
  systemPrompt: string;
  capabilities: string[];
  mcpToolsAllowed: string[]; // List of MCP tool names this agent can invoke
  status: AgentStatus;
  avatarIcon?: string;
  color?: string;
}

export interface MultiAgentTaskTraceStep {
  id: string;
  timestamp: string;
  stepNumber: number;
  agentId: string;
  agentName: string;
  agentRole: string;
  action: 'received' | 'delegating' | 'mcp_tool_execution' | 'hitl_review' | 'completed' | 'failed';
  title: string;
  description: string;
  a2aMessageId?: string;
  mcpToolCalled?: string;
  mcpToolArgs?: Record<string, any>;
  output?: string;
  durationMs?: number;
  hitlApproved?: boolean;
  hitlFeedback?: string;
}

export interface MultiAgentTaskExecution {
  id: string;
  taskPrompt: string;
  coordinatorId: string;
  status: 'running' | 'completed' | 'failed';
  startTime: string;
  endTime?: string;
  steps: MultiAgentTaskTraceStep[];
  finalResult?: string;
  messages: A2AMessage[];
}
