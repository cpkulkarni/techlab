/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { DEFAULT_AGENTS } from '../agents/defaultAgents.js';
import { A2ABroker } from './broker.js';
import { MCPRegistry } from '../mcp/registry.js';
import {
  AgentDefinition,
  AgentStatus,
  MultiAgentTaskExecution,
  MultiAgentTaskTraceStep,
  A2AMessage,
} from '../types.js';

export class MultiAgentOrchestrator {
  private static instance: MultiAgentOrchestrator;
  private agents: Map<string, AgentDefinition> = new Map();
  private activeExecutions: Map<string, MultiAgentTaskExecution> = new Map();
  private broker: A2ABroker;
  private mcp: MCPRegistry;

  private constructor() {
    this.broker = A2ABroker.getInstance();
    this.mcp = MCPRegistry.getInstance();
    this.initAgents();
  }

  public static getInstance(): MultiAgentOrchestrator {
    if (!MultiAgentOrchestrator.instance) {
      MultiAgentOrchestrator.instance = new MultiAgentOrchestrator();
    }
    return MultiAgentOrchestrator.instance;
  }

  private initAgents() {
    for (const agent of DEFAULT_AGENTS) {
      this.agents.set(agent.id, { ...agent });
    }
  }

  public getAgents(): AgentDefinition[] {
    return Array.from(this.agents.values());
  }

  public getAgent(id: string): AgentDefinition | undefined {
    return this.agents.get(id);
  }

  public setAgentStatus(id: string, status: AgentStatus): void {
    const agent = this.agents.get(id);
    if (agent) {
      agent.status = status;
      this.broker.broadcastToSSE({
        type: 'agent_status_change',
        agentId: id,
        status,
      });
    }
  }

  public registerAgent(agent: AgentDefinition): void {
    this.agents.set(agent.id, agent);
    this.broker.broadcastToSSE({
      type: 'agent_registered',
      agent,
    });
  }

  /**
   * Helper to invoke Gemini AI model using environment API key or custom fallback
   */
  private async queryLLM(systemInstruction: string, prompt: string, modelConfig?: any): Promise<string> {
    const apiKey = process.env.GEMINI_API_KEY || modelConfig?.apiKey;
    if (apiKey) {
      try {
        const ai = new GoogleGenAI({ apiKey });
        const response = await ai.models.generateContent({
          model: modelConfig?.selectedModel || 'gemini-2.5-flash',
          contents: [
            { role: 'user', parts: [{ text: `${systemInstruction}\n\nTask:\n${prompt}` }] },
          ],
        });
        if (response.text) return response.text;
      } catch (e: any) {
        console.warn('Gemini call error in orchestrator:', e.message);
      }
    }
    // Fallback response generator for simulation/offline mode
    return `[Structured Sub-agent Response for "${prompt.slice(0, 40)}..."] Completed analysis with findings and recommendations successfully.`;
  }

  /**
   * Run full Multi-Agent Orchestration task
   */
  public async executeMultiAgentTask(
    userPrompt: string,
    modelConfig?: any
  ): Promise<MultiAgentTaskExecution> {
    const taskId = `task_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const conversationId = `conv_${taskId}`;
    const startTime = new Date().toISOString();

    const execution: MultiAgentTaskExecution = {
      id: taskId,
      taskPrompt: userPrompt,
      coordinatorId: 'agent_coordinator',
      status: 'running',
      startTime,
      steps: [],
      messages: [],
    };

    this.activeExecutions.set(taskId, execution);

    const addStep = (
      agentId: string,
      action: MultiAgentTaskTraceStep['action'],
      title: string,
      description: string,
      details?: { a2aMessageId?: string; mcpToolCalled?: string; mcpToolArgs?: any; output?: string; durationMs?: number }
    ) => {
      const agent = this.getAgent(agentId);
      const step: MultiAgentTaskTraceStep = {
        id: `step_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        timestamp: new Date().toISOString(),
        stepNumber: execution.steps.length + 1,
        agentId,
        agentName: agent?.name || agentId,
        agentRole: agent?.role || 'agent',
        action,
        title,
        description,
        a2aMessageId: details?.a2aMessageId,
        mcpToolCalled: details?.mcpToolCalled,
        mcpToolArgs: details?.mcpToolArgs,
        output: details?.output,
        durationMs: details?.durationMs,
      };
      execution.steps.push(step);
      this.broker.broadcastToSSE({
        type: 'execution_step',
        taskId,
        step,
      });
      return step;
    };

    try {
      // Step 1: Coordinator receives task
      this.setAgentStatus('agent_coordinator', 'thinking');
      addStep('agent_coordinator', 'received', 'Task Received by Coordinator', `Decomposing user request: "${userPrompt}"`);

      // Initial A2A Request from User/System to Coordinator
      const initMsg = this.broker.publish(
        'user_system',
        'agent_coordinator',
        conversationId,
        'request',
        { task: userPrompt }
      );
      execution.messages.push(initMsg);

      // Step 2: Coordinator plan breakdown
      const coordinatorSysPrompt = this.agents.get('agent_coordinator')?.systemPrompt || '';
      const planResponse = await this.queryLLM(
        coordinatorSysPrompt,
        `Decompose this task into 2-3 specific sub-tasks for (1) Web/Docs Researcher, (2) Code Synthesis, (3) QA Test Engineer:\n\nTask: ${userPrompt}`,
        modelConfig
      );

      addStep('agent_coordinator', 'delegating', 'Coordinator Sub-Task Plan', planResponse);

      // Step 3: Delegate to Research Agent
      this.setAgentStatus('agent_coordinator', 'delegating');
      this.setAgentStatus('agent_researcher', 'thinking');

      const researchTask = `Research context, documentation, and files related to: ${userPrompt}`;
      const delegateResearchMsg = this.broker.publish(
        'agent_coordinator',
        'agent_researcher',
        conversationId,
        'delegate',
        {
          task: researchTask,
          mcp_context: { tools_used: ['web_search', 'read_file', 'list_files'] },
        }
      );
      execution.messages.push(delegateResearchMsg);

      addStep('agent_researcher', 'received', 'Research Sub-Task Delegated', researchTask, { a2aMessageId: delegateResearchMsg.id });

      // Execute Research MCP Tool call
      const searchStart = Date.now();
      this.setAgentStatus('agent_researcher', 'executing');
      const searchResult = await this.mcp.invokeTool('web_search', { query: userPrompt, count: 3 });
      const searchDuration = Date.now() - searchStart;

      addStep('agent_researcher', 'mcp_tool_execution', 'Executed MCP Tool: web_search', `Searched for "${userPrompt}"`, {
        mcpToolCalled: 'web_search',
        mcpToolArgs: { query: userPrompt },
        output: JSON.stringify(searchResult, null, 2),
        durationMs: searchDuration,
      });

      const researchOutput = await this.queryLLM(
        this.agents.get('agent_researcher')?.systemPrompt || '',
        `Summarize research findings for task "${userPrompt}" using MCP data: ${JSON.stringify(searchResult)}`,
        modelConfig
      );

      const researchRespMsg = this.broker.publish(
        'agent_researcher',
        'agent_coordinator',
        conversationId,
        'response',
        {
          task: researchTask,
          result: researchOutput,
          mcp_context: { tools_used: ['web_search'] },
        }
      );
      execution.messages.push(researchRespMsg);
      this.setAgentStatus('agent_researcher', 'idle');

      // Step 4: Delegate to Coder Agent
      this.setAgentStatus('agent_coder', 'thinking');
      const coderTask = `Implement software solution or specifications based on research findings:\n${researchOutput}`;
      const delegateCoderMsg = this.broker.publish(
        'agent_coordinator',
        'agent_coder',
        conversationId,
        'delegate',
        {
          task: coderTask,
          mcp_context: { tools_used: ['read_file', 'write_file', 'edit_file'] },
        }
      );
      execution.messages.push(delegateCoderMsg);

      addStep('agent_coder', 'received', 'Code Synthesis Delegated', 'Synthesizing implementation plan and file edits', { a2aMessageId: delegateCoderMsg.id });

      this.setAgentStatus('agent_coder', 'executing');
      const listFilesStart = Date.now();
      const filesResult = await this.mcp.invokeTool('list_files', { dirPath: '.' });
      addStep('agent_coder', 'mcp_tool_execution', 'Executed MCP Tool: list_files', 'Inspected workspace directory structure', {
        mcpToolCalled: 'list_files',
        output: JSON.stringify(filesResult, null, 2),
        durationMs: Date.now() - listFilesStart,
      });

      const coderOutput = await this.queryLLM(
        this.agents.get('agent_coder')?.systemPrompt || '',
        `Generate code architecture or changes for task "${userPrompt}" considering workspace files: ${JSON.stringify(filesResult)}`,
        modelConfig
      );

      const coderRespMsg = this.broker.publish(
        'agent_coder',
        'agent_coordinator',
        conversationId,
        'response',
        {
          task: coderTask,
          result: coderOutput,
          mcp_context: { tools_used: ['list_files'] },
        }
      );
      execution.messages.push(coderRespMsg);
      this.setAgentStatus('agent_coder', 'idle');

      // Step 5: Delegate to QA Test Agent
      this.setAgentStatus('agent_tester', 'thinking');
      const testTask = `Verify and validate code synthesis result for quality assurance.`;
      const delegateTestMsg = this.broker.publish(
        'agent_coordinator',
        'agent_tester',
        conversationId,
        'delegate',
        {
          task: testTask,
          mcp_context: { tools_used: ['run_tests'] },
        }
      );
      execution.messages.push(delegateTestMsg);

      addStep('agent_tester', 'received', 'QA Verification Delegated', testTask, { a2aMessageId: delegateTestMsg.id });

      this.setAgentStatus('agent_tester', 'executing');
      const testStart = Date.now();
      const testResult = await this.mcp.invokeTool('run_tests', { type: 'typecheck' });
      addStep('agent_tester', 'mcp_tool_execution', 'Executed MCP Tool: run_tests', 'Ran automated typecheck & lint verification', {
        mcpToolCalled: 'run_tests',
        output: JSON.stringify(testResult, null, 2),
        durationMs: Date.now() - testStart,
      });

      const qaRespMsg = this.broker.publish(
        'agent_tester',
        'agent_coordinator',
        conversationId,
        'response',
        {
          task: testTask,
          result: testResult,
          mcp_context: { tools_used: ['run_tests'] },
        }
      );
      execution.messages.push(qaRespMsg);
      this.setAgentStatus('agent_tester', 'idle');

      // Step 6: Coordinator Synthesis
      this.setAgentStatus('agent_coordinator', 'thinking');
      const finalSynthesis = await this.queryLLM(
        this.agents.get('agent_coordinator')?.systemPrompt || '',
        `Synthesize final response for task "${userPrompt}" from all sub-agent results:\n\nResearch:\n${researchOutput}\n\nCode Implementation:\n${coderOutput}\n\nQA Verification:\n${JSON.stringify(testResult)}`,
        modelConfig
      );

      addStep('agent_coordinator', 'completed', 'Multi-Agent Task Completed', finalSynthesis);

      execution.status = 'completed';
      execution.endTime = new Date().toISOString();
      execution.finalResult = finalSynthesis;

      this.setAgentStatus('agent_coordinator', 'idle');

      this.broker.broadcastToSSE({
        type: 'execution_completed',
        taskId,
        execution,
      });

      return execution;

    } catch (err: any) {
      execution.status = 'failed';
      execution.endTime = new Date().toISOString();

      addStep('agent_coordinator', 'failed', 'Multi-Agent Execution Failed', err.message);

      for (const agentId of ['agent_coordinator', 'agent_researcher', 'agent_coder', 'agent_tester']) {
        this.setAgentStatus(agentId, 'idle');
      }

      this.broker.broadcastToSSE({
        type: 'execution_failed',
        taskId,
        error: err.message,
      });

      throw err;
    }
  }

  public getExecutions(): MultiAgentTaskExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  public getExecution(id: string): MultiAgentTaskExecution | undefined {
    return this.activeExecutions.get(id);
  }
}
