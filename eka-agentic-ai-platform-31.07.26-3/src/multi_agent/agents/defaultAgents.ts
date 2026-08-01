/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { AgentDefinition } from '../types.js';

export const DEFAULT_AGENTS: AgentDefinition[] = [
  {
    id: 'agent_coordinator',
    name: 'Coordinator Agent',
    role: 'coordinator',
    description: 'Master orchestrator that breaks down user tasks, assigns sub-tasks to domain agents via A2A protocol, and synthesizes final responses.',
    systemPrompt: `You are the Coordinator Agent. Analyze the incoming user task, decompose it into parallel or sequential sub-tasks for specialized agents (Research Agent, Code Synthesis Agent, QA Agent), delegate via A2A protocol, and combine their outputs into a cohesive final result.`,
    capabilities: ['task_decomposition', 'a2a_delegation', 'response_synthesis', 'mcp_orchestration'],
    mcpToolsAllowed: ['read_file', 'list_files', 'run_workflow'],
    status: 'idle',
    avatarIcon: 'Cpu',
    color: '#6366f1', // Indigo
  },
  {
    id: 'agent_researcher',
    name: 'Web & Docs Research Agent',
    role: 'researcher',
    description: 'Specialist agent for deep web searching, documentation retrieval, code pattern lookup, and knowledge extraction.',
    systemPrompt: `You are the Web & Docs Research Agent. Your job is to search the web, fetch technical documentation, analyze existing codebase files, and provide structured insights back to the requesting agent via A2A.`,
    capabilities: ['web_search', 'doc_retrieval', 'file_inspection', 'summarization'],
    mcpToolsAllowed: ['web_search', 'read_file', 'list_files', 'read_workspace_logs'],
    status: 'idle',
    avatarIcon: 'Globe',
    color: '#06b6d4', // Cyan
  },
  {
    id: 'agent_coder',
    name: 'Code Synthesis Agent',
    role: 'coder',
    description: 'Specialist software architect and developer for reading codebase files, writing clean TypeScript/React code, and executing file edits.',
    systemPrompt: `You are the Code Synthesis Agent. You take code specifications or feature requests, inspect existing source code, generate robust TypeScript implementations, and write or edit files using MCP tools.`,
    capabilities: ['code_generation', 'file_creation', 'file_editing', 'refactoring'],
    mcpToolsAllowed: ['read_file', 'write_file', 'edit_file', 'list_files', 'execute_command'],
    status: 'idle',
    avatarIcon: 'Code2',
    color: '#10b981', // Emerald
  },
  {
    id: 'agent_tester',
    name: 'QA & Test Engineer Agent',
    role: 'tester',
    description: 'Specialist agent that validates code changes, executes automated tests, analyzes runtime errors, and reports quality assurance results.',
    systemPrompt: `You are the QA & Test Engineer Agent. You verify code changes, generate test specifications, run unit/integration tests using MCP tools, and verify that all features work without errors.`,
    capabilities: ['test_runner', 'error_analysis', 'linting', 'verification'],
    mcpToolsAllowed: ['run_tests', 'read_file', 'execute_command', 'read_workspace_logs'],
    status: 'idle',
    avatarIcon: 'CheckSquare',
    color: '#f59e0b', // Amber
  },
  {
    id: 'agent_hitl_verifier',
    name: 'Human-in-the-Loop Verifier Agent',
    role: 'hitl_verifier',
    description: 'Gatekeeper agent that intercepts sub-task outputs, enforces human review checkpoints, records human approvals/modifications, and directs subtask re-execution.',
    systemPrompt: `You are the Human-in-the-Loop Verifier Agent. Your responsibility is to pause multi-agent workflows at key milestones, present generated outputs for human operator inspection, collect human verification feedback, and emit approval or change signals back to the Coordinator.`,
    capabilities: ['output_interception', 'human_verification', 'feedback_ingestion', 'checkpoint_approval'],
    mcpToolsAllowed: ['read_file', 'list_files', 'read_workspace_logs'],
    status: 'idle',
    avatarIcon: 'ShieldCheck',
    color: '#ec4899', // Pink
  },
  {
    id: 'agent_scheduler',
    name: 'Temporal & Trigger.dev Scheduler Agent',
    role: 'specialist',
    description: 'Specialist agent that interfaces with Temporal.io and Trigger.dev scheduler servers to register cron/interval schedules, execute code snippets, open documents, and query job execution status.',
    systemPrompt: `You are the Temporal & Trigger.dev Scheduler Specialist Agent. You receive outputs from other agents or user requests, determine whether the payload is code or a local file path, construct Temporal/Trigger.dev schedules (cron expressions or interval seconds), register jobs, and query job execution statuses.`,
    capabilities: ['schedule_registration', 'payload_classification', 'temporal_trigger_integration', 'job_status_monitoring'],
    mcpToolsAllowed: ['schedule_temporal_trigger_job', 'get_scheduler_job_status', 'trigger_scheduled_job_now', 'cancel_scheduled_job', 'read_file'],
    status: 'idle',
    avatarIcon: 'Clock',
    color: '#8b5cf6', // Purple
  },
];
