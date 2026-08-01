/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type AgentMode = 'chat' | 'code' | 'research' | 'multimodal' | 'documentation' | 'testing';

export interface FileAttachment {
  id: string;
  name: string;
  size: number;
  type: string;
  content?: string;
  dataUrl?: string;
}

export type MultimodalFeature =
  | 'text_to_image'
  | 'image_to_text'
  | 'stt'
  | 'tts'
  | 'text_to_video'
  | 'video_to_text'
  | 'text_to_3d'
  | 'image_to_3d'
  | 'translation';

export interface FileNode {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: FileNode[];
  content?: string;
  size?: number;
  /** When true, the item belongs to the locked source-code group: no delete, no rename. */
  locked?: boolean;
}

export type ServerType = 'gemini' | 'local_llm' | 'ollama' | 'openai';

export type LocalLLMProvider = 'ollama' | 'vllm' | 'lmstudio' | 'llamacpp' | 'custom';

export interface LocalLLMProviderConfig {
  provider: LocalLLMProvider;
  name: string;
  baseUrl: string;
  apiKey: string;
  selectedModel: string;
  availableModels: string[];
  isOnline: boolean;
  defaultPort: string;
  description: string;
}

export const DEFAULT_LOCAL_CONFIGS: Record<LocalLLMProvider, LocalLLMProviderConfig> = {
  ollama: {
    provider: 'ollama',
    name: 'Ollama',
    baseUrl: 'http://localhost:11434',
    apiKey: '',
    selectedModel: 'llama3',
    availableModels: ['llama3', 'mistral', 'codegemma', 'llama2', 'phi3'],
    isOnline: false,
    defaultPort: '11434',
    description: 'Local Ollama server instance with native API & OpenAI compatibility layer'
  },
  vllm: {
    provider: 'vllm',
    name: 'vLLM',
    baseUrl: 'http://localhost:8000/v1',
    apiKey: '',
    selectedModel: 'meta-llama/Llama-3-8B-Instruct',
    availableModels: ['meta-llama/Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.2', 'Qwen/Qwen2.5-7B-Instruct'],
    isOnline: false,
    defaultPort: '8000',
    description: 'High-throughput vLLM serving engine running OpenAI-compatible API server'
  },
  lmstudio: {
    provider: 'lmstudio',
    name: 'LM Studio',
    baseUrl: 'http://localhost:1234/v1',
    apiKey: '',
    selectedModel: 'local-model',
    availableModels: ['local-model', 'qwen2.5-7b-instruct', 'gemma-2-9b-it', 'llama-3.2-3b-instruct'],
    isOnline: false,
    defaultPort: '1234',
    description: 'LM Studio local inference server with OpenAI endpoint format'
  },
  llamacpp: {
    provider: 'llamacpp',
    name: 'llama.cpp',
    baseUrl: 'http://localhost:8080/v1',
    apiKey: '',
    selectedModel: 'default',
    availableModels: ['default', 'llama-2-7b-chat', 'codellama-7b-instruct', 'mistral-7b-v0.1'],
    isOnline: false,
    defaultPort: '8080',
    description: 'Lightweight C/C++ llama.cpp server (`llama-server`) with OpenAI API endpoint'
  },
  custom: {
    provider: 'custom',
    name: 'Custom Local',
    baseUrl: 'http://localhost:5000/v1',
    apiKey: '',
    selectedModel: 'default',
    availableModels: ['default'],
    isOnline: false,
    defaultPort: '5000',
    description: 'Custom local OpenAI-compatible endpoint (TextGenWebUI, Jan.ai, LocalAI, KoboldCPP)'
  },
};

export interface ModelServerConfig {
  type: ServerType;
  baseUrl: string;
  apiKey: string;
  selectedModel: string;
  isOnline: boolean;
  availableModels: string[];
  // Local LLM sub-provider settings stored separately per provider
  activeLocalProvider?: LocalLLMProvider;
  localConfigs?: Record<LocalLLMProvider, LocalLLMProviderConfig>;
  // Search Engine properties
  searchEngine?: string; // 'duckduckgo' | 'google_cse' | 'bing' | 'brave' | 'serper'
  searchEntryCount?: number; // Number of search results to fetch (1-20, default 5)
  googleCseApiKey?: string;
  googleCseCx?: string;
  bingApiKey?: string;
  braveApiKey?: string;
  serperApiKey?: string;
}

export interface GroundingChunk {
  web: {
    uri: string;
    title: string;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  mode?: AgentMode;
  attachments?: FileAttachment[];
  citations?: Array<{ title: string; url: string }>;
  stepLogs?: string[]; // Log outputs specifically for step-by-step display
}

export interface PlanStep {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'approved' | 'rejected' | 'running' | 'completed' | 'failed';
  type: 'create' | 'edit' | 'delete' | 'command' | 'test';
  target: string; // File path, command string, or test spec
  codeContent?: string; // Generated code to be written/applied
  command?: string; // Terminal command to run
  approvalRequired: boolean;
  logs?: string;
  /** LLM's explanation of why this step was chosen — shown in the UI thought-process panel */
  reasoning?: string;
}

export interface AgentWorkflow {
  taskId: string;
  prompt: string;
  mode: AgentMode;
  status: 'idle' | 'planning' | 'waiting_plan_approval' | 'waiting_approval' | 'executing' | 'testing' | 'correcting' | 'completed' | 'failed';
  plan: PlanStep[];
  currentStepIndex: number;
  logs: string[];
  /** Real-time thought process lines shown in the chat UI as the agent works */
  thinkingLines: string[];
}

// ==========================================
// PIPELINE WORKFLOW BUILDER TYPES
// ==========================================

export type WFNodeType =
  | 'input'
  | 'loop'
  | 'human_intervention'
  | 'decision'
  | 'db_read'
  | 'db_write'
  | 'api_call'
  | 'code_execution'
  | 'test_runner'
  | 'rag_vector_db'
  | 'rag_elastic'
  | 'rag_search_engine'
  | 'rag_local_files'
  | 'llm'
  | 'output'
  | 'email_send'
  | 'email_receive'
  | 'scheduler';

/** Decision / Conditional node config — evaluates Yes/No branching condition */
export interface WFDecisionConfig {
  questionPrompt: string;    // Condition / Question to evaluate e.g. "Is the input valid code?" or "Did tests pass?"
  evalType: 'contains_yes' | 'contains_text' | 'llm_boolean' | 'js_expression';
  expectedValue?: string;   // Text to match if evalType is contains_text
  yesLabel?: string;        // Branch label for YES
  noLabel?: string;         // Branch label for NO
}

/** Config shapes per node type */

export interface WFInputConfig {
  inputText: string;       // The user-provided text / data that flows into connected nodes
  label: string;           // Human-readable label shown in the pipeline
}

export interface WFCodeExecutionConfig {
  fileName: string;       // relative path inside workspace
  language: string;       // 'python' | 'typescript' | 'javascript' | 'bash'
  args: string;           // optional CLI args
}

export interface WFTestRunnerConfig {
  directory: string;      // workspace directory to scan for tests
  framework: string;      // 'pytest' | 'unittest' | 'jest' | 'vitest' | 'mocha'
  pattern: string;        // file glob, e.g. "test_*.py"
}

export interface WFRagVectorDbConfig {
  dbType: string;         // 'chroma' | 'pinecone' | 'weaviate' | 'qdrant' | 'milvus' | 'pgvector'
  query: string;          // Query text for embedding-based retrieval
  host: string;
  port: string;
  apiKey: string;
  collectionName: string;
  embeddingModel: string; // 'text-embedding-3-small' | 'gemini-embedding' etc.
  topK: number;
}

export interface WFRagElasticConfig {
  host: string;           // e.g. https://my-cluster.es.io
  port: string;
  apiKey: string;
  username: string;
  password: string;
  indexName: string;
  query: string;          // Full-text / semantic search query
  topK: number;
  tlsVerify: boolean;
}

export interface WFRagSearchEngineConfig {
  engine: string;         // 'duckduckgo' | 'google_cse' | 'bing' | 'brave' | 'serper'
  query: string;          // The search query (can be overridden by upstream input node)
  apiKey: string;
  googleCseApiKey: string;
  googleCseCx: string;
  bingApiKey: string;
  braveApiKey: string;
  serperApiKey: string;
  topK: number;
}

export interface WFRagLocalFilesConfig {
  directory: string;      // local dir path
  fileTypes: string;      // comma-separated, e.g. "txt,md,pdf"
  query: string;          // Query / filter text for retrieval
  chunkSize: number;
  chunkOverlap: number;
}

export interface WFLlmConfig {
  prompt: string;
  systemInstruction: string;
  temperature: number;    // 0-1
  maxTokens: number;
  searchEnabled: boolean; // use internet search to augment the prompt
}

export interface WFOutputConfig {
  format: 'text' | 'markdown' | 'code' | 'json' | 'csv' | 'html';
  fileName: string;       // output file relative path
  language: string;       // for code output: 'python' | 'typescript' etc.
  appendMode: boolean;
}

/** Send an email via SMTP */
export interface WFEmailSendConfig {
  sendMode?: 'local' | 'smtp' | 'direct';
  smtpHost: string;
  smtpPort: string;
  smtpUser: string;
  smtpPass: string;
  fromAddress: string;
  toAddresses: string;    // comma-separated
  ccAddresses: string;
  subject: string;        // may contain {{input}} / {{context}}
  body: string;           // may contain {{input}} / {{context}}
  isHtml: boolean;
  tlsMode: 'starttls' | 'ssl' | 'none';
}

/** Receive/poll emails via IMAP */
export interface WFEmailReceiveConfig {
  imapHost: string;
  imapPort: string;
  imapUser: string;
  imapPass: string;
  mailbox: string;
  filterFrom: string;
  filterSubject: string;
  maxMessages: number;
  markRead: boolean;
  tlsMode: 'ssl' | 'starttls' | 'none';
}

/** Loop connector — repeats the connected node N times, passing each iteration's output forward */
export interface WFLoopConfig {
  loopCount: number;        // how many times to iterate (1–50)
  loopVariable: string;     // variable name exposed as {{loopIndex}} inside the loop body
  breakCondition: string;   // optional JS-like expression: stop early if truthy (e.g. "output.includes('done')")
}

/** Human intervention — pauses pipeline and waits for a human to confirm or provide input */
export interface WFHumanInterventionConfig {
  prompt: string;           // message shown to the human
  mode: 'confirm' | 'input' | 'review'; // confirm = yes/no, input = free text, review = show output and confirm
  timeoutSeconds: number;   // 0 = wait indefinitely
  defaultAction: 'approve' | 'reject'; // what to do on timeout
}

/** Database read — query any relational DB */
export interface WFDbReadConfig {
  dbType: string;           // 'postgres' | 'mysql' | 'sqlite' | 'mssql' | 'oracle'
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  ssl: boolean;
  query: string;            // SQL SELECT query
  maxRows: number;
  outputFormat: 'json' | 'csv' | 'text';
}

/** Database write — insert/update/upsert any relational DB */
export interface WFDbWriteConfig {
  dbType: string;
  host: string;
  port: string;
  database: string;
  username: string;
  password: string;
  ssl: boolean;
  query: string;            // SQL INSERT/UPDATE/DELETE statement; use {{input}} for upstream value
  tableName: string;        // helper label
}

/** API Call — HTTP request to any external REST endpoint */
/** Temporal.io & Trigger.dev Scheduler integration node config */
export interface WFSchedulerConfig {
  jobName: string;
  schedulerServer: 'temporal' | 'trigger_dev' | 'embedded';
  scheduleType: 'cron' | 'interval' | 'one_shot';
  cronExpression: string;    // e.g. "*/5 * * * *"
  intervalSeconds: number;
  actionType: 'auto_detect' | 'code_execution' | 'file_action' | 'pipeline_workflow';
  targetPayload: string;     // code snippet, local file path, or empty to auto-inherit upstream node output
  temporalNamespace?: string;
  triggerDevEnvironment?: string;
}

export interface WFApiCallConfig {
  url: string;              // may contain {{input}} substitution
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  headers: string;          // JSON string: {"Authorization":"Bearer {{apiKey}}"}
  body: string;             // JSON/text body; may contain {{input}} or {{context}}
  apiKey: string;           // convenience field injected as Bearer if headers is empty
  timeoutMs: number;
  outputPath: string;       // JSONPath to extract from response, e.g. "data.results"
}

export type WFNodeConfig =
  | WFInputConfig
  | WFLoopConfig
  | WFHumanInterventionConfig
  | WFDecisionConfig
  | WFDbReadConfig
  | WFDbWriteConfig
  | WFApiCallConfig
  | WFCodeExecutionConfig
  | WFTestRunnerConfig
  | WFRagVectorDbConfig
  | WFRagElasticConfig
  | WFRagSearchEngineConfig
  | WFRagLocalFilesConfig
  | WFLlmConfig
  | WFOutputConfig
  | WFEmailSendConfig
  | WFEmailReceiveConfig
  | WFSchedulerConfig;

export interface WFNode {
  id: string;
  type: WFNodeType;
  label: string;
  config: Partial<WFNodeConfig>;
  position: { x: number; y: number };
  status?: 'idle' | 'running' | 'completed' | 'failed';
  outputPreview?: string;
}

export interface WFEdge {
  id: string;
  sourceId: string;
  targetId: string;
  /** If set, this edge is a loop-back: the target node will run loopCount times */
  loopCount?: number;
  /** Optional label shown on the edge mid-point */
  label?: string;
}

export interface WFWorkflow {
  id: string;
  name: string;
  description: string;
  nodes: WFNode[];
  edges: WFEdge[];
  createdAt: string;
  updatedAt: string;
  lastRunAt?: string;
  status?: 'idle' | 'running' | 'completed' | 'failed';
  runLogs?: string[];
  /** Format version for import/export compatibility */
  version?: string;
  exportedAt?: string;
}
