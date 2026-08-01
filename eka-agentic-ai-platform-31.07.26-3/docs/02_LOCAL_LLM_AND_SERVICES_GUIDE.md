# 🦙 Local LLM Engines & Services Integration Guide

The **Autonomous AI Studio & Multi-Agent Development Platform** includes native support for running locally hosted LLM inference engines alongside Google Gemini and OpenAI cloud APIs. This guide details supported engines, isolated setting storage, installation instructions, and troubleshooting.

---

## 💻 1. Supported Local LLM Frameworks

| Provider ID | Provider Name | Default Base URL | Protocol Format | Default Port | Typical Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ollama` | **Ollama** | `http://localhost:11434` | Native Ollama API & `/v1` | 11434 | Local lightweight models (`llama3`, `mistral`, `codegemma`, `phi3`) |
| `vllm` | **vLLM** | `http://localhost:8000/v1` | OpenAI API Compatible | 8000 | High-throughput GPU inference engine for enterprise Llama/Qwen models |
| `lmstudio` | **LM Studio** | `http://localhost:1234/v1` | OpenAI API Compatible | 1234 | Desktop GUI for GGUF model serving with one-click local server |
| `llamacpp` | **llama.cpp** | `http://localhost:8080/v1` | OpenAI API Compatible | 8080 | C/C++ `llama-server` lightweight binary execution |
| `custom` | **Custom OpenAI** | `http://localhost:5000/v1` | OpenAI API Compatible | Custom | TextGenWebUI, Jan.ai, LocalAI, KoboldCPP, FastChat, vLLM custom ports |

---

## 🗄️ 2. Isolated Provider Settings Architecture

Unlike monolithic applications that overwrite global endpoints when switching between local servers, this platform maintains **completely isolated configurations for every local provider**.

### Configuration Data Schema (`ModelServerConfig`)
In `src/types.ts`, local settings are stored independently in the `localConfigs` dictionary:

```typescript
export type LocalLLMProvider = 'ollama' | 'vllm' | 'lmstudio' | 'llamacpp' | 'custom';

export interface LocalLLMProviderConfig {
  provider: LocalLLMProvider;
  name: string;
  baseUrl: string;         // E.g. 'http://localhost:8000/v1'
  apiKey: string;          // Optional token
  selectedModel: string;   // Active model for this provider
  availableModels: string[];// Dynamically fetched models list
  isOnline: boolean;       // Health check status
  defaultPort: string;     // Default port indicator
  description: string;     // Overview description
}
```

### Benefits of Isolated Settings Storage
1. **Zero Re-configuration**: Switching from `vLLM` on Port 8000 to `LM Studio` on Port 1234 preserves your custom model selections, base URLs, and secret tokens for both engines.
2. **Independent Health Checking**: Clicking **Test Connection** tests the specific provider selected without disturbing other provider profiles.
3. **Smart Reset**: Each provider tab includes a **Reset Default** button to restore the standard base URL and port if customized.

---

## 🚀 3. Setup & Installation for Local LLM Engines

### 3.1 Ollama
- **Installation**: Download from [ollama.com](https://ollama.com).
- **Start Command**:
  ```bash
  ollama serve
  ```
- **Pull Models**:
  ```bash
  ollama pull llama3
  ollama pull codegemma
  ```
- **Health Check Endpoint**: `GET http://localhost:11434/api/tags` or `GET http://localhost:11434/v1/models`

### 3.2 vLLM (High-Performance GPU Engine)
- **Installation**:
  ```bash
  pip install vllm
  ```
- **Start Command**:
  ```bash
  python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --port 8000
  ```
- **Health Check Endpoint**: `GET http://localhost:8000/v1/models`

### 3.3 LM Studio
- **Installation**: Download LM Studio from [lmstudio.ai](https://lmstudio.ai).
- **Setup**:
  1. Download your desired GGUF model (e.g. Qwen 2.5 7B, Llama 3.2).
  2. Navigate to the **Local Server** tab (`<->` icon).
  3. Set Port to `1234` and enable **CORS**.
  4. Click **Start Server**.
- **Health Check Endpoint**: `GET http://localhost:1234/v1/models`

### 3.4 llama.cpp (`llama-server`)
- **Build / Download**: Clone [llama.cpp](https://github.com/ggerganov/llama.cpp) and build `llama-server`.
- **Start Command**:
  ```bash
  ./llama-server -m models/llama-3-8b-instruct.Q4_K_M.gguf --port 8080 --host 0.0.0.0
  ```
- **Health Check Endpoint**: `GET http://localhost:8080/v1/models`

### 3.5 Custom OpenAI-Compatible Endpoints
- Connect any OpenAI-compliant local server (TextGenWebUI, Jan.ai, LocalAI, KoboldCPP, FastChat).
- Specify custom Base URL (e.g., `http://localhost:5000/v1`) and optional API key.

---

## 📧 4. Local Python SMTP Mail Server (Port 1025)

The platform includes an integrated local Python SMTP mail server for testing email generation, notification templates, and agent alerts without external SMTP credentials.

### How it Works
1. **Control via UI**: In the **Settings** sidebar under *Local Python Mail Server*, click **Start Mail Server**.
2. **Backend Execution**: The Express server invokes Python's `asyncio` / `smtpd` module:
   ```bash
   python3 -m smtpd -c DebuggingServer -n 127.0.0.1:1025
   ```
3. **Log Interception**: Incoming emails sent to `127.0.0.1:1025` are captured and streamed to the application's Terminal log view.
4. **API Management**:
   - `GET /api/mailserver/status` — Returns `{ success: true, running: boolean }`
   - `POST /api/mailserver/start` — Spawns local SMTP server process
   - `POST /api/mailserver/stop` — Terminates local SMTP server process

---

## 🔧 5. Troubleshooting Connection Issues

| Error Message | Cause | Solution |
| :--- | :--- | :--- |
| `Could not connect to Ollama: fetch failed` | Ollama service is not running locally. | Run `ollama serve` in your terminal and ensure port `11434` is bound. |
| `Server returned status 404` | Incorrect API path suffix (e.g. missing `/v1`). | Click **Reset Default** to automatically set the compliant `/v1` endpoint path. |
| `CORS Error / Connection Refused` | Local LLM server blocking browser requests or localhost origins. | Enable CORS in your local server settings (e.g. check "Enable CORS" in LM Studio). |
| `API Key Unauthorized (401)` | Engine requires bearer token. | Enter your server's secret token in the **API Token / Key** field. |
