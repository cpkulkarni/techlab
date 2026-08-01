# Knowledge Graph Application — Build Plan

## Top-Level Overview

Build a fully **client-side, portable single `index.html`** knowledge graph application. The user enters a topic, the app queries a chosen LLM (Ollama local / OpenAI / Gemini) using a natural-language prompt that asks the LLM to return structured JSON, then renders an interactive force-directed graph using D3.js. Clicking a node expands it with downstream concepts. A small ℹ️ icon on each node shows a 5-line summary popup with a "More Details →" DuckDuckGo search link. The graph is persistable as a JSON file and reloadable. When Ollama (local) is selected, the app auto-enriches the LLM prompt with DuckDuckGo Instant Answer results.

**Constraints:**
- No server, no backend, no build step required — runs from a single HTML file in any modern browser
- All external dependencies loaded from CDN (D3.js)
- API keys stored only in `localStorage`, never transmitted anywhere except the chosen LLM provider

---

## Sub-Tasks

---

### Sub-Task 1 — Project Shell & Graph Canvas

**Intent:** Create the single `index.html` with responsive layout, D3.js force-directed graph canvas with dummy data, zoom/pan/drag, and basic node/edge rendering. This establishes the visual foundation everything else builds on.

**Expected Outcomes:**
- `index.html` opens in browser and displays a sample graph with ~6 dummy nodes and edges
- Graph is zoomable, pannable, and nodes are draggable
- Layout is responsive and usable on mobile, tablet, and desktop
- Nodes display a label; edges display a relationship label

**Todo List:**
1. Create `knowledge-graph-app/index.html` with inlined CSS and JS sections
2. Add a top toolbar area (for topic input, settings button, save/load controls)
3. Add an SVG canvas that fills the remaining viewport height
4. Load D3.js v7 from CDN
5. Implement a force-directed simulation with dummy nodes/edges
6. Render nodes as circles with label text beneath
7. Render edges as lines with a small midpoint relationship label
8. Add zoom/pan behavior on the SVG
9. Add drag behavior on nodes
10. Ensure SVG viewBox scales correctly on window resize

**Relevant Context:**
- All code lives in `knowledge-graph-app/index.html`
- D3.js CDN: `https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js`
- Force simulation: `d3.forceSimulation`, `d3.forceLink`, `d3.forceManyBody`, `d3.forceCenter`

**Status:** [ ] pending

---

### Sub-Task 2 — LLM Provider Settings Panel

**Intent:** Add a settings panel (modal or sidebar) where the user can choose their LLM provider (Ollama / OpenAI / Gemini), configure the model name, and enter API keys. Settings are persisted to `localStorage`.

**Expected Outcomes:**
- A ⚙️ settings button in the toolbar opens the settings panel
- User can select provider: Ollama (local) | OpenAI (ChatGPT) | Gemini
- For Ollama: input for model name (default: `llama3`) and base URL (default: `http://localhost:11434`)
- For OpenAI: input for API key and model (default: `gpt-4o-mini`)
- For Gemini: input for API key and model (default: `gemini-1.5-flash`)
- All values save to `localStorage` on close and restore on page load
- Panel closes on backdrop click or close button

**Todo List:**
1. Add a ⚙️ settings button to the toolbar
2. Create a modal overlay with a settings form
3. Add provider radio selector (Ollama / OpenAI / Gemini)
4. Conditionally show relevant fields based on selected provider
5. Implement `saveSettings()` to write to `localStorage`
6. Implement `loadSettings()` to restore from `localStorage` on page load
7. Wire open/close behavior for the modal

**Relevant Context:**
- `localStorage` keys: `kg_provider`, `kg_model`, `kg_api_key`, `kg_ollama_url`
- No API keys should appear in any URL or console log

**Status:** [ ] pending

---

### Sub-Task 3 — Topic Input → LLM Query → Graph Render

**Intent:** Wire the topic input field to query the selected LLM provider using a natural-language prompt that instructs the LLM to return a specific JSON schema. Parse the response and render the real graph.

**Expected Outcomes:**
- User types a topic and presses Enter or clicks "Generate"
- App shows a loading indicator
- LLM is called with a natural-language prompt asking for JSON output
- Response is parsed; nodes and edges are extracted
- D3 graph is re-rendered with the real data (replaces dummy data)
- If parsing fails, a user-friendly error is shown

**Todo List:**
1. Add topic text input and "Generate" button to the toolbar
2. Implement `buildPrompt(topic)` — natural language prompt instructing LLM to return JSON with `nodes[]` and `edges[]`
3. Implement `callOllama(prompt)` — fetch to `http://localhost:11434/api/generate`
4. Implement `callOpenAI(prompt)` — fetch to OpenAI chat completions endpoint
5. Implement `callGemini(prompt)` — fetch to Gemini generateContent endpoint
6. Implement `callLLM(prompt)` — dispatches to the correct provider based on settings
7. Implement `parseGraphJSON(rawText)` — extract JSON block from LLM response text (handle markdown code fences)
8. Implement `renderGraph(nodes, edges)` — update D3 simulation with new data
9. Add loading spinner overlay during fetch
10. Add error toast for failed calls or unparseable responses

**Relevant Context:**
- Prompt instructs: extract 5–8 key concepts, respond ONLY with valid JSON, no text outside JSON
- JSON schema: `{ "nodes": [{ "id", "label", "summary" }], "edges": [{ "source", "target", "relation" }] }`
- Ollama API: POST `/api/generate` with `{ model, prompt, stream: false }`
- OpenAI API: POST `/v1/chat/completions` with messages array, Authorization Bearer header
- Gemini API: POST `/v1beta/models/{model}:generateContent` with API key as query param

**Status:** [ ] pending

---

### Sub-Task 4 — Node Click → Expand Downstream Nodes

**Intent:** Clicking a node's body triggers a new LLM query for that node's topic and appends the returned nodes/edges to the existing graph (no duplicates, no reset).

**Expected Outcomes:**
- Clicking a node body (not the ℹ️ icon) triggers expansion
- A loading spinner appears on the clicked node while fetching
- New nodes and edges are appended to the existing graph
- Duplicate node IDs are skipped (existing nodes are not replaced)
- The graph re-simulates smoothly with new nodes added
- Already-explored nodes get a visual indicator (e.g. darker border)

**Todo List:**
1. Add click handler on node circles that calls `expandNode(node)`
2. `expandNode(node)` calls `callLLM(buildPrompt(node.label))` and shows spinner on node
3. On response, call `parseGraphJSON()` then `mergeGraph(newNodes, newEdges)` 
4. Implement `mergeGraph(newNodes, newEdges)` — adds only nodes/edges with IDs not already in graph
5. Mark the expanded node with `explored: true` and update its visual style
6. Re-run D3 simulation with updated data
7. Distinguish click-to-expand (node body) from click-to-info (ℹ️ icon) clearly

**Relevant Context:**
- Node data model: `{ id, label, summary, explored, depth }`
- Edge data model: `{ id, source, target, relation }`
- D3 simulation `.nodes()` and `.force("link").links()` must be updated before restarting

**Status:** [ ] pending

---

### Sub-Task 5 — ℹ️ Icon → Knowledge Summary Popup

**Intent:** Each graph node has a small ℹ️ icon. Clicking it shows a popup card with the node's 5-line summary and a "More Details →" link to a DuckDuckGo search for that topic.

**Expected Outcomes:**
- Each node renders a small ℹ️ icon (top-right of circle)
- Clicking ℹ️ opens a floating popup card anchored near the node
- Popup shows: node label as title, summary text (~5 sentences), "More Details →" link
- "More Details →" opens `https://duckduckgo.com/?q=<node+label>` in a new tab
- Popup is smart-positioned (repositioned if near screen edge)
- Clicking anywhere outside the popup dismisses it
- Only one popup can be open at a time

**Todo List:**
1. Render a small `ℹ` text/icon element per node in the D3 node group, offset top-right
2. Add click handler on ℹ️ icon that calls `showSummaryPopup(node, screenX, screenY)`
3. Create a `<div id="summary-popup">` in the HTML, initially hidden
4. `showSummaryPopup()` populates the popup with title, summary, and search link then positions it
5. Implement edge-detection: if popup would overflow right/bottom of viewport, flip position
6. Add `document.addEventListener('click')` to dismiss popup on outside click
7. Stop click propagation on the popup itself and the ℹ️ icon click
8. Style the popup as a clean card (shadow, rounded corners, max-width ~280px)

**Relevant Context:**
- DuckDuckGo search URL: `https://duckduckgo.com/?q=${encodeURIComponent(node.label)}`
- Summary comes from `node.summary` field populated during LLM parse
- Screen coordinates available from D3 node's `x`/`y` + SVG transform

**Status:** [ ] pending

---

### Sub-Task 6 — Internet Search Enrichment for Ollama

**Intent:** When Ollama (local) is the selected provider, the app first calls the DuckDuckGo Instant Answer API to fetch context about the topic, then injects those snippets into the LLM prompt to compensate for the local model's limited knowledge.

**Expected Outcomes:**
- When provider is Ollama, `buildPrompt(topic)` automatically fetches DuckDuckGo results first
- Top 3 DuckDuckGo abstract/related topic snippets are prepended to the prompt as context
- If DuckDuckGo returns no results, the app falls back gracefully to a prompt without enrichment
- Cloud providers (OpenAI, Gemini) skip this step entirely

**Todo List:**
1. Implement `fetchDuckDuckGoContext(topic)` — call `https://api.duckduckgo.com/?q=<topic>&format=json&no_html=1&skip_disambig=1` via CORS proxy or direct
2. Extract `AbstractText` and up to 3 `RelatedTopics[].Text` snippets
3. Implement `buildEnrichedPrompt(topic, contextSnippets)` — prepends snippets as "Background context" before the JSON instruction
4. In `callLLM()`, when provider is Ollama, call `fetchDuckDuckGoContext()` first then `buildEnrichedPrompt()`
5. Add graceful fallback: if fetch fails or returns empty, use standard `buildPrompt(topic)` without enrichment
6. Note: DuckDuckGo Instant Answer API supports CORS and requires no API key

**Relevant Context:**
- DuckDuckGo API: `https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1`
- This API returns `AbstractText` (short summary) and `RelatedTopics` array
- CORS: DuckDuckGo API does support cross-origin requests from browsers

**Status:** [ ] pending

---

### Sub-Task 7 — Save & Load Knowledge Graph JSON

**Intent:** Allow the user to save the current graph as a `.json` file and reload a previously saved graph from a file. Also auto-save to `localStorage` as a backup.

**Expected Outcomes:**
- "Save" button downloads the graph as `knowledge-graph.json`
- "Load" button opens a file picker; selecting a valid JSON file restores and re-renders the graph
- Graph auto-saves to `localStorage` on every change
- On page load, if `localStorage` has a saved graph, a "Resume last session?" prompt appears
- Saved JSON includes all nodes (with summaries, properties, explored state) and all edges

**Todo List:**
1. Add "💾 Save" and "📂 Load" buttons to the toolbar
2. Implement `saveGraph()` — serialize current graph state to JSON and trigger download via `<a download>`
3. Define final JSON schema: `{ version, created, nodes: [{id, label, summary, explored, depth, properties}], edges: [{id, source, target, relation}] }`
4. Implement `loadGraph(jsonString)` — parse JSON, validate structure, call `renderGraph()`
5. Wire file input `<input type="file" accept=".json">` to `loadGraph()`
6. Implement `autoSave()` — write graph JSON to `localStorage` key `kg_graph_data` after every expand/generate
7. On page load, check `localStorage` for existing graph and show restore prompt if found

**Relevant Context:**
- File download: create a `Blob`, make an object URL, click a hidden `<a>` element
- File upload: `FileReader.readAsText()` on the selected file
- `localStorage` key: `kg_graph_data`

**Status:** [ ] pending

---

### Sub-Task 8 — Polish & UX Refinement

**Intent:** Final visual and UX improvements: node color-coding by depth, smooth animations, mobile usability, and overall fit-and-finish.

**Expected Outcomes:**
- Nodes are color-coded by depth level (root = one color, depth 1 = another, etc.)
- Explored nodes have a distinct visual style (e.g. thicker border or subtle fill difference)
- New nodes animate in smoothly when added
- Graph fits to screen after initial generation ("fit to view" behavior)
- Toolbar is clean and usable on mobile (hamburger menu if needed)
- App title and brief usage hint visible on first load (dismissable)

**Todo List:**
1. Define a color scale by depth using `d3.scaleOrdinal` with a color palette
2. Apply depth-based fill color to node circles; update on graph changes
3. Style explored nodes with a distinct border/stroke
4. Add entrance transition for new nodes (scale from 0 to full size)
5. After initial graph render, call a `fitToView()` function that adjusts zoom/translate to fit all nodes
6. Add a welcome/hint banner on first load explaining: "Enter a topic → click Generate → click nodes to expand"
7. Review mobile layout: ensure toolbar controls stack or scroll cleanly on small screens
8. Final CSS cleanup and consistent color theme (dark or light, choose one)

**Relevant Context:**
- D3 depth: track `node.depth` when adding nodes (root = 0, children of root = 1, etc.)
- D3 zoom transform: `svg.transition().call(zoom.transform, d3.zoomIdentity.translate(...).scale(...))`

**Status:** [ ] pending
