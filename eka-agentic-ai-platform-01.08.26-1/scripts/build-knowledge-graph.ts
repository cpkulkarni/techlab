/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { generateKnowledgeGraph } from '../src/api/shared/knowledgeGraph.js';

async function main() {
  console.log('Generating Eka Source Code Knowledge Graph...');
  const kg = await generateKnowledgeGraph({ saveToFile: true });
  console.log('Knowledge Graph Generation Complete:');
  console.log(`- Files Processed: ${kg.metadata.totalFiles}`);
  console.log(`- Total Lines of Code: ${kg.metadata.totalLinesOfCode}`);
  console.log(`- Total Object Definitions: ${kg.metadata.totalDefinitions}`);
  console.log(`- Output File: eka_src_code_knowledge_graph.json`);
}

main().catch(err => {
  console.error('Failed to generate Knowledge Graph:', err);
  process.exit(1);
});
