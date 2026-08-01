/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Router } from 'express';

const router = Router();

export interface GraphEntity {
  id: string;
  name: string;
  type: 'concept' | 'component' | 'module' | 'service' | 'vocabulary_term';
  description: string;
  relationships: Array<{ targetId: string; relation: string }>;
}

export interface KnowledgeDocument {
  id: string;
  title: string;
  category: string;
  content: string;
}

// In-memory knowledge store with defaults
const knowledgeStore: {
  entities: GraphEntity[];
  documents: KnowledgeDocument[];
} = {
  entities: [
    {
      id: 'e1',
      name: 'Agentic Workflow',
      type: 'concept',
      description: 'Autonomous execution framework coordinating multi-step planning and tool usage.',
      relationships: [{ targetId: 'e2', relation: 'uses' }]
    },
    {
      id: 'e2',
      name: 'MCP Protocol',
      type: 'service',
      description: 'Model Context Protocol standardizing tool and resource access across agents.',
      relationships: []
    }
  ],
  documents: [
    {
      id: 'd1',
      title: 'Multi-Agent Architecture Guidelines',
      category: 'Architecture',
      content: 'Multi-agent orchestration separates responsibilities into Coordinator, Coder, Researcher, and Tester.'
    }
  ]
};

// GET /api/multi-agent/knowledge - Fetch full Knowledge Graph & Vault
router.get('/', (req, res) => {
  return res.json({
    success: true,
    entities: knowledgeStore.entities,
    documents: knowledgeStore.documents
  });
});

// POST /api/multi-agent/knowledge/entity - Add or update Knowledge Entity / Vocabulary Term
router.post('/entity', (req, res) => {
  const { name, type = 'concept', description, relationships = [] } = req.body;
  if (!name || !description) {
    return res.status(400).json({ success: false, error: 'name and description are required' });
  }

  const newEntity: GraphEntity = {
    id: `e_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
    name,
    type,
    description,
    relationships
  };

  knowledgeStore.entities.push(newEntity);
  return res.json({ success: true, entity: newEntity });
});

// POST /api/multi-agent/knowledge/document - Add Knowledge Vault Document
router.post('/document', (req, res) => {
  const { title, category = 'General', content } = req.body;
  if (!title || !content) {
    return res.status(400).json({ success: false, error: 'title and content are required' });
  }

  const newDoc: KnowledgeDocument = {
    id: `d_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`,
    title,
    category,
    content
  };

  knowledgeStore.documents.push(newDoc);
  return res.json({ success: true, document: newDoc });
});

export default router;
