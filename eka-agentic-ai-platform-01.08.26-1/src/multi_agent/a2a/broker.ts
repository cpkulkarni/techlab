/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { EventEmitter } from 'events';
import { A2AMessage, A2AMessageType, A2APayload } from '../types.js';

export type A2AListener = (message: A2AMessage) => void;

export class A2ABroker {
  private static instance: A2ABroker;
  private eventEmitter: EventEmitter;
  private messageHistory: A2AMessage[] = [];
  private sseClients: Set<(data: string) => void> = new Set();
  private maxHistorySize = 500;

  private constructor() {
    this.eventEmitter = new EventEmitter();
    this.eventEmitter.setMaxListeners(100);
  }

  public static getInstance(): A2ABroker {
    if (!A2ABroker.instance) {
      A2ABroker.instance = new A2ABroker();
    }
    return A2ABroker.instance;
  }

  /**
   * Constructs and publishes a valid A2A protocol message.
   */
  public publish(
    senderId: string,
    recipientId: string | 'broadcast',
    conversationId: string,
    messageType: A2AMessageType,
    payload: A2APayload
  ): A2AMessage {
    const message: A2AMessage = {
      id: `a2a_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      sender_id: senderId,
      recipient_id: recipientId,
      conversation_id: conversationId,
      message_type: messageType,
      payload,
      timestamp: new Date().toISOString(),
    };

    // Store in history
    this.messageHistory.push(message);
    if (this.messageHistory.length > this.maxHistorySize) {
      this.messageHistory.shift();
    }

    // Emit event locally for agents
    if (recipientId === 'broadcast') {
      this.eventEmitter.emit('broadcast', message);
    } else {
      this.eventEmitter.emit(`agent:${recipientId}`, message);
      this.eventEmitter.emit(`conversation:${conversationId}`, message);
    }
    this.eventEmitter.emit('all_messages', message);

    // Broadcast to connected SSE stream clients for live UI update
    this.broadcastToSSE({ type: 'a2a_message', message });

    return message;
  }

  /**
   * Subscribe an agent to receive messages targeted to them or broadcast
   */
  public subscribeAgent(agentId: string, listener: A2AListener): () => void {
    const agentHandler = (msg: A2AMessage) => listener(msg);
    const broadcastHandler = (msg: A2AMessage) => {
      if (msg.sender_id !== agentId) {
        listener(msg);
      }
    };

    this.eventEmitter.on(`agent:${agentId}`, agentHandler);
    this.eventEmitter.on('broadcast', broadcastHandler);

    return () => {
      this.eventEmitter.off(`agent:${agentId}`, agentHandler);
      this.eventEmitter.off('broadcast', broadcastHandler);
    };
  }

  /**
   * Subscribe to all messages for live UI inspector / log tracing
   */
  public subscribeAll(listener: A2AListener): () => void {
    this.eventEmitter.on('all_messages', listener);
    return () => {
      this.eventEmitter.off('all_messages', listener);
    };
  }

  /**
   * Get filtered history logs
   */
  public getHistory(options?: {
    conversationId?: string;
    agentId?: string;
    messageType?: A2AMessageType;
    limit?: number;
  }): A2AMessage[] {
    let filtered = [...this.messageHistory];

    if (options?.conversationId) {
      filtered = filtered.filter(m => m.conversation_id === options.conversationId);
    }
    if (options?.agentId) {
      filtered = filtered.filter(m => m.sender_id === options.agentId || m.recipient_id === options.agentId || m.recipient_id === 'broadcast');
    }
    if (options?.messageType) {
      filtered = filtered.filter(m => m.message_type === options.messageType);
    }

    const limit = options?.limit || 100;
    return filtered.slice(-limit);
  }

  public clearHistory(): void {
    this.messageHistory = [];
  }

  // SSE Stream Management
  public registerSSEClient(sendFn: (data: string) => void): () => void {
    this.sseClients.add(sendFn);
    return () => {
      this.sseClients.delete(sendFn);
    };
  }

  public broadcastToSSE(eventData: Record<string, any>): void {
    const formatted = `data: ${JSON.stringify(eventData)}\n\n`;
    for (const client of this.sseClients) {
      try {
        client(formatted);
      } catch {
        this.sseClients.delete(client);
      }
    }
  }
}
