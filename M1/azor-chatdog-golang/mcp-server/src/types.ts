export interface SessionLogEntry {
  role: "user" | "model";
  timestamp: string;
  text: string;
}

export interface SessionLogData {
  session_id: string;
  model: string;
  system_role: string;
  history: SessionLogEntry[];
}

export interface SessionMetadata {
  id: string;
  messagesCount: number;
  lastActivity: string;
  model: string;
  error?: string;
}

export interface WALEntry {
  timestamp: string;
  session_id: string;
  model: string;
  prompt: string;
  response: string;
  tokens_used: number;
}

export interface SessionSummary {
  id: string;
  messagesCount: number;
  lastActivity: string;
  model: string;
  totalTokens: number;
  firstMessage?: string;
  lastMessage?: string;
}
