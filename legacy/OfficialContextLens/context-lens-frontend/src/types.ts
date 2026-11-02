
export enum TipCategory {
  BODY_LANGUAGE = 'Body Language',
  TONE = 'Tone & Pacing',
  CONTENT = 'Content Quality',
  ENGAGEMENT = 'Engagement'
}

export interface TranscriptItem {
  id: string;
  startTime: number;
  endTime: number;
  text: string;
  isTip?: boolean;
  category?: TipCategory;
  sentiment?: 'positive' | 'neutral' | 'negative';
}

export interface AnalysisState {
  isProcessing: boolean;
  currentTime: number;
  duration: number;
  confidence: number;
  activeTips: TranscriptItem[];
}
