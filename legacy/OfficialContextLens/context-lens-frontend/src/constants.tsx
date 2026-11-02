import React from 'react';
import { TranscriptItem, TipCategory } from './types.ts';

export const MOCK_TRANSCRIPT: TranscriptItem[] = [
  { id: '1', startTime: 1, endTime: 4, text: "Welcome to our deep dive session on architectural patterns." },
  { id: '2', startTime: 5, endTime: 8, text: "Today we are exploring how serverless functions scale under high pressure." },
  { id: 'tip-1', startTime: 7, endTime: 11, text: "You seem a bit tense. Try to relax your shoulders and use more hand gestures to emphasize key points.", isTip: true, category: TipCategory.BODY_LANGUAGE },
  { id: '3', startTime: 9, endTime: 13, text: "Notice how the event-driven architecture allows for decouple logic." },
  { id: '4', startTime: 14, endTime: 18, text: "If we look at the latencies here, we can see a significant drop compared to monoliths." },
  { id: 'tip-2', startTime: 16, endTime: 20, text: "Great pacing! Your words per minute is currently optimal for listener retention.", isTip: true, category: TipCategory.TONE },
  { id: '5', startTime: 19, endTime: 23, text: "This brings us to our primary case study: a retail giant handling Black Friday traffic." },
  { id: '6', startTime: 24, endTime: 28, text: "The complexity arises when managing shared state across multiple availability zones." },
  { id: 'tip-3', startTime: 27, endTime: 32, text: "You're using technical jargon frequently. Consider explaining 'Availability Zones' for a broader audience.", isTip: true, category: TipCategory.CONTENT },
  { id: '7', startTime: 29, endTime: 33, text: "To solve this, we implemented a distributed caching layer using Redis." },
  { id: '8', startTime: 34, endTime: 38, text: "As you can see on the screen, the performance metrics improved by nearly 40%." },
  { id: '9', startTime: 39, endTime: 43, text: "Next week we'll be discussing the implications of Cold Starts in AWS Lambda." }
];

export const ICONS = {
  Activity: (props: any) => <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>,
  MessageSquare: (props: any) => <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>,
  Zap: (props: any) => <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>,
  Upload: (props: any) => <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>,
  Brain: (props: any) => <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9.5 2A5 5 0 0 1 12 8a5 5 0 0 1 2.5-6"/><path d="M6 15a4 4 0 1 1 0-8c.9 0 1.7.3 2.4.8"/><path d="M18 15a4 4 0 1 0 0-8c-.9 0-1.7.3-2.4.8"/><path d="M12 13V7"/><path d="M12 13c-2.4 0-4 1.6-4 4v3a1 1 0 0 0 1 1h6a1 1 0 0 0 1-1v-3c0-2.4-1.6-4-4-4Z"/></svg>
};
