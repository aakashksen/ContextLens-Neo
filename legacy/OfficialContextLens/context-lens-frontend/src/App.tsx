import React, { useState, useEffect } from 'react';
import TranscriptDisplay from './components/TranscriptDisplay';
import { TranscriptItem } from './types';
import testingData from './components/testing.json';

const App: React.FC = () => {
  const [transcript, setTranscript] = useState<TranscriptItem[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const realtime = import.meta.env.VITE_REALTIME === '1';

  useEffect(() => {
    if (realtime) {
      setTranscript([]);
      setDuration(Number.POSITIVE_INFINITY);
      setIsPlaying(true);
      return;
    }

    // Convert test data to transcript format
    const transcriptItems: TranscriptItem[] = testingData.items
      .filter((item: any) => item.isFinal)
      .map((item: any) => ({
        id: item.id,
        startTime: item.tStartMs / 1000,
        endTime: item.tEndMs / 1000,
        text: item.isTip ? item.text : `${item.speaker}: ${item.text}`,
        isTip: !!item.isTip,
        category: item.category
      }));

    setTranscript(transcriptItems);
    const maxTime = Math.max(...transcriptItems.map(item => item.endTime));
    setDuration(maxTime);
    setIsPlaying(true); // Auto-start playback
  }, [realtime]);

  useEffect(() => {
    if (!realtime) return;
    let counter = 0;
    const start = Date.now();
    const source = new EventSource('/realtime');
    source.onmessage = (event) => {
      const line = (event.data || '').trim();
      if (!line) return;
      const nowSeconds = (Date.now() - start) / 1000;
      counter += 1;
      if (line.startsWith('TIP:')) {
        const text = line.replace(/^TIP:\s*/, '');
        setTranscript(prev => ([
          ...prev,
          {
            id: `tip_${counter}`,
            startTime: nowSeconds,
            endTime: nowSeconds + 8,
            text,
            isTip: true
          }
        ]));
      } else if (line.includes(': ')) {
        setTranscript(prev => ([
          ...prev,
          {
            id: `rt_${counter}`,
            startTime: nowSeconds,
            endTime: nowSeconds + 6,
            text: line
          }
        ]));
      }
    };
    return () => {
      source.close();
    };
  }, [realtime]);

  // Simulate video playback with timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isPlaying && currentTime < duration) {
      interval = setInterval(() => {
        setCurrentTime(prev => {
          if (prev >= duration) {
            setIsPlaying(false);
            return duration;
          }
          return prev + 0.1;
        });
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying, currentTime, duration]);

  return (
    <div className="w-full h-screen bg-[#121212] flex items-center justify-center p-4 overflow-hidden relative">
      {/* Subtle Depth Elements - Simulates room lighting on Quest */}
      <div className="absolute top-1/4 left-1/4 w-[60%] h-[60%] bg-white/[0.02] rounded-full blur-[120px]"></div>
      
      <div className="w-full max-w-2xl h-full z-10 flex items-center justify-center">
        <TranscriptDisplay 
          transcript={transcript} 
          currentTime={currentTime} 
        />
      </div>
    </div>
  );
};

export default App;
