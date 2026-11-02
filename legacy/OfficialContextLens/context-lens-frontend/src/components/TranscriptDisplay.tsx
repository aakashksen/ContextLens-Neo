import React, { useEffect, useRef, useState } from 'react';
import { TranscriptItem } from '../types';
import { ICONS } from '../constants';

interface TranscriptDisplayProps {
  transcript: TranscriptItem[];
  currentTime: number;
}

const TranscriptDisplay: React.FC<TranscriptDisplayProps> = ({ transcript, currentTime }) => {
  const [isOpen, setIsOpen] = useState(true);
  const [activeTip, setActiveTip] = useState<TranscriptItem | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const tipTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Filter out tips from the main transcript list
  const visibleItems = transcript
    .filter(item => !item.isTip && item.startTime <= currentTime)
    .slice(-5);

  const lastItem = visibleItems[visibleItems.length - 1];

  // Handle Tip Logic: Detection and Auto-dismiss
  useEffect(() => {
    const currentTip = transcript.find(item => 
      item.isTip && 
      currentTime >= item.startTime && 
      currentTime <= item.startTime + 0.15 // Capture the moment it triggers
    );

    if (currentTip && currentTip.id !== activeTip?.id) {
      if (tipTimeoutRef.current) clearTimeout(tipTimeoutRef.current);
      
      setActiveTip(currentTip);

      // Auto-dismiss after 5 seconds
      tipTimeoutRef.current = setTimeout(() => {
        setActiveTip(null);
      }, 8000);
    }
  }, [currentTime, transcript, activeTip]);

  useEffect(() => {
    if (scrollRef.current && isOpen) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: 'smooth'
      });
    }
  }, [visibleItems.length, currentTime, isOpen]);

  return (
    <div className="flex flex-col items-center w-full max-w-md space-y-4">
      {/* Main Transcript Window */}
      <div className={`bg-zinc-800/20 backdrop-blur-[60px] border border-white/10 rounded-[2rem] overflow-hidden shadow-2xl transition-all duration-700 ease-[cubic-bezier(0.23,1,0.32,1)] w-full ${isOpen ? 'h-[440px]' : 'h-[72px]'} ring-1 ring-white/5`}>
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/[0.05] transition-colors text-left"
        >
          <div className="flex items-center space-x-3">
            <div className={`w-1.5 h-1.5 rounded-full ${isOpen ? 'bg-violet-400 shadow-[0_0_8px_rgba(167,139,250,0.6)]' : 'bg-white/20'}`}></div>
            <h3 className="text-[11px] font-bold tracking-[0.15em] uppercase text-white/80">ContextLens</h3>
          </div>
          {!isOpen && lastItem && (
            <p className="text-[11px] text-violet-300/60 truncate max-w-[150px] font-medium italic">
              {lastItem.text.split(': ')[1]}
            </p>
          )}
        </button>
        
        {isOpen && (
          <div className="flex flex-col h-[calc(440px-72px)]">
            <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-2 space-y-6 no-scrollbar">
              {visibleItems.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center opacity-20">
                  <ICONS.Activity className="w-5 h-5 text-white animate-pulse" />
                </div>
              ) : (
                visibleItems.map((item) => {
                  const isActive = currentTime >= item.startTime && currentTime <= item.endTime;
                  const [speaker, ...textParts] = item.text.split(': ');
                  const messageText = textParts.join(': ');
                  
                  return (
                    <div key={item.id} className={`transition-all duration-700 ${isActive ? 'opacity-100 scale-100' : 'opacity-20 blur-[0.5px] scale-[0.98]'}`}>
                      <div className="flex space-x-3 items-start">
                        <span className={`mt-1.5 text-[8px] font-mono tabular-nums ${isActive ? 'text-violet-400' : 'text-white/30'}`}>
                          {Math.floor(item.startTime / 60)}:{(Math.floor(item.startTime) % 60).toString().padStart(2, '0')}
                        </span>
                        <p className={`text-[14px] leading-relaxed font-semibold transition-all duration-500 ${isActive ? 'text-white' : 'text-white/60'}`}>
                          <span className={isActive ? 'text-violet-400/90' : 'text-white/40'}>{speaker}:</span> {messageText}
                        </p>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
            
            <div className="px-6 py-4 bg-white/5 border-t border-white/5">
              <div className="flex items-center justify-between">
                <span className="text-[8px] font-bold text-white/20 uppercase tracking-widest">Meta Horizon Passthrough</span>
                <ICONS.Zap className="w-3 h-3 text-violet-400" />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Insight Card (Tips) */}
      <div className={`w-full transition-all duration-500 ease-[cubic-bezier(0.23,1,0.32,1)] transform ${activeTip ? 'translate-y-0 opacity-100 scale-100' : 'translate-y-4 opacity-0 scale-95 pointer-events-none'}`}>
        <div className="bg-violet-900/10 backdrop-blur-[40px] border border-violet-500/20 p-5 rounded-[1.5rem] shadow-xl ring-1 ring-violet-500/10">
          <div className="flex items-center space-x-3 mb-2">
            <div className="p-1.5 bg-violet-500/20 rounded-lg">
              <ICONS.Brain className="w-4 h-4 text-violet-400" />
            </div>
            <span className="text-[9px] font-black uppercase tracking-[0.2em] text-violet-400">Cognitive Insight</span>
          </div>
          <p className="text-sm font-semibold text-white/95 leading-snug italic tracking-tight">
            "{activeTip?.text}"
          </p>
        </div>
      </div>
    </div>
  );
};

export default TranscriptDisplay;
