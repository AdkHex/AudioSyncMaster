import { useEffect, useState } from "react";

interface WaveformVisualizerProps {
  isActive: boolean;
  label?: string;
}

export const WaveformVisualizer = ({ isActive, label }: WaveformVisualizerProps) => {
  const bars = 40;
  const [heights, setHeights] = useState<number[]>(Array(bars).fill(15));

  useEffect(() => {
    if (!isActive) {
      setHeights(Array(bars).fill(15));
      return;
    }

    const interval = setInterval(() => {
      setHeights(prev => 
        prev.map(() => 15 + Math.random() * 85)
      );
    }, 100);

    return () => clearInterval(interval);
  }, [isActive]);

  return (
    <div className="relative">
      {label && (
        <span className="absolute -top-6 left-0 text-xs font-medium text-muted-foreground">
          {label}
        </span>
      )}
      <div className="flex items-end justify-center gap-0.5 h-12 px-4 py-2 bg-secondary/50 rounded-lg">
        {heights.map((height, i) => (
          <div
            key={i}
            className="w-1 bg-primary rounded-full transition-all duration-100"
            style={{ 
              height: `${height}%`,
              opacity: isActive ? 0.5 + (height / 200) : 0.3
            }}
          />
        ))}
      </div>
    </div>
  );
};
