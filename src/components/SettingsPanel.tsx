import { Settings, Clock, FileCode, Key, ChevronDown } from "lucide-react";
import { useState } from "react";

interface SettingsPanelProps {
  segmentDuration: number;
  onSegmentChange: (value: number) => void;
  customPattern: string;
  onPatternChange: (value: string) => void;
  password: string;
  onPasswordChange: (value: string) => void;
}

export const SettingsPanel = ({
  segmentDuration,
  onSegmentChange,
  customPattern,
  onPatternChange,
  password,
  onPasswordChange
}: SettingsPanelProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-border bg-card overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between px-5 py-4 hover:bg-muted/30 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Settings className="w-5 h-5 text-muted-foreground" />
          <span className="font-medium">Advanced Settings</span>
        </div>
        <ChevronDown 
          className={`w-4 h-4 text-muted-foreground transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}
        />
      </button>
      
      {/* Content */}
      {isExpanded && (
        <div className="px-5 pb-5 space-y-4 border-t border-border pt-4 animate-slide-up">
          {/* Segment Duration */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Clock className="w-4 h-4 text-muted-foreground" />
              Segment Duration (seconds)
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min="60"
                max="600"
                step="30"
                value={segmentDuration}
                onChange={(e) => onSegmentChange(Number(e.target.value))}
                className="flex-1 h-2 bg-secondary rounded-full appearance-none cursor-pointer
                           [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 
                           [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full 
                           [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:shadow-md
                           [&::-webkit-slider-thumb]:cursor-pointer"
              />
              <span className="w-16 text-center text-sm font-mono bg-muted px-2 py-1 rounded-md">
                {segmentDuration}s
              </span>
            </div>
          </div>
          
          {/* Custom Pattern */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <FileCode className="w-4 h-4 text-muted-foreground" />
              Custom Match Pattern (regex)
            </label>
            <input
              type="text"
              value={customPattern}
              onChange={(e) => onPatternChange(e.target.value)}
              placeholder="e.g., S(\d+)E(\d+)"
              className="w-full px-4 py-2.5 bg-secondary rounded-lg text-sm placeholder:text-muted-foreground
                         focus:outline-none focus:ring-2 focus:ring-primary/50 font-mono"
            />
          </div>
          
          {/* Password */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium">
              <Key className="w-4 h-4 text-muted-foreground" />
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => onPasswordChange(e.target.value)}
              placeholder="Enter script password"
              className="w-full px-4 py-2.5 bg-secondary rounded-lg text-sm placeholder:text-muted-foreground
                         focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
        </div>
      )}
    </div>
  );
};
