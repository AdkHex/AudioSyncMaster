import { ReactNode } from "react";

interface WindowFrameProps {
  children: ReactNode;
  title?: string;
}

export const WindowFrame = ({ children, title = "AudioSync" }: WindowFrameProps) => {
  return (
    <div className="w-full max-w-5xl mx-auto animate-slide-up">
      <div className="rounded-xl overflow-hidden shadow-apple-lg bg-card border border-border">
        {/* Windows-style Title Bar */}
        <div className="h-10 bg-secondary/50 flex items-center justify-between px-4 border-b border-border">
          {/* App Icon and Title */}
          <div className="flex items-center gap-2">
            <div className="w-5 h-5 rounded bg-primary flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" className="text-primary-foreground">
                <path d="M9 18V5l12-2v13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                <circle cx="6" cy="18" r="3" stroke="currentColor" strokeWidth="2"/>
                <circle cx="18" cy="16" r="3" stroke="currentColor" strokeWidth="2"/>
              </svg>
            </div>
            <span className="text-sm font-medium text-foreground">{title}</span>
          </div>
          
          {/* Windows Controls (visual only) */}
          <div className="flex items-center gap-1">
            <div className="w-8 h-6 flex items-center justify-center hover:bg-muted rounded transition-colors cursor-pointer">
              <svg width="10" height="1" viewBox="0 0 10 1" className="text-muted-foreground">
                <rect width="10" height="1" fill="currentColor"/>
              </svg>
            </div>
            <div className="w-8 h-6 flex items-center justify-center hover:bg-muted rounded transition-colors cursor-pointer">
              <svg width="10" height="10" viewBox="0 0 10 10" className="text-muted-foreground">
                <rect x="0.5" y="0.5" width="9" height="9" stroke="currentColor" fill="none"/>
              </svg>
            </div>
            <div className="w-8 h-6 flex items-center justify-center hover:bg-destructive/80 rounded transition-colors cursor-pointer group">
              <svg width="10" height="10" viewBox="0 0 10 10" className="text-muted-foreground group-hover:text-white">
                <path d="M1 1L9 9M9 1L1 9" stroke="currentColor" strokeWidth="1.5"/>
              </svg>
            </div>
          </div>
        </div>
        
        {/* Content */}
        <div className="p-6">
          {children}
        </div>
      </div>
    </div>
  );
};
