import { Folder, Music, Film, Upload } from "lucide-react";

interface FileDropZoneProps {
  type: "video" | "audio" | "folder";
  label: string;
  description: string;
  value?: string;
  onSelect: () => void;
}

export const FileDropZone = ({ type, label, description, value, onSelect }: FileDropZoneProps) => {
  const Icon = type === "audio" ? Music : type === "video" ? Film : Folder;
  
  const iconColors = {
    video: "text-primary",
    audio: "text-success",
    folder: "text-warning"
  };

  return (
    <button
      onClick={onSelect}
      className="w-full p-6 rounded-xl border-2 border-dashed border-border hover:border-primary/50 
                 bg-secondary/30 hover:bg-accent/30 transition-all duration-300 group text-left
                 hover:scale-[1.01] active:scale-[0.99]"
    >
      <div className="flex items-start gap-4">
        <div className={`p-3 rounded-xl bg-card shadow-apple-sm ${iconColors[type]}`}>
          <Icon className="w-6 h-6" />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-foreground">{label}</span>
            <Upload className="w-4 h-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
          </div>
          
          {value ? (
            <p className="text-sm text-primary font-medium truncate">{value}</p>
          ) : (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
      </div>
    </button>
  );
};
