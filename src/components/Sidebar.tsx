import { Play, Square } from "lucide-react";
import { memo } from "react";

import { FilePanel } from "@/components/FilePanel";
import { cx } from "@/lib/cx";
import type { FileItem, MediaProbe, SyncMode, TrackListing } from "@/lib/types";

interface SidebarProps {
  mode: SyncMode;
  videoFiles: FileItem[];
  audioFiles: FileItem[];
  videoFolder: string | null;
  audioFolder: string | null;
  recentFolders: { video: string | null; audio: string | null };
  probes: Record<string, MediaProbe>;
  dragTarget: "video" | "audio" | null;
  busy: boolean;

  /** Audio streams per file path, and the chosen stream for each. Per file
   *  rather than per side: a selection can mix sources whose track layouts
   *  have nothing in common. */
  listings: Record<string, TrackListing>;
  trackChoices: Record<string, number>;
  onTrackChange: (path: string, index: number) => void;

  onBrowse: (kind: "video" | "audio") => void;
  onRemove: (kind: "video" | "audio", id: string) => void;
  onClear: (kind: "video" | "audio") => void;
  onDragEnter: (kind: "video" | "audio") => void;

  /** What the run button does and whether it can. */
  runLabel: string;
  canRun: boolean;
  runBlockedReason?: string;
  onRun: () => void;
  onStop: () => void;
}

/** Everything that defines a run, in one column that never scrolls away.
 *
 *  Inputs used to sit above the results and push them off screen once a run
 *  finished. Keeping them beside the results means the files and their
 *  measurements are visible at the same time. */
export const Sidebar = memo(function Sidebar({
  mode,
  videoFiles,
  audioFiles,
  videoFolder,
  audioFolder,
  recentFolders,
  probes,
  dragTarget,
  busy,
  listings,
  trackChoices,
  onTrackChange,
  onBrowse,
  onRemove,
  onClear,
  onDragEnter,
  runLabel,
  canRun,
  runBlockedReason,
  onRun,
  onStop,
}: SidebarProps) {
  return (
    <aside className="flex w-[288px] shrink-0 flex-col border-r border-border">
      <div
        className="min-h-0 flex-1 overflow-y-auto px-4 py-[18px]"
        onDragOver={(event) => event.preventDefault()}
      >
        <div onDragEnter={() => onDragEnter("video")}>
          <FilePanel
            kind="video"
            title="Video"
            hint="Drop files, or click to browse"
            files={videoFiles}
            folder={videoFolder}
            recentFolder={recentFolders.video}
            probes={probes}
            listings={listings}
            trackChoices={trackChoices}
            onTrackChange={onTrackChange}
            dragActive={dragTarget === "video"}
            disabled={busy}
            onBrowse={() => onBrowse("video")}
            onRemove={(id) => onRemove("video", id)}
            onClear={() => onClear("video")}
          />
        </div>

        <hr className="my-5 border-border" />

        <div onDragEnter={() => onDragEnter("audio")}>
          <FilePanel
            kind="audio"
            title={mode === "compare" ? "Dubs to test" : "Dub"}
            hint={
              mode === "movie"
                ? "Drop the audio track"
                : mode === "compare"
                  ? "Drop the tracks to test"
                  : "Drop the folder of dubs"
            }
            files={audioFiles}
            folder={audioFolder}
            recentFolder={recentFolders.audio}
            probes={probes}
            listings={listings}
            trackChoices={trackChoices}
            onTrackChange={onTrackChange}
            dragActive={dragTarget === "audio"}
            disabled={busy}
            onBrowse={() => onBrowse("audio")}
            onRemove={(id) => onRemove("audio", id)}
            onClear={() => onClear("audio")}
          />
        </div>

      </div>

      <div className="shrink-0 px-4 pb-4 pt-2">
        <button
          type="button"
          onClick={busy ? onStop : onRun}
          disabled={!busy && !canRun}
          title={!busy && !canRun ? runBlockedReason : undefined}
          className={cx(
            "flex w-full items-center justify-center gap-2 rounded-[9px] px-4 py-[11px]",
            "text-[13px] font-semibold transition-colors",
            busy
              ? "bg-elevated text-foreground hover:bg-secondary"
              : "bg-primary text-primary-foreground hover:bg-primary/90",
            "disabled:bg-elevated disabled:text-muted-foreground",
          )}
        >
          {busy ? (
            <>
              <Square className="h-3 w-3 fill-current" aria-hidden />
              Stop
            </>
          ) : (
            <>
              <Play className="h-3.5 w-3.5 fill-current" aria-hidden />
              {runLabel}
            </>
          )}
        </button>

        {!busy && !canRun && runBlockedReason && (
          <p className="mt-2 text-center text-[11.5px] text-muted-foreground">
            {runBlockedReason}
          </p>
        )}
      </div>
    </aside>
  );
});
