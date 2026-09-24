import { X } from "lucide-react";
import { memo, useMemo, useState } from "react";

import { cx } from "@/lib/cx";
import {
  formatDuration,
  formatSize,
  streamSummary,
  type FileItem,
  type MediaProbe,
  type TrackListing,
} from "@/lib/types";

interface FilePanelProps {
  kind: "video" | "audio";
  /** What a file in this slot must contain. Defaults to the kind; the video
   *  slot of a dub sync only needs audio, since a bare original-language
   *  track serves as well as the video itself. */
  needs?: "video" | "audio";
  title: string;
  hint: string;
  files: FileItem[];
  folder: string | null;
  recentFolder: string | null;
  probes: Record<string, MediaProbe>;
  /** Audio streams per file path, for the per-file track picker. */
  listings: Record<string, TrackListing>;
  /** Chosen stream index per file path. Absent means the first stream. */
  trackChoices: Record<string, number>;
  onTrackChange: (path: string, index: number) => void;
  dragActive: boolean;
  disabled: boolean;
  onBrowse: () => void;
  /** Adds a file by its full path, typed or pasted: for a keyboard, for a
   *  path copied from elsewhere, and where no file dialog can be used. */
  onAddPath?: (path: string) => void;
  /** Media files in the folder last used, offered one click each while the
   *  panel is empty: the next episode's files are usually beside the last. */
  suggestions?: FileItem[];
  onRemove: (id: string) => void;
  onClear: () => void;
}

/** One input group in the sidebar.
 *
 *  A plain list rather than a bordered card: the heading and the spacing
 *  already group these rows, so a box around them is a third device doing the
 *  same job. */
export const FilePanel = memo(function FilePanel({
  kind,
  needs = kind,
  title,
  hint,
  files,
  folder,
  recentFolder,
  probes,
  listings,
  trackChoices,
  onTrackChange,
  dragActive,
  disabled,
  onBrowse,
  onAddPath,
  suggestions,
  onRemove,
  onClear,
}: FilePanelProps) {
  const [typedPath, setTypedPath] = useState("");
  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + (file.size ?? 0), 0),
    [files],
  );
  // The folder the offered files are in: a side whose own last folder is
  // gone is offered the other side's, and must not name the one it lost.
  const shownFolder =
    onAddPath && suggestions && suggestions.length > 0
      ? suggestions[0].path.replace(/[\\/][^\\/]*$/, "")
      : recentFolder;

  return (
    <section aria-label={title}>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="text-[11px] font-semibold text-muted-foreground">{title}</h2>
        {files.length > 0 ? (
          <div className="flex items-baseline gap-2.5">
            <button
              type="button"
              onClick={onClear}
              disabled={disabled}
              className="text-[11.5px] text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={onBrowse}
              disabled={disabled}
              className="text-[11.5px] font-medium text-primary transition-opacity hover:opacity-80 disabled:opacity-40"
            >
              Change
            </button>
          </div>
        ) : null}
      </div>

      {files.length === 0 ? (
        <button
          type="button"
          onClick={onBrowse}
          disabled={disabled}
          className={cx(
            "w-full rounded-[9px] border border-dashed px-3.5 py-6 text-center text-xs transition-colors disabled:opacity-40",
            dragActive
              ? "border-primary/60 bg-primary/5 text-primary"
              : "border-border-strong text-muted-foreground hover:border-muted-foreground/40 hover:text-foreground",
          )}
        >
          {dragActive ? `Drop to add ${needs === kind ? kind : "the"} file${needs === kind ? "s" : ""}` : hint}
        </button>
      ) : (
        <>
          <ul className="flex flex-col gap-1">
            {files.map((file) => {
              const probe = probes[file.path];
              const missing =
                probe && (needs === "video" ? !probe.hasVideo : !probe.hasAudio);
              const listing = listings[file.path];
              const tracks = listing?.tracks ?? [];
              const chosen = trackChoices[file.path] ?? 0;
              const details = streamSummary(probe, listing, kind);

              return (
                <li key={file.id} className="group">
                  <div className="flex items-baseline gap-2">
                    <span
                      className="min-w-0 flex-1 truncate text-[12.5px]"
                      title={file.name}
                    >
                      {file.name}
                    </span>

                    {missing ? (
                      <span className="shrink-0 text-[11px] text-destructive">
                        no {needs === "video" ? "video" : "audio"}
                      </span>
                    ) : probe?.duration ? (
                      <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                        {formatDuration(probe.duration)}
                      </span>
                    ) : null}

                    <span className="tabular shrink-0 font-mono text-[11px] text-muted-foreground">
                      {formatSize(file.size)}
                    </span>

                    <button
                      type="button"
                      onClick={() => onRemove(file.id)}
                      disabled={disabled}
                      className="-mr-1 shrink-0 rounded p-0.5 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-0"
                    >
                      <X className="h-3 w-3" aria-hidden />
                      <span className="sr-only">Remove {file.name}</span>
                    </button>
                  </div>

                  {/* What the file actually contains. Codec, rate and channels
                      were probed already and thrown away; seeing them is how
                      you notice you loaded the commentary or a stereo downmix. */}
                  {details && (
                    <p className="font-mono text-[10.5px] leading-snug text-muted-foreground/80">
                      {details}
                    </p>
                  )}

                  {/* Per file, not per side: a selection can mix a REMUX
                      carrying five languages with a WEB-DL carrying one. */}
                  {tracks.length > 1 && (
                    <select
                      value={chosen}
                      disabled={disabled}
                      onChange={(event) =>
                        onTrackChange(file.path, Number(event.target.value))
                      }
                      aria-label={`Audio stream for ${file.name}`}
                      className="mt-1 w-full appearance-none rounded-md bg-elevated px-2 py-1 pr-6 text-[11.5px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
                      style={{
                        backgroundImage:
                          "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23969696' stroke-width='2.5'><path d='m6 9 6 6 6-6'/></svg>\")",
                        backgroundRepeat: "no-repeat",
                        backgroundPosition: "right 7px center",
                      }}
                    >
                      {tracks.map((track) => (
                        <option key={track.index} value={track.index}>
                          {track.label}
                        </option>
                      ))}
                    </select>
                  )}
                </li>
              );
            })}
          </ul>

          <p className="mt-1.5 flex items-baseline justify-between gap-2 font-mono text-[10.5px] text-muted-foreground">
            <span className="min-w-0 truncate" title={folder ?? undefined}>
              {folder ?? ""}
            </span>
            {files.length > 1 && (
              <span className="tabular shrink-0">{formatSize(totalSize)}</span>
            )}
          </p>
        </>
      )}

      {files.length === 0 && shownFolder && (
        <p
          className="mt-2 truncate font-mono text-[10.5px] text-muted-foreground"
          title={shownFolder}
        >
          {shownFolder}
        </p>
      )}
      {onAddPath && files.length === 0 && suggestions && suggestions.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5" aria-label={`${title}: files in the last folder`}>
          {suggestions.slice(0, 8).map((file) => (
            <button
              key={file.path}
              type="button"
              disabled={disabled}
              onClick={() => onAddPath(file.path)}
              title={file.path}
              aria-label={`Add ${file.name}`}
              className="max-w-full truncate rounded-md border border-border px-2 py-1 font-mono text-[11px] text-muted-foreground transition-colors hover:border-border-strong hover:text-foreground disabled:opacity-40"
            >
              + {file.name}
            </button>
          ))}
        </div>
      )}
      {onAddPath && (
        <input
          type="text"
          value={typedPath}
          disabled={disabled}
          onChange={(event) => setTypedPath(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            // A path copied from a terminal or Finder often comes quoted.
            const path = typedPath.trim().replace(/^["']|["']$/g, "");
            if (!path) return;
            onAddPath(path);
            setTypedPath("");
          }}
          placeholder="…or type a file's full path and press Enter"
          aria-label={`${title}: add a file by its path`}
          spellCheck={false}
          className="mt-2 h-7 w-full rounded-md border border-border bg-input px-2 font-mono text-[11px] text-foreground placeholder:font-sans placeholder:text-muted-foreground/70 focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-40"
        />
      )}
    </section>
  );
});
