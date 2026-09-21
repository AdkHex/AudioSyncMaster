import { Play, Square } from "lucide-react";
import { memo, useId } from "react";

import { FilePanel } from "@/components/FilePanel";
import { cx } from "@/lib/cx";
import {
  DUB_CODECS,
  DUB_RATES,
  type AppSettings,
  type DubScope,
  type FileItem,
  type MediaProbe,
  type SyncMode,
  type TrackListing,
} from "@/lib/types";

/** The dub sync choices that belong beside the files rather than in Settings:
 *  what to write, and whether to put it into a copy of the video. */
export type DubOutputOptions = Pick<
  AppSettings,
  "dubCodec" | "dubMux" | "dubLanguage" | "maxWorkers" | "dubRate"
>;

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

  /** What the dub sync tab pairs: movies by filename, series by episode. */
  dubScope: DubScope;
  onDubScopeChange: (scope: DubScope) => void;

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

  /** Dub sync output options; shown in that mode only. */
  dubOptions: DubOutputOptions;
  onDubOptionsChange: (patch: Partial<DubOutputOptions>) => void;

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
  dubOptions,
  onDubOptionsChange,
  runLabel,
  canRun,
  runBlockedReason,
  onRun,
  onStop,
  dubScope,
  onDubScopeChange,
}: SidebarProps) {
  const dubsync = mode === "dubsync";
  return (
    <aside className="flex w-[288px] shrink-0 flex-col border-r border-border">
      <div
        className="min-h-0 flex-1 overflow-y-auto px-4 py-[18px]"
        onDragOver={(event) => event.preventDefault()}
      >
        {dubsync && (
          <>
            <DubScopeSwitch
              scope={dubScope}
              disabled={busy}
              onChange={onDubScopeChange}
            />
            <div className="mt-4" />
          </>
        )}

        <div onDragEnter={() => onDragEnter("video")}>
          <FilePanel
            kind="video"
            // A bare original-language track works as well as the video here:
            // only its audio is compared, and only its audio fills the gaps.
            needs={dubsync ? "audio" : "video"}
            title={
              dubsync
                ? dubScope === "movies"
                  ? "Movies"
                  : "Episodes"
                : "Video"
            }
            hint={
              dubsync
                ? dubScope === "movies"
                  ? "Select the movies (multi-select works), or drop them"
                  : "Drop the episode folder, or click to browse"
                : "Drop files, or click to browse"
            }
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
            title={
              mode === "compare"
                ? "Dubs to test"
                : dubsync
                  ? dubScope === "movies"
                    ? "Movie dubs"
                    : "Episode dubs"
                  : "Dub"
            }
            hint={
              mode === "movie"
                ? "Drop the audio track"
                : mode === "compare"
                  ? "Drop the tracks to test"
                  : dubsync && dubScope === "movies"
                    ? "Select the movie dubs (multi-select works), or drop them"
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

        {dubsync && (
          <>
            <hr className="my-5 border-border" />
            <DubOutput options={dubOptions} disabled={busy} onChange={onDubOptionsChange} />
          </>
        )}
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

const SELECT_ARROW =
  "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23969696' stroke-width='2.5'><path d='m6 9 6 6 6-6'/></svg>\")";

/** What the dub tab is pairing, chosen before the files: movies are paired
 *  by filename, episodes by season/episode number. The labels of both slots
 *  follow the choice, since a season and a folder of movies ask different
 *  questions. */
function DubScopeSwitch({
  scope,
  disabled,
  onChange,
}: {
  scope: "movies" | "series";
  disabled: boolean;
  onChange: (scope: "movies" | "series") => void;
}) {
  const options: { id: "movies" | "series"; label: string }[] = [
    { id: "movies", label: "Movies" },
    { id: "series", label: "Series" },
  ];
  return (
    <div
      role="radiogroup"
      aria-label="What to sync"
      className="grid grid-cols-2 gap-1 rounded-[9px] bg-elevated p-1"
    >
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          role="radio"
          aria-checked={scope === option.id}
          disabled={disabled}
          onClick={() => onChange(option.id)}
          className={cx(
            "rounded-[7px] px-3 py-1.5 text-[12px] font-medium transition-colors",
            scope === option.id
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
            "disabled:opacity-50",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/** What the synced track is written as, and whether it also goes into a
 *  copy of the video. The track always lands beside the video; the video
 *  itself is never touched. */
function DubOutput({
  options,
  disabled,
  onChange,
}: {
  options: DubOutputOptions;
  disabled: boolean;
  onChange: (patch: Partial<DubOutputOptions>) => void;
}) {
  const ids = useId();
  return (
    <section aria-label="Output">
      <h2 className="mb-2 text-[11px] font-semibold text-muted-foreground">Output</h2>

      <label htmlFor={`${ids}-codec`} className="block text-[11.5px] text-muted-foreground">
        Write the synced track as
      </label>
      <select
        id={`${ids}-codec`}
        value={options.dubCodec}
        disabled={disabled}
        onChange={(event) => onChange({ dubCodec: event.target.value as DubOutputOptions["dubCodec"] })}
        className="mt-1 w-full appearance-none rounded-md bg-elevated px-2 py-1.5 pr-6 text-[12px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        style={{
          backgroundImage: SELECT_ARROW,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "right 7px center",
        }}
      >
        {DUB_CODECS.map((codec) => (
          <option key={codec.id} value={codec.id}>
            {codec.label}
          </option>
        ))}
      </select>
      <p className="mt-1 text-[10.5px] leading-snug text-muted-foreground/80">
        Written beside the video, the video's exact length. A dub in a format that
        cannot be written back (TrueHD, DTS) becomes FLAC.
      </p>

      <label className="mt-3 flex cursor-pointer items-start gap-2 text-[12px]">
        <input
          type="checkbox"
          checked={options.dubMux}
          disabled={disabled}
          onChange={(event) => onChange({ dubMux: event.target.checked })}
          className="mt-[3px] h-[15px] w-[15px] accent-primary"
        />
        <span>
          Also write a copy of the video with this track added
          <span className="block text-[10.5px] leading-snug text-muted-foreground/80">
            Every original stream is kept; the synced dub is added as one more.
          </span>
        </span>
      </label>

      {options.dubMux && (
        <div className="mt-2.5 flex items-center gap-2.5">
          <label htmlFor={`${ids}-lang`} className="text-[11.5px] text-muted-foreground">
            Language tag
          </label>
          <input
            id={`${ids}-lang`}
            type="text"
            value={options.dubLanguage}
            disabled={disabled}
            maxLength={3}
            placeholder="hin"
            spellCheck={false}
            onChange={(event) =>
              onChange({ dubLanguage: event.target.value.toLowerCase().replace(/[^a-z]/g, "") })
            }
            className="w-16 rounded-md border border-border-strong bg-input px-2 py-1 font-mono text-xs focus:outline-none focus:ring-2 focus:ring-ring/40 disabled:opacity-50"
          />
        </div>
      )}

      <label htmlFor={`${ids}-rate`} className="mt-3 block text-[11.5px] text-muted-foreground">
        The dub was mastered at
      </label>
      <select
        id={`${ids}-rate`}
        value={options.dubRate === null ? "auto" : String(options.dubRate)}
        disabled={disabled}
        onChange={(event) =>
          onChange({ dubRate: event.target.value === "auto" ? null : Number(event.target.value) })
        }
        className="mt-1 w-full appearance-none rounded-md bg-elevated px-2 py-1.5 pr-6 text-[12px] text-foreground focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
        style={{
          backgroundImage: SELECT_ARROW,
          backgroundRepeat: "no-repeat",
          backgroundPosition: "right 7px center",
        }}
      >
        <option value="auto">Find out from the audio</option>
        {DUB_RATES.map((rate) => (
          <option key={rate.value} value={String(rate.value)}>
            {rate.label}
          </option>
        ))}
      </select>
      <p className="mt-1 text-[10.5px] leading-snug text-muted-foreground/80">
        Leave it to the audio unless the plan says the rate could not be confirmed. A dub
        timed to a 23.976 fps master on a 24 fps video runs 0.1% slow, a millisecond a second.
      </p>

      <p className="mt-3 text-[10.5px] leading-snug text-muted-foreground/80">
        Pairs are synced in parallel, up to {options.maxWorkers} at a time (Settings → Workers).
      </p>
    </section>
  );
}
