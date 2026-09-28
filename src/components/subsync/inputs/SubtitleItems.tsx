import { Eye, Film, Subtitles, X } from "lucide-react";

import { LinkButton, SelectBox } from "@/components/subsync/controls";
import { cx } from "@/lib/cx";
import {
  isVideoFile,
  trackLabel,
  usableTracks,
  type InputItem,
  type InputPool,
  type PoolFile,
} from "@/lib/subsync/inputs";
import { audioTrackLabel } from "@/lib/subsync/format";
import type { MediaRef, SubTask } from "@/lib/subsync/types";

import { FileDetails } from "./FileDetails";

interface SubtitleItemsProps {
  task: SubTask;
  pool: InputPool;
  items: InputItem[];
  disabled: boolean;
  /** Whether the audio of the paired video is listened to (sync). */
  listensToAudio: boolean;
  onPair: (subtitle: string, video: string | null) => void;
  onTrack: (video: string, track: number) => void;
  onAudio: (video: string, track: number) => void;
  onRemove: (path: string) => void;
  onPreview: (ref: MediaRef, name: string) => void;
}

/** The queue of a subtitle tool: each subtitle (a file, or a track of a
 *  video) with the video it goes with. */
export function SubtitleItems({
  task,
  pool,
  items,
  disabled,
  listensToAudio,
  onPair,
  onTrack,
  onAudio,
  onRemove,
  onPreview,
}: SubtitleItemsProps) {
  const files = new Map(pool.files.map((file) => [file.path, file]));
  const videos = pool.files.filter(isVideoFile);
  // Videos used only as the partner of a subtitle file have no row of their
  // own; they are listed after the queue so they can still be removed.
  const shown = new Set(items.filter((item) => item.embedded).map((item) => item.video?.path));
  const partners = videos.filter((video) => !shown.has(video.path));

  return (
    <>
      <ul aria-label="Queue">
        {items.map((item) => {
          const own = item.embedded ? files.get(item.video!.path) : files.get(item.subtitle?.path ?? "");
          const video = item.video ? files.get(item.video.path) : undefined;
          return (
            <li key={item.key} className="group border-b border-border px-[18px] py-2.5">
              <div className="flex items-center gap-2">
                {item.embedded ? (
                  <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                ) : (
                  <Subtitles className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                )}
                <span className="min-w-0 flex-1 truncate text-[12.5px]" title={own?.path}>
                  {own?.name ?? item.subtitleName}
                </span>
                {item.subtitle && !own?.pending && (
                  <LinkButton
                    onClick={() => onPreview(item.subtitle!, item.embedded ? `${own?.name} ${item.subtitleName}` : (own?.name ?? ""))}
                    className="flex items-center gap-1"
                  >
                    <Eye className="h-3 w-3" aria-hidden />
                    Cues
                  </LinkButton>
                )}
                <RemoveButton name={own?.name ?? ""} disabled={disabled} onClick={() => own && onRemove(own.path)} />
              </div>
              {own && (
                <div className="ml-[22px] mt-0.5">
                  <FileDetails file={own} />
                </div>
              )}

              <div className="ml-[22px] mt-1.5 grid grid-cols-[64px_minmax(0,1fr)] items-center gap-x-2 gap-y-1">
                {item.embedded && own ? (
                  <TrackRow file={own} task={task} value={item.subtitle?.track ?? null} disabled={disabled} onChange={(track) => onTrack(own.path, track)} />
                ) : (
                  <>
                    <label htmlFor={`${item.key}-video`} className="text-[11.5px] text-muted-foreground">
                      Video
                    </label>
                    <SelectBox
                      id={`${item.key}-video`}
                      value={item.video?.path ?? ""}
                      disabled={disabled}
                      onChange={(event) => onPair(item.subtitle!.path, event.target.value || null)}
                    >
                      <option value="">No video</option>
                      {videos.map((file) => (
                        <option key={file.path} value={file.path}>
                          {file.name}
                        </option>
                      ))}
                    </SelectBox>
                  </>
                )}
                {listensToAudio && video && video.audioTracks.length > 1 && (
                  <AudioRow file={video} value={item.video?.audioTrack ?? 0} disabled={disabled} onChange={(track) => onAudio(video.path, track)} />
                )}
              </div>

              {item.problem && <Problem text={item.problem} />}
            </li>
          );
        })}
      </ul>

      {partners.length > 0 && (
        <section aria-label="Videos" className="px-[18px] py-3">
          <h3 className="mb-1.5 text-[11px] font-semibold text-muted-foreground">Videos</h3>
          <ul className="flex flex-col gap-1.5">
            {partners.map((file) => (
              <li key={file.path} className="group">
                <div className="flex items-center gap-2">
                  <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                  <span className="min-w-0 flex-1 truncate text-[12px]" title={file.path}>
                    {file.name}
                  </span>
                  <RemoveButton name={file.name} disabled={disabled} onClick={() => onRemove(file.path)} />
                </div>
                <div className="ml-[22px]">
                  <FileDetails file={file} />
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
  );
}

function TrackRow({
  file,
  task,
  value,
  disabled,
  onChange,
}: {
  file: PoolFile;
  task: SubTask;
  value: number | null;
  disabled: boolean;
  onChange: (track: number) => void;
}) {
  const tracks = usableTracks(file, task);
  if (tracks.length === 0) return null;
  const id = `track-${file.path}`;
  return (
    <>
      <label htmlFor={id} className="text-[11.5px] text-muted-foreground">
        Track
      </label>
      <SelectBox id={id} value={value ?? tracks[0].index} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))}>
        {tracks.map((track) => (
          <option key={track.index} value={track.index}>
            {trackLabel(track)}
          </option>
        ))}
      </SelectBox>
    </>
  );
}

export function AudioRow({
  file,
  value,
  disabled,
  onChange,
}: {
  file: PoolFile;
  value: number;
  disabled: boolean;
  onChange: (track: number) => void;
}) {
  const id = `audio-${file.path}`;
  return (
    <>
      <label htmlFor={id} className="text-[11.5px] text-muted-foreground">
        Audio
      </label>
      <SelectBox id={id} value={value} disabled={disabled} onChange={(event) => onChange(Number(event.target.value))}>
        {file.audioTracks.map((track) => (
          <option key={track.index} value={track.index}>
            {audioTrackLabel(track)}
          </option>
        ))}
      </SelectBox>
    </>
  );
}

export function Problem({ text, className }: { text: string; className?: string }) {
  return <p className={cx("ml-[22px] mt-1 text-[11.5px] text-warning", className)}>{text}</p>;
}

export function RemoveButton({ name, disabled, onClick }: { name: string; disabled: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="-mr-1 shrink-0 rounded p-0.5 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover:opacity-100 disabled:opacity-0"
    >
      <X className="h-3 w-3" aria-hidden />
      <span className="sr-only">Remove {name}</span>
    </button>
  );
}
