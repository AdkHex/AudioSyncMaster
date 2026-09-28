import { Film, Subtitles } from "lucide-react";

import { CheckRow, INPUT_CLASS, SelectBox } from "@/components/subsync/controls";
import { cx } from "@/lib/cx";
import { isVideoFile, type InputItem, type InputPool } from "@/lib/subsync/inputs";
import type { MuxTrackMeta } from "@/lib/subsync/types";

import { FileDetails } from "./FileDetails";
import { Problem, RemoveButton } from "./SubtitleItems";

/** Mux: one video, and every subtitle to add to it with its track flags. */
export function MuxInputs({
  pool,
  item,
  disabled,
  onVideo,
  onMeta,
  onRemove,
}: {
  pool: InputPool;
  item: InputItem | undefined;
  disabled: boolean;
  onVideo: (path: string) => void;
  onMeta: (path: string, patch: MuxTrackMeta) => void;
  onRemove: (path: string) => void;
}) {
  const videos = pool.files.filter(isVideoFile);
  const files = new Map(pool.files.map((file) => [file.path, file]));
  const video = item?.video ? files.get(item.video.path) : undefined;
  const others = pool.files.filter((file) => file.kind !== "video" && file.kind !== "subtitle");

  return (
    <div>
      <section aria-label="Video" className="border-b border-border px-[18px] py-3">
        <div className="flex items-center gap-2">
          <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
          {videos.length > 1 ? (
            <SelectBox
              aria-label="Video to add the subtitles to"
              value={video?.path ?? ""}
              disabled={disabled}
              onChange={(event) => onVideo(event.target.value)}
              className="flex-1"
            >
              {videos.map((file) => (
                <option key={file.path} value={file.path}>
                  {file.name}
                </option>
              ))}
            </SelectBox>
          ) : (
            <span className="min-w-0 flex-1 truncate text-[12.5px]">{video?.name ?? "No video yet"}</span>
          )}
          {video && <RemoveButton name={video.name} disabled={disabled} onClick={() => onRemove(video.path)} />}
        </div>
        {video && (
          <div className="ml-[22px] mt-0.5">
            <FileDetails file={video} />
          </div>
        )}
      </section>

      <ul aria-label="Subtitles to add">
        {(item?.subtitles ?? []).map((sub) => {
          const file = files.get(sub.path);
          return (
            <li key={sub.path} className="group border-b border-border px-[18px] py-2.5">
              <div className="flex items-center gap-2">
                <Subtitles className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
                <span className="min-w-0 flex-1 truncate text-[12.5px]" title={sub.path}>
                  {sub.name}
                </span>
                <RemoveButton name={sub.name} disabled={disabled} onClick={() => onRemove(sub.path)} />
              </div>
              {file && (
                <div className="ml-[22px] mt-0.5">
                  <FileDetails file={file} />
                </div>
              )}
              <div className="ml-[22px] mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1.5">
                <label className="flex items-center gap-1.5 text-[11.5px] text-muted-foreground">
                  Language
                  <input
                    type="text"
                    value={sub.language ?? ""}
                    maxLength={3}
                    placeholder="und"
                    spellCheck={false}
                    disabled={disabled}
                    onChange={(event) => onMeta(sub.path, { language: event.target.value.toLowerCase().replace(/[^a-z]/g, "") })}
                    className={cx(INPUT_CLASS, "w-14 font-mono")}
                  />
                </label>
                <label className="flex min-w-[160px] flex-1 items-center gap-1.5 text-[11.5px] text-muted-foreground">
                  Title
                  <input
                    type="text"
                    value={sub.title ?? ""}
                    placeholder="e.g. English SDH"
                    disabled={disabled}
                    onChange={(event) => onMeta(sub.path, { title: event.target.value })}
                    className={cx(INPUT_CLASS, "min-w-0 flex-1")}
                  />
                </label>
                <CheckRow label="Default" checked={!!sub.default} disabled={disabled} onChange={(value) => onMeta(sub.path, { default: value })} />
                <CheckRow label="Forced" checked={!!sub.forced} disabled={disabled} onChange={(value) => onMeta(sub.path, { forced: value })} />
                <CheckRow label="SDH" checked={!!sub.hearingImpaired} disabled={disabled} onChange={(value) => onMeta(sub.path, { hearingImpaired: value })} />
              </div>
            </li>
          );
        })}
      </ul>

      {others.map((file) => (
        <div key={file.path} className="group flex items-center gap-2 border-b border-border px-[18px] py-2">
          <span className="min-w-0 flex-1 truncate text-[12px] text-muted-foreground">{file.name}</span>
          <span className="text-[11px] text-warning">Not a video or subtitle</span>
          <RemoveButton name={file.name} disabled={disabled} onClick={() => onRemove(file.path)} />
        </div>
      ))}

      {item?.problem && <Problem text={item.problem} className="ml-[18px] py-2" />}
    </div>
  );
}
