import { Film } from "lucide-react";

import type { InputItem, InputPool } from "@/lib/subsync/inputs";

import { FileDetails } from "./FileDetails";
import { AudioRow, Problem, RemoveButton } from "./SubtitleItems";

/** The queue of a video tool (Generate, Tone-map, Extract): one row per
 *  video, with its audio stream where speech is what matters. */
export function VideoItems({
  pool,
  items,
  disabled,
  chooseAudio,
  onAudio,
  onRemove,
}: {
  pool: InputPool;
  items: InputItem[];
  disabled: boolean;
  chooseAudio: boolean;
  onAudio: (video: string, track: number) => void;
  onRemove: (path: string) => void;
}) {
  return (
    <ul aria-label="Queue">
      {items.map((item, index) => {
        const file = pool.files[index];
        if (!file) return null;
        return (
          <li key={item.key} className="group border-b border-border px-[18px] py-2.5">
            <div className="flex items-center gap-2">
              <Film className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden />
              <span className="min-w-0 flex-1 truncate text-[12.5px]" title={file.path}>
                {file.name}
              </span>
              <RemoveButton name={file.name} disabled={disabled} onClick={() => onRemove(file.path)} />
            </div>
            <div className="ml-[22px] mt-0.5">
              <FileDetails file={file} />
            </div>
            {chooseAudio && file.audioTracks.length > 1 && (
              <div className="ml-[22px] mt-1.5 grid grid-cols-[64px_minmax(0,1fr)] items-center gap-x-2">
                <AudioRow
                  file={file}
                  value={item.video?.audioTrack ?? 0}
                  disabled={disabled}
                  onChange={(track) => onAudio(file.path, track)}
                />
              </div>
            )}
            {item.problem && <Problem text={item.problem} />}
          </li>
        );
      })}
    </ul>
  );
}
