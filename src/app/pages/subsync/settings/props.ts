import type { Capabilities, ProbedFile, SubTask, TaskOptions } from "@/lib/subsync/types";

/** What the workspace hands every tool's settings panel. */
export interface SettingsPanelProps<K extends SubTask> {
  options: TaskOptions[K];
  onChange: (patch: Partial<TaskOptions[K]>) => void;
  /** Engine availability and packs; null while loading. */
  caps: Capabilities | null;
  /** True while a run is in progress. */
  disabled: boolean;
  /** Opens Preferences › Subtitles at that pack ("keys" = the API keys). */
  onInstallPack: (packId: string) => void;
  /** For reference inputs. */
  pickFile: (accept: "video" | "subtitle" | "any") => Promise<ProbedFile | null>;
  /** fps of the first queued video, if known. */
  videoFps: number | null;
}
