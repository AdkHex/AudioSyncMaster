/** Per-tool settings panels of Subsync. The workspace renders
 *  SETTINGS_PANELS[task] with that task's options and the shared props. */

import type { SubTask } from "@/lib/subsync/types";

import { FpsPanel } from "./FpsPanel";
import { GeneratePanel } from "./GeneratePanel";
import { HdrSubsPanel } from "./HdrSubsPanel";
import { OcrPanel } from "./OcrPanel";
import type { SettingsPanelProps } from "./props";
import { ConvertPanel, ExtractPanel, MuxPanel } from "./SmallPanels";
import { StylePanel } from "./StylePanel";
import { SyncPanel } from "./SyncPanel";
import { TonemapPanel } from "./TonemapPanel";
import { TranslatePanel } from "./TranslatePanel";

export type { SettingsPanelProps } from "./props";

export const SETTINGS_PANELS: { [K in SubTask]: React.ComponentType<SettingsPanelProps<K>> } = {
  sync: SyncPanel,
  fps: FpsPanel,
  ocr: OcrPanel,
  translate: TranslatePanel,
  generate: GeneratePanel,
  style: StylePanel,
  hdrSubs: HdrSubsPanel,
  tonemap: TonemapPanel,
  convert: ConvertPanel,
  extract: ExtractPanel,
  mux: MuxPanel,
};
