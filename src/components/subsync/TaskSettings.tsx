import { SETTINGS_PANELS, type SettingsPanelProps } from "@/components/subsync/settings";
import type { SubTask } from "@/lib/subsync/types";

/** The open task's settings panel. The map is typed per task; the
 *  workspace holds the task as a union, so this is the one place the two
 *  are joined. */
export function TaskSettings({ task, ...props }: SettingsPanelProps<SubTask> & { task: SubTask }) {
  const Panel = SETTINGS_PANELS[task] as unknown as React.ComponentType<SettingsPanelProps<SubTask>>;
  return <Panel {...props} />;
}
