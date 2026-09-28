import { MuxInputs } from "@/components/subsync/inputs/MuxInputs";
import { SubtitleItems } from "@/components/subsync/inputs/SubtitleItems";
import { VideoItems } from "@/components/subsync/inputs/VideoItems";
import type { InputItem, InputPool, PoolAction } from "@/lib/subsync/inputs";
import { inputShape, needsVideo } from "@/lib/subsync/tools";
import type { MediaRef, SubTask, TaskOptions } from "@/lib/subsync/types";

/** The open task's queue, in the shape that task takes its inputs. */
export function QueueList({
  task,
  pool,
  items,
  options,
  disabled,
  dispatch,
  onPreview,
}: {
  task: SubTask;
  pool: InputPool;
  items: InputItem[];
  options: TaskOptions[SubTask];
  disabled: boolean;
  dispatch: (action: PoolAction) => void;
  onPreview: (ref: MediaRef, name: string) => void;
}) {
  const remove = (path: string) => dispatch({ type: "remove", task, path });
  const audio = (video: string, track: number) => dispatch({ type: "setAudio", task, video, track });

  switch (inputShape(task)) {
    case "mux":
      return (
        <MuxInputs
          pool={pool}
          item={items[0]}
          disabled={disabled}
          onVideo={(video) => dispatch({ type: "setMuxVideo", task, video })}
          onMeta={(path, patch) => dispatch({ type: "setMuxMeta", task, path, patch })}
          onRemove={remove}
        />
      );
    case "video":
      return (
        <VideoItems
          pool={pool}
          items={items}
          disabled={disabled}
          chooseAudio={task === "generate"}
          onAudio={audio}
          onRemove={remove}
        />
      );
    default:
      return (
        <SubtitleItems
          task={task}
          pool={pool}
          items={items}
          disabled={disabled}
          listensToAudio={task === "sync" && needsVideo(task, options)}
          onPair={(subtitle, video) => dispatch({ type: "setPair", task, subtitle, video })}
          onTrack={(video, track) => dispatch({ type: "setTrack", task, video, track })}
          onAudio={audio}
          onRemove={remove}
          onPreview={onPreview}
        />
      );
  }
}
