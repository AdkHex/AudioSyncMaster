import { Check } from "lucide-react";

import { Dialog } from "@/components/Dialog";
import { Button, ProgressBar, Spinner } from "@/components/ui";

export interface ApplyState {
  /** Files finished so far, most recent last. */
  written: string[];
  /** The file currently being written, if any. */
  current: string | null;
  done: number;
  total: number;
}

interface ApplyProgressDialogProps {
  state: ApplyState | null;
  onCancel: () => void;
  cancelling: boolean;
}

/** Writing corrected files can take minutes on multi-GB sources. The original
 *  app showed nothing at all until the final toast, so a long write looked
 *  indistinguishable from a hang. */
export function ApplyProgressDialog({ state, onCancel, cancelling }: ApplyProgressDialogProps) {
  if (!state) return null;

  const { written, current, done, total } = state;
  const percent = total > 0 ? (done / total) * 100 : 0;

  return (
    <Dialog
      open
      title="Writing corrected files"
      description={
        total > 0 ? `${done} of ${total} written` : "Preparing…"
      }
      // Dismissing mid-write would leave a half-written file with no way back
      // to this dialog; stopping is offered explicitly instead.
      dismissible={false}
      className="max-w-md"
      footer={
        <>
          <span className="flex-1" />
          <Button size="sm" onClick={onCancel} disabled={cancelling}>
            {cancelling ? "Stopping…" : "Stop after this file"}
          </Button>
        </>
      }
    >
      <div className="space-y-4 py-3">
        <ProgressBar percent={percent} label="Writing progress" />

        <ul className="flex flex-col gap-2">
          {written.map((path) => (
            <li key={path} className="flex items-center gap-2.5 text-xs text-muted-foreground">
              <Check className="h-3.5 w-3.5 shrink-0 text-success" aria-hidden />
              <span className="min-w-0 flex-1 truncate" title={path}>
                {path.replace(/^.*[\\/]/, "")}
              </span>
            </li>
          ))}

          {current && (
            <li className="flex items-center gap-2.5 text-xs font-medium">
              <Spinner className="h-3.5 w-3.5 border-[1.8px]" />
              <span className="min-w-0 flex-1 truncate" title={current}>
                {current.replace(/^.*[\\/]/, "")}
              </span>
            </li>
          )}
        </ul>

        <p className="text-[11.5px] text-muted-foreground">
          Source files are never modified — each correction is written as a new file.
        </p>
      </div>
    </Dialog>
  );
}
