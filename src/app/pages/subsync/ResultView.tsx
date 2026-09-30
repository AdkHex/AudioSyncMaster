/** What one finished job did, in the side panel's Result tab: its summary,
 *  anything worth a second look, the numbers behind it, and the files it
 *  wrote, each with Reveal, Open and "Use as input for…". */

import {
  DocumentTextRegular,
  FolderOpenRegular,
  FolderRegular,
  OpenRegular,
  VideoClipRegular,
  WarningFilled,
  WindowConsoleRegular,
} from "@fluentui/react-icons";

import type { RunJob } from "@/lib/subsync/reducer";
import { TASK_LABELS, useAsTargets } from "@/lib/subsync/tools";
import type { SubTask, TaskOutputFile } from "@/lib/subsync/types";
import { Cmd, Combo, DL, Links, MidText, baseName } from "@/ui/kit";

import { reportRows } from "./rows";

export interface ResultActions {
  onReveal: (path: string) => void;
  onOpen: (path: string) => void;
  onUseAs: (output: TaskOutputFile, task: SubTask) => void;
  onOpenConsole: () => void;
}

export function ResultView({ job, actions }: { job: RunJob; actions: ResultActions }) {
  const outcome = job.outcome;
  if (!outcome) {
    return (
      <span className="t3">
        {job.status === "cancelled" ? "Stopped before it finished." : job.status === "running" ? "Running…" : "Waiting to run."}
      </span>
    );
  }
  if (outcome.error) {
    return (
      <>
        <div className="ss-fld">
          <span className="ss-sum bad">This file failed.</span>
          <span className="t2">{outcome.error}</span>
        </div>
        <Links>
          <Cmd sm icon={<WindowConsoleRegular />} onClick={actions.onOpenConsole}>
            Open the output
          </Cmd>
        </Links>
      </>
    );
  }

  const rows = reportRows(outcome);
  return (
    <>
      {outcome.summary && <p className="ss-sum">{outcome.summary}</p>}

      {outcome.warnings.length > 0 && (
        <ul className="ss-warnings" aria-label="Warnings">
          {outcome.warnings.map((warning, index) => (
            <li key={index} className="row">
              <WarningFilled className="warn" aria-hidden />
              <span>{warning}</span>
            </li>
          ))}
        </ul>
      )}

      {rows.length > 0 && <DL rows={rows.map(([k, v]) => [k, <span title={v}>{v}</span>])} />}

      {outcome.outputs.length > 0 && (
        <div className="ss-sec">
          <span className="sec">Written</span>
          <ul className="ss-files" aria-label="Files written">
            {outcome.outputs.map((output) => (
              <OutputRow key={output.path} output={output} actions={actions} />
            ))}
          </ul>
        </div>
      )}
    </>
  );
}

function OutputRow({ output, actions }: { output: TaskOutputFile; actions: ResultActions }) {
  const targets = useAsTargets(output);
  const name = baseName(output.path);
  const icon = output.kind === "video" ? <VideoClipRegular /> : output.kind === "folder" ? <FolderRegular /> : <DocumentTextRegular />;
  return (
    <li className="ss-fld">
      <span className="cell" title={output.path}>
        <span className="fi" aria-hidden>{icon}</span>
        <MidText text={name} tail={16} />
      </span>
      {output.label && <span className="sm t3 ss-indent">{output.label}</span>}
      <div className="ss-indent">
        <Links>
          <Cmd sm icon={<FolderOpenRegular />} title={`Show ${name} in its folder`} onClick={() => actions.onReveal(output.path)}>
            Reveal
          </Cmd>
          {output.kind !== "folder" && (
            <Cmd sm icon={<OpenRegular />} title={`Open ${name}`} onClick={() => actions.onOpen(output.path)}>
              Open
            </Cmd>
          )}
        </Links>
      </div>
      {targets.length > 0 && (
        <div className="ss-indent">
          <Combo<SubTask>
            sm
            ghost
            w={170}
            label={`Use ${name} as input for another tool`}
            value={null}
            placeholder="Use as input for…"
            options={targets.map((task) => ({ value: task, label: TASK_LABELS[task] }))}
            onChange={(task) => actions.onUseAs(output, task)}
            style={{ marginLeft: -11 }}
          />
        </div>
      )}
    </li>
  );
}
