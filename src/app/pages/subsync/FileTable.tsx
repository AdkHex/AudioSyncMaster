/** The Files list of a Subsync tool: one row per job, each with what it is
 *  paired with (or how it will be read) under its name, its result and its
 *  status. Mux has its own Tracks list instead, one row per track, with the
 *  language, title and flags each gets. */

import { DocumentTextRegular, VideoClipRegular } from "@fluentui/react-icons";
import type { ReactNode, SyntheticEvent } from "react";

import { usableTracks, type InputItem, type InputPool, type PoolAction, type PoolFile } from "@/lib/subsync/inputs";
import type { RunJob } from "@/lib/subsync/reducer";
import { inputShape, needsVideo } from "@/lib/subsync/tools";
import type { MuxTrackMeta, SubTask, TaskOptions } from "@/lib/subsync/types";
import { Chk, Combo, MidText, Status, TBox, Table, Tr } from "@/ui/kit";

import { audioText, formatName, itemName, keyPath, languageLabel, rowStatus, shortResult, subtitleTrackText, videoFacts } from "./rows";
import { formatFps } from "./settings/logic";

const plural = (n: number, one: string, many = `${one}s`) => `${n.toLocaleString()} ${n === 1 ? one : many}`;

/** Controls inside a row keep their own clicks and keys (Space, Enter). */
const stop = (event: SyntheticEvent) => event.stopPropagation();

export interface FilesProps {
  task: SubTask;
  pool: InputPool;
  items: InputItem[];
  options: TaskOptions[SubTask];
  jobs: Map<string, RunJob>;
  ranHere: boolean;
  running: boolean;
  current: string | null;
  onSelect: (key: string) => void;
  disabled: boolean;
  dispatch: (action: PoolAction) => void;
}

export function FilesTable(props: FilesProps) {
  const { task, pool, items, jobs, ranHere, running, current, onSelect } = props;
  const files = new Map(pool.files.map((file) => [file.path, file]));
  const video = inputShape(task) === "video";
  return (
    <Table cols="minmax(0,1fr) 90px 180px" head={["File", " Result", "Status"]} label="Files">
      {items.map((item) => {
        const file = files.get(keyPath(item.key));
        const job = jobs.get(item.key) ?? null;
        const status = rowStatus(item, file, job, ranHere, running);
        const result = shortResult(job?.outcome);
        const name = itemName(item);
        return (
          <Tr key={item.key} on={item.key === current} onClick={() => onSelect(item.key)} style={{ height: 50 }} label={name}>
            <span className="col" style={{ minWidth: 0, gap: 1 }}>
              <span className="cell">
                <span className="fi" aria-hidden>
                  {video || item.embedded ? <VideoClipRegular /> : <DocumentTextRegular />}
                </span>
                <MidText text={name} tail={16} />
              </span>
              <span className="cell t3 sm ss-line2">
                <Details {...props} item={item} file={file} files={files} />
              </span>
            </span>
            <span className="r num" style={{ display: "flex" }}>
              {result ?? <span className="t3">—</span>}
            </span>
            <span className="ss-status" title={status.title}>
              <Status s={status.s} text={status.text} pct={status.pct} />
            </span>
          </Tr>
        );
      })}
    </Table>
  );
}

/** The second line of a row: what the file goes with, or how it is read. */
function Details({
  task,
  item,
  file,
  files,
  pool,
  options,
  disabled,
  dispatch,
}: FilesProps & { item: InputItem; file: PoolFile | undefined; files: Map<string, PoolFile> }) {
  if (file?.pending) return <span>Reading…</span>;
  if (!file) return null;
  const shape = inputShape(task);
  const videos = pool.files.filter((f) => f.kind === "video");
  const setAudio = (path: string, track: number) => dispatch({ type: "setAudio", task, video: path, track });

  if (shape === "video") {
    if (file.kind !== "video") return <span>{file.kind === "unknown" ? "Not a media file" : `A ${file.kind} file`}</span>;
    if (task === "generate") return <AudioChoice file={file} value={item.video?.audioTrack ?? 0} disabled={disabled} onChange={setAudio} />;
    if (task === "extract") {
      const facts = [plural(file.subtitleTracks.length, "subtitle track"), videoFacts(file)].filter(Boolean).join(" · ");
      return <MidText text={facts} tail={22} />;
    }
    return <MidText text={videoFacts(file) || file.name} tail={22} />;
  }

  const parts: ReactNode[] = [];
  const paired = item.video ? files.get(item.video.path) : undefined;

  if (item.embedded) {
    const tracks = usableTracks(file, task);
    if (tracks.length > 1) {
      parts.push(
        <span key="track" onClick={stop} onKeyDown={stop}>
          <Combo<number>
            sm
            ghost
            w={240}
            disabled={disabled}
            label={`Subtitle track of ${file.name}`}
            value={item.subtitle?.track ?? tracks[0].index}
            options={tracks.map((track) => ({ value: track.index, label: subtitleTrackText(track) }))}
            onChange={(track) => dispatch({ type: "setTrack", task, video: file.path, track })}
          />
        </span>,
      );
    } else if (tracks[0]) {
      parts.push(<MidText key="track" text={subtitleTrackText(tracks[0])} tail={22} />);
    }
  } else {
    const detail = subtitleDetail(task, file, options, paired);
    if (task === "sync") {
      parts.push(
        <Pairing
          key="video"
          item={item}
          videos={videos}
          required={needsVideo(task, options)}
          disabled={disabled}
          onPair={(path) => dispatch({ type: "setPair", task, subtitle: file.path, video: path })}
        />,
      );
    } else {
      if (detail) parts.push(<span key="detail" className="ss-nowrap">{detail}</span>);
      // The video goes with the subtitle by itself when there is one; the
      // choice shows when there are several, or none was taken.
      if (videos.length > 1 || (videos.length === 1 && !item.video)) {
        parts.push(
          <Pairing
            key="video"
            item={item}
            videos={videos}
            required={false}
            disabled={disabled}
            onPair={(path) => dispatch({ type: "setPair", task, subtitle: file.path, video: path })}
          />,
        );
      }
    }
  }

  // Sync listens to the paired video: which of its audio tracks.
  if (task === "sync" && needsVideo(task, options) && paired && paired.audioTracks.length > 1) {
    parts.push(<AudioChoice key="audio" file={paired} value={item.video?.audioTrack ?? 0} disabled={disabled} onChange={setAudio} />);
  }

  return (
    <>
      {parts.map((part, index) => (
        <span key={index} className="cell" style={{ minWidth: 0, flexShrink: index === 0 ? 1 : 0 }}>
          {index > 0 && <span aria-hidden>·</span>}
          {part}
        </span>
      ))}
    </>
  );
}

/** How a subtitle will be read by this tool: "Korean → English". */
function subtitleDetail(task: SubTask, file: PoolFile, options: TaskOptions[SubTask], video: PoolFile | undefined): string | null {
  const sub = file.subtitle;
  const format = formatName(sub?.format ?? file.path.replace(/^.*\./, ""));
  const language = languageLabel(sub?.language);
  switch (task) {
    case "ocr":
      return [format, sub?.cues != null ? plural(sub.cues, "image") : null].filter(Boolean).join(" · ");
    case "translate": {
      const o = options as TaskOptions["translate"];
      const from = o.source === "auto" ? (language ?? "Detect") : (languageLabel(o.source) ?? o.source);
      return `${from} → ${languageLabel(o.target) ?? o.target}`;
    }
    case "fps": {
      const o = options as TaskOptions["fps"];
      if (o.twoPoint) return "Two known points";
      const from = o.from === "auto" ? "Auto" : formatFps(o.from);
      const toRate = o.to === "auto" ? (video?.video?.fps ?? null) : o.to;
      return toRate ? `${from} → ${formatFps(toRate)} fps` : `${from} fps → the video's rate`;
    }
    case "style": {
      const o = options as TaskOptions["style"];
      const lang = o.language === "auto" ? language : languageLabel(o.language);
      return [sub?.cues != null ? plural(sub.cues, "cue") : format, lang].filter(Boolean).join(" · ");
    }
    case "hdrSubs": {
      const hdr = video?.video && video.video.hdr !== "sdr" ? videoFacts(video).split(" · ").find((part) => /HDR|Dolby|HLG/.test(part)) : null;
      return hdr ? `${format} · for a ${hdr} video` : format;
    }
    default:
      return [format, sub?.cues != null ? plural(sub.cues, "cue") : null, language].filter(Boolean).join(" · ");
  }
}

/** The video a subtitle goes with: its name when there is one choice, a
 *  combo box when there are several or none is chosen. */
function Pairing({
  item,
  videos,
  required,
  disabled,
  onPair,
}: {
  item: InputItem;
  videos: PoolFile[];
  required: boolean;
  disabled: boolean;
  onPair: (path: string | null) => void;
}) {
  const current = item.video?.path ?? null;
  if (current && videos.length === 1) return <MidText text={item.videoName ?? current} tail={22} />;
  const name = item.subtitleName ?? "the subtitle";
  return (
    <span onClick={stop} onKeyDown={stop} style={{ minWidth: 0, display: "flex" }}>
      {required ? (
        <Combo<string>
          sm
          ghost
          w={200}
          disabled={disabled || videos.length === 0}
          label={`Video for ${name}`}
          placeholder="Choose a video"
          value={current}
          options={videos.map((video) => ({ value: video.path, label: video.name }))}
          onChange={(path) => onPair(path)}
        />
      ) : (
        <Combo<string>
          sm
          ghost
          w={200}
          disabled={disabled}
          label={`Video for ${name}`}
          value={current ?? ""}
          options={[{ value: "", label: "No video" }, ...videos.map((video) => ({ value: video.path, label: video.name }))]}
          onChange={(path) => onPair(path || null)}
        />
      )}
    </span>
  );
}

function AudioChoice({
  file,
  value,
  disabled,
  onChange,
}: {
  file: PoolFile;
  value: number;
  disabled: boolean;
  onChange: (path: string, track: number) => void;
}) {
  if (file.audioTracks.length === 0) return <span>No audio</span>;
  if (file.audioTracks.length === 1) return <MidText text={audioText(file.audioTracks[0])} tail={22} />;
  return (
    <span onClick={stop} onKeyDown={stop} style={{ display: "flex", minWidth: 0 }}>
      <Combo<number>
        sm
        ghost
        w={240}
        disabled={disabled}
        label={`Audio track of ${file.name}`}
        value={value}
        options={file.audioTracks.map((track) => ({ value: track.index, label: audioText(track) }))}
        onChange={(track) => onChange(file.path, track)}
      />
    </span>
  );
}

// -------------------------------------------------------------------- mux

/** "Video, 2 audio, 1 subtitle" */
function streams(file: PoolFile): string {
  if (file.pending) return "Reading…";
  return ["Video", file.audioTracks.length ? `${file.audioTracks.length} audio` : null, file.subtitleTracks.length ? plural(file.subtitleTracks.length, "subtitle") : null]
    .filter(Boolean)
    .join(", ");
}

export function MuxTable({
  task,
  pool,
  item,
  current,
  onSelect,
  disabled,
  dispatch,
}: {
  task: SubTask;
  pool: InputPool;
  item: InputItem | undefined;
  current: string | null;
  onSelect: (key: string) => void;
  disabled: boolean;
  dispatch: (action: PoolAction) => void;
}) {
  const videos = pool.files.filter((file) => file.kind === "video");
  const video = videos.find((file) => file.path === item?.video?.path) ?? null;
  const others = pool.files.filter((file) => file.kind !== "video" && file.kind !== "subtitle");
  const files = new Map(pool.files.map((file) => [file.path, file]));
  return (
    <Table cols="minmax(0,1fr) 72px 170px 64px 64px 48px" head={["Track", "Language", "Title", "Default", "Forced", "SDH"]} label="Tracks">
      {video && (
        <Tr on={current === video.path} onClick={() => onSelect(video.path)} label={video.name}>
          <span className="cell">
            <span className="fi" aria-hidden><VideoClipRegular /></span>
            {videos.length > 1 ? (
              <span onClick={stop} onKeyDown={stop} style={{ minWidth: 0, display: "flex" }}>
                <Combo<string>
                  sm
                  ghost
                  w={320}
                  disabled={disabled}
                  label="Video to add the subtitles to"
                  value={video.path}
                  options={videos.map((file) => ({ value: file.path, label: file.name }))}
                  onChange={(path) => dispatch({ type: "setMuxVideo", task, video: path })}
                />
              </span>
            ) : (
              <MidText text={video.name} />
            )}
          </span>
          <span className="t3">—</span>
          <span className="t3 truncate">{streams(video)}</span>
          <span />
          <span />
          <span />
        </Tr>
      )}
      {(item?.subtitles ?? []).map((sub) => {
        const file = files.get(sub.path);
        const set = (patch: MuxTrackMeta) => dispatch({ type: "setMuxMeta", task, path: sub.path, patch });
        return (
          <Tr key={sub.path} on={current === sub.path} onClick={() => onSelect(sub.path)} label={sub.name}>
            <span className="cell" style={{ paddingLeft: 24 }}>
              <span className="fi" aria-hidden><DocumentTextRegular /></span>
              <MidText text={sub.name} />
              {file?.pending && <span className="t3 sm">Reading…</span>}
            </span>
            <span onClick={stop} onKeyDown={stop}>
              <TBox
                className="ss-cellbox num"
                w="100%"
                label={`Language of ${sub.name}`}
                value={sub.language ?? ""}
                maxLength={3}
                placeholder="und"
                spellCheck={false}
                disabled={disabled}
                onChange={(value) => set({ language: value.toLowerCase().replace(/[^a-z]/g, "") })}
              />
            </span>
            <span onClick={stop} onKeyDown={stop}>
              <TBox
                className="ss-cellbox"
                w="100%"
                label={`Title of ${sub.name}`}
                value={sub.title ?? ""}
                placeholder="Title"
                disabled={disabled}
                onChange={(title) => set({ title })}
              />
            </span>
            <Chk name={`${sub.name} is the default track`} on={!!sub.default} disabled={disabled} onChange={(value) => set({ default: value })} />
            <Chk name={`${sub.name} is forced`} on={!!sub.forced} disabled={disabled} onChange={(value) => set({ forced: value })} />
            <Chk name={`${sub.name} is for the deaf and hard of hearing`} on={!!sub.hearingImpaired} disabled={disabled} onChange={(value) => set({ hearingImpaired: value })} />
          </Tr>
        );
      })}
      {others.map((file) => (
        <Tr key={file.path} on={current === file.path} onClick={() => onSelect(file.path)} label={file.name}>
          <span className="cell">
            <span className="fi" aria-hidden><DocumentTextRegular /></span>
            <MidText text={file.name} className="t3" />
          </span>
          <span />
          <Status s="warn" text={file.pending ? "Reading…" : "Not a video or subtitle"} />
          <span />
          <span />
          <span />
        </Tr>
      ))}
    </Table>
  );
}
