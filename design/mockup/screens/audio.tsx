import {
  AddRegular,
  ArrowClockwiseRegular,
  CheckmarkCircleFilled,
  CircleRegular,
  CopyRegular,
  DeleteRegular,
  DismissRegular,
  DocumentSearchRegular,
  ErrorCircleFilled,
  FolderAddRegular,
  FolderOpenRegular,
  MoreHorizontalRegular,
  MusicNote2Regular,
  PlayRegular,
  SettingsRegular,
  SkipForwardTabRegular,
  StopRegular,
  VideoClipRegular,
} from "@fluentui/react-icons";
import type { ReactNode } from "react";

import { Btn, Cmd, Combo, DL, Empty, InfoBar, MidText, Ring, Seg, Status, TBox, Table, Toggle, Tr, cx, type St } from "../kit";
import { AutoGo, Box, DockTabs, Window, useProgress, useProto, type LcdProps } from "../shell";

/* Sample data in FPSConv's real formats: its output names, its resolve_encode
 * labels and notes, its stage names and its job-log command lines. */

const OUT = "D:\\Encodes";

/* ------------------------------- shared parts ------------------------------- */

/** A labelled control in the strip under the toolbar. */
const L = ({ label, children }: { label: string; children: ReactNode }) => (
  <span className="row" style={{ gap: 8, marginRight: 12 }}>
    <span className="t3">{label}</span>
    {children}
  </span>
);

/** A chosen file or folder in the strip: click to change, × to clear. */
const PathPick = ({ icon, value, ph, clear, w }: { icon: ReactNode; value?: string; ph: string; clear?: boolean; w: number }) => (
  <button type="button" className={cx("combo pick", !value && "placeholder")} style={{ width: w }}>
    <span className="ic">{icon}</span>
    <span className="v">{value ? <MidText text={value} tail={20} /> : ph}</span>
    {clear && value && <span className="ic x"><DismissRegular /></span>}
  </button>
);

/** A row of the queue table that names a section: Running 2, Queued 1 … */
const Group = ({ label, n }: { label: string; n: number }) => (
  <div className="grp"><span>{label}</span><span className="t3 num">{n}</span></div>
);

type Stage = { name: string; state: "done" | "run" | "wait" | "skip" | "failed"; took?: string; pct?: number; note?: string };

function StageList({ stages }: { stages: Stage[] }) {
  return (
    <div className="col" style={{ padding: "6px 16px" }}>
      {stages.map((s) => (
        <div key={s.name} className="row" style={{ gap: 10, height: 30, color: s.state === "wait" || s.state === "skip" ? "var(--text-3)" : undefined }}>
          <span style={{ width: 16, display: "grid", fontSize: 16 }}>
            {s.state === "done" ? <CheckmarkCircleFilled className="ok" /> : s.state === "run" ? <Ring size={16} /> : s.state === "failed" ? <ErrorCircleFilled className="bad" /> : s.state === "skip" ? <SkipForwardTabRegular /> : <CircleRegular style={{ color: "var(--text-4)" }} />}
          </span>
          <span style={{ width: 170 }} className="truncate">{s.name}</span>
          <span className="grow truncate t3">{s.note}</span>
          {s.state === "run" && <span className="pbar" style={{ width: 120, display: "block" }}><i style={{ width: `${s.pct}%` }} /></span>}
          <span className="num t2" style={{ width: 64, textAlign: "right" }}>{s.state === "run" ? `${Math.round(s.pct ?? 0)}%` : s.took ?? ""}</span>
        </div>
      ))}
    </div>
  );
}

function JobLog({ lines }: { lines: string[] }) {
  return (
    <div className="logv">
      {lines.map((l, i) => (
        <div key={i} className={cx(l.includes(": $ ") && "cmdline")}>{l}</div>
      ))}
    </div>
  );
}

function AudioDock({ on, stages, log, who }: { on: string; stages: Stage[]; log: string[]; who: string }) {
  const { go } = useProto();
  return (
    <div style={{ height: 250, display: "flex", flexDirection: "column" }}>
      <DockTabs
        tabs={["Progress", "Job log", "Output", "History"]}
        on={on}
        onTab={(t) => go(t === "Job log" ? "encode-log" : "encode-run")}
        tools={<span className="t3 sm" style={{ marginRight: 8 }}><MidText text={who} tail={24} /></span>}
      />
      {on === "Job log" ? <JobLog lines={log} /> : <StageList stages={stages} />}
    </div>
  );
}

function MoreFly({ fps }: { fps?: boolean }) {
  return (
    <>
      <div className="fly more-fly">
        <div className="col" style={{ gap: 6 }}>
          <span className="t3">Files at once</span>
          <Seg items={["1", "2", "3", "4", "6", "8"]} value={1} />
        </div>
        {fps && (
          <div className="col" style={{ gap: 6 }}>
            <span className="t3">Bitrate</span>
            <div className="row" style={{ gap: 8 }}><Combo value="Same as the source" w={170} /><TBox ph="kbps" w={84} /></div>
          </div>
        )}
        <div className="col" style={{ gap: 6 }}>
          <span className="t3">When the output exists</span>
          <Combo value="Write a numbered copy" w="100%" />
        </div>
      </div>
    </>
  );
}

/* ---------------------------------- Encode ---------------------------------- */

type EPhase = "empty" | "ready" | "blocked" | "run" | "log" | "done" | "more" | "mediainfo" | "import";

type EJob = { f: string; stream: string; src: string; res: string; s: St; t?: string; pct?: number; group: "Running" | "Queued" | "Ready" | "Finished"; size?: string };

const ENCODE_READY: EJob[] = [
  { f: "Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX.mkv", stream: "#0 TrueHD Atmos · 8 ch · eng", src: "TrueHD Atmos 7.1 · 48 kHz", res: "DDP 7.1 Atmos flat 1536k", s: "ready", group: "Ready" },
  { f: "Oppenheimer.2023.1080p.BluRay.mkv", stream: "#1 DTS-HD MA · 6 ch · eng", src: "DTS-HD MA 5.1 · 48 kHz", res: "DDP 5.1 1024k", s: "ready", group: "Ready" },
  { f: "Arrival.2016.Commentary.flac", stream: "#0 FLAC · 2 ch", src: "FLAC 2.0 · 96 kHz", res: "DDP 2.0 256k", s: "ready", group: "Ready" },
  { f: "Tenet.2020.WEB-DL.DDP5.1.Atmos.mka", stream: "#0 E-AC-3 Atmos · 6 ch · eng", src: "E-AC-3 Atmos 5.1 · 48 kHz", res: "DDP 5.1 Atmos 768k", s: "ready", group: "Ready" },
];

const FLAT71_STAGES = (p: number): Stage[] => [
  { name: "Extracting TrueHD", state: "done", took: "0:41" },
  { name: "truehdd decode", state: p < 100 ? "run" : "done", pct: p, note: `${(p * 0.21).toFixed(1)} GB written to temp`, took: "18:02" },
  { name: "Patching DEE", state: "wait" },
  { name: "DEE measure", state: "wait" },
  { name: "DEE encode", state: "wait" },
  { name: "Surround EX flag", state: "wait" },
  { name: "Finishing", state: "wait" },
];

const FLAT71_LOG = [
  "14:02:11  job started: Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX.mkv  →  Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX_DDP7.1Atmos_1536k.ec3",
  "14:02:11  source: TrueHD Atmos · 8 ch · 48000 Hz · 7.1 · 02:46:01 · stream #0",
  "14:02:11  target: DDP 7.1 Atmos flat 1536k",
  "14:02:11  note: flat 7.1: Lb Rb via the patched DEE (truehdd → DEE 5.2.1 patch → Surround EX flag)",
  "14:02:11  Extracting TrueHD (1/5): $ C:\\Tools\\eac3to\\eac3to.exe D:\\Films\\Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX.mkv 2: C:\\Users\\me\\AppData\\Local\\com.audiosyncmaster.app\\encode\\temp\\fpsjob_3f9c2a\\fpsconv_3f9c2a.thd -progressnumbers",
  "14:02:52  Extracting TrueHD (1/5): finished in 00:00:41",
  "14:02:52  truehdd decode (2/5): $ C:\\Tools\\truehdd.exe --progress decode --output-path C:\\…\\fpsjob_3f9c2a\\master --warp-mode normal --bed-conform C:\\…\\fpsjob_3f9c2a\\fpsconv_3f9c2a.thd",
  "14:02:52  truehdd: truehdd decode started",
  "14:20:54  truehdd: truehdd decode  ·  13.2 GB written to temp  ·  00:18:02 in this stage",
];

export function Encode({ phase, overlay }: { phase: EPhase; overlay?: ReactNode }) {
  const { go } = useProto();
  const p = useProgress(63, 6000);
  const running = phase === "run" || phase === "log";
  const done = phase === "done";

  const jobs: EJob[] = running
    ? [
        { ...ENCODE_READY[0], s: "run", t: "truehdd decode", pct: p * 0.52, group: "Running" },
        { ...ENCODE_READY[1], s: "run", t: "DEE: encode", pct: 64 + p * 0.3, group: "Running" },
        { ...ENCODE_READY[2], s: "wait", group: "Queued" },
        { ...ENCODE_READY[3], s: "ok", t: "Done", size: "623 MB", group: "Finished" },
      ]
    : done
      ? [
          { ...ENCODE_READY[0], s: "bad", t: "Failed", group: "Finished" },
          { ...ENCODE_READY[1], s: "ok", t: "Done", size: "1.21 GB", group: "Finished" },
          { ...ENCODE_READY[2], s: "ok", t: "Done", size: "41.3 MB", group: "Finished" },
          { ...ENCODE_READY[3], s: "ok", t: "Done", size: "623 MB", group: "Finished" },
        ]
      : ENCODE_READY;
  const sel = done ? 0 : 0;

  const lcd: LcdProps =
    phase === "empty"
      ? { l1: "Drop audio or video files" }
      : running
        ? { icon: "run", l1: "Encoding 2 of 4 · truehdd decode", l2: "Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX.mkv", pct: 38 + p * 0.3, time: "41 min left" }
        : done
          ? { icon: "bad", l1: "3 done · 1 failed", l2: "Dune.Part.Two: this DEE build is not the one the 7.1 patcher is validated for" }
          : phase === "blocked"
            ? { icon: "warn", l1: "4 ready · DEE is not set", l2: "Every DD and DDP encode runs through Dolby Encoding Engine" }
            : { l1: "4 ready", l2: "DDP · channels as source · Atmos flat 7.1 · Film light" };

  const strip = (
    <>
      <L label="Format"><Seg items={["DDP", "DD"]} value={0} /></L>
      <L label="Channels"><Combo value="As source" w={112} /></L>
      <L label="Bitrate"><Combo value="Auto" w={88} /></L>
      <L label="Atmos"><Toggle on /></L>
      <L label="Layout"><Combo value="Flat 7.1" w={104} /></L>
      <L label="DRC"><Combo value="Film light" w={120} /></L>
      <span className="grow" />
      <PathPick icon={<FolderOpenRegular />} value={OUT} ph="Output folder" w={190} />
      <Cmd icon={<MoreHorizontalRegular />} title="More options" onClick={() => go("encode-more")} />
    </>
  );

  const groups = ["Running", "Queued", "Ready", "Finished"] as const;
  const selJob = jobs[sel];

  let inspector: ReactNode = null;
  if (done && selJob.s === "bad")
    inspector = (
      <Box title={<MidText text={selJob.f} tail={18} />}>
        <DL rows={[["Stream", selJob.src], ["Asked for", selJob.res], ["Took", "0:02"]]} />
        <div className="errtext">flat 7.1 Atmos: this DEE build is not the one the 7.1 patcher is validated for (dee_audio_filter_ddp_atmos.dll SHA-256 9f3a0c11e2b7…, expected 3d66bcec3603…); use DEE 5.2.1-5994839 or choose the DEE-default layout</div>
        <div className="row" style={{ gap: 8 }}>
          <Btn icon={<ArrowClockwiseRegular />}>Retry</Btn>
          <Btn icon={<SettingsRegular />} onClick={() => go("prefs-encode")}>Encode settings</Btn>
        </div>
      </Box>
    );
  else if (phase !== "empty")
    inspector = (
      <Box title={<MidText text={selJob.f} tail={18} />}>
        <DL
          rows={[
            ["Container", "MKV · 2:46:01 · 71.4 GB"],
            ["Stream", selJob.src],
            ["Writes", selJob.res],
            ["As", <MidText key="n" text="Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX_DDP7.1Atmos_1536k.ec3" tail={26} />],
          ]}
        />
        <div className="note-line">flat 7.1: Lb Rb via the patched DEE (truehdd → DEE 5.2.1 patch → Surround EX flag)</div>
        <div className="links">
          <button type="button" className="cmd" onClick={() => go("encode-mediainfo")}><span className="ic"><DocumentSearchRegular /></span>MediaInfo</button>
          {running && <button type="button" className="cmd"><span className="ic"><StopRegular /></span>Cancel this job</button>}
        </div>
      </Box>
    );

  return (
    <Window
      page="encode"
      busy={running ? "encode" : undefined}
      lcd={lcd}
      strip={phase === "empty" ? undefined : strip}
      cols={phase === "empty" ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"}
      dock={running ? <AudioDock on={phase === "log" ? "Job log" : "Progress"} stages={FLAT71_STAGES(p)} log={FLAT71_LOG} who={jobs[0].f} /> : undefined}
      tools={
        <>
          <Cmd icon={<AddRegular />}>Add files</Cmd>
          <Cmd icon={<FolderAddRegular />}>Add folder</Cmd>
          {phase !== "empty" && <Cmd icon={<DeleteRegular />} title="Remove (Delete)" />}
          {phase !== "empty" && <Cmd icon={<DocumentSearchRegular />} title="MediaInfo" onClick={() => go("encode-mediainfo")} />}
        </>
      }
      primary={
        running ? (
          <Btn icon={<StopRegular />} onClick={() => go("encode-done")}>Stop all</Btn>
        ) : (
          <Btn accent icon={<PlayRegular />} disabled={phase === "empty"} kbd="Ctrl+Enter" onClick={() => go("encode-run")}>{done ? "Run again" : "Start 4"}</Btn>
        )
      }
      overlay={overlay ?? (phase === "more" ? <MoreFly /> : phase === "mediainfo" ? <MediaInfo /> : phase === "import" ? <ImportOffer /> : undefined)}
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<MusicNote2Regular />} title="Drop audio or video files">
            <Btn icon={<AddRegular />} onClick={() => go("encode-ready")}>Add files</Btn>
          </Empty>
        </section>
      ) : (
        <>
          <Box body={false} title="Queue" sub={`${jobs.length}`}>
            {phase === "blocked" && (
              <InfoBar tone="warn" actions={<Btn onClick={() => go("prefs-encode")}>Set DEE</Btn>}>Dolby Encoding Engine is not set</InfoBar>
            )}
            <Table cols="minmax(0,1fr) 160px 170px 196px" head={["File", "Source", "Writes", "Status"]}>
              {groups.map((g) => {
                const rows = jobs.filter((j) => j.group === g);
                if (!rows.length) return null;
                return (
                  <div key={g} style={{ display: "contents" }}>
                    <Group label={g} n={rows.length} />
                    {rows.map((j) => (
                      <Tr key={j.f} on={j === selJob} style={{ height: 50 }}>
                        <span className="col" style={{ minWidth: 0, gap: 1 }}>
                          <span className="cell"><span className="fi">{j.f.endsWith(".mkv") ? <VideoClipRegular /> : <MusicNote2Regular />}</span><MidText text={j.f} tail={18} /></span>
                          <span className="cell" style={{ paddingLeft: 24 }}>{j.s === "ready" ? <Combo sm ghost value={j.stream} w={210} /> : <span className="t3 sm">{j.stream}</span>}</span>
                        </span>
                        <span className="t2 truncate">{j.src}</span>
                        <span className="truncate">{j.res}{j.size && <span className="t3"> · {j.size}</span>}</span>
                        <Status s={j.s} text={j.t ? (j.s === "run" ? `${Math.round(j.pct ?? 0)}% · ${j.t}` : j.t) : undefined} pct={j.pct} />
                      </Tr>
                    ))}
                  </div>
                );
              })}
            </Table>
          </Box>
          {inspector}
        </>
      )}
      {phase === "run" && <AutoGo to="encode-done" ms={6400} />}
    </Window>
  );
}

function MediaInfo() {
  const { go } = useProto();
  const report = `General
Complete name                            : D:\\Encodes\\Dune.Part.Two.2024.2160p.UHD.BluRay.REMUX_DDP7.1Atmos_1536k.ec3
Format                                   : E-AC-3
Commercial name                          : Dolby Digital Plus with Dolby Atmos
File size                                : 1.79 GiB
Duration                                 : 2 h 46 min
Overall bit rate                         : 1 536 kb/s

Audio
Format                                   : E-AC-3 JOC
Commercial name                          : Dolby Digital Plus with Dolby Atmos
Bit rate                                 : 1 503 kb/s
Channel(s)                               : 8 channels
Channel layout                           : L R C LFE Ls Rs Lb Rb
Sampling rate                            : 48.0 kHz
Compression mode                         : Lossy
Number of dynamic objects                : 15
Bed channel count                        : 1 channel
Bed channel configuration                : LFE
Dialog Normalization                     : -27 dB`;
  return (
    <div className="smoke">
      <div className="dialog wide">
        <div className="dh">
          <span className="grow truncate">MediaInfo · Dune.Part.Two…REMUX_DDP7.1Atmos_1536k.ec3</span>
          <Cmd icon={<CopyRegular />} sm>Copy</Cmd>
          <button type="button" className="cap x" aria-label="Close" onClick={() => go("encode-done")}>
            <svg width="10" height="10"><path d="M.5.5l9 9M9.5.5l-9 9" stroke="currentColor" /></svg>
          </button>
        </div>
        <pre className="mi">{report}</pre>
      </div>
    </div>
  );
}

function ImportOffer() {
  const { go } = useProto();
  return (
    <div className="smoke">
      <div className="dialog">
        <div className="db">
          <div className="dt">Bring over your FPSConv setup?</div>
          <div>Tool paths, encode defaults, the output folder and the DEE path.</div>
        </div>
        <div className="df">
          <Btn accent onClick={() => go("encode-ready")}>Import</Btn>
          <Btn onClick={() => go("encode-empty")}>Not now</Btn>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------- FPSConv ---------------------------------- */

type FPhase = "empty" | "ready" | "run" | "done" | "more";

type FJob = { f: string; stream: string; src: string; fps: string; conv: string; why: string; s: St; t?: string; pct?: number; out: string; group: "Running" | "Queued" | "Ready" | "Finished" };

const FPS_READY: FJob[] = [
  { f: "Blade.Runner.2049.2017.1080p.BluRay.mkv", stream: "#0 E-AC-3 · 6 ch · eng", src: "E-AC-3 5.1 · 640 kbps", fps: "23.976 · video", conv: "23.976 → 25", why: "suggested", out: "Blade.Runner.2049.2017.1080p.BluRay_23_976-25.ec3", s: "ready", group: "Ready" },
  { f: "Blade.Runner.2049.Hindi.AC3.5.1.ac3", stream: "#0 AC-3 · 6 ch", src: "AC-3 5.1 · 448 kbps", fps: "unknown", conv: "24 → 25", why: "suggested · Δ 0.4 s", out: "Blade.Runner.2049.Hindi.AC3.5.1_24-25.ac3", s: "ready", group: "Ready" },
  { f: "Blade.Runner.2049.Commentary.m4a", stream: "#0 AAC · 2 ch · eng", src: "AAC 2.0 · 192 kbps", fps: "23.976 · name", conv: "23.976 → 25", why: "suggested", out: "Blade.Runner.2049.Commentary_23_976-25.aac", s: "ready", group: "Ready" },
  { f: "Blade.Runner.2049.PAL.DD.ac3", stream: "#0 AC-3 · 6 ch", src: "AC-3 5.1 · 384 kbps", fps: "25 · tag", conv: "23.976 → 25", why: "no change needed", out: "Blade.Runner.2049.PAL.DD_23_976-25.ac3", s: "ready", group: "Ready" },
];

export function FpsConv({ phase }: { phase: FPhase }) {
  const { go } = useProto();
  const p = useProgress(58, 5000);
  const running = phase === "run";
  const done = phase === "done";
  const jobs: FJob[] = running
    ? [
        { ...FPS_READY[0], s: "run", t: "DEE: encode", pct: 40 + p * 0.5, group: "Running" },
        { ...FPS_READY[1], s: "run", t: "Extracting WAV", pct: p * 0.6, group: "Running" },
        { ...FPS_READY[3], s: "wait", group: "Queued" },
        { ...FPS_READY[2], s: "ok", t: "Done", group: "Finished" },
      ]
    : done
      ? FPS_READY.map((j) => ({ ...j, s: "ok" as St, t: "Done", group: "Finished" as const }))
      : FPS_READY;
  const lcd: LcdProps =
    phase === "empty"
      ? { l1: "Drop audio tracks to re-time" }
      : running
        ? { icon: "run", l1: "Converting 2 of 4 · DEE: encode", l2: "Blade.Runner.2049.2017.1080p.BluRay.mkv", pct: 30 + p * 0.6, time: "3 min left" }
        : done
          ? { icon: "ok", l1: "4 converted", l2: `Saved to ${OUT}` }
          : { l1: "4 ready · 1 already fits the target", l2: "Target video: Blade.Runner.2049.2017.2160p.UHD.mkv · 25 fps · 2:43:48" };

  const strip = (
    <>
      <L label="Conversion"><Combo value="23.976 → 25 fps" w={150} /></L>
      <span className="t3 num" style={{ marginRight: 16 }}>×1.0427 · eac3to -23.976 -changeTo25.000</span>
      <PathPick icon={<VideoClipRegular />} value="Blade.Runner.2049.2017.2160p.UHD.mkv · 25 fps" ph="Target video" clear w={300} />
      <span className="grow" />
      <PathPick icon={<FolderOpenRegular />} value={OUT} ph="Output folder" w={190} />
      <Cmd icon={<MoreHorizontalRegular />} title="More options" onClick={() => go("fps-more")} />
    </>
  );
  const groups = ["Running", "Queued", "Ready", "Finished"] as const;
  const selJob = jobs[0];

  return (
    <Window
      page="fpsconv"
      busy={running ? "fpsconv" : undefined}
      lcd={lcd}
      strip={phase === "empty" ? undefined : strip}
      cols={phase === "empty" ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"}
      dock={running ? <AudioDock on="Progress" stages={[{ name: "Extracting WAV", state: "done", took: "1:12" }, { name: "DEE: measure", state: "done", took: "0:48" }, { name: "DEE: encode", state: "run", pct: 40 + p * 0.5 }, { name: "Finishing", state: "wait" }]} log={[]} who={selJob.f} /> : undefined}
      tools={
        <>
          <Cmd icon={<AddRegular />}>Add files</Cmd>
          <Cmd icon={<FolderAddRegular />}>Add folder</Cmd>
          {phase !== "empty" && <Cmd icon={<DeleteRegular />} title="Remove (Delete)" />}
          {phase !== "empty" && <Cmd icon={<DocumentSearchRegular />} title="MediaInfo" />}
        </>
      }
      primary={
        running ? (
          <Btn icon={<StopRegular />} onClick={() => go("fps-done")}>Stop all</Btn>
        ) : (
          <Btn accent icon={<PlayRegular />} disabled={phase === "empty"} kbd="Ctrl+Enter" onClick={() => go("fps-run")}>{done ? "Run again" : "Start 4"}</Btn>
        )
      }
      overlay={phase === "more" ? <MoreFly fps /> : undefined}
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<MusicNote2Regular />} title="Drop audio tracks to re-time">
            <Btn icon={<AddRegular />} onClick={() => go("fps-ready")}>Add files</Btn>
          </Empty>
        </section>
      ) : (
        <>
          <Box body={false} title="Queue" sub={`${jobs.length}`}>
            <Table cols="minmax(0,1fr) 150px 104px 150px 150px" head={["File", "Source", "Frame rate", "Conversion", "Status"]}>
              {groups.map((g) => {
                const rows = jobs.filter((j) => j.group === g);
                if (!rows.length) return null;
                return (
                  <div key={g} style={{ display: "contents" }}>
                    <Group label={g} n={rows.length} />
                    {rows.map((j) => (
                      <Tr key={j.f} on={j === selJob} style={{ height: 50 }}>
                        <span className="col" style={{ minWidth: 0, gap: 1 }}>
                          <span className="cell"><span className="fi">{j.f.endsWith(".mkv") ? <VideoClipRegular /> : <MusicNote2Regular />}</span><MidText text={j.f} tail={18} /></span>
                          <span className="cell t3 sm" style={{ paddingLeft: 24 }}>{j.stream}</span>
                        </span>
                        <span className="t2 truncate">{j.src}</span>
                        <span className={cx("truncate", j.fps === "unknown" && "t3")}>{j.fps}</span>
                        <span className="col" style={{ minWidth: 0, gap: 1 }}>
                          {j.s === "ready" ? <Combo sm ghost value={j.conv} w={132} /> : <span>{j.conv}</span>}
                          <span className={cx("sm", j.why === "no change needed" ? "t3" : "acc")} style={{ paddingLeft: j.s === "ready" ? 8 : 0 }}>{j.why}</span>
                        </span>
                        <Status s={j.s} text={j.t ? (j.s === "run" ? `${Math.round(j.pct ?? 0)}% · ${j.t}` : j.t) : undefined} pct={j.pct} />
                      </Tr>
                    ))}
                  </div>
                );
              })}
            </Table>
          </Box>
          <Box title={<MidText text={selJob.f} tail={18} />}>
            <DL
              rows={[
                ["Stream", "E-AC-3 5.1 · 640 kbps · 48 kHz"],
                ["Frame rate", "23.976 fps, from the video track"],
                ["Duration", "2:43:48"],
                ["Target", "25 fps · 2:37:13"],
                ["Path", "eac3to wav → deew ddp"],
                ["Writes", <MidText key="o" text={selJob.out} tail={24} />],
              ]}
            />
            <div className="note-line">audio is 23.976 fps, video is 25 fps</div>
          </Box>
        </>
      )}
      {running && <AutoGo to="fps-done" ms={5200} />}
    </Window>
  );
}
