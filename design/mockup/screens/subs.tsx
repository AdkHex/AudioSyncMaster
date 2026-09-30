import {
  AddRegular,
  BrightnessHighRegular,
  CheckmarkRegular,
  ChevronDownRegular,
  ChevronRightRegular,
  DeleteRegular,
  DocumentMultipleRegular,
  DocumentTextRegular,
  FolderAddRegular,
  GaugeRegular,
  HdrRegular,
  MicRegular,
  PlayRegular,
  ScanTextRegular,
  StopRegular,
  TextEditStyleRegular,
  TimerRegular,
  TranslateRegular,
  VideoClipRegular,
  WarningFilled,
} from "@fluentui/react-icons";
import type { ReactNode } from "react";

import { CUES, OCR } from "../data";
import { Btn, Chk, Cmd, Combo, Empty, MidText, Seg, Slider, Status, TBox, Table, Toggle, Tr, cx, type St } from "../kit";
import { AutoGo, Box, DockTabs, SUB_TOOLS, Window, useProgress, useProto, type LcdProps } from "../shell";

/* ------------------------- settings pane building blocks ------------------------- */

const F = ({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) => (
  <div className="col" style={{ gap: 6 }}>
    <span className="t3">{label}</span>
    {children}
    {hint && <span className="sm t3">{hint}</span>}
  </div>
);
const T = ({ label, on }: { label: string; on: boolean }) => (
  <div className="row" style={{ gap: 12 }}><span className="grow">{label}</span><Toggle on={on} /></div>
);
const S = ({ title, children }: { title: string; children: ReactNode }) => (
  <div className="col" style={{ gap: 12 }}>
    <span className="sec">{title}</span>
    {children}
  </div>
);
const More = () => (
  <button type="button" className="row t2" style={{ gap: 8 }}><ChevronRightRegular /> More options</button>
);

const SETTINGS: Record<string, ReactNode> = {
  sync: (
    <>
      <F label="Match against" hint={<>Speech recognition needs Whisper (1.6 GB). <span className="acc">Install</span></>}><Combo value="Speech in the video" w="100%" /></F>
      <S title="Timing">
        <T label="Detect frame-rate changes" on />
        <T label="Different offset per scene" on />
        <F label="Split penalty"><div className="row" style={{ gap: 12 }}><Slider pct={31} /><span className="num" style={{ width: 16 }}>7</span></div></F>
      </S>
      <More />
    </>
  ),
  ocr: (
    <>
      <F label="Engine" hint="Apple Vision is only on macOS."><Combo value="RapidOCR" w="100%" /></F>
      <F label="Language"><Combo value="Japanese" w="100%" /></F>
      <S title="Text">
        <T label="Remove furigana" on />
        <T label="Detect italics" on />
        <T label="Fix common OCR errors" on />
      </S>
      <F label="Flag lines below"><div className="row" style={{ gap: 12 }}><Slider pct={60} /><span className="num" style={{ width: 32 }}>60%</span></div></F>
      <More />
    </>
  ),
  translate: (
    <>
      <F label="Engine"><Combo value="Claude · claude-opus-5-5" w="100%" /></F>
      <div className="row" style={{ gap: 8 }}>
        <F label="From"><Combo value="Detect" w={126} /></F>
        <F label="To"><Combo value="English" w={126} /></F>
      </div>
      <F label="Glossary"><TBox area mono value={"김신 = Kim Shin\n저승사자 = Grim Reaper"} /></F>
      <F label="Honorifics"><Seg items={["Keep", "Localize"]} value={0} /></F>
      <T label="Fit reading speed" on />
      <T label="Also write a bilingual file" on={false} />
      <More />
    </>
  ),
  fps: (
    <>
      <F label="Convert using"><Seg items={["Frame rates", "Two points"]} value={0} /></F>
      <F label="Subtitle was timed for"><Combo value="25 fps (PAL)" w="100%" /></F>
      <F label="Video plays at"><Combo value="23.976 fps (from the video)" w="100%" /></F>
      <F label="Conversion"><Seg items={["Speed change", "Keep frames"]} value={0} /></F>
      <div className="row"><span className="grow t3">Speed factor</span><span className="num">×1.042709</span></div>
      <div className="row"><span className="grow t3">Then shift by</span><TBox value="0" unit="ms" w={100} /></div>
    </>
  ),
  generate: (
    <>
      <F label="Engine" hint="Runs on the GPU."><Combo value="faster-whisper · Large v3" w="100%" /></F>
      <F label="Spoken language"><Combo value="Detect" w="100%" /></F>
      <F label="Output"><Seg items={["Transcribe", "Translate to English"]} value={0} /></F>
      <T label="Separate voices from music first" on={false} />
      <T label="Skip non-speech" on />
      <F label="Names and terms"><TBox ph="Deckard, Tyrell, Wallace" /></F>
      <More />
    </>
  ),
  style: (
    <>
      <F label="Rules"><Seg items={["Netflix", "Custom"]} value={0} /></F>
      <F label="When a rule is broken"><Seg items={["Fix", "Report only"]} value={0} /></F>
      <T label="Snap to shot changes" on />
      <T label="For deaf and hard of hearing" on={false} />
      <div className="col" style={{ gap: 6 }}>
        {[["Characters per line", "42"], ["Reading speed", "20 cps"], ["Lines", "2"], ["Shortest", "0.833 s"], ["Longest", "7 s"], ["Gap", "2 frames"]].map(([k, v]) => (
          <div key={k} className="row"><span className="grow t3">{k}</span><span className="num">{v}</span></div>
        ))}
      </div>
    </>
  ),
  hdr: (
    <>
      <F label="Set brightness by"><Seg items={["Nits", "Percentage"]} value={0} /></F>
      <div className="row"><span className="grow">Subtitle white</span><TBox value="203" unit="nits" w={110} /></div>
      <div className="row"><span className="grow">Display peak</span><TBox value="1000" unit="nits" w={110} /></div>
      <F label="Colour"><Combo value="Keep the original colours" w="100%" /></F>
      <F label="Write as"><Combo value="Same format as the input" w="100%" /></F>
    </>
  ),
  tonemap: (
    <>
      <F label="Engine"><Combo value="Automatic (libplacebo)" w="100%" /></F>
      <F label="Dolby Vision"><Seg items={["Use metadata", "Base layer"]} value={0} /></F>
      <div className="row" style={{ gap: 8 }}>
        <F label="Resolution"><Combo value="1080p" w={126} /></F>
        <F label="Curve"><Combo value="BT.2390" w={126} /></F>
      </div>
      <div className="row" style={{ gap: 8 }}>
        <F label="Encoder"><Combo value="H.264" w={126} /></F>
        <F label="Quality"><TBox value="18" unit="CRF" w={126} /></F>
      </div>
      <F label="Subtitles"><Seg items={["Copy", "None", "Burn in"]} value={0} /></F>
      <More />
    </>
  ),
  mux: (
    <>
      <F label="Task"><Seg items={["Convert", "Extract", "Mux"]} value={2} /></F>
      <F label="Container"><Seg items={["MKV", "MP4"]} value={0} /></F>
      <T label="Keep the video's own subtitles" on />
    </>
  ),
};

function SettingsBox({ tool }: { tool: string }) {
  return (
    <Box title="Settings">
      {SETTINGS[tool]}
      <div className="hr" />
      <S title="Output">
        <F label="Save to"><Combo value="Beside each source" w="100%" /></F>
        {["sync", "fps", "style", "hdr", "tonemap", "mux"].includes(tool) && (
          <div className="row"><span className="grow t3">Name suffix</span><TBox mono value={{ sync: ".synced", fps: ".retimed", style: ".styled", hdr: ".hdr", tonemap: ".sdr", mux: ".muxed" }[tool]} w={110} /></div>
        )}
      </S>
    </Box>
  );
}

const TOOL_ICON: Record<string, ReactNode> = {
  sync: <TimerRegular />,
  ocr: <ScanTextRegular />,
  translate: <TranslateRegular />,
  fps: <GaugeRegular />,
  generate: <MicRegular />,
  style: <TextEditStyleRegular />,
  hdr: <HdrRegular />,
  tonemap: <BrightnessHighRegular />,
  mux: <DocumentMultipleRegular />,
};

/** The tool strip: Subsync's tools as a toolbar, not a sidebar. */
function Tools({ on }: { on: string }) {
  const { go } = useProto();
  return (
    <>
      {SUB_TOOLS.map((t) => (
        <button key={t.id} type="button" className={cx("tool", t.id === on && "on")} onClick={() => go(`sub-${t.id}`)}>
          <span className="ic">{TOOL_ICON[t.id]}</span>
          {t.label}
        </button>
      ))}
    </>
  );
}

type Job = { f: string; meta?: string; with?: ReactNode; res?: string; s: St; t?: string; pct?: number };

const JOBS: Record<string, Job[]> = {
  sync: [
    { f: "Blade.Runner.2049.en.srt", with: "Blade.Runner.2049.2017.1080p.BluRay.mkv", s: "ready" },
    { f: "Blade.Runner.2049.en.forced.srt", with: "Blade.Runner.2049.2017.1080p.BluRay.mkv", s: "ready" },
    { f: "Blade.Runner.2049.hi.srt", with: <Combo key="c" sm ghost placeholder value="Choose a video" w={200} />, s: "warn", t: "Needs a video" },
  ],
  ocr: [{ f: "Your.Name.2016.jpn.sup", with: "PGS · 1,204 images", s: "ready" }],
  translate: [
    { f: "Goblin.S01E01.ko.srt", with: "Korean → English", s: "ready" },
    { f: "Goblin.S01E02.ko.srt", with: "Korean → English", s: "ready" },
  ],
  fps: [{ f: "Blade.Runner.2049.PAL.en.srt", with: "25 → 23.976 fps", s: "ready" }],
  generate: [{ f: "Blade.Runner.2049.2017.1080p.WEB-DL.mkv", with: <Combo key="c" sm ghost value="Audio 1 · English · E-AC3 5.1" w={240} />, s: "ready" }],
  style: [{ f: "Goblin.S01E01.en.srt", with: "1,088 cues · English", s: "ready" }],
  hdr: [{ f: "Dune.Part.Two.en.sup", with: "PGS · for a DV / HDR10 video", s: "ready" }],
  tonemap: [{ f: "Dune.Part.Two.2024.2160p.UHD.BluRay.DV.HDR10.mkv", with: "2160p · Dolby Vision 8.1 · 71.4 GB", s: "ready" }],
};

const LABEL = (id: string) => SUB_TOOLS.find((t) => t.id === id)?.label ?? id;

function MuxTable() {
  return (
    <Table cols="minmax(0,1fr) 72px 170px 64px 64px" head={["Track", "Language", "Title", "Default", "Forced"]}>
      <Tr on>
        <span className="cell"><span className="fi"><VideoClipRegular /></span><MidText text="Blade.Runner.2049.2017.1080p.BluRay.mkv" /></span>
        <span className="t3">—</span><span className="t3">Video, 2 audio, 1 subtitle</span><span /><span />
      </Tr>
      {[["Blade.Runner.2049.en.srt", "eng", "English", true, false], ["Blade.Runner.2049.en.forced.srt", "eng", "English (forced)", false, true], ["Blade.Runner.2049.hi.srt", "hin", "Hindi", false, false]].map(([f, l, t, d, fo]) => (
        <Tr key={String(f)}>
          <span className="cell" style={{ paddingLeft: 24 }}><span className="fi"><DocumentTextRegular /></span><MidText text={String(f)} /></span>
          <span className="num">{l}</span>
          <span>{t}</span>
          <Chk on={!!d} />
          <Chk on={!!fo} />
        </Tr>
      ))}
    </Table>
  );
}

type Phase = "empty" | "ready" | "run" | "done";

export function SubTool({ tool, phase = "ready" }: { tool: string; phase?: Phase }) {
  const { go } = useProto();
  const p = useProgress(58, 4200);
  const runs = phase === "run";
  const done = phase === "done";
  const label = LABEL(tool);
  const jobs = (JOBS[tool] ?? []).map((j, i): Job => {
    if (tool !== "sync") return j;
    if (runs) return i === 2 ? { ...j, s: "wait", t: "Skipped · no video" } : p > (i + 1) * 45 ? { ...j, s: "ok", t: "Done", res: "+3.250 s" } : { ...j, s: "run", pct: Math.min(99, 2 + (p - i * 30) * 2) };
    if (done) return i === 2 ? { ...j, s: "warn", t: "Skipped · no video" } : { ...j, s: i === 0 ? "warn" : "ok", t: i === 0 ? "Done · 2 scenes cut" : "Done", res: "+3.250 s" };
    return j;
  });
  const ready = jobs.filter((j) => j.s === "ready").length;

  const lcd: LcdProps =
    phase === "empty"
      ? { l1: `${label}: drop subtitles and their videos` }
      : runs
        ? { icon: "run", l1: `${label}: listening for speech`, l2: "Blade.Runner.2049.en.srt", pct: p, time: `${Math.round(p)}%` }
        : done
          ? { icon: "warn", l1: "2 synced · 1 skipped", l2: "Blade.Runner.2049.en.srt: two scenes were cut" }
          : tool === "sync"
            ? { icon: "warn", l1: "2 of 3 ready", l2: "Blade.Runner.2049.hi.srt needs a video" }
            : { l1: `${label} · ${tool === "mux" ? "1 video, 3 subtitles" : `${jobs.length} ${jobs.length === 1 ? "file" : "files"}`}`, l2: "Ready" };

  return (
    <Window
      page="subs"
      lcd={lcd}
      strip={<Tools on={tool} />}
      cols={phase === "empty" ? "minmax(0,1fr)" : "minmax(0,1fr) 300px"}
      dock={done && tool === "sync" ? <CuesDock /> : undefined}
      tools={
        <>
          <Cmd icon={<AddRegular />} disabled={runs}>Add files</Cmd>
          <Cmd icon={<FolderAddRegular />} disabled={runs}>Add folder</Cmd>
          {phase !== "empty" && <Cmd icon={<DeleteRegular />} disabled={runs} title="Remove" />}
        </>
      }
      primary={
        runs ? (
          <Btn icon={<StopRegular />} onClick={() => go("sub-sync")}>Stop</Btn>
        ) : (
          <Btn accent icon={<PlayRegular />} disabled={phase === "empty"} onClick={() => go(tool === "sync" ? "sub-sync-run" : tool === "ocr" ? "sub-ocr-done" : `sub-${tool}`)}>
            {tool === "mux" ? "Mux" : done ? "Run again" : ready > 1 ? `Run ${ready}` : "Run"}
          </Btn>
        )
      }
    >
      {phase === "empty" ? (
        <section className="box">
          <Empty icon={<DocumentTextRegular />} title="Drop subtitles and their videos">
            <Btn icon={<AddRegular />} onClick={() => go("sub-sync")}>Add files</Btn>
          </Empty>
        </section>
      ) : (
        <>
          <Box body={false} title={tool === "mux" ? "Tracks" : "Files"} sub={tool === "mux" ? "into one MKV" : `${jobs.length}`}>
            {tool === "mux" ? (
              <MuxTable />
            ) : (
              <Table cols="minmax(0,1fr) 90px 180px" head={["File", " Result", "Status"]}>
                {jobs.map((j, i) => (
                  <Tr key={j.f} on={i === 0} style={{ height: 50 }}>
                    <span className="col" style={{ minWidth: 0, gap: 1 }}>
                      <span className="cell"><span className="fi">{tool === "tonemap" || tool === "generate" ? <VideoClipRegular /> : <DocumentTextRegular />}</span><MidText text={j.f} tail={16} /></span>
                      <span className="cell t3 sm" style={{ paddingLeft: 24 }}>{typeof j.with === "string" ? <MidText text={j.with} tail={22} /> : j.with}</span>
                    </span>
                    <span className="r num" style={{ display: "flex" }}>{j.res ?? <span className="t3">—</span>}</span>
                    <Status s={j.s} text={j.t} pct={j.pct} />
                  </Tr>
                ))}
              </Table>
            )}
          </Box>
          <SettingsBox tool={tool} />
        </>
      )}
      {runs && <AutoGo to="sub-sync-done" ms={4400} />}
    </Window>
  );
}

function CuesDock() {
  return (
    <div style={{ height: 260, display: "flex", flexDirection: "column" }}>
      <DockTabs tabs={["Cues", "Output", "History"]} on="Cues" tools={<><span className="t2 sm" style={{ marginRight: 8 }}>Only lines to check (1)</span><Toggle on={false} /></>} />
      <Table cols="44px 96px 96px minmax(0,1fr) 48px" head={["#", "In", "Out", "Text", " CPS"]}>
        {CUES.map((c) => (
          <Tr key={c.n} style={{ height: 30 }}>
            <span className="t3 num">{c.n}</span>
            <span className="num">{c.a.slice(3)}</span>
            <span className="num t2">{c.b.slice(3)}</span>
            <span className="cell">{c.warn && <WarningFilled className="warn" style={{ fontSize: 14 }} />}<span className="truncate">{c.t}</span></span>
            <span className={cx("r num", c.warn ? "warn" : "t3")} style={{ display: "flex" }}>{c.cps}</span>
          </Tr>
        ))}
      </Table>
    </div>
  );
}

/** OCR finished: every line with how sure the engine was; the right panel edits one. */
export function OcrReview() {
  return (
    <Window
      page="subs"
      strip={<Tools on="ocr" />}
      lcd={{ icon: "warn", l1: "1,204 lines read · 17 to check", l2: "Your.Name.2016.jpn.sup → Your.Name.2016.ja.srt" }}
      tools={<><Cmd icon={<AddRegular />}>Add files</Cmd><Cmd icon={<ChevronDownRegular />}>Next to check</Cmd></>}
      primary={<Btn accent icon={<CheckmarkRegular />}>Save</Btn>}
    >
      <Box body={false} title="Lines" sub="Your.Name.2016.jpn.sup">
        <Table cols="44px 96px minmax(0,1fr) 64px" head={["#", "In", "Text", " Sure"]}>
          {OCR.map((c) => (
            <Tr key={c.n} on={c.n === 90}>
              <span className="t3 num">{c.n}</span>
              <span className="num t2">{c.a.slice(3)}</span>
              <span className="cell">{c.c < 60 && <WarningFilled className="warn" style={{ fontSize: 14 }} />}<span className="truncate" style={{ fontSize: 14 }}>{c.t}</span></span>
              <span className={cx("r num", c.c < 60 ? "warn" : "t3")} style={{ display: "flex" }}>{c.c}%</span>
            </Tr>
          ))}
        </Table>
      </Box>
      <Box title="Line 90" end={<span className="warn sm" style={{ paddingRight: 6 }}>43% sure</span>}>
        <div style={{ height: 88, borderRadius: 4, display: "grid", placeItems: "center", background: "repeating-conic-gradient(#2b2b2b 0 25%, #232323 0 50%) 0 0 / 14px 14px" }}>
          <span style={{ fontSize: 24, fontWeight: 700, color: "#fff", letterSpacing: 1, textShadow: "0 0 2px #000, 1px 1px 0 #000, -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000", fontFamily: "'Yu Gothic UI','Hiragino Sans','Meiryo',sans-serif" }}>そんな場所、ないよ。</span>
        </div>
        <F label="Text"><TBox focus value="そんな場所、ないよ。" /></F>
        <div className="row" style={{ gap: 8 }}><Btn accent>Looks right</Btn><Btn>Next</Btn></div>
      </Box>
    </Window>
  );
}

