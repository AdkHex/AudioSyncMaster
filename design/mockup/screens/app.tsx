import {
  ArrowDownloadRegular,
  ArrowSyncRegular,
  ClosedCaptionRegular,
  DataHistogramRegular,
  HeadphonesSoundWaveRegular,
  SettingsRegular,
} from "@fluentui/react-icons";
import type { ReactNode } from "react";

import { Btn, Combo, PBar, Slider, TBox, Toggle, cx } from "../kit";
import { AutoGo, useProgress, useProto } from "../shell";
import { Movies } from "./analyse";

const Row = ({ h, d, children, sub }: { h: ReactNode; d?: ReactNode; children?: ReactNode; sub?: boolean }) => (
  <div className={cx("srow", sub && "sub")}>
    <div className="tx">
      <div>{h}</div>
      {d && <div className="d">{d}</div>}
    </div>
    {children}
  </div>
);
const SliderCtl = ({ pct, v }: { pct: number; v: string }) => (
  <div className="row" style={{ gap: 12, width: 240 }}><Slider pct={pct} /><span className="num" style={{ width: 36, textAlign: "right" }}>{v}</span></div>
);

type PTab = "general" | "analysis" | "dub" | "subs" | "updates";
const TABS: { id: PTab; label: string; icon: ReactNode }[] = [
  { id: "general", label: "General", icon: <SettingsRegular /> },
  { id: "analysis", label: "Analysis", icon: <DataHistogramRegular /> },
  { id: "dub", label: "Dub sync", icon: <HeadphonesSoundWaveRegular /> },
  { id: "subs", label: "Subtitles", icon: <ClosedCaptionRegular /> },
  { id: "updates", label: "Updates", icon: <ArrowSyncRegular /> },
];

function PrefsBody({ tab }: { tab: PTab }) {
  const p = useProgress(42, 8000, 20);
  if (tab === "analysis")
    return (
      <>
        <div className="group-h">Measuring</div>
        <div className="group">
          <Row h="Sample windows" d="More windows catch drift better, and take longer."><SliderCtl pct={40} v="6" /></Row>
          <Row h="Window length"><SliderCtl pct={20} v="45 s" /></Row>
          <Row h="Largest offset"><SliderCtl pct={19} v="60 s" /></Row>
        </div>
        <div className="group-h">Checks</div>
        <div className="group">
          <Row h="Cut check" d="Find where a scene was cut."><Toggle on={false} /></Row>
          <Row h="Whole-timeline check" d="Lists every cut and gap. Slower."><Toggle on={false} /></Row>
          <Row h="Frame-rate check" d="Tries 25 ↔ 23.976 and the other standard rates."><Toggle on /></Row>
          <Row h="Episode pattern" d="S01E01 and 1x01 are read automatically."><TBox ph="Automatic" w={180} /></Row>
        </div>
      </>
    );
  if (tab === "dub")
    return (
      <>
        <div className="group-h">Dub sync</div>
        <div className="group">
          <Row h="Replace stretches that don't match" d="Fill them from the original instead."><Toggle on={false} /></Row>
          <Row h="Move voices onto the lips" d="Finds scenes where the dub's voices were cut apart from its music."><Toggle on /></Row>
          <Row sub h="Voice tools" d={<span className="row" style={{ gap: 10, marginTop: 6 }}><PBar pct={p} w={200} /><span className="num">{Math.round(p * 10)} MB of 1.0 GB</span></span>}><Btn>Stop</Btn></Row>
        </div>
      </>
    );
  if (tab === "subs")
    return (
      <>
        <div className="group-h">Engines</div>
        <div className="group">
          <Row h="Speech recognition" d="faster-whisper · Large v3 · 1.6 GB"><span className="t3" style={{ marginRight: 8 }}>Installed</span><Btn>Remove</Btn></Row>
          <Row h="OCR" d="RapidOCR · 180 MB"><span className="t3" style={{ marginRight: 8 }}>Installed</span><Btn>Remove</Btn></Row>
          <Row h="Tone-mapping" d="FFmpeg with libplacebo · about 140 MB"><Btn icon={<ArrowDownloadRegular />}>Install</Btn></Row>
        </div>
        <div className="group-h">API keys</div>
        <div className="group">
          <Row h="Anthropic" d="Claude translation and OCR"><TBox value="•••••••••••• saved" w={200} /><Btn>Remove</Btn></Row>
          <Row h="DeepL"><TBox ph="Paste a key" w={200} /><Btn>Save</Btn></Row>
          <Row h="OpenAI"><TBox ph="Paste a key" w={200} /><Btn>Save</Btn></Row>
        </div>
      </>
    );
  if (tab === "updates")
    return (
      <>
        <div className="group-h">Updates</div>
        <div className="group">
          <Row h="Version 2.15.0 is ready" d="Scene-by-scene lip check, faster E-AC3 reading. 48 MB."><UpdateBtn /></Row>
          <Row h="Check automatically" d="Once a day, on launch."><Toggle on /></Row>
        </div>
        <div className="group-h">About</div>
        <div className="group">
          <Row h="AudioSyncMaster 2.14.0" d="FFmpeg 6.1 · engine 2.14.0" />
        </div>
      </>
    );
  return (
    <>
      <div className="group-h">Appearance</div>
      <div className="group">
        <Row h="Theme"><Combo value="Use system setting" w={200} /></Row>
      </div>
      <div className="group-h">Files</div>
      <div className="group">
        <Row h="Name suffix" d="Added to every corrected file. Sources are never changed."><TBox mono value=".synced" w={160} /></Row>
        <Row h="Files at once" d="In every page."><Combo value="3" w={96} /></Row>
        <Row h="Save to"><Combo value="Beside each source" w={200} /></Row>
      </div>
    </>
  );
}

function UpdateBtn() {
  const { go } = useProto();
  return <Btn accent icon={<ArrowDownloadRegular />} onClick={() => go("update-install")}>Update and restart</Btn>;
}

function PrefsWindow({ tab }: { tab: PTab }) {
  const { go } = useProto();
  return (
    <div className="smoke">
      <div className="prefs">
        <div className="pt">
          Preferences
          <button type="button" className="cap x" style={{ marginLeft: "auto" }} aria-label="Close" onClick={() => go("movies-done")}>
            <svg width="10" height="10"><path d="M.5.5l9 9M9.5.5l-9 9" stroke="currentColor" /></svg>
          </button>
        </div>
        <div className="ptabs">
          {TABS.map((t) => (
            <button key={t.id} type="button" className={cx("ptab", t.id === tab && "on")} onClick={() => go(t.id === "general" ? "prefs" : `prefs-${t.id}`)}>
              <span className="ic">{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>
        <div className="pbody"><PrefsBody tab={tab} /></div>
        <div className="pfoot">
          <Btn>Reset to defaults</Btn>
          <span style={{ flex: 1 }} />
          <Btn accent style={{ minWidth: 96 }} onClick={() => go("movies-done")}>Done</Btn>
        </div>
      </div>
    </div>
  );
}

export function Prefs({ tab }: { tab: PTab }) {
  return <Movies phase="done" overlay={<PrefsWindow tab={tab} />} />;
}

function Installing() {
  const p = useProgress(26, 4000, 4);
  return (
    <div className="smoke">
      <AutoGo to="movies-done" ms={4300} />
      <div className="dialog">
        <div className="db">
          <div className="dt">Updating to 2.15.0</div>
          <div className="row"><span className="grow">{p < 85 ? "Downloading" : "Installing"}</span><span className="t3 num">{(p * 0.48).toFixed(1)} of 48 MB</span></div>
          <PBar pct={p} />
          <div className="t3">AudioSyncMaster restarts when it's done.</div>
        </div>
      </div>
    </div>
  );
}

export function UpdateInstall() {
  return <Movies phase="done" overlay={<Installing />} />;
}
