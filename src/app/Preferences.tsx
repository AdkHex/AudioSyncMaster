/** Preferences: its own window with icon tabs (General, Analysis, Dub sync,
 *  Subtitles, Updates), as in the approved mockup. Every value applies to the
 *  next run; nothing here touches files already written. */

import {
  ArrowDownloadRegular,
  ArrowSyncRegular,
  ClosedCaptionRegular,
  DataHistogramRegular,
  HeadphonesSoundWaveRegular,
  SettingsRegular,
} from "@fluentui/react-icons";
import { useEffect, useState, type ReactNode } from "react";

import { useTheme } from "@/components/ThemeProvider";
import * as api from "@/lib/api";
import { DEFAULT_SETTINGS, formatSize, type AppSettings, type VoiceToolsStatus } from "@/lib/types";
import type { UpdateInfo } from "@/lib/updater";
import { Modal } from "@/ui/frame";
import { Btn, Combo, PBar, Slider, TBox, Toggle, cx } from "@/ui/kit";

export type PrefsTab = "general" | "analysis" | "dub" | "subs" | "updates";

const TABS: { id: PrefsTab; label: string; icon: ReactNode }[] = [
  { id: "general", label: "General", icon: <SettingsRegular /> },
  { id: "analysis", label: "Analysis", icon: <DataHistogramRegular /> },
  { id: "dub", label: "Dub sync", icon: <HeadphonesSoundWaveRegular /> },
  { id: "subs", label: "Subtitles", icon: <ClosedCaptionRegular /> },
  { id: "updates", label: "Updates", icon: <ArrowSyncRegular /> },
];

const Row = ({ h, d, children, sub }: { h: ReactNode; d?: ReactNode; children?: ReactNode; sub?: boolean }) => (
  <div className={cx("srow", sub && "sub")}>
    <div className="tx">
      <div>{h}</div>
      {d && <div className="d">{d}</div>}
    </div>
    {children}
  </div>
);

const SliderCtl = ({
  value,
  min,
  max,
  step,
  display,
  label,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  label: string;
  onChange: (value: number) => void;
}) => (
  <div className="row" style={{ gap: 12, width: 240 }}>
    <Slider value={value} min={min} max={max} step={step} onChange={onChange} label={label} />
    <span className="num" style={{ width: 36, textAlign: "right" }}>{display}</span>
  </div>
);

/** Why a series pattern cannot be used, or null when it can. */
export function patternProblem(value: string): string | null {
  if (!value.trim()) return null;
  try {
    const compiled = new RegExp(value, "i");
    // A pattern with no capture group can never produce a match key.
    const groups = new RegExp(`${compiled.source}|`).exec("")?.length ?? 1;
    return groups - 1 < 1 ? "Add a capture group, e.g. S(\\d+)E(\\d+)" : null;
  } catch (error) {
    return error instanceof Error ? error.message : "Invalid pattern";
  }
}

export interface PreferencesProps {
  tab: PrefsTab;
  onTab: (tab: PrefsTab) => void;
  settings: AppSettings;
  onChange: (settings: AppSettings) => void;
  onClose: () => void;
  version: string;
  /** A run is going: the voice tools cannot be installed or removed meanwhile. */
  busy: boolean;
  update: UpdateInfo | null;
  onInstallUpdate: () => void;
  onSkipUpdate: () => void;
  onCheckForUpdate: () => void;
  checkingUpdate: boolean;
  ffmpeg: string | null;
  /** The Subtitles tab: engines and API keys, owned by the Subsync page. */
  subtitles: ReactNode;
}

export function PreferencesWindow(props: PreferencesProps) {
  const { tab, onTab, settings, onChange, onClose } = props;
  return (
    <Modal onClose={onClose}>
      <div className="prefs" role="dialog" aria-modal="true" aria-label="Preferences">
        <div className="pt">
          Preferences
          <button type="button" className="cap x" style={{ marginLeft: "auto" }} aria-label="Close" onClick={onClose}>
            <svg width="10" height="10" aria-hidden><path d="M.5.5l9 9M9.5.5l-9 9" stroke="currentColor" /></svg>
          </button>
        </div>
        <div className="ptabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              type="button"
              role="tab"
              aria-selected={t.id === tab}
              className={cx("ptab", t.id === tab && "on")}
              onClick={() => onTab(t.id)}
            >
              <span className="ic" aria-hidden>{t.icon}</span>
              {t.label}
            </button>
          ))}
        </div>
        <div className="pbody" role="tabpanel" aria-label={TABS.find((t) => t.id === tab)?.label}>
          <Body {...props} />
        </div>
        <div className="pfoot">
          {/* Theme is owned by the ThemeProvider, so it is not reset here. */}
          <Btn onClick={() => onChange({ ...DEFAULT_SETTINGS, theme: settings.theme })}>Reset to defaults</Btn>
          <span style={{ flex: 1 }} />
          <Btn accent style={{ minWidth: 96 }} onClick={onClose}>Done</Btn>
        </div>
      </div>
    </Modal>
  );
}

function Body({
  tab,
  settings,
  onChange,
  version,
  busy,
  update,
  onInstallUpdate,
  onSkipUpdate,
  onCheckForUpdate,
  checkingUpdate,
  ffmpeg,
  subtitles,
}: PreferencesProps) {
  const set = (patch: Partial<AppSettings>) => onChange({ ...settings, ...patch });
  const { theme, setTheme } = useTheme();

  if (tab === "analysis") {
    const problem = patternProblem(settings.matchPattern);
    return (
      <>
        <div className="group-h">Measuring</div>
        <div className="group">
          <Row h="Sample windows" d="More windows catch drift better, and take longer.">
            <SliderCtl label="Sample windows" value={settings.windowCount} min={2} max={12} step={1} display={String(settings.windowCount)} onChange={(windowCount) => set({ windowCount })} />
          </Row>
          <Row h="Window length">
            <SliderCtl label="Window length" value={settings.windowSeconds} min={10} max={180} step={5} display={`${settings.windowSeconds} s`} onChange={(windowSeconds) => set({ windowSeconds })} />
          </Row>
          <Row h="Largest offset">
            <SliderCtl
              label="Largest offset"
              value={Math.round(settings.maxOffsetMs / 1000)}
              min={5}
              max={300}
              step={5}
              display={`${Math.round(settings.maxOffsetMs / 1000)} s`}
              onChange={(seconds) => set({ maxOffsetMs: seconds * 1000 })}
            />
          </Row>
        </div>
        <div className="group-h">Checks</div>
        <div className="group">
          <Row h="Cut check" d="Find where a scene was cut.">
            <Toggle name="Cut check" on={settings.cutCheck} onChange={(cutCheck) => set({ cutCheck })} />
          </Row>
          <Row h="Whole-timeline check" d="Lists every cut and gap. Slower.">
            <Toggle name="Whole-timeline check" on={settings.timelineCheck} onChange={(timelineCheck) => set({ timelineCheck })} />
          </Row>
          <Row h="Frame-rate check" d="Tries 25 ↔ 23.976 and the other standard rates.">
            <Toggle name="Frame-rate check" on={settings.rateCheck} onChange={(rateCheck) => set({ rateCheck })} />
          </Row>
          <Row h="Episode pattern" d={problem ? <span className="bad">{problem}</span> : "S01E01 and 1x01 are read automatically."}>
            <TBox
              label="Episode pattern"
              mono
              w={180}
              value={settings.matchPattern}
              placeholder="Automatic"
              spellCheck={false}
              aria-invalid={problem ? true : undefined}
              onChange={(matchPattern) => set({ matchPattern })}
            />
          </Row>
        </div>
      </>
    );
  }

  if (tab === "dub")
    return (
      <>
        <div className="group-h">Dub sync</div>
        <div className="group">
          <Row h="Replace stretches that don't match" d="Fill them from the original instead.">
            <Toggle name="Replace stretches that don't match" on={settings.dubFillUnmatched} onChange={(dubFillUnmatched) => set({ dubFillUnmatched })} />
          </Row>
          <Row h="Move voices onto the lips" d="Finds scenes where the dub's voices were cut apart from its music.">
            <Toggle name="Move voices onto the lips" on={settings.fixVoices} onChange={(fixVoices) => set({ fixVoices })} />
          </Row>
          <VoiceTools busy={busy} />
        </div>
      </>
    );

  if (tab === "subs") return <>{subtitles}</>;

  if (tab === "updates")
    return (
      <>
        <div className="group-h">Updates</div>
        <div className="group">
          {update && (
            <Row h={`Version ${update.version} is ready`} d={update.notes ? firstLine(update.notes) : `You have ${update.currentVersion}.`}>
              <Btn onClick={onSkipUpdate}>Skip</Btn>
              <Btn accent icon={<ArrowDownloadRegular />} onClick={onInstallUpdate}>Update and restart</Btn>
            </Row>
          )}
          <Row h="Check automatically" d="On launch, at most every six hours.">
            <Toggle name="Check automatically" on={settings.autoUpdateCheck !== false} onChange={(autoUpdateCheck) => set({ autoUpdateCheck })} />
          </Row>
        </div>
        <div className="group-h">About</div>
        <div className="group">
          <Row h={`AudioSyncMaster ${version}`} d={ffmpeg ? `FFmpeg ${ffmpeg}` : undefined}>
            <Btn onClick={onCheckForUpdate} disabled={checkingUpdate}>{checkingUpdate ? "Checking…" : "Check for updates"}</Btn>
          </Row>
        </div>
      </>
    );

  return (
    <>
      <div className="group-h">Appearance</div>
      <div className="group">
        <Row h="Theme">
          <Combo
            label="Theme"
            w={200}
            value={theme}
            options={[
              { value: "system", label: "Use system setting" },
              { value: "light", label: "Light" },
              { value: "dark", label: "Dark" },
            ]}
            onChange={(value) => setTheme(value as "system" | "light" | "dark")}
          />
        </Row>
      </div>
      <div className="group-h">Files</div>
      <div className="group">
        <Row h="Name suffix" d="Added to every corrected file. Sources are never changed.">
          <TBox label="Name suffix" mono w={160} value={settings.outputSuffix} spellCheck={false} onChange={(outputSuffix) => set({ outputSuffix })} />
        </Row>
        <Row h="Files at once" d="In every page.">
          <Combo
            label="Files at once"
            w={96}
            value={settings.maxWorkers}
            options={[1, 2, 3, 4, 5, 6, 7, 8].map((n) => ({ value: n, label: String(n) }))}
            onChange={(maxWorkers) => set({ maxWorkers })}
          />
        </Row>
        <Row h="Save to">
          <Combo
            label="Save to"
            w={200}
            value={settings.outputDir ?? BESIDE}
            options={[
              { value: BESIDE, label: "Beside each source" },
              ...(settings.outputDir ? [{ value: settings.outputDir, label: folderName(settings.outputDir) }] : []),
              { value: CHOOSE, label: "Choose a folder…" },
            ]}
            onChange={(value) => {
              if (value === BESIDE) set({ outputDir: null });
              else if (value === CHOOSE) void chooseFolder().then((outputDir) => outputDir && set({ outputDir }));
            }}
          />
        </Row>
      </div>
    </>
  );
}

const BESIDE = "__beside__";
const CHOOSE = "__choose__";
const folderName = (path: string) => path.replace(/[\\/]+$/, "").replace(/^.*[\\/]/, "") || path;

/** The system's folder picker; null when cancelled or outside the desktop app. */
async function chooseFolder(): Promise<string | null> {
  if (!api.isDesktop()) return null;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const picked = await invoke<string | string[] | null>("plugin:dialog|open", { options: { directory: true, multiple: false, title: "Save corrected copies to" } });
    return Array.isArray(picked) ? (picked[0] ?? null) : picked;
  } catch {
    return null;
  }
}

const firstLine = (text: string) => text.split(/\r?\n/).find((line) => line.trim())?.replace(/^[#*\-\s]+/, "") ?? "";

/** The optional voice tools (PyTorch, Demucs, Silero VAD, about 1 GB) that the
 *  voice check needs, downloaded once into the app's data folder. */
function VoiceTools({ busy }: { busy: boolean }) {
  const [status, setStatus] = useState<VoiceToolsStatus | null>(null);
  const [progress, setProgress] = useState<{ percent: number; stage: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!api.isDesktop()) return;
    let active = true;
    api
      .voiceTools("status")
      .then((result) => active && setStatus(result.status))
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  const install = async () => {
    setError(null);
    setProgress({ percent: 0, stage: "Starting…" });
    const unlisten = await api.subscribeToVoiceToolsProgress((event) => setProgress({ percent: event.percent, stage: event.stage }));
    try {
      const result = await api.voiceTools("install");
      setStatus(result.status);
      if (result.error && result.error !== "cancelled") setError(result.error);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      unlisten();
      setProgress(null);
    }
  };

  const remove = async () => {
    setError(null);
    try {
      const result = await api.voiceTools("remove");
      setStatus(result.status);
      if (result.error && result.error !== "cancelled") setError(result.error);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  if (!api.isDesktop()) return <Row sub h="Voice tools" d="Installed and removed from the desktop app." />;

  const installed = status?.installed ?? false;
  const outdated = status?.outdated ?? false;
  const where = status?.device === "cuda" || status?.device === "mps" ? "the GPU" : "the CPU";

  if (progress)
    return (
      <Row
        sub
        h="Voice tools"
        d={
          <span className="row" style={{ gap: 10, marginTop: 6 }}>
            <PBar pct={progress.percent} w={200} />
            <span className="num">{progress.stage.charAt(0).toUpperCase() + progress.stage.slice(1)}</span>
          </span>
        }
      >
        <Btn onClick={() => void api.cancelSync()}>Stop</Btn>
      </Row>
    );

  return (
    <Row
      sub
      h="Voice tools"
      d={
        error ? (
          <span className="bad">{error}</span>
        ) : outdated ? (
          "Out of date"
        ) : installed ? (
          `Installed · runs on ${where}${status?.sizeBytes ? ` · ${formatSize(status.sizeBytes)}` : ""}`
        ) : (
          "Not installed · about 1 GB"
        )
      }
    >
      {installed && <Btn onClick={() => void remove()} disabled={busy}>Remove</Btn>}
      {(!installed || outdated) && (
        <Btn onClick={() => void install()} disabled={busy} icon={<ArrowDownloadRegular />}>
          {outdated ? "Reinstall" : "Install"}
        </Btn>
      )}
    </Row>
  );
}
