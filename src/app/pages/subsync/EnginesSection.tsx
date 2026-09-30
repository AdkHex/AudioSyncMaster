/** Preferences › Subtitles: the optional engines Subsync downloads on demand,
 *  and the keys for online services. Laid out like the rest of Preferences
 *  (design/mockup/screens/app.tsx, prefs-subs): a heading, then a group of
 *  rows, each with its name, one line about it, and its control. */

import { ArrowDownloadRegular } from "@fluentui/react-icons";
import { useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { toast } from "sonner";

import { ShellContext } from "@/app/shell";
import * as subsApi from "@/lib/subsync/api";
import type { PackProgress } from "@/lib/subsync/reducer";
import type { Capabilities, PackStatus, SecretName } from "@/lib/subsync/types";
import { Btn, PBar, TBox, cx } from "@/ui/kit";

import { packBusy, runPack, saveSecret, setFocus, useEngines } from "./engines";
import { formatBytes } from "./settings/logic";

/** In the mockup's order; only Anthropic, used by two tools, says what for
 *  under its name. The others say it as their tooltip. */
const SECRETS: { name: SecretName; label: string; hint: string; shown?: boolean }[] = [
  { name: "anthropic", label: "Anthropic", hint: "Claude translation and OCR", shown: true },
  { name: "deepl", label: "DeepL", hint: "DeepL translation (Free or Pro key)" },
  { name: "openai", label: "OpenAI", hint: "OpenAI-compatible translation" },
  { name: "google", label: "Google", hint: "Google Cloud Translation" },
];

const PLATFORM: Record<Capabilities["platform"], string> = { macos: "macOS", windows: "Windows", linux: "Linux" };

const Row = ({
  h,
  d,
  children,
  sub,
  anchor,
  title,
}: {
  h: ReactNode;
  d?: ReactNode;
  children?: ReactNode;
  sub?: boolean;
  anchor?: string;
  title?: string;
}) => (
  <div className={cx("srow", sub && "sub")} data-anchor={anchor} title={title}>
    <div className="tx">
      <div>{h}</div>
      {d && <div className="d">{d}</div>}
    </div>
    {children}
  </div>
);

export function EnginesSection() {
  const shell = useContext(ShellContext);
  const { caps, packs, secrets, focus } = useEngines();
  const ref = useRef<HTMLDivElement>(null);
  const desktop = subsApi.isDesktop();

  // Opened from an Install / Add API key link: show that row.
  useEffect(() => {
    if (!focus) return;
    const frame = window.requestAnimationFrame(() => {
      const target = ref.current?.querySelector<HTMLElement>(`[data-anchor="${CSS.escape(focus)}"]`);
      if (!target) return;
      target.scrollIntoView({ block: "center" });
      target.querySelector<HTMLElement>("button:not(:disabled), input")?.focus();
      setFocus(null);
    });
    return () => window.cancelAnimationFrame(frame);
  }, [focus, caps]);

  // The engine runs one command at a time: packs wait for any run.
  const engineBusy = shell?.enginePage != null;
  const locked = engineBusy || packBusy(packs);

  const onPack = async (action: "install" | "remove", pack: PackStatus, model?: string) => {
    if (shell && !shell.claimEngine("subsync")) {
      toast.error("Another page is running. Install engines once it finishes.");
      return;
    }
    try {
      const error = await runPack(action, pack.id, model);
      if (error && error !== "cancelled") toast.error(error);
    } finally {
      shell?.releaseEngine("subsync");
    }
  };

  if (!desktop) {
    return (
      <>
        <div className="group-h">Engines</div>
        <div className="group">
          <Row h="Subtitle engines" d="Installed and removed from the desktop app." />
        </div>
      </>
    );
  }

  return (
    <div ref={ref}>
      <div className="group-h">Engines</div>
      <div className="group">
        {!caps ? (
          <Row h="Reading what is installed…" />
        ) : caps.packs.length === 0 ? (
          <Row h="No engines to install" d="Everything this system can run is built in." />
        ) : (
          // A pack this platform can never run is not offered at all.
          caps.packs.filter((pack) => pack.supported || pack.installed).map((pack) => (
            <PackRows
              key={pack.id}
              pack={pack}
              progress={packs[pack.id]}
              platform={PLATFORM[caps.platform]}
              locked={locked}
              onPack={(action, model) => void onPack(action, pack, model)}
            />
          ))
        )}
      </div>

      <div className="group-h" data-anchor="keys">API keys</div>
      <div className="group">
        {SECRETS.map((secret) => (
          <KeyRow key={secret.name} {...secret} saved={secrets?.[secret.name] ?? false} />
        ))}
      </div>
    </div>
  );
}

function sizeLine(pack: PackStatus): string | null {
  if (pack.installed) return pack.sizeBytes ? formatBytes(pack.sizeBytes) : null;
  return pack.downloadBytes ? `about ${formatBytes(pack.downloadBytes)}` : null;
}

function PackRows({
  pack,
  progress,
  platform,
  locked,
  onPack,
}: {
  pack: PackStatus;
  progress: PackProgress | undefined;
  platform: string;
  locked: boolean;
  onPack: (action: "install" | "remove", model?: string) => void;
}) {
  const working = !!progress?.busy;
  const size = sizeLine(pack);
  // As the mockup: the pack's name and its size under it; the description
  // and version are the tooltip.
  const facts = size ?? "";
  const about = [pack.description, pack.installed && pack.version ? `Version ${pack.version}` : null].filter(Boolean).join(" · ");

  let d: ReactNode = facts || undefined;
  if (working) {
    const bytes =
      progress.bytes !== null
        ? progress.totalBytes
          ? `${formatBytes(progress.bytes) ?? "0 MB"} of ${formatBytes(progress.totalBytes)}`
          : formatBytes(progress.bytes)
        : null;
    d = (
      <span className="row" style={{ gap: 10, marginTop: 6 }}>
        <PBar pct={progress.percent} ind={progress.percent === null} w={200} />
        <span className="num truncate">
          {bytes ?? progress.stage ?? (progress.action === "remove" ? "Removing…" : "Starting…")}
        </span>
      </span>
    );
  } else if (progress?.error) {
    d = <span className="bad">{progress.error}</span>;
  } else if (pack.outdated) {
    d = `Out of date${facts ? ` · ${facts}` : ""}`;
  }

  return (
    <>
      <Row h={pack.label} d={d} anchor={pack.id} title={about || undefined}>
        {working ? (
          <Btn onClick={() => void subsApi.cancel()}>Stop</Btn>
        ) : !pack.supported ? (
          <span className="t3">Not available on {platform}</span>
        ) : pack.installed ? (
          <>
            {pack.outdated ? (
              <Btn icon={<ArrowDownloadRegular />} disabled={locked} onClick={() => onPack("install")}>
                Reinstall
              </Btn>
            ) : (
              <span className="t3" style={{ marginRight: 8 }}>
                {pack.external ? "Provided by the system" : "Installed"}
              </span>
            )}
            {!pack.external && (
              <Btn disabled={locked} onClick={() => onPack("remove")} aria-label={`Remove ${pack.label}`}>
                Remove
              </Btn>
            )}
          </>
        ) : (
          <Btn icon={<ArrowDownloadRegular />} disabled={locked} onClick={() => onPack("install")} aria-label={`Install ${pack.label}`}>
            Install
          </Btn>
        )}
      </Row>
      {/* Models still to download; the rest need nothing. */}
      {pack.installed &&
        !working &&
        (pack.models ?? []).filter((model) => !model.installed).map((model) => (
          <Row
            key={model.id}
            sub
            h={model.label}
            d={model.installed ? "Downloaded" : model.downloadBytes ? `about ${formatBytes(model.downloadBytes)}` : "Not downloaded"}
          >
            {!model.installed && (
              <Btn disabled={locked} onClick={() => onPack("install", model.id)} aria-label={`Download the ${model.label} model`}>
                Download
              </Btn>
            )}
          </Row>
        ))}
    </>
  );
}

/** A key goes straight to the shell's store. What is typed lives in this
 *  row only until it is saved, and is cleared right after. */
function KeyRow({ name, label, hint, shown, saved }: { name: SecretName; label: string; hint: string; shown?: boolean; saved: boolean }) {
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);

  const save = async (value: string) => {
    setSaving(true);
    try {
      await saveSecret(name, value);
      toast.success(value ? "Key saved" : "Key removed");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save the key");
    } finally {
      setDraft("");
      setSaving(false);
    }
  };

  return (
    <Row h={label} d={shown ? hint : undefined} title={shown ? undefined : hint} anchor={`key-${name}`}>
      <form
        className="row"
        style={{ gap: 8 }}
        onSubmit={(event) => {
          event.preventDefault();
          if (draft.trim()) void save(draft.trim());
        }}
      >
        <TBox
          type="password"
          autoComplete="off"
          spellCheck={false}
          label={`${label} API key`}
          w={200}
          value={draft}
          placeholder={saved ? "•••••••••••• saved" : "Paste a key"}
          onChange={setDraft}
        />
        {saved && !draft ? (
          <Btn disabled={saving} onClick={() => void save("")} aria-label={`Remove the ${label} key`}>
            Remove
          </Btn>
        ) : (
          <Btn type="submit" disabled={!draft.trim() || saving}>
            Save
          </Btn>
        )}
      </form>
    </Row>
  );
}
