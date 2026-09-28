import { CheckCircle2, Download, KeyRound, Trash2 } from "lucide-react";
import { useCallback, useEffect, useId, useRef, useState } from "react";
import { toast } from "sonner";

import { Dialog } from "@/components/Dialog";
import { INPUT_CLASS } from "@/components/subsync/controls";
import { Button, ProgressBar, Spinner, Tag } from "@/components/ui";
import { cx } from "@/lib/cx";
import * as subsApi from "@/lib/subsync/api";
import { formatBytes } from "@/lib/subsync/format";
import type { PackProgress, PacksState } from "@/lib/subsync/reducer";
import type { Capabilities, EngineStatus, PackStatus, SecretName } from "@/lib/subsync/types";

const SECRETS: { name: SecretName; label: string; hint: string }[] = [
  { name: "anthropic", label: "Anthropic", hint: "Claude translation and Claude vision OCR." },
  { name: "openai", label: "OpenAI", hint: "OpenAI-compatible translation." },
  { name: "deepl", label: "DeepL", hint: "DeepL translation (Free or Pro key)." },
  { name: "google", label: "Google", hint: "Google Cloud Translation." },
];

const ENGINE_GROUPS: { key: keyof Capabilities["engines"]; label: string }[] = [
  { key: "sync", label: "Sync" },
  { key: "vad", label: "Speech detection" },
  { key: "ocr", label: "OCR" },
  { key: "translate", label: "Translate" },
  { key: "generate", label: "Speech recognition" },
  { key: "tonemap", label: "Tone-mapping" },
];

const PLATFORM: Record<Capabilities["platform"], string> = { macos: "macOS", windows: "Windows", linux: "Linux" };

interface EnginesPanelProps {
  open: boolean;
  /** A pack id to scroll to, or "keys" for the API keys. */
  focus: string | null;
  caps: Capabilities | null;
  packs: PacksState;
  /** A batch is running: the engine takes one command at a time. */
  busy: boolean;
  onPack: (action: "install" | "remove", pack: string, model?: string) => void;
  onRefresh: () => void;
  onClose: () => void;
}

/** Optional engines and the keys for online services, in one sheet. */
export function EnginesPanel({ open, focus, caps, packs, busy, onPack, onRefresh, onClose }: EnginesPanelProps) {
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open || !focus) return;
    const frame = window.requestAnimationFrame(() => {
      const target = bodyRef.current?.querySelector<HTMLElement>(`[data-anchor="${CSS.escape(focus)}"]`);
      target?.scrollIntoView({ block: "center" });
      target?.querySelector<HTMLElement>("button, input")?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [open, focus]);

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title="Engines"
      description="Optional engines are downloaded on demand; keys are kept by the app, never in its settings file."
      className="max-w-2xl"
    >
      <div ref={bodyRef}>
        {!caps ? (
          <p className="flex items-center gap-2 py-6 text-[12px] text-muted-foreground">
            <Spinner className="h-3.5 w-3.5" /> Reading what is installed…
          </p>
        ) : (
          <>
            <Group title="Packs">
              {caps.packs.length === 0 && <p className="text-[12px] text-muted-foreground">No packs are offered on this system.</p>}
              <ul className="flex flex-col">
                {caps.packs.map((pack) => (
                  <PackRow
                    key={pack.id}
                    pack={pack}
                    progress={packs[pack.id]}
                    platform={PLATFORM[caps.platform]}
                    busy={busy}
                    highlighted={focus === pack.id}
                    onPack={onPack}
                  />
                ))}
              </ul>
              {busy && <p className="mt-2 text-[11px] text-muted-foreground">Packs can be changed once the current run has finished.</p>}
            </Group>

            <Group title="Engines">
              <div className="grid grid-cols-1 gap-x-6 gap-y-3 sm:grid-cols-2">
                {ENGINE_GROUPS.map(({ key, label }) => (
                  <EngineList key={key} label={label} engines={caps.engines[key] ?? []} />
                ))}
              </div>
              <p className="mt-3 font-mono text-[10.5px] text-muted-foreground">
                FFmpeg: {caps.ffmpeg.path ? `${caps.ffmpeg.version ?? "unknown version"} · ${caps.ffmpeg.path}` : "not found"}
              </p>
            </Group>
          </>
        )}

        <Group title="API keys" anchor="keys">
          <SecretsSection open={open} onSaved={onRefresh} />
        </Group>
      </div>
    </Dialog>
  );
}

function Group({ title, anchor, children }: { title: string; anchor?: string; children: React.ReactNode }) {
  return (
    <section data-anchor={anchor} className="border-b border-border py-4 last:border-0">
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.05em] text-muted-foreground">{title}</h3>
      {children}
    </section>
  );
}

function PackRow({
  pack,
  progress,
  platform,
  busy,
  highlighted,
  onPack,
}: {
  pack: PackStatus;
  progress: PackProgress | undefined;
  platform: string;
  busy: boolean;
  highlighted: boolean;
  onPack: (action: "install" | "remove", pack: string, model?: string) => void;
}) {
  const working = !!progress?.busy;
  const size = pack.installed ? pack.sizeBytes : pack.downloadBytes;
  const locked = busy || working || !pack.supported;

  return (
    <li
      data-anchor={pack.id}
      className={cx(
        "-mx-2 rounded-lg border-t border-border px-2 py-3 first:border-t-0",
        highlighted && "bg-primary/[0.05]",
        !pack.supported && "opacity-60",
      )}
    >
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <p className="flex items-baseline gap-2 text-[13px] font-medium">
            {pack.label}
            {pack.installed ? (
              <Tag tone="success">Installed{pack.version ? ` · ${pack.version}` : ""}</Tag>
            ) : !pack.supported ? (
              <Tag>Not available on {platform}</Tag>
            ) : (
              <Tag>Not installed</Tag>
            )}
          </p>
          <p className="mt-0.5 text-[11.5px] leading-relaxed text-muted-foreground">{pack.description}</p>
          {size ? (
            <p className="mt-0.5 font-mono text-[10.5px] text-muted-foreground">
              {pack.installed ? `${formatBytes(size)} on disk` : `About ${formatBytes(size)} to download`}
            </p>
          ) : null}
        </div>
        {pack.supported &&
          (pack.installed ? (
            <Button size="sm" variant="ghost" disabled={locked} onClick={() => onPack("remove", pack.id)} aria-label={`Remove ${pack.label}`}>
              <Trash2 className="h-3.5 w-3.5" aria-hidden />
              Remove
            </Button>
          ) : (
            <Button size="sm" variant="primary" disabled={locked} onClick={() => onPack("install", pack.id)} aria-label={`Install ${pack.label}`}>
              <Download className="h-3.5 w-3.5" aria-hidden />
              Install
            </Button>
          ))}
      </div>

      {working && <PackProgressLine progress={progress!} label={pack.label} />}
      {progress?.error && !working && <p className="mt-1.5 text-[11.5px] text-destructive">{progress.error}</p>}

      {pack.installed && pack.models && pack.models.length > 0 && (
        <ul className="mt-2 flex flex-col gap-1" aria-label={`${pack.label} models`}>
          {pack.models.map((model) => (
            <li key={model.id} className="flex items-center gap-2 text-[12px]">
              {model.installed ? (
                <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-success" aria-hidden />
              ) : (
                <span className="h-3.5 w-3.5 shrink-0 rounded-full border border-border-strong" aria-hidden />
              )}
              <span className="min-w-0 flex-1 truncate">{model.label}</span>
              {model.downloadBytes && !model.installed ? (
                <span className="font-mono text-[10.5px] text-muted-foreground">{formatBytes(model.downloadBytes)}</span>
              ) : null}
              {!model.installed && (
                <Button
                  size="sm"
                  className="h-6 px-2 text-[11px]"
                  disabled={locked}
                  onClick={() => onPack("install", pack.id, model.id)}
                  aria-label={`Download the ${model.label} model`}
                >
                  Download
                </Button>
              )}
            </li>
          ))}
        </ul>
      )}
    </li>
  );
}

function PackProgressLine({ progress, label }: { progress: PackProgress; label: string }) {
  const bytes =
    progress.bytes !== null
      ? progress.totalBytes
        ? `${formatBytes(progress.bytes)} of ${formatBytes(progress.totalBytes)}`
        : formatBytes(progress.bytes)
      : null;
  return (
    <div className="mt-2" aria-live="polite">
      {progress.percent !== null ? (
        <ProgressBar percent={progress.percent} label={`${label} progress`} />
      ) : (
        <Spinner className="h-3 w-3 border-[1.5px]" />
      )}
      <p className="mt-1 flex gap-2 text-[11px] text-muted-foreground">
        <span className="min-w-0 flex-1 truncate">{progress.stage ?? (progress.action === "remove" ? "Removing…" : "Starting…")}</span>
        {bytes && <span className="tabular font-mono">{bytes}</span>}
        {progress.percent !== null && <span className="tabular font-mono">{Math.round(progress.percent)}%</span>}
      </p>
    </div>
  );
}

function EngineList({ label, engines }: { label: string; engines: EngineStatus[] }) {
  if (engines.length === 0) return null;
  return (
    <div>
      <p className="mb-1 text-[11.5px] font-medium">{label}</p>
      <ul className="flex flex-col gap-0.5">
        {engines.map((engine) => (
          <li key={engine.id} className="flex items-baseline gap-2 text-[11.5px]">
            <span className={cx("h-1.5 w-1.5 shrink-0 translate-y-[-1px] rounded-full", engine.available ? "bg-success" : "bg-border-strong")} aria-hidden />
            <span className={engine.available ? "" : "text-muted-foreground"}>{engine.label}</span>
            {!engine.available && engine.reason && (
              <span className="min-w-0 truncate text-[10.5px] text-muted-foreground/80" title={engine.reason}>
                {engine.reason}
              </span>
            )}
            <span className="sr-only">{engine.available ? "available" : "not available"}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Keys go straight to the shell's store. The typed value lives in this
 *  component only until it is saved, and is cleared right after. */
function SecretsSection({ open, onSaved }: { open: boolean; onSaved: () => void }) {
  const [status, setStatus] = useState<Record<SecretName, boolean> | null>(null);
  const [drafts, setDrafts] = useState<Partial<Record<SecretName, string>>>({});
  const [saving, setSaving] = useState<SecretName | null>(null);
  const ids = useId();

  const refresh = useCallback(() => {
    subsApi
      .secretStatus()
      .then(setStatus)
      .catch(() => setStatus(null));
  }, []);

  useEffect(() => {
    if (open) refresh();
    // Nothing typed survives closing the sheet.
    else setDrafts({});
  }, [open, refresh]);

  const save = async (name: SecretName, value: string) => {
    setSaving(name);
    try {
      await subsApi.setSecret(name, value);
      toast.success(value ? "Key saved" : "Key removed");
      onSaved();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not save the key");
    } finally {
      setDrafts((current) => ({ ...current, [name]: "" }));
      setSaving(null);
      refresh();
    }
  };

  return (
    <ul className="flex flex-col gap-3">
      {SECRETS.map(({ name, label, hint }) => {
        const draft = drafts[name] ?? "";
        const saved = status?.[name] ?? false;
        const inputId = `${ids}-${name}`;
        return (
          <li key={name}>
            <div className="flex flex-wrap items-center gap-2">
              <KeyRound className="h-3.5 w-3.5 text-muted-foreground" aria-hidden />
              <label htmlFor={inputId} className="w-[84px] text-[12.5px] font-medium">
                {label}
              </label>
              <form
                className="flex min-w-[240px] flex-1 items-center gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  if (draft.trim()) void save(name, draft.trim());
                }}
              >
                <input
                  id={inputId}
                  type="password"
                  autoComplete="off"
                  spellCheck={false}
                  value={draft}
                  placeholder={saved ? "•••••••• saved" : "Paste the key"}
                  onChange={(event) => setDrafts((current) => ({ ...current, [name]: event.target.value }))}
                  className={cx(INPUT_CLASS, "min-w-0 flex-1 font-mono")}
                />
                <Button type="submit" size="sm" disabled={!draft.trim() || saving === name}>
                  Save
                </Button>
              </form>
              {saved && (
                <Button size="sm" variant="ghost" disabled={saving === name} onClick={() => void save(name, "")} aria-label={`Remove the ${label} key`}>
                  Remove
                </Button>
              )}
            </div>
            <p className="ml-[22px] mt-0.5 text-[11px] text-muted-foreground">
              {status === null ? "" : saved ? <span className="text-success">Saved. </span> : "Not set. "}
              {hint}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
