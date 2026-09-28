import { Pill, Spinner } from "@/components/ui";
import { formatDuration, formatSize } from "@/lib/types";
import { formatFps, hdrBadge } from "@/lib/subsync/format";
import type { PoolFile } from "@/lib/subsync/inputs";

/** "2160p", "1080p", or WxH for odd sizes. */
function resolution(width: number, height: number): string {
  const standard: Record<number, string> = { 3840: "2160p", 4096: "2160p", 2560: "1440p", 1920: "1080p", 1280: "720p", 720: "480p" };
  return standard[width] ?? `${width}×${height}`;
}

/** What a probed file is: the facts that decide which tool can use it. */
export function FileDetails({ file }: { file: PoolFile }) {
  if (file.pending) {
    return (
      <p className="flex items-center gap-1.5 text-[10.5px] text-muted-foreground">
        <Spinner className="h-2.5 w-2.5 border-[1.5px]" />
        Reading…
      </p>
    );
  }
  if (file.error) {
    return <p className="text-[11px] text-destructive">{file.error}</p>;
  }

  const facts: string[] = [];
  let badge: string | null = null;
  if (file.subtitle) {
    facts.push(file.subtitle.format.toUpperCase());
    if (file.subtitle.cues !== null) facts.push(`${file.subtitle.cues.toLocaleString()} cues`);
    if (file.subtitle.language) facts.push(file.subtitle.language);
    if (file.subtitle.encoding) facts.push(file.subtitle.encoding);
  }
  if (file.video) {
    facts.push(resolution(file.video.width, file.video.height));
    const fps = formatFps(file.video.fps);
    if (fps) facts.push(fps);
    facts.push(file.video.codec);
    badge = hdrBadge(file.video);
  }
  if (file.duration) facts.push(formatDuration(file.duration));
  if (file.kind !== "subtitle") {
    const text = file.subtitleTracks.filter((track) => track.kind === "text").length;
    const image = file.subtitleTracks.length - text;
    if (file.subtitleTracks.length === 0) facts.push("no subtitle tracks");
    else facts.push([text && `${text} text`, image && `${image} image`].filter(Boolean).join(" + ") + " subtitle tracks");
  }
  if (file.size) facts.push(formatSize(file.size));

  return (
    <p className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5 font-mono text-[10.5px] leading-snug text-muted-foreground/80">
      {badge && (
        <Pill tone={badge === "SDR" ? "neutral" : "accent"} className="px-1.5 py-0 font-sans text-[10px]">
          {badge}
        </Pill>
      )}
      <span className="min-w-0">{facts.join(" · ")}</span>
    </p>
  );
}
