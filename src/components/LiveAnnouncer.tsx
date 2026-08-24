import { useEffect, useRef, useState } from "react";

import type { Announcement } from "@/lib/announce";

interface LiveAnnouncerProps {
  announcement: Announcement | null;
}

/** Speaks status changes to screen readers.
 *
 *  Analysis is long-running and almost entirely visual -- a progress bar and a
 *  table that fills in. Without this, a screen-reader user has no way to know a
 *  run finished, failed, or found nothing.
 *
 *  Two regions rather than one: a live region's politeness is read when the
 *  region is created, so toggling `aria-live` on a single element is unreliable
 *  across screen readers. Both stay mounted and the message goes to whichever
 *  urgency it needs.
 */
export function LiveAnnouncer({ announcement }: LiveAnnouncerProps) {
  const [polite, setPolite] = useState("");
  const [assertive, setAssertive] = useState("");
  const lastRef = useRef<Announcement | null>(null);

  useEffect(() => {
    if (!announcement || announcement === lastRef.current) return;
    lastRef.current = announcement;

    const setter = announcement.politeness === "assertive" ? setAssertive : setPolite;

    // Clear first, then set on the next frame. An identical string written
    // twice in a row is not re-announced, which would silently swallow a
    // repeated message such as two consecutive failures.
    setter("");
    const timer = window.setTimeout(() => setter(announcement.message), 60);
    return () => window.clearTimeout(timer);
  }, [announcement]);

  return (
    <>
      <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">
        {polite}
      </div>
      <div role="alert" aria-live="assertive" aria-atomic="true" className="sr-only">
        {assertive}
      </div>
    </>
  );
}
