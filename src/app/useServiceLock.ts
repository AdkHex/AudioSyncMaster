import { useEffect, useState } from "react";

import { OPEN, serviceLock, subscribeToServiceLock, type ServiceLock } from "@/lib/service";

/** The remote lock as the shell reports it (docs/REMOTE-STATUS.md): read
 *  once, then follow its event. Null until the first answer. */
export function useServiceLock(): [ServiceLock | null, (lock: ServiceLock) => void] {
  const [lock, setLock] = useState<ServiceLock | null>(null);
  useEffect(() => {
    let alive = true;
    let heard = false;
    let dispose: (() => void) | undefined;
    void subscribeToServiceLock((next) => {
      heard = true;
      if (alive) setLock(next);
    }).then((unlisten) => {
      if (alive) dispose = unlisten;
      else unlisten();
      // Read after listening, so a change in between is not missed; an
      // event that already arrived is newer than this answer.
      return serviceLock()
        .then((current) => alive && !heard && setLock(current))
        .catch(() => alive && !heard && setLock(OPEN));
    });
    return () => {
      alive = false;
      dispose?.();
    };
  }, []);
  return [lock, setLock];
}
