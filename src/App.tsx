import { useEffect, useState } from "react";
import { Toaster } from "sonner";

import { ServiceLockScreen } from "@/app/ServiceLock";
import { useServiceLock } from "@/app/useServiceLock";
import { ThemeProvider, useTheme } from "@/components/ThemeProvider";
import Index from "@/pages/Index";
import "@/ui/ui.css";

/** Single-window desktop app: no router, and one toast system.
 *
 *  The pages mount only once the app has been open (docs/REMOTE-STATUS.md):
 *  a copy that starts paused or asking for the password shows just the lock
 *  screen, so nothing on the pages asks the shell for anything it would
 *  refuse. A lock that starts later covers the pages and keeps them, so the
 *  work it stopped can carry on when it lifts. */
export default function App() {
  const [lock, setLock] = useServiceLock();
  const [opened, setOpened] = useState(false);
  useEffect(() => {
    if (lock && !lock.locked) setOpened(true);
  }, [lock]);
  return (
    <ThemeProvider defaultTheme="dark" storageKey="audiosync.theme">
      {opened && <Index />}
      {lock?.locked && <ServiceLockScreen lock={lock} onLock={setLock} />}
      <Toasts />
    </ThemeProvider>
  );
}

/** Notifications as Fluent flyouts, above the page bar. */
function Toasts() {
  const { theme } = useTheme();
  return (
    <Toaster
      position="bottom-right"
      offset={64}
      closeButton
      theme={theme === "system" ? "system" : theme}
      toastOptions={{ duration: 4000, className: "asm-toast" }}
    />
  );
}
