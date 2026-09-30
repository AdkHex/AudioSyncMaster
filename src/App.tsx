import { Toaster } from "sonner";

import { ThemeProvider, useTheme } from "@/components/ThemeProvider";
import Index from "@/pages/Index";
import "@/ui/ui.css";

/** Single-window desktop app: no router, and one toast system. */
export default function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="audiosync.theme">
      <Index />
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
