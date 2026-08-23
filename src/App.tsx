import { Toaster } from "sonner";

import { ThemeProvider } from "@/components/ThemeProvider";
import Index from "@/pages/Index";

/** Single-window desktop app: no router, and one toast system.
 *  The original mounted two competing toasters and a BrowserRouter whose 404
 *  route was unreachable inside a Tauri window. */
export default function App() {
  return (
    <ThemeProvider defaultTheme="dark" storageKey="audiosync.theme">
      <Index />
      <Toaster
        position="bottom-right"
        closeButton
        richColors
        toastOptions={{ duration: 4000 }}
      />
    </ThemeProvider>
  );
}
