import { createRoot } from "react-dom/client";
import App from "./App.tsx";

async function boot() {
  // Dev only: `/?demo=movies` mocks the desktop shell so the real app renders
  // with fixture data in a plain browser. Vite replaces `import.meta.env.DEV`
  // with `false` in a production build and drops this block, chunk and all.
  if (import.meta.env.DEV && new URLSearchParams(window.location.search).has("demo")) {
    const { installDemo } = await import("./dev/demo");
    installDemo(new URLSearchParams(window.location.search).get("demo") || "movies");
  }
  createRoot(document.getElementById("root")!).render(<App />);
}

void boot();
