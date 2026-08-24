import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

import pkg from "./package.json";

// https://vitejs.dev/config/
export default defineConfig(() => ({
  server: {
    host: "::",
    port: 8081,
    strictPort: true,
  },
  // Injected from package.json so the version shown in Settings can never
  // drift from the one CI tags the release with.
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
