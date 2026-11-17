import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on :8000; Vite proxies /api so the front end uses relative URLs everywhere.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: { "/api": "http://localhost:8000" } },
  build: { outDir: "dist", chunkSizeWarningLimit: 1200 },
});
