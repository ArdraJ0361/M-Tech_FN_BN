import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// The dev server must stay on port 5173: the backend's CORS_ORIGINS allows exactly this origin.
export default defineConfig({
  plugins: [react()],
  server: { host: "localhost", port: 5173, strictPort: true },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
