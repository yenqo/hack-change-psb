import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // В разработке запросы /api/* уходят на FastAPI
    proxy: { "/api": "http://localhost:8000" },
  },
});
