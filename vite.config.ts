import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig(() => ({
  server: {
    host: "::",
    port: 8081,
    allowedHosts: [
      "trackai-app.eu.ngrok.io",
      "trackai-frontend.loca.lt",
      "fa44db5269c86bf8-185-104-115-196.serveousercontent.com",
      "localhost",
    ],
  },
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
