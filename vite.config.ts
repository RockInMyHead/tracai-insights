import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "fs";
import crypto from "crypto";

const productionGraphApiPlugin = () => ({
  name: "trackai-production-graph-api",
  configureServer(server) {
    const assets = path.resolve(__dirname, "backend/assets/floorplans");
    const metadataPath = path.join(assets, "kerama_marazzi_2025.json");

    const readJson = (filePath: string) => JSON.parse(
      fs.readFileSync(filePath, "utf8"),
    );
    const sha256 = (buffer: Buffer | string) => crypto
      .createHash("sha256")
      .update(buffer)
      .digest("hex");
    const stableStringify = (value: any): string => {
      if (Array.isArray(value)) {
        return `[${value.map(stableStringify).join(",")}]`;
      }
      if (value && typeof value === "object") {
        return `{${Object.keys(value).sort().map((key) => (
          `${JSON.stringify(key)}:${stableStringify(value[key])}`
        )).join(",")}}`;
      }
      if (typeof value === "number" && Number.isInteger(value)) {
        return `${value}.0`;
      }
      return JSON.stringify(value);
    };
    const graphGeometrySha256 = (graph: Record<string, any>) => {
      const payload = {
        schema_version: graph.schema_version,
        map_id: graph.map_id,
        nodes: (graph.nodes || []).map((node: Record<string, any>) => ({
          id: String(node.id),
          kind: String(node.kind),
          x: Number(node.x),
          y: Number(node.y),
          enabled: Boolean(node.enabled ?? true),
        })),
        edges: (graph.edges || []).map((edge: Record<string, any>) => ({
          id: String(edge.id),
          source_edge_id: String(edge.source_edge_id || edge.id),
          from: String(edge.from),
          to: String(edge.to),
          points: (edge.points || []).map((point: number[]) => [
            Number(point[0]),
            Number(point[1]),
          ]),
          bidirectional: Boolean(edge.bidirectional ?? true),
          enabled: Boolean(edge.enabled ?? true),
        })),
      };
      return sha256(stableStringify(payload));
    };
    const sendJson = (res: any, status: number, body: unknown) => {
      const encoded = Buffer.from(JSON.stringify(body));
      res.statusCode = status;
      res.setHeader("Content-Type", "application/json");
      res.setHeader("Content-Length", String(encoded.length));
      res.end(encoded);
    };
    const readBody = async (req: any) => new Promise<string>((resolve, reject) => {
      let body = "";
      req.setEncoding("utf8");
      req.on("data", (chunk: string) => {
        body += chunk;
      });
      req.on("end", () => resolve(body));
      req.on("error", reject);
    });

    server.middlewares.use(async (req, res, next) => {
      const url = req.url || "";
      const route = "/api/admin/floorplans/kerama_marazzi_2025/topology-graph/production";
      if (!url.startsWith(route)) {
        next();
        return;
      }
      try {
        const metadata = readJson(metadataPath);
        const graphName = metadata.topology_graph_file
          || "kerama_marazzi_2025.topology-graph.v1.json";
        const graphPath = path.join(assets, graphName);
        if (req.method === "GET") {
          const graph = readJson(graphPath);
          const fileSha = sha256(fs.readFileSync(graphPath));
          const geometrySha = graphGeometrySha256(graph);
          graph.production_validation = {
            file_sha256: fileSha,
            expected_file_sha256: metadata.topology_graph_sha256 || "",
            file_sha256_matches: fileSha === (metadata.topology_graph_sha256 || ""),
            geometry_sha256: geometrySha,
            embedded_geometry_sha256: graph.source?.geometry_sha256 || "",
          };
          sendJson(res, 200, graph);
          return;
        }
        if (req.method === "POST") {
          const graph = JSON.parse(await readBody(req));
          graph.production_validation = undefined;
          const geometrySha = graphGeometrySha256(graph);
          graph.source = {
            ...(graph.source || {}),
            geometry_sha256: geometrySha,
            saved_from_admin: true,
            saved_at_unix: Math.round(Date.now()) / 1000,
          };
          const encoded = Buffer.from(
            `${JSON.stringify(graph, null, 2)}\n`,
            "utf8",
          );
          const fileSha = sha256(encoded);
          fs.writeFileSync(graphPath, encoded);
          fs.writeFileSync(
            metadataPath,
            `${JSON.stringify({
              ...metadata,
              topology_graph_file: graphName,
              topology_graph_sha256: fileSha,
            }, null, 2)}\n`,
          );
          graph.production_validation = {
            file_sha256: fileSha,
            expected_file_sha256: fileSha,
            file_sha256_matches: true,
            geometry_sha256: geometrySha,
            embedded_geometry_sha256: geometrySha,
          };
          sendJson(res, 200, graph);
          return;
        }
        sendJson(res, 405, { detail: "Method not allowed" });
      } catch (error) {
        sendJson(res, 500, {
          detail: error instanceof Error ? error.message : String(error),
        });
      }
    });
  },
});

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // The web app may be opened from nested routes such as /downloads/.
  // Absolute asset URLs prevent the browser from requesting
  // /downloads/assets/* and receiving the HTML fallback instead of JS/CSS.
  base: mode === 'desktop' ? './' : '/',
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
  plugins: [react(), productionGraphApiPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
