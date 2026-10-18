// Tarayıcıda çalışan kütüphanelerin dosyalarını public/ altına kopyalar (her build'den önce).
// Böylece ONNX Runtime'ın wasm'ı ve VAD modeli kendi sitemizden yüklenir, dışarıdaki bir CDN'e bağımlı kalmayız.
import { cpSync, mkdirSync, readdirSync } from "node:fs";

const copy = (from, to, keep) => {
  mkdirSync(to, { recursive: true });
  const files = readdirSync(from).filter(keep);
  for (const f of files) cpSync(`${from}/${f}`, `${to}/${f}`);
  return files.length;
};

const ort = copy("node_modules/onnxruntime-web/dist", "public/ort",
  (f) => f.startsWith("ort-wasm-simd-threaded") && (f.endsWith(".wasm") || f.endsWith(".mjs")));
const vad = copy("node_modules/@ricky0123/vad-web/dist", "public/vad",
  (f) => f === "vad.worklet.bundle.min.js" || f === "silero_vad_v6.onnx");
console.log(`public/ort: ${ort} dosya, public/vad: ${vad} dosya kopyalandı`);
