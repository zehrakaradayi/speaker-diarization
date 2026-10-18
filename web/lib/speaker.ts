// Kazanan model (Model B: donuk ECAPA + sınıflandırıcı, int8) tarayıcıda: ham ses → kişi olasılıkları.
// Eğitimdeki (ortak.py) kuralların aynısı: 16 kHz, 2 sn pencere, 0,5 sn kaydırma, her pencere -20 dBFS RMS.
import * as ort from "onnxruntime-web/wasm";

export const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const SR = 16000;
export const LOW_CONF = 0.6; // bunun altında isim yerine "Tanıyamadım"

export type ModelInfo = {
  labels: string[];
  window_samples: number;
  hop_samples: number;
  files: { int8: string };
  model: { name: string; precision: string; der: number; test_acc: number };
};

export type Segment = { start: number; end: number; k: number; conf: number; probs: number[] };

export type Speaker = {
  info: ModelInfo;
  classify: (audio: Float32Array) => Promise<{ segments: Segment[]; msPerWindow: number; windows: number }>;
};

// onnxruntime-web'in wasm dosyaları public/ort/ altından (scripts/copy-assets.mjs) yüklenir
ort.env.wasm.wasmPaths = BASE + "/ort/";
ort.env.wasm.numThreads = typeof self !== "undefined" && self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

function normalize(x: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i] * x[i];
  const gain = 10 ** (-20 / 20) / (Math.sqrt(sum / x.length) + 1e-9);
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = Math.max(-1, Math.min(1, x[i] * gain));
  return out;
}

/** Bir konuşma bölümünün 2 sn pencereleri; son pencere bölümün sonuna hizalanır, kısa bölüm sessizlikle doldurulur. */
function windowsFor(audio: Float32Array, win: number, hop: number) {
  const last = Math.max(0, audio.length - win);
  const starts: number[] = [];
  for (let s = 0; s <= last; s += hop) starts.push(s);
  if (starts[starts.length - 1] !== last) starts.push(last);
  const batch = new Float32Array(starts.length * win);
  starts.forEach((s, i) => batch.set(normalize(audio.subarray(s, Math.min(s + win, audio.length))), i * win));
  return { starts, batch };
}

export async function loadSpeaker(onProgress: (p: number) => void): Promise<Speaker> {
  const info: ModelInfo = await (await fetch(`${BASE}/models/labels.json`)).json();
  const res = await fetch(`${BASE}/models/${info.files.int8}`);
  if (!res.ok || !res.body) throw new Error(`model indirilemedi (${res.status})`);
  const total = Number(res.headers.get("content-length")) || 22_000_000;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(Math.min(1, got / total));
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const K = info.labels.length;
  const win = info.window_samples, hop = info.hop_samples;

  async function classify(audio: Float32Array) {
    const { starts, batch } = windowsFor(audio, win, hop);
    const t0 = performance.now();
    const out = await session.run({ waveform: new ort.Tensor("float32", batch, [starts.length, win]) });
    const msPerWindow = (performance.now() - t0) / starts.length;
    const probs = out.probs.data as Float32Array;
    // her 0,5 sn dilimin kişisi = üstüne düşen pencerelerin olasılık ortalaması; aynı kişinin art arda dilimleri birleşir
    const segments: Segment[] = [];
    const len = audio.length;
    for (let fs = 0; fs < len; fs += hop) {
      const fe = Math.min(fs + hop, len);
      const p = new Array(K).fill(0);
      let n = 0;
      starts.forEach((s, i) => {
        if (s < fe && Math.min(s + win, len) > fs) {
          for (let k = 0; k < K; k++) p[k] += probs[i * K + k];
          n++;
        }
      });
      for (let k = 0; k < K; k++) p[k] /= n;
      const k = p.indexOf(Math.max(...p));
      const d = (fe - fs) / SR;
      const prev = segments[segments.length - 1];
      if (prev && prev.k === k) {
        const w = prev.end - prev.start;
        prev.probs = prev.probs.map((v, j) => (v * w + p[j] * d) / (w + d));
        prev.end += d;
        prev.conf = prev.probs[k];
      } else {
        segments.push({ start: fs / SR, end: fs / SR + d, k, conf: p[k], probs: p });
      }
    }
    return { segments, msPerWindow, windows: starts.length };
  }

  return { info, classify };
}
