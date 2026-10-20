"use client";

import { MicVAD } from "@ricky0123/vad-web";
import { useCallback, useEffect, useRef, useState } from "react";
import { BASE, LOW_CONF, loadSpeaker, SR, type Segment, type Speaker } from "@/lib/speaker";

const COLORS = ["#7c3aed", "#0d9488", "#d97706", "#e11d48", "#2563eb", "#65a30d"];
const UNKNOWN = "#8a8798";
// Eğitim kayıtları ham alındı: tarayıcının ses işleme özellikleri kapalı, yoksa model farklı bir ses duyar
const MIC: MediaTrackConstraints = { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false };

type Status = "loading" | "ready" | "starting" | "listening" | "error";
type Placed = Segment & { t0: number }; // t0: oturum başından itibaren saniye

const display = (n: string) => n.charAt(0).toLocaleUpperCase("tr") + n.slice(1);
const fmt = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function Diarizer() {
  const [status, setStatus] = useState<Status>("loading");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [speaking, setSpeaking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [placed, setPlaced] = useState<Placed[]>([]);
  const [elapsed, setElapsed] = useState(0);
  const [ms, setMs] = useState<number | null>(null);
  const speaker = useRef<Speaker | null>(null);
  const vad = useRef<MicVAD | null>(null);
  const startedAt = useRef(0);
  const queue = useRef(Promise.resolve());
  const heard = useRef(0);

  useEffect(() => {
    loadSpeaker(setProgress)
      .then((s) => { speaker.current = s; setStatus("ready"); })
      .catch((e) => { setError(`Model yüklenemedi: ${e.message ?? e}`); setStatus("error"); });
    return () => { vad.current?.destroy(); };
  }, []);

  useEffect(() => {
    if (status !== "listening") return;
    const id = setInterval(() => setElapsed((performance.now() - startedAt.current) / 1000), 250);
    return () => clearInterval(id);
  }, [status]);

  const onSpeechEnd = useCallback((audio: Float32Array) => {
    setSpeaking(false);
    const end = (performance.now() - startedAt.current) / 1000;
    const t0 = Math.max(0, end - audio.length / SR);
    heard.current++;
    // konuşmalar sırayla işlenir: bir pencere bilgisayarda ~0,15 sn, yavaş telefonda ~0,6 sn
    queue.current = queue.current.then(async () => {
      setBusy(true);
      const r = await speaker.current!.classify(audio);
      setMs((m) => (m === null ? r.msPerWindow : 0.7 * m + 0.3 * r.msPerWindow));
      setPlaced((p) => [...p, ...r.segments.map((s) => ({ ...s, t0: t0 + s.start }))]);
      setNotice("");
      setBusy(false);
    }).catch((e) => { setBusy(false); setError(`Tahmin yapılamadı: ${e.message ?? e}`); });
  }, []);

  async function start() {
    setError(""); setNotice(""); setStatus("starting");
    try {
      vad.current = await MicVAD.new({
        model: "v6",
        // Varsayılanlar (1,4 sn bekleme, 0,8 sn ön pay) sohbette sıraları birleştirir ve önceki konuşmacıyı da katar:
        // 0,5 sn sessizlikte konuşma biter, başa sadece 0,1 sn eklenir. Durdur'a basınca yarım kalan konuşma da işlenir.
        redemptionMs: 500,
        preSpeechPadMs: 100,
        submitUserSpeechOnPause: true,
        baseAssetPath: BASE + "/vad/",
        onnxWASMBasePath: BASE + "/ort/",
        getStream: () => navigator.mediaDevices.getUserMedia({ audio: MIC }),
        resumeStream: () => navigator.mediaDevices.getUserMedia({ audio: MIC }),
        onSpeechStart: () => setSpeaking(true),
        onVADMisfire: () => { setSpeaking(false); setNotice("Konuşma bulunamadı, biraz daha uzun ve yüksek sesle tekrar dene."); },
        onSpeechEnd,
      });
      startedAt.current = performance.now() - elapsed * 1000;
      heard.current = 0;
      setStatus("listening");
    } catch (e) {
      const denied = e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "SecurityError");
      const missing = e instanceof DOMException && e.name === "NotFoundError";
      setError(denied ? "Mikrofona erişilemedi. Lütfen izin ver."
        : missing ? "Mikrofon bulunamadı. Bir mikrofon bağlayıp tekrar dene."
        : `Mikrofon başlatılamadı: ${(e as Error).message ?? e}`);
      setStatus("ready");
    }
  }

  async function stop() {
    await vad.current?.destroy();
    vad.current = null;
    setSpeaking(false);
    setStatus("ready");
    if (heard.current === 0) setNotice("Konuşma bulunamadı, biraz daha yüksek sesle tekrar dene.");
  }

  function reset() {
    setPlaced([]); setElapsed(0); setNotice("");
  }

  const info = speaker.current?.info;
  const labels = info?.labels ?? [];
  const last = placed[placed.length - 1];
  const known = last && last.conf >= LOW_CONF;
  const span = Math.max(30, elapsed, last ? last.t0 + last.end - last.start : 0);
  const listening = status === "listening";

  return (
    <section className="app">
      {status === "loading" && (
        <div className="state">
          <div className="state-title">Model yükleniyor... %{Math.round(progress * 100)}</div>
          <div className="progress"><i style={{ width: `${progress * 100}%` }} /></div>
          <div className="muted">İlk açılışta model (22 MB) bir kez iner.</div>
        </div>
      )}
      {error && <div className="state error" role="alert">{error}</div>}

      <div className="stage">
        <button
          className={`mic${listening ? " on" : ""}${speaking ? " speaking" : ""}`}
          onClick={listening ? stop : start}
          disabled={status === "loading" || status === "starting" || status === "error"}
          aria-label={listening ? "Durdur" : "Başlat"}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            {listening ? <rect x="7" y="7" width="10" height="10" rx="2" />
              : <path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V21h2v-2.07A7 7 0 0 0 19 12h-2Z" />}
          </svg>
          <span>{status === "starting" ? "Açılıyor..." : listening ? "Durdur" : "Başlat"}</span>
        </button>
        <div className="hint">
          {listening ? (speaking ? "Dinliyorum..." : busy ? "Düşünüyorum..." : "Konuş, cümlen bitince sonucu göstereceğim.")
            : status === "ready" ? "Başlat'a bas ve konuş." : ""}
        </div>
        {notice && <div className="notice">{notice}</div>}
      </div>

      <div className="now" style={{ ["--c" as string]: last ? (known ? COLORS[last.k % COLORS.length] : UNKNOWN) : UNKNOWN }}>
        <div className="eyebrow">SON KONUŞAN</div>
        {last ? (
          <>
            <div className="who">{known ? display(labels[last.k]) : "Tanıyamadım"}</div>
            <div className="muted">
              {known ? `%${Math.round(last.conf * 100)} emin`
                : `En yakın: ${display(labels[last.k])} %${Math.round(last.conf * 100)}. Datasette olmayan biri olabilir.`}
            </div>
            <div className="bars">
              {labels.map((n, k) => (
                <div key={n} className={`bar${k === last.k ? " top" : ""}`}>
                  <span>{display(n)}</span>
                  <div><i style={{ width: `${last.probs[k] * 100}%`, background: COLORS[k % COLORS.length] }} /></div>
                  <em>%{Math.round(last.probs[k] * 100)}</em>
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="who idle">—</div>
        )}
      </div>

      <div className="timeline">
        <div className="tl-head">
          <span className="eyebrow">ZAMAN ÇİZELGESİ</span>
          <span className="tl-right">
            <span className="mono">{fmt(span)}</span>
            {placed.length > 0 && !listening && <button className="link" onClick={reset}>Temizle</button>}
          </span>
        </div>
        <div className="lane">
          {placed.map((s, i) => {
            const ok = s.conf >= LOW_CONF;
            return (
              <div key={i} className={`seg${ok ? "" : " unknown"}`}
                style={{ left: `${(s.t0 / span) * 100}%`, width: `${((s.end - s.start) / span) * 100}%`,
                  background: ok ? COLORS[s.k % COLORS.length] : undefined }}
                title={`${ok ? display(labels[s.k]) : "Tanıyamadım"} %${Math.round(s.conf * 100)} · ${s.t0.toFixed(1)} sn`}>
                <b>{ok ? display(labels[s.k]) : "?"}</b>
                <span>%{Math.round(s.conf * 100)}</span>
              </div>
            );
          })}
          {listening && <div className="cursor" style={{ left: `${(elapsed / span) * 100}%` }} />}
          {placed.length === 0 && <div className="empty">Konuşmalar bittikçe burada dolacak.</div>}
        </div>
        <div className="legend">
          {labels.map((n, k) => (
            <span key={n}><i style={{ background: COLORS[k % COLORS.length] }} />{display(n)}</span>
          ))}
          <span><i className="unk" />Tanıyamadım (%{LOW_CONF * 100} altı)</span>
        </div>
      </div>

      {info && (
        <aside className="card" id="model">
          <div className="eyebrow">MODEL</div>
          <div className="card-grid">
            <div><small>Kullanılan model</small><b>{info.model.name}</b></div>
            <div><small>DER (20 sahte sohbet)</small><b>%{(info.model.der * 100).toFixed(1)}</b></div>
            <div><small>Ağırlıklar</small><b>{info.model.precision}, 22 MB</b></div>
            <div><small>Bir pencere bu cihazda</small><b>{ms === null ? "—" : `${Math.round(ms)} ms`}</b></div>
          </div>
          <p className="muted">Ses cihazından çıkmaz: model tarayıcının içinde çalışır. Tanıdığı kişiler: {labels.map(display).join(", ")}.</p>
        </aside>
      )}
    </section>
  );
}
