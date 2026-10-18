"""Aşama 5 — Kim Konuşuyor? Model Arena: iki modeli aynı ses üstünde yan yana yarıştır.

Akış notebook'takiyle birebir aynı (ortak.py):
1. Ses 16 kHz mono'ya çevrilir (hazirla.load16k)
2. Silero VAD konuşmayı bulur (ortak.speech_regions)
3. Konuşma 2 sn pencereyle, 0,5 sn kaydırarak iki modele aynı anda dinletilir; her pencere -20 dBFS'e eşitlenir
4. Her 0,5 sn'nin kişisi, üstüne düşen pencerelerin ortalamasıyla seçilir; aynı kişinin art arda anları birleşir

Modeller: training.ipynb'nin kaydettiği models/model_a.pt ve models/model_b_head.pt (+ donuk ECAPA).
DER'ler models/sonuclar.json'dan okunur. Notebook yeniden çalıştırılınca uygulamayı yeniden başlat.

Çalıştırma:  .venv/bin/python app.py   →  https://student.exposureai.org/lab/proxy/7860/
"""

import html
import json
import threading
from pathlib import Path

import gradio as gr
import numpy as np
import torch

import ortak
from hazirla import SR, load16k

ROOT = Path(__file__).resolve().parent
PORT = 7860
COLORS = ["#7c3aed", "#0d9488", "#f59e0b", "#e11d48", "#2563eb", "#65a30d"]
DER = {"A": None, "B": None}  # sahte sohbet DER'leri (%), models/sonuclar.json'dan


def people() -> list[str]:
    """Kişi listesi (canlı skor seçimi için); modeller yüklenmeden de bilinir."""
    try:
        return torch.load(ROOT / "models" / "model_a.pt", weights_only=False)["names"]
    except FileNotFoundError:
        return []


class Trained:
    def __init__(self, name: str, labels: list[str], predict):
        self.name, self.labels, self.predict = name, labels, predict


def load_models() -> dict:
    names, predict_a, predict_b = ortak.load_models()
    try:
        res = json.load(open(ROOT / "models" / "sonuclar.json"))
        DER.update({k: res[k]["DER"] * 100 for k in DER})
    except (FileNotFoundError, KeyError):
        pass
    return {"A": Trained("Model A — Sıfırdan CNN", names, predict_a),
            "B": Trained("Model B — ECAPA (pretrained)", names, predict_b)}


MODELS: dict = {}
READY = threading.Event()
LOAD_ERROR: list[str] = []


def _load_in_background() -> None:
    try:
        MODELS.update(load_models())
    except Exception as e:  # ör. data/ yok → arayüz hatayı gösterir
        LOAD_ERROR.append(f"Modeller yüklenemedi: {e}")
    READY.set()


# ---------- diarization (notebook'la aynı kod: ortak.py) ----------
def diarize(x: np.ndarray) -> dict | None:
    regions = ortak.speech_regions(x)
    if not regions:
        return None
    labels = MODELS["A"].labels
    out = ortak.diarize_detailed(x, {k: MODELS[k].predict for k in ("A", "B")}, len(labels), regions)
    out["duration"] = len(x) / SR
    return out


# ---------- arayüz parçaları ----------
def color(k: int) -> str:
    return COLORS[k % len(COLORS)]


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").upper()


def display(name: str) -> str:
    """Klasör adı → ekrandaki isim (Türkçe büyük harfle: elif → Elif, irem → İrem)."""
    return html.escape(tr_upper(name[:1]) + name[1:])


def fmt_t(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def timeline_html(result: dict, labels: list[str]) -> str:
    total = result["duration"]
    rows = []
    for key in ("A", "B"):
        der = f"DER %{DER[key]:.0f}" if DER[key] is not None else "DER —"
        blocks = []
        for seg in result[key]["segments"]:
            t0, t1 = seg["start"] / SR, seg["end"] / SR
            name, conf = display(labels[seg["k"]]), seg["p"][seg["k"]] * 100
            blocks.append(
                f'<div class="blk" style="left:{t0 / total * 100:.2f}%;width:{(t1 - t0) / total * 100:.2f}%;'
                f'background:{color(seg["k"])}" title="{name} %{conf:.0f} · {t0:.1f}–{t1:.1f} sn">'
                f'<b>{name}</b><span>%{conf:.0f}</span></div>')
        rows.append(f'<div class="trk-head"><span>{html.escape(MODELS[key].name)}</span><span class="der">{der}</span></div>'
                    f'<div class="trk">{"".join(blocks)}</div>')
    ticks = "".join(f'<span style="left:{i * 25}%">{fmt_t(total * i / 4)}</span>' for i in range(5))
    legend = "".join(f'<span><i style="background:{color(k)}"></i>{display(n)}</span>' for k, n in enumerate(labels))
    return (f'<div class="tl"><div class="tl-title">KAYIT · {fmt_t(total)}</div>{"".join(rows)}'
            f'<div class="axis">{ticks}</div><div class="legend">{legend}</div></div>')


def now_html(result: dict | None, labels: list[str]) -> str:
    """Son konuşma bölümü: her model kimi duydu ve kime ne kadar emin."""
    if result is None:
        return '<div class="card"><div class="card-title">ŞU AN KONUŞAN</div><div class="muted">Henüz kayıt yok</div></div>'
    parts = []
    for key in ("A", "B"):
        seg = result[key]["segments"][-1]
        bars = "".join(
            f'<div class="bar"><span>{display(n)}</span><div><i style="width:{seg["p"][k] * 100:.0f}%;'
            f'background:{color(k)}"></i></div><span>%{seg["p"][k] * 100:.0f}</span></div>'
            for k, n in enumerate(labels))
        parts.append(f'<div class="who" style="color:{color(seg["k"])}">{display(labels[seg["k"]])}</div>'
                     f'<div class="muted">Model {key} · %{seg["p"][seg["k"]] * 100:.0f} emin</div>{bars}')
    return f'<div class="card"><div class="card-title">ŞU AN KONUŞAN</div>{"<hr>".join(parts)}</div>'


def score_html(score: dict, person: str | None) -> str:
    if not person:
        return ('<div class="card"><div class="card-title">CANLI SKOR</div>'
                '<div class="muted">Yukarıdan şu an kimin konuştuğunu seç, sonra konuş.</div></div>')
    rows = []
    for key in ("A", "B"):
        ok, n = score[key]
        rows.append(f'<div class="bar"><span>Model {key}</span><div><i style="width:{ok / n * 100 if n else 0:.0f}%;'
                    f'background:var(--color-accent)"></i></div><span>{ok}/{n}</span></div>')
    return (f'<div class="card"><div class="card-title">CANLI SKOR · KONUŞAN: {html.escape(tr_upper(person))}</div>'
            f'{"".join(rows)}<div class="muted">Kaç parça doğru kişiyi buldu</div></div>')


def status_html(text: str, kind: str = "info") -> str:
    return f'<div class="status {kind}">{html.escape(text)}</div>'


# ---------- olaylar ----------
def analyze(path: str | None, person: str | None, score: dict):
    if not READY.is_set():
        yield status_html("Modeller yükleniyor..."), gr.skip(), gr.skip(), gr.skip(), score
        READY.wait()
    if LOAD_ERROR:
        yield status_html(LOAD_ERROR[0], "error"), gr.skip(), gr.skip(), gr.skip(), score
        return
    labels = MODELS["A"].labels
    if not path:
        return
    yield status_html("İki model dinliyor..."), gr.skip(), gr.skip(), gr.skip(), score
    try:
        x = load16k(Path(path))
    except Exception:
        yield status_html("Ses dosyası okunamadı. Başka bir dosya dene.", "error"), "", now_html(None, labels), gr.skip(), score
        return
    result = diarize(x)
    if result is None:
        yield (status_html("Konuşma bulunamadı, biraz daha yüksek sesle tekrar dene", "warn"), "",
               now_html(None, labels), gr.skip(), score)
        return
    if person:
        target = labels.index(person)
        score = {k: [score[k][0] + int((result[k]["window_pred"] == target).sum()),
                     score[k][1] + len(result[k]["window_pred"])] for k in ("A", "B")}
    n = len(result["A"]["window_pred"])
    yield (status_html(f"Hazır · {n} pencere dinlendi", "ok"), timeline_html(result, labels),
           now_html(result, labels), score_html(score, person), score)


def change_person(person: str | None):
    score = {"A": [0, 0], "B": [0, 0]}
    return score, score_html(score, person)


def on_load():
    if not READY.is_set():
        yield status_html("Modeller yükleniyor...")
        READY.wait()
    if LOAD_ERROR:
        yield status_html(LOAD_ERROR[0], "error")
        return
    yield status_html("Modeller hazır. Mikrofona konuş ya da bir kayıt yükle.", "ok")


# Gradio mikrofonu tarayıcı varsayılanlarıyla açar (gürültü azaltma, yankı engelleme, otomatik seviye AÇIK).
# Eğitim kayıtlarında bunlar yoktu → getUserMedia'yı sarıp üçünü kapatıyoruz. İzin reddedilirse uyarı gösteriyoruz.
HEAD = """
<script>
(() => {
  const md = navigator.mediaDevices;
  if (!md || !md.getUserMedia) return;
  const orig = md.getUserMedia.bind(md);
  md.getUserMedia = async (c) => {
    if (c && c.audio) {
      const a = typeof c.audio === "object" ? c.audio : {};
      c = {...c, audio: {...a, echoCancellation: false, noiseSuppression: false, autoGainControl: false}};
    }
    try {
      const s = await orig(c);
      document.getElementById("mic-error")?.remove();
      return s;
    } catch (e) {
      if (!document.getElementById("mic-error")) {
        const d = document.createElement("div");
        d.id = "mic-error";
        d.textContent = "Mikrofona erişilemedi. Lütfen izin ver.";
        document.body.prepend(d);
      }
      throw e;
    }
  };
})();
</script>
"""

CSS = """
#mic-error { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; padding: 10px 16px; text-align: center;
  background: #e11d48; color: #fff; font-weight: 600; }
.status { padding: 8px 12px; border-radius: 8px; font-size: 14px; background: var(--background-fill-secondary); }
.status.ok { color: #15803d; } .status.warn { color: #b45309; } .status.error { color: #e11d48; font-weight: 600; }
.tl { padding: 4px 2px; }
.tl-title, .card-title { font-size: 11px; letter-spacing: .08em; color: var(--body-text-color-subdued); font-weight: 600; margin-bottom: 8px; }
.trk-head { display: flex; justify-content: space-between; font-weight: 600; font-size: 14px; margin: 12px 0 4px; }
.trk-head .der { color: var(--body-text-color-subdued); font-weight: 400; }
.trk { position: relative; height: 44px; border-radius: 6px; background: var(--background-fill-secondary); }
.blk { position: absolute; top: 3px; bottom: 3px; border-radius: 5px; color: #fff; overflow: hidden;
  font-size: 11px; line-height: 1.2; padding: 4px 5px; box-sizing: border-box; white-space: nowrap; }
.blk b { display: block; font-size: 12px; } .blk span { opacity: .9; }
.axis { position: relative; height: 18px; margin-top: 6px; font-size: 11px; color: var(--body-text-color-subdued); }
.axis span { position: absolute; transform: translateX(-50%); }
.axis span:first-child { transform: none; } .axis span:last-child { transform: translateX(-100%); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; margin-top: 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
.card { padding: 4px 2px; }
.card hr { border: none; border-top: 1px solid var(--border-color-primary); margin: 12px 0; }
.who { font-size: 30px; font-weight: 700; line-height: 1.1; }
.muted { font-size: 12px; color: var(--body-text-color-subdued); margin: 2px 0 8px; }
.bar { display: grid; grid-template-columns: 64px 1fr 40px; align-items: center; gap: 8px; font-size: 12px; margin: 4px 0; }
.bar div { height: 6px; border-radius: 3px; background: var(--background-fill-secondary); overflow: hidden; }
.bar i { display: block; height: 100%; } .bar span:last-child { text-align: right; }
"""

with gr.Blocks(title="Kim Konuşuyor? — Model Arena") as demo:
    gr.Markdown("# Kim Konuşuyor? — Model Arena\nKonuş ya da bir kayıt yükle, iki model kimin konuştuğunu tahmin etsin.")
    status = gr.HTML(status_html("Modeller yükleniyor..."))
    score_state = gr.State({"A": [0, 0], "B": [0, 0]})
    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Mikrofon ya da ses dosyası")
            timeline = gr.HTML()
        with gr.Column(scale=1, min_width=260):
            person = gr.Dropdown(choices=[("Seçilmedi", "")] + [(tr_upper(n[:1]) + n[1:], n) for n in people()], value="", label="Şu an kim konuşuyor? (canlı skor için)")
            now = gr.HTML(now_html(None, []))
            score = gr.HTML(score_html({}, None))
            reset = gr.Button("Skoru sıfırla", size="sm")

    outputs = [status, timeline, now, score, score_state]
    audio.stop_recording(analyze, [audio, person, score_state], outputs)
    audio.upload(analyze, [audio, person, score_state], outputs)
    person.change(change_person, person, [score_state, score])
    reset.click(change_person, person, [score_state, score])
    demo.load(on_load, None, status)


if __name__ == "__main__":
    threading.Thread(target=_load_in_background, daemon=True).start()
    print(f"Model Arena: https://student.exposureai.org/lab/proxy/{PORT}/")
    demo.launch(server_name="127.0.0.1", server_port=PORT, root_path=f"/lab/proxy/{PORT}",
                head=HEAD, css=CSS, theme=gr.themes.Soft())
