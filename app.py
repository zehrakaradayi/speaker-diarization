"""Aşama 5 — Kim Konuşuyor? Model Arena: iki modeli aynı ses üstünde yan yana yarıştır.

Akış notebook'takiyle birebir aynı (ortak.py):
1. Ses 16 kHz mono'ya çevrilir (hazirla.load16k)
2. Silero VAD konuşmayı bulur (ortak.speech_regions)
3. Konuşma 2 sn pencereyle, 0,5 sn kaydırarak iki modele aynı anda dinletilir; her pencere -20 dBFS'e eşitlenir
4. Her 0,5 sn'nin kişisi, üstüne düşen pencerelerin ortalamasıyla seçilir; aynı kişinin art arda anları birleşir

Modeller: training.ipynb'nin kaydettiği models/model_a.pt ve models/model_b_head.pt (+ donuk ECAPA).
DER'ler models/sonuclar.json'dan okunur. Notebook yeniden çalıştırılınca uygulamayı yeniden başlat.
Görünüş (tema, CSS, HTML parçaları, yerleşim) arayuz.py'de; burada sadece mantık var.

Çalıştırma:  .venv/bin/python app.py   →  https://student.exposureai.org/lab/proxy/7860/
"""

import threading
from pathlib import Path

import gradio as gr
import numpy as np
import torch

import ortak
from arayuz import CSS, THEME, build, empty_timeline_html, now_html, score_html, status_html, timeline_html
from hazirla import SR, load16k

ROOT = Path(__file__).resolve().parent
PORT = 7860


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
    return {"A": Trained("Model A — Sıfırdan CNN", names, predict_a),
            "B": Trained("Model B — ECAPA (pretrained)", names, predict_b)}


MODELS: dict = {}
READY = threading.Event()
LOAD_ERROR: list[str] = []


def _load_in_background() -> None:
    try:
        MODELS.update(load_models())
    except Exception as e:  # ör. models/ yok → arayüz hatayı gösterir
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
        yield (status_html("Ses dosyası okunamadı. Başka bir dosya dene.", "error"), empty_timeline_html(),
               now_html(None, labels), gr.skip(), score)
        return
    result = diarize(x)
    if result is None:
        yield (status_html("Konuşma bulunamadı, biraz daha yüksek sesle tekrar dene", "warn"), empty_timeline_html(),
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

demo = build(analyze, change_person, on_load, people())


if __name__ == "__main__":
    threading.Thread(target=_load_in_background, daemon=True).start()
    print(f"Model Arena: https://student.exposureai.org/lab/proxy/{PORT}/")
    demo.launch(server_name="127.0.0.1", server_port=PORT, root_path=f"/lab/proxy/{PORT}",
                head=HEAD, css=CSS, theme=THEME)
