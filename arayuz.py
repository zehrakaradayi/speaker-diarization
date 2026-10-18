"""Model Arena'nın görünüşü: tema, CSS ve HTML parçaları (app.py'deki mantıktan ayrı).

Konsept "dövüş gecesi": iki model iki köşede (A turuncu, B mavi), zaman çizelgeleri teyp şeritleri,
canlı skor bir skorbord. Model ne kadar eminse bölüm o kadar dolu; %60'ın altı taralı çizilir.
İki şeridin arasındaki "anlaşma şeridi" modellerin farklı kişi dediği anları kırmızıyla gösterir.

app.py mantığı (modeller, diarization, canlı skor) çalıştırır; görünüş için bu dosyadaki status_html,
timeline_html, now_html, score_html, empty_timeline_html parçalarını ve build() yerleşimini kullanır.
"""

import html
import json
from pathlib import Path

import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent
SR = 16000
STEP = SR // 2  # anlaşma şeridi 0,5 sn'lik dilimlerle (diarization ile aynı)
LOW_CONF = 0.6  # bunun altı "emin değil": taralı çizilir
COLORS = ["#a78bfa", "#2dd4bf", "#fbbf24", "#fb7185", "#60a5fa", "#a3e635"]
CORNERS = {"A": ("Sıfırdan CNN", "kendi verimizle, sıfırdan"), "B": ("ECAPA", "7.000 kişiyle önceden eğitilmiş")}


def color(k: int) -> str:
    return COLORS[k % len(COLORS)]


def tr_upper(s: str) -> str:
    return s.replace("i", "İ").upper()


def display(name: str) -> str:
    """Klasör adı → ekrandaki isim (Türkçe büyük harfle: elif → Elif, irem → İrem)."""
    return html.escape(tr_upper(name[:1]) + name[1:])


def fmt_t(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def der() -> dict:
    """Sahte sohbet DER'leri (%), notebook'un yazdığı models/sonuclar.json'dan; yoksa boş."""
    try:
        res = json.load(open(ROOT / "models" / "sonuclar.json"))
        return {k: res[k]["DER"] * 100 for k in ("A", "B")}
    except (FileNotFoundError, KeyError, ValueError):
        return {}


def der_text(key: str) -> str:
    d = der().get(key)
    return f"DER %{d:.0f}" if d is not None else "DER —"


# ---------- parçalar ----------
def hero_html() -> str:
    a, b = (f'<div class="corner c-{k.lower()}"><div class="corner-tag">{k}</div>'
            f'<div><div class="corner-name">{name}</div><div class="corner-sub">{sub}</div></div>'
            f'<div class="corner-der">{der_text(k)}</div></div>'
            for k, (name, sub) in CORNERS.items())
    return (f'<div class="hero"><div class="kicker"><span class="live-dot"></span>MODEL ARENA · CANLI</div>'
            f'<h1>Kim <em>Konuşuyor?</em></h1>'
            f'<p>Konuş ya da bir kayıt yükle. İki model aynı sesi dinlesin, kimin konuştuğunu tahmin etsin.</p>'
            f'<div class="corners">{a}<div class="vs">VS</div>{b}</div></div>')


def status_html(text: str, kind: str = "info") -> str:
    return f'<div class="status {kind}"><span class="dot"></span>{html.escape(text)}</div>'


def empty_timeline_html() -> str:
    lanes = "".join(
        f'<div class="track t-{k.lower()}"><div class="track-label"><span class="tag">{k}</span>'
        f'<div><b>{name}</b><small>{der_text(k)}</small></div></div><div class="lane empty"></div></div>'
        for k, (name, _) in CORNERS.items())
    return (f'<div class="tl"><div class="tl-top"><span class="eyebrow">ZAMAN ÇİZELGESİ</span></div>{lanes}'
            f'<div class="tl-hint">Kayıt bitince iki şerit de burada dolacak.</div></div>')


def _frame_labels(segments: list[dict], n: int) -> np.ndarray:
    """Her 0,5 sn dilimin kişisi (-1 = sessizlik)."""
    lab = np.full(n, -1)
    for s in segments:
        lab[s["start"] // STEP:-(-s["end"] // STEP)] = s["k"]
    return lab


def _agreement(result: dict) -> tuple[str, float | None]:
    n = int(-(-result["duration"] * SR // STEP))
    a, b = _frame_labels(result["A"]["segments"], n), _frame_labels(result["B"]["segments"], n)
    state = np.where((a < 0) & (b < 0), 0, np.where(a == b, 1, 2))  # 0 sessiz, 1 aynı, 2 farklı
    runs, i = [], 0
    while i < n:
        j = i
        while j < n and state[j] == state[i]:
            j += 1
        if state[i]:
            runs.append(f'<i class="{"same" if state[i] == 1 else "diff"}" '
                        f'style="left:{i / n * 100:.2f}%;width:{(j - i) / n * 100:.2f}%"></i>')
        i = j
    talk = state > 0
    return "".join(runs), (float((state[talk] == 1).mean()) if talk.any() else None)


def timeline_html(result: dict, labels: list[str]) -> str:
    total = result["duration"]
    tracks = []
    for key, (name, _) in CORNERS.items():
        blocks = []
        for seg in result[key]["segments"]:
            t0, t1 = seg["start"] / SR, seg["end"] / SR
            conf = float(seg["p"][seg["k"]])
            who = display(labels[seg["k"]])
            low = " low" if conf < LOW_CONF else ""
            blocks.append(
                f'<div class="blk{low}" style="left:{t0 / total * 100:.2f}%;width:{(t1 - t0) / total * 100:.2f}%;'
                f'--c:{color(seg["k"])};--f:{0.35 + 0.65 * conf:.2f}" title="{who} %{conf * 100:.0f} · {t0:.1f}–{t1:.1f} sn">'
                f'<b>{who}</b><span>%{conf * 100:.0f}</span></div>')
        tracks.append(f'<div class="track t-{key.lower()}"><div class="track-label"><span class="tag">{key}</span>'
                      f'<div><b>{name}</b><small>{der_text(key)}</small></div></div>'
                      f'<div class="lane">{"".join(blocks)}</div></div>')
    runs, agree = _agreement(result)
    strip = (f'<div class="track agree-row"><div class="track-label"><small>anlaşma</small></div>'
             f'<div class="agree">{runs}</div></div>')
    ticks = "".join(f'<span style="left:{i * 25}%">{fmt_t(total * i / 4)}</span>' for i in range(5))
    legend = "".join(f'<span><i style="background:{color(k)}"></i>{display(n)}</span>' for k, n in enumerate(labels))
    agree_txt = f"Aynı kişiyi duydukları süre: <b>%{agree * 100:.0f}</b>" if agree is not None else ""
    return (f'<div class="tl"><div class="tl-top"><span class="eyebrow">ZAMAN ÇİZELGESİ</span>'
            f'<span class="dur">{fmt_t(total)}</span></div>{tracks[0]}{strip}{tracks[1]}'
            f'<div class="track"><div class="track-label"></div><div class="axis">{ticks}</div></div>'
            f'<div class="tl-foot"><div class="legend">{legend}<span class="lg-low"><i></i>%60 altı</span>'
            f'<span class="lg-diff"><i></i>farklı kişi</span></div><div class="agree-txt">{agree_txt}</div></div></div>')


def now_html(result: dict | None, labels: list[str]) -> str:
    """Son konuşma bölümü: iki model kimi duydu, kime ne kadar emin."""
    if result is None:
        return ('<div class="now idle"><span class="eyebrow">ŞU AN KONUŞAN</span>'
                '<div class="eq">' + "<i></i>" * 5 + '</div><div class="now-name">—</div>'
                '<div class="now-sub">Henüz kayıt yok. Mikrofona bir şey söyle.</div></div>')
    last = {k: result[k]["segments"][-1] for k in ("A", "B")}
    ka, kb = last["A"]["k"], last["B"]["k"]
    main = kb  # büyük yazı Model B'nin (sahte sohbetlerde DER'i daha düşük) tahmini
    verdict = ('<span class="pill ok">İkisi de aynı fikirde</span>' if ka == kb else
               f'<span class="pill diff">Anlaşamadılar! A: {display(labels[ka])}</span>')
    cols = []
    for key in ("A", "B"):
        p = last[key]["p"]
        bars = "".join(
            f'<div class="pb{" top" if k == last[key]["k"] else ""}"><span>{display(n)}</span>'
            f'<div><i style="width:{p[k] * 100:.0f}%;background:{color(k)}"></i></div><em>%{p[k] * 100:.0f}</em></div>'
            for k, n in enumerate(labels))
        cols.append(f'<div class="pcol c-{key.lower()}"><div class="pcol-h"><span class="tag">{key}</span>'
                    f'{display(labels[last[key]["k"]])}</div>{bars}</div>')
    conf = last["B"]["p"][main] * 100
    return (f'<div class="now" style="--c:{color(main)}"><span class="eyebrow">ŞU AN KONUŞAN</span>'
            f'<div class="now-row"><div class="eq">' + "<i></i>" * 5 + f'</div>'
            f'<div class="now-name">{display(labels[main])}</div></div>'
            f'<div class="now-sub">Model B · %{conf:.0f} emin · {verdict}</div>'
            f'<div class="pcols">{"".join(cols)}</div></div>')


def score_html(score: dict, person: str | None) -> str:
    if not person:
        return ('<div class="board idle"><span class="eyebrow">CANLI SKOR</span>'
                '<div class="board-hint">Yukarıdan şu an kimin konuştuğunu seç, sonra konuş. '
                'Her kayıtta iki modelin kaç parçayı doğru bildiğini sayarım.</div></div>')
    pct = {k: score[k][0] / score[k][1] if score[k][1] else None for k in ("A", "B")}
    lead = None
    if pct["A"] is not None and pct["A"] != pct["B"]:
        lead = "A" if pct["A"] > pct["B"] else "B"
    sides = []
    for key in ("A", "B"):
        ok, n = score[key]
        sides.append(f'<div class="side s-{key.lower()}{" lead" if lead == key else ""}">'
                     f'<small>MODEL {key}{" · ÖNDE" if lead == key else ""}</small>'
                     f'<div class="digits">{ok}</div><small>/ {n} parça'
                     f'{f" · %{pct[key] * 100:.0f}" if pct[key] is not None else ""}</small></div>')
    return (f'<div class="board"><span class="eyebrow">CANLI SKOR</span><div class="board-row">{sides[0]}'
            f'<div class="board-mid"><small>KONUŞAN</small><b>{html.escape(tr_upper(person))}</b></div>{sides[1]}</div>'
            f'<div class="board-hint">Kaç parça doğru kişiyi buldu</div></div>')


# ---------- yerleşim ----------
def build(analyze, change_person, on_load, people: list[str]) -> gr.Blocks:
    with gr.Blocks(title="Kim Konuşuyor? — Model Arena") as demo:
        gr.HTML(hero_html())
        status = gr.HTML(status_html("Modeller yükleniyor..."))
        score_state = gr.State({"A": [0, 0], "B": [0, 0]})
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Mikrofon ya da ses dosyası",
                                 elem_classes="mic",
                                 waveform_options=gr.WaveformOptions(waveform_color="#5b5872",
                                                                     waveform_progress_color="#ff6b4a"))
                timeline = gr.HTML(empty_timeline_html())
            with gr.Column(scale=2, min_width=300):
                now = gr.HTML(now_html(None, []))
                person = gr.Dropdown(choices=[("Seçilmedi", "")] + [(tr_upper(n[:1]) + n[1:], n) for n in people],
                                     value="", label="Şu an kim konuşuyor? (canlı skor için)")
                score = gr.HTML(score_html({}, None))
                reset = gr.Button("Skoru sıfırla", size="sm", variant="secondary")
        gr.HTML('<div class="foot">Konuşmayı Silero VAD bulur · model 2 sn pencereyle, 0,5 sn kaydırarak dinler · '
                'mikrofonda gürültü azaltma, yankı engelleme ve otomatik seviye kapalı</div>')

        outputs = [status, timeline, now, score, score_state]
        audio.stop_recording(analyze, [audio, person, score_state], outputs)
        audio.upload(analyze, [audio, person, score_state], outputs)
        person.change(change_person, person, [score_state, score])
        reset.click(change_person, person, [score_state, score])
        demo.load(on_load, None, status)
    return demo


def _theme() -> gr.themes.Base:
    """Koyu sahne: açık ve koyu modda aynı renkler (tasarım tek, koyu bir arena)."""
    t = gr.themes.Base(primary_hue="orange", neutral_hue="slate", radius_size="lg",
                       font=[gr.themes.GoogleFont("Bricolage Grotesque"), "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"])
    vals = {
        "body_background_fill": "#0c0b14", "body_text_color": "#ecebf5", "body_text_color_subdued": "#8f8ca8",
        "background_fill_primary": "#14131f", "background_fill_secondary": "#1c1a2b",
        "block_background_fill": "#14131f", "block_border_color": "#2a2840", "border_color_primary": "#2a2840",
        "block_label_background_fill": "#1c1a2b", "block_label_text_color": "#b5b2cc", "block_title_text_color": "#b5b2cc",
        "input_background_fill": "#1c1a2b", "input_border_color": "#2a2840",
        "button_secondary_background_fill": "#1c1a2b", "button_secondary_background_fill_hover": "#262438",
        "button_secondary_text_color": "#ecebf5", "button_secondary_border_color": "#2a2840",
        "color_accent": "#ff6b4a", "color_accent_soft": "#3a2019",
    }
    return t.set(**vals, **{f"{k}_dark": v for k, v in vals.items() if k != "color_accent"})


THEME = _theme()

CSS = """
.gradio-container { max-width: 1240px !important; margin: 0 auto !important;
  background: radial-gradient(900px 420px at 0% -10%, rgba(255,107,74,.16), transparent 60%),
              radial-gradient(900px 420px at 100% -10%, rgba(56,189,248,.16), transparent 60%), #0c0b14 !important; }
footer { display: none !important; }
.eyebrow { font: 600 11px/1 var(--font-mono); letter-spacing: .14em; color: #8f8ca8; }

/* başlık ve köşeler */
.hero { padding: 18px 4px 6px; }
.kicker { display: inline-flex; align-items: center; gap: 8px; font: 600 11px/1 var(--font-mono); letter-spacing: .16em;
  color: #ff8e72; padding: 6px 10px; border: 1px solid #3a2a2a; border-radius: 99px; background: rgba(255,107,74,.08); }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #ff6b4a; box-shadow: 0 0 0 0 rgba(255,107,74,.6);
  animation: ping 1.6s infinite; }
@keyframes ping { 70% { box-shadow: 0 0 0 8px rgba(255,107,74,0); } 100% { box-shadow: 0 0 0 0 rgba(255,107,74,0); } }
.hero h1 { font-size: clamp(40px, 7vw, 76px); line-height: .95; letter-spacing: -.035em; margin: 14px 0 8px; font-weight: 800; }
.hero h1 em { font-style: normal; background: linear-gradient(90deg, #ff6b4a, #fbbf24 45%, #38bdf8);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #a9a6c2; font-size: 16px; margin: 0 0 18px; max-width: 560px; }
.corners { display: grid; grid-template-columns: 1fr auto 1fr; align-items: stretch; gap: 12px; }
.corner { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border-radius: 16px;
  background: #14131f; border: 1px solid #2a2840; }
.c-a.corner { border-left: 4px solid #ff6b4a; } .c-b.corner { border-right: 4px solid #38bdf8; flex-direction: row-reverse; text-align: right; }
.corner-tag, .tag { display: grid; place-items: center; width: 34px; height: 34px; border-radius: 10px; flex: none;
  font: 800 16px/1 var(--font-mono); color: #0c0b14; }
.c-a .corner-tag, .t-a .tag, .c-a .tag, .s-a .tag { background: #ff6b4a; }
.c-b .corner-tag, .t-b .tag, .c-b .tag, .s-b .tag { background: #38bdf8; }
.corner-name { font-weight: 700; font-size: 18px; } .corner-sub { color: #8f8ca8; font-size: 12px; }
.corner-der { margin-left: auto; font: 600 13px/1 var(--font-mono); color: #ecebf5; background: #1c1a2b;
  padding: 7px 9px; border-radius: 8px; white-space: nowrap; }
.c-b .corner-der { margin-left: 0; margin-right: auto; }
.vs { display: grid; place-items: center; font: 800 20px/1 var(--font-mono); color: #0c0b14; width: 52px; border-radius: 16px;
  background: linear-gradient(160deg, #ff6b4a, #38bdf8); }

/* durum */
.status { display: inline-flex; align-items: center; gap: 9px; padding: 8px 14px; border-radius: 99px; font-size: 14px;
  background: #14131f; border: 1px solid #2a2840; color: #cfcce3; }
.status .dot { width: 8px; height: 8px; border-radius: 50%; background: #8f8ca8; animation: blink 1s infinite alternate; }
.status.ok .dot { background: #4ade80; animation: none; } .status.ok { color: #bbf7d0; }
.status.warn .dot { background: #fbbf24; animation: none; } .status.warn { color: #fde68a; border-color: #4a3d1a; }
.status.error .dot { background: #fb7185; animation: none; } .status.error { color: #fecdd3; border-color: #4a1f2a; font-weight: 600; }
@keyframes blink { to { opacity: .25; } }
#mic-error { position: fixed; top: 0; left: 0; right: 0; z-index: 1000; padding: 12px 16px; text-align: center;
  background: #fb7185; color: #0c0b14; font-weight: 700; }

/* zaman çizelgesi */
.tl { padding: 16px; border-radius: 18px; background: #14131f; border: 1px solid #2a2840; }
.tl-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.tl-top .dur { font: 600 13px/1 var(--font-mono); color: #ecebf5; }
.track { display: grid; grid-template-columns: 150px 1fr; gap: 12px; align-items: center; }
.track + .track { margin-top: 6px; }
.track-label { display: flex; align-items: center; gap: 9px; min-width: 0; }
.track-label b { display: block; font-size: 14px; } .track-label small { color: #8f8ca8; font: 500 11px var(--font-mono); }
.track .tag { width: 28px; height: 28px; font-size: 13px; border-radius: 8px; }
.lane { position: relative; height: 54px; border-radius: 10px; overflow: hidden;
  background: repeating-linear-gradient(90deg, #1c1a2b 0 calc(25% - 1px), #24223a calc(25% - 1px) 25%); }
.lane.empty::after { content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,.05), transparent); animation: sweep 2.4s infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(100%); } }
.blk { position: absolute; top: 5px; bottom: 5px; border-radius: 7px; overflow: hidden; padding: 6px 7px; box-sizing: border-box;
  white-space: nowrap; color: #0c0b14; font-size: 11px; line-height: 1.15;
  background: color-mix(in srgb, var(--c) calc(var(--f) * 100%), #1c1a2b); box-shadow: inset 0 0 0 1px var(--c);
  animation: grow .45s ease-out both; transform-origin: left; }
.blk b { display: block; font-size: 12.5px; font-weight: 800; } .blk span { font-family: var(--font-mono); font-weight: 600; }
.blk.low { color: var(--c); background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c) 30%, transparent) 0 5px,
  transparent 5px 10px), #1c1a2b; }
@keyframes grow { from { transform: scaleX(0); opacity: 0; } }
.agree-row .track-label small { color: #6d6a86; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }
.agree { position: relative; height: 8px; border-radius: 4px; background: #1c1a2b; overflow: hidden; }
.agree i { position: absolute; top: 0; bottom: 0; }
.agree .same { background: #3b3854; }
.agree .diff { background: repeating-linear-gradient(135deg, #fb7185 0 4px, #9f3a4d 4px 8px); }
.axis { position: relative; height: 16px; font: 500 11px var(--font-mono); color: #8f8ca8; }
.axis span { position: absolute; transform: translateX(-50%); }
.axis span:first-child { transform: none; } .axis span:last-child { transform: translateX(-100%); }
.tl-foot { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 10px; margin-top: 12px; padding-top: 12px;
  border-top: 1px dashed #2a2840; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: #cfcce3; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }
.lg-low i { background: repeating-linear-gradient(135deg, #8f8ca8 0 2px, transparent 2px 4px); box-shadow: inset 0 0 0 1px #8f8ca8; }
.lg-diff i { background: #fb7185; }
.agree-txt { font-size: 13px; color: #a9a6c2; } .agree-txt b { color: #ecebf5; font-family: var(--font-mono); }
.tl-hint { margin-top: 12px; font-size: 13px; color: #6d6a86; }

/* şu an konuşan */
.now { padding: 18px; border-radius: 18px; background: #14131f; border: 1px solid #2a2840; position: relative; overflow: hidden; }
.now:not(.idle)::before { content: ""; position: absolute; inset: -40% -20% auto auto; width: 260px; height: 260px;
  background: radial-gradient(circle, color-mix(in srgb, var(--c) 30%, transparent), transparent 70%); }
.now-row { display: flex; align-items: center; gap: 14px; margin-top: 10px; position: relative; }
.now-name { font-size: clamp(40px, 6vw, 58px); font-weight: 800; letter-spacing: -.03em; line-height: 1; color: var(--c, #4b4864); }
.now-sub { font-size: 13px; color: #a9a6c2; margin: 8px 0 14px; position: relative; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.eq { display: flex; align-items: flex-end; gap: 4px; height: 40px; }
.eq i { width: 6px; border-radius: 3px; background: var(--c, #4b4864); animation: eq 0.9s ease-in-out infinite alternate; }
.eq i:nth-child(1) { animation-delay: -.2s } .eq i:nth-child(2) { animation-delay: -.5s } .eq i:nth-child(3) { animation-delay: -.1s }
.eq i:nth-child(4) { animation-delay: -.7s } .eq i:nth-child(5) { animation-delay: -.35s }
.now.idle .eq i { animation: none; height: 6px; }
.now.idle .eq { margin-top: 10px; height: 20px; }
@keyframes eq { from { height: 8px; } to { height: 40px; } }
.pill { padding: 3px 9px; border-radius: 99px; font-size: 12px; font-weight: 600; }
.pill.ok { background: rgba(74,222,128,.12); color: #86efac; } .pill.diff { background: rgba(251,113,133,.14); color: #fda4af; }
.pcols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; position: relative; }
.pcol-h { display: flex; align-items: center; gap: 8px; font-weight: 700; margin-bottom: 8px; font-size: 14px; }
.pcol .tag { width: 22px; height: 22px; font-size: 11px; border-radius: 6px; }
.pb { display: grid; grid-template-columns: 50px 1fr 34px; gap: 6px; align-items: center; font-size: 12px; color: #8f8ca8; margin: 5px 0; }
.pb.top { color: #ecebf5; font-weight: 600; }
.pb div { height: 6px; border-radius: 3px; background: #1c1a2b; overflow: hidden; }
.pb i { display: block; height: 100%; border-radius: 3px; transition: width .5s; }
.pb em { font: normal 500 11px var(--font-mono); text-align: right; }

/* skorbord */
.board { padding: 16px; border-radius: 18px; background: #14131f; border: 1px solid #2a2840; }
.board-row { display: grid; grid-template-columns: 1fr auto 1fr; gap: 8px; align-items: center; margin-top: 12px; }
.side { text-align: center; padding: 12px 6px; border-radius: 14px; background: #0c0b14; border: 1px solid #2a2840; transition: .3s; }
.side small { display: block; font: 600 10px/1.4 var(--font-mono); letter-spacing: .1em; color: #8f8ca8; }
.digits { font: 800 44px/1.05 var(--font-mono); margin: 4px 0; }
.s-a .digits { color: #ff6b4a; } .s-b .digits { color: #38bdf8; }
.side.lead.s-a { border-color: #ff6b4a; box-shadow: 0 0 24px -6px #ff6b4a; }
.side.lead.s-b { border-color: #38bdf8; box-shadow: 0 0 24px -6px #38bdf8; }
.board-mid { text-align: center; } .board-mid small { display: block; font: 600 10px var(--font-mono); letter-spacing: .1em; color: #6d6a86; }
.board-mid b { font-size: 16px; }
.board-hint { margin-top: 10px; font-size: 13px; color: #8f8ca8; }
.board.idle .board-hint { margin-top: 10px; }

.mic { border-radius: 18px !important; }
.foot { text-align: center; font: 500 11px/1.6 var(--font-mono); color: #6d6a86; padding: 18px 0 6px; }

@media (max-width: 720px) {
  .corners { grid-template-columns: 1fr; } .vs { width: auto; height: 34px; }
  .c-b.corner { flex-direction: row; text-align: left; border-right: 1px solid #2a2840; border-left: 4px solid #38bdf8; }
  .c-b .corner-der { margin-left: auto; margin-right: 0; }
  .track { grid-template-columns: 1fr; gap: 6px; } .agree-row .track-label { display: none; }
  .track + .track { margin-top: 10px; }
  .pcols { grid-template-columns: 1fr; }
}
"""
