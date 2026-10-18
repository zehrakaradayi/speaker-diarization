"""training.ipynb ve app.py'nin ortak kullandığı parçalar: model tanımları ve diarization.

Eğitimde ve canlı denemede AYNI kodun çalışması için tek yerde durur:
- normalize: her 2 sn pencere -20 dBFS RMS (hazirla.py ve web_export ile aynı kural)
- build_model_a / LogMel: Model A (sıfırdan CNN) ve girdisi
- load_ecapa / build_head_b: Model B (donuk ECAPA + sınıflandırıcı)
- diarize: Silero VAD → 2 sn pencere, 0,5 sn kaydırma → her 0,5 sn'nin kişisi → zaman çizelgesi
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core import Annotation, Segment
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parent
SR = 16000
WIN = 2 * SR  # model her seferinde tam 2 sn (32000 örnek) duyar
HOP = SR // 2  # pencere 0,5 sn kaydırılır
TARGET_DBFS = -20.0
CHANNELS = (16, 32, 64, 64, 128)  # Aşama 3'te onaylanan orta boy


def normalize(x: np.ndarray, dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return np.clip(x * (10 ** (dbfs / 20) / rms), -1, 1).astype(np.float32)


# ---------- Model A ----------
class LogMel(nn.Module):
    """Ham ses [B, 32000] → log-mel spektrogram [B, 1, 64, 201]. export_model_a ile aynı adımlar."""

    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=400, hop_length=160, n_mels=64)

    def forward(self, wav):
        return torch.log(self.mel(wav) + 1e-6).unsqueeze(1)


def build_model_a(n_classes: int, channels=CHANNELS) -> nn.Sequential:
    """İlk 5 eleman Conv 3x3 → BatchNorm → ReLU → MaxPool blokları (canlı panel bunlara bakar),
    sonra global average pooling ve tek Linear. Sınıflandırıcı ayrı bir Sequential'a SARILMAZ."""
    c = (1, *channels)
    blocks = [nn.Sequential(nn.Conv2d(c[i], c[i + 1], 3, padding=1, bias=False), nn.BatchNorm2d(c[i + 1]),
                            nn.ReLU(), nn.MaxPool2d(2)) for i in range(5)]
    return nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                         nn.Linear(channels[-1], n_classes))


# ---------- Model B ----------
def load_ecapa():
    """speechbrain 1.0.3 repo adıyla indiremiyor: önce snapshot_download, sonra o klasörden yükle."""
    from huggingface_hub import snapshot_download
    from speechbrain.inference.speaker import EncoderClassifier
    src = snapshot_download("speechbrain/spkrec-ecapa-voxceleb")
    ecapa = EncoderClassifier.from_hparams(source=src, savedir=str(ROOT / "pretrained_models" / "ecapa"),
                                           overrides={"pretrained_path": src}, run_opts={"device": "cpu"})
    ecapa.eval()
    for p in ecapa.mods.parameters():
        p.requires_grad = False  # donuk: ECAPA hiç eğitilmez
    return ecapa


class HeadB(nn.Module):
    """L2-normalize edilmiş 192 sayılık embedding → Linear → kişi skorları."""

    def __init__(self, n_classes: int, dim: int = 192):
        super().__init__()
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, emb):
        return self.linear(F.normalize(emb, dim=-1))


@torch.no_grad()
def ecapa_embed(ecapa, wavs: np.ndarray, batch: int = 256) -> np.ndarray:
    out = [ecapa.encode_batch(torch.from_numpy(wavs[i:i + batch])).squeeze(1) for i in range(0, len(wavs), batch)]
    return torch.cat(out).numpy()


# ---------- kaydetme / yükleme ----------
def load_models():
    """models/ içindeki .pt dosyalarından iki modelin tahmin fonksiyonunu kurar (app.py kullanır)."""
    ca = torch.load(ROOT / "models" / "model_a.pt", weights_only=False)
    model_a = build_model_a(len(ca["names"]), ca["channels"])
    model_a.load_state_dict(ca["state_dict"])
    model_a.eval()
    logmel = LogMel()
    cb = torch.load(ROOT / "models" / "model_b_head.pt", weights_only=False)
    head = HeadB(len(cb["names"]))
    head.load_state_dict(cb["state_dict"])
    head.eval()
    ecapa = load_ecapa()

    @torch.no_grad()
    def predict_a(x: np.ndarray) -> np.ndarray:
        return model_a(logmel(torch.from_numpy(x))).softmax(1).numpy()

    @torch.no_grad()
    def predict_b(x: np.ndarray) -> np.ndarray:
        return head(torch.from_numpy(ecapa_embed(ecapa, x))).softmax(1).numpy()

    return ca["names"], predict_a, predict_b


# ---------- diarization ----------
_VAD = None


def speech_regions(wav: np.ndarray) -> list[tuple[int, int]]:
    """Silero VAD: konuşma olan bölgeler (örnek indeksleri)."""
    global _VAD
    if _VAD is None:
        _VAD = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(wav), _VAD, sampling_rate=SR)
    return [(t["start"], t["end"]) for t in ts]


def windows_for(wav: np.ndarray, regions) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Her konuşma bölgesinde 2 sn pencereler, 0,5 sn kaydırarak. Bölge 2 sn'den kısaysa sessizlikle
    doldurulur. Son pencere bölgenin sonuna hizalanır, böylece bölgenin her anı en az bir pencerede."""
    wins, meta = [], []
    for r, (s, e) in enumerate(regions):
        last = max(s, e - WIN)
        starts = list(range(s, last + 1, HOP))
        if starts[-1] != last:
            starts.append(last)
        for st in starts:
            seg = wav[st:min(st + WIN, e)]
            w = np.zeros(WIN, np.float32)
            w[:len(seg)] = normalize(seg)
            wins.append(w)
            meta.append((r, st, min(st + WIN, e)))
    return (np.stack(wins) if wins else np.zeros((0, WIN), np.float32)), meta


def diarize_detailed(wav: np.ndarray, predicts: dict, n_classes: int, regions=None) -> dict:
    """wav: 16 kHz mono float32. predicts: {"A": fn, "B": fn}, fn(pencereler [N, 32000]) → olasılık [N, K].
    Her 0,5 sn'lik dilimin kişisi, o dilimin üstüne düşen pencerelerin olasılık ortalamasıyla seçilir; aynı kişinin
    art arda dilimleri birleşir. Her model için: segments = [{start, end (örnek), k, p (ortalama olasılıklar)}],
    window_pred = her pencerenin tahmini (Gradio'daki canlı skor bunu sayar)."""
    regions = speech_regions(wav) if regions is None else regions
    wins, meta = windows_for(wav, regions)
    out = {}
    for key, fn in predicts.items():
        probs = fn(wins) if len(wins) else np.zeros((0, n_classes))
        segs = []
        for r, (s, e) in enumerate(regions):
            idx = [i for i, m in enumerate(meta) if m[0] == r]
            for fs in range(s, e, HOP):
                fe = min(fs + HOP, e)
                cover = [i for i in idx if meta[i][1] < fe and meta[i][2] > fs]
                p = probs[cover].mean(0)
                k = int(p.argmax())
                if segs and segs[-1]["k"] == k and segs[-1]["end"] == fs:
                    segs[-1]["end"] = fe
                    segs[-1]["ps"].append((p, fe - fs))
                else:
                    segs.append({"k": k, "start": fs, "end": fe, "ps": [(p, fe - fs)]})
        for seg in segs:  # bölümün olasılığı = dilimlerin süreyle ağırlıklı ortalaması
            ps = seg.pop("ps")
            seg["p"] = sum(p * n for p, n in ps) / sum(n for _, n in ps)
        out[key] = {"segments": segs, "window_pred": probs.argmax(1)}
    return out


def diarize(wav: np.ndarray, predicts: dict, names: list[str], regions=None) -> dict:
    """Her model için: [(başlangıç sn, bitiş sn, isim, güven), ...] (notebook'taki DER ve çizelgeler için)."""
    d = diarize_detailed(wav, predicts, len(names), regions)
    return {key: [(s["start"] / SR, s["end"] / SR, names[s["k"]], float(s["p"][s["k"]])) for s in r["segments"]]
            for key, r in d.items()}


def to_annotation(segs, uri: str = "x") -> Annotation:
    ann = Annotation(uri=uri)
    for a, b, who, *_ in segs:
        ann[Segment(a, b)] = who
    return ann


def read_rttm(path) -> Annotation:
    ann = None
    for line in open(path):
        f = line.split()
        ann = ann or Annotation(uri=f[1])
        ann[Segment(float(f[3]), float(f[3]) + float(f[4]))] = f[7]
    return ann
