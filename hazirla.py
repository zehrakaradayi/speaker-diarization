"""Aşama 2 — Data cleaning: dataset/ ve oda_sesi/ → data/ (modelin öğrenebileceği hale).

1. Tek format: her kayıt 16 kHz mono WAV'a çevrilir            → data/wav16/
2. Sessizliği at: Silero VAD'ın bulduğu konuşmalar birleştirilir → data/speech/
3. 2 saniyelik parçalar + seviye eşitleme (her parça -20 dBFS RMS; web_export ile aynı kural)
4. Bölme: oturum 1-2 train, 4 validation, 3 test (aynı kaydın parçaları asla farklı bölüme düşmez)
5. Augmentation: her train parçasının temiz + gürültülü + yankılı + seviyesi oynatılmış kopyası
6. Sahte sohbetler: sadece test oturumlarından, 20 × 1 dk + RTTM doğru cevap dosyaları → data/sohbetler/
Kontrol noktası: parça sayısı tablosu + her kişiden bir spektrogram (data/spektrogramlar.png).

Çalıştırma:  .venv/bin/python hazirla.py      (data/ her seferinde baştan üretilir)
"""

import csv
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve
from silero_vad import get_speech_timestamps, load_silero_vad

ROOT = Path(__file__).resolve().parent
DATASET, ROOM, OUT = ROOT / "dataset", ROOT / "oda_sesi", ROOT / "data"
SR = 16000
CHUNK = 2 * SR
TARGET_DBFS = -20.0  # tarayıcıdaki modelle aynı: her 2 sn pencere -20 dBFS RMS'e eşitlenir
SPLITS = {1: "train", 2: "train", 4: "val", 3: "test"}
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".webm", ".flac", ".mp4"}
FILE_RE = re.compile(r"^oturum(\d)_(telefon|laptop)$")
N_CONVS, CONV_SEC = 20, 60.0
rng = np.random.default_rng(42)


def load16k(path: Path) -> np.ndarray:
    """Her formatı (m4a, mp3, wav…) ffmpeg ile 16 kHz mono float32'ye çevirir."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).copy()


def normalize(x: np.ndarray, dbfs: float = TARGET_DBFS) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2)) + 1e-9
    return np.clip(x * (10 ** (dbfs / 20) / rms), -1, 1).astype(np.float32)


def write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, SR, subtype="PCM_16")


# ---------- augmentation ----------
def add_noise(x: np.ndarray, noises: list[np.ndarray]) -> np.ndarray:
    """Oda sesinden rastgele bir parça, 5-20 dB SNR ile."""
    n = noises[rng.integers(len(noises))]
    start = rng.integers(0, len(n) - len(x))
    seg = n[start:start + len(x)]
    snr = rng.uniform(5, 20)
    seg = seg * np.sqrt(np.mean(x**2) / (np.mean(seg**2) + 1e-12) / 10 ** (snr / 10))
    return normalize(x + seg)


def add_reverb(x: np.ndarray) -> np.ndarray:
    """Sentetik oda yankısı: üstel sönen gürültü (RT60 0,2-0,7 sn) + doğrudan ses."""
    rt60 = rng.uniform(0.2, 0.7)
    t = np.arange(int(rt60 * SR)) / SR
    rir = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60) * 0.3
    rir[0] = 1.0
    return normalize(fftconvolve(x, rir)[: len(x)])


def vary_level(x: np.ndarray) -> np.ndarray:
    """Parça içinde yavaşça değişen seviye (±8 dB): konuşurken mikrofona yaklaşıp uzaklaşmak gibi.
    Sabit bir kazanç her pencere -20 dBFS'e eşitlendiği için iz bırakmaz; bu yüzden zarf kullanılır."""
    pts = rng.uniform(-8, 8, size=rng.integers(3, 6))
    env_db = np.interp(np.linspace(0, 1, len(x)), np.linspace(0, 1, len(pts)), pts)
    return normalize(x * 10 ** (env_db / 20))


# ---------- sahte sohbetler ----------
def make_conversation(idx: int, speech: dict[str, np.ndarray], noise: np.ndarray) -> list[tuple]:
    """Test oturumlarından 2-6 sn sıralar; art arda aynı kişi yok, aralarda 0,3-1 sn oda sesi.
    Sohbetin tamamının altında laptop oda sesi var (test de demo da laptopla yapılıyor)."""
    total = int(CONV_SEC * SR)
    start = rng.integers(0, len(noise) - 1)
    audio = np.tile(noise, total // len(noise) + 2)[start:start + total].copy()
    turns, t, prev = [], rng.uniform(0.3, 1.0), None
    people = sorted(speech)
    while t < CONV_SEC - 0.5:
        who = rng.choice([p for p in people if p != prev])
        dur = min(rng.uniform(2, 6), CONV_SEC - t)
        src = speech[who]
        n = int(dur * SR)
        off = rng.integers(0, len(src) - n)
        seg = normalize(src[off:off + n], TARGET_DBFS + rng.uniform(-3, 3))
        s = int(t * SR)
        audio[s:s + n] += seg
        turns.append((t, dur, who))
        t += dur + rng.uniform(0.3, 1.0)
        prev = who
    name = f"sohbet_{idx:02d}"
    write(OUT / "sohbetler" / f"{name}.wav", np.clip(audio, -1, 1))
    with open(OUT / "sohbetler" / f"{name}.rttm", "w") as f:
        for t0, d, who in turns:
            f.write(f"SPEAKER {name} 1 {t0:.3f} {d:.3f} <NA> <NA> {who} <NA> <NA>\n")
    return turns


def spectrogram_figure(examples: dict[str, Path]) -> None:
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(examples), figsize=(4 * len(examples), 3.2), sharey=True)
    for ax, (who, path) in zip(np.atleast_1d(axes), examples.items()):
        x, _ = sf.read(path)
        mel = librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=SR, n_fft=400, hop_length=160, n_mels=64), ref=np.max)
        librosa.display.specshow(mel, sr=SR, hop_length=160, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(f"{who} — {path.name}")
    fig.tight_layout()
    fig.savefig(OUT / "spektrogramlar.png", dpi=110)


def main() -> None:
    for sub in ("wav16", "speech", "oda", "chunks", "sohbetler"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    vad = load_silero_vad()

    noises = []
    for p in sorted(ROOM.iterdir()):
        if p.suffix.lower() in AUDIO_EXT:
            x = load16k(p)
            write(OUT / "oda" / f"{p.stem}.wav", x)
            noises.append(x)
    laptop_noise = load16k(next(ROOM.glob("ofis_laptop.*")))

    rows, test_speech, report = [], {}, []
    for person_dir in sorted(d for d in DATASET.iterdir() if d.is_dir()):
        who = person_dir.name
        for p in sorted(person_dir.iterdir()):
            m = FILE_RE.match(p.stem)
            if p.suffix.lower() not in AUDIO_EXT or not m:
                print(f"  atlandı (isim kurala uymuyor): {p.relative_to(ROOT)}")
                continue
            session, device = int(m[1]), m[2]
            split = SPLITS[session]
            x = load16k(p)
            write(OUT / "wav16" / who / f"{p.stem}.wav", x)
            ts = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=SR,
                                       min_silence_duration_ms=300, speech_pad_ms=60)
            speech = np.concatenate([x[t["start"]:t["end"]] for t in ts]) if ts else np.zeros(0, np.float32)
            write(OUT / "speech" / who / f"{p.stem}.wav", speech)
            if split == "test":
                test_speech[who] = speech
            n = len(speech) // CHUNK
            report.append((who, p.stem, len(x) / SR, len(speech) / SR, n))
            for i in range(n):
                chunk = normalize(speech[i * CHUNK:(i + 1) * CHUNK])
                versions = {"temiz": chunk}
                if split == "train":
                    versions |= {"gurultu": add_noise(chunk, noises), "yanki": add_reverb(chunk),
                                 "seviye": vary_level(chunk)}
                for aug, y in versions.items():
                    out = OUT / "chunks" / split / who / f"{p.stem}_{i:03d}_{aug}.wav"
                    write(out, y)
                    rows.append({"path": str(out.relative_to(ROOT)), "person": who, "session": session,
                                 "device": device, "split": split, "aug": aug, "chunk": i})

    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nKayıt → konuşma (VAD sonrası)")
    print(f"{'kişi':8s} {'kayıt':16s} {'ham':>6s} {'konuşma':>8s} {'parça':>6s}")
    for who, stem, raw, sp, n in report:
        print(f"{who:8s} {stem:16s} {raw:6.1f} {sp:8.1f} {n:6d}")

    people = sorted({r["person"] for r in rows})
    c = Counter((r["person"], r["split"], r["aug"] == "temiz") for r in rows)
    print("\nParça sayıları (2 sn)")
    print(f"{'kişi':8s} {'train temiz':>12s} {'train +aug':>11s} {'val':>5s} {'test':>5s}")
    for who in people:
        tr = c[(who, "train", True)]
        print(f"{who:8s} {tr:12d} {tr + c[(who, 'train', False)]:11d} {c[(who, 'val', True)]:5d} {c[(who, 'test', True)]:5d}")
        for split in ("train", "val", "test"):
            if not c[(who, split, True)]:
                print(f"  UYARI: {who} için {split} parçası yok")

    print(f"\nSahte sohbetler (test oturumları: {', '.join(sorted(test_speech))})")
    (OUT / "sohbetler").mkdir(parents=True, exist_ok=True)
    talk = Counter()
    for k in range(1, N_CONVS + 1):
        for _, d, who in make_conversation(k, test_speech, laptop_noise):
            talk[who] += d
    with open(OUT / "sohbetler" / "hepsi.rttm", "w") as f:
        for p in sorted((OUT / "sohbetler").glob("sohbet_*.rttm")):
            f.write(p.read_text())
    print("  " + ", ".join(f"{w}: {s:.0f} sn" for w, s in sorted(talk.items())))

    examples = {}
    for who in people:
        clean = [r for r in rows if r["person"] == who and r["split"] == "train" and r["aug"] == "temiz"]
        if clean:
            examples[who] = ROOT / clean[len(clean) // 2]["path"]
    spectrogram_figure(examples)
    print(f"\nYazıldı: data/manifest.csv ({len(rows)} parça), data/sohbetler/ ({N_CONVS} sohbet), data/spektrogramlar.png")


if __name__ == "__main__":
    main()
