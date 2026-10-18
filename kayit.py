"""Aşama 1 — Kayıt sayfası: laptop mikrofonuyla tek tıkla kayıt al, dosya doğru klasöre düşsün.

Tarayıcı sesi gürültü azaltma / yankı engelleme / otomatik seviye KAPALI olarak kaydeder, WAV'a çevirir
ve buraya gönderir → dataset/<isim>/oturum2_laptop.wav gibi doğru yere yazılır.
Telefon kayıtlarını (m4a) elle sürükle; sağdaki tablo onları da sayar (Aşama 1 kontrol noktası).

Çalıştırma:  .venv/bin/python kayit.py   →  https://student.exposureai.org/lab/proxy/8003/
"""

import json
import re
import subprocess
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "kayit.html"
DATASET = ROOT / "dataset"
ROOM = ROOT / "oda_sesi"
PORT = 8003
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".ogg", ".webm", ".flac"}
# Sayfadan kaydedilebilen dosyalar: laptop oturumları ve laptop oda sesi
TARGETS = {"oturum2_laptop", "oturum3_laptop", "ofis_laptop"}
NAME_RE = re.compile(r"^[a-z0-9_]{1,32}$")
MAX_BYTES = 60 * 1024 * 1024  # 48 kHz 16-bit mono ~5 dakika


def duration(path: Path) -> float:
    """Ses dosyasının süresi (saniye). WAV'ı doğrudan, diğerlerini ffprobe ile okur."""
    if path.suffix == ".wav":
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except (wave.Error, EOFError):
            pass  # wave modülünün okuyamadığı WAV'lar (ör. float) ffprobe'a düşer
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def audio_files(folder: Path) -> list[dict]:
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in AUDIO_EXT]
    return [{"name": p.name, "sec": round(duration(p), 1)} for p in files]


def status() -> dict:
    """Her kişinin dosyaları + oda sesleri: sayfadaki kontrol tablosu bunu çizer."""
    DATASET.mkdir(exist_ok=True)
    ROOM.mkdir(exist_ok=True)
    people = [{"person": d.name, "files": audio_files(d)} for d in sorted(DATASET.iterdir()) if d.is_dir()]
    return {"people": people, "room": audio_files(ROOM)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self._json(200, status())
        else:
            self._json(404, {"error": "bulunamadı"})

    def do_POST(self) -> None:
        url = urlparse(self.path)
        if url.path != "/api/save":
            return self._json(404, {"error": "bulunamadı"})
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        target, person = q.get("target", ""), q.get("person", "")
        if target not in TARGETS:
            return self._json(400, {"error": "geçersiz oturum"})
        if target == "ofis_laptop":
            out = ROOM / "ofis_laptop.wav"
        else:
            if not NAME_RE.match(person):
                return self._json(400, {"error": "isim sadece küçük harf, rakam ve _ içermeli"})
            out = DATASET / person / f"{target}.wav"
        if out.exists() and q.get("overwrite") != "1":
            return self._json(409, {"error": "exists", "path": str(out.relative_to(ROOT))})
        size = int(self.headers.get("Content-Length", 0))
        if not 44 < size <= MAX_BYTES:
            return self._json(400, {"error": "kayıt boş ya da çok büyük"})
        data = self.rfile.read(size)
        if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return self._json(400, {"error": "WAV değil"})
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        self._json(200, {"path": str(out.relative_to(ROOT)), "sec": round(duration(out), 1)})

    def log_message(self, fmt: str, *args) -> None:
        if "/api/save" in self.path:
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print(f"Kayıt sayfası: https://student.exposureai.org/lab/proxy/{PORT}/")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
