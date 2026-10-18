"""Yeni arayüzün (arayuz.py) önizlemesi: app.py'ye dokunmadan, onun mantığıyla ayrı portta çalışır.

app.py'nin HTML fonksiyonları arayuz.py'dekilerle değiştirilir; analyze, VAD ve modeller aynen app.py'den.
Tasarım onaylanınca app.py arayuz.py'yi doğrudan kullanacak ve bu dosya silinecek.

Çalıştırma:  .venv/bin/python onizleme.py   →  https://student.exposureai.org/lab/proxy/7861/
"""

import threading

import app
import arayuz

PORT = 7861

for fn in ("status_html", "timeline_html", "now_html", "score_html"):
    setattr(app, fn, getattr(arayuz, fn))

demo = arayuz.build(app.analyze, app.change_person, app.on_load, app.people())

if __name__ == "__main__":
    threading.Thread(target=app._load_in_background, daemon=True).start()
    print(f"Önizleme: https://student.exposureai.org/lab/proxy/{PORT}/")
    demo.launch(server_name="127.0.0.1", server_port=PORT, root_path=f"/lab/proxy/{PORT}",
                head=app.HEAD, css=arayuz.CSS, theme=arayuz.THEME)
