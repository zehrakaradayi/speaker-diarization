# Deney Günlüğü — Speaker Diarization

## Aşama 1 — Ses Toplama
- Kişiler: zehra, elif, onur, cem (4 kişi)
- Laptop oturumları `kayit.py` sayfasıyla (tarayıcının gürültü azaltma / yankı engelleme / otomatik seviye ayarları kapalı), telefon oturumları telefonun ses kayıt uygulamasıyla alındı.
- Oda sesi: `ofis_laptop.wav` (0:30), `ofis_telefon.m4a` (0:30)

| Kişi | Oturum 1 (telefon) | Oturum 2 (laptop) | Oturum 3 (laptop) | Oturum 4 (telefon) | Toplam |
|---|---|---|---|---|---|
| zehra | 2:14 | 2:25 | 2:01 | 2:01 | 8:42 |
| elif | 1:50 | 2:23 | 2:07 | 2:00 | 8:19 |
| onur | 1:28 | 2:31 | 2:04 | 2:33 | 8:35 |
| cem | bekleniyor | 2:20 | 2:20 | bekleniyor | 4:39 |

Kararlar ve sapmalar:
- **Onur:** dört oturumun hepsi aynı iPad'le (Apple Ses Kayıtları) kaydedildi; "laptop" oturumları da iPad'den. Yeniden kayıt yerine olduğu gibi kullanmaya karar verdim. Risk: model Onur'u sesinden değil iPad mikrofonundan tanıyabilir. Test oturumu (3) da iPad olduğu için test skoru bu riski gizleyebilir; canlı testte (laptop mikrofonu) Onur'a özellikle bakılacak.
- **Cem:** telefon kaydı WhatsApp sesli mesajı olarak geldi (sıkıştırılmış, oturumu belirsiz), kullanılmadı. Oturum 1 ve 4 telefonla yeniden kaydedilecek.
- Zehra'nın paylaşılan klasöründeki, Onur'un oturum 3 kaydının kopyası olan dosya silindi; veriye girmedi.

## Aşama 2 — Data Cleaning

## Aşama 3 — Model Seçimi
- Model A (sıfırdan CNN) mimarisi:
- Model B (ECAPA + sınıflandırıcı) planı:
- Tahminim:

## Aşama 4 — Training ve DER Karşılaştırma

## Aşama 5 — Dene ve Karar Ver

## Aşama 6 — Productize ve Deploy

## Aşama 7 — Modelin Kulağından Dinle
