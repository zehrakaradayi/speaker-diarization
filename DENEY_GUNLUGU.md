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
| cem | 2:01 | 2:20 | 2:20 | 1:41 | 8:22 |

Kararlar ve sapmalar:
- **Onur:** dört oturumun hepsi aynı iPad'le (Apple Ses Kayıtları) kaydedildi; "laptop" oturumları da iPad'den. Yeniden kayıt yerine olduğu gibi kullanmaya karar verdim. Risk: model Onur'u sesinden değil iPad mikrofonundan tanıyabilir. Test oturumu (3) da iPad olduğu için test skoru bu riski gizleyebilir; canlı testte (laptop mikrofonu) Onur'a özellikle bakılacak.
- **Cem:** ilk telefon kaydı (`telefon.mp4`, 1:32) tek dosyaydı ve oturumu belirsizdi, kullanılmadı. Oturum 1 ve 4 ayrı ayrı yeniden kaydedildi (WhatsApp'tan geldi ama 133 kbps AAC, 18 kHz bant; diğer telefon kayıtlarıyla aynı kalite). Hangisinin hangi oturum olduğunu dinleyerek doğruladım.
- Zehra'nın paylaşılan klasöründeki, Onur'un oturum 3 kaydının kopyası olan dosya silindi; veriye girmedi.

## Aşama 2 — Data Cleaning
Script: `hazirla.py` (her çalıştırmada `data/` baştan üretilir, seed 42).

- **Tek format:** her kayıt ffmpeg ile 16 kHz mono'ya çevrildi.
- **Sessizlik:** Silero VAD (min sessizlik 300 ms, 60 ms pay); konuşma bölümleri birleştirildi. Ham sürenin %80–95'i konuşma çıktı.
- **Seviye:** her 2 sn parça -20 dBFS RMS'e eşitlendi, sonra [-1, 1]'e kırpıldı. Bu, `web_export` dosyasının tarayıcıda her pencereye uyguladığı kuralın aynısı; eğitim ve canlı kullanım aynı sesi görür.
- **Parçalar:** konuşma örtüşmesiz 2 sn parçalara bölündü, sondaki 2 sn'den kısa kalan atıldı.
- **Bölme:** oturum 1–2 train, 4 validation, 3 test. Bir kaydın tüm parçaları tek bölümde.
- **Augmentation (sadece train):** temiz + `gurultu` (oda sesi, 5–20 dB SNR) + `yanki` (sentetik oda yankısı, RT60 0,2–0,7 sn) + `seviye` (parça içinde yavaşça değişen ±8 dB kazanç). Seviye için sabit kazanç yerine zarf kullanıldı, çünkü sabit kazanç pencere normalizasyonunda tamamen silinirdi. Hız/perde değişikliği yok.
- **Sahte sohbetler:** sadece test (oturum 3) konuşmasından 2–6 sn sıralar, art arda aynı kişi yok, aralarda 0,3–1 sn; sohbetin tamamının altında laptop oda sesi. 20 × 60 sn, `data/sohbetler/sohbet_XX.rttm` + `hepsi.rttm`. Test konuşması kişi başı ~2 dk olduğu için sohbetler aynı test sesini farklı yerlerden tekrar kullanıyor. Silero VAD'ın bu sohbetlerde RTTM'e göre konuşma tespiti hatası ~%6 (cümle içi kısa duraklamalar).

| Kişi | Train (temiz) | Train (+3 aug) | Val | Test |
|---|---|---|---|---|
| cem | 103 | 412 | 31 | 53 |
| elif | 102 | 408 | 50 | 50 |
| onur | 109 | 436 | 72 | 57 |
| zehra | 125 | 500 | 59 | 48 |

Gözlemler:
- **Cem:** oturum 4 (serbest konuşma) 1:41'in sadece 63 sn'si konuşma çıktı (çok duraksama), bu yüzden validation'ı en küçük (31 parça). Ayrıca sesi beklenenden alçak/pes (medyan perde ~96 Hz, diğer oturumlarında ~114 Hz); "enerjik ve tiz" tonu tam tutturulamamış.
- **Onur'un spektrogramı:** 3,5 kHz üstünde kara (boş) bantlar var. iPad'in AAC sıkıştırması yüksek frekansları atıyor. Diğer üç kişide bu yok. Model Onur'u bu "kara bant" izinden tanımayı öğrenebilir. Aşama 3'te buna karşı önlem konuşulacak (ör. herkese rastgele alçak geçiren filtre / codec augmentation).

## Aşama 3 — Model Seçimi
- Model A (sıfırdan CNN) mimarisi:
- Model B (ECAPA + sınıflandırıcı) planı:
- Tahminim:

## Aşama 4 — Training ve DER Karşılaştırma

## Aşama 5 — Dene ve Karar Ver

## Aşama 6 — Productize ve Deploy

## Aşama 7 — Modelin Kulağından Dinle
