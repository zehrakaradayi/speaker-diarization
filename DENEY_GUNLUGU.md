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
Veri: 4 kişi, 439 benzersiz train parçası (augmentation'la 1.756), 212 val, 208 test parçası. CPU: 48 vCPU, GPU yok.

### Model A — sıfırdan CNN (onaylanan: orta boy, ~135 bin parametre)
Girdi: log-mel spektrogram, `MelSpectrogram(16000, n_fft=400, hop_length=160, n_mels=64)` → `log(m + 1e-6)` → 1 × 64 × 201.

| Katman | Yapı | Çıkış |
|---|---|---|
| Blok 1 | Conv 3×3 (1→16) → BatchNorm → ReLU → MaxPool 2×2 | 16 × 32 × 100 |
| Blok 2 | Conv 3×3 (16→32) → BatchNorm → ReLU → MaxPool | 32 × 16 × 50 |
| Blok 3 | Conv 3×3 (32→64) → BatchNorm → ReLU → MaxPool | 64 × 8 × 25 |
| Blok 4 | Conv 3×3 (64→64) → BatchNorm → ReLU → MaxPool | 64 × 4 × 12 |
| Blok 5 | Conv 3×3 (64→128) → BatchNorm → ReLU → MaxPool | 128 × 2 × 6 |
| Sınıflandırıcı | AdaptiveAvgPool (global average pooling) → Flatten → Dropout 0,3 → Linear 128→4 | 4 |

- Tek `nn.Sequential`; ilk 5 eleman blok (canlı panel bunu bekliyor), sınıflandırıcı ayrı bir Sequential'a sarılmadı.
- Eğitim: AdamW lr 1e-3, batch 64, ~30 epoch, en iyi validation accuracy'li epoch saklanır. `torch.set_num_threads(48)`, `num_workers=4`.
- Neden bu boy: 246 bin parametrelik büyük sürüm bu kadar az veride ezberleme riski taşır; 62 bin parametrelik küçük sürüm ince ses farklarını kaçırabilir.

### Model B — pretrained ECAPA + sınıflandırıcı
- `snapshot_download("speechbrain/spkrec-ecapa-voxceleb")` → o klasörden `EncoderClassifier` yüklenir, tamamen dondurulur.
- Her parçanın (train'in 4 kopyası, val, test) 192 sayılık embedding'i bir kez hesaplanıp kaydedilir.
- Sınıflandırıcı: L2-normalize edilmiş embedding → Linear 192→4 (772 parametre). Saniyeler içinde eğitilir. Web'e giderse `export_model_b(..., l2_normalize_embedding=True)`.

### Adil karşılaştırma
İki model aynı train/val/test parçalarıyla ve aynı augmentation kopyalarıyla eğitilir; SpecAugment gibi ek augmentation hiçbirine eklenmez. DER için ikisi de aynı VAD'ı ve aynı pencereyi (2 sn, 0,5 sn kaydırma) kullanır.

### Onur'un iPad izi
Alçak geçiren filtre augmentation'ı önerildi, eklenmemesine karar verdim; veri olduğu gibi kalıyor. Etkisi Aşama 4'te karışıklık matrisinde (Onur satırı/sütunu) ve Aşama 5'te laptop mikrofonuyla canlı testte kontrol edilecek.

### Tahminim
**Model B az farkla kazanır.** Pretraining fark yaratır ama kişi sayısı az (4) ve sesler ayrışık olduğu için sıfırdan CNN de iyi öğrenir; DER farkının birkaç puan olmasını bekliyorum.

## Aşama 4 — Training ve DER Karşılaştırma

## Aşama 5 — Dene ve Karar Ver

## Aşama 6 — Productize ve Deploy

## Aşama 7 — Modelin Kulağından Dinle
