# Algı (perception) alanında yaptığımız her şey: özet

Kapsam: Ekim - Kasım 2026, kendi kameramız ve kendi IMU'muzla yarış drone'u hattı (Isaac simülasyonu).
Kaynaklar: vault `01-Projects/DroneRacing/droneracing-perception-80-goal.md` (55 bölümlük günlük), bu oturumdaki ölçümler, `runs/cam_eval/*.txt`.
Her rakamın yanında nerede ölçüldüğü yazılı. Ölçülmeyen şeyi ölçülmüş gibi yazmadım.

## 0. Okurken bilmen gerekenler

- **İsim çakışması:** Ekim günlüğündeki "g17", "g18", "g19" **politika** (uçuran ağ) adlarıdır. Kasım'daki "g15 → g16 → g17 → g18 → g19" ise **dedektör** (kapı köşelerini bulan ağ) adlarıdır. Bu belgede ikisini ayrı yazdım: "politika g17" ve "dedektör g17".
- **Ölçüm gürültüsü:** Her seed 4 drone ile uçurulur (820×616 render sınırı). Aynı seed, aynı komutla bir koşuda %25, ötekinde %100 verdi. 64 uçuşun standart hatası yaklaşık ±4.7 puan. Bu yüzden küçük farklar kanıt değildir.
- **Metrikler:** "Kapı bulundu" = kapının görünen köşelerinden en az 2'si, doğru türde bir tepe ile 4 piksel içinde eşleşti. "Algı ≥1 m" = kapıya 1 m'den uzak karelerde bulma oranı. "Yol hızı" = parkur yolu / tur süresi.

## 1. Algı hattı (şu anki hali)

| Blok | Ne yapar | Ayar |
|---|---|---|
| Kamera | 820×616 görüntü, tam görüş alanı (yaklaşık 128°), 45° yukarı eğim | 30 Hz, 50 ms gecikme |
| Dedektör (GateNet) | Kamera roll açısıyla görüntüyü döndürür (derotasyon), 384×384 kırpımdan kapı köşe tepelerini bulur | `droneracing/gatenet.py` |
| Tahminci (GateEKF5) | IMU ile durumu ilerletir, köşe ölçümlerini süzer, **gecikmeyi durum geri-sarma ile telafi eder** | `droneracing/ekf5.py`, `n_map` 32 |
| Politika | Hız, yön ve sonraki 3 kapının konumunu alıp motor komutu üretir | g37, g38d... |

Gerçekçilik katmanı (test ve eğitim): rolling shutter 20 ms, %10 bozuk kare, düşük pozlama, 3° bilinmeyen kamera montaj hatası (offline kalibrasyonla düzeltilmiş).

## 2. Ekim: hattı çalışır hale getirmek

### 2.1 Gecikme: en büyük tek kayıp (bölüm 2-6)
İlk kendi-kamera koşusunda gecikme yok sayılınca **%0 başarı** (kapıların %9.5'i geçildi, kapı hatası 0.6-1.4 m). Tekrar oynatma (offline) ile ölçüm, kapı konum hatası p90 (0-3 m):

| Durum | Hata |
|---|---|
| Gecikme yok (referans) | 0.38 / 0.36 m |
| 50 ms gecikme, yok sayıldı | 1.87 / 2.43 m |
| 50 ms gecikme, **telafi edildi (GateEKF5)** | 0.38 / 0.36 m |

Telafi, gecikmeyi doğru bilmeyi gerektirir: 20 ms'lik yanlış gecikme başarıyı 75 → 37.5 %'e düşürdü. Yöntem MonoRace'in gecikme telafisidir (makalede hata 0.289 → 0.103 m).

### 2.2 İki hata bulundu ve düzeltildi
- **Test görüntüsünde titreşim** (kullanıcı videolardan fark etti): eğitim artırımı her karede yeniden çekiliyordu. Düzeltme: `--aug-stable` (görünüm drone başına bir kez çekilir). Etki: +7.3 puan.
- **Kapı düzeltme tablosu 16 yuvalıydı, parkur 22 kapı** (2. tur bozuktu): `n_map` 32. Etki: +9.4 puan.

### 2.3 Sonuç ve tekrar edememe (bölüm 21-22)
İkisiyle seed 21-24'te %80.2'ye çıkıldı, **ama seed 41-44'te %69.8 çıktı**. Sekiz seed toplamı **%75.0** (yaklaşık 16 m/s). Yani ilk iyi sonuç kısmen şanstı.

### 2.4 Kamera hızı (bölüm 11)
30 Hz: %63.5, 60 Hz: %71.9 (+8.4 puan). 60 Hz, Jetson'un standart sürücüsünde tam görüş alanıyla yok (1280×720 modu dikey görüşü kırpar), bu yüzden donanıma bağlı.

## 3. Gerçekçi stres: hat çöktü

Üç stres birlikte (rolling shutter 20 ms + %10 bozuk kare + 3° montaj hatası): **%75 → %16.7** (bölüm 23). Tek başına 3° montaj hatası **%29.2**. Yani kaybın çoğu montaj hatasından.

### 3.1 Montaj hatası: ana gerçek-kamera riski (bölüm 30-32, 45)
| Bilinmeyen montaj hatası | Başarı |
|---|---|
| 0° | %75.0 |
| 1° | %50.0 |
| 2° | %33.4 |
| 3° | %16.7 |

- **Çevrim-içi tahmin (GateEKF6)** denendi, çalışmadı.
- **Çözüm: offline kalibrasyon** (`droneracing/extrinsic.py`, `scripts/calib_extrinsic.py`). Bir uçuş kaydından, yer gerçeği olmadan, köşe kalıntısını en aza indirerek montaj dönmesi bulunur. Isaac'te dört drone için 3.80 / 2.19 / 1.33 / 2.84° → **0.56 / 0.59 / 0.39 / 0.20°** (ortalama 0.44°). Hedef: 0.5° altı.

### 3.2 Bozuk kareler ve rolling shutter
- **Kare reddi (bölüm 44):** bütün kareyi atmak (%15 kare gidiyor) algıyı %73.8-80.3'e düşürdü; hiç kontrol etmemek %83.4-83.9; **sadece bozuk bant satırlarını reddetmek (`--band-check`) %86.3-85.1**. Bant dedektörü: temiz karelerde satırların %0.18'i yanlış işaretli, bant satırlarının %84.7'si bulundu.
- **Rolling shutter telafisi (`droneracing/rscomp.py`):** her köşe, jiro hızı ve satır zamanıyla kare ortası zamanına geri taşınır. Sentetik doğrulamada 14 piksele kadar kayma 0.65 pikselin altına iniyor.

## 4. Dedektör çalışmaları

### 4.1 Soy ağacı ve offline sonuçlar (held-out kareler, gerçek-kamera etkili)
| Dedektör | Ne eklendi | Kapı bulundu |
|---|---|---|
| g5 | temiz eğitim, 410×308 | %72.6 (yanlış tepe 2.4/kare) |
| g13 | gerçek-kamera artırımı | %85.5 (1.3) |
| g14 | + 820×616, 384 kırpım, roll derotasyonu | **%91.0** (0.9) |
| g15 | + yakın mesafe verisi (0.8-3.5 m, burun aşağı) | %90.3 (karma test) |

Gerçek-kamera artırımı (`droneracing/realcam_aug.py`): rolling shutter (dönme ve ileri hareketten), hareket bulanıklığı, kısa pozlama, lens kenarı bulanıklığı, bozuk bantlar. Etiketler görüntüyle birlikte kaydırılır (tutarlı).
Düşük tepe eşiği çare değil: eşik 0.15'te %87.6 bulur ama kare başına 5.9 yanlış tepe verir.

### 4.2 GateSeg (segmentasyon ağı): bırakıldı (bölüm 24-28)
Kapalı çevrimde aynı politikayla karşılaştırma: 0-3 m'de GateSeg %62, GateNet %70; 3-6 m'de GateSeg %39, GateNet %61. Köşe hatası benzer ama bulma oranı düşük.

### 4.3 MonoRace yeniden uygulaması (bölüm 38-41)
GateNetMR (U-Net), QuAdGate (çizgi çıkarma ve eşleme), GateEKF8 (köşe başına piksel kalıntısı) uygulandı. **Paylaşılan hata düzeltmesi** bulundu: aynı kapının köşeleri bağımsız sayılınca kovaryans çöküyordu (filtre sapması 4 cm, gerçek hata 1 m); ortak harita terimi köşe sayısıyla çarpılınca düzeldi. Eğitim yarıda (40 epoch'un 20'sinde) kalmıştı. Bu belgede, nihai uçuş hattının GateNet tabanlı (g14/g15 sonrası g17) olduğu dışında MR ağının kapalı çevrim sonucu yok; onu iddia etmiyorum.

## 5. Neyin parasını ödüyoruz? (ablasyon, bölüm 34-35)

Yer gerçeğini kanal kanal vererek (politika g17, seed 21/22):

| Gerçek verilen | Başarı |
|---|---|
| Hiçbiri (kamera hattı) | %75.0 |
| Sadece tutum | %68.8 (kazanç yok) |
| **Sadece hız** | **%87.5** |
| **Sadece kapı konumu ve normali** | **%87.5** |
| Üçü birden | **%100** |

Okuma: hız hatası (0.4 m/s) ve kapı konum hatası yaklaşık 12.5'er puan, tutum hatası sıfır puan kaybettiriyor. Mükemmel algıyla politika %100 uçuyor: **sınır algıdır, politikanın kendisi değil**.

## 6. İşe yaramayanlar (kazanç yok)
- Filtre gürültü ayarı (chi2, sig_aided, kappa süpürmesi).
- İvme bias rastgele yürüyüşü.
- Daha doğru kapı haritası (0.10 m yerine 0.25 m hata): %78.1'e karşı %80.2.
- Kapıya yakın IMU ile uçuş (algıyı kesmek): kazanç yok.
- Sürükleme destekli hız (`ekf9`): bu IMU modeliyle kullanılamaz (hız hatası 0.235 → 0.53 m/s).
- GateSeg, çevrim-içi montaj tahmini (GateEKF6).

## 7. Algı ile politika arasındaki köprü
- **Algı iyileşince uçuş orantılı artmıyor** (bölüm 50): g15 ile algı %90.0'a çıktı, uçuş %56-66'da kaldı; temiz hat (stressiz) stresli hattan iyi değildi. Kasım'daki 47 seed satırında başarı ile algı oranının korelasyonu **+0.12**; hız hatası -0.41, tutum hatası -0.39, yakın mesafe kapı hatası -0.27.
- **MonoRace algı cezası** (`--perc-pen`, bir sonraki kapıyı görüş alanında tutar): politika g17 → g27, aynı hatta **%56.3 → %71.9**.
- **Tahminci-döngüde eğitim** (politika, GateEKF5'in çıktısını görür): benzetim Isaac'e 1-4 puan içinde uyar, ama kendi içinde eğitilmiş politikaları fazla iyi gösterir (g34: benzetim %88-91, Isaac %78-82). Sıralama her zaman Isaac'te yapıldı.

## 8. Kasım: dedektörün on-policy turları

Dedektörü, politikanın gerçekten uçtuğu görüntü dağılımıyla ince ayarlamak:

| Adım | Politika | Dedektör | 16 seed Isaac başarı |
|---|---|---|---|
| g34 + g15 | g34 | g15 | %76.6 |
| g34 + g16 | g34 | g16 (g34'ün kareleriyle) | %79.7 |
| g34 + g17 | g34 | g17 (g16'nın kareleriyle) | %82.8 |
| g34 + g18 | g34 | g18 (bir tur daha) | 8 seed'de g17'nin altında, bırakıldı |
| **g37 + g17** | g37 | g17 | **%87.5** (16 seed); taze seed'lerle birleşik **90/100** |

Her tur yaklaşık +3 puan verdi, üçüncü tur geriledi: **on-policy dedektör turlarının getirisi tükendi**. Bundan sonraki kayıp dedektörden değil, hız tahmin hatasından ve kapı konum gürültüsünden.

## 9. Kasım: hız arttıkça algı bozuluyor (g38d)

Hızlı politika (g38d, 19 m/s) Isaac'te, aynı seed'lerde (71-77) g37 ile:

| Ölçü | g37 (15-16 m/s) | g38d (19 m/s) |
|---|---|---|
| Algı ≥1 m | %92.7 | **%89.0** |
| Hız tahmin hatası | 0.35 m/s | **0.44** (+26%) |
| Yakın mesafe (0-2 m) kapı hatası | 0.076 m | **0.099** (+30%) |
| Başarı | %93 | %75 |

Benzetim bunu göstermiyor (yapay algı modeli kullanıyor). Karşı hamle: **dedektör g19** = g17, g38d'nin kendi uçuşlarından toplanan karelerle (seed 91-94) ince ayarlandı. İlk epoch'larda zor durumlarda recall arttı (1 köşe görünürken 0.526 → 0.632). Isaac'te seed 71'de algı %93.0 (g17: %90.6). **8 seed'lik uçuş sonucu bekliyor.**

## 10. Açık algı riskleri (gerçek drona geçişte)

| Risk | Durum |
|---|---|
| Montaj hatası | Offline kalibrasyon yöntemi var (0.44° Isaac'te). Gerçek dronda Kalibr ile 0.5° altı gerekir. |
| Gecikme | Telafi doğru gecikmeyi gerektirir. LED testi ile ölçülecek. Yanlış 20 ms başarıyı yarıya indirdi. |
| Bozuk kare türü | Bizim modelimiz **yatay satır bantları**. Bugün gösterdiğin FPV karesinde **dağınık blok bozulması** vardı; gerçek kameramız blok bozulması verirse dedektör görmedi. Önce ham kare kaydı (`jetson/capture_ts.py`) ile ölçmek gerekir. |
| Jetson hızı | g17 ONNX hazır (`jetson/onnx/gatenet_g17_384.onnx`), TensorRT süresi ölçülmedi. Bütçe 85 fps'te kare başına 11.7 ms. |
| Kamera modu | 820×616 85 fps için register tablosu ve device-tree hazır, kernel modülüne eklenmedi. |
| Başlangıç çarpmaları | İncelenen 5 çarpmanın 5'i kapı 0'da, başlangıçtan ~50 adım sonra. Sebep ayrılmadı (politika manevrası mı, yeniden başlatma sonrası render/tahminci gecikmesi mi). |

## 11. Ana dersler

1. **Gecikmeyi telafi etmek** hattı %0'dan çalışır hale getiren tek en büyük adımdı.
2. **Montaj hatası** en büyük gerçek-kamera riski; çözüm görüntü artırımı değil **kalibrasyon**.
3. Mükemmel algıyla %100 uçuluyor: **sınır algı ve tahmin**, politika değil.
4. Algı oranı tek başına uçuşu belirlemiyor; **hız ve yakın kapı hatası** daha ilişkili.
5. Küçük örnekle karar verilmez: seed başına 4 drone gürültülüdür, 16+ seed ve ayrık (hiç kullanılmamış) seed gerekir.
6. Benzetim, kendi içinde eğitilmiş politikaları iyimser gösterir; **Isaac'te ölç**.
7. Dedektörün on-policy turlarının getirisi 3 turda tükendi.

## 12. Dosya ve araç haritası

| Ne | Nerede |
|---|---|
| Tahminciler (gecikme telafisi, köşe ölçümü) | `droneracing/ekf5.py`, `ekf8.py` (ekf6, ekf7, ekf9 denendi) |
| Gerçek-kamera artırımı | `droneracing/realcam_aug.py` |
| Rolling shutter telafisi | `droneracing/rscomp.py` |
| Montaj kalibrasyonu | `droneracing/extrinsic.py`, `scripts/calib_extrinsic.py`, `runs_host/calib/cal15_41.pt` |
| Bozuk bant dedektörü | `droneracing/framecheck.py` (`--band-check`) |
| Dedektör eğitimi | `scripts/train_gatenet2.py` (g16/g17/g19 için `run_onpolicy16.sh`, `run_g17.sh`, `run_g19.sh`) |
| Isaac değerlendirme ve algı metriği | `isaac/eval_vision.py`, `scripts/cam_eval.sh` |
| Kaza analizi | `scripts/fail_anatomy.py`, `scripts/crash_anatomy.py` |
| Jetson araçları | `jetson/` (ONNX, kamera modu, zaman damgası, LED gecikme, IMU) |
| Ayrıntılı günlük (İngilizce) | vault `01-Projects/DroneRacing/droneracing-perception-80-goal.md` |
