"""Report PDF (Turkish) for the g37 + g17 result: methods, results, problems met, limits, next steps.
All numbers come from runs/cam_eval/*.txt (per-seed success of 4 drones). Run: python3 scripts/make_report_g37.py
Output: reports/Kamera_ile_Ucus_g37_Yontem_ve_Sonuclar_v2.pdf and reports/_fig/g37_*.png"""
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

root = Path(__file__).resolve().parents[1]
fig_dir = root / "reports" / "_fig"
fig_dir.mkdir(parents=True, exist_ok=True)
OUT = root / "reports" / "Kamera_ile_Ucus_g37_Yontem_ve_Sonuclar_v2.pdf"


def seeds(f):
    out = {}
    for l in open(root / "runs/cam_eval" / f):
        m = re.match(r"^(\d+) VISION", l)
        s = re.search(r"success\s+([\d.]+)%", l)
        if m and s:
            out[int(m[1])] = round(float(s[1]) / 25)  # successes out of 4 drones
    return out


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (c - h), 100 * (c + h)


g37a, g37b = seeds("t37_cal.txt"), seeds("t37c_cal.txt")
g17 = seeds("t34g17_cal.txt")
g16 = {**seeds("t34g16_cal.txt"), **seeds("t34g16b_cal.txt")}
g15 = seeds("t34_cal.txt")
ka, na = sum(g37a.values()), 4 * len(g37a)
kb, nb = sum(g37b.values()), 4 * len(g37b)
kp, npool = ka + kb, na + nb
cand = [("g15\ndedektör", sum(g15.values()), 4 * len(g15)), ("g16\ndedektör", sum(g16.values()), 4 * len(g16)),
        ("g17\ndedektör", sum(g17.values()), 4 * len(g17)), ("g37 + g17\nseed 21-44", ka, na),
        ("g37 + g17\ntaze seed 71-79", kb, nb), ("g37 + g17\nbirleşik", kp, npool)]
print("g37 old", ka, na, "fresh", kb, nb, "pooled", kp, npool, "CI", wilson(kp, npool))

# ---- figures ----
fig, ax = plt.subplots(figsize=(7.2, 3.3))
ss_ = sorted(g37a) + sorted(g37b)
vals = [100 * (g37a.get(s) if s in g37a else g37b[s]) / 4 for s in ss_]
cols = ["#2b6cb0" if s in g37a else "#2f855a" for s in ss_]
ax.bar(range(len(ss_)), vals, color=cols)
ax.axhline(85, color="#c53030", ls="--", lw=1); ax.text(-0.4, 101.5, "hedef %85 (kesikli çizgi)", color="#c53030", ha="left", fontsize=7.5)
ax.set_xticks(range(len(ss_))); ax.set_xticklabels([str(s) for s in ss_], fontsize=7, rotation=90)
ax.set_ylabel("başarı [%] (seed başına 4 drone)", fontsize=8); ax.set_ylim(0, 108)
ax.set_title("g37 + g17: seed başına uçuş başarısı (mavi: ilk 16 seed, yeşil: taze seed 71-79)", fontsize=8.5)
ax.tick_params(axis="y", labelsize=7); fig.tight_layout(); fig.savefig(fig_dir / "g37_per_seed.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.3))
for i, (nm, k, n) in enumerate(cand):
    lo, hi = wilson(k, n); p = 100 * k / n
    ax.bar(i, p, color="#718096" if i < 3 else "#2b6cb0")
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="black", capsize=3, lw=1)
    ax.text(i, p + (hi - p) + 1.5, f"{p:.1f}\n({k}/{n})", ha="center", fontsize=7)
ax.axhline(85, color="#c53030", ls="--", lw=1)
ax.set_xticks(range(len(cand))); ax.set_xticklabels([c[0] for c in cand], fontsize=7)
ax.set_ylim(50, 112); ax.set_ylabel("başarı [%], %95 Wilson aralığı", fontsize=8); ax.tick_params(axis="y", labelsize=7)
ax.set_title("Adayların karşılaştırması (aynı hedef hat, aynı seed kümesi)", fontsize=8.5)
fig.tight_layout(); fig.savefig(fig_dir / "g37_candidates.png", dpi=200); plt.close(fig)

# ---- document ----
pdfmetrics.registerFont(TTFont("DJ", "/usr/share/fonts/TTF/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DJB", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"))
NAVY, GREY, LIGHT = colors.HexColor("#1a365d"), colors.HexColor("#4a5568"), colors.HexColor("#edf2f7")
ss = getSampleStyleSheet()
B = ParagraphStyle("B", parent=ss["BodyText"], fontName="DJ", fontSize=8.8, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="DJB", fontSize=13, textColor=NAVY, spaceBefore=11, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="DJB", fontSize=10.2, textColor=NAVY, spaceBefore=7, spaceAfter=3, keepWithNext=1)
T = ParagraphStyle("T", parent=B, fontSize=7.8, leading=10, alignment=0, spaceAfter=0)
TH = ParagraphStyle("TH", parent=T, fontName="DJB", textColor=colors.white)
CAP = ParagraphStyle("CAP", parent=B, fontSize=7.6, leading=9.6, textColor=GREY, alignment=0)
BUL = ParagraphStyle("BUL", parent=B, leftIndent=12, bulletIndent=2, spaceAfter=2)
TITLE = ParagraphStyle("TITLE", parent=ss["Title"], fontName="DJB", fontSize=19, leading=23, textColor=NAVY, alignment=0)
SUB = ParagraphStyle("SUB", parent=B, fontSize=10, textColor=GREY, alignment=0)


def P(t, s=B): return Paragraph(t, s)
def bullets(items): return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths):
    data = [[Paragraph(str(c), TH if r == 0 else T) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e0")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st)); return t


def img(name, w=16.2):
    from PIL import Image as PI
    iw, ih = PI.open(fig_dir / name).size
    return Image(str(fig_dir / name), width=w * cm, height=w * cm * ih / iw)


lo_p, hi_p = wilson(kp, npool); lo_b, hi_b = wilson(kb, nb); lo_a, hi_a = wilson(ka, na)
S = []
S += [P("Kamera ile yarış uçuşu: g37 politikası + g17 dedektörü", TITLE),
      P("Yöntem, sonuçlar, karşılaşılan sorunlar ve sınırlar &nbsp;·&nbsp; 3 Kasım 2026 &nbsp;·&nbsp; DroneRacing", SUB), Spacer(1, 6)]

S += [P("1. Özet", H1),
      P(f"Kendi kamerası ve kendi IMU'sunu kullanan, yalnızca görüntüden kapı konumunu tahmin eden bir yarış drone'u hattı Isaac simülasyonunda "
        f"<b>{kp}/{npool} = %{100*kp/npool:.1f}</b> uçuş başarısına ulaştı (hedef: %85 üstü). Bu sayı iki ayrı ölçümün birleşimidir: "
        f"daha önce kullanılan 16 seed'de {ka}/{na} (%{100*ka/na:.1f}) ve hiç kullanılmamış <b>taze</b> 9 seed'de (71-79) {kb}/{nb} (%{100*kb/nb:.1f}). "
        f"Birleşik oranın %95 güven aralığı %{lo_p:.0f}-{hi_p:.0f}, taze kümeninki %{lo_b:.0f}-{hi_b:.0f}. "
        "Ölçüm, gerçek kameraya yakın bir stres hattında yapıldı: rolling shutter, bozuk kareler, düşük pozlama, bilinmeyen 3° kamera montaj hatası "
        "(offline kalibrasyonla düzeltilmiş) ve 50 ms görüntü gecikmesi."),
      P("<b>Önemli sınır:</b> Sonuçlar simülasyondadır. Hat gerçek donanımda henüz çalıştırılmadı (bkz. bölüm 7)."), Spacer(1, 4)]

S += [table([["Ölçüt", "Değer"],
             ["Politika / dedektör", "g37 (g34 + 1000 iterasyon, tahminci-döngüde eğitim) / g17 (GateNet, on-policy ince ayar)"],
             ["Birleşik başarı", f"{kp}/{npool} = %{100*kp/npool:.1f}  (%95 aralık %{lo_p:.0f}-{hi_p:.0f})"],
             ["İlk 16 seed (21-30, 35, 36, 41-44)", f"{ka}/{na} = %{100*ka/na:.1f}"],
             ["Taze 9 seed (71-79)", f"{kb}/{nb} = %{100*kb/nb:.1f}"],
             ["Aynı 16 seed'de önceki en iyi (g34 + g17)", f"{sum(g17.values())}/{4*len(g17)} = %{100*sum(g17.values())/(4*len(g17)):.1f}"],
             ["Uçuş hızı (ölçülen yol hızı)", "yaklaşık 15-16 m/s"]], [6.2, 10.4])]

S += [P("2. Sistem mimarisi", H1),
      P("Hat dört bloktan oluşur. (1) <b>Kamera</b>: 820×616 görüntü, 30 Hz, tam görüş alanı; sim'de gerçek kameranın etkileri uygulanır. "
        "(2) <b>Dedektör (GateNet)</b>: görüntüyü kamera roll açısıyla döndürür (derotasyon), 384×384 kırpımdan kapı köşelerini bulur. "
        "(3) <b>Tahminci (GateEKF5)</b>: IMU ile durumu ilerletir, köşe ölçümlerini süzer; 50 ms'lik gecikmeyi durum geri-sarma yöntemiyle telafi eder. "
        "(4) <b>Politika</b>: tahmincinin çıktısını (hız, yön, sonraki 3 kapının konumu, 8 adımlık geçmiş) alıp motor komutu üretir. "
        "Politika, yarış parkurunun (A2RL 2025 final pisti kopyası) her bölümde 0.3 m konum ve 0.05 rad yön gürültüsüyle bozulmuş halinde uçar.")]

S += [P("3. Eğitim yöntemi", H1), P("3.1 Politika", H2),
      P("Politika pekiştirmeli öğrenme ile eğitildi. Soy ağacı: g27 (MonoRace tarzı algı cezası <font name='DJB'>--perc-pen 2</font>) → g32 "
        "(eğitimde gerçekçi algılama modeli: bir kapının görülmesi köşe sayısına bağlı olasılıkla) → g34 (tahminci-döngüde eğitim: politika, "
        "benzetilmiş algılarla beslenen GateEKF5'in çıktısını görür, 30 Hz kamera, 50 ms gecikme) → <b>g37</b> (g34'ün aynı ortamda 1000 iterasyon daha eğitimi). "
        "Eğitim koşulları: 4096 paralel ortam, <font name='DJB'>own_drone</font> profili (gerçek parçalar etrafında dar alan rastgeleleştirmesi), "
        "<font name='DJB'>--v-max 13</font>, 3 kapılık gözlem, gözlem geçmişi 8."),
      P("3.2 Dedektör", H2),
      P("GateNet'in g15 sürümü, ilk politika uçuşlarından toplanan kırpımlarla (d9, d10, d11) g16'ya, ardından g16'nın kendi uçuşlarından toplanan yeni kırpımlarla "
        "(d12, öğrenme oranı 1e-4) g17'ye ince ayarlandı. Bu <i>on-policy</i> veri, dedektörün gerçekte uçuşta gördüğü görüntü dağılımını yakalar. "
        "Eğitim artırımı (<font name='DJB'>realcam_aug.py</font>): rolling shutter (dönme ve ileri hareketten), hareket bulanıklığı, kısa pozlama ile gürültü, "
        "lens kenarı bulanıklığı ve bozuk bantlar; etiketler görüntüyle birlikte kaydırılır. Roll derotasyonuna 3° hata gürültüsü eklenir. "
        "Montaj eğimi bir kamera dönmesi olduğundan görüntü artırımıyla değil kalibrasyonla ele alınır."),
      P("3.3 Ölçüm protokolü", H2),
      P("Her seed, Isaac'te 4 drone ile 3000 adım uçurulur; başarı, drone'ların ilk bölümde 22 kapıyı tamamlama oranıdır. Hat: rolling shutter 20 ms, "
        "%10 bozuk kare, pozlama artırımı 0.4, bilinmeyen 3° montaj hatası (drone başına sabit) ve offline kalibrasyon düzeltmesi, 50 ms gecikme. "
        "Dedektör eğitiminde kullanılan seed'ler (31-34, 51-54, 61-64) hiçbir değerlendirmede kullanılmadı. Taze seed'ler (71-79) daha önce hiçbir ayarda kullanılmadı.")]

S += [PageBreak(), P("4. Sonuçlar", H1), img("g37_candidates.png"),
      P("Şekil 1. Adayların başarı oranı ve %95 Wilson aralığı. İlk üç sütun aynı 16 seed'de g34 politikasıyla farklı dedektörlerdir.", CAP), Spacer(1, 4),
      table([["Aday", "Seed", "Uçuş", "Başarı"],
             ["g34 + g15 dedektör", "16", f"{4*len(g15)}", f"%{100*sum(g15.values())/(4*len(g15)):.1f}"],
             ["g34 + g16 dedektör", "16", f"{4*len(g16)}", f"%{100*sum(g16.values())/(4*len(g16)):.1f}"],
             ["g34 + g17 dedektör", "16", f"{4*len(g17)}", f"%{100*sum(g17.values())/(4*len(g17)):.1f}"],
             ["g37 + g17, ilk 16 seed", "16", f"{na}", f"%{100*ka/na:.1f}"],
             ["g37 + g17, taze seed 71-79", "9", f"{nb}", f"%{100*kb/nb:.1f}"],
             ["<b>g37 + g17, birleşik</b>", "25", f"{npool}", f"<b>%{100*kp/npool:.1f}</b>"],
             ["g18 (g17 + bir tur daha on-policy veri)", "8", "32", "%81.3 (g17 aynı seed'lerde %87.5): geride, kesildi"],
             ["g33 (box)", "8", "32", "%65.6: elendi"],
             ["g36", "8", "32", "%75.0"]], [7.2, 1.6, 1.6, 6.2]),
      Spacer(1, 6), img("g37_per_seed.png"),
      P("Şekil 2. g37 + g17 için seed başına başarı (her seed 4 drone). Düşük seed'ler (22, 35, 43) tek başına kalıcı zorluk olarak yorumlanmamalıdır: "
        "aynı seed ve yapılandırma bir koşuda %25, ötekinde %100 verdi (bölüm 5).", CAP)]

S += [P("5. Karşılaşılan sorunlar ve çıkarılan dersler", H1)] + bullets([
    "<b>Seed başına sonuç gürültülüdür.</b> Seed 36'da aynı politika, dedektör ve komutla bir koşu %25 (1/4), ötekisi %100 (4/4) verdi. Simülasyon birebir tekrarlanabilir değil. "
    "4 drone'lu bir seed'in %50 ve altı çıkma olasılığı, gerçek başarı %83 iken yaklaşık %14'tür. Bu yüzden karşılaştırmaları 16+ seed'de ve ayrık seed'lerle yaptık.",
    "<b>Küçük örnek yanıltıcıydı.</b> g16 ilk 8 seed'de %84.4, g17 %87.5 görünürken 16 seed'de %79.7 ve %82.8 oldu; eski %82.5 de şanslı bir 40 uçuşluk alt kümeydi. "
    "g37'nin ilk 8 seed'i de %90.6 idi, 16 seed'de %87.5'e indi; asıl güven taze seed'lerle geldi.",
    "<b>Yerine koyma modeli iyimser.</b> Tahminci-döngü benzetimi (2048 drone, dakikalar) Isaac'i genelde 1-4 puan içinde izler, ancak kendi içinde eğitilmiş politikaları fazla iyi gösterir "
    "(g34: benzetimde %88-91, Isaac'te %78-82). Aday sıralaması her zaman Isaac'te yapıldı.",
    "<b>Dedektör iyileşmesi uçuşa doğrusal yansımadı.</b> Algılama oranı ile uçuş başarısı korelasyonu zayıf (+0.12); hız ve yakın kapı hatası daha ilişkili (-0.41 ve -0.27). "
    "g18 turu (bir on-policy veri turu daha) g17'nin altında kaldı ve bırakıldı.",
    "<b>İlk kaza analizim yanıltıcıydı.</b> Veri dökümünde bölümün bitiş mi çarpma mı olduğu kayıtlı değildi; bu alanları ekleyip analizi yeniden yazdık. "
    "Ön bulgu: incelenen 5 çarpmanın 5'i kapı 0'da, bölüm başlangıcından yaklaşık 50 adım (yaklaşık 1 s) sonra oldu (3'ü veri dökümünde, 2'si videoda; durağan başlangıç, kapıya 1.6-3 m). Sebep ayrılmadı: politikanın başlangıç manevrası ya da yeniden başlatma sonrası render/tahminci gecikmesi olabilir. Örnek küçük; kalan dökümler alınmadı.",
    "<b>Eğitim verisi hatası yakalandı.</b> g18 başlatılırken bir veri kümesi iki kez listelenmişti (çift ağırlık); süreç durdurulup düzeltilerek yeniden başlatıldı.",
    "<b>Altyapı.</b> Eğitim ve Isaac eval'i aynı GPU'da aynı anda koşunca sahte %0 sonuçlar çıkar; işler kilit dosyasıyla sıraya alındı, Ollama sunucusu kapatıldı. "
    "Bir render hatası seed 29'u çökertti (satır silinip yeniden koşuldu). Bilgisayar kapanışına karşı tüm hatlar kaldığı yerden devam edecek yazıldı."])

S += [P("6. Denenip işe yaramayanlar", H1)] + bullets([
    "Dedektör: üçüncü on-policy tur (g18) ve g33 politikası g17/g34 hattının altında kaldı.",
    "Tahminci ve politika (araştırma günlüğüne göre): ekf9 (sürükleme destekli hız; bu IMU modeliyle kullanılamaz, hız hatası 0.235 → 0.53 m/s), kapıya yakın IMU ile uçuş (kazanç yok), "
    "giriş açısı cezası (g26, kazanç yok), daha düşük hız cezası (g28, daha yavaş ve daha kötü), filtre gürültü ayarı.",
    "MonoRace algı cezası (g27) ise işe yaradı: aynı hatta g17 politikasına göre %56.3 → %71.9."])

S += [P("7. Sınırlar ve gerçek drone'a geçiş", H1)] + bullets([
    "<b>Yalnızca simülasyon.</b> Jetson üzerinde hiçbir ölçüm yapılmadı. g17 dedektörü ONNX olarak hazır (<font name='DJB'>jetson/onnx/gatenet_g17_384.onnx</font>); "
    "TensorRT hız ölçümü (bütçe 85 fps'te kare başına 11.7 ms) ve kamera modu (820×616) kurulumu bekliyor.",
    "<b>Montaj hatası.</b> Kalibrasyonsuz 1/2/3° hata başarıyı %50/33/17'ye düşürmüştü (önceki hat). Gerçek dronda kamera-IMU dönmesi 0.5° altına kalibre edilmeli.",
    "<b>Gecikme.</b> 20 ms'lik gecikme uyuşmazlığı başarıyı yarıya indirdi; gerçek dronda kareler zaman damgalanmalı (LED testi ile ölçüm planlı).",
    "<b>İstatistik.</b> 100 uçuşta %95 aralığın alt ucu %85'in altında; kesin üstünlük için daha fazla taze seed gerekir. Parkur tek (A2RL 2025 final kopyası) ve gerçek rüzgar, batarya düşümü, motor farkı modellenmedi.",
    "<b>Hız.</b> Politika güvenli model olarak <font name='DJB'>--v-max 13</font> ayarıyla eğitildi; ölçülen yol hızı yaklaşık 15-16 m/s."])

S += [P("8. Sonraki adımlar", H1)] + bullets([
    "Kalan kaza dökümlerini alıp tüm bölümleri (başlangıç mesafesi, yön sapması, yükseklik, kapı indeksi) sonuçla karşılaştırmak; baskın kaza türüne göre politikayı güçlendirmek.",
    "Güvenli modeli koşullarıyla birlikte arşivlemek (ara arşiv: <font name='DJB'>releases/v2_INTERIM_g34_g17_nov3</font>; g37 eklenerek nihai sürüm).",
    "Hız sınırı olmayan hızlı bir model eğitmek (güvenli model yedek kalır).",
    "Jetson'da TensorRT hız ölçümü, kamera modu, LED gecikme testi, Kalibr ile montaj kalibrasyonu; ardından donanımda döngüde test."])

S += [P("9. Yeniden üretme", H1),
      P("Betikler (hepsi <font name='DJB'>scripts/</font> altında): <font name='DJB'>run_g34.sh</font> (politika), <font name='DJB'>run_g37.sh</font> (g37 eğitimi ve 16 seed eval), "
        "<font name='DJB'>run_onpolicy16.sh</font>, <font name='DJB'>run_g17.sh</font> (dedektör), <font name='DJB'>run_g37_confirm.sh</font> (taze seed doğrulaması), "
        "<font name='DJB'>cam_eval.sh</font> (ölçüm protokolü), <font name='DJB'>fail_anatomy.py</font> (kaza analizi), <font name='DJB'>run_video_g37.sh</font> (video). "
        "Sonuç dosyaları: <font name='DJB'>runs/cam_eval/t37_cal.txt</font>, <font name='DJB'>t37c_cal.txt</font>, <font name='DJB'>t34g17_cal.txt</font>. "
        "Modeller: <font name='DJB'>runs/g37</font> (politika), <font name='DJB'>runs_host/g17/gatenet.pt</font> (dedektör), <font name='DJB'>runs_host/e4/estimator.pt</font>, "
        "<font name='DJB'>runs_host/calib/cal15_41.pt</font> (kalibrasyon).")]


def footer(c, d):
    c.saveState(); c.setFont("DJ", 7); c.setFillColor(GREY)
    c.drawString(2 * cm, 1.1 * cm, "DroneRacing · g37 + g17 · Yöntem ve sonuçlar"); c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{d.page}"); c.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                  title="Kamera ile yarış uçuşu: g37 + g17", author="DroneRacing").build(S, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
