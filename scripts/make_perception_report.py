"""Builds the perception-error report PDF (Turkish): problems measured, non-training solutions tried, results.

Run: python3 scripts/make_perception_report.py
All numbers come from the vault notes droneracing-perception-80-goal, droneracing-vision-drop-analysis,
droneracing-literature-comparison, droneracing-a2rl-plan and from handover.md / runs logs.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

root = Path(__file__).resolve().parents[1]
out = root / "reports"
figdir = out / "_fig"
figdir.mkdir(parents=True, exist_ok=True)

FD = "/usr/share/fonts/noto/"
pdfmetrics.registerFont(TTFont("N", FD + "NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NB", FD + "NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NI", FD + "NotoSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DJ", "/usr/share/fonts/TTF/DejaVuSans.ttf"))
pdfmetrics.registerFontFamily("N", normal="N", bold="NB", italic="NI", boldItalic="NB")

NAVY = colors.HexColor("#12304f")
H1 = ParagraphStyle("h1", fontName="NB", fontSize=19, leading=24, spaceAfter=4, textColor=NAVY)
H2 = ParagraphStyle("h2", fontName="NB", fontSize=13.5, leading=18, spaceBefore=10, spaceAfter=4, textColor=NAVY, keepWithNext=1)
B = ParagraphStyle("b", fontName="N", fontSize=10, leading=14.5, spaceAfter=4)
SM = ParagraphStyle("sm", fontName="N", fontSize=8.6, leading=11.8, textColor=colors.HexColor("#444444"), spaceAfter=3)
BL = ParagraphStyle("bl", parent=B, leftIndent=12, bulletIndent=2)
TC = ParagraphStyle("tc", fontName="N", fontSize=8.4, leading=11)
TB = ParagraphStyle("tb", fontName="NB", fontSize=8.4, leading=11, textColor=colors.white)

GREEN, AMBER, RED, GREY = "#d9f0d8", "#fdeccb", "#f6d4d0", "#e6e6e6"


import re  # noqa: E402


def fx(t):
    return re.sub(r"[→≈≥≤]", lambda m: f'<font name="DJ">{m.group(0)}</font>', str(t))


def P(t, s=B):
    return Paragraph(fx(t), s)


def bullets(items):
    return [Paragraph(fx(t), BL, bulletText="•") for t in items]


def table(rows, widths, verdict_col=None):
    data = [[Paragraph(fx(c), TB if i == 0 else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79"))]
    for i in range(1, len(rows)):
        if verdict_col is None and i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef3f8")))
        if verdict_col is not None:
            v = str(rows[i][verdict_col])
            c = GREEN if v.startswith("İşe yaradı") else RED if v.startswith("Yaramadı") else GREY if v.startswith(("Denenmedi", "Bekliyor", "Geçersiz")) else AMBER
            st.append(("BACKGROUND", (verdict_col, i), (verdict_col, i), colors.HexColor(c)))
    t.setStyle(TableStyle(st))
    return t


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- figures
def fig_delay(path):
    bins = ["0-3 m", "3-6 m", "6-10 m"]
    nod, ign, comp = [0.38, 0.26, 0.23], [1.87, 1.81, 1.46], [0.38, 0.30, 0.25]
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    w = 0.26
    for k, (v, c, l) in enumerate(((nod, "#9bb3c9", "Gecikme yok (referans)"), (ign, "#c0392b", "50 ms gecikme, yok sayıldı"), (comp, "#1f6a3a", "50 ms gecikme, telafi edildi"))):
        xs = [i + (k - 1) * w for i in range(3)]
        ax.bar(xs, v, w, color=c, label=l)
        for x, y in zip(xs, v):
            ax.text(x, y + 0.03, f"{y:.2f}".replace(".", ","), ha="center", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(bins, fontsize=8.5)
    ax.set_ylabel("Kapı konum hatası, p90 (m)", fontsize=8.5)
    ax.set_ylim(0, 2.5)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("Görüntü gecikmesi (tekrar oynatma, uçuş a25_41, 50 ms)", fontsize=9.5, color="#12304f")
    clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_iso(path):
    labels = ["100 Hz tespit\ngecikme yok", "30 Hz tespit\ngecikme yok", "30 Hz + 50 ms\ngecikme yok sayıldı", "30 Hz + 50 ms\ngecikme telafi (EKF5)"]
    vals = [83.3, 66.7, 0.0, 70.8]
    cols = ["#9bb3c9", "#7fa3c4", "#c0392b", "#1f6a3a"]
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=200)
    ax.bar(range(4), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.8, f"%{v:.1f}".replace(".", ","), ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Başarı %", fontsize=8.5)
    ax.set_title("Kendi kamera hattı, tek tohum (21), 24 dron", fontsize=9.5, color="#12304f")
    clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(path):
    names = ["Kapı konum gürültüsü", "  bunun AR(1) korelasyonu", "Kare kaybı (dropout)", "Durum (hız, tutum)", "Gecikme", "Kapı normal gürültüsü", "Harita yanlılığı", "Rüzgâr", "IMU"]
    vals = [11, 9, 4, 4, 2, 0, 0, 0, 0]
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=200)
    ys = list(range(len(names)))[::-1]
    ax.barh(ys, vals, color=["#c0392b", "#c0392b", "#e08e45", "#e08e45", "#e0c445", "#9bb3c9", "#9bb3c9", "#9bb3c9", "#9bb3c9"])
    for y, v in zip(ys, vals):
        ax.text(v + 0.2, y, f"+{v}", va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Başarı artışı (puan), ilgili hata sıfırlanınca (taban %61,8)", fontsize=8)
    ax.set_title("Torch ablasyonu: hangi algı hatası ne kadar başarı yiyor", fontsize=9.5, color="#12304f")
    clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_perm(path):
    seeds = ["21", "22", "23", "24", "41", "42", "43", "44"]
    a = [83.3, 66.7, 62.5, 66.7, 70.8, 70.8, 79.2, 75.0]
    b = [87.5, 58.3, 79.2, 83.3, 75.0, 95.8, 91.7, 79.2]
    fig, ax = plt.subplots(figsize=(6.6, 2.9), dpi=200)
    w = 0.38
    ax.bar([i - w / 2 for i in range(8)], a, w, color="#9bb3c9", label="Düzeltme yok (ort. %71,9)")
    ax.bar([i + w / 2 for i in range(8)], b, w, color="#1f6a3a", label="aided_perm (ort. %81,3)")
    ax.set_xticks(range(8))
    ax.set_xticklabels(seeds, fontsize=8)
    ax.set_xlabel("Tohum", fontsize=8.5)
    ax.set_ylabel("Başarı %", fontsize=8.5)
    ax.set_ylim(0, 125)
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", ncol=2)
    ax.set_title("Köşe sırası düzeltmesi, 2025 pist kopyası, 192 dron", fontsize=9.5, color="#12304f")
    clean(ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


f1, f2, f3, f4 = (figdir / n for n in ("pr_delay.png", "pr_iso.png", "pr_ablation.png", "pr_perm.png"))
fig_delay(f1)
fig_iso(f2)
fig_ablation(f3)
fig_perm(f4)

# ---------------------------------------------------------------- content
S = []
S.append(P("Algı Hataları: Ne Yaptık, Ne Bozuldu, Hangi Çözümü Denedik", H1))
S.append(P("DroneRacing — kamera + IMU ile RL yarış dronu. Kapsam: <b>eğitim (politika yeniden eğitimi) dışındaki</b> tüm algı hatası çözümleri ve sonuçları. "
           "Veriler 19–25 Ekim 2026 deney kayıtlarından alındı.", SM))

S.append(P("1. Özet", H2))
S += bullets([
    "<b>Sorun tek cümleyle:</b> politika kapının merkezini yanlış biliyor ve <b>yanlış bildiğini bilmiyor</b>. Kenara çarpan 32 vakanın <b>tamamında</b> "
    "politika kapının içinden geçtiğini sanıyordu; algı hatası 0,55–0,90 m idi (ortalama 0,66 m, kapı açıklığı 1,5 m, dron gövdesi ~0,3 m).",
    "<b>Kusursuz algıyla</b> aynı politika %100 (kendi dron profilinde, Torch, 4096 dron) başarıyor. Yani kayıp politikadan değil algıdan geliyor.",
    "Hata <b>gürültü değil, kalıcı kayma ve ağır kuyruk</b>: kapı konum hatasının zamansal korelasyonu AR(1) = 0,93 (~14 adım). "
    "Torch ablasyonunda bu korelasyonu sıfırlamak +9 puan veriyor.",
    "Hatanın en büyük kaynağı <b>kapıya yakınken kapının görüş alanından çıkması</b>: 1,5 m içinde yalnızca %11 adımda dört köşe de görünüyor.",
    "<b>İşe yarayan eğitim dışı çözümler (ölçülen kazanç):</b> harita tabanlı EKF (%2–8 → %12–17), GateNet g3 geniş tutum verisi (~+17 puan), "
    "köşe sırası düzeltmesi <i>aided_perm</i> (%71,9 → %81,3, 8 tohum), görüntü gecikmesi telafisi <i>EKF5</i> (%0 → %70,8; hata 1,87 → 0,38 m).",
    "<b>İşe yaramayanlar:</b> sıkı köşe yeniden izdüşüm filtreleri (EKF2/3/4), chi² eşiği gevşetme, doğrudan PnP, decoy maskesi, 155° / 384 kırpma, köşe ısı haritası detektörü (kapalı döngüde).",
    "<b>Açık hedef:</b> sağlam kamera hattında %80–85. Kendi kamera hattı (30 Hz, 50 ms gecikme) şu an <b>%63,5–68,8</b>; kuyruk hatası (p90 0,5–0,9 m) hâlâ duruyor.",
])

S.append(P("2. Hat ve test düzeni", H2))
S.append(P("Hat: kamera → GateNet (kapı köşeleri) → dönme destekli PnP → yaklaşık haritalı EKF → durum tabanlı PPO politikası (itki + gövde hızları). "
           "Harita hatası ±0,25 m / ±5° kabul edildi (MonoRace tarzı). Test: Isaac Sim fiziği + gerçek render edilen kamera karesi, tohum başına 24 dron. "
           "<b>Dikkat:</b> 24 dronluk bir hücrenin standart hatası ~±10 puan; 4 tohumda ~±5 puan. Bu rapordaki birçok fark gürültü sınırında ve bu işaretlendi.", B))

S.append(P("3. Hatayı ölçtük: ne biliyoruz?", H2))
S.append(P("Önerilen 1. adım (hatayı ölç, modelle) büyük ölçüde yapıldı. Sorulara bizim ölçümlerimizle cevap:", B))
S.append(table([
    ["Soru", "Ölçülen cevap", "Kaynak"],
    ["Hata kapıya mesafeyle nasıl değişiyor?",
     "Kapalı döngü, sonraki kapının düzlem içi hatası: medyan 4–15 cm, <b>p90 0,30–0,57 m</b>; en kötü 0–3 m (ort. 0,21, p90 0,48 m). "
     "Kendi kamera hattı (30 Hz, telafili): p90 0,52 / 0,86 / 0,55 m (0–2 / 2–4 / 4–8 m). 2025 pist kopyasında 2–5 m p90 0,73–0,83 m; split-S alt kapısında 1,6–1,7 m.",
     "ekf_gate_err, cam_eval, rep_x21"],
    ["Son tespitten beri geçen süre / tespit yokken?",
     "Tespit varken filtre doğrudan PnP kadar doğru (ort. 0,12–0,19 m). Tespit yokken: <b>0,35 m</b> (0–3 m, adımların %51'i), <b>0,79 m</b> (3–6 m, %21), 6–15 m'de >1 m. "
     "Hata tespit eksikliğinden geliyor, güncelleme kuralından değil. (Eski GRU kestirici ise sıfırlamadan beri sürekli kayıyordu: 2 s'de 1,2 m, 8 s'de 9 m.)",
     "ekf_replay, E3"],
    ["Kapı görüş alanından çıkınca ne oluyor?",
     "1,5 m içinde yalnızca <b>%11 adımda 4 köşe görünür</b>. Kısmi kapı tespiti %4–14 (GateNet2'den önce). Kör dönemler (2'den az köşe): 1,5–10 m aralığında adımların %3–8'i, hata 0,8–1,7 m. "
     "Tespit oranı: 0–3 m %85, 3–6 m %96, 6–10 m %97, 10–20 m %62. Kapıya yakın köşe hatası: 60–400 px genişlikte ort. 8,4 px (p90 23 px).",
     "diag_det, ekf_replay"],
    ["Hata kare kare titriyor mu, kayıyor mu?",
     "<b>Kayıyor.</b> AR(1) korelasyonu 0,93 (~14 adım). Hatalar ağır kuyruklu: medyan 0,12–0,27 m, p99 2,6–4,6 m. Kapı başına kalıcı kuyruk var. "
     "Torch'ta korelasyonu sıfırlamak +9 puan, konum gürültüsünü ×0,5 yapmak %60,8 → %70,9.",
     "diag_noise, E5"],
], [34, 112, 28]))
S.append(Spacer(1, 4))
S.append(Image(str(f3), width=140 * mm, height=140 * mm * 3.1 / 6.6))
S.append(P("Şekil 1. Tek bir hata grubu sıfırlanınca başarı artışı (Torch, g6, 4096 dron). Kayıp kapı konumu hatasında ve onun zamansal korelasyonunda.", SM))

S.append(P("Önce doğrulama: çarpma anatomisi", H2))
S.append(P("Öneri: her çarpmayı ikiye ayırın — tahmin mi yanlıştı, yoksa tahmin doğruyken politika mı kenara gitti? Bunu <i>scripts/crash_anatomy.py</i> ile yaptık "
           "(g15, 2025 kopyası, sağlam kamera hattı, 4 uçuş dökümü):", B))
S.append(table([
    ["Büyüklük (çerçeve çarpması vakaları, ort.)", "Değer"],
    ["Kapı merkezinin gerçek kaçırma mesafesi", "0,79 m"],
    ["Politikanın inandığı kaçırma mesafesi", "0,16 m"],
    ["İnanç hatası (kapı merkezi tahmin hatası)", "0,66 m (p25 0,55; p50 0,65; p75 0,77; p90 0,87)"],
    ["İnanç hatası 0,2 m altında olan vakalar", "<b>%0</b>"],
    ["Politikanın kendini kapı içinde sandığı vakalar", "<b>%100</b>"],
    ["Yaklaşma anındaki hız", "13,3 m/s"],
], [100, 74]))
S.append(P("Okuma: 113 kapıya yakın bitişin 32'si çerçeve çarpması; hepsinde politika tahmin ettiği merkeze nişan aldı ve tahmin yanlıştı. "
           "Kapıyı küçültmek 0,25 m pay ekler; inanç hatası 0,66 m olduğundan pay yetmez. "
           "<b>Sınırlar:</b> tek politika (g15), sağlam kamera hattı, 32 olay; en yakın noktadaki ortalama yaklaşma açısı büyük (60°), bir kısmı yandan veya geçişten sonraki temas olabilir. "
           "Bekleyen kontrol: g17 ile Isaac'te gerçek gözlem (<i>oracle_isaac_g17</i>, algı dışı kayıp payı) ve kendi kamera dökümleri için aynı analiz — sonuçları bu rapora girmedi. "
           "Ek destek: kusursuz algıyla kendi dron profilinde %100 (0 çarpma).", B))

S.append(PageBreak())
S.append(P("4. Önce ölçüm hataları (algı sayılarını bozan sorunlar)", H2))
S.append(P("Bunlar algı çözümü değil; ama algı sonuçlarını haftalarca yanlış okuttu. Aynı hatayı tekrarlamamak için listelendi.", B))
S.append(table([
    ["Sorun", "Etki", "Düzeltme"],
    ["Kamera prim'i dron gövdesini izlemiyordu (FabricFrameView dünya konumunu yenilemedi); kamera doğuş noktasında boş gri sahne çizdi.",
     "21 Ekim ~17:50'den sonraki tüm kamera değerlendirmeleri %0. Sürücü, Docker, VRAM, pist karışımı boşuna suçlandı.", "<i>VisionRig.follow()</i> her adım kamera dünya konumunu yazıyor. O dönemin sayıları geçersiz."],
    ["Değerlendirme eğitimle veya başka GPU kullanıcısıyla aynı GPU'da koştu.", "%60–100 boş kare, sahte %0 sonuçlar ve yanlış 'zor başlangıç' açıklaması.", "GPU boşken koş (<1,5 GB kullanım); boş/donuk kare sayaçları eklendi; ısınma başarısızsa değerlendirme iptal."],
    ["PnP, 155° kamera için 130° odak uzaklığıyla çalıştı.", "İlk crop384 değerlendirmesi geçersiz.", "<i>hfov_deg=a.hfov</i> geçiriliyor."],
    ["Veri toplamada kapılar reset öncesi konumda çizildi.", "Eğitim etiketleri kaymış.", "<i>--sync 1</i> varsayılan."],
    ["Kapı geçişi dronu nokta kabul etti.", "Başarı iyimser: 58,5 → 27,4 (Torch). Algı hatasının gerçek maliyeti küçümsendi.", "<i>--gate-body</i> (yarıçap 0,155–0,195 m)."],
], [58, 62, 54]))

S.append(P("5. Eğitim dışı algı çözümleri ve sonuçları", H2))
S.append(P("Renk: yeşil = işe yaradı, sarı = kısmen veya gürültü sınırında, kırmızı = yaramadı, gri = denenmedi/bekliyor.", SM))
rows = [["#", "Sorun / ölçülen neden", "Çözüm", "Sonuç", "Hüküm"]]
rows += [
    ["1", "GRU kestirici sürekli kayıyordu: hız hatası 2,6–3,2 m/s, tutum 4,2°; gizli haritayla başarı %2–8.",
     "Harita tabanlı EKF (<i>ekf.py</i>): IMU + PnP konum/normal güncellemesi, Mahalanobis ilişkilendirme.",
     "Hız hatası 0,9 m/s, tutum 2,7–2,9°. Başarı %12–17 (politika hâlâ 0,25 m üstü hataya dayanıksızdı).", "İşe yaradı (ilk adım)"],
    ["2", "PnP uzak kapıda kötü: 6–10 m medyan hata 0,96 m.", "Dönme destekli PnP (filtre tutumu bilinen, köşe ışınlarında en küçük kareler).",
     "Medyan hata: 6–10 m 0,96 → 0,31 m; 10–20 m 1,74 → 0,57; 3–6 m 0,29 → 0,16. Tutum hatası 5°'ye duyarsız.", "İşe yaradı"],
    ["3", "GateNet eğitim verisi gövde eğimi 0–50° (kapalı döngüde kareler %63 50–90°, %16 ters).", "GateNet g3: politikasız rastgele geniş tutum verisi d2 (50k kare, %12 ters).",
     "PnP medyan hata 0–6 m 0,12–0,16 m (g2: 0,17–0,23). g3 ≈ +17 puan; boru hattı toplamı %64,3 (168 bölüm).", "İşe yaradı"],
    ["4", "Kısmi kapıda köşe tespiti: 1/2/3/4 köşe görünürken geri çağırma %3,7/48/79/90.", "GateNet2: görünen her köşe için ısı haritası (merkez görüntüde olmak zorunda değil).",
     "Geri çağırma %68/85/84/91. Kapalı döngü: g3 ortalama %62,5, GateNet2 %62,5 — fark yok.", "Kısmen (detektör iyileşti, başarı değişmedi)"],
    ["5", "Gevşek bağlı güncelleme; köşe yeniden izdüşüm hatası doğrudan kullanılabilir (ADR-VINS).", "EKF2 (sıkı), EKF3 (tek köşe 2B kalıntı + Huber + gate offset), EKF4 (köşe sayısına bağlı gürültü).",
     "Tekrar oynatmada EKF2 tabandan iyi değil (0–3 m 0,32 m). EKF4 %58,3/58,3 (EKF3 ile aynı). Sentetik tepe üst sınırı: 0,25 → 0,13 m, ama gerçek detektörle kazanç yok.", "Yaramadı"],
    ["6", "Filtre sert eşik yüzünden ölçümü atıyor olabilir.", "chi² 14 → 40 gevşetme (tekrar oynatma).", "Daha kötü: 0–3 m 0,40 m.", "Yaramadı"],
    ["7", "Filtre PnP'den iyi olabilir mi?", "PnP'yi doğrudan politikaya ver.", "Kazanç yok (tespit varken filtre PnP kadar doğru).", "Yaramadı"],
    ["8", "Kapı offset EMA (<i>kappa</i>) çok yavaş: 0,6.", "kappa 0,6 → 0,3 → 0,15.", "Yakın kapı p90: 0,48–0,66 → 0,37–0,42 → 0,28–0,39 m. Kamera başarısı 54,2 / 58,3 / 51,4 (gürültü içinde).", "Kısmen (hata azaldı, başarı değişmedi)"],
    ["9", "Kamera 130°, 256×192 (~2 px/°); MonoRace 155°, 820×616 (~5 px/°).", "155° / 616×820 / 384 kırpma veri seti d5, GateNet g10.",
     "3–10 m tespit %89–93, medyan hata 0,14–0,18 m; yakında 0,30 m. 12 tohum ort. %47,9 (iki çöken tohum çıkınca %57,5) = 130° hattıyla aynı.", "Yaramadı (başarıda kazanç yok)"],
    ["10", "Geçilen kapının sahte tespitleri (decoy) ilişkilendirmeyi bozuyor olabilir.", "<i>--decoy-prev</i>: geçilen kapının tespitlerini at.", "Split-S kapı hatası 1,7–2,8 m, başarı %25; kazanç yok → sorun ilişkilendirme değil.", "Yaramadı"],
    ["11", "<b>Split-S alt kapısı</b>: kamera yuvarlanınca veya kapı aynalı görününce PnP köşe sırası yanlış eşleşiyor; tek tespit medyan hata 1,81 m.",
     "<i>aided_perm</i>: 4 dönme × 2 ayna = 8 köşe sırası dene, en düşük yeniden izdüşüm hatalısını seç.",
     "Tek tespit medyan 1,81 → 0,06 m (<0,5 m olanlar %44 → %82). Kamera hattı 8 tohum, 192 dron: <b>%71,9 → %81,3</b>, 7/8 tohum iyi. Hız 15,7 m/s sabit.", "İşe yaradı (en büyük tek kazanç)"],
    ["12", "<b>Görüntü gecikmesi</b> (kendi kamera: 30 Hz + ~50 ms): yok sayılınca hata 1,5–2,6 m.",
     "GateEKF5 (<i>ekf5.py</i>): son adımların filtre durumunu sakla, çekim anına geri sar, güncelle, düzeltmeyi bugüne taşı (MonoRace yöntemi).",
     "4/4 uçuşta gecikmesiz doğruluk geri geldi (0–3 m p90: 1,87–2,43 → 0,38 m). Kapalı döngü (30 Hz, tohum 21): %0 → <b>%70,8</b>. 4 tohum: ort. %63,5. "
     "<b>Gerçek gecikme gerekli:</b> 20 ms yanlış → 2,5× hata.", "İşe yaradı"],
    ["13", "Filtre ayarları gerçek PnP'den kötümser (PnP medyan 0,07 m).", "chi² 9, sig_aided 0,07 + 0,02 d, kappa 0,2 (tekrar oynatma taraması, ayrı uçuşta doğrulandı).",
     "Tekrar oynatma p90 0,34 → 0,22 m. Kapalı döngü (4 tohum): %63,5 → %68,8 (+5,2 puan, 1 standart hata içinde).", "Kısmen"],
    ["14", "Kendi kamera: 30 Hz (organizatör 90–100 Hz).", "60 Hz modu (donanım sınırı: stok Jetson sürücüsünde 60 FPS dikey görüş alanını kırpıyor).",
     "4 tohum: 30 Hz %63,5 → 60 Hz %71,9 (+8,4). 100 Hz tespit, tohum 21–22: %83,3 / %62,5 (4 tohumlu kontrol sürüyor).", "Kısmen (donanımla sınırlı)"],
]
S.append(table(rows, [8, 39, 42, 61, 24], verdict_col=4))
S.append(Spacer(1, 6))
S.append(Image(str(f4), width=135 * mm, height=135 * mm * 2.9 / 6.6))
S.append(P("Şekil 2. En büyük tek algı kazancı: köşe sırası düzeltmesi (aided_perm). 7 tohumda iyileşme, ortalama +9,4 puan.", SM))

S.append(P("6. Görüntü gecikmesi ve kamera hızı ayrıntısı", H2))
S.append(Image(str(f1), width=135 * mm, height=135 * mm * 3.0 / 6.6))
S.append(P("Şekil 3. Gecikme telafisi (GateEKF5) 50 ms'de gecikmesiz hatayı geri getiriyor. Aynı sonuç 4 uçuşta doğrulandı (a25_41…44).", SM))
S.append(table([
    ["Gecikme yok sayıldı mı?", "0–3 m p90", "3–6 m p90", "6–10 m p90"],
    ["Filtrede 0 adım (yok sayıldı)", "1,87 m", "1,81 m", "1,46 m"],
    ["3 adım (20 ms kısa)", "1,13 m", "0,84 m", "0,73 m"],
    ["<b>5 adım (doğru)</b>", "<b>0,38 m</b>", "<b>0,30 m</b>", "<b>0,25 m</b>"],
    ["7 adım (20 ms uzun)", "1,00 m", "0,84 m", "0,69 m"],
], [60, 38, 38, 38]))
S.append(P("Gerçek dronda her kareyi IMU ile <b>aynı saatle, çekim anında</b> damgala (Jetson donanım zaman damgası) ve gecikmeyi LED testiyle ölç. "
           "Sabit varsayılan gecikme yetmez. Olası iyileştirme: her kare için birkaç gecikme dene, en küçük yenilik (innovation) olanı seç — uygulanmadı.", B))
S.append(Image(str(f2), width=135 * mm, height=135 * mm * 3.0 / 6.6))
S.append(P("Şekil 4. Kendi kamera hattı izolasyonu (g17, GateNet g5, tek tohum, 24 dron, GPU boş): profil, eğim ve görüş açısı sorun değil; 30 Hz + yok sayılan gecikme hattı kırıyor.", SM))
S.append(P("Kendi kamera hattı çoklu tohum (30 Hz, 50 ms telafili, 4 tohum × 24 dron, GateNet g5): taban %63,5, ayarlı filtre %68,8, 60 Hz %71,9; "
           "aynı tohumlarda sağlam kamera hattı referansı %77,1. Hedef %80–85 <b>ulaşılmadı</b>. Torch modeli kendi kamera için hattı ~20 puan fazla gösteriyor "
           "(Torch %83,3, kamera %62,5): kendi kameranın hata modeli Torch simülatöründe yok.", B))

S.append(P("7. Eğitim tarafı denemeleri (kapsam dışı, yalnızca bağlam için)", H2))
S.append(P("Önerinin 2–4. adımlarına (hatalı gözlemle eğitim, belirsizlik girdisi, kapı küçültme) yakın denemeler vardı. Eğitim tarafı olduğu için ayrıntılı rapora alınmadı; sonuçlar kısaca:", B))
S.append(table([
    ["Deneme", "Sonuç"],
    ["Gürültü profili kamera hattına uyduruldu (g6, sensör ölçeği 0,6)", "+8 puan (kamera ~%58–62); en iyi gerçek boyutlu kapı politikası"],
    ["Swift tarzı artık-hata bankası ile eğitim (g12)", "Kötü: Torch parametrik %39,5 (g6 %64,5), kamera %43,8 (g6 %63,5). Banka yalnızca değerlendirme için iyi bir vekil (Torch %57,0 ≈ kamera %59–63)"],
    ["Sürekli kapı başına ağır kuyruk gürültüsüyle eğitim", "Çok sert: g6 %18'e düşüyor"],
    ["Kapı küçültme (g3, 0,25 m), algı cezası (g2), son tespit yaşı girdisi (g4), ödül varyantları", "Hiçbiri gürültü sınırını aşmadı (Torch %27,6–33,2 hepsi)"],
    ["Politika gecikmesi 0–2 adım (g18)", "Kamera hattı %62,5 (g17 %63,5): fark yok"],
], [80, 94]))
S.append(P("Çıkarım: bu denemeler önerinin 2–4. adımlarını destekleyen veya çürüten kanıt değildir; hatanın <b>zamansal yapısı</b> doğru modellenmeden eğitim kazanç vermiyor (g12, kuyruk gürültüsü). "
           "Hata kaynağını küçültmek (algı tarafı) eğitimden daha çok kazandırdı.", B))

S.append(P("8. Henüz denenmeyenler ve bu rapora göre sıralı öneriler", H2))
S.append(table([
    ["Sıra", "Çözüm", "Durum / neden"],
    ["1", "Kapıya yakın kör dönemler: kısmi kapı ve görüntü dışı köşe için daha iyi detektör (GateSeg: MonoRace maske + çizgi uydurma QuAdGate; KAIST tarzı YOLOv8-pose anahtar nokta).",
     "Denenmedi: kod hazır (<i>yolopose.py</i>), zincir 15/16 GPU dolu olduğu için hiç koşmadı. Neden: 1,5 m içinde yalnızca %11 adımda 4 köşe görünüyor."],
    ["2", "Gerçek kovaryansı güvenilirlik için doğrula; kapı başına öğrenilmiş tespit kovaryansı (log-varyans başı) EKF'ye; son tespitten beri geçen süre ve kovaryans politika girdisi.",
     "Denenmedi (girdi olarak politika eğitimi gerekir; EKF'de kovaryans doğrulaması eğitimsiz yapılabilir). Torch'ta yaş girdisi tek başına fayda vermedi (g4/g5)."],
    ["3", "Chi² eşiği yerine dayanıklı yeniden ağırlıklandırma (Huber/Student-t).", "Denenmedi (chi² gevşetme kötü çıktı, ama yumuşak ağırlık farklı)."],
    ["4", "Kamera-IMU ve kamera açısı kalibrasyonu: gerçek dronda Kalibr; MonoRace çevrimdışı ~1° iyileştirme; SkyDreamer ±5° dış parametre rastgeleleştirme.",
     "Denenmedi (simülasyonda yalnızca d5 verisinde 3° yuva hatası var, değerlendirmede yok). Öneri: ±1–3° tilt/yaw hatasını değerlendirmeye ekle, etkiyi ölç."],
    ["5", "Rolling shutter, hareket bulanıklığı, bozuk kare simülasyonu (hızlı dönüşte görüntü eğilmesi).", "Denenmedi: kamera değerlendirmesinde yok. Gerçek kamerada hata kaynağı olabilir."],
    ["6", "Gecikmeyi kare başına tara (en küçük yenilik) + donanım zaman damgası + LED gecikme testi.", "Denenmedi; 20 ms uyumsuzluk hatayı 2,5× tutuyor, bu yüzden gerçek dronda kritik."],
    ["7", "Çevrimdışı faktör grafiği yumuşatıcı (ADR-FGO: 0,060 m vs filtre 0,143 m) gerçek uçuş etiketleri için.", "Denenmedi; gerçek uçuş verisi gelince anlamlı."],
    ["8", "Güvenlik yedeği: belirsizlik eşiğini aşınca yavaş ve güvenli politikaya geç.", "Yok. Güvenli (g15) ve hızlı (g16) model stratejisi var, otomatik geçiş yok. Eşik için EKF kovaryansı (madde 2) gerekli."],
    ["9", "Hafif VIO hız kaynağı.", "Düşük öncelik: kendi ablasyonumuzda kusursuz hız tek başına kazanç vermedi."],
], [14, 86, 74]))
S.append(P("<b>Beklenen en yüksek kazanç, önerilen sırayla:</b> (1) kapıya yakın görünürlük (detektör + görüş alanı), (4–5) kamera kalibrasyon ve bozulma gerçekçiliği, (2–3) kovaryans ve dayanıklı ağırlık. "
           "Bunlar tahminlerdir; kazançlar bizim kapalı döngümüzde ölçülmeden kesin değildir.", B))

S.append(P("9. Sınırlar ve dürüstlük notları", H2))
S += bullets([
    "Tek tohum, 24 dron: ±10 puan gürültü. 4 tohum: ±5 puan. Hedefe 16 puan kalan sonuç bile birkaç tohumda değişebilir.",
    "Sayıların çoğu <b>2025 A2RL pist kopyasında</b> ve organizatör kamera hattında; rastgele pistlerde en iyi kamera sonucu %63,5. Rakamlar farklı test setlerini birbirine karıştırmamalı (Torch ve Isaac kamera sayıları doğrudan karşılaştırılamaz).",
    "Çarpma anatomisi tek politika ve 32 olaya dayanıyor; yaklaşma açısı büyük olduğu için bir kısmı yan temas olabilir.",
    "Gerçek dron verisi yok; kendi kamera için gecikme (50 ms), görüş açısı (~128×96°) ve tilt değerleri tahmindir ve tezgâh testinden sonra güncellenmeli.",
    "Literatür kıyası: MonoRace gerçek uçuşta gecikme telafisiyle yörünge RMS hatasını 0,289 → 0,103 m indirdi; bizim tekrar oynatmamız aynı yönde (1,87 → 0,38 m p90). Yayın sayıları farklı veri ve pistlerdedir.",
])

doc = SimpleDocTemplate(str(out / "Algi_Hatalari_Cozumler_ve_Sonuclar.pdf"), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                        title="Algı Hataları: Çözümler ve Sonuçlar", author="DroneRacing")


def footer(c, d):
    c.saveState()
    c.setFont("N", 8)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(18 * mm, 9 * mm, "DroneRacing — algı hataları raporu, 25 Ekim 2026")
    c.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Sayfa {d.page}")
    c.restoreState()


doc.build(S, onFirstPage=footer, onLaterPages=footer)
print(out / "Algi_Hatalari_Cozumler_ve_Sonuclar.pdf")
