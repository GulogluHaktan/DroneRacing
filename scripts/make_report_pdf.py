"""Builds the method + results PDF (Turkish) and the improvement-roadmap PDF from the frozen v1 results.

Run: python3 scripts/make_report_pdf.py [--frame path/to/frame.jpg]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frame", default=None)
ap.add_argument("--out-dir", default="reports")
a = ap.parse_args()
root = Path(__file__).resolve().parents[1]
out = root / a.out_dir
out.mkdir(exist_ok=True)

FD = "/usr/share/fonts/noto/"
pdfmetrics.registerFont(TTFont("N", FD + "NotoSans-Regular.ttf"))
pdfmetrics.registerFont(TTFont("NB", FD + "NotoSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("NI", FD + "NotoSans-Italic.ttf"))
pdfmetrics.registerFontFamily("N", normal="N", bold="NB", italic="NI", boldItalic="NB")

H1 = ParagraphStyle("h1", fontName="NB", fontSize=19, leading=24, spaceAfter=4, textColor=colors.HexColor("#12304f"))
H2 = ParagraphStyle("h2", fontName="NB", fontSize=13.5, leading=18, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#12304f"))
B = ParagraphStyle("b", fontName="N", fontSize=10.5, leading=15, spaceAfter=4)
SM = ParagraphStyle("sm", fontName="N", fontSize=8.8, leading=12, textColor=colors.HexColor("#444444"))
BL = ParagraphStyle("bl", parent=B, leftIndent=12, bulletIndent=2)
TC = ParagraphStyle("tc", fontName="N", fontSize=9.4, leading=12.5)
TB = ParagraphStyle("tb", fontName="NB", fontSize=9.4, leading=12.5, textColor=colors.white)


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items):
    return [Paragraph(t, BL, bulletText="•") for t in items]


def table(rows, widths, head=True, zebra=True):
    data = [[Paragraph(str(c), TB if (head and i == 0) else TC) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1 if head else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c4d0")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if head:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")))
    if zebra:
        for i in range(1, len(rows)):
            if i % 2 == 0:
                st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#eef3f8")))
    t.setStyle(TableStyle(st))
    return t


# ---------------------------------------------------------------- figures
def fig_pipeline(path):
    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    boxes = [(1, 22, 15, 12, "Kamera\n256×192, 130°", "#dbe9f6"), (20, 22, 16, 12, "GateNet\nkapı köşeleri", "#dbe9f6"),
             (40, 22, 18, 12, "Dönme destekli\nPnP", "#dbe9f6"), (62, 22, 17, 12, "Harita tabanlı\nEKF", "#fde9c9"),
             (83, 22, 16, 12, "RL politikası\n(PPO)", "#d9f0d8")]
    for x, y, w, h, t, c in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=c, ec="#33506b", lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.2)
    for x0, x1 in ((16.5, 19.5), (36.5, 39.5), (58.5, 61.5), (79.5, 82.5)):
        ax.annotate("", xy=(x1, 28), xytext=(x0, 28), arrowprops=dict(arrowstyle="->", lw=1.4, color="#33506b"))
    ax.add_patch(FancyBboxPatch((62, 4), 17, 10, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#fde9c9", ec="#33506b", lw=1.1))
    ax.text(70.5, 9, "IMU (jiro + ivmeölçer)", ha="center", va="center", fontsize=7.6)
    ax.annotate("", xy=(70.5, 21.5), xytext=(70.5, 14.7), arrowprops=dict(arrowstyle="->", lw=1.4, color="#33506b"))
    ax.add_patch(FancyBboxPatch((40, 4), 18, 10, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#eeeeee", ec="#33506b", lw=1.1))
    ax.text(49, 9, "Yaklaşık pist haritası", ha="center", va="center", fontsize=7.6)
    ax.annotate("", xy=(49, 21.5), xytext=(49, 14.7), arrowprops=dict(arrowstyle="->", lw=1.2, color="#777777"))
    ax.annotate("", xy=(64, 12.5), xytext=(58.7, 9), arrowprops=dict(arrowstyle="->", lw=1.2, color="#777777"))
    ax.text(89.5, 17, "itki + gövde\nhızları", ha="right", va="top", fontsize=7.2, color="#2d7a2d")
    ax.annotate("", xy=(91, 12), xytext=(91, 21.5), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2d7a2d"))
    ax.add_patch(FancyBboxPatch((82, 3), 17, 8.5, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#d9f0d8", ec="#33506b", lw=1.1))
    ax.text(90.5, 7.2, "Uçuş kontrolcüsü\n→ motorlar", ha="center", va="center", fontsize=7.6)
    ax.text(50, 38, "Kamera + IMU ile uçuş hattı", ha="center", fontsize=10.5, fontweight="bold", color="#12304f")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_progress(path):
    labels = ["Başlangıç\n(GRU + gizli harita)\n48 bölüm", "+ Harita tabanlı\nEKF\n48 bölüm", "+ Dönme destekli PnP\n+ GateNet g3\n24 bölüm", "+ RL ince ayar\n(t10)\n168 bölüm"]
    vals = [2.1, 14.6, 54.2, 64.3]
    ns = ["48 bölüm", "48 bölüm", "24 bölüm", "168 bölüm"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=200)
    bars = ax.bar(range(4), vals, color=["#9bb3c9", "#7fa3c4", "#4f86b8", "#1f6a3a"], width=0.62)
    for i, (v, n) in enumerate(zip(vals, ns)):
        ax.text(i, v + 1.6, f"%{v:.1f}".replace(".", ","), ha="center", fontsize=10, fontweight="bold")
    ax.axhline(60, color="#c0392b", lw=1.2, ls="--")
    ax.text(-0.45, 61.5, "hedef %60", ha="left", fontsize=8, color="#c0392b")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 78)
    ax.set_ylabel("Başarı (turu bitirme) %", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Gerçek kamera, Isaac Sim: adım adım başarı", fontsize=10.5, color="#12304f")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_seeds(path):
    seeds = [21, 22, 23, 24, 25, 26, 27]
    v = [62.5, 83.3, 58.3, 75.0, 58.3, 50.0, 62.5]
    fig, ax = plt.subplots(figsize=(7.0, 2.4), dpi=200)
    ax.bar([str(s) for s in seeds], v, color="#2e7d4f", width=0.6)
    ax.axhline(64.3, color="#12304f", lw=1.2)
    ax.text(-0.4, 92, "ortalama %64,3 (çizgi)  ·  hedef %60 (kesikli)", ha="left", fontsize=8, color="#12304f")
    ax.axhline(60, color="#c0392b", lw=1, ls="--")
    for i, x in enumerate(v):
        ax.text(i, x + 1.5, f"{x:.0f}".replace(".", ","), ha="center", fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("değerlendirme seed'i (her biri 24 bölüm)", fontsize=8)
    ax.set_ylabel("Başarı %", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


tmp = out / "_fig"
tmp.mkdir(exist_ok=True)
fig_pipeline(tmp / "pipeline.png")
fig_progress(tmp / "progress.png")
fig_seeds(tmp / "seeds.png")


def img(p, w_mm):
    from PIL import Image as PI
    w, h = PI.open(p).size
    return Image(str(p), width=w_mm * mm, height=w_mm * mm * h / w)


# ---------------------------------------------------------------- PDF 1: methods + results
def build_methods(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Kamera ile Drone Yarışı: Yöntem ve Sonuçlar (v1)", author="DroneRacing")
    s = [P("Kamera ile Drone Yarışı", H1), P("Yöntem ve sonuçlar · v1 (dondurulmuş sürüm) · 19 Ekim 2026", SM), Spacer(1, 6)]
    s += [P("Özet", H2),
          P("Drone, yarış pistini <b>yalnızca kamera görüntüsü ve IMU</b> ile uçuyor: kapıları kameradan buluyor, konumunu kendisi kestiriyor ve "
            "pekiştirmeli öğrenmeyle (RL) eğitilmiş bir politika ile uçuyor. Isaac Sim'de, gerçek render edilen kamera karelerinde, 7 farklı seed "
            "üzerinde toplam <b>168 bölümde %64,3 başarı</b> (turu çarpmadan bitirme), ortalama <b>11–12,7 m/s</b> hızla elde edildi. "
            "Bu çalışmanın başındaki aynı koşuldaki başarı %2–8 idi."),
          img(tmp / "progress.png", 165)]
    s += [P("Uçuş hattı", H2), img(tmp / "pipeline.png", 170),
          P("Politika dışarıdan gerçek konum bilgisi almıyor. Elinde yalnızca kamera, IMU ve <b>yaklaşık</b> bir pist haritası var "
            "(kapı konumları ±0,25 m, yön ±5° hatalı). Başlangıç noktası biliniyor.")]
    if a.frame and Path(a.frame).exists():
        s += [P("Uçuştan bir kare", H2), img(a.frame, 170),
              P("Solda takip kamerası, sağda drone'un kendi kamerası. Sarı: GateNet'in bulduğu kapı köşeleri. Camgöbeği: EKF'in tahmin ettiği sonraki kapı.", SM)]
    s += [P("Kullanılan yöntemler", H2)]
    rows = [["Aşama", "Ne yapıyor", "Dayandığı çalışma"],
            ["GateNet", "Kamera görüntüsünden kapının 4 iç köşesini bulan evrişimli ağ (ısı haritası + alt piksel kayması). "
                        "Eğitim verisi: Isaac'te rastgele poz, eğim (25–80° burun aşağı, %12 ters), mesafe (2–20 m) ve arka plan ile "
                        "üretilen toplam 130 bin görüntü (eğitimde 90 bini kullanıldı); hareket bulanıklığı ve gürültü eklenir.", "Loquercio vd. 2019 (T-RO): rastgele benzetim verisiyle algı; AlphaPilot (2005.12813)"],
            ["Dönme destekli PnP", "4 köşeden kapı merkezi konumunu, EKF'in dönme tahmini ve haritadaki kapı yönü bilinerek çözer "
                                   "(köşe ışınları üzerinde kapalı biçimli en küçük kareler). Benzetimde uzak kapılarda (6–20 m) konum hatasını yaklaşık 3 kat azaltır.", "MonoRace (2601.15222)"],
            ["Harita tabanlı EKF", "IMU ile durumu ilerletir (konum, hız, yönelim, sensör sapmaları). Kapı ölçümlerini haritadaki en yakın "
                                   "kapıyla Mahalanobis testiyle eşler; testi geçemeyen yanlış tespitleri atar. Hata zamanla büyümez.", "MonoRace, AlphaPilot (2005.12813)"],
            ["RL politikası", "PPO ile eğitilmiş küçük sinir ağı. Girdi: kestirilen hız, yönelim, jiro, önceki komut, sonraki 2 kapı ve son 8 adım. "
                              "Çıktı: itki ve gövde açısal hızları. Eğitimde kritik ağ gerçek durumu görür (asimetrik actor-critic); "
                              "politika EKF'in gerçekçi hata düzeyiyle eğitilir.", "Swift (Kaufmann vd., Nature 2023); asimetrik AC (2105.11674)"],
            ["Dayanıklılık", "Kütle, itki, boyut, sürükleme, gecikme, motor ve mikser dinamiği, rüzgâr ve batarya düşüşü rastgele değiştirilir "
                             "(A2RL benzeri aralıklar). 4096 paralel ortamda eğitim, Isaac Sim fiziğinde doğrulama.", "One Net (2504.21586); MonoRace"]]
    s += [table(rows, [28, 100, 46])]
    s += [P("Sonuçlar", H2), img(tmp / "seeds.png", 150),
          table([["Koşul (Isaac, gerçek kamera)", "Başarı", "Bölüm"],
                 ["Başlangıç: GRU kestirici + gizli harita", "%2,1", "48"],
                 ["Harita tabanlı EKF (eski GateNet g2)", "%14,6", "48"],
                 ["+ dönme destekli PnP + GateNet g3 (eski politika t8snap)", "%54,2", "24"],
                 ["Politika t10 + g2 (eski GateNet)", "%45,8", "24"],
                 ["<b>Politika t10 + g3 + dönme destekli EKF (v1)</b>", "<b>%64,3</b>", "<b>168</b>"]], [118, 26, 30]),
          P("%95 güven aralığı yaklaşık %57–71. Seed'ler arası dalgalanma %50–83. Aşamalar tek tek temiz ablasyon değil: bazı adımlar birlikte "
            "değişti ve örnek sayıları farklı.", SM)]
    s += [P("Varsayımlar ve sınırlar", H2)] + bullets([
        "Yaklaşık pist haritası veriliyor (MonoRace gibi). Harita tamamen bilinmeseydi sonuç çok daha düşüktü (%2–8).",
        "Başlangıç pozu biliniyor; kapılar A2RL tipi, pistte direk yok; tek drone.",
        "Tüm sonuçlar benzetimde (Isaac Sim). Gerçek donanım, merceğin bozulması ve satır tarama (rolling shutter) denenmedi.",
        "Kameranın gerçek dünyaya aktarımı için ayrıca gerçek görüntüyle doğrulama gerekir.",
        "Çarpma oranı %35,7. Bu sürümde çarpmaların nedenleri ayrıca incelenmedi; bir sonraki adım bu analiz."])
    s += [P("Kaynaklar", H2),
          P("MonoRace arXiv:2601.15222 · SkyDreamer 2510.14783 · AlphaPilot 2005.12813 · Dream to Fly 2501.14377 · One Net 2504.21586 · "
            "Loquercio vd., IEEE T-RO 2019 (doi:10.1109/tro.2019.2942989) · Kaufmann vd., Nature 2023 (doi:10.1038/s41586-023-06419-4) · "
            "Baisero &amp; Amato 2105.11674.", SM)]
    doc.build(s)


# ---------------------------------------------------------------- PDF 2: improvement roadmap
def build_roadmap(path):
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
                            title="Sonuçları iyileştirme yol haritası", author="DroneRacing")
    s = [P("Sonuçları iyileştirmek mümkün mü?", H1), P("Literatür taraması · v1 sonrası yol haritası · 19 Ekim 2026", SM), Spacer(1, 6)]
    s += [P("Kısa cevap", H2),
          P("<b>Evet.</b> Şampiyon düzeyindeki sistemler aynı temel hattı kullanıyor ama bizden fazla iki şey yapıyor: kestiriciyi kameranın ham "
            "köşe ölçümlerine daha sıkı bağlıyorlar ve yüksek çözünürlüklü, tahmine dayalı kırpma ile köşeleri daha hassas buluyorlar. "
            "MonoRace (A2RL 2025 birincisi) 28 m/s, ADR-VINS (A2RL Sezon 2) 20,9 m/s'ye çıktı; bizim hızımız 11–12,7 m/s. "
            "Aşağıdaki sıra, bizim hatamıza (kapı konum hatası ve kapı görünmediğinde kayma) en doğrudan çare olanlardan başlıyor. "
            "Beklenen kazançlar bizim tahminimiz, literatürde bu sayılar yok.")]
    rows = [["#", "İyileştirme", "Literatür", "Bizde durum", "Tahmini etki / maliyet"],
            ["1", "<b>Köşe piksel hatasını doğrudan EKF'e sokmak</b> (PnP yerine). En az 2 köşe yeter, sağlam yeniden ağırlıklandırma",
             "ADR-VINS 2603.02742: RMS 0,143 m, 20,9 m/s'de uçtu; PnP'nin 4 köşe şartını ve bilgi kaybını kaldırıyor",
             "Dört köşe şart, kısmen görünen kapı atılıyor", "Yüksek / orta"],
            ["2", "<b>Politikayı ölçülen EKF hatasıyla yeniden eğitmek</b> (kalibre profil, ikinci tur)",
             "Swift (Nature 2023): gerçek uçuştan öğrenilmiş artık gözlem modeli",
             "Profil ilk koşulardan geçici; Isaac dökümünden kalibre edilmedi", "Orta / düşük"],
            ["3", "<b>Kapı konumunu çok gözlemle birleştirmek</b> (yer işareti başına çoklu ölçüm, çift graf)",
             "2604.15168: VIO'ya göre %56–74 daha az hata; tek grafa göre %10–12 daha iyi",
             "Kapı düzeltmesi tek adımlık düşük kazançlı süzgeç", "Orta / düşük"],
            ["4", "<b>Yüksek çözünürlük + tahmine dayalı kırpma</b>",
             "MonoRace: kırpma ile %36 daha çok köşe, %25 daha çok geçerli PnP karesi",
             "Düz 256×192; 12 m'de kapı ~14 piksel", "Yüksek / orta-yüksek (render maliyeti)"],
            ["5", "<b>Algı farkındalıklı ödül:</b> kapıya bakmayı ödüllendir",
             "MonoRace ödül cezası; Song 2022 (2210.01841); Dream to Fly kendiliğinden öğrendi",
             "Küçük bir algı ödülü var (0,02)", "Orta / düşük"],
            ["6", "<b>Kapı görünmediğinde odometri:</b> görsel-eylemsiz odometri ve kayma modeli",
             "Novák vd. 2602.01860 (A2RL finalist): VIO'nun tüm durumlarını düzeltir; Azhari vd. 2512.20475",
             "Sadece IMU + kapı; kapısız aralıkta kayma", "Orta / yüksek"],
            ["7", "<b>Daha yüksek kontrol frekansı, doğrudan motor komutu</b> (G&amp;CNet)",
             "MonoRace: 500 Hz, 3×64 ağ; Ferede 2311.16948, 2504.21586",
             "50 Hz, gövde hızı + itki komutu, ayrı iç döngü", "Hız için yüksek / yüksek"],
            ["8", "<b>Uçtan uca dünya modeli</b> (algı + kestirim + kontrol birlikte)",
             "SkyDreamer 2510.14783: 21 m/s gerçek uçuş; Dream to Fly 2501.14377",
             "Pikselden PPO denendi, %0", "Belirsiz / çok yüksek"]]
    s += [table(rows, [7, 50, 50, 38, 29])]
    s += [P("Önerilen sıra", H2)] + bullets([
        "<b>Adım 1 (düşük maliyet):</b> 2 ve 3 numaralı iyileştirmeler. Isaac'ten EKF çıktı hatası dökümü alıp profili kalibre et, kapı konumunu çoklu gözlemle birleştir, "
        "politikayı bir tur daha eğit (t11). Aynı 7 seed'de karşılaştır.",
        "<b>Adım 2:</b> 1 numara. Doğrudan köşe yeniden izdüşüm güncellemesi ve sağlam ağırlıklandırma. Kısmen görünen kapıları da kullanır.",
        "<b>Adım 3:</b> 5 numara (algı ödülü) ve 4 numara (çözünürlük + kırpma) birlikte, hız hedefini artırmadan önce güvenilirlik %75+ olunca.",
        "<b>Uzun vade:</b> 7 ve 8. Hız sınırı (11–12 m/s) büyük olasılıkla kontrol yapısından ve algı gecikmesinden geliyor."])
    s += [P("Dürüst notlar", H2)] + bullets([
        "Literatürdeki hız ve hata sayıları gerçek donanımda, farklı pist ve koşulda; bizim benzetim sonuçlarımızla doğrudan kıyaslanamaz.",
        "Mevcut %64,3'ün güven aralığının alt ucu ~%57. Her iyileştirmeyi aynı seed kümesinde ve en az 168 bölümle ölçmek gerekir.",
        "Haritanın yaklaşık verilmesi bir varsayım. Gerçek yarışta harita hatası daha büyükse sonuç düşer; bunu ayrıca test etmek gerekir."])
    s += [P("Kaynaklar", H2),
          P("ADR-VINS 2603.02742 · Novák vd. 2602.01860 · Azhari vd. 2512.20475 · Dual pose-graph 2604.15168 · MonoRace 2601.15222 · "
            "SkyDreamer 2510.14783 · Dream to Fly 2501.14377 · Ferede vd. 2311.16948, 2504.21586 · Song vd. 2210.01841 · Swift (Nature 2023).", SM)]
    doc.build(s)


build_methods(out / "Kamera_ile_Ucus_Yontem_ve_Sonuclar_v1.pdf")
build_roadmap(out / "Iyilestirme_Yol_Haritasi_v1.pdf")
print("PDFs written to", out)
