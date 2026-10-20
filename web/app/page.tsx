import ClientDiarizer from "@/components/ClientDiarizer";
import SculptureHero from "@/components/ui/liquid-sculpture-hero";

// Hero görseli: sohbet_01'in ilk 20 saniyesi, her konuşmacı kendi renginde (gerçek RTTM etiketleri)
const SOHBET = (process.env.NEXT_PUBLIC_BASE_PATH ?? "") + "/hero/sohbet.png";
const REPO = "https://github.com/zehrakaradayi/speaker-diarization";

export default function Home() {
  return (
    <>
      <SculptureHero
        src={SOHBET}
        darkSrc={SOHBET}
        mobileSrc={SOHBET}
        mobileDarkSrc={SOHBET}
        imageAlt="Dört konuşmacılı bir sohbetin dalga formu, her konuşmacı ayrı renkte"
        brand="Kim Konuşuyor?"
        navLinks={[
          { label: "Dene", href: "#uygulama" },
          { label: "Model", href: "#model" },
        ]}
        navCta={{ label: "GitHub", href: REPO }}
        themeToggle={false}
        // Renkler globals.css'ten gelir: açık/koyu tema sayfanın geri kalanıyla birlikte değişir
        background="var(--bg)"
        textColor="var(--ink)"
        accentColor="var(--accent)"
        loadFonts={false}
        headingFont="var(--font-geist), system-ui, sans-serif"
        bodyFont="var(--font-geist), system-ui, sans-serif"
        accentFont="var(--font-serif), Georgia, serif"
        eyebrow="Speaker diarization — tarayıcıda"
        headline={"Kim\nkonuşuyor?"}
        headlineAccent="konuşuyor?"
        body="Konuş, seni tanıyayım. Model sesinden kim olduğunu bulur ve ne kadar emin olduğunu zaman çizelgesine yazar. Ses cihazından çıkmaz."
        primaryCta={{ label: "Konuşmaya başla", href: "#uygulama" }}
        secondaryCta={{ label: "Model nasıl çalışıyor", href: "#model" }}
        hint="İmleci sesin üzerinde gezdir"
        hintTouch="Sese dokun"
        backdropText="kim?"
        particleDensity={1.2}
      />
      <main id="uygulama">
        <header>
          <div className="eyebrow">CANLI DENEME</div>
          <h2>Başlat'a bas ve konuş</h2>
        </header>
        <ClientDiarizer />
      </main>
    </>
  );
}
