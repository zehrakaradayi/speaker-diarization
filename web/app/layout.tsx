import type { Metadata, Viewport } from "next";
import { Geist, Instrument_Serif } from "next/font/google";
import "./globals.css";

// Fontlar build sırasında indirilip siteyle birlikte sunulur: COEP (require-corp) dış CSS'i engellerdi
const geist = Geist({ subsets: ["latin", "latin-ext"], variable: "--font-geist" });
const serif = Instrument_Serif({ subsets: ["latin", "latin-ext"], weight: "400", style: "italic", variable: "--font-serif" });

export const metadata: Metadata = {
  title: "Kim Konuşuyor?",
  description: "Konuş, seni tanıyayım: tarayıcıda çalışan konuşmacı tanıma (speaker diarization).",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f5f2" },
    { media: "(prefers-color-scheme: dark)", color: "#121118" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr" className={`${geist.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
