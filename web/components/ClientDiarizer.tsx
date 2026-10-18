"use client";

import dynamic from "next/dynamic";

// Model ve mikrofon sadece tarayıcıda çalışır: sunucuda hiç yüklenmesin
const Diarizer = dynamic(() => import("./Diarizer"), {
  ssr: false,
  loading: () => <div className="state"><div className="state-title">Model yükleniyor...</div></div>,
});

export default function ClientDiarizer() {
  return <Diarizer />;
}
