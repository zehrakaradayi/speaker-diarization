import ClientDiarizer from "@/components/ClientDiarizer";

export default function Home() {
  return (
    <main>
      <header>
        <h1>Kim Konuşuyor?</h1>
        <p>Konuş, seni tanıyayım. Model kimin konuştuğunu ve ne kadar emin olduğunu zaman çizelgesine yazar.</p>
      </header>
      <ClientDiarizer />
    </main>
  );
}
