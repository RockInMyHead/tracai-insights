import { Link } from "react-router-dom";
import { Activity, ArrowLeft, Download, Monitor, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";

const DISPLAY_VERSION = "1.18";
const WIN_INSTALLER = "TrackAI-Setup-1.0.0.exe";
const MAC_INSTALLER = "TrackAI-1.0.0.dmg";

const DownloadPage = () => {
  const base = typeof window !== "undefined" ? window.location.origin : "";
  const winUrl = `${base}/downloads/${WIN_INSTALLER}`;
  const macUrl = `${base}/downloads/${MAC_INSTALLER}`;
  const [available, setAvailable] = useState({ win: true, mac: true });

  useEffect(() => {
    const check = async (url: string) => {
      try {
        const response = await fetch(url, { method: "HEAD", cache: "no-store" });
        const type = response.headers.get("content-type") || "";
        return response.ok && !type.includes("text/html");
      } catch {
        return false;
      }
    };

    Promise.all([check(winUrl), check(macUrl)]).then(([win, mac]) => {
      setAvailable({ win, mac });
    });
  }, [winUrl, macUrl]);

  return (
    <main className="min-h-screen bg-[#07111f] text-white">
      <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col px-6 py-8">
        <header className="flex items-center justify-between">
          <Link to="/" className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-cyan-400 text-slate-950">
              <Activity className="h-6 w-6" />
            </div>
            <span className="text-2xl font-bold">
              Track<span className="text-cyan-300">AI</span>
            </span>
          </Link>
          <Link
            to="/trajectory"
            className="inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 hover:border-cyan-300/70 hover:text-cyan-200"
          >
            <ArrowLeft className="h-4 w-4" />
            Вернуться в сервис
          </Link>
        </header>

        <section className="grid flex-1 items-center gap-10 py-14 lg:grid-cols-[1.05fr_0.95fr]">
          <div>
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-cyan-300/20 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100">
              <Sparkles className="h-4 w-4" />
              Десктопная версия TrackAI
            </div>
            <h1 className="max-w-3xl text-5xl font-bold leading-tight tracking-tight md:text-6xl">
              Скачайте приложение для загрузки видео в TrackAI
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-8 text-slate-300">
              Приложение открывает рабочий экран «Траектория», отправляет видео на сервер TrackAI и помечает загрузку
              как desktop в админке.
            </p>

            <div className="mt-9 flex flex-col gap-3 sm:flex-row">
              <a
                href={available.win ? winUrl : undefined}
                download={available.win ? WIN_INSTALLER : undefined}
                aria-disabled={!available.win}
                className={`inline-flex h-14 items-center justify-center gap-3 rounded-xl px-7 text-base font-semibold shadow-[0_0_40px_rgba(34,211,238,0.25)] ${
                  available.win
                    ? "bg-cyan-300 text-slate-950 hover:bg-cyan-200"
                    : "pointer-events-none border border-white/10 bg-white/5 text-slate-500 shadow-none"
                }`}
              >
                <Download className="h-5 w-5" />
                {available.win ? "Скачать для Windows" : "Windows файл не загружен"}
              </a>
              <a
                href={available.mac ? macUrl : undefined}
                download={available.mac ? MAC_INSTALLER : undefined}
                aria-disabled={!available.mac}
                className={`inline-flex h-14 items-center justify-center gap-3 rounded-xl border px-7 text-base font-semibold ${
                  available.mac
                    ? "border-white/12 bg-white/5 text-white hover:border-cyan-300/70 hover:bg-white/10"
                    : "pointer-events-none border-white/10 bg-white/5 text-slate-500"
                }`}
              >
                <Download className="h-5 w-5" />
                {available.mac ? "Скачать для macOS" : "macOS файл не загружен"}
              </a>
            </div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-5 shadow-2xl shadow-black/30">
            <div className="rounded-xl border border-cyan-300/20 bg-[#0a1627] p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="text-sm uppercase tracking-[0.25em] text-cyan-200/70">TrackAI Desktop</p>
                  <h2 className="mt-2 text-2xl font-semibold">Версия {DISPLAY_VERSION}</h2>
                </div>
                <Monitor className="h-9 w-9 text-cyan-300" />
              </div>

              <div className="space-y-3">
                {[
                  ["Windows installer", WIN_INSTALLER],
                  ["macOS image", MAC_INSTALLER],
                  ["Сервер", base || "текущий домен"],
                ].map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between rounded-lg border border-white/8 bg-white/[0.03] px-4 py-3">
                    <span className="text-slate-400">{label}</span>
                    <span className="max-w-[55%] truncate font-mono text-sm text-cyan-100">{value}</span>
                  </div>
                ))}
              </div>

            </div>
          </div>
        </section>
      </div>
    </main>
  );
};

export default DownloadPage;
