import Navbar from "@/components/Navbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Download, Monitor, MonitorSmartphone, CheckCircle2 } from "lucide-react";
import { Link } from "react-router-dom";

const APP_VERSION = "1.0.0";
const WIN_INSTALLER = `TrackAI-Setup-${APP_VERSION}.exe`;
const MAC_INSTALLER = `TrackAI-${APP_VERSION}.dmg`;

const DownloadPage = () => {
  const base = typeof window !== 'undefined' ? window.location.origin : '';
  const winUrl = `${base}/downloads/${WIN_INSTALLER}`;
  const macUrl = `${base}/downloads/${MAC_INSTALLER}`;

  return (
    <div className="min-h-screen bg-gradient-dark">
      <Navbar />
      <div className="container mx-auto px-6 pt-24 pb-16">
        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold mb-4">Скачать TrackAI</h1>
            <p className="text-lg text-muted-foreground">
              Десктопное приложение для Windows и macOS. Работает быстрее веб-версии.
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 mb-12">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Monitor className="h-6 w-6 text-primary" />
                  <CardTitle>Windows</CardTitle>
                </div>
                <CardDescription>{WIN_INSTALLER}</CardDescription>
              </CardHeader>
              <CardContent>
                <a
                  href={winUrl}
                  download={WIN_INSTALLER}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-md bg-primary px-8 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                >
                  <Download className="h-5 w-5" />
                  Скачать для Windows
                </a>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <MonitorSmartphone className="h-6 w-6 text-primary" />
                  <CardTitle>macOS</CardTitle>
                </div>
                <CardDescription>{MAC_INSTALLER}</CardDescription>
              </CardHeader>
              <CardContent>
                <a
                  href={macUrl}
                  download={MAC_INSTALLER}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-md border border-input bg-background px-8 text-sm font-medium hover:bg-accent"
                >
                  <Download className="h-5 w-5" />
                  Скачать для macOS
                </a>
              </CardContent>
            </Card>
          </div>

          <Card className="mb-8">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CheckCircle2 className="h-5 w-5 text-emerald-500" />
                Преимущества десктопной версии
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-muted-foreground">
              <p>• Работает без браузера — отдельное окно приложения</p>
              <p>• Удобная загрузка файлов — перетаскивание в окно</p>
              <p>• Подключение к серверу — анализ видео на сервере TrackAI</p>
              <p>• Требуется интернет для анализа (сервер обрабатывает видео)</p>
            </CardContent>
          </Card>

          <div className="text-center">
            <Link to="/">
              <Button variant="ghost">← Вернуться на главную</Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DownloadPage;
