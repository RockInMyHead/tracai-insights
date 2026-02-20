import { useState } from "react";
import { useNavigate } from "react-router-dom";
import SupportChat from "@/components/SupportChat";

const Support = () => {
  const navigate = useNavigate();
  const [isChatOpen, setIsChatOpen] = useState(true);

  const handleClose = () => {
    setIsChatOpen(false);
    // Небольшая задержка перед переходом, чтобы анимация закрытия успела проиграться
    setTimeout(() => {
      navigate('/');
    }, 300);
  };

  return (
    <div className="min-h-screen pt-16 bg-gradient-to-br from-background via-background to-secondary/20">
      <div className="container mx-auto px-6 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold mb-4">
              Техническая <span className="text-gradient">поддержка</span>
            </h1>
            <p className="text-lg text-muted-foreground">
              Получите помощь от нашего ИИ-специалиста по работе с системой TrackAI
            </p>
          </div>

          <div className="bg-card rounded-lg border border-border p-6" style={{ minHeight: 'calc(100vh - 200px)' }}>
            <SupportChat isOpen={isChatOpen} onClose={handleClose} fullPage={true} />
          </div>

          <div className="mt-8 text-center">
            <p className="text-sm text-muted-foreground">
              Если проблема не решена, вы можете обратиться к разработчикам напрямую
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Support;