import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { ArrowRight, Play, Shield, Zap, BarChart3 } from "lucide-react";

const HeroSection = () => {
  return (
    <section className="relative min-h-screen pt-16 overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30" />
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[800px] bg-gradient-glow" />
      
      <div className="container relative mx-auto px-6 pt-20 pb-32">
        <div className="max-w-4xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-secondary/50 border border-border/50 mb-8 animate-fade-in">
            <span className="flex h-2 w-2 rounded-full bg-primary animate-pulse" />
            <span className="text-sm text-muted-foreground">Новое поколение аналитики движения</span>
          </div>
          
          {/* Heading */}
          <h1 className="text-5xl md:text-7xl font-bold mb-6 animate-fade-in" style={{ animationDelay: "0.1s" }}>
            Отслеживайте
            <br />
            <span className="text-gradient">каждое движение</span>
          </h1>
          
          <p className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 animate-fade-in" style={{ animationDelay: "0.2s" }}>
            AI-powered платформа для мониторинга и анализа перемещений персонала. 
            Оптимизируйте рабочие процессы и повышайте безопасность на предприятии.
          </p>
          
          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16 animate-fade-in" style={{ animationDelay: "0.3s" }}>
            <Link to="/trajectory">
              <Button variant="hero" size="xl" className="gap-2">
                Начать анализ
                <ArrowRight className="h-5 w-5" />
              </Button>
            </Link>
            <Button variant="outline" size="xl" className="gap-2">
              <Play className="h-5 w-5" />
              Смотреть демо
            </Button>
          </div>
          
          {/* Stats */}
          <div className="grid grid-cols-3 gap-8 max-w-2xl mx-auto animate-fade-in" style={{ animationDelay: "0.4s" }}>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-gradient mb-1">99.9%</div>
              <div className="text-sm text-muted-foreground">Точность</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-gradient mb-1">24/7</div>
              <div className="text-sm text-muted-foreground">Мониторинг</div>
            </div>
            <div className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-gradient mb-1">50ms</div>
              <div className="text-sm text-muted-foreground">Задержка</div>
            </div>
          </div>
        </div>
        
        {/* Features preview */}
        <div className="mt-24 grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          {[
            { icon: Zap, title: "Реальное время", desc: "Мгновенное отслеживание перемещений" },
            { icon: Shield, title: "Безопасность", desc: "Контроль доступа в опасные зоны" },
            { icon: BarChart3, title: "Аналитика", desc: "Детальные отчёты и статистика" },
          ].map((feature, i) => (
            <div 
              key={feature.title}
              className="group p-6 rounded-2xl bg-gradient-card border border-border/50 hover:border-primary/50 transition-all duration-300 animate-fade-in"
              style={{ animationDelay: `${0.5 + i * 0.1}s` }}
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-primary/10 text-primary mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground">{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Footer note */}
        <div className="mt-16 pt-8 border-t border-border/50">
          <div className="text-center">
            <p className="text-lg font-medium text-primary mb-2">
              Специально для
            </p>
            <p className="text-2xl font-bold text-gradient">
              Kerama Marazzi
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HeroSection;
