import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Activity, LayoutDashboard, Route, MessageCircle, Download } from "lucide-react";

const Navbar = () => {
  const location = useLocation();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto px-6">
        <div className="flex h-16 items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-primary">
              <Activity className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold text-foreground">
              Track<span className="text-gradient">AI</span>
            </span>
          </Link>
          
          <div className="hidden md:flex items-center gap-1">
            <Link to="/">
              <Button 
                variant={isActive("/") ? "secondary" : "ghost"} 
                size="sm"
              >
                Главная
              </Button>
            </Link>
            <Link to="/dashboard">
              <Button 
                variant={isActive("/dashboard") ? "secondary" : "ghost"} 
                size="sm"
                className="gap-2"
              >
                <LayoutDashboard className="h-4 w-4" />
                Кабинет
              </Button>
            </Link>
            <Link to="/trajectory">
              <Button
                variant={isActive("/trajectory") ? "secondary" : "ghost"}
                size="sm"
                className="gap-2"
              >
                <Route className="h-4 w-4" />
                Траектория
              </Button>
            </Link>
            <Link to="/download">
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
              >
                <Download className="h-4 w-4" />
                Скачать
              </Button>
            </Link>
            <Link to="/support">
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
              >
                <MessageCircle className="h-4 w-4" />
                Техническая поддержка
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
