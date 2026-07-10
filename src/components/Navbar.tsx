import { Link, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Activity, LayoutDashboard, Route, MessageCircle, Download, Printer } from "lucide-react";

const isDesktopClient = () => {
  if (typeof window === "undefined") return false;
  const trackai = (window as unknown as { trackai?: { isDesktop?: boolean } }).trackai;
  const queryDesktop = new URLSearchParams(window.location.search).get("desktop") === "1";
  if (trackai?.isDesktop === true || queryDesktop) {
    window.sessionStorage.setItem("trackai_desktop_client", "1");
    return true;
  }
  return window.sessionStorage.getItem("trackai_desktop_client") === "1";
};

const Navbar = () => {
  const location = useLocation();
  const desktopClient = isDesktopClient();
  
  const isActive = (path: string) => location.pathname === path;
  
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/50 bg-background/80 backdrop-blur-xl">
      <div className="container mx-auto px-6">
        <div className="flex h-16 items-center justify-between">
          <Link to={desktopClient ? "/trajectory" : "/"} className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-primary">
              <Activity className="h-5 w-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-bold text-foreground">
              Track<span className="text-gradient">AI</span>
            </span>
          </Link>
          
          <div className="hidden md:flex items-center gap-1">
            {!desktopClient && (
              <>
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
              </>
            )}
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
            {!desktopClient && (
              <>
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
                <Link to="/logs">
                  <Button
                    variant={isActive("/logs") ? "secondary" : "ghost"}
                    size="sm"
                    className="gap-2"
                  >
                    <Printer className="h-4 w-4" />
                    SLM Printer
                  </Button>
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
