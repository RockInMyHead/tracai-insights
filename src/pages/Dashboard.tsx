import Navbar from "@/components/Navbar";
import StatsCard from "@/components/StatsCard";
import ActivityChart from "@/components/ActivityChart";
import { Users, MapPin, AlertTriangle, Clock, Settings, Bell } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-gradient-dark">
      <Navbar />
      
      <main className="container mx-auto px-6 pt-24 pb-12">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 gap-4">
          <div className="flex items-center gap-4">
            <Avatar className="h-16 w-16 border-2 border-primary/30">
              <AvatarFallback className="bg-primary/10 text-primary text-xl">АД</AvatarFallback>
            </Avatar>
            <div>
              <h1 className="text-2xl font-bold">Добро пожаловать, Администратор</h1>
              <p className="text-muted-foreground">Управляйте мониторингом предприятия</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Button variant="outline" size="icon">
              <Bell className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          <StatsCard
            title="Активные сотрудники"
            value="24"
            change="+3 за час"
            changeType="positive"
            icon={Users}
          />
          <StatsCard
            title="Среднее время в зоне"
            value="47 мин"
            change="+8%"
            changeType="neutral"
            icon={Clock}
          />
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <ActivityChart />
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
