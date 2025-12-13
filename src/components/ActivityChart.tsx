import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const data = [
  { time: "08:00", активность: 20 },
  { time: "09:00", активность: 45 },
  { time: "10:00", активность: 65 },
  { time: "11:00", активность: 80 },
  { time: "12:00", активность: 40 },
  { time: "13:00", активность: 30 },
  { time: "14:00", активность: 75 },
  { time: "15:00", активность: 85 },
  { time: "16:00", активность: 70 },
  { time: "17:00", активность: 50 },
  { time: "18:00", активность: 25 },
];

const ActivityChart = () => {
  return (
    <div className="p-6 rounded-2xl bg-gradient-card border border-border/50">
      <h3 className="text-lg font-semibold mb-4">Активность за день</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorActivity" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(187, 85%, 53%)" stopOpacity={0.3} />
                <stop offset="95%" stopColor="hsl(187, 85%, 53%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 18%)" />
            <XAxis 
              dataKey="time" 
              stroke="hsl(215, 20%, 55%)" 
              fontSize={12}
              tickLine={false}
            />
            <YAxis 
              stroke="hsl(215, 20%, 55%)" 
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(222, 47%, 8%)",
                border: "1px solid hsl(222, 30%, 18%)",
                borderRadius: "12px",
                color: "hsl(210, 40%, 98%)",
              }}
            />
            <Area
              type="monotone"
              dataKey="активность"
              stroke="hsl(187, 85%, 53%)"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#colorActivity)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ActivityChart;
