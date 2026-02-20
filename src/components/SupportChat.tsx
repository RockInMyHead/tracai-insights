import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { X, Send, Bot, User, Trash2 } from "lucide-react";

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const SYSTEM_PROMPT = `Ты - вежливый и профессиональный специалист технической поддержки для системы TrackAI.

TrackAI - это система анализа траекторий движения, которая:
- Анализирует видео файлы для отслеживания перемещений людей
- Обрабатывает файлы MP4, AVI, MOV, MKV до 10GB
- Использует компьютерное зрение и ИИ для анализа траекторий
- Предоставляет детальные отчеты о движении персонала
- Поддерживает стабилизацию видео и SLAM-анализ

Инструкции для общения:
- Будь максимально вежливым и профессиональным
- Отвечай кратко, но информативно
- Если не знаешь ответ, предложи связаться с разработчиками
- Используй русский язык для общения
- Помогай пользователям с техническими вопросами о системе`;

interface SupportChatProps {
  isOpen: boolean;
  onClose: () => void;
  fullPage?: boolean;
}

const STORAGE_KEY = 'trackai_support_chat_history';

const SupportChat = ({ isOpen, onClose, fullPage = false }: SupportChatProps) => {
  const [messages, setMessages] = useState<Message[]>(() => {
    // Загружаем историю из localStorage при инициализации
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        return parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
      } catch (e) {
        console.error('Failed to load chat history:', e);
      }
    }
    // Если истории нет, показываем приветственное сообщение
    return [
      {
        id: '1',
        content: 'Здравствуйте! Я специалист технической поддержки TrackAI. Чем могу помочь?',
        sender: 'bot',
        timestamp: new Date()
      }
    ];
  });
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Сохраняем историю в localStorage при изменении сообщений
  useEffect(() => {
    if (messages.length > 0) {
      const toSave = messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp.toISOString()
      }));
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    }
  }, [messages]);

  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  // Отправка в Telegram теперь происходит через бэкенд

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Подготавливаем историю сообщений для контекста
      const messageHistory = messages.slice(-10).map(msg => ({
        role: msg.sender === 'user' ? 'user' : 'assistant',
        content: msg.content
      }));

      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          system_prompt: SYSTEM_PROMPT,
          history: messageHistory
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const data = await response.json();

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.response,
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        content: 'Извините, произошла ошибка. Попробуйте позже или обратитесь к разработчикам.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearHistory = () => {
    if (confirm('Вы уверены, что хотите очистить историю чата?')) {
      const welcomeMessage: Message = {
        id: '1',
        content: 'Здравствуйте! Я специалист технической поддержки TrackAI. Чем могу помочь?',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
      localStorage.removeItem(STORAGE_KEY);
    }
  };

  if (!isOpen) return null;

  const cardContent = (
    <>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <CardTitle className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          Техническая поддержка TrackAI
        </CardTitle>
        <div className="flex items-center gap-2">
          <Button 
            variant="ghost" 
            size="sm" 
            onClick={clearHistory}
            title="Очистить историю"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>

      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        <div 
          className="flex-1 px-4 overflow-y-auto"
          style={{ 
            height: fullPage ? 'calc(100vh - 280px)' : 'calc(600px - 180px)'
          }}
        >
            <div className="space-y-4 py-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-3 ${
                    message.sender === 'user' ? 'justify-end' : 'justify-start'
                  }`}
                >
                  <div
                    className={`flex gap-2 max-w-[80%] ${
                      message.sender === 'user' ? 'flex-row-reverse' : ''
                    }`}
                  >
                    <div
                      className={`flex h-8 w-8 items-center justify-center rounded-full ${
                        message.sender === 'user'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-secondary text-secondary-foreground'
                      }`}
                    >
                      {message.sender === 'user' ? (
                        <User className="h-4 w-4" />
                      ) : (
                        <Bot className="h-4 w-4" />
                      )}
                    </div>
                    <div
                      className={`rounded-lg px-3 py-2 ${
                        message.sender === 'user'
                          ? 'bg-primary text-primary-foreground'
                          : 'bg-muted'
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                      <p className="text-xs opacity-70 mt-1">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="flex gap-2 max-w-[80%]">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-secondary text-secondary-foreground">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="rounded-lg px-3 py-2 bg-muted">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce"></div>
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                        <div className="w-2 h-2 bg-current rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="border-t p-4">
            <div className="flex gap-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Опишите вашу проблему..."
                disabled={isLoading}
                className="flex-1"
              />
              <Button
                onClick={sendMessage}
                disabled={!inputMessage.trim() || isLoading}
                size="sm"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardContent>
    </>
  );

  if (fullPage) {
    return (
      <Card className="w-full flex flex-col" style={{ height: 'calc(100vh - 200px)' }}>
        {cardContent}
      </Card>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <Card className="w-full max-w-2xl h-[600px] flex flex-col">
        {cardContent}
      </Card>
    </div>
  );
};

export default SupportChat;