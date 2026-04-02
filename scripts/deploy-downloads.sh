#!/bin/bash
# Загружает установщики на сервер (nginx root: ~/dist)
# Использование: ./scripts/deploy-downloads.sh user1@176.123.167.109

SERVER="${1:-user1@176.123.167.109}"

if [ ! -d "release" ]; then
  echo "Сначала выполните: npm run build:desktop:win"
  exit 1
fi

echo "Загрузка установщиков на $SERVER..."
ssh "$SERVER" "mkdir -p ~/dist/downloads"
rsync -avz "release/TrackAI-Setup-1.0.0.exe" "$SERVER:~/dist/downloads/"
if [ -f "release/TrackAI-1.0.0.dmg" ]; then
  rsync -avz "release/TrackAI-1.0.0.dmg" "$SERVER:~/dist/downloads/"
fi
echo "Готово. Файлы доступны по адресу /downloads/"
echo "Проверьте: http://176.123.167.109/downloads/TrackAI-Setup-1.0.0.exe"
