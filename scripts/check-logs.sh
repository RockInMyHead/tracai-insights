#!/bin/bash
# Скрипт для просмотра логов TrackAI на сервере
# Использование: ./scripts/check-logs.sh [количество строк]

SERVER="user1@176.123.167.109"
LINES=${1:-150}

echo "=== Логи trackai-backend (последние $LINES строк) ==="
ssh $SERVER "journalctl -u trackai-backend -n $LINES --no-pager"

echo ""
echo "=== Последние ошибки ==="
ssh $SERVER "journalctl -u trackai-backend -n 500 --no-pager | grep -i -E 'error|exception|traceback|failed' | tail -30"
