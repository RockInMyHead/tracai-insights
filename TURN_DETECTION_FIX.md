# Диагностика проблемы с поворотами (90° → 180°)

## Выявленные проблемы

### 1. **Функция `_detect_enhanced_turns()` использует arccosine**
```python
raw_angle = np.degrees(np.arccos(cos_ang))  # Всегда 0-180°!
```
- **Проблема**: arccosine возвращает угол в диапазоне [0°, 180°]
- **Следствие**: не различает 90° влево от 90° вправо (оба дают 90°)
- **Решение**: использовать atan2 для ориентированного угла

### 2. **Вычисление dtheta из матрицы преобразования неправильное**
```python
dtheta = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))  # atan2 от верхней строки
```
- **Проблема**: матрица аффинного преобразования [a, b; c, d] содержит масштаб и поворот вместе
- **Правильно**: должно быть `atan2(M[1, 0] / ||M||, M[0, 0] / ||M||)` или использовать SVD
- **Следствие**: dtheta может быть полностью неправильным при масштабировании

### 3. **Срезание угла после вычисления**
```python
if abs(dtheta) > self.max_heading_step_deg:  # 15°
    return 0.0, 0.0, 0.0, ratio, False
```
- **Проблема**: если истинный поворот > 15°, он отбрасывается!
- **Следствие**: при повороте > 15° берется 0 или неправильное значение

### 4. **Детекция типа поворота (левый/правый) игнорирует heading**
```python
# Вычисляется только из координат траектории (x, y)
cos_ang = np.dot(vec1[:2], vec2[:2]) / (n1 * n2)
raw_angle = np.degrees(np.arccos(cos_ang))  # 0-180°, нет информации о направлении!
```
- **Решение**: использовать cross product для ориентированного угла

---

## Рекомендуемые исправления

### Fix 1: Правильное вычисление dtheta из матрицы
```python
# Вместо:
dtheta = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

# Правильно:
a, b = float(M[0, 0]), float(M[0, 1])
c, d = float(M[1, 0]), float(M[1, 1])
scale = np.sqrt(a*a + b*b)  # Норма строк
if scale > 0:
    cos_theta = a / scale
    sin_theta = c / scale
    dtheta = float(np.degrees(np.arctan2(sin_theta, cos_theta)))
else:
    dtheta = 0.0
```

### Fix 2: Повысить max_heading_step_deg или быть толерантнее
```python
# Текущее: 15° — слишком мало для резких поворотов
self.max_heading_step_deg = 30.0  # → 30° (или более)
```

### Fix 3: Использовать signed angle в _detect_enhanced_turns
```python
# Вместо:
cos_ang = np.dot(vec1[:2], vec2[:2]) / (n1 * n2)
raw_angle = np.degrees(np.arccos(cos_ang))

# Правильно (ориентированный угол):
v1 = vec1[:2]
v2 = vec2[:2]
cos_ang = np.dot(v1, v2) / (n1 * n2)
sin_ang = v1[0] * v2[1] - v1[1] * v2[0]  # cross product
raw_angle = float(np.degrees(np.arctan2(sin_ang, cos_ang)))  # -180 to +180
```

### Fix 4: Явно использовать sign от cross product
```python
# Определение типа поворота:
cross = v1[0] * v2[1] - v1[1] * v2[0]
turn_type = 'left' if cross > 0 else 'right'
angle = abs(raw_angle)  # Абсолютный угол
```

---

## План действий

1. ✅ **Исправить вычисление dtheta** (из матрицы преобразования)
2. ✅ **Повысить max_heading_step_deg** (15° → 30°)
3. ✅ **Использовать signed angle** в _detect_enhanced_turns
4. ✅ **Проверить гейтинг порогов** (может быть слишком жесткие)
5. ✅ **Логировать angle / raw_angle** для отладки
