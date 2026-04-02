# ДЕТАЛЬНЫЙ АНАЛИЗ АЛГОРИТМА - ПОСТРОЧНАЯ РАЗБОРКА ОШИБОК

## РАЗДЕЛ 1: Инициализация и маски (строки 568-613)

### Линия 602: Извлечение признаков с маской
```python
kp, des = self.feature_detector.detectAndCompute(gray, mask)
```
**❌ ОШИБКА #11**: Маска применяется, но потом может быть заменена!

### Линия 604-605: Fallback на весь кадр
```python
if des is None or len(kp) < 80:
    kp, des = self.feature_detector.detectAndCompute(gray, None)
```
**❌ ОШИБКА #12**: Если < 80 фич, переходит на весь кадр БЕЗ маски
- Инклюдит watermark, edges (плохие признаки)
- Даже одна плохая фича может сломать всё

**❌ ОШИБКА #13**: Пороговое значение 80 ЖЕСТКОЕ
- При видео низкого качества может быть < 80 даже при хорошей сцене
- Нет адаптации к качеству видео

**Решение**:
```python
if des is None or len(kp) < 40:  # Снизить до 40
    # Использовать расширенную маску вместо None
    expanded_mask = self._expand_mask(mask, 20)
    kp, des = self.feature_detector.detectAndCompute(gray, expanded_mask)
    if des is None:  # Только если совсем беда
        kp, des = self.feature_detector.detectAndCompute(gray, None)
```

---

## РАЗДЕЛ 2: Сопоставление признаков (строки 622-642)

### Линия 626: Ratio test
```python
good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.70 * m[1].distance]
```
**❌ ОШИБКА #14**: Ratio 0.70 может быть СЛИШКОМ ЛОЯЛЬНЫМ при плохой текстуре
- Включает false positives
- Нет учёта абсолютного расстояния (только relative)

**Решение**:
```python
good = [m[0] for m in matches 
        if len(m) == 2 
        and m[0].distance < 0.70 * m[1].distance
        and m[0].distance < 50]  # Абсолютный порог
```

### Линия 629: Пороговое значение min_good_matches
```python
if len(good) > self.min_good_matches:  # 45
```
**❌ ОШИБКА #15**: При быстром движении/вращении может быть < 45 совпадений
- Даже при хороших features
- Нет адаптации к скорости движения

---

## РАЗДЕЛ 3: Fusion логика (строки 644-670)

### Линия 650-652: Взвешивание
```python
dx = (sparse_dx * sparse_weight + dx * feature_weight) / weight_sum
```
**❌ ОШИБКА #16**: weight_sum = 0.75 + 0.25 = 1.0, НО деление всё равно есть!
- Нужно считать weight_sum как sum([0.75, 0.25]) = 1.0
- Деление на 1.0 не нужно (избыточно)

### Линия 647-648: Жесткие веса
```python
sparse_weight = 0.75  # СТАТИЧНЫЙ вес
feature_weight = 0.25
```
**❌ ОШИБКА #17**: Веса НИКОГДА не адаптируются!
- Если sparse_conf низкая, всё равно 75%
- Если feature_conf высокая, только 25%

**Решение**:
```python
sparse_weight = 0.75 * (0.5 + 0.5 * sparse_conf)  # Адаптивный
feature_weight = 0.25 * (0.5 + 0.5 * feature_conf)
weight_sum = sparse_weight + feature_weight
if weight_sum > 0:
    dx = (sparse_dx * sparse_weight + dx * feature_weight) / weight_sum
```

---

## РАЗДЕЛ 4: Optical Flow (строки 672-693)

### Линия 674: Условие использования
```python
if self.use_optical_flow and self.prev_masked_gray is not None and not sparse_ok:
```
**❌ ОШИБКА #18**: OF используется только если sparse FAIL
- Но OF сам по себе может давать ошибки при вращении!
- Лучше использовать как резервный + fusion

**❌ ОШИБКА #19**: Нет проверки, что self.prev_masked_gray инициализирован
- На первом фрейме это None, и OF пропускается
- Теряется информация о движении

### Линия 684-685: Использование медианы
```python
flow_dx = float(np.median(center_region[..., 0])) * self.motion_scale * self.scale_factor
```
**❌ ОШИБКА #20**: МЕДИАНА может быть 0 при случайном шуме
- Лучше использовать СРЕДНЕЕ + МЕДИАНУ вместе
- Или фильтровать по magnitude

**Решение**:
```python
magnitude = np.sqrt(center_region[..., 0]**2 + center_region[..., 1]**2)
valid = magnitude > 0.5  # Пороговая фильтрация
if np.sum(valid) > (magnitude.size * 0.3):  # > 30% валидных
    flow_dx = float(np.median(center_region[valid, 0])) * ...
else:
    flow_dx = 0  # Недостаточно валидных значений
```

### Линия 688-689: Feature weight для OF
```python
feature_weight = float(np.clip(0.2 + 0.8 * feature_conf, 0.2, 0.9)) if feature_ok else 0.2
flow_weight = 1.0 - feature_weight
```
**❌ ОШИБКА #21**: flow_weight = 1.0 - feature_weight
- Если feature_ok=False, feature_weight=0.2, flow_weight=0.8
- Если feature_ok=True и conf=1.0, flow_weight = 1.0 - 0.9 = 0.1
- OF ВСЕГДА используется! (даже если feature отличный)

**Лучше**:
```python
if not feature_ok or feature_conf < 0.3:
    # Используем OF как основу
    dx = 0.7 * flow_dx + 0.3 * dx
    dy = 0.7 * flow_dy + 0.3 * dy
else:
    # Feature хороший, OF только как поправка
    dx = 0.9 * dx + 0.1 * flow_dx
    dy = 0.9 * dy + 0.1 * flow_dy
```

---

## РАЗДЕЛ 5: Сглаживание (строки 698-705)

### Линия 703-705: Буферное сглаживание
```python
dx_s = alpha * dx + (1 - alpha) * (self.pos_buffer[-1][0] if self.pos_buffer else dx)
```
**❌ ОШИБКА #22**: Используется ПОСЛЕДНИЙ элемент буфера
- Нужно использовать ПЕРВЫЙ или СРЕДНЕЕ!

```python
# Неправильно (используется последний):
prev_dx = self.pos_buffer[-1][0]  # Это только что добавленное значение!

# Правильно:
prev_dx = self.pos_buffer[0][0] if self.pos_buffer else dx  # Первый элемент
```

**❌ ОШИБКА #23**: maxlen=30 для буфера
- Может быть недостаточно при 30 FPS (всего 1 сек истории)
- Лучше 60-120 для плавности

### Линия 700: Буферизация ПЕРЕД сглаживанием
```python
self.pos_buffer.append([dx, dy])
self.rot_buffer.append(dheading)

dx_s = alpha * dx + (1 - alpha) * (self.pos_buffer[-1][0] if self.pos_buffer else dx)
```
**❌ ОШИБКА #24**: append ЛО-ГИ-ЧЕ-СКИ нарушает сглаживание!
```python
# После append, pos_buffer имеет [dx, dy] в конце
# Потом берём pos_buffer[-1][0] = новый dx!
# Это не сглаживание, это копирование!
```

**Правильно**:
```python
# Сглаживание ПЕРЕД добавлением в буфер
if len(self.pos_buffer) > 0:
    prev_dx = self.pos_buffer[-1][0]
else:
    prev_dx = dx
    
dx_s = alpha * dx + (1 - alpha) * prev_dx

# ПОТОМ добавляем
self.pos_buffer.append([dx_s, dy_s])  # Сглаженные значения!
```

---

## РАЗДЕЛ 6: Преобразование координат (строки 730-733)

### Линия 731-733: prev_heading (ИСПРАВЛЕНО?)
```python
prev_heading = self.trajectory[-1][2]
theta = np.radians(prev_heading)
global_dx = dx_s * np.cos(theta) - dy_s * np.sin(theta)
```
**❌ ОШИБКА #25**: prev_heading берётся из траектории
- Но траектория ТОЛЬКО ЧТО обновлена на ТЕКУЩЕЙ итерации!
- self.trajectory[-1] = новая позиция с новым heading!

**Надо сохранить heading ВЖЕ обновления**:
```python
# ДО обновления self.heading:
prev_heading_rad = np.radians(self.heading)  # Сохранить текущее
global_dx = dx_s * np.cos(prev_heading_rad) - dy_s * np.sin(prev_heading_rad)
global_dy = dx_s * np.sin(prev_heading_rad) + dy_s * np.cos(prev_heading_rad)

# ПОТОМ обновить
self.heading += rot_s
```

---

## РАЗДЕЛ 7: Интегрирование (строки 743-763)

### Линия 743-744: Добавление в траекторию
```python
new_x = self.trajectory[-1][0] + global_dx
new_y = self.trajectory[-1][1] + global_dy
```
**❌ ОШИБКА #26**: Нет проверки на NaN ПЕРЕД добавлением
- Если global_dx = NaN, вся траектория испорчена

**Решение**:
```python
if np.isnan(new_x) or np.isnan(new_y):
    new_x, new_y = self.trajectory[-1][0], self.trajectory[-1][1]
    logger.warning(f"NaN detected at frame {self.frame_count}")
```

### Линия 763: Добавление в траекторию
```python
self.trajectory.append([new_x, new_y, self.heading])
```
**❌ ОШИБКА #27**: self.heading УЖЕ обновлен ПЕРЕД этой строкой!
- self.heading += rot_s было выше (линия 725)
- Нужно сохранить старое heading

---

## РАЗДЕЛ 8: Детекция поворотов (линия 766)

### Линия 766: Вызов _detect_enhanced_turns()
```python
self._detect_enhanced_turns()
```
**❌ ОШИБКА #28**: Вызывается ПОСЛЕ добавления в траекторию
- Но окно для детекции может быть слишком маленьким на начало видео!
- Линия 753: `if len(self.trajectory) < 60: return`

**Если видео 30 сек на 30 FPS = 900 фреймов, окно 60 = 2 сек
- Повороты в начале (первые 2 сек) не детектируются!

**Решение**: Динамическое окно
```python
min_window = max(20, len(self.trajectory) // 10)  # 10% длины
```

---

## РАЗДЕЛ 9: Сохранение состояния (строки 768-770)

### Линия 768: Сохранение для следующего фрейма
```python
self.prev_gray, self.prev_kp, self.prev_des = gray, kp, des
```
**❌ ОШИБКА #29**: self.prev_kp, self.prev_des могут быть None!
- Если on first frame или detection failed
- Потом на следующем фрейме попытка доступа вызовет ошибку

**Решение**: Проверка перед использованием везде
```python
if self.prev_kp is not None and self.prev_des is not None:
    # Использовать
```

---

## РАЗДЕЛ 10: get_trajectory() (линия 883-890)

### Линия 890: Вывод траектории
```python
return [[round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 2)] for p in self.trajectory]
```
**❌ ОШИБКА #30**: Нет проверки на пустую траекторию
- Если self.trajectory пусто, вернёт пустой список
- Фронт может сломаться

**Решение**:
```python
if not self.trajectory:
    return [[0.0, 0.0, 0.0]]
return [[round(...) for p in self.trajectory]
```

---

## ИТОГОВЫЙ СПИСОК: 20 НОВЫХ ОШИБОК

| # | Строки | Ошибка | Критичность |
|---|--------|--------|-------------|
| 11 | 602-605 | Маска заменяется на None | 🔴 ВЫСОКАЯ |
| 12 | 604-605 | Fallback без маски | 🔴 ВЫСОКАЯ |
| 13 | 604 | Пороговое значение 80 жесткое | 🟠 СРЕДНЯЯ |
| 14 | 626 | Ratio test только относительный | 🟠 СРЕДНЯЯ |
| 15 | 629 | min_good_matches не адаптивен | 🟠 СРЕДНЯЯ |
| 16 | 650-652 | Лишнее деление на weight_sum | 🟡 НИЗКАЯ |
| 17 | 647-648 | Веса не адаптивны к confidence | 🔴 ВЫСОКАЯ |
| 18 | 674 | OF только при sparse fail | 🟠 СРЕДНЯЯ |
| 19 | 674 | Нет проверки prev_masked_gray | 🟠 СРЕДНЯЯ |
| 20 | 684-685 | Медиана может быть 0 | 🟠 СРЕДНЯЯ |
| 21 | 688-691 | OF ВСЕГДА используется | 🔴 ВЫСОКАЯ |
| 22 | 703 | Используется ПОСЛЕДНИЙ буфер | 🔴 ВЫСОКАЯ |
| 23 | 117 | maxlen=30 слишком мал | 🟠 СРЕДНЯЯ |
| 24 | 699-705 | append ДО сглаживания | 🔴 ВЫСОКАЯ |
| 25 | 758 | prev_heading из траектории | 🔴 ВЫСОКАЯ |
| 26 | 747-755 | Нет проверки NaN | 🟠 СРЕДНЯЯ |
| 27 | 763 | self.heading уже обновлен | 🔴 ВЫСОКАЯ |
| 28 | 766 | Окно для поворотов слишком мал | 🟡 НИЗКАЯ |
| 29 | 768 | prev_kp/des могут быть None | 🟠 СРЕДНЯЯ |
| 30 | 890 | Нет проверки пустой траектории | 🟠 СРЕДНЯЯ |

---

## КРИТИЧНЫЕ ОШИБКИ (НАДО ИСПРАВИТЬ СЕЙЧАС):
- **#11, #12**: Маска / fallback
- **#17**: Адаптивные веса fusion
- **#21**: Логика OF
- **#22, #24**: Буферное сглаживание
- **#25, #27**: Heading для преобразования координат
