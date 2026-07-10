c# Полный анализ алгоритма трекинга - Поиск ошибок

## ЭТАП 1: Входные данные
**Что происходит**: Video от первого лица (POV)  
**Сложность**: Быстрые движения, повороты, изменения масштаба

---

## ЭТАП 2: Предобработка кадра

### ✅ Хорошо:
- CLAHE (адаптивное выравнивание контраста)
- Масштабирование до 900px макс
- Маскирование (удаление краев, watermark)

### ❌ ОШИБКА #1: Маски конфликтуют
```python
# Line 598: Выбор маски зависит от mode (person vs ego)
mask = person_mask if mode == "person" else self._build_scene_mask(gray)
```
**Проблема**: `_build_scene_mask()` обрезает ДНО на 6% (line 376), но именно там может быть **опорная плоскость для определения поворотов**!

**Ошибка**: Обрезание нижней части убирает перспективные линии → no rotation info

**Решение**: Снизить обрезку нижней части
```python
bottom_cut = int(h * 0.02)  # Было 0.06 → 0.02
```

---

## ЭТАП 3: Извлечение признаков

### ✅ AKAZE вместо ORB
- Более инвариантен к вращению
- Хорош для быстрых движений

### ❌ ОШИБКА #2: Fallback стирает маску!
```python
# Line 604-605:
if des is None or len(kp) < 80:
    kp, des = self.feature_detector.detectAndCompute(gray, None)  # ❌ None!
```
**Проблема**: Если мало features в маске (< 80), переходит на весь кадр БЕЗ маски
- Инклюдит watermark, black areas
- Плохие features → плохие совпадения

**Решение**: Снизить порог или использовать расширенную маску
```python
if des is None or len(kp) < 40:  # Было 80 → 40 (менее жесткий порог)
    kp, des = self.feature_detector.detectAndCompute(gray, mask)  # Использовать маску!
```

---

## ЭТАП 4: Трекинг движения (3 источника)

### 4A: Sparse KLT (Lucas-Kanade)

#### ✅ Хорошо:
- Может работать при плохой текстуре
- Быстрый

#### ❌ ОШИБКА #3: KLT НЕ вычисляет вращение!
```python
# Line 439-495 (_estimate_sparse_flow_motion):
# Используется только для dx, dy!
# dtheta вычисляется через _estimate_motion(src, dst) 
# но это требует хороших точек на протяженности
```
**Проблема**: При быстром вращении KLT отслеживает точки неправильно (они скакыот вокруг оптического центра)

**Решение**: 
```python
# Для вращения использовать SVD разложения лучше:
def _estimate_rotation_from_flow(self, src, dst):
    """SVD для чистого вращения"""
    centroid_src = np.mean(src, axis=0)
    centroid_dst = np.mean(dst, axis=0)
    H = (dst - centroid_dst).T @ (src - centroid_src)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    # Извлечь угол из R
```

---

### 4B: Feature-based VO (AKAZE + Homography)

#### ❌ ОШИБКА #4: findHomography неправильно извлекает угол!
```python
# Line 513-519:
a, b = float(M[0, 0]), float(M[0, 1])
c, d = float(M[1, 0]), float(M[1, 1])
scale = np.sqrt(a*a + b*b)
cos_theta = a / scale
sin_theta = c / scale
dtheta = float(np.degrees(np.arctan2(sin_theta, cos_theta)))
```
**Проблема 1**: Матрица гомографии масштабирована и может быть нестабильна!  
**Проблема 2**: Масштаб может быть > 1 (при zoom in), что искажает угол

**Лучше**: Использовать SVD разложение для нормализации
```python
U, S, Vt = np.linalg.svd(M[:2, :2])
R = U @ Vt  # Чистое вращение (без масштаба!)
dtheta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
```

---

### 4C: Optical Flow (Farneback)

#### ❌ ОШИБКА #5: Flow используется как резервный, но его параметры плохие
```python
# Line 131-137:
flow_params = dict(
    pyr_scale=0.5,
    levels=3,      # Может быть недостаточно
    winsize=15,    # Может быть слишком большой для быстрого движения
    ...
)
```
**Проблема**: При больших углах вращения (> 30°) optical flow дает артефакты

---

## ЭТАП 5: Слияние (Fusion)

### ❌ ОШИБКА #6: Неправильная логика ветвления!
```python
# Line 644-661:
if sparse_ok:
    if feature_ok:
        # fusion: 75% sparse, 25% feature
    else:
        dx, dy, dheading = sparse_dx, sparse_dy, sparse_heading
elif not feature_ok and mode == "ego":
    self.total_ransac_failures += 1
```
**Проблема**: 
- Если sparse_ok=False и feature_ok=False → ничего не происходит!
- dx, dy, dheading остаются 0
- **Поворот теряется!**

**Решение**: Явно обработать все комбинации
```python
if feature_ok and sparse_ok:
    # Fusion
elif feature_ok:
    dx, dy, dheading = dx, dy, dheading  # feature only
elif sparse_ok:
    dx, dy, dheading = sparse_dx, sparse_dy, sparse_heading  # sparse only
# else: остаток от инициализации (0, 0, 0) - движения нет
```

---

## ЭТАП 6: Сглаживание

### ❌ ОШИБКА #7: Двойное сглаживание разрушает крутые повороты!
```python
# Line 682-685:
alpha = 0.40
dx_s = alpha * dx + (1 - alpha) * (self.pos_buffer[-1][0] if self.pos_buffer else dx)
dy_s = alpha * dy + (1 - alpha) * (self.pos_buffer[-1][1] if self.pos_buffer else dy)
rot_s = alpha * dheading + (1 - alpha) * (self.rot_buffer[-1] if self.rot_buffer else dheading)
```
**Потом еще раз:**
```python
# Line 763-766:
smooth_x = 0.15 * new_x + (1 - 0.15) * prev_x
smooth_y = 0.15 * new_y + (1 - 0.15) * prev_y
smooth_h = 0.15 * self.heading + (1 - 0.15) * prev_h
```
**Проблема**: Поворот на 45° за фрейм будет отфильтрован дважды:
1. alpha=0.40 → остается 40% от 45° = 18°
2. alpha=0.15 → остается 15% от 18° = 2.7°
**Итог: 45° → 2.7°** ❌

**Решение**: 
```python
# Использовать ОДНО сглаживание, но адаптивное:
if abs(dheading) > 15:  # Крутой поворот
    alpha_rot = 0.7  # Минимум фильтрации
else:
    alpha_rot = 0.3  # Нормальное сглаживание
rot_s = alpha_rot * dheading + (1 - alpha_rot) * rot_buffer[-1]

# Убрать вторичное сглаживание в trajectory!
self.trajectory.append([new_x, new_y, self.heading])  # Без дополнительного alpha
```

---

## ЭТАП 7: Интегрирование позиции

### ❌ ОШИБКА #8: Преобразование в глобальные координаты неправильно!
```python
# Line 726-728:
theta = np.radians(self.heading)
global_dx = dx_s * np.cos(theta) - dy_s * np.sin(theta)
global_dy = dx_s * np.sin(theta) + dy_s * np.cos(theta)
```
**Проблема**: `self.heading` обновляется ПЕРЕД преобразованием, но используется heading ТЕКУЩЕГО фрейма
- Нужно использовать heading ПРЕДЫДУЩЕГО фрейма!

**Решение**:
```python
# Сохранить prev_heading
prev_heading_rad = np.radians(self.trajectory[-1][2])  # heading последней точки
global_dx = dx_s * np.cos(prev_heading_rad) - dy_s * np.sin(prev_heading_rad)
global_dy = dx_s * np.sin(prev_heading_rad) + dy_s * np.cos(prev_heading_rad)

# Потом обновить heading
self.heading += rot_s
```

---

## ЭТАП 8: Детекция поворотов

### ❌ ОШИБКА #9: `_detect_enhanced_turns()` смотрит на КООРДИНАТЫ, а не на HEADING!
```python
# Line 791-812:
# Вычисляет угол между векторами траектории (x, y)
vec1 = np.array(self.trajectory[mid]) - np.array(self.trajectory[start])
vec2 = np.array(self.trajectory[i]) - np.array(self.trajectory[mid])
```
**Проблема**: Если heading обновляется неправильно, то даже если координаты П-образные, угол не будет зафиксирован правильно!

**Решение**: Использовать heading напрямую
```python
# Вместо вычисления угла через cross product координат:
heading_change = self.trajectory[i][2] - self.trajectory[start][2]
# Нормализовать угол в [-180, 180]
heading_change = ((heading_change + 180) % 360) - 180

angle = abs(heading_change)
turn_type = 'left' if heading_change > 0 else 'right'
```

---

## ЭТАП 9: Вывод траектории

### ❌ ОШИБКА #10: Final moving-average ОПЯТЬ сглаживает!
```python
# Line 869-877:
win = 13
kernel = np.ones(win, dtype=np.float32) / win
x = np.convolve(arr[:, 0], kernel, mode='same')
y = np.convolve(arr[:, 1], kernel, mode='same')
h = arr[:, 2]  # heading не сглаживается, но x,y сглаживаются!
```
**Проблема**: Окончательное сглаживание деформирует П-образный путь!

**Решение**: Не сглаживать для вывода, или сглаживать минимально
```python
# Вариант 1: Вообще не сглаживать для вывода
return [[float(p[0]), float(p[1]), float(p[2])] for p in self.trajectory]

# Вариант 2: Очень мягкое сглаживание (окно 3)
if len(self.trajectory) < 3:
    return [[float(p[0]), float(p[1]), float(p[2])] for p in self.trajectory]
win = 3
kernel = np.ones(win) / win
x = np.convolve(arr[:, 0], kernel, mode='same')
y = np.convolve(arr[:, 1], kernel, mode='same')
```

---

## ИТОГОВЫЙ СПИСОК ОШИБОК

| # | Этап | Ошибка | Критичность | Решение |
|---|------|--------|-------------|---------|
| 1 | Маски | Обрезка нижней части | 🔴 ВЫСОКАЯ | Снизить bottom_cut до 2% |
| 2 | Признаки | Fallback без маски | 🔴 ВЫСОКАЯ | Использовать маску всегда |
| 3 | KLT | Не вычисляет вращение | 🔴 ВЫСОКАЯ | Использовать SVD для rotation |
| 4 | Homography | Неправильный atan2 | 🟠 СРЕДНЯЯ | SVD разложение матрицы |
| 5 | Optical Flow | Плохие параметры при вращении | 🟠 СРЕДНЯЯ | Адаптивные параметры |
| 6 | Fusion | Потеря dheading при обе ошибках | 🔴 ВЫСОКАЯ | Явная обработка все комбинаций |
| 7 | Сглаживание | Двойное сглаживание | 🔴 ВЫСОКАЯ | Одно адаптивное сглаживание |
| 8 | Интегрирование | Неправильный heading для преобразования | 🔴 ВЫСОКАЯ | Использовать prev_heading |
| 9 | Детекция поворотов | Вычисление через координаты | 🟠 СРЕДНЯЯ | Использовать heading напрямую |
| 10 | Вывод | Final moving-average | 🔴 ВЫСОКАЯ | Не сглаживать или минимально |

---

## ПРИОРИТЕТ ИСПРАВЛЕНИЙ

**Критичные (должны быть исправлены):**
1. ❌ Ошибка #8 (prev_heading)
2. ❌ Ошибка #7 (двойное сглаживание)
3. ❌ Ошибка #6 (fusion логика)
4. ❌ Ошибка #1 (маска - обрезка)
5. ❌ Ошибка #10 (final smoothing)
