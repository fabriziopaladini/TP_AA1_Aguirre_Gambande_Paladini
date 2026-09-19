# Experimentos sobre el TP1 de regresión

Este documento describe los cambios hechos sobre el notebook `TP-regresion-AA1.ipynb`
en dos ramas de prueba y los resultados obtenidos. Ambas salen de `aguirre`
(commit `da4c1b8`), que conserva el notebook original.

## Ramas

| Rama | Parte de | Qué contiene |
|---|---|---|
| `aguirre` | – | Notebook original: split sin estratificar, CHAS imputada con CatBoost, hiperparámetros de Gradient Descent elegidos mirando el R² de test. |
| `prueba-estratificacion` | `aguirre` | Split estratificado por CHAS, CHAS imputada con la moda, RAD excluida, `lr`/`epochs` de GD con K-Fold. |
| `prueba-colinealidad` | `aguirre` | Notebook original (sin estratificar, con CatBoost) + RAD excluida + `lr`/`epochs` de GD con K-Fold. |

Para ver las diferencias:

```bash
git diff aguirre prueba-estratificacion -- tp1-regresion/TP-regresion-AA1.ipynb
git log --oneline aguirre..prueba-estratificacion
git log --oneline aguirre..prueba-colinealidad
```

El `diff` de un `.ipynb` es ruidoso porque incluye las salidas de las celdas. Las celdas de
código modificadas son pocas y están comentadas.

## Cambios en el notebook

1. **Estratificación por CHAS** (solo `prueba-estratificacion`). El split usa
   `stratify=X["CHAS"].fillna(-1)`. El `fillna(-1)` solo vive dentro del argumento `stratify`
   (una serie temporal): `X` conserva los NaN. Los faltantes forman un tercer estrato.
2. **Imputación de CHAS con la moda** (solo `prueba-estratificacion`). `SimpleImputer(most_frequent)`
   con `fit` en train y el mismo valor fijo aplicado a train y test. Reemplaza al
   clasificador CatBoost; se eliminó la celda `!pip install catboost`.
3. **Exclusión de RAD** (ambas ramas). RAD y TAX tienen correlación 0.88 y VIF de 7.5 y 9.0.
   La lista `cols_excluir` (en la celda del split) controla qué variables se sacan; con
   `cols_excluir = []` se usan todas.
4. **Selección de `lr` y `epochs` de Gradient Descent con K-Fold** (ambas ramas). Antes se
   elegían mirando el R² de test (fuga de información). Ahora se usa `KFold(5, shuffle=True)`
   dentro de train, eligiendo por MSE de validación promedio, igual que `LassoCV`, `RidgeCV`
   y `ElasticNetCV`. `gradient_descent` recibe `plot=False` para no graficar durante la búsqueda.
5. **Textos**: se actualizaron los comentarios y conclusiones del notebook que citaban métricas
   anteriores.

Lo que **no** cambió: `RobustScaler` + `KNNImputer` para las numéricas (con `fit` solo en train),
`log1p` sobre CRIM y DIS, y `StandardScaler` final (`fit` solo en train).

## Cómo se evaluó

Con un único split el R² de test varía mucho según la semilla (entre ≈ 0.43 y 0.74 para
LinearRegression). Por eso cada configuración se corrió con **5 semillas** (`RANDOM_STATE` =
42, 1, 2, 3, 4). Para comparar configuraciones dentro de un mismo notebook se usó la
**diferencia pareada por semilla** (misma partición train/test), cuyo desvío es mucho menor
que el desvío bruto entre semillas (≈ 0.003–0.011 contra ≈ 0.11). Los notebooks estratificado
y no estratificado **no** se pueden parear: con la misma semilla las filas de test son distintas.

Los valores de `alpha` de Lasso/Ridge/ElasticNet se eligen con `cv=5` dentro de train en cada
semilla; test se usa una única vez para la métrica final.

## Resultados

### 1. Estratificación e imputación (LinearRegression, R² de test)

| Configuración | Semilla 42 |
|---|---|
| Original (sin estratificar, CatBoost) | 0.589 |
| Sin estratificar, moda | 0.603 |
| Estratificado, CatBoost | 0.709 |
| Estratificado, moda | 0.709 |

Promedio de 5 semillas: original 0.562 ± 0.113 y estratificado + moda 0.596 ± 0.119 (RMSE
5.88 y 6.07). La imputación por moda cambia el resultado ≈ 0.014; el salto de la semilla 42
proviene de que el split estratificado deja otras filas en test. No hay evidencia de que estratificar
mejore el modelo ni de que reduzca la variabilidad entre semillas.

### 2. Colinealidad (promedio de 5 semillas, LinearRegression, R² de test / RMSE de test)

| Notebook | Todas | Sin RAD | Sin TAX | Sin ambas |
|---|---|---|---|---|
| Original | 0.562 / 5.880 | 0.554 / 5.937 | 0.554 / 5.959 | 0.550 / 5.987 |
| Estratificado + moda | 0.596 / 6.066 | 0.593 / 6.092 | 0.582 / 6.180 | 0.580 / 6.195 |

Diferencia pareada del R² de test contra "todas" (media ± desvío de la diferencia entre semillas):

| Notebook | Sin RAD | Sin TAX | Sin ambas |
|---|---|---|---|
| Original | −0.008 ± 0.008 | −0.008 ± 0.053 | −0.013 ± 0.058 |
| Estratificado + moda | −0.003 ± 0.004 | −0.014 ± 0.010 | −0.016 ± 0.008 |

Coeficientes estandarizados de LinearRegression (media ± desvío entre semillas):

| Notebook | Modelo | TAX | RAD |
|---|---|---|---|
| Original | Todas | −2.99 ± 0.69 | +1.11 ± 0.34 |
| Original | Sin RAD | −2.38 ± 0.52 | – |
| Original | Sin TAX | – | −0.79 ± 0.15 |
| Estratificado + moda | Todas | −2.74 ± 0.29 | +1.00 ± 0.34 |
| Estratificado + moda | Sin RAD | −2.19 ± 0.15 | – |
| Estratificado + moda | Sin TAX | – | −0.73 ± 0.14 |

Conclusión: sacar RAD cuesta muy poco en R² (−0.003 a −0.008) y estabiliza el coeficiente de
TAX. El signo de RAD pasa de + a − cuando se saca TAX, evidencia de colinealidad. Sacar TAX
cuesta más porque tiene más relación con MEDV (r = −0.44 contra −0.34 de RAD). Ninguna variante
mejora el rendimiento predictivo: la exclusión se justifica por interpretación, no por métricas.

### 3. Transformación Yeo-Johnson sobre B (R² de test, promedio de 5 semillas)

| Notebook | LinearRegression sin transformar | Con Yeo-Johnson |
|---|---|---|
| Original | 0.562 | 0.554 |
| Estratificado + moda | 0.596 | 0.590 |

B tiene una cola izquierda marcada (skew ≈ −2.5); Yeo-Johnson (λ ≈ 3.4, ajustado solo en
train) lo deja en ≈ −1.7. No mejora las métricas (en el original es peor en 5 de 5 semillas),
por lo que B se deja sin transformar.

### 4. Selección de `lr` y `epochs` de Gradient Descent (R² de test de GD, promedio de 5 semillas)

| Método de selección | Original | Estratificado |
|---|---|---|
| Mirando test (con fuga, versión anterior) | 0.554 | 0.593 |
| Holdout 80/20 dentro de train | 0.551 | 0.586 |
| K-Fold (5 folds) dentro de train | 0.550 | 0.591 |
| LinearRegression (referencia) | 0.554 | 0.593 |

El valor obtenido eligiendo con test era levemente optimista. Con K-Fold la elección es más
estable (7 de 10 corridas eligen `lr=0.01, epochs=500`). Cuando GD converge (lr de 0.05 a 0.1)
coincide con la solución de mínimos cuadrados, y `lr=0.3` diverge. Los resultados de GD quedan a
menos de 0.005 de R² de LinearRegression en promedio.

## Notas y limitaciones

- Con solo 5 semillas los estadísticos son orientativos; las diferencias son chicas (≤ 0.02 de R²).
- Las cifras del notebook corresponden a la semilla 42; las tablas de este documento resumen
  corridas con las 5 semillas que se hicieron cambiando `RANDOM_STATE`.
- `prueba-colinealidad` usa CatBoost para imputar CHAS. `catboost` no figura en `requirements.txt`;
  el notebook lo instala con `!pip install catboost`, y localmente hace falta `pip install catboost`.
- Los R² y RMSE de este documento pueden diferir en la tercera cifra decimal de los del notebook
  según la versión de las librerías.
