# Experimentos sobre el TP1 de regresión

Este documento resume las decisiones de criterio que se tomaron en el notebook del TP
(`TP-regresion-AA1.ipynb`) y los experimentos que las respaldan. **Los experimentos se
pueden ver y volver a ejecutar en [`experimentos/experimentos-AA1.ipynb`](experimentos/experimentos-AA1.ipynb)**:
ahí está el código de cada prueba y todas las tablas y gráficos se generan al ejecutarlo
(los resultados también se guardan como CSV en [`experimentos/resultados/`](experimentos/resultados/)).
Las cifras de este documento salen de ese notebook.

## Ramas

| Rama | Parte de | Qué contiene |
|---|---|---|
| `aguirre` | – | Notebook original: split sin estratificar, CHAS imputada con CatBoost, hiperparámetros de Gradient Descent elegidos mirando el R² de test. |
| `prueba-estratificacion` | `aguirre` | Split estratificado por CHAS, CHAS imputada con la moda, RAD excluida, `lr`/`epochs` de GD con K-Fold, y el notebook de experimentos. |
| `prueba-colinealidad` | `aguirre` | Notebook original (sin estratificar, con CatBoost) + RAD excluida + `lr`/`epochs` de GD con K-Fold, y el notebook de experimentos. |

Para ver las diferencias entre ramas:

```bash
git diff aguirre prueba-estratificacion -- tp1-regresion/TP-regresion-AA1.ipynb
git log --oneline aguirre..prueba-estratificacion
git log --oneline aguirre..prueba-colinealidad
```

El `diff` de un `.ipynb` es ruidoso porque incluye las salidas de las celdas. Las celdas de
código modificadas son pocas y están comentadas.

## Cambios en el notebook del TP

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

Con un único split el R² de test varía mucho según la semilla (entre ≈ 0.43 y 0.75 para
LinearRegression). Por eso cada configuración se corrió con **5 semillas** (`random_state` del
split = 42, 1, 2, 3, 4), y con 10 en el experimento 6. Para comparar configuraciones dentro de un
mismo experimento se usó la **diferencia pareada por semilla** (misma partición train/test), cuyo
desvío es mucho menor que el desvío bruto entre semillas. Los esquemas de estratificación (experimento 6)
no se pueden parear, porque cambian las filas de test.

Cada experimento cambia **un solo factor** respecto de una configuración base (split estratificado
por CHAS, CHAS con la moda, todas las variables, `lr`/`epochs` con K-Fold y grilla por defecto de
`RidgeCV`). Los valores de `alpha` de Lasso/Ridge/ElasticNet se eligen con `cv=5` dentro de train
en cada semilla; test se usa una única vez para la métrica final. El notebook de experimentos
incluye una celda que comprueba que su pipeline reproduce los resultados del notebook del TP
para la semilla 42.

## Resultados

Todas las cifras son medias entre semillas; "±" es el desvío entre semillas (o, en las diferencias
pareadas, el desvío de la diferencia). "Peor en n/5" cuenta en cuántas semillas la configuración
empeora respecto de la base.

### 1. Estratificación e imputación de CHAS (LinearRegression, 5 semillas)

| Configuración | R² de test | RMSE de test |
|---|---|---|
| Sin estratificar + CatBoost (notebook original) | 0.562 ± 0.113 | 5.880 ± 0.875 |
| Sin estratificar + moda | 0.564 ± 0.116 | 5.867 ± 0.910 |
| Estratificado + CatBoost | 0.595 ± 0.118 | 6.074 ± 1.007 |
| Estratificado + moda | 0.596 ± 0.119 | 6.066 ± 1.004 |

Imputar con la moda en lugar de CatBoost, con el mismo split: +0.002 ± 0.010 de R² sin estratificar
(peor en 2/5 semillas) y +0.001 ± 0.005 estratificando (peor en 1/5). O sea, no hay diferencia apreciable.
Estratificar por CHAS deja otras filas en test, por lo que su columna no es comparable de forma pareada
con la de sin estratificar; en promedio no hay evidencia de que mejore el modelo.

### 2. Colinealidad

VIF (filas completas): TAX 9.0, RAD 7.5, NOX 4.4, INDUS 4.0, DIS 4.0; correlación RAD–TAX 0.88.

R² de test de LinearRegression, 5 semillas, y diferencia pareada contra "todas":

| Base | Todas | Sin RAD | Sin TAX | Sin ambas |
|---|---|---|---|---|
| Estratificado + moda (TP) | 0.596 ± 0.119 | 0.593 ± 0.116 | 0.582 ± 0.115 | 0.580 ± 0.113 |
| Δ pareada | – | −0.003 ± 0.004 (peor en 4/5) | −0.014 ± 0.010 (peor en 5/5) | −0.016 ± 0.008 (peor en 5/5) |
| Original (sin estratificar + CatBoost) | 0.562 ± 0.113 | 0.554 ± 0.114 | 0.554 ± 0.090 | 0.550 ± 0.097 |
| Δ pareada | – | −0.008 ± 0.008 (peor en 5/5) | −0.008 ± 0.053 (peor en 4/5) | −0.013 ± 0.058 (peor en 4/5) |

Coeficientes estandarizados de LinearRegression (base estratificado + moda, media ± desvío entre semillas):

| Modelo | TAX | RAD |
|---|---|---|
| Todas | −2.74 ± 0.29 | +1.00 ± 0.34 |
| Sin RAD | −2.19 ± 0.15 | – |
| Sin TAX | – | −0.73 ± 0.14 |

Sacar RAD cuesta muy poco en R² y estabiliza el coeficiente de TAX (el desvío baja de 0.29 a 0.15).
El signo de RAD pasa de + a − cuando se saca TAX, evidencia de colinealidad. Sacar TAX cuesta más.
Ninguna variante mejora el rendimiento predictivo: la exclusión se justifica por interpretación, no por métricas.

### 3. Transformación Yeo-Johnson sobre B (base estratificado + moda, 5 semillas)

B tiene skew −2.57; con Yeo-Johnson queda en −1.76. R² de test de LinearRegression: 0.596 ± 0.119 sin
transformar y 0.590 ± 0.122 con Yeo-Johnson (Δ pareada −0.006 ± 0.010, peor en 3/5 semillas). No mejora
las métricas, por lo que B se deja sin transformar.

### 4. Selección de `lr` y `epochs` de Gradient Descent (base estratificado + moda, 5 semillas)

| Método de selección | R² de test de GD |
|---|---|
| `lr=0.1, epochs=200` (elegidos mirando test, con fuga) | 0.596 ± 0.119 |
| Holdout 80/20 dentro de train | 0.589 ± 0.112 |
| K-Fold (5 folds) dentro de train | 0.594 ± 0.115 |
| LinearRegression (referencia) | 0.596 ± 0.119 |

Diferencia pareada de GD contra los valores con fuga: K-Fold −0.002 ± 0.005 (peor en 4/5) y holdout
−0.008 ± 0.013 (peor en 3/5). La combinación elegida cambia con la semilla (K-Fold elige `lr=0.01, epochs=500`,
`lr=0.1, epochs=500`, `lr=0.05, epochs=200` o `lr=0.01, epochs=200`). Cuando GD converge coincide con la solución
de mínimos cuadrados, y `lr=0.3` diverge; el efecto de elegir con fuga es pequeño.

### 5. Grilla de `RidgeCV` (base estratificado + moda, 5 semillas)

`RidgeCV(cv=5)` usa por defecto `alphas=(0.1, 1, 10)` y elige siempre `alpha = 10`, el extremo superior.
Con `np.logspace(-2, 3, 50)` el `alpha` elegido es mayor a 10 en las 5 semillas (de 11.5 a 59.6), pero el
rendimiento en test no mejora: Δ R² −0.004 ± 0.013 (peor en 3/5) y Δ RMSE +0.044 ± 0.112. La grilla por defecto
estaba acotada, pero ampliarla no cambia las conclusiones.

### 6. Esquemas de estratificación (10 semillas, LinearRegression)

| Esquema | Media | Desvío entre semillas | Mín – máx |
|---|---|---|---|
| Sin estratificar | 0.564 | 0.094 | 0.433 – 0.672 |
| Por CHAS (NaN como tercer estrato) | 0.600 | 0.088 | 0.471 – 0.736 |
| Por MEDV (5 quintiles) | 0.616 | 0.070 | 0.501 – 0.748 |
| Por CHAS × MEDV (3 terciles) | 0.631 | 0.058 | 0.547 – 0.718 |

Estratificar por el target reduce la variabilidad de la evaluación entre semillas: el desvío pasa de
0.094 a 0.070 con MEDV y a 0.058 combinando CHAS y MEDV; estratificar solo por CHAS lo reduce poco.
Ridge, Lasso y ElasticNet muestran el mismo patrón. El modelo es el mismo en todos los casos; las
diferencias entre esquemas no son pareables y parte del aumento de la media se debe a que un split
aleatorio a veces deja un test con poca dispersión de MEDV, lo que baja el R². Con 10 semillas la
estimación de un desvío es imprecisa, así que el resultado es sugerente pero no concluyente.

## Notas y limitaciones

- Con solo 5 semillas (10 en el experimento 6) los estadísticos son orientativos; las diferencias son chicas
  (≤ 0.02 de R²) salvo las del experimento 6.
- Los experimentos parten de una base con todas las variables, mientras que el notebook del TP en
  `prueba-estratificacion` excluye RAD; por eso algunas cifras (por ejemplo las de Gradient Descent)
  no coinciden en la tercera cifra decimal con las del TP.
- `prueba-colinealidad` usa CatBoost para imputar CHAS. `catboost` no figura en `requirements.txt`;
  el notebook lo instala con `!pip install catboost`, y localmente hace falta `pip install catboost`.
  El notebook de experimentos también lo necesita solo para las configuraciones con CatBoost (si no está
  instalado, las saltea).
- Los R² y RMSE pueden diferir en la tercera cifra decimal según la versión de las librerías.
