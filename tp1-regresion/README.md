# TP1 — Regresión: predicción de precios de casas

Aprendizaje Automático 1 — Tecnicatura en Inteligencia Artificial (FCEIA, UNR)

**Integrantes:** Aguirre, Gambande, Paladini

Notebook: [`TP-regresion-AA1.ipynb`](TP-regresion-AA1.ipynb)
Datos: [`data/house-prices-tp.csv`](data/house-prices-tp.csv)

## De qué se trata

Predecir el precio (`MEDV`) de casas de Boston a partir de sus características (criminalidad
de la zona, cantidad de habitaciones, cercanía a centros de empleo, etc.). Consigna completa
en el PDF de la cátedra.

## Qué hicimos

- EDA: análisis de cada variable, datos faltantes (se imputan con KNN, y CHAS con la moda),
  outliers y correlación entre variables.
- Split estratificado por `CHAS`, todo lo que se ajusta a los datos (imputación, escalado) se
  hace después del split, solo con train, para no filtrar información de test.
- Modelos: LinearRegression, Gradiente Descendente (implementado a mano), Ridge, Lasso y
  ElasticNet.
- Los hiperparámetros de GD (lr, epochs) y los alphas de Ridge/Lasso/ElasticNet se eligen con
  validación cruzada dentro de train, sin mirar el conjunto de test.
- Se mantienen todas las variables, incluida RAD, a pesar de su colinealidad con TAX (se
  discute el efecto en la sección 5.2).

## Resultado

Los cinco modelos quedan muy parejos (RMSE de test entre 5.08 y 5.11). El mejor por esa
métrica es LinearRegression (R² test 0.71, RMSE 5.08), pero la diferencia con los demás es
menor que la variabilidad que introduce la propia partición train/test (lo medimos en la
sección 6), así que no se puede decir que uno sea claramente mejor que el resto.

## Cómo correrlo

Instalación del entorno en el [README del repo](../README.md). Con el kernel activado, abrir
el notebook y correr todo de arriba a abajo (Restart Kernel + Run All).
