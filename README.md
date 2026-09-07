# TP_AA1_Aguirre_Gambande_Paladini

Trabajo Práctico: Predicción de precios de casas — Aprendizaje Automático 1
Tecnicatura en Inteligencia Artificial — FCEIA (UNR)

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Registrar el entorno como kernel de Jupyter (una sola vez):

```bash
python -m ipykernel install --user --name=tp-aa1
```

Luego, al abrir `TP-regresion-AA1.ipynb`, seleccionar el kernel `tp-aa1`.

## Objetivos

Familiarizarse con la biblioteca scikit-learn y las herramientas que brinda para el
preprocesamiento de datos, la implementación de modelos de regresión lineal con diversos
hiperparámetros y la evaluación de métricas.

## Dataset

El dataset se llama `house-prices.csv` y contiene información de precios de casas de Boston,
además de otras variables características, detalladas a continuación:

**Características de entrada (en orden):**

1. **CRIM**: tasa de criminalidad per cápita por ciudad
2. **ZN**: proporción de terrenos residenciales zonificados para lotes de más de 25,000 pies cuadrados
3. **INDUS**: proporción de acres de negocios no minoristas por ciudad
4. **CHAS**: variable dummy del río Charles (1 si el tramo limita con el río; 0 de lo contrario)
5. **NOX**: concentración de óxidos de nitrógeno (partes por 10 millones) [parts/10M]
6. **RM**: número promedio de habitaciones por vivienda
7. **AGE**: proporción de unidades ocupadas por sus propietarios construidas antes de 1940
8. **DIS**: distancias ponderadas a cinco centros de empleo de Boston
9. **RAD**: índice de accesibilidad a las autopistas radiales
10. **TAX**: tasa de impuesto sobre la propiedad a valor completo por $10,000 [$/10k]
11. **PTRATIO**: proporción alumno-maestro por ciudad
12. **B**: resultado de la ecuación B=1000(Bk - 0.63)^2, donde Bk es la proporción de negros por ciudad
13. **LSTAT**: % de población de menor estatus socioeconómico

**Variable de salida (target):**

14. **MEDV**: valor mediano de las viviendas ocupadas por sus propietarios, en miles de dólares [k$]

> Para todos los ítems, incorporar una cantidad de texto adecuado en forma de comentarios,
> ya sea para la comprensión del código (usualmente una línea de comentario por cada celda)
> como para explicar las decisiones tomadas a lo largo del trabajo (por ejemplo, la
> justificación de la imputación de valores faltantes, la elección de las métricas
> adecuadas, entre otros). Mantener la coherencia con los comentarios.

## Consignas

1. Armar grupos de tres personas para la realización del trabajo práctico. En caso de no
   tener compañero, informar al cuerpo docente. Se recomienda que al menos dos integrantes
   hayan aprobado Fundamentos de Ciencias de Datos.
2. Crear un repositorio que se llame "TP_AA1_Apellido1_Apellido2_apellido3" en GitHub.
3. Realizar un análisis descriptivo, que ayude a la comprensión del problema, de cada una de
   las variables involucradas en el problema detallando características, comportamiento y
   rango de variación.

   Debe incluir:
   - Análisis y decisión sobre datos faltantes.
   - Visualización de datos (por ejemplo histogramas, scatterplots entre variables, diagramas de caja).
   - Codificación de variables categóricas (si se van a utilizar para predicción).
   - Matriz de correlación de variables.
   - Estandarización o escalado de datos.
   - Validación cruzada train - test. Realizar una división del conjunto de datos en
     conjuntos de entrenamiento y prueba (y si se quiere, se puede incluir validación, que
     luego será útil) en el MOMENTO donde ustedes lo crean adecuado.

4. Implementar la solución del problema de regresión con regresión lineal múltiple.
   - Probar con el método LinearRegression.
   - Probar con métodos de gradiente descendente. ¿Algún cambio? Incorporar gráficas de
     Error vs Iteraciones (loss vs epochs). Agregar comentarios.
   - Probar con métodos de regularización (Lasso, Ridge, Elastic Net).
   - Obtener las métricas adecuadas (entre R² Score, MSE, RMSE, MAE, MAPE, elegir) tanto
     para entrenamiento como para prueba. ¿Por qué para ambos conjuntos?
   - ¿Creen que han conseguido un buen fitting?

5. Optimizar la selección de hiperparámetros.
   - Variar los hiperparámetros de gradiente descendente. ¿Qué observa?
   - Variar los hiperparámetros de Lasso y Ridge. ¿Qué observa?

6. Comparación de modelos.
   - Incluyan en su análisis una comparación de modelos: de todos los modelos de regresión,
     ¿cuál es el mejor? Escoger una métrica adecuada para poder compararlos.

7. Escribir una conclusión del trabajo.

8. Preparar una defensa del trabajo práctico: la defensa consiste en preguntas hechas por el
   cuerpo docente que pueden ser: explicar una parte del código, explicar alguno de los
   métodos utilizados, preguntas de índole teórica, preguntas de índole práctica. Son tanto
   grupales como individuales.

## Entrega

El repositorio debe llamarse "TP_AA1_Apellido1_Apellido2_Apellido3" sin excepciones.

**Respetar los siguientes nombres:**

Notebook de trabajo: `TP-regresion-AA1.ipynb`

**Fecha de entrega:** Lunes 28/09/2026

Cada entrega puede demorar hasta dos días después de la fecha pactada, **con disminución de
la nota final del trabajo práctico**.

**No se aceptan entregas finales con fecha posterior al 30/09/2026.** En caso de no tener
todos los ítems entregados para esta fecha, la condición es automáticamente desaprobada.

La defensa de los TP se hará de forma presencial y virtual, en horarios de clase, separados
por turnos que la cátedra asignará según el orden en el que se fue entregando. En caso de
detectar errores o una presentación en la que falten conocimientos sobre el trabajo
realizado que se consideren lo suficientemente graves, se pactará una fecha para una
segunda defensa en mesas de examen (donde deberán estar realizadas las correcciones y más
acertada la presentación). En caso de reprobar en esta segunda instancia de defensa, la
condición es de libre.

En caso de no aprobar ni la defensa del TP de regresión ni el de clasificación, la condición
es de libre.
