# SUSTENTACI-N_inteligencia_codigo

# Análisis Comparativo: t-SNE + DBSCAN vs GMM en MNIST

Este proyecto implementa y compara dos enfoques de clustering no supervisado, **t-SNE + DBSCAN** y **Gaussian Mixture Models (GMM)**, aplicados al dataset de dígitos manuscritos MNIST. El objetivo principal es evaluar la capacidad de estos algoritmos para descubrir la estructura latente en datos de alta dimensión y agruparlos de manera coherente con sus etiquetas reales, así como analizar su estabilidad y rendimiento.

## Contenido del Notebook

El notebook está estructurado en las siguientes secciones:

### 1. Encabezado y Configuraciones
Se importan las librerías necesarias (NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, etc.) y se definen parámetros globales configurables para el experimento, como el número de muestras, semillas para reproducibilidad, y configuraciones específicas para PCA, t-SNE, DBSCAN y GMM.

### 2. Carga y Preprocesamiento del Dataset
Se descarga y carga el dataset MNIST (submuestreado a 10,000 muestras para entrenamiento y 2,000 para prueba). Los datos son normalizados y se presenta un resumen de las formas y distribución de clases de los conjuntos de entrenamiento y prueba.

### 3. Análisis Exploratorio Inicial
Se realiza un análisis básico de las estadísticas de intensidad de píxeles y se visualizan ejemplos de dígitos para cada clase. También se examinan las distribuciones de intensidad media y desviación estándar por clase.

### 4. Análisis Exploratorio en Alta Dimensión (PCA)
Se aplica un análisis de componentes principales (PCA) para comprender la varianza de los datos en alta dimensión. Se generan gráficos de scree plot y varianza acumulada para determinar un número adecuado de componentes. También se visualiza una proyección 3D de las primeras componentes principales.

### 5. Embeddings: PCA y t-SNE
Se utiliza PCA para reducir la dimensionalidad de los datos a 50 componentes (según la configuración inicial) y, posteriormente, se aplica t-SNE para obtener una representación bidimensional de los datos. Se evalúa el impacto de diferentes hiperparámetros de t-SNE (perplexidad y learning rate) en la calidad de la visualización y se elige una configuración base para el análisis posterior.

### 6. Medidas de Preservación Local/Global
Se evalúa la calidad de la reducción de dimensionalidad de t-SNE mediante métricas como la preservación de k-vecinos más cercanos (k-NN preservation) y el análisis de distancias intra-clase e inter-clase en los espacios original (PCA) y embebido (t-SNE).

### 7. DBSCAN: Análisis y Clustering
Se explora el algoritmo DBSCAN. Se utiliza un k-distance plot para ayudar en la selección del parámetro `eps` y se realiza un barrido de parámetros (`eps` y `min_samples`) para encontrar la configuración óptima basada en métricas de clustering como ARI (Adjusted Rand Index), NMI (Normalized Mutual Information) y Silhouette Score. Se visualizan los resultados del mejor modelo.

### 8. GMM (EM): Ajuste y Evaluación
Se implementa Gaussian Mixture Models (GMM). Se realiza un barrido de parámetros (`n_components` y `covariance_type`) para encontrar la mejor configuración utilizando métricas como BIC (Bayesian Information Criterion) y AIC (Akaike Information Criterion), así como ARI. Se visualizan los clusters resultantes, la entropía por muestra (indicador de ambigüedad) y los centros de los componentes.

### 9. Validación en Conjunto de Prueba y Matriz de Confusión Real
Se evalúa el rendimiento de los modelos DBSCAN y GMM en un conjunto de prueba independiente. Para ello, se genera un embedding t-SNE para el conjunto de prueba y se mapean los clusters a las etiquetas reales del dataset de entrenamiento. Se calculan métricas como ARI, NMI y Accuracy, y se generan matrices de confusión detalladas para identificar las confusiones específicas entre dígitos.

### 10. Evaluación Comparativa y Métricas
Esta sección presenta una comparación directa de DBSCAN y GMM utilizando métricas como ARI, NMI, Silhouette Score, Davies-Bouldin Index, número de clusters y tiempo de ejecución. Se realizan evaluaciones con múltiples semillas para analizar la estabilidad de los resultados y se utilizan tests estadísticos (t-test pareado, Wilcoxon) para determinar si existen diferencias significativas en el rendimiento de los algoritmos. También se visualizan la distribución de los coeficientes de Silhouette y los tamaños de los clusters.

### 11. Casos de Estudio
Se exploran casos específicos como puntos con alta ambigüedad detectados por GMM y outliers identificados por DBSCAN, visualizando sus imágenes reales. También se muestran mosaicos de imágenes por cluster para GMM y se analizan las confusiones más comunes entre pares de dígitos.

### 12. Resumen Final y Conclusiones
Se presenta una tabla resumen final que consolida todos los hallazgos y se extraen conclusiones sobre el rendimiento de cada método, destacando sus fortalezas y debilidades. Se recomienda un método basado en el rendimiento de las métricas clave.

## Resultados Clave

Basado en la evaluación comparativa, los resultados promedio (media ± desviación estándar) de múltiples ejecuciones con diferentes semillas son los siguientes:

| Métrica      | DBSCAN          | GMM             |
|--------------|-----------------|-----------------|
| ARI          | 0.802 ± 0.018   | 0.456 ± 0.015   |
| NMI          | 0.832 ± 0.008   | 0.636 ± 0.016   |
| Silhouette   | -0.109 ± 0.020  | 0.052 ± 0.002   |

**Conclusiones Principales:**

*   **DBSCAN** mostró una **superioridad significativa** en el **Adjusted Rand Index (ARI)**, indicando una mejor concordancia con las etiquetas reales de los dígitos. Su capacidad para identificar clusters de densidad variable y marcar outliers fue ventajosa para este dataset.
*   **GMM** obtuvo un **mejor coeficiente de Silhouette promedio**, lo que sugiere que sus clusters son más densos y mejor separados internamente. Sin embargo, su ARI fue considerablemente inferior, lo que indica que sus agrupaciones no se alinearon tan bien con las clases de dígitos reales.
*   **Estabilidad:** Ambos métodos mostraron una variabilidad relativamente baja en sus métricas a través de diferentes semillas, con GMM ligeramente más estable en ARI.
*   **Rendimiento en Test:** La validación en el conjunto de prueba reflejó el rendimiento observado en el entrenamiento, con DBSCAN mostrando mejor capacidad de generalización en términos de precisión cuando los clusters se mapean a etiquetas.

## Uso

Para ejecutar este análisis:

1.  Abre el notebook en Google Colab.
2.  Asegúrate de tener las dependencias listadas en la primera celda (`!pip install`) instaladas.
3.  Ejecuta las celdas secuencialmente para replicar el análisis.

--- 
