# ToDo – Experimentos de Alto Impacto para el Paper

## 🧪 1. VLM vs Detectores (CRÍTICO)
**Objetivo:** Comparar VLMs con detectores clásicos en CPU.

**Modelos:**
- VLMs actuales
- YOLOv8n (CPU)
- YOLOv5n (CPU)

**Tasks:**
- Detección binaria de texto (sí/no)
- Opcional: localización aproximada

**Métricas:**
- Latencia
- Memoria
- Accuracy

---

## 🧪 2. Scaling Law en CPU
**Objetivo:** Mostrar relación entre tamaño del modelo y eficiencia.

**Ejes:**
- X: tamaño modelo
- Y: latencia / accuracy / efficiency

**Resultado esperado:**
- Curva no lineal
- Punto óptimo

---

## 🧪 3. Robustez a Prompts
**Objetivo:** Medir sensibilidad a variaciones de prompt.

**Prompts:**
1. What does the image say?
2. Extract all text
3. Read the text exactly

**Métricas:**
- Variación de accuracy
- Variación de latencia

---

## 🧪 4. Warm vs Cold Start
**Objetivo:** Medir latencia real en producción.

**Comparar:**
- Primera inferencia
- Inferencias siguientes

**Métrica:**
- Diferencia de latencia (%)

---

## 🧪 5. Batch vs Single
**Objetivo:** Evaluar escalabilidad en CPU.

**Configuración:**
- Batch = 1
- Batch = 4 / 8

**Métricas:**
- Latencia por imagen
- Throughput

---

## 🧪 6. Métrica de Eficiencia
**Propuesta:**

Efficiency Score = Accuracy / (Latency × Memory)

Opcional:
Cost Efficiency = Accuracy / (Latency × Energy)

---

## 🧪 7. Análisis de Errores
**Clasificación:**
- OCR failure
- Hallucination
- Partial extraction
- Format errors

**Output:**
Tabla comparativa por modelo

---

## 🧪 8. Cuantización Controlada
**Objetivo:** Validar impacto real.

**Setup:**
- FP16
- INT8
- GGUF (si posible)

**Métricas:**
- Latencia
- Accuracy drop

---

## 🚀 Plan mínimo recomendado

Priorizar:
1. VLM vs YOLO
2. Scaling law
3. Prompt robustness
4. Efficiency score

---

## 🎯 Objetivo final

Pasar de:
“Esto funciona así”

A:
“Estas son las leyes de eficiencia de VLMs en CPU”
