"""
=============================================================================
MÓDULO 6: EVALUACIÓN Y CALIBRACIÓN DEL SISTEMA
=============================================================================
Herramientas para medir y mejorar la calidad del sistema:
  - Métricas estándar de clasificación (precision, recall, F1)
  - Análisis de falsos positivos y negativos
  - Curvas de calibración de umbrales
  - Framework de anotación humana para ground truth
  - Detección de sesgo en grupos demográficos
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict
import json

try:
    import numpy as np
    from sklearn.metrics import (
        precision_recall_fscore_support,
        confusion_matrix,
        roc_auc_score,
        classification_report,
    )
    from sklearn.calibration import calibration_curve
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from data_structures import ConflictSeverity


@dataclass
class AnnotatedMessage:
    """Mensaje con etiqueta humana para ground truth."""
    msg_id:          str
    text:            str
    human_label:     ConflictSeverity   # Etiqueta del anotador humano
    predicted_label: ConflictSeverity   # Predicción del sistema
    annotator_id:    str                # Para medir acuerdo inter-anotador
    notes:           str = ""           # Comentarios del anotador


class SystemEvaluator:
    """
    Evalúa el rendimiento del sistema contra anotaciones humanas.

    Métricas clave para sistemas de moderación:
      - Recall alto en CRITICAL/HIGH → no perder casos graves
      - Precision alta en general   → no sobrecargar a moderadores
      - F1-beta con beta>1          → penaliza más los falsos negativos
    """

    def __init__(self, beta: float = 1.5):
        """
        Args:
            beta: Peso del recall en F-beta. >1 → penaliza más FN que FP.
                  Recomendado: 1.5 para moderación de comunidades.
        """
        self.beta = beta

    def evaluate(
        self, annotations: List[AnnotatedMessage]
    ) -> Dict:
        """
        Calcula todas las métricas de evaluación.
        Devuelve un diccionario estructurado con resultados y análisis.
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn no disponible — instalar con pip install scikit-learn"}

        y_true = [a.human_label.value for a in annotations]
        y_pred = [a.predicted_label.value for a in annotations]

        # Métricas por clase
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4],
            zero_division=0,
        )

        # F-beta global
        _, _, fbeta, _ = precision_recall_fscore_support(
            y_true, y_pred,
            beta=self.beta, average="weighted", zero_division=0
        )

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

        # Análisis de falsos positivos/negativos por severidad
        fp_fn_analysis = self._analyze_errors(annotations)

        results = {
            "overall": {
                f"f{self.beta}_score": float(fbeta),
                "accuracy": sum(1 for a in annotations
                                if a.human_label == a.predicted_label) / len(annotations),
            },
            "per_class": {
                severity.name: {
                    "precision": float(precision[i]),
                    "recall":    float(recall[i]),
                    "f1":        float(f1[i]),
                    "support":   int(support[i]),
                }
                for i, severity in enumerate(ConflictSeverity)
            },
            "confusion_matrix": cm.tolist(),
            "error_analysis":   fp_fn_analysis,
            "recommendations":  self._generate_recommendations(
                precision, recall, f1
            ),
        }
        return results

    def _analyze_errors(
        self, annotations: List[AnnotatedMessage]
    ) -> Dict:
        """
        Clasifica los errores para entender los puntos débiles del sistema.
        Categorías de error más importantes para moderación:
          - Falsos negativos CRÍTICOS: perder un HIGH/CRITICAL
          - Falsos positivos en NONE: marcar como conflicto lo que no lo es
        """
        false_negatives_critical = []
        false_positives_normal   = []
        over_classifications     = []

        for ann in annotations:
            true_val = ann.human_label.value
            pred_val = ann.predicted_label.value

            # Error grave: predijo LOW/NONE pero era HIGH/CRITICAL
            if true_val >= 3 and pred_val <= 1:
                false_negatives_critical.append({
                    "msg_id": ann.msg_id,
                    "text":   ann.text[:100],
                    "true":   ann.human_label.name,
                    "pred":   ann.predicted_label.name,
                })

            # Error de sobre-alerta: predijo MODERATE+ pero era NONE
            elif true_val == 0 and pred_val >= 2:
                false_positives_normal.append({
                    "msg_id": ann.msg_id,
                    "text":   ann.text[:100],
                    "true":   ann.human_label.name,
                    "pred":   ann.predicted_label.name,
                })

            # Sobre-clasificación (predijo un nivel más arriba)
            elif pred_val == true_val + 2:
                over_classifications.append({
                    "msg_id": ann.msg_id,
                    "text":   ann.text[:100],
                    "over_by": pred_val - true_val,
                })

        return {
            "critical_misses":        false_negatives_critical,
            "false_alarms":           false_positives_normal,
            "over_classifications":   over_classifications,
            "summary": {
                "critical_misses_count":      len(false_negatives_critical),
                "false_alarms_count":         len(false_positives_normal),
                "over_classifications_count": len(over_classifications),
            }
        }

    def _generate_recommendations(self, precision, recall, f1) -> List[str]:
        """Genera recomendaciones automáticas basadas en los resultados."""
        recs = []
        sev_names = [s.name for s in ConflictSeverity]

        for i, name in enumerate(sev_names):
            if i == 0:
                continue  # Ignorar NONE
            if recall[i] < 0.60:
                recs.append(
                    f"Recall bajo ({recall[i]:.0%}) para {name}. "
                    f"Reducir umbral o añadir más ejemplos de entrenamiento."
                )
            if precision[i] < 0.50 and i <= 2:
                recs.append(
                    f"Precisión baja ({precision[i]:.0%}) para {name}. "
                    f"Aumentar umbral para reducir falsos positivos."
                )
        return recs

    def compute_inter_annotator_agreement(
        self, annotations_a: List[AnnotatedMessage],
        annotations_b: List[AnnotatedMessage],
    ) -> Dict:
        """
        Mide el acuerdo entre dos anotadores humanos usando Cohen's Kappa.
        Valores de referencia:
          κ < 0.20: acuerdo pobre
          κ 0.40–0.60: acuerdo moderado (aceptable para tareas complejas)
          κ > 0.80: acuerdo excelente
        """
        if not SKLEARN_AVAILABLE:
            return {}
        from sklearn.metrics import cohen_kappa_score

        labels_a = [a.human_label.value for a in annotations_a]
        labels_b = [a.human_label.value for a in annotations_b]

        kappa = cohen_kappa_score(labels_a, labels_b,
                                   weights="quadratic")

        return {
            "cohen_kappa": float(kappa),
            "interpretation": (
                "Excelente" if kappa > 0.8 else
                "Bueno"     if kappa > 0.6 else
                "Moderado"  if kappa > 0.4 else
                "Pobre — revisar guía de anotación"
            ),
        }


class ThresholdOptimizer:
    """
    Encuentra los umbrales óptimos para minimizar falsos positivos
    mientras mantiene un recall mínimo en casos graves.

    Útil para recalibrar el sistema con datos reales de tu comunidad.
    """

    def optimize(
        self,
        toxicity_scores: List[float],
        true_labels: List[int],
        min_recall_high: float = 0.85,   # Recall mínimo para HIGH+
    ) -> Dict:
        """
        Busca el umbral de toxicidad que:
          1. Mantiene recall ≥ min_recall_high para casos graves
          2. Maximiza la precisión general

        Args:
            toxicity_scores: Puntuaciones brutas del modelo [0,1]
            true_labels:     Etiquetas verdaderas (0-4)
            min_recall_high: Recall mínimo para HIGH/CRITICAL

        Returns:
            Diccionario con umbral óptimo y curva precision-recall.
        """
        if not SKLEARN_AVAILABLE:
            return {"optimal_threshold": 0.60}

        thresholds = np.arange(0.30, 0.90, 0.05)
        results = []

        high_mask = np.array([l >= 3 for l in true_labels])

        for thresh in thresholds:
            predicted = [1 if s >= thresh else 0 for s in toxicity_scores]
            predicted_high = [1 if s >= thresh else 0 for s in toxicity_scores]

            if sum(predicted) == 0:
                continue

            # Recall en HIGH+
            true_pos  = sum(1 for p, h in zip(predicted_high, high_mask) if p and h)
            false_neg = sum(1 for p, h in zip(predicted_high, high_mask) if not p and h)
            recall_high = true_pos / max(true_pos + false_neg, 1)

            # Precision general
            false_pos = sum(
                1 for p, l in zip(predicted, true_labels) if p and l == 0
            )
            precision = sum(predicted) / max(sum(predicted), 1)

            if recall_high >= min_recall_high:
                results.append({
                    "threshold":    float(thresh),
                    "recall_high":  float(recall_high),
                    "precision":    float(precision),
                    "f1":           2 * precision * recall_high / max(precision + recall_high, 1e-8),
                })

        if not results:
            return {"optimal_threshold": 0.60, "note": "No se encontró umbral con recall suficiente"}

        # Elegir el que maximice F1 manteniendo recall
        best = max(results, key=lambda x: x["f1"])
        return {
            "optimal_threshold": best["threshold"],
            "expected_recall_high": best["recall_high"],
            "expected_precision":   best["precision"],
            "all_results": results,
        }


# ---------------------------------------------------------------------------
# Guía de anotación — para construir el ground truth
# ---------------------------------------------------------------------------

ANNOTATION_GUIDE = """
═══════════════════════════════════════════════════════════════
GUÍA DE ANOTACIÓN PARA MODERADORES
═══════════════════════════════════════════════════════════════

REGLA FUNDAMENTAL: Anota lo que ves en el texto, no lo que
supones sobre la intención del autor.

NIVELES DE SEVERIDAD:

  NONE (0):
    • Conversación normal, debate de ideas
    • Desacuerdo expresado con respeto
    • Crítica constructiva
    Ejemplos: "Creo que te equivocas porque...", "No comparto esa opinión"

  LOW (1):
    • Tono ligeramente elevado pero no hostil
    • Frustración expresada de forma controlada
    • Sarcasmo muy leve, ambiguo
    Ejemplos: "No entiendo cómo puedes pensar eso", "Esto es un poco ridículo"

  MODERATE (2):
    • Hostilidad clara aunque sin insultos directos
    • Pasivo-agresivo evidente
    • Descalificación general del otro ("siempre haces esto")
    Ejemplos: "Típico de ti no entender", "Claro, como siempre tienes razón..."

  HIGH (3):
    • Insultos directos a la persona (no a las ideas)
    • Amenazas implícitas
    • Ataques sostenidos
    Ejemplos: "Eres un idiota", "Gente como tú no debería estar aquí"

  CRITICAL (4):
    • Amenazas explícitas
    • Acoso sistemático
    • Lenguaje de odio
    Ejemplos: [Amenazas directas, doxxing, incitación a la violencia]

CASOS DIFÍCILES:
  • Humor negro → LOW como máximo si no hay objetivo personal
  • Debate acalorado de ideas → máximo LOW aunque el tono sea fuerte
  • Ironía → requiere contexto; en duda, anota LOW
  • Lenguaje de comunidad/jerga → consultar con otro anotador

ACUERDO INTER-ANOTADOR:
  • Discrepancias de ±1 nivel son normales
  • Discrepancias de ±2 niveles requieren discusión
  • Objetivo: κ ≥ 0.60 (Cohen's Kappa)
═══════════════════════════════════════════════════════════════
"""

print(ANNOTATION_GUIDE)
