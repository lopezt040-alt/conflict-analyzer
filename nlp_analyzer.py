"""
=============================================================================
MÓDULO 2: ANALIZADOR NLP MULTI-MODELO
=============================================================================
Combina varios modelos especializados para obtener señales complementarias.
Cada señal tiene su propio umbral y peso, evitando que un solo modelo
domine la decisión final (reduce falsos positivos).

Modelos usados:
  1. Detoxify         → toxicidad multi-clase (rápido, ligero)
  2. cardiffnlp/twitter-roberta → sentimiento contextual
  3. martin-ha/toxic-comment-model → segunda opinión de toxicidad
  4. Heurísticas lingüísticas → pasivo-agresivo, provocación
"""

import re
import logging
from typing import Tuple
import numpy as np

# Importaciones opcionales: el sistema funciona en modo degradado si falta alguna
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers no disponible — usando solo heurísticas")

try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    DETOXIFY_AVAILABLE = False
    logging.warning("detoxify no disponible")

from data_structures import Message, ConflictSeverity


# ---------------------------------------------------------------------------
# Patrones lingüísticos para heurísticas de bajo coste
# (complementan los modelos; no reemplazan)
# ---------------------------------------------------------------------------

# Pasivo-agresivo: sarcasmo, condescendencia velada, invalidación
PASSIVE_AGGRESSIVE_PATTERNS = [
    r"\blo que sea\b", r"\bsi tú lo dices\b", r"\bclaro, claro\b",
    r"\bpues nada\b", r"\bcomo siempre\b", r"\btípico de ti\b",
    r"\bsuerte con eso\b", r"\bno esperaba menos\b",
    r"\btú sabrás\b", r"\bimpresionante.*ironía\b",
    # Inglés (para comunidades mixtas)
    r"\bwhatever\b", r"\bif you say so\b", r"\bsure jan\b",
    r"\btypical\b", r"\bgood luck with that\b", r"\bfine\b",
]

# Provocación directa: desafíos, invitaciones a pelear
PROVOCATION_PATTERNS = [
    r"\ba ver si\b.*\bte atreves\b", r"\bdemuéstralo\b",
    r"\bvente si puedes\b", r"\bno puedes\b", r"\bcobarde\b",
    r"\bbusca pelea\b", r"\bprovoca\b",
    r"come at me", r"fight me", r"prove it", r"i dare you",
]

# Escaladores de hostilidad: insultos directos
HOSTILITY_PATTERNS = [
    r"\bidiota\b", r"\bestúpido\b", r"\bimbécil\b", r"\bcretino\b",
    r"\bmente enferma\b", r"\bpatético\b", r"\bmaldito\b",
    r"\bidiot\b", r"\bstupid\b", r"\bmoron\b", r"\bjerk\b",
    r"\basshole\b", r"\bscum\b",
]

COMPILED_PA  = [re.compile(p, re.IGNORECASE) for p in PASSIVE_AGGRESSIVE_PATTERNS]
COMPILED_PRO = [re.compile(p, re.IGNORECASE) for p in PROVOCATION_PATTERNS]
COMPILED_HOS = [re.compile(p, re.IGNORECASE) for p in HOSTILITY_PATTERNS]


# ---------------------------------------------------------------------------
# Clase principal del analizador
# ---------------------------------------------------------------------------

class ConflictNLPAnalyzer:
    """
    Motor de análisis NLP. Aplica múltiples señales y las combina
    con un esquema de votación ponderada para minimizar falsos positivos.

    Parámetros de calibración:
        toxicity_threshold   : mínimo para considerar toxicidad real (0.6 por defecto)
        ensemble_weights     : peso de cada señal en la puntuación final
        min_confidence       : confianza mínima para escalar la severidad
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.60,
        min_confidence:     float = 0.55,
        device: int = -1,           # -1 = CPU, 0 = GPU
    ):
        self.toxicity_threshold = toxicity_threshold
        self.min_confidence     = min_confidence
        self.device             = device

        # Pesos de ensemble: ajustar según performance en tu corpus
        self.weights = {
            "detoxify":     0.40,   # Modelo más preciso para toxicidad
            "transformer":  0.35,   # Segunda opinión
            "heuristics":   0.25,   # Rápidas pero con más ruido
        }

        self._load_models()

    def _load_models(self):
        """Carga los modelos de forma lazy con fallback gracioso."""
        self.detoxify_model  = None
        self.sentiment_pipe  = None
        self.toxicity_pipe   = None

        if DETOXIFY_AVAILABLE:
            try:
                # 'multilingual' soporta español e inglés
                self.detoxify_model = Detoxify('multilingual')
                logging.info("✓ Detoxify multilingual cargado")
            except Exception as e:
                logging.warning(f"Detoxify no se pudo cargar: {e}")

        if TRANSFORMERS_AVAILABLE:
            try:
                # Sentimiento robusto para redes sociales
                self.sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
                logging.info("✓ Modelo de sentimiento cargado")
            except Exception as e:
                logging.warning(f"Modelo de sentimiento no disponible: {e}")

    # -----------------------------------------------------------------------
    # API pública
    # -----------------------------------------------------------------------

    def analyze_message(self, message: Message) -> Message:
        """
        Punto de entrada principal. Enriquece un Message con todas las
        puntuaciones NLP y devuelve el mismo objeto modificado.

        El flujo es:
          1. Extraer señales individuales de cada modelo/heurística
          2. Combinar con promedio ponderado
          3. Aplicar umbrales conservadores
          4. Asignar ConflictSeverity final con explicación
        """
        text = message.text

        # 1. Obtener señales
        tox_scores  = self._detoxify_scores(text)
        sent_score  = self._sentiment_score(text)
        heur_scores = self._heuristic_scores(text)

        # 2. Combinar toxicidad
        raw_toxicity = (
            self.weights["detoxify"]    * tox_scores["toxicity"] +
            self.weights["heuristics"]  * heur_scores["hostility"]
        )

        # 3. Asignar puntuaciones al mensaje
        message.toxicity_score     = min(raw_toxicity, 1.0)
        message.aggression_score   = tox_scores.get("severe_toxicity", 0.0)
        message.passive_aggression = heur_scores["passive_aggressive"]
        message.provocation_score  = heur_scores["provocation"]
        message.sentiment_polarity = sent_score

        # 4. Determinar severidad con umbral conservador
        message.conflict_severity, message.analysis_explanation = (
            self._classify_severity(message)
        )

        return message

    # -----------------------------------------------------------------------
    # Señales individuales
    # -----------------------------------------------------------------------

    def _detoxify_scores(self, text: str) -> dict:
        """
        Devuelve diccionario con: toxicity, severe_toxicity,
        obscene, threat, insult, identity_attack.
        Fallback a ceros si el modelo no está disponible.
        """
        if self.detoxify_model is None:
            return {k: 0.0 for k in [
                "toxicity", "severe_toxicity", "obscene",
                "threat", "insult", "identity_attack"
            ]}
        try:
            # Detoxify espera texto, devuelve dict[str, float]
            results = self.detoxify_model.predict(text)
            return {k: float(v) for k, v in results.items()}
        except Exception as e:
            logging.debug(f"Error en Detoxify: {e}")
            return {"toxicity": 0.0, "severe_toxicity": 0.0}

    def _sentiment_score(self, text: str) -> float:
        """
        Devuelve polaridad [-1, 1]: -1 muy negativo, +1 muy positivo.
        Usa el modelo de sentimiento si está disponible.
        """
        if self.sentiment_pipe is None:
            return self._simple_sentiment(text)
        try:
            result = self.sentiment_pipe(text[:512])[0]
            label  = result["label"].upper()
            score  = result["score"]
            if "POSITIVE" in label or "POS" in label:
                return score
            elif "NEGATIVE" in label or "NEG" in label:
                return -score
            return 0.0
        except Exception:
            return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> float:
        """Heurística de sentimiento ultraligera sin dependencias."""
        positive_words = ["gracias", "bien", "genial", "perfecto", "acuerdo",
                          "thanks", "great", "good", "agree", "nice"]
        negative_words = ["mal", "horrible", "pésimo", "error", "culpa",
                          "bad", "terrible", "wrong", "fault", "hate"]
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def _heuristic_scores(self, text: str) -> dict:
        """
        Aplica los patrones regex compilados y devuelve puntuaciones [0,1].
        IMPORTANTE: las heurísticas tienen recall alto pero precisión media,
        por eso tienen peso reducido en el ensemble.
        """
        def match_ratio(patterns, text):
            hits = sum(1 for p in patterns if p.search(text))
            return min(hits / max(len(patterns) * 0.1, 1), 1.0)

        return {
            "passive_aggressive": match_ratio(COMPILED_PA,  text),
            "provocation":        match_ratio(COMPILED_PRO, text),
            "hostility":          match_ratio(COMPILED_HOS, text),
        }

    # -----------------------------------------------------------------------
    # Clasificación final de severidad
    # -----------------------------------------------------------------------

    def _classify_severity(self, msg: Message) -> Tuple[ConflictSeverity, str]:
        """
        Lógica de clasificación multi-criterio.
        Se aplica el PRINCIPIO DE MÍNIMA ASUNCIÓN:
          - No escalar salvo que múltiples señales coincidan
          - Preferir sub-clasificar (LOW) a sobre-clasificar (HIGH)
        """
        t  = msg.toxicity_score
        a  = msg.aggression_score
        pa = msg.passive_aggression
        pr = msg.provocation_score
        s  = msg.sentiment_polarity

        explanations = []

        # CRITICAL: toxicidad severa verificada por múltiples señales
        if t > 0.85 and a > 0.70:
            explanations.append(f"Toxicidad severa: {t:.2f}, agresión: {a:.2f}")
            return ConflictSeverity.CRITICAL, " | ".join(explanations)

        # HIGH: hostilidad clara
        if t > self.toxicity_threshold and (a > 0.5 or pr > 0.5):
            explanations.append(
                f"Toxicidad {t:.2f} + agresión {a:.2f} o provocación {pr:.2f}"
            )
            return ConflictSeverity.HIGH, " | ".join(explanations)

        # MODERATE: varias señales medias simultáneas
        signals_moderate = sum([
            t > 0.40,
            pa > 0.30,
            pr > 0.30,
            s < -0.50,
        ])
        if signals_moderate >= 2:
            explanations.append(
                f"Múltiples señales moderadas: tox={t:.2f}, PA={pa:.2f}, "
                f"prov={pr:.2f}, sent={s:.2f}"
            )
            return ConflictSeverity.MODERATE, " | ".join(explanations)

        # LOW: una sola señal débil
        if t > 0.25 or pa > 0.20 or pr > 0.20 or s < -0.40:
            explanations.append("Señal leve única — monitorizar")
            return ConflictSeverity.LOW, " | ".join(explanations)

        return ConflictSeverity.NONE, "Sin señales de conflicto"
