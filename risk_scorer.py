"""
=============================================================================
MÓDULO 4: SISTEMA DE PUNTUACIÓN DE RIESGO INTERPRETABLE
=============================================================================
Genera una puntuación de riesgo [0-100] para cada usuario, explicable
factor por factor. Inspirado en modelos de scoring crediticio pero
adaptado para análisis de conflictos comunitarios.

Principios de diseño:
  - Toda puntuación tiene una explicación en lenguaje natural
  - Las puntuaciones decaen con el tiempo (comportamiento pasado < reciente)
  - Se distingue el contexto: discutir ideas ≠ atacar personas
  - Umbrales revisables por moderadores humanos
"""

from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

from data_structures import Message, UserProfile, ConflictSeverity


@dataclass
class RiskFactor:
    """Un factor individual que contribuye a la puntuación de riesgo."""
    name:        str          # Nombre legible del factor
    value:       float        # Valor medido [0, 1]
    weight:      float        # Peso en la puntuación final
    contribution: float       # Contribución numérica real
    explanation: str          # Explicación en lenguaje natural
    evidence:    List[str]    # Ejemplos concretos (msg IDs o fragmentos)


class RiskScorer:
    """
    Calcula y explica la puntuación de riesgo de un usuario.

    La puntuación se descompone en factores independientes para que
    los moderadores puedan entender y cuestionar cada contribución.

    Fórmula:
        risk_score = Σ (factor_value × factor_weight × decay) × 100

    El decay temporal reduce la influencia de eventos antiguos:
        decay = exp(-λ × días_desde_evento)
    """

    FACTOR_WEIGHTS = {
        "toxicity_frequency":    0.25,  # ¿Con qué frecuencia es tóxico?
        "severe_incidents":      0.20,  # ¿Cuántos incidentes graves?
        "escalation_initiation": 0.18,  # ¿Inicia escaladas?
        "targeted_harassment":   0.15,  # ¿Ataca repetidamente al mismo usuario?
        "passive_aggression":    0.10,  # ¿Usa hostilidad encubierta?
        "recovery_deficit":      0.07,  # ¿No mejora tras advertencias?
        "provocation_pattern":   0.05,  # ¿Provoca deliberadamente?
    }

    DECAY_LAMBDA = 0.05   # Constante de decaimiento: ~14 días para reducir al 50%

    RISK_LABELS = [
        (0,  20,  "low",      "Comportamiento dentro de la norma"),
        (20, 45,  "medium",   "Algunos patrones a monitorizar"),
        (45, 70,  "high",     "Riesgo elevado — requiere revisión"),
        (70, 100, "critical", "Acción de moderación recomendada"),
    ]

    def __init__(self, reference_date: datetime = None):
        """
        Args:
            reference_date: Fecha de referencia para calcular el decay.
                            Por defecto: ahora.
        """
        self.reference_date = reference_date or datetime.now()

    def compute_risk(
        self,
        user: UserProfile,
        messages: List[Message],
        conflict_events: list,
    ) -> Tuple[float, str, List[RiskFactor]]:
        """
        Calcula la puntuación de riesgo completa de un usuario.

        Returns:
            (score, label, factors) donde:
              score   = puntuación [0, 100]
              label   = 'low' | 'medium' | 'high' | 'critical'
              factors = lista de RiskFactor explicando cada contribución
        """
        user_messages = [m for m in messages if m.user_id == user.user_id]

        if not user_messages:
            return 0.0, "low", []

        factors = []

        # --- Factor 1: Frecuencia de toxicidad ---
        factors.append(self._factor_toxicity_frequency(user_messages))

        # --- Factor 2: Incidentes severos ---
        factors.append(self._factor_severe_incidents(user_messages))

        # --- Factor 3: Iniciación de escaladas ---
        factors.append(self._factor_escalation_initiation(
            user_messages, conflict_events
        ))

        # --- Factor 4: Acoso dirigido ---
        factors.append(self._factor_targeted_harassment(user_messages))

        # --- Factor 5: Pasivo-agresivo ---
        factors.append(self._factor_passive_aggression(user_messages))

        # --- Factor 6: Déficit de recuperación ---
        factors.append(self._factor_recovery_deficit(user, user_messages))

        # --- Factor 7: Patrón de provocación ---
        factors.append(self._factor_provocation(user_messages))

        # Suma ponderada
        total_score = sum(f.contribution for f in factors)
        total_score = max(0.0, min(100.0, total_score * 100))

        # Determinar label
        label = self._score_to_label(total_score)

        return total_score, label, factors

    # -----------------------------------------------------------------------
    # Factores individuales
    # -----------------------------------------------------------------------

    def _factor_toxicity_frequency(self, messages: List[Message]) -> RiskFactor:
        """% de mensajes con toxicidad ≥ MODERATE, con decaimiento temporal."""
        toxic_msgs = [
            m for m in messages
            if m.conflict_severity.value >= ConflictSeverity.MODERATE.value
        ]

        if not messages:
            value = 0.0
        else:
            # Aplicar decay: mensajes recientes pesan más
            decayed_count = sum(
                self._time_decay(m.timestamp) for m in toxic_msgs
            )
            decayed_total = sum(
                self._time_decay(m.timestamp) for m in messages
            )
            value = decayed_count / max(decayed_total, 1)

        weight = self.FACTOR_WEIGHTS["toxicity_frequency"]
        evidence = [m.id for m in toxic_msgs[-3:]]  # Últimos 3 ejemplos

        return RiskFactor(
            name="Frecuencia de toxicidad",
            value=value,
            weight=weight,
            contribution=value * weight,
            explanation=(
                f"{len(toxic_msgs)} de {len(messages)} mensajes con nivel "
                f"MODERATE+ ({value*100:.1f}%). "
                f"{'Preocupante' if value > 0.15 else 'Aceptable'}."
            ),
            evidence=evidence,
        )

    def _factor_severe_incidents(self, messages: List[Message]) -> RiskFactor:
        """Cuenta y pondera incidentes HIGH o CRITICAL."""
        severe = [
            m for m in messages
            if m.conflict_severity.value >= ConflictSeverity.HIGH.value
        ]
        # Score: función logarítmica para que no explote con muchos incidentes
        raw = len(severe)
        value = min(math.log1p(raw) / math.log1p(10), 1.0)  # Normaliza a [0,1]

        weight = self.FACTOR_WEIGHTS["severe_incidents"]

        return RiskFactor(
            name="Incidentes severos",
            value=value,
            weight=weight,
            contribution=value * weight,
            explanation=(
                f"{raw} incidentes de alta severidad (HIGH/CRITICAL). "
                f"Umbral de alerta: 3+"
            ),
            evidence=[m.id for m in severe[-5:]],
        )

    def _factor_escalation_initiation(
        self, messages: List[Message], conflict_events: list
    ) -> RiskFactor:
        """¿Con qué frecuencia este usuario inicia eventos de escalada?"""
        user_id = messages[0].user_id if messages else ""

        # Un usuario "inicia" una escalada si su mensaje tiene
        # la mayor severidad en la primera mitad del evento
        initiations = 0
        relevant_events = [
            e for e in conflict_events if user_id in e.participants
        ]

        for event in relevant_events:
            if event.participants and event.participants[0] == user_id:
                initiations += 1

        value = min(initiations / max(len(relevant_events), 1), 1.0)
        weight = self.FACTOR_WEIGHTS["escalation_initiation"]

        return RiskFactor(
            name="Iniciación de escaladas",
            value=value,
            weight=weight,
            contribution=value * weight,
            explanation=(
                f"Inició o fue primer participante en {initiations} "
                f"de {len(relevant_events)} eventos de escalada detectados."
            ),
            evidence=[e.event_id for e in relevant_events[:3]],
        )

    def _factor_targeted_harassment(self, messages: List[Message]) -> RiskFactor:
        """
        Detecta si hay un patrón de ataques repetidos hacia el mismo usuario.
        Indica acoso dirigido vs conflicto general.
        """
        # Contar hacia qué usuarios se dirigen los mensajes conflictivos
        target_counts: Dict[str, int] = {}
        conflict_msgs = [
            m for m in messages
            if m.conflict_severity.value >= ConflictSeverity.MODERATE.value
            and m.reply_to_id
        ]

        for m in conflict_msgs:
            # En un pipeline completo, resolveríamos el reply_to al user_id
            # Aquí usamos reply_to_id como proxy
            target_counts[m.reply_to_id] = target_counts.get(m.reply_to_id, 0) + 1

        max_concentration = max(target_counts.values()) if target_counts else 0
        # Si >40% de los conflictos van dirigidos a la misma persona → alerta
        total_conflicts = len(conflict_msgs)
        concentration = max_concentration / max(total_conflicts, 1)
        value = concentration if max_concentration >= 3 else 0.0

        weight = self.FACTOR_WEIGHTS["targeted_harassment"]

        return RiskFactor(
            name="Acoso dirigido",
            value=value,
            weight=weight,
            contribution=value * weight,
            explanation=(
                f"{'Patrón de acoso detectado' if value > 0.4 else 'Sin patrón dirigido'}. "
                f"Máxima concentración en un objetivo: {max_concentration} mensajes conflictivos."
            ),
            evidence=[],
        )

    def _factor_passive_aggression(self, messages: List[Message]) -> RiskFactor:
        """Promedio de pasivo-agresividad en los mensajes con score > 0."""
        pa_msgs = [m for m in messages if m.passive_aggression > 0.15]
        value = sum(m.passive_aggression for m in pa_msgs) / max(len(messages), 1)
        weight = self.FACTOR_WEIGHTS["passive_aggression"]

        return RiskFactor(
            name="Pasivo-agresividad",
            value=min(value * 3, 1.0),  # Amplificar ligeramente (señal débil)
            weight=weight,
            contribution=min(value * 3, 1.0) * weight,
            explanation=(
                f"{len(pa_msgs)} mensajes con rasgos pasivo-agresivos detectados "
                f"(sarcasmo, condescendencia, invalidación velada)."
            ),
            evidence=[m.id for m in pa_msgs[-3:]],
        )

    def _factor_recovery_deficit(
        self, user: UserProfile, messages: List[Message]
    ) -> RiskFactor:
        """
        ¿Sigue el comportamiento conflictivo tras advertencias?
        Un déficit de recuperación es más grave que un incidente aislado.
        """
        if user.warnings_received == 0:
            return RiskFactor(
                name="Déficit de recuperación",
                value=0.0, weight=self.FACTOR_WEIGHTS["recovery_deficit"],
                contribution=0.0,
                explanation="Sin advertencias previas registradas.",
                evidence=[],
            )

        # Mensajes conflictivos después del momento de la última advertencia
        # (En implementación real, habría timestamp de advertencias)
        recent_conflicts = sum(
            1 for m in messages[-20:]
            if m.conflict_severity.value >= ConflictSeverity.MODERATE.value
        )
        value = min(recent_conflicts / 5, 1.0) if user.warnings_received > 0 else 0.0
        weight = self.FACTOR_WEIGHTS["recovery_deficit"]

        return RiskFactor(
            name="Déficit de recuperación",
            value=value,
            weight=weight,
            contribution=value * weight,
            explanation=(
                f"{user.warnings_received} advertencia(s) previas. "
                f"{recent_conflicts} conflictos recientes tras ellas. "
                f"{'Preocupante: no hay mejora aparente.' if value > 0.5 else 'Mejoría parcial observada.'}"
            ),
            evidence=[],
        )

    def _factor_provocation(self, messages: List[Message]) -> RiskFactor:
        """Frecuencia de mensajes con alta puntuación de provocación."""
        prov_msgs = [m for m in messages if m.provocation_score > 0.25]
        value = len(prov_msgs) / max(len(messages), 1)
        weight = self.FACTOR_WEIGHTS["provocation_pattern"]

        return RiskFactor(
            name="Patrón de provocación",
            value=min(value * 5, 1.0),
            weight=weight,
            contribution=min(value * 5, 1.0) * weight,
            explanation=(
                f"{len(prov_msgs)} mensajes con indicadores de provocación intencional."
            ),
            evidence=[m.id for m in prov_msgs[-3:]],
        )

    # -----------------------------------------------------------------------
    # Utilidades
    # -----------------------------------------------------------------------

    def _time_decay(self, timestamp: datetime) -> float:
        """Función de decaimiento exponencial: eventos recientes > pasados."""
        days_ago = (self.reference_date - timestamp).days
        return math.exp(-self.DECAY_LAMBDA * days_ago)

    def _score_to_label(self, score: float) -> str:
        for lo, hi, label, _ in self.RISK_LABELS:
            if lo <= score < hi:
                return label
        return "critical"

    def generate_report(
        self,
        user: UserProfile,
        score: float,
        label: str,
        factors: List[RiskFactor],
    ) -> str:
        """
        Genera un informe de texto legible para moderadores.
        IMPORTANTE: El informe siempre incluye contexto y limitaciones.
        """
        lines = [
            f"═══════════════════════════════════════",
            f"INFORME DE RIESGO: @{user.username}",
            f"═══════════════════════════════════════",
            f"Puntuación: {score:.1f}/100  |  Nivel: {label.upper()}",
            f"Mensajes analizados: {user.total_messages}",
            f"",
            f"FACTORES CONTRIBUYENTES:",
        ]

        # Ordenar factores por contribución descendente
        for f in sorted(factors, key=lambda x: x.contribution, reverse=True):
            bar = "█" * int(f.contribution * 200) + "░" * (10 - int(f.contribution * 200))
            lines.append(
                f"  [{bar[:10]}] {f.contribution*100:.1f}pts — {f.name}"
            )
            lines.append(f"    → {f.explanation}")

        lines += [
            "",
            "LIMITACIONES Y CONTEXTO:",
            "  • Este análisis es orientativo, no definitivo.",
            "  • Un moderador humano debe revisar los mensajes antes de actuar.",
            "  • El modelo puede no capturar ironía, humor o contexto cultural.",
            "  • Puntuaciones altas indican revisión necesaria, no culpa.",
            "═══════════════════════════════════════",
        ]
        return "\n".join(lines)
