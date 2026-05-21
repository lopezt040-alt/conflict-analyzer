"""
=============================================================================
MÓDULO 5: PIPELINE PRINCIPAL Y EJEMPLOS DE USO
=============================================================================
Orquesta todos los módulos en un pipeline coherente.
Incluye ejemplos de uso con datos simulados.
"""

import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import random

from data_structures import (
    Message, UserProfile, ConflictSeverity, CommunityReport
)
from nlp_analyzer import ConflictNLPAnalyzer
from graph_analyzer import ConflictGraphAnalyzer
from risk_scorer import RiskScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Generador de datos de ejemplo
# ---------------------------------------------------------------------------

def generate_sample_messages() -> List[Message]:
    """
    Genera un dataset simulado con diferentes patrones de conflicto.
    Útil para desarrollo, testing y evaluación del sistema.
    """
    base_time = datetime.now() - timedelta(days=7)

    # Usuarios con diferentes perfiles de comportamiento
    users = {
        "alice":   "Alice",        # Usuaria normal
        "bob":     "Bob",          # Usuario ocasionalmente conflictivo
        "charlie": "Charlie",      # Agitador frecuente
        "diana":   "Diana",        # Mediadora
        "eve":     "Eve",          # Víctima frecuente de ataques
    }

    # Mensajes simulados con diferentes tonos
    sample_messages = [
        # Conversación normal
        ("alice",   None,       "thread_1",
         "¿Alguien ha probado la nueva API? Funciona bastante bien."),
        ("bob",     "msg_001",  "thread_1",
         "Sí, la usé ayer. Tiene algunas limitaciones pero en general está bien."),
        ("charlie", "msg_002",  "thread_1",
         "Está bien si no te importa que sea una basura comparada con lo que teníamos."),

        # Escalada de tensión
        ("alice",   "msg_003",  "thread_1",
         "Charlie, eso es una exageración. ¿Puedes explicar qué problemas encontraste?"),
        ("charlie", "msg_004",  "thread_1",
         "No tengo que explicarte nada. Si no lo ves es que no entiendes de esto."),
        ("bob",     "msg_005",  "thread_1",
         "Oye Charlie, eso no es necesario. Podemos hablar de esto tranquilamente."),
        ("charlie", "msg_006",  "thread_1",
         "Claro, claro, ahora el defensor de las causas perdidas. Típico de ti Bob."),

        # Pasivo-agresivo
        ("charlie", "msg_007",  "thread_2",
         "Muy interesante propuesta. Suerte llevándola a cabo. Si tú lo dices..."),
        ("eve",     "msg_007",  "thread_2",
         "¿Tienes alguna crítica constructiva o solo sarcasmo?"),
        ("charlie", "msg_008",  "thread_2",
         "No esperaba menos de ti. Sigue en tu mundo."),

        # Ataque directo (HIGH severity)
        ("charlie", "msg_009",  "thread_3",
         "Eres un completo idiota si crees que eso va a funcionar."),
        ("eve",     "msg_010",  "thread_3",
         "No me llames idiota. Ataca las ideas, no a las personas."),
        ("diana",   "msg_010",  "thread_3",
         "Vamos a calmarnos. Charlie, ese lenguaje no está bien. Eve tiene razón."),

        # Debate normal (NO conflict — importante para calibrar)
        ("alice",   None,       "thread_4",
         "Creo que el enfoque A es mejor que el B porque tiene mejor rendimiento."),
        ("bob",     "msg_013",  "thread_4",
         "Discrepo, el enfoque B tiene mejor mantenibilidad aunque sea más lento."),
        ("alice",   "msg_014",  "thread_4",
         "Buen punto. ¿Tienes benchmarks que lo respalden?"),
        ("bob",     "msg_015",  "thread_4",
         "Aquí los tienes. En producción, el 73% de los casos prefieren B."),
    ]

    messages = {}
    msg_list  = []

    for i, (user_id, reply_to, thread_id, text) in enumerate(sample_messages):
        msg_id = f"msg_{i+1:03d}"
        msg = Message(
            id=msg_id,
            user_id=user_id,
            text=text,
            timestamp=base_time + timedelta(hours=i * 2),
            thread_id=thread_id,
            reply_to_id=reply_to,
            platform="forum",
        )
        messages[msg_id] = msg
        msg_list.append(msg)

    return msg_list, messages


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class ConflictAnalysisPipeline:
    """
    Orquesta el análisis completo de una comunidad.

    Flujo:
      1. Ingerir mensajes (batch o stream)
      2. Análisis NLP de cada mensaje
      3. Actualizar grafo de interacciones
      4. Calcular perfiles de usuario
      5. Generar puntuaciones de riesgo
      6. Producir informe de comunidad
    """

    def __init__(
        self,
        toxicity_threshold: float = 0.60,
        escalation_window_minutes: int = 30,
    ):
        self.nlp_analyzer   = ConflictNLPAnalyzer(
            toxicity_threshold=toxicity_threshold
        )
        self.graph_analyzer = ConflictGraphAnalyzer(
            escalation_window_minutes=escalation_window_minutes
        )
        self.risk_scorer    = RiskScorer()

        self.messages:      Dict[str, Message]      = {}
        self.user_profiles: Dict[str, UserProfile]  = {}

    def ingest_messages(self, messages: List[Message]) -> List[Message]:
        """
        Procesa una lista de mensajes en orden cronológico.
        Devuelve los mensajes enriquecidos con puntuaciones NLP.
        """
        # Ordenar cronológicamente para que las escaladas se detecten bien
        sorted_msgs = sorted(messages, key=lambda m: m.timestamp)

        analyzed = []
        for msg in sorted_msgs:
            # 1. Análisis NLP
            enriched = self.nlp_analyzer.analyze_message(msg)
            self.messages[msg.id] = enriched

            # 2. Actualizar grafo
            self.graph_analyzer.process_message(enriched, self.messages)

            # 3. Actualizar perfil del usuario
            self._update_user_profile(enriched)

            analyzed.append(enriched)
            logging.debug(
                f"[{msg.id}] @{msg.user_id}: {msg.conflict_severity.name} "
                f"(tox={msg.toxicity_score:.2f})"
            )

        return analyzed

    def _update_user_profile(self, msg: Message):
        """Actualiza o crea el perfil del usuario con los datos del mensaje."""
        uid = msg.user_id
        if uid not in self.user_profiles:
            self.user_profiles[uid] = UserProfile(
                user_id=uid,
                username=uid,
            )

        profile = self.user_profiles[uid]
        profile.total_messages += 1

        # Actualizar promedios con media acumulada
        n = profile.total_messages
        profile.avg_toxicity = (
            (profile.avg_toxicity * (n-1) + msg.toxicity_score) / n
        )
        profile.avg_aggression = (
            (profile.avg_aggression * (n-1) + msg.aggression_score) / n
        )
        profile.avg_passive_aggression = (
            (profile.avg_passive_aggression * (n-1) + msg.passive_aggression) / n
        )

        if msg.conflict_severity.value >= ConflictSeverity.MODERATE.value:
            profile.conflict_initiated += 1

    def compute_risk_scores(self) -> Dict[str, dict]:
        """
        Calcula puntuaciones de riesgo para todos los usuarios.
        Devuelve {user_id: {score, label, factors, report}}
        """
        all_messages  = list(self.messages.values())
        conflict_evts = self.graph_analyzer.conflict_events
        results       = {}

        for uid, profile in self.user_profiles.items():
            score, label, factors = self.risk_scorer.compute_risk(
                user=profile,
                messages=all_messages,
                conflict_events=conflict_evts,
            )
            profile.risk_score = score
            profile.risk_label = label
            profile.risk_factors = [f.explanation for f in factors]

            report_text = self.risk_scorer.generate_report(
                profile, score, label, factors
            )

            results[uid] = {
                "score":   score,
                "label":   label,
                "factors": [
                    {
                        "name":        f.name,
                        "value":       round(f.value, 3),
                        "contribution": round(f.contribution * 100, 2),
                        "explanation": f.explanation,
                        "evidence":    f.evidence,
                    }
                    for f in factors
                ],
                "report":  report_text,
            }

        return results

    def generate_community_report(self) -> CommunityReport:
        """Genera el informe de salud de la comunidad."""
        all_msgs      = list(self.messages.values())
        conflict_msgs = [
            m for m in all_msgs
            if m.conflict_severity.value >= ConflictSeverity.MODERATE.value
        ]

        conflict_rate = len(conflict_msgs) / max(len(all_msgs), 1)

        # Top usuarios por puntuación de riesgo
        top_risk = sorted(
            self.user_profiles.values(),
            key=lambda u: u.risk_score,
            reverse=True,
        )[:5]

        # Puntuación de salud: inverso de la tasa de conflicto
        health_score = max(0.0, (1 - conflict_rate * 3)) * 100

        ts = [m.timestamp for m in all_msgs]
        report = CommunityReport(
            report_id=str(uuid.uuid4())[:8],
            period_start=min(ts) if ts else datetime.now(),
            period_end=max(ts) if ts else datetime.now(),
            total_messages=len(all_msgs),
            conflict_rate=conflict_rate,
            top_risk_users=[u.user_id for u in top_risk],
            conflict_events=self.graph_analyzer.conflict_events,
            health_score=health_score,
            recommendations=self._generate_recommendations(
                conflict_rate, top_risk
            ),
        )
        return report

    def _generate_recommendations(self, rate, top_risk) -> List[str]:
        recs = []
        if rate > 0.20:
            recs.append(
                "La tasa de conflicto supera el 20% — considerar revisar "
                "las normas de la comunidad."
            )
        critical_users = [u for u in top_risk if u.risk_label in ("high", "critical")]
        if critical_users:
            names = ", ".join(f"@{u.username}" for u in critical_users)
            recs.append(
                f"Usuarios con riesgo elevado requieren revisión manual: {names}"
            )
        if not recs:
            recs.append("La comunidad muestra un comportamiento saludable.")
        return recs


# ---------------------------------------------------------------------------
# Ejecución de ejemplo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sistema de Análisis de Conflictos — Demo ===\n")

    # 1. Generar datos de ejemplo
    msg_list, msg_dict = generate_sample_messages()
    print(f"Mensajes cargados: {len(msg_list)}")

    # 2. Inicializar pipeline
    pipeline = ConflictAnalysisPipeline(
        toxicity_threshold=0.55,
        escalation_window_minutes=120,
    )

    # 3. Procesar mensajes
    print("\n--- Analizando mensajes ---")
    analyzed = pipeline.ingest_messages(msg_list)

    for msg in analyzed:
        if msg.conflict_severity.value > 0:
            print(
                f"  [{msg.conflict_severity.name:8}] @{msg.user_id:8} "
                f"tox={msg.toxicity_score:.2f} | {msg.text[:60]}..."
            )

    # 4. Calcular puntuaciones de riesgo
    print("\n--- Puntuaciones de Riesgo ---")
    risk_results = pipeline.compute_risk_scores()

    for uid, data in sorted(
        risk_results.items(), key=lambda x: x[1]["score"], reverse=True
    ):
        print(f"  @{uid:10} → {data['score']:5.1f}/100  [{data['label'].upper():8}]")
        for factor in data["factors"][:2]:
            if factor["contribution"] > 0:
                print(f"             {factor['contribution']:4.1f}pts — {factor['name']}")

    # 5. Informe de comunidad
    print("\n--- Informe de Comunidad ---")
    report = pipeline.generate_community_report()
    print(f"  Mensajes totales:   {report.total_messages}")
    print(f"  Tasa de conflicto: {report.conflict_rate*100:.1f}%")
    print(f"  Salud comunidad:   {report.health_score:.0f}/100")
    print(f"  Eventos detectados: {len(report.conflict_events)}")
    print("\n  Recomendaciones:")
    for rec in report.recommendations:
        print(f"    • {rec}")

    # 6. Métricas del grafo
    print("\n--- Métricas del Grafo de Interacciones ---")
    graph_metrics = pipeline.graph_analyzer.compute_graph_metrics()
    for uid, metrics in graph_metrics.items():
        print(
            f"  @{uid:10} out={metrics['conflict_out']:.1f}  "
            f"in={metrics['conflict_in']:.1f}  "
            f"between={metrics['betweenness']:.3f}"
        )

    print("\n✓ Pipeline completo. Revisa los informes detallados con pipeline.compute_risk_scores()")
