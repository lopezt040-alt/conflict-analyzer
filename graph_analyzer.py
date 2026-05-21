"""
=============================================================================
MÓDULO 3: ANÁLISIS DE GRAFOS Y REDES SOCIALES
=============================================================================
Modela las interacciones como un grafo dirigido ponderado donde:
  - Nodos = usuarios
  - Aristas = interacciones (reply / mención)
  - Peso  = intensidad de conflicto acumulada

Métricas clave extraídas:
  - Betweenness centrality → mediadores y agitadores
  - In-degree de hostilidad → usuarios frecuentemente atacados
  - Clustering de conflicto → grupos antagónicos
  - Detección de escaladas temporales (sliding window)
"""

import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False
    logging.warning("networkx no disponible — análisis de grafos limitado")

from data_structures import Message, ConflictEvent, ConflictSeverity, InteractionType


class ConflictGraphAnalyzer:
    """
    Construye y analiza el grafo de interacciones conflictivas.

    El grafo es DIRIGIDO: A → B significa "A respondió/mencionó a B"
    El peso de cada arista acumula la severidad promedio de esas interacciones.
    """

    def __init__(
        self,
        escalation_window_minutes: int = 30,   # Ventana para detectar escalada
        min_conflict_weight: float = 0.30,     # Umbral mínimo para añadir arista
    ):
        self.escalation_window  = timedelta(minutes=escalation_window_minutes)
        self.min_conflict_weight = min_conflict_weight

        # Grafo principal: nodos=usuarios, aristas ponderadas por conflicto
        self.graph = nx.DiGraph() if NX_AVAILABLE else None

        # Historial por hilo para detectar escaladas temporales
        # {thread_id: deque[(timestamp, user_id, severity_value)]}
        self.thread_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Registro de eventos de conflicto detectados
        self.conflict_events: List[ConflictEvent] = []
        self._event_counter = 0

    # -----------------------------------------------------------------------
    # Procesamiento incremental de mensajes
    # -----------------------------------------------------------------------

    def process_message(self, message: Message, all_messages: Dict[str, Message]):
        """
        Procesa un mensaje y actualiza el grafo.
        Llamar en orden cronológico para detectar escaladas correctamente.

        Args:
            message:      El mensaje a procesar (ya analizado por NLP).
            all_messages: Diccionario {msg_id: Message} para lookup de contexto.
        """
        severity_val = message.conflict_severity.value

        # Actualizar historial del hilo
        self.thread_history[message.thread_id].append((
            message.timestamp,
            message.user_id,
            severity_val,
        ))

        # Si el mensaje es una respuesta, crear arista en el grafo
        if message.reply_to_id and message.reply_to_id in all_messages:
            target_msg = all_messages[message.reply_to_id]
            target_user = target_msg.user_id

            # Solo añadir arista si hay algún nivel de conflicto
            if severity_val >= 1 and message.user_id != target_user:
                self._update_edge(
                    source=message.user_id,
                    target=target_user,
                    severity=severity_val,
                    msg_id=message.id,
                    timestamp=message.timestamp,
                )

        # Detectar escaladas en este hilo
        self._detect_escalation(message.thread_id, message)

    def _update_edge(
        self,
        source: str,
        target: str,
        severity: int,
        msg_id: str,
        timestamp: datetime,
    ):
        """
        Actualiza o crea una arista en el grafo de conflictos.
        Usa promedio exponencial ponderado para suavizar el historial.
        """
        if not NX_AVAILABLE:
            return

        if self.graph.has_edge(source, target):
            edge = self.graph[source][target]
            # Promedio exponencial: da más peso a interacciones recientes
            alpha = 0.3
            edge["weight"]      = alpha * severity + (1 - alpha) * edge["weight"]
            edge["count"]      += 1
            edge["last_time"]   = timestamp
            edge["message_ids"].append(msg_id)
        else:
            self.graph.add_edge(
                source, target,
                weight=float(severity),
                count=1,
                first_time=timestamp,
                last_time=timestamp,
                message_ids=[msg_id],
            )

        # Asegurar que los nodos existen con atributos básicos
        for node in [source, target]:
            if node not in self.graph.nodes:
                self.graph.add_node(node)

    # -----------------------------------------------------------------------
    # Detección de escaladas temporales
    # -----------------------------------------------------------------------

    def _detect_escalation(self, thread_id: str, trigger_msg: Message):
        """
        Detecta si la ventana temporal reciente muestra una escalada.
        Una escalada se define como:
          - ≥3 mensajes en la ventana
          - La severidad promedio sube significativamente respecto al inicio
          - ≥2 usuarios distintos involucrados

        Si se detecta, registra un ConflictEvent.
        """
        history = self.thread_history[thread_id]
        now     = trigger_msg.timestamp
        cutoff  = now - self.escalation_window

        # Filtrar solo mensajes dentro de la ventana
        window = [(ts, uid, sev) for ts, uid, sev in history if ts >= cutoff]

        if len(window) < 3:
            return  # Ventana demasiado pequeña

        users_in_window = {uid for _, uid, _ in window}
        if len(users_in_window) < 2:
            return  # Solo un usuario, no hay conflicto interpersonal

        severities = [sev for _, _, sev in window]
        avg_severity = np.mean(severities)
        max_severity = max(severities)

        # Gradiente: ¿está subiendo la tensión?
        if len(severities) >= 4:
            first_half = np.mean(severities[:len(severities)//2])
            second_half = np.mean(severities[len(severities)//2:])
            escalation_gradient = second_half - first_half
        else:
            escalation_gradient = 0.0

        # Condición de escalada: tensión media moderada + tendencia al alza
        is_escalating = (
            avg_severity >= 1.5 and
            max_severity >= 2 and
            escalation_gradient >= 0.3
        )

        if is_escalating:
            # Verificar que no sea duplicado del mismo hilo reciente
            recent_events = [
                e for e in self.conflict_events[-10:]
                if e.thread_id == thread_id
                and (now - e.start_time) < self.escalation_window * 2
            ]
            if not recent_events:
                self._register_conflict_event(
                    thread_id=thread_id,
                    participants=list(users_in_window),
                    messages=window,
                    start_time=window[0][0],
                    peak_severity=max_severity,
                )

    def _register_conflict_event(
        self,
        thread_id: str,
        participants: list,
        messages: list,
        start_time: datetime,
        peak_severity: int,
    ):
        """Registra un evento de conflicto detectado."""
        self._event_counter += 1
        severity = ConflictSeverity(min(peak_severity, 4))

        event = ConflictEvent(
            event_id=f"evt_{self._event_counter:04d}",
            thread_id=thread_id,
            participants=participants,
            start_time=start_time,
            peak_severity=severity,
            interaction_type=self._infer_interaction_type(peak_severity),
            summary=(
                f"Escalada detectada en hilo {thread_id} "
                f"entre {len(participants)} usuarios. "
                f"Severidad pico: {severity.name}"
            )
        )
        self.conflict_events.append(event)
        logging.info(f"ConflictEvent registrado: {event.event_id} — {event.summary}")

    def _infer_interaction_type(self, severity: int) -> InteractionType:
        mapping = {
            0: InteractionType.NEUTRAL,
            1: InteractionType.DISAGREEMENT,
            2: InteractionType.TENSION,
            3: InteractionType.PROVOCATION,
            4: InteractionType.HOSTILITY,
        }
        return mapping.get(severity, InteractionType.TENSION)

    # -----------------------------------------------------------------------
    # Métricas del grafo
    # -----------------------------------------------------------------------

    def compute_graph_metrics(self) -> Dict[str, dict]:
        """
        Calcula métricas de centralidad para identificar:
          - Agitadores: alto out-degree ponderado
          - Víctimas recurrentes: alto in-degree ponderado
          - Mediadores: alta betweenness centrality (opcional)

        Devuelve {user_id: {métrica: valor}}
        """
        if not NX_AVAILABLE or self.graph.number_of_nodes() == 0:
            return {}

        metrics = {}

        # Out-degree ponderado: cuánto conflicto genera cada usuario
        out_deg = dict(self.graph.out_degree(weight="weight"))

        # In-degree ponderado: cuánto conflicto recibe
        in_deg  = dict(self.graph.in_degree(weight="weight"))

        # Betweenness: usuarios que están en caminos entre grupos conflictivos
        try:
            betweenness = nx.betweenness_centrality(self.graph, weight="weight")
        except Exception:
            betweenness = {n: 0.0 for n in self.graph.nodes}

        for node in self.graph.nodes:
            metrics[node] = {
                "conflict_out": out_deg.get(node, 0.0),
                "conflict_in":  in_deg.get(node, 0.0),
                "betweenness":  betweenness.get(node, 0.0),
                "net_conflict": out_deg.get(node, 0) - in_deg.get(node, 0),
            }

        return metrics

    def detect_antagonistic_clusters(self) -> List[Tuple[set, set]]:
        """
        Identifica pares de grupos de usuarios con alta hostilidad mutua.
        Usa detección de comunidades en el grafo no dirigido de conflictos.

        Devuelve lista de (grupo_A, grupo_B) con alta tensión entre ellos.
        """
        if not NX_AVAILABLE or self.graph.number_of_nodes() < 4:
            return []

        # Convertir a no dirigido para detectar comunidades
        undirected = self.graph.to_undirected()

        try:
            # Louvain community detection (requiere networkx >= 3.0)
            from networkx.algorithms.community import louvain_communities
            communities = list(louvain_communities(undirected, weight="weight"))
        except (ImportError, AttributeError):
            # Fallback: componentes conectados
            communities = [
                c for c in nx.connected_components(undirected)
                if len(c) >= 2
            ]

        # Detectar pares de comunidades con muchas aristas entre ellas
        antagonistic_pairs = []
        for i, comm_a in enumerate(communities):
            for comm_b in communities[i+1:]:
                cross_weight = sum(
                    self.graph[u][v]["weight"]
                    for u in comm_a for v in comm_b
                    if self.graph.has_edge(u, v)
                )
                if cross_weight > 2.0:
                    antagonistic_pairs.append((comm_a, comm_b))

        return antagonistic_pairs
