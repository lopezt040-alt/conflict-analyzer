"""
=============================================================================
MÓDULO 1: ESTRUCTURAS DE DATOS
Sistema de Análisis de Conflictos en Comunidades
=============================================================================
Define los modelos de datos centrales usando dataclasses y Pydantic.
La estructura es inmutable y trazable: cada mensaje conserva su contexto
completo para poder auditar cualquier puntuación generada.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enumeraciones de dominio
# ---------------------------------------------------------------------------

class ConflictSeverity(Enum):
    """Niveles de severidad calibrados para minimizar falsos positivos."""
    NONE       = 0   # Conversación normal
    LOW        = 1   # Tensión leve / desacuerdo razonado
    MODERATE   = 2   # Lenguaje elevado pero no tóxico
    HIGH       = 3   # Hostilidad clara, ataques personales
    CRITICAL   = 4   # Acoso, amenazas, toxicidad severa


class InteractionType(Enum):
    """Tipo de interacción detectada entre dos usuarios."""
    NEUTRAL       = "neutral"
    DISAGREEMENT  = "disagreement"    # Desacuerdo normal (saludable)
    TENSION       = "tension"         # Escalada de tono
    PROVOCATION   = "provocation"     # Intento deliberado de irritar
    HOSTILITY     = "hostility"       # Ataque directo
    HARASSMENT    = "harassment"      # Patrón repetido contra un usuario


# ---------------------------------------------------------------------------
# Estructuras de datos principales
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    Unidad atómica del sistema. Cada mensaje del foro/chat.

    Attributes:
        id:            Identificador único del mensaje.
        user_id:       Usuario que lo escribió.
        text:          Texto original, sin modificar.
        timestamp:     Cuándo se publicó.
        thread_id:     Hilo o conversación al que pertenece.
        reply_to_id:   ID del mensaje al que responde (si aplica).
        platform:      Origen: 'discord', 'forum', 'slack', etc.
    """
    id:           str
    user_id:      str
    text:         str
    timestamp:    datetime
    thread_id:    str
    reply_to_id:  Optional[str] = None
    platform:     str = "unknown"

    # Campos enriquecidos por el pipeline de análisis (se rellenan después)
    toxicity_score:        float = 0.0   # [0, 1] — probabilidad de toxicidad
    aggression_score:      float = 0.0   # [0, 1] — agresividad directa
    passive_aggression:    float = 0.0   # [0, 1] — hostilidad encubierta
    provocation_score:     float = 0.0   # [0, 1] — intento de provocar
    sentiment_polarity:    float = 0.0   # [-1, 1] — positivo vs negativo
    conflict_severity:     ConflictSeverity = ConflictSeverity.NONE
    analysis_explanation:  str = ""      # Por qué se asignó esta puntuación


@dataclass
class UserProfile:
    """
    Perfil acumulado de un usuario en la comunidad.
    Se actualiza de forma incremental conforme llegan mensajes nuevos.
    Todas las métricas se normalizan entre 0 y 1.
    """
    user_id:   str
    username:  str

    # --- Contadores básicos ---
    total_messages:    int = 0
    total_threads:     int = 0

    # --- Historial de conflictos ---
    conflict_initiated:  int = 0   # Veces que inició una escalada
    conflict_received:   int = 0   # Veces que fue objetivo de hostilidad
    warnings_received:   int = 0   # Avisos formales de moderación

    # --- Puntuaciones promedio (ventana deslizante últimos 30 días) ---
    avg_toxicity:          float = 0.0
    avg_aggression:        float = 0.0
    avg_passive_aggression: float = 0.0
    avg_provocation:       float = 0.0

    # --- Redes de interacción ---
    # Usuarios con los que más conflictos tiene: {user_id: frecuencia}
    conflict_partners:  dict = field(default_factory=dict)

    # --- Puntuación de riesgo global (0–100) ---
    risk_score:    float = 0.0
    risk_label:    str = "low"      # 'low' | 'medium' | 'high' | 'critical'
    risk_factors:  list = field(default_factory=list)  # Explicaciones


@dataclass
class ConflictEvent:
    """
    Un episodio de conflicto detectado: involucra ≥2 usuarios,
    tiene un inicio, posiblemente un pico, y se resuelve (o no).
    """
    event_id:       str
    thread_id:      str
    participants:   list              # Lista de user_ids
    start_time:     datetime
    end_time:       Optional[datetime] = None
    peak_severity:  ConflictSeverity = ConflictSeverity.LOW
    interaction_type: InteractionType = InteractionType.TENSION
    message_ids:    list = field(default_factory=list)   # Mensajes del evento
    resolved:       bool = False
    summary:        str = ""          # Descripción legible del conflicto


@dataclass
class CommunityReport:
    """
    Informe agregado del estado de salud de la comunidad.
    Generado periódicamente (diario / semanal / mensual).
    """
    report_id:      str
    period_start:   datetime
    period_end:     datetime
    total_messages: int = 0
    conflict_rate:  float = 0.0      # % mensajes con conflicto
    top_risk_users: list = field(default_factory=list)
    conflict_events: list = field(default_factory=list)
    health_score:   float = 100.0    # 0 (muy conflictiva) – 100 (saludable)
    recommendations: list = field(default_factory=list)
