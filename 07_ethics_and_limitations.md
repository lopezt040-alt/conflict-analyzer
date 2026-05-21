# Consideraciones Éticas y Limitaciones Técnicas

## Limitaciones Técnicas

### 1. Comprensión del contexto
- Los modelos no entienden el historial de una relación entre usuarios
- El sarcasmo, la ironía y el humor negro generan falsos positivos
- Diferencias culturales y de argot pueden sesgar las puntuaciones
- Los modelos preentrenados en inglés tienen rendimiento inferior en español

### 2. Sesgos del modelo
- Detoxify y modelos similares están entrenados principalmente en texto en inglés
- Ciertos acentos regionales, jerga juvenil o expresiones coloquiales se marcan incorrectamente
- Los grupos minoritarios pueden recibir puntuaciones más altas si el modelo fue entrenado con datos sesgados
- **Mitigación**: evaluar el sistema separando métricas por idioma y subgrupo demográfico

### 3. Gaming del sistema
- Usuarios hostiles pueden aprender a evitar palabras clave
- La hostilidad encubierta (dog whistles, jerga de in-group) es difícil de detectar
- El sistema mide síntomas, no intenciones

### 4. Deriva temporal
- El lenguaje evoluciona; un modelo de hace 2 años puede estar obsoleto
- Reentrenar periódicamente con datos recientes de tu comunidad

---

## Consideraciones Éticas

### Principio 1: El sistema es una herramienta, no un juez
- Ninguna puntuación debe traducirse automáticamente en una sanción
- Un moderador humano debe revisar siempre los casos de riesgo alto/crítico
- El sistema ayuda a priorizar la revisión, no a reemplazarla

### Principio 2: Transparencia
- Los usuarios deben saber que existe moderación asistida por IA
- Las razones de una sanción no deben ser "el algoritmo lo dijo"
- Las puntuaciones son orientativas, no evidencia

### Principio 3: Derecho a recurrir
- Cualquier usuario debe poder impugnar una clasificación
- El proceso de apelación debe ser humano y explicable
- Guardar logs de decisiones para auditoría

### Principio 4: Privacidad y retención de datos
- Los mensajes analizados son datos personales (RGPD/GDPR en Europa)
- Definir política de retención: ¿cuánto tiempo guardar los análisis?
- Anonimizar datos antes de usarlos para entrenamiento
- No compartir puntuaciones individuales con terceros

### Principio 5: Equidad algorítmica
- Medir si el sistema penaliza más a ciertos grupos (idioma, género, origen)
- Usar métricas de equidad: equalized odds, demographic parity
- Documentar y publicar los resultados de equidad

### Principio 6: Escala mínima necesaria
- No desplegar en comunidades donde no hay moderadores humanos
- El sistema no es adecuado para plataformas con usuarios vulnerables sin supervisión
- Evitar usar para decisiones de alto impacto (expulsión permanente) sin revisión

---

## Checklist antes del despliegue

- [ ] Evaluado con datos reales de tu comunidad (no solo demos)
- [ ] Acuerdo inter-anotador κ ≥ 0.60 en el ground truth
- [ ] Métricas de equidad calculadas por idioma y subgrupo
- [ ] Proceso de apelación documentado
- [ ] Política de retención de datos definida
- [ ] Moderadores entrenados en cómo interpretar las puntuaciones
- [ ] Monitorización de deriva del modelo activada
- [ ] Usuarios informados de la existencia del sistema
