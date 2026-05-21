import glob
from datetime import datetime
from bs4 import BeautifulSoup
from pipeline import ConflictAnalysisPipeline
from data_structures import Message

html_files = sorted(glob.glob("ChatExport_2026-05-21/messages*.html"))
print(f"Archivos encontrados: {len(html_files)}")

messages = []
for filepath in html_files:
    with open(filepath, encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    for msg in soup.find_all("div", class_="message"):
        if "service" in msg.get("class", []):
            continue
        try:
            user = msg.find("div", class_="from_name")
            text = msg.find("div", class_="text")
            date = msg.find("div", class_="date")

            if not user or not text or not text.get_text().strip():
                continue

            fecha_str = date.get("title", "") if date else ""
            try:
                timestamp = datetime.strptime(fecha_str, "%d.%m.%Y %H:%M:%S")
            except Exception:
                timestamp = datetime(2026, 1, 1)

            messages.append(Message(
                id=msg.get("id", str(len(messages))),
                user_id=user.get_text().strip(),
                text=text.get_text().strip(),
                timestamp=timestamp,
                thread_id="telegram_group",
                reply_to_id=None,
                platform="telegram",
            ))
        except Exception:
            continue

print(f"Mensajes cargados: {len(messages)}")

pipeline = ConflictAnalysisPipeline()
pipeline.ingest_messages(messages)
risks = pipeline.compute_risk_scores()
report = pipeline.generate_community_report()

print("\n--- Top 10 usuarios por riesgo ---")
for uid, data in sorted(risks.items(), key=lambda x: x[1]["score"], reverse=True)[:10]:
    print(f"  {uid[:25]:25} {data['score']:5.1f}/100  [{data['label']}]")

print(f"\nSalud comunidad: {report.health_score:.0f}/100")
print(f"Tasa de conflicto: {report.conflict_rate*100:.1f}%")