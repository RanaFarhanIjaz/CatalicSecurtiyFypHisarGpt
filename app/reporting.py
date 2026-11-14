from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from xml.sax.saxutils import escape

def create_pdf(result: dict) -> str:
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{result['assessment_id']}.pdf"
    styles = getSampleStyleSheet()
    story = [
        Paragraph("HisarGPT Assessment Report", styles["Title"]),
        Paragraph("Local authorized test — automated candidates require human review.", styles["Italic"]),
        Spacer(1, 12),
        Paragraph(f"Assessment ID: {escape(result['assessment_id'])}", styles["BodyText"]),
        Paragraph(f"Target: {escape(result['target_url'])}", styles["BodyText"]),
        Paragraph(f"Mode: {escape(result['mode'])}", styles["BodyText"]),
        Paragraph(f"Score: {result['score']}/100 — {escape(result['grade'])}", styles["Heading2"]),
        Paragraph("Review status: Pending human review", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("AI Judge Summary", styles["Heading2"]),
        Paragraph(escape(result["judge_summary"]), styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Observations", styles["Heading2"]),
    ]
    rows = [["Agent", "Candidate", "Target response"]]
    for o in result["observations"]:
        agent = o["agent"] if isinstance(o, dict) else o.agent
        candidate = o["candidate"] if isinstance(o, dict) else o.candidate
        response = o["response"] if isinstance(o, dict) else o.response
        rows.append([agent, "Yes" if candidate else "No", Paragraph(escape(response[:700]), styles["BodyText"])])
    table = Table(rows, colWidths=[110, 55, 330], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(table)
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return str(path)
