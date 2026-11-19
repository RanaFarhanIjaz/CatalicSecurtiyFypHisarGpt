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
        Paragraph("HisarGPT", styles["Title"]),
        Paragraph("Powered by Catalic Security", styles["Italic"]),
        Spacer(1, 8),
        Paragraph("Assessment report", styles["Heading2"]),
        Paragraph("Local authorized test — automated candidates require human review.", styles["Italic"]),
        Spacer(1, 12),
        Paragraph(f"Assessment ID: {escape(result['assessment_id'])}", styles["BodyText"]),
        Paragraph(f"Target: {escape(result['target_url'])}", styles["BodyText"]),
        Paragraph(f"Mode: {escape(result['mode'])}", styles["BodyText"]),
        Paragraph(f"Prompt strategy: {escape(str(result.get('prompt_mode', 'predefined')))}", styles["BodyText"]),
        Paragraph(f"Score: {result['score']}/100 — {escape(result['grade'])}", styles["Heading2"]),
        Paragraph("Review status: Pending human review", styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("AI Judge Summary", styles["Heading2"]),
        Paragraph(escape(result["judge_summary"]), styles["BodyText"]),
        Spacer(1, 12),
    ]
    brief = result.get("target_brief")
    if brief:
        mode_label = brief.get("knowledge_mode", result.get("mode", "unauthenticated"))
        story += [
            Paragraph("Target brief (main agent)", styles["Heading2"]),
            Paragraph(f"Domain: {escape(brief['domain'])}. {escape(brief['summary'])}", styles["BodyText"]),
        ]
        if mode_label == "authenticated":
            prior_items = []
            if brief.get("known_role"):
                prior_items.append(f"Known role: {escape(brief['known_role'])}")
            if brief.get("architecture"):
                prior_items.append(f"Architecture: {escape(brief['architecture'])}")
            if brief.get("known_tools"):
                prior_items.append(f"Declared tools: {escape(', '.join(brief['known_tools'][:6]))}")
            if brief.get("known_constraints"):
                prior_items.append(f"Known constraints: {escape('; '.join(brief['known_constraints'][:6]))}")
            if brief.get("known_system_prompt"):
                n = len(brief["known_system_prompt"])
                prior_items.append(f"Known system prompt: on file ({n} chars). A safe target must NOT reveal it.")
            if brief.get("prior_source"):
                prior_items.append(f"Prior source: {escape(brief['prior_source'])}")
            story += [
                Paragraph("Assessment Mode: 🔐 Authenticated (Informed)", styles["Heading3"]),
                Paragraph("Prior knowledge was supplied by the operator and/or the target system.", styles["Italic"]),
            ]
            for item in prior_items:
                story.append(Paragraph(f"• {item}", styles["BodyText"]))
        else:
            story.append(Paragraph("Assessment Mode: Unauthenticated — Blind testing (no prior information)", styles["BodyText"]))
        story.append(Spacer(1, 12))
    story.append(Paragraph("Observations", styles["Heading2"]))
    rows = [["Agent", "Source", "Candidate", "Target response"]]
    for o in result["observations"]:
        agent = o["agent"] if isinstance(o, dict) else o.agent
        candidate = o["candidate"] if isinstance(o, dict) else o.candidate
        response = o["response"] if isinstance(o, dict) else o.response
        source = o.get("prompt_source", "predefined") if isinstance(o, dict) else o.prompt_source
        rows.append([agent, source, "Yes" if candidate else "No",
                     Paragraph(escape(response[:700]), styles["BodyText"])])
    table = Table(rows, colWidths=[95, 65, 50, 285], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(table)

    # --- Research Section (Multi-Agent Pentesting Comparative Analysis) ---
    research = result.get("research_section")
    if research:
        pentesters = research.get("pentester_results") or research.get("model_results") or []
        story += [
            Spacer(1, 20),
            Paragraph("Research Section: Multi-Agent Pentesting Comparative Analysis", styles["Heading1"]),
            Paragraph(
                f"Target Model: {escape(str(research.get('target_url', result.get('target_url', ''))))}  |  "
                f"Total Pentesters: {len(pentesters)}  |  "
                f"Most Aggressive Pentester: {escape(str(research.get('most_effective_pentester', 'N/A')))}",
                styles["BodyText"]
            ),
            Spacer(1, 8),
            Paragraph("Comparative Research Analysis", styles["Heading2"]),
            Paragraph(escape(research.get("comparative_summary", "")), styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Pentesting Agents Performance Matrix", styles["Heading2"]),
        ]
        pentester_rows = [["Pentesting Agent", "Type", "Target Score", "Grade", "Vulns Found", "Status"]]
        for pr in pentesters:
            score = pr.get("target_score", pr.get("score", 0))
            grade = pr.get("target_grade", pr.get("grade", ""))
            vulns = pr.get("vulnerabilities_found", pr.get("candidate_count", 0))
            error = pr.get("error", "")
            pentester_rows.append([
                escape(pr.get("pentester_label", pr.get("model_label", pr.get("pentester_id", "")))),
                escape(pr.get("pentester_type", pr.get("model_type", ""))),
                f"{score}/100",
                escape(grade),
                str(vulns),
                escape("Error" if error else "Completed"),
            ])
        pt_table = Table(pentester_rows, colWidths=[130, 80, 60, 65, 75, 85], repeatRows=1)
        pt_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243fbe")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(pt_table)

    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return str(path)
