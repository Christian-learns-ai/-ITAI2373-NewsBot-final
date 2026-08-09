import json

def export_insights_report(insights_data: dict, output_path: str = "report.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(insights_data, f, indent=4)
    return output_path
