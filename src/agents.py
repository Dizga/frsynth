import os
import random
import yaml

from openai import OpenAI
from schema import GeneratedReport, output_config_for

MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def generate_report():

    with open("data/attribute_grid.yaml") as f:
        grid = yaml.safe_load(f)

    category = random.choice(grid["categories"])
    severity = random.choice(grid["severities"])
    sector = random.choice(grid["sectors"])
    register = random.choice(grid["registers"])
    length = random.choice(grid["lengths"])
    noise = random.choice(grid["noise_levels"])
    has_distractor = random.choices(list(grid["distractor_weights"].keys()), weights=grid["distractor_weights"].values())[0]

    prompt = (
        f"Génère un rapport d'incident industriel réaliste, en français, dans la catégorie {category} et de sévérité {severity}. "
        f"Le secteur concerné est {sector} et le registre est {register}. "
        f"Le rapport doit être de longueur {length} et contenir un niveau de nature {noise}. "
        "Ne mentionne pas explicitement la catégorie ou la sévérité dans le texte du rapport. "
        "Réponds uniquement en JSON selon le schéma fourni."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **output_config_for(GeneratedReport),
    )

    raw = response.choices[0].message.content
    try:
        report = GeneratedReport.model_validate_json(raw)
        return report.report
    except Exception as e:
        print(f"Could not parse generated report: {e}")
        return None