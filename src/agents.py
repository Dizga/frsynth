import os
import random
import yaml

from openai import OpenAI
from schema import GeneratedReport, LabeledReport, output_config_for

MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class Generator:
    def __init__(self, model: str = MODEL, grid_path: str = "data/attribute_grid.yaml"):
        self.model = model
        with open(grid_path) as f:
            self.grid = yaml.safe_load(f)

    def generate_report(self) -> LabeledReport | None:

        category = random.choice(self.grid["categories"])
        severity = random.choice(self.grid["severities"])
        sector = random.choice(self.grid["sectors"])
        register = random.choice(self.grid["registers"])
        length = random.choice(self.grid["lengths"])
        noise = random.choice(self.grid["noise_levels"])
        has_distractor = random.choices(
            list(self.grid["distractor_weights"].keys()),
            weights=list(self.grid["distractor_weights"].values()),
        )[0]

        distractor_category = None
        distractor_prompt = ""
        if has_distractor:
            distractor_category = random.choice(self.grid["categories"])
            while distractor_category == category:
                distractor_category = random.choice(self.grid["categories"])

            distractor_prompt = f"Génère aussi une phrase d'information plus pertinente à la catégorie {distractor_category}, pour rendre la tâche de classification plus difficile. "

        prompt = (
            f"Génère un rapport d'incident industriel réaliste, en français, de catégorie {category} et de sévérité {severity}. "
            f"Le secteur concerné est {sector} avec un registre de type {register}. "
            f"Le rapport doit être de longueur {length} et de nature {noise}. "
            f"{distractor_prompt}"
            "Ne mentionne pas explicitement la catégorie ou la sévérité dans le texte du rapport. "
            "Réponds uniquement en JSON selon le schéma fourni."
        )

        response = client.chat.completions.create(
            model=self.model,
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
        except Exception as e:
            print(f"Could not parse generated report: {e}")
            return None

        return LabeledReport(
            report=report.report,
            category=category,
            severity=severity,
            sector=sector,
            register_=register,
            length=length,
            noise=noise,
            has_distractor=has_distractor,
            distractor_category=distractor_category,
        )