import os
import random
import yaml

from openai import OpenAI
from schema import Category, CriticVerdict, GeneratedReport, LabeledReport, Severity, output_config_for

MODEL = "deepseek/deepseek-v4-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GRID_PATH = "data/attribute_grid.yaml"

client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def load_grid(path: str = GRID_PATH) -> dict:
    with open(path) as f:
        grid = yaml.safe_load(f)
    return grid


def _rubric_block(title: str, definitions: dict[str, str]) -> str:
    lines = "\n".join(f"  - {label} : {desc}" for label, desc in definitions.items())
    return f"{title} :\n{lines}"

class Generator:
    def __init__(self, model: str = MODEL, grid_path: str = GRID_PATH):
        self.model = model
        self.grid = load_grid(grid_path)

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

            distractor_prompt = f"Inclus un détail trompeur qui évoque les symptômes typiques d'une panne de nature {distractor_category}, intégré naturellement au récit, sans jamais nommer cette catégorie ni la signaler comme une note à part. "

        category_def = self.grid["category_definitions"][category]
        severity_def = self.grid["severity_definitions"][severity]

        prompt = (
            f"Génère un rapport d'incident industriel réaliste, en français, de catégorie {category} ({category_def}) "
            f"et de sévérité {severity} ({severity_def}). "
            f"Le secteur concerné est {sector} avec un registre de type {register}. "
            f"Le rapport doit être de longueur {length} et de nature {noise}. "
            f"{distractor_prompt}"
            "N'explique jamais la catégorie ni la sévérité : pas de formule du type « classé comme… », pas de mention d'une catégorie. "
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
    

class Critic:
    def __init__(self, model: str = MODEL, grid_path: str = GRID_PATH):
        self.model = model
        self.grid = load_grid(grid_path)

    def critique(self, report: LabeledReport) -> CriticVerdict:
        category_rubric = _rubric_block("Catégories possibles", self.grid["category_definitions"])
        severity_rubric = _rubric_block("Sévérités possibles", self.grid["severity_definitions"])
        prompt = (
            f"Voici un rapport d'incident industriel :\n\n{report.report}\n\n"
            f"En te basant uniquement sur le contenu du rapport, sans faire d'inférences externes, réponds aux questions suivantes :\n"
            f"1. Quelle est la catégorie de l'incident ? Choisis parmi :\n{category_rubric}\n"
            f"2. Quelle est la sévérité de l'incident ? Choisis parmi :\n{severity_rubric}\n"
            f"3. Le rapport est-il plausible et cohérent pour un incident de cette nature ? (true or false)\n"
            f"4. Fournis toute note ou observation pertinente sur les éléments du rapport qui ont guidé tes jugements, en particulier en cas de doute ou d'ambiguïté.\n\n"
            "Réponds uniquement en JSON selon le schéma fourni."
        )

        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            **output_config_for(CriticVerdict),
        )

        raw = response.choices[0].message.content
        try:
            verdict = CriticVerdict.model_validate_json(raw)
        except Exception as e:
            print(f"Could not parse critic verdict: {e}")
            raise

        return verdict