from typing import Literal

from pydantic import BaseModel, ConfigDict

Category = Literal[
    "mécanique",
    "électrique",
    "hydraulique/pneumatique",
    "logiciel/contrôle",
    "sécurité/humain",
    "qualité/process",
]
Severity = Literal["mineur", "majeur", "critique"]


class GeneratedReport(BaseModel):

    model_config = ConfigDict(extra="forbid")

    report: str


class CriticVerdict(BaseModel):

    model_config = ConfigDict(extra="forbid")

    judged_category: Category
    judged_severity: Severity
    plausible: bool
    notes: str


class PredictedLabel(BaseModel):

    model_config = ConfigDict(extra="forbid")

    category: Category
    severity: Severity


def output_config_for(model_cls: type[BaseModel]) -> dict:

    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "schema": model_cls.model_json_schema(),
                "strict": True,
            },
        }
    }
