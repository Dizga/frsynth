import random
import json

from agents import Critic
from schema import LabeledReport

with open("data/raw.jsonl", encoding="utf-8") as f:
    reports = [LabeledReport.model_validate_json(line) for line in f if line.strip()]

random.shuffle(reports)

samples_reports = reports[:150]

pro_critic = Critic("deepseek/deepseek-v4-pro")
standard_critic = Critic("deepseek/deepseek-v4-flash")
pro_verdicts = []
standard_verdicts = []
passed_by_pro = 0
passed_by_standard = 0
mismatchs = 0

for report in samples_reports:
    pro_verdict = pro_critic.critique(report)
    standard_verdict = standard_critic.critique(report)

    passed_by_pro+=pro_verdict.plausible
    passed_by_standard+=standard_verdict.plausible
    mismatchs+= pro_verdict.plausible != standard_verdict.plausible

    pro_verdicts.append({"report": report.report, **pro_verdict.model_dump()})
    standard_verdicts.append({"report": report.report, **standard_verdict.model_dump()})


with open("data/pro_verdicts.jsonl", "w", encoding="utf-8") as f:
    for item in pro_verdicts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("data/standard_verdicts.jsonl", "w", encoding="utf-8") as f:
    for item in standard_verdicts:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Passed by pro: {passed_by_pro}/{len(samples_reports)}")
print(f"Passed by standard: {passed_by_standard}/{len(samples_reports)}")
print(f"Mismatchs: {mismatchs}/{len(samples_reports)}")