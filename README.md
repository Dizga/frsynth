# FRSynth

Specialization of a small open model for French industrial-incident classification using only synthetic data.

The pipeline generates schema-valid labeled examples with a frontier LLM, filters them with a generator–critic loop, LoRA-trains smaller model on the curated set, and measures the result against a hand verified test set.

## Disclamer

This is a pet project to get a better grasp of the full pipeline of training a small open model with synthetic data.