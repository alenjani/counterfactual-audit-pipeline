"""Demographic prompt construction for counterfactual generation.

Care taken with phrasing to:
  - Avoid stereotyped associations (e.g., do not bind ethnicity to occupation)
  - Use neutral, descriptive language
  - Hold non-target attributes constant across counterfactuals
"""
from __future__ import annotations

from typing import Any

# Fitzpatrick skin type → descriptive prompt fragment
FITZPATRICK_PROMPTS = {
    1: "very fair skin tone, Fitzpatrick type I",
    2: "fair skin tone, Fitzpatrick type II",
    3: "medium-light skin tone, Fitzpatrick type III",
    4: "medium skin tone, Fitzpatrick type IV",
    5: "medium-dark skin tone, Fitzpatrick type V",
    6: "dark skin tone, Fitzpatrick type VI",
}

GENDER_PROMPTS = {
    "male": "male presentation",
    "female": "female presentation",
}

AGE_PROMPTS = {
    20: "in their early 20s",
    30: "in their 30s",
    40: "in their 40s",
    50: "in their 50s",
    60: "in their 60s",
    70: "in their 70s",
}


def build_demographic_prompt(
    base_attributes: dict[str, str], demographic_attributes: dict[str, Any]
) -> str:
    """Compose a generation prompt holding base attributes constant and varying demographics."""
    fragments = ["a portrait photograph of a person"]

    if "skin_tone" in demographic_attributes:
        fragments.append("with " + FITZPATRICK_PROMPTS[demographic_attributes["skin_tone"]])
    if "gender" in demographic_attributes:
        fragments.append(GENDER_PROMPTS[demographic_attributes["gender"]])
    if "age" in demographic_attributes:
        fragments.append(AGE_PROMPTS[demographic_attributes["age"]])

    pose = base_attributes.get("pose", "frontal")
    expression = base_attributes.get("expression", "neutral expression")
    lighting = base_attributes.get("lighting", "soft even studio lighting")
    fragments.extend([f"{pose} pose", expression, lighting])
    fragments.append("high detail, photorealistic, sharp focus")

    return ", ".join(fragments)
