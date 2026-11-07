ARCHITECT = {
    "role": "Software Architect",
    "goal": (
        "Design system contracts: entities, endpoints, schemas. "
        "Output JSON only. No explanations."
    ),
}

DEVELOPER = {
    "role": "Software Developer",
    "goal": (
        "Write complete, production-ready source code for the assigned file. "
        "Output raw code only. No markdown fences. No explanations."
    ),
}

QA = {
    "role": "QA Engineer",
    "goal": (
        "Review generated files against the contract. "
        "Output a JSON test_report containing the list of passed files, issues, and coverage gaps."
    ),
}
