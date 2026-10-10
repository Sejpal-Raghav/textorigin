from src.services.execution import ExecutionContext, context_or_default

LABELS = ("human", "ai_written", "ai_paraphrased")


class ComparisonService:
    def __init__(self, classification, humanization):
        self.classification = classification
        self.humanization = humanization

    def compare(self, text: str, *, style="natural", strength="light", use_llm=False,
                api_key=None, similarity_threshold=None, seed=None, context: ExecutionContext | None = None) -> dict:
        context = context_or_default(context)
        context.progress("classifying_original")
        original = self.classification.classify(text, include_signals=False, context=context)
        context.progress("rewriting_for_compare")
        rewrite = self.humanization.humanize(
            text, style=style, strength=strength, use_llm=use_llm,
            api_key=api_key, similarity_threshold=similarity_threshold, seed=seed, context=context,
        )
        context.progress("classifying_revised")
        revised = self.classification.classify(rewrite["humanized_text"], include_signals=False, context=context)
        if original["model_version"] != revised["model_version"]:
            raise ValueError("Compare requires the same classifier version")
        context.check()
        return {"schema_version": "1.1", "operation": "compare", "style": style,
                "strength": strength, "original_text": text,
                "humanized_text": rewrite["humanized_text"],
                "original_scores": original["scores"], "revised_scores": revised["scores"],
                "delta_percentage_points": {key: (revised["scores"][key] - original["scores"][key]) * 100
                                            for key in LABELS},
                "original_analysis": original, "revised_analysis": revised,
                "rewrite": rewrite, "complete": True}
