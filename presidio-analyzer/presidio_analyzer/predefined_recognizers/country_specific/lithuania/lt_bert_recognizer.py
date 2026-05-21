import torch
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification

from presidio_analyzer.entity_recognizer import EntityRecognizer
from presidio_analyzer.recognizer_result import RecognizerResult


DEFAULT_LT_ENTITY_MAPPING: Dict[str, str] = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
}


class LtBertRecognizer(EntityRecognizer):
    ENTITIES = list(DEFAULT_LT_ENTITY_MAPPING.values())

    def __init__(
        self,
        model_name: str = "VSSA-SDSA/LT-NER-modernBERT",
        label_mapping: Optional[Dict[str, str]] = None,
        supported_entities: Optional[List[str]] = None,
        name: str = "LtBertRecognizer",
        supported_language: str = "lt",
        threshold: float = 0.70,
        **kwargs,
    ):
        if label_mapping is None:
            label_mapping = DEFAULT_LT_ENTITY_MAPPING

        if supported_entities is None:
            supported_entities = list(label_mapping.values())

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name,
        )

        self.model_name = model_name
        self.label_map = label_mapping
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)

    def analyze(self, text, entities, nlp_artifacts=None):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True,
        )

        offsets = inputs.pop("offset_mapping")[0]
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        pred_ids = torch.argmax(logits, dim=-1).tolist()
        scores = torch.softmax(logits, dim=-1).max(dim=-1).values.tolist()

        raw_predictions = []

        for idx, pred_id in enumerate(pred_ids):
            label = self.model.config.id2label[pred_id]
            score = float(scores[idx])

            if label == "O":
                continue

            clean_label = label
            if label.startswith("B-") or label.startswith("I-"):
                clean_label = label[2:]

            mapped = self.label_map.get(clean_label)
            if mapped is None:
                continue

            if entities and mapped not in entities:
                continue

            if score < self.threshold:
                continue

            start, end = offsets[idx].tolist()
            if start == end:
                continue

            span_text = text[start:end]
            if not span_text.strip():
                continue

            raw_predictions.append(
                {
                    "entity_type": mapped,
                    "start": start,
                    "end": end,
                    "score": score,
                }
            )

        merged = self._merge_predictions(raw_predictions, text)

        return [
            RecognizerResult(
                entity_type=item["entity_type"],
                start=item["start"],
                end=item["end"],
                score=item["score"],
            )
            for item in merged
        ]

    def _merge_predictions(self, predictions, text):
        if not predictions:
            return []

        predictions = sorted(predictions, key=lambda x: (x["start"], x["end"]))
        merged = [predictions[0]]

        for current in predictions[1:]:
            prev = merged[-1]

            same_entity = current["entity_type"] == prev["entity_type"]
            close_or_touching = current["start"] <= prev["end"] + 1

            if same_entity and close_or_touching:
                prev["end"] = max(prev["end"], current["end"])
                prev["score"] = max(prev["score"], current["score"])
            else:
                merged.append(current)

        cleaned = []
        for item in merged:
            span_text = text[item["start"]:item["end"]]
            if not span_text.strip():
                continue
            if len(span_text.strip()) < 2:
                continue
            cleaned.append(item)

        return cleaned