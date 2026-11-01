"""spaCy NER processor for extracting entities from article titles."""
import logging
from typing import Any

import spacy

logger = logging.getLogger(__name__)

_KEEP_LABELS = {"ORG", "GPE", "PERSON"}
_MIN_ENTITY_LEN = 2


class NERProcessor:
    def __init__(self, model: str = "en_core_web_sm") -> None:
        self._nlp = spacy.load(model, disable=["parser", "lemmatizer"])
        logger.info("NER model loaded: %s", model)

    def extract_batch(
        self, titles: list[str], batch_size: int = 64
    ) -> list[list[dict[str, str]]]:
        """Run NER on a list of titles; return one entity list per title.

        Each entity dict has keys 'name' and 'type' (ORG, GPE, or PERSON).
        """
        results: list[list[dict[str, str]]] = []
        for doc in self._nlp.pipe(titles, batch_size=batch_size):
            entities = [
                {"name": ent.text.strip(), "type": ent.label_}
                for ent in doc.ents
                if ent.label_ in _KEEP_LABELS
                and len(ent.text.strip()) >= _MIN_ENTITY_LEN
            ]
            results.append(entities)
        return results
