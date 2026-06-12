"""
Load-time preprocessor for [TASK NAME].

In CALT's pipeline, the dataset_load_preprocessor runs once when loading
the dataset from disk, before the UnifiedLexer tokenizes the strings.

If your data is stored as plain text (formatter.py already returns strings),
this file may be minimal (just split on ' # ').

If your data is stored as pickle (SageMath objects), this file does the
conversion: pickle dict → (input_text, target_text).

TODO: Choose the appropriate base and implement process_sample.
"""

from typing import Any

from .formatter import format_input, format_target


class TaskParser:
    """
    Converts stored (problem, answer) to (input_text, target_text).

    Used as: io_pipeline.dataset_load_preprocessor = TaskParser()
    """

    def process_sample(self, source: str | dict[str, Any]) -> tuple[str, str]:
        """
        Convert one stored sample to a (input_text, target_text) pair.

        Parameters
        ----------
        source : str | dict
            str  → plain text line "input # output"
            dict → {"problem": ..., "answer": ...} from JSON Lines or pickle

        Returns
        -------
        (input_text, target_text) — both tokenizable by the UnifiedLexer
        """
        # TODO: implement for your storage format (text or pickle)
        # Example for text format:
        # if isinstance(source, str):
        #     parts = source.strip().split(" # ", maxsplit=1)
        #     return parts[0].strip(), parts[1].strip()

        # Example for pickle format:
        # if isinstance(source, dict):
        #     problem = source["problem"]
        #     answer = source["answer"]
        #     return format_input(problem), format_target(answer)

        raise NotImplementedError
