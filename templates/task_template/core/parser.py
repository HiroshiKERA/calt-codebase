"""
Load-time preprocessor for [TASK NAME].

In CALT's pipeline, the dataset_load_preprocessor runs once when loading
the dataset from disk, before the UnifiedLexer tokenizes the strings.

Plain text data ("input # output" lines, the default output of
DatasetPipeline) needs no parser: CALT splits each line on '#' itself, and
core/train.py leaves the parser off. Use this file when you need to
  - read the JSON Lines files (dataset.save_json: true writes *_data.jsonl,
    one {"problem": ..., "answer": ...} object per line), or
  - transform samples at load time (e.g. rebuild SageMath objects from the
    strings and change their representation).

TODO: implement process_sample, then enable it in core/train.py.
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
            dict → {"problem": ..., "answer": ...} from JSON Lines

        Returns
        -------
        (input_text, target_text) — both tokenizable by the UnifiedLexer
        """
        # TODO: implement for your storage format (text or JSON Lines)
        # Example for text format:
        # if isinstance(source, str):
        #     parts = source.strip().split(" # ", maxsplit=1)
        #     return parts[0].strip(), parts[1].strip()

        # Example for JSON Lines (problem/answer as stored by DatasetPipeline):
        # if isinstance(source, dict):
        #     problem = source["problem"]
        #     answer = source["answer"]
        #     return format_input(problem), format_target(answer)

        raise NotImplementedError
