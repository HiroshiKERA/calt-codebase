"""
Convert math objects to tokenizable strings for [TASK NAME].

This module defines the STRING FORMAT of the task:
  - How is the input represented as text?
  - How is the output represented as text?

These strings are later tokenized by CALT's UnifiedLexer.
Make sure the tokens you use are in the lexer.yaml vocabulary.

Format
------
Input  : [describe format]
Output : [describe format]

Example
-------
[concrete example: object → "string"]
"""

# TODO: define INNER_SEP if you have multi-element inputs/outputs
INNER_SEP = " | "


def format_input(problem) -> str:
    """
    Convert the problem object to its input string representation.

    TODO: implement this.
    """
    raise NotImplementedError


def format_target(answer) -> str:
    """
    Convert the answer object to its target string representation.

    TODO: implement this.
    """
    raise NotImplementedError
