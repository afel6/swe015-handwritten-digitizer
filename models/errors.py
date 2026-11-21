"""Exceptions whose message is written for the end user and may be shown verbatim in an HTTP 400."""

from __future__ import annotations

__all__ = ["EmptyAnswerError", "UserInputError"]


class UserInputError(ValueError):
    """A problem with what the user sent (sentence, empty boxes, input too long). Never wraps library text."""


class EmptyAnswerError(UserInputError):
    """The submitted answer contains no boxes with visible handwriting."""
