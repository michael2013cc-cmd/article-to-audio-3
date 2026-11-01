#!/usr/bin/env python3
"""a2a: article-to-audio - turn a written document into a verified mp3.

The pipeline is:

    listening script  ->  speech  ->  chaptered, phone-loud mp3
                                        + a transcript with a timestamp
                                          for every sentence
                                        + a listen-back that checks the
                                          figures were actually spoken

Modules are usable individually (``numbers``, ``keys``, ``match``,
``synth``, ``audio``, ``tags``, ``verify``, ``transcript``, ``audit``,
``stt``, ``content``) or together through ``pipeline.run``.
"""

import sys

from . import align, audio, audit, content, keys, match, numbers
from . import pipeline, script, stt, synth, tags, transcript, verify
from .script import Chapter, dump, load, parse

__all__ = [
    "Chapter", "align", "audio", "audit", "content", "dump", "keys",
    "load", "match", "numbers", "parse", "pipeline", "script", "stt",
    "synth", "tags", "transcript", "verify",
]

__version__ = "0.1.0"


def _force_utf8_stdio() -> None:
    """Make the stdio streams UTF-8, whatever the console's code page is.

    Windows consoles default to a legacy code page (cp950 on a zh-TW box).
    Printing a chapter title containing a character that page cannot encode
    raises UnicodeEncodeError and kills the run - and because output is
    usually piped, Python decodes with the ANSI code page rather than the
    console's UTF-8 one. This runs at import time so that both the CLI and
    direct library use are covered.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # pragma: no cover - odd streams
            pass


_force_utf8_stdio()
