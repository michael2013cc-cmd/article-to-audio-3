#!/usr/bin/env python3
"""Command line: ``python -m a2a make examples/sample.txt -o out/sample.mp3``"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import audit as auditmod
from . import content as contentmod
from . import keys as keymod
from . import pipeline, script, stt, synth, verify
from .lock import WorkdirBusy


def _spoken_lines(script_path: str) -> list[str]:
    """The script body without the chapter markers.

    The markers are never read aloud, so their digits and Latin letters
    must not take part in key extraction or auditing.
    """
    with open(script_path, encoding="utf-8") as fh:
        text = fh.read()
    return [
        line for line in text.splitlines()
        if not line.startswith("@@")
    ]


def _cmd_make(args: argparse.Namespace) -> int:
    keys = None
    if args.keys:
        keys = contentmod.load_keys(args.keys)
    elif args.auto_keys:
        lines = _spoken_lines(args.script)
        candidates = keymod.extract_candidates(lines)
        # Model-name digits (MI355X's 355) are guaranteed listen-back
        # misses, not data. Drop them BEFORE ranking so the budget still
        # fills with real figures - and say what was dropped, never
        # silently.
        suspects = set(keymod.suspect_model_keys(candidates, lines))
        if suspects:
            print(
                f"⚠ 剔除 {len(suspects)} 个型号/版本号内部的假键："
                f"{', '.join(sorted(suspects))}（如确要验证，用 --keys 手工指定）",
                flush=True,
            )
            candidates = [c for c in candidates if c["key"] not in suspects]
        picked = keymod.rank_candidates(candidates, args.auto_keys)
        keys = [{"key": c["key"], "alts": c["alts"]} for c in picked]
        print(f"自动挑了 {len(keys)} 个验证键", flush=True)

    try:
        report = pipeline.run(
            args.script,
            args.output,
            workdir=args.workdir,
            voice=args.voice,
            rate=args.rate,
            bitrate=args.bitrate,
            title=args.title,
            album=args.album,
            artist=args.artist,
            year=args.year,
            resume=args.resume,
            write_transcript=not args.no_transcript,
            word_level=args.word_level,
            keys=keys,
            keys_path=args.keys,
            stt_model=args.stt_model,
            stt_device=args.stt_device,
        )
    except WorkdirBusy as busy:
        # Exit code 5, same contract as v1: the caller (or the agent
        # driving this) sees "busy" and picks another output name or
        # waits - instead of silently corrupting the other run.
        print(f"\n{busy}", file=sys.stderr)
        return 5
    if not report["ok"]:
        print("\nfinished, but the file did not pass verification:", file=sys.stderr)
        for problem in report["problems"]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    content_report = report.get("content")
    if content_report and not content_report.get("ok", True):
        print("\n回听发现关键数字缺失：", file=sys.stderr)
        for row in content_report.get("missing", []):
            print(f"  - {row['key']}", file=sys.stderr)
        return 1
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    report = verify.check_file(args.file, expect_chapters=args.chapters)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _cmd_parse(args: argparse.Namespace) -> int:
    chapters = script.load(args.script)
    if args.json:
        print(json.dumps(
            [{"id": c.id, "title": c.title, "chars": len(c.body)} for c in chapters],
            ensure_ascii=False, indent=2,
        ))
    else:
        for i, c in enumerate(chapters):
            print(f"{i:2d}  @{c.id:<4} {len(c.body):>6} chars  {c.title}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    report = auditmod.audit(args.script, args.source, allow=args.allow or [])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["problems"] and not args.force_ok:
        print("\nreview the flagged items before synthesizing.", file=sys.stderr)
        return 1
    return 0


def _cmd_keys(args: argparse.Namespace) -> int:
    lines = _spoken_lines(args.script)
    candidates = keymod.extract_candidates(lines)
    if args.lint:
        warns = keymod.lint_numbers(lines)
        print(json.dumps(warns, ensure_ascii=False, indent=2))
        return 0
    picked = keymod.rank_candidates(candidates, args.top)
    # A human is reading this table, so suspects are FLAGGED, not dropped -
    # unlike --auto-keys, where nobody is watching.
    suspects = set(keymod.suspect_model_keys(picked, lines))
    if args.json:
        print(json.dumps(
            [{**c, "suspect": c["key"] in suspects} for c in picked],
            ensure_ascii=False, indent=2,
        ))
        return 0
    for c in picked:
        alt = c["alts"][1] if len(c["alts"]) > 1 else ""
        mark = "  ⚠ 型号内部" if c["key"] in suspects else ""
        print(f"{c['key']:<12} tier {c['tier']}  score {c['score']:>3}  {c['context'][:40]}"
              + (f"   (alt {alt})" if alt else "") + mark)
    if suspects:
        print(f"\n⚠ {len(suspects)} 个键只出现在型号/版本号内部，回听大概率报未命中（确认后删除）："
              f"{', '.join(sorted(suspects))}", flush=True)
    if args.out:
        contentmod.save_keys(
            args.out,
            [{"key": c["key"], "alts": c["alts"], "tier": c["tier"],
              "context": c["context"], "suspect": c["key"] in suspects}
             for c in picked],
        )
        print(f"键已写入 {args.out}", flush=True)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    """Check a hand-written key file against the script before spending time
    on synthesis."""
    lines = _spoken_lines(args.script)
    keys = contentmod.load_keys(args.keys)
    bogus = keymod.preflight(keys, lines)
    if bogus:
        print("这些键不在稿子里，回听时永远不会命中：", file=sys.stderr)
        for label in bogus:
            print(f"  - {label}", file=sys.stderr)
        return 1
    # A key that IS in the script but only inside a model name is the
    # subtler failure: it will be spoken, mangled, and reported as a miss.
    # Warning, not hard failure - a real figure can sit next to an English
    # word; a human should look.
    suspects = keymod.suspect_model_keys(keys, lines)
    if suspects:
        print("⚠ 这些键只出现在型号/版本号内部，回听大概率报未命中（确认后删除）：")
        for label in suspects:
            print(f"  - {label}")
    print(f"{len(keys)} 个键都在稿子里（其中 {len(suspects)} 个疑似型号假键）。")
    return 0


def _cmd_listen(args: argparse.Namespace) -> int:
    keys = contentmod.load_keys(args.keys) if args.keys else []
    report = contentmod.check_content(
        args.file,
        keys,
        model_size=args.stt_model,
        device=args.stt_device,
        transcript_path=args.out,
        quiet=False,
    )
    print(json.dumps(
        {
            "ok": report.get("ok"),
            "total": report.get("total"),
            "found": report.get("found"),
            "missing": [r["key"] for r in report.get("missing", [])],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    # a2a/__init__.py already forced UTF-8 on stdout/stderr, because a
    # cp950 console cannot encode a chapter title and would kill the run.
    parser = argparse.ArgumentParser(
        prog="a2a",
        description="listening script -> chaptered, verified mp3",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("make", help="build the mp3 (and the transcript)")
    p.add_argument("script", help="path to the @@ID|Title script")
    p.add_argument("-o", "--output", required=True, help="output .mp3")
    p.add_argument("--workdir", help="scratch dir (default: <output>_work)")
    p.add_argument("--voice", default=synth.DEFAULT_VOICE)
    p.add_argument("--rate", default=synth.DEFAULT_RATE)
    p.add_argument("--bitrate", default="64k")
    p.add_argument("--title", help="ID3 title; also the transcript heading")
    p.add_argument("--album")
    p.add_argument("--artist")
    p.add_argument("--year")
    p.add_argument("--resume", action="store_true",
                   help="keep already-synthesized chapters in the workdir")
    p.add_argument("--no-transcript", action="store_true")
    p.add_argument("--word-level", action="store_true",
                   help="also write a word-by-word script")
    p.add_argument("--keys", help="JSON file of verification keys")
    p.add_argument("--auto-keys", type=int, metavar="N",
                   help="pick N verification keys from the script itself")
    p.add_argument("--stt-model", default=stt.DEFAULT_MODEL,
                   help="faster-whisper model (small / medium / large-v3)")
    p.add_argument("--stt-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=_cmd_make)

    p = sub.add_parser("check", help="measure an existing mp3")
    p.add_argument("file")
    p.add_argument("--chapters", type=int, help="expected chapter count")
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("parse", help="show the chapter table of a script")
    p.add_argument("script")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_parse)

    p = sub.add_parser("audit", help="flag numbers and stray tokens in a script")
    p.add_argument("script")
    p.add_argument("--source", help="original document to diff numbers against")
    p.add_argument("--allow", action="append", default=[],
                   help="extra Latin token to accept (repeatable)")
    p.add_argument("--force-ok", action="store_true", help="always exit 0")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("keys", help="extract verification keys from a script")
    p.add_argument("script")
    p.add_argument("--top", type=int, default=25, help="how many keys to keep")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", help="write the keys to this JSON file")
    p.add_argument("--lint", action="store_true",
                   help="only report risky number spellings")
    p.set_defaults(func=_cmd_keys)

    p = sub.add_parser("preflight", help="check a key file against the script")
    p.add_argument("script")
    p.add_argument("keys")
    p.set_defaults(func=_cmd_preflight)

    p = sub.add_parser("listen", help="listen back to an mp3 and check the keys")
    p.add_argument("file")
    p.add_argument("--keys", help="JSON file of verification keys")
    p.add_argument("--out", help="write the heard transcript here")
    p.add_argument("--stt-model", default=stt.DEFAULT_MODEL)
    p.add_argument("--stt-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.set_defaults(func=_cmd_listen)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
