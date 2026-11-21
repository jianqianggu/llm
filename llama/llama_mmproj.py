#!/usr/bin/env python3
"""Shared mmproj discovery for the 7900 XTX llama-proxy / llama-switch stack.

llama.cpp vision requires a separate multimodal projector GGUF (`--mmproj`).
Unsloth (and Fara) ship that file next to the text weights, but the filename
rarely matches the quant of the chat GGUF (e.g. IQ4_XS weights + BF16 mmproj).
"""
from __future__ import annotations

import glob
import os
import re

MODELS_DIR = os.environ.get("LLAMA_MODELS_DIR", "/mnt/fast_models/unsloth")
MODELS_DIR_MIRROR = os.environ.get("LLAMA_MODELS_DIR_MIRROR", "/mnt/LLM_Models/unsloth")

MODEL_ALIASES = {
    "qwen3.5-35b": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "qwen3.6-35b": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "qwen3.8-27b": "Qwen3.8-27B-Q4_K_M.gguf",
    "qwen3.8-27b-uncensored": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
    "qwen3.8-uncensored": "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf",
    "fara1.5": "Fara1.5-27B-Q4_K_M.gguf",
    "fara1.5-27b": "Fara1.5-27B-Q4_K_M.gguf",
    "qwen3.5-4b": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
}

# Last-resort relative paths under MODELS_DIR when auto-detect has nothing
# nearby. Filenames come from the live host layout in endpoints.json.
KNOWN_MMPROJ_BY_GGUF = {
    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf": (
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/"
        "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
        "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-GGUF/"
        "mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
    ),
    "Fara1.5-27B-Q4_K_M.gguf": (
        "Fara1.5-27B-GGUF/Fara1.5-27.mmproj-q8_0.gguf",
        "Fara1.5-27B-GGUF/Fara1.5-27B.mmproj-q8_0.gguf",
    ),
}

_QUANT_TOKENS = {
    "bf16",
    "f16",
    "fp16",
    "f32",
    "q2",
    "q3",
    "q3_k_m",
    "q3_k_s",
    "q4",
    "q4_0",
    "q4_1",
    "q4_k",
    "q4_k_m",
    "q4_k_s",
    "q4_k_xl",
    "q5",
    "q5_0",
    "q5_k_m",
    "q5_k_s",
    "q6",
    "q6_k",
    "q8",
    "q8_0",
    "iq1_s",
    "iq2_xxs",
    "iq2_xs",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "ud",
    "xl",
    "k",
    "m",
    "s",
    "xs",
    "xxs",
    "nl",
    "gguf",
    "mt",
    "mtp",
}

_GENERIC_MMPROJ_TOKENS = {"mmproj", "clip", "merger", "vision", "projector", "proj"}


def is_mmproj_filename(name):
    if not name:
        return False
    lower = name.lower()
    return "mmproj" in lower and lower.endswith(".gguf")


def clean_model_name(raw_name):
    if not raw_name:
        return ""
    name = re.sub(r"^external::[^:]+::", "", raw_name)
    return os.path.basename(name).strip()


def llama_switch_command(model_id, mmproj=None):
    cmd = ["llama-switch", model_id]
    if mmproj:
        cmd.extend(["--mmproj", mmproj])
    return cmd


def iter_chat_ggufs(models_dir):
    pattern = os.path.join(models_dir, "**", "*.gguf")
    for path in sorted(glob.glob(pattern, recursive=True)):
        if is_mmproj_filename(os.path.basename(path)):
            continue
        yield path


def _tokens(name):
    stem = os.path.basename(name)
    stem = re.sub(r"\.gguf$", "", stem, flags=re.I)
    stem = re.sub(r"^mmproj[-_.]*", "", stem, flags=re.I)
    stem = re.sub(r"[-_.]*mmproj[-_.]*", "-", stem, flags=re.I)
    parts = re.split(r"[-_.]+", stem.lower())
    tokens = set()
    for part in parts:
        if not part or part in _QUANT_TOKENS or part in _GENERIC_MMPROJ_TOKENS:
            continue
        if part.isdigit():
            continue
        tokens.add(part)
    return tokens


def _models_dir(explicit=None):
    if explicit:
        return os.path.abspath(explicit)
    return os.path.abspath(os.environ.get("LLAMA_MODELS_DIR", MODELS_DIR))


def _mirror_roots(models_dir=None):
    roots = []
    models_dir = _models_dir(models_dir)
    for src, dst in (
        ("/mnt/fast_models/", "/mnt/LLM_Models/"),
        ("/mnt/LLM_Models/", "/mnt/fast_models/"),
    ):
        if src in models_dir + os.sep:
            roots.append(models_dir.replace(src, dst, 1))
    custom = os.environ.get("LLAMA_MODELS_DIR_MIRROR")
    if custom:
        roots.append(os.path.abspath(custom))
    # Host default mirror only when we are on that layout (not unit-test tmpdirs).
    if os.path.abspath(MODELS_DIR) == models_dir or "/mnt/fast_models/" in models_dir:
        roots.append(os.path.abspath(MODELS_DIR_MIRROR))
    return roots


def _mirror_path(path, models_dir=None):
    replacements = (
        ("/mnt/fast_models/", "/mnt/LLM_Models/"),
        ("/mnt/LLM_Models/", "/mnt/fast_models/"),
    )
    mirrors = []
    path = os.path.abspath(path)
    for src, dst in replacements:
        if src in path:
            mirrors.append(path.replace(src, dst, 1))
    models_dir = _models_dir(models_dir)
    prefix = models_dir.rstrip(os.sep) + os.sep
    if path.startswith(prefix):
        rel = os.path.relpath(path, models_dir)
        for root in _mirror_roots(models_dir):
            mirrors.append(os.path.join(root, rel))
    return mirrors


def _add_dir(dirs, path):
    if not path:
        return
    path = os.path.abspath(path)
    if path not in dirs and os.path.isdir(path):
        dirs.append(path)


def _quant_bonus(name):
    lower = name.lower()
    for token, bonus in (
        ("bf16", 6),
        ("fp16", 5),
        ("f16", 5),
        ("q8_0", 4),
        ("q8", 3),
        ("q6", 2),
        ("q5", 1),
        ("q4", 0),
    ):
        if token in lower:
            return bonus
    return 0


def _candidate_dirs(model_path, models_dir=None):
    model_path = os.path.abspath(model_path)
    model_dir = os.path.dirname(model_path)
    model_tokens = _tokens(os.path.basename(model_path))
    folder_tokens = _tokens(os.path.basename(model_dir))
    dirs = []
    _add_dir(dirs, model_dir)
    _add_dir(dirs, os.path.dirname(model_dir))

    for mirrored in _mirror_path(model_path, models_dir=models_dir):
        _add_dir(dirs, os.path.dirname(mirrored))
        _add_dir(dirs, os.path.dirname(os.path.dirname(mirrored)))

    try:
        for child in os.listdir(model_dir):
            child_path = os.path.join(model_dir, child)
            if not os.path.isdir(child_path):
                continue
            child_tokens = _tokens(child)
            if child_tokens & (model_tokens | folder_tokens):
                _add_dir(dirs, child_path)
    except OSError:
        pass

    parent = os.path.dirname(model_dir)
    try:
        for sibling in os.listdir(parent):
            sibling_path = os.path.join(parent, sibling)
            if not os.path.isdir(sibling_path):
                continue
            sibling_tokens = _tokens(sibling)
            if sibling_tokens & (model_tokens | folder_tokens):
                _add_dir(dirs, sibling_path)
    except OSError:
        pass

    roots = [_models_dir(models_dir)]
    roots.extend(_mirror_roots(models_dir))

    # Conventional Unsloth: <models_dir>/<Family>-GGUF/mmproj-*.gguf
    seen_roots = set()
    for root in roots:
        root = os.path.abspath(root) if root else ""
        if not root or root in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(root)
        try:
            for child in os.listdir(root):
                child_path = os.path.join(root, child)
                if not os.path.isdir(child_path):
                    continue
                if _tokens(child) & model_tokens:
                    _add_dir(dirs, child_path)
        except OSError:
            continue

    return dirs


def _list_mmproj(directory):
    found = []
    try:
        for name in os.listdir(directory):
            if is_mmproj_filename(name):
                found.append(os.path.join(directory, name))
    except OSError:
        return []
    return found


def _score_mmproj(model_path, mmproj_path):
    model_path = os.path.abspath(model_path)
    mmproj_path = os.path.abspath(mmproj_path)
    model_dir = os.path.dirname(model_path)
    mmproj_dir = os.path.dirname(mmproj_path)
    model_tokens = _tokens(os.path.basename(model_path))
    mmproj_tokens = _tokens(os.path.basename(mmproj_path))
    overlap = model_tokens & mmproj_tokens
    extra = mmproj_tokens - model_tokens

    score = 0
    if mmproj_dir == model_dir:
        score += 10000
    elif os.path.basename(mmproj_dir) and (
        _tokens(os.path.basename(mmproj_dir)) & model_tokens
    ):
        score += 400
    score += 100 * len(overlap)
    score += _quant_bonus(os.path.basename(mmproj_path))

    if extra:
        score -= 80 * len(extra)
        if mmproj_dir != model_dir:
            score -= 4000
    elif mmproj_dir != model_dir and not overlap:
        score -= 5000
    return score


def _known_mmproj(model_path, models_dir=None):
    name = os.path.basename(model_path)
    rels = KNOWN_MMPROJ_BY_GGUF.get(name, ())
    roots = [_models_dir(models_dir)]
    roots.extend(_mirror_roots(models_dir))
    roots.append(os.path.dirname(os.path.abspath(model_path)))
    roots.append(os.path.dirname(os.path.dirname(os.path.abspath(model_path))))
    for rel in rels:
        for root in roots:
            candidate = os.path.join(root, rel)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            candidate = os.path.join(root, os.path.basename(rel))
            if os.path.isfile(candidate) and is_mmproj_filename(os.path.basename(candidate)):
                return os.path.abspath(candidate)
    return None


def find_mmproj(model_path, models_dir=None):
    """Return the best mmproj GGUF for a chat GGUF, or None if text-only."""
    if not model_path:
        return None
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        known = _known_mmproj(model_path, models_dir=models_dir)
        return known

    candidates = []
    for directory in _candidate_dirs(model_path, models_dir=models_dir):
        candidates.extend(_list_mmproj(directory))

    # De-dupe while preserving order.
    seen = set()
    unique = []
    for path in candidates:
        path = os.path.abspath(path)
        if path == model_path or path in seen:
            continue
        if not os.path.isfile(path):
            continue
        seen.add(path)
        unique.append(path)

    if unique:
        unique.sort(key=lambda path: (_score_mmproj(model_path, path), path), reverse=True)
        best = unique[0]
        if _score_mmproj(model_path, best) > 0:
            return best

    return _known_mmproj(model_path, models_dir=models_dir)


def resolve_model_filename(requested_model_id, installed_filenames):
    query = clean_model_name(requested_model_id)
    if not query:
        return None
    if query.lower() in MODEL_ALIASES:
        target_name = MODEL_ALIASES[query.lower()]
        for name in installed_filenames:
            if name.lower() == target_name.lower():
                return name
    for name in installed_filenames:
        if name == query:
            return name
    case_insensitive = [name for name in installed_filenames if name.lower() == query.lower()]
    if len(case_insensitive) == 1:
        return case_insensitive[0]
    partial = [
        name
        for name in installed_filenames
        if query.lower() in name.lower() or name.lower() in query.lower()
    ]
    if len(partial) == 1:
        return partial[0]
    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: llama_mmproj.py <model.gguf>", file=sys.stderr)
        raise SystemExit(2)
    found = find_mmproj(sys.argv[1])
    if found:
        print(found)
    else:
        print("NO_MMPROJ")
        raise SystemExit(1)
