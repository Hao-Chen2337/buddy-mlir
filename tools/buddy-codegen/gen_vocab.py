#!/usr/bin/env python3
# ===- gen_vocab.py - Export a tokenizer's vocab to buddy vocab.txt ---------===//
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===//
#
# buddy-mlir's decoder runtime reads a plain-text vocab.txt in which each line
# is one token and the 0-based line number is the token id (see
# frontend/Interfaces/buddy/LLM/TextContainer.h, Text<T, N>::loadVocab).
# This script turns a tokenizer file into that format:
#
#   * SentencePiece binary model  (.model) -> pieces 0..N-1 in id order
#   * HuggingFace fast tokenizer  (.json)   -> model.vocab ids, then any
#       added_tokens ids beyond the base vocab
#
# The output must use LF line endings: some vocabs (e.g. Mistral, Phi) contain
# tokens with a literal '\r' inside them, and a CRLF file would append '\r' to
# every token read back by getline().  This script always writes '\n' and
# refuses to write a token that itself contains '\n', which would break the
# line-based format.
#
# Usage:
#   python3 gen_vocab.py tokenizer.model -o vocab.txt
#   python3 gen_vocab.py tokenizer.json  -o vocab.txt
#
# ===----------------------------------------------------------------------===//

import argparse
import json
import os
import sys


def tokens_from_sentencepiece(path):
    """Return the SentencePiece pieces in id order (piece 0 = <unk>, ...)."""
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=path)
    return [processor.id_to_piece(i) for i in range(processor.get_piece_size())]


def tokens_from_hf_json(path):
    """Return the token strings in id order from a HF fast tokenizer.json.

    The base vocabulary lives in ``model.vocab`` (token -> id); ``added_tokens``
    may extend it with higher ids (e.g. Phi-3's 32000..32010) or restate
    special tokens that are already part of the base vocab.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{path}: no 'model' section (not a tokenizer.json?)")

    vocab = model.get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"{path}: model.vocab is missing or not an object")

    id_to_token = {int(tok_id): token for token, tok_id in vocab.items()}

    for entry in data.get("added_tokens", []):
        if (
            not isinstance(entry, dict)
            or "id" not in entry
            or "content" not in entry
        ):
            raise ValueError(f"{path}: malformed added_tokens entry: {entry!r}")
        id_to_token[int(entry["id"])] = entry["content"]

    max_id = max(id_to_token)
    tokens = []
    for i in range(max_id + 1):
        if i not in id_to_token:
            raise ValueError(
                f"{path}: vocab has a gap at token id {i} (max id {max_id})"
            )
        tokens.append(id_to_token[i])
    return tokens


def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".model":
        return "sentencepiece"
    if ext == ".json":
        return "hf-json"
    raise ValueError(f"{path}: cannot infer format from extension {ext!r}")


def write_vocab(tokens, out_path):
    if not tokens:
        raise ValueError("tokenizer produced an empty vocab")

    for i, token in enumerate(tokens):
        if not token:
            raise ValueError(f"empty token at id {i}")
        if "\n" in token:
            raise ValueError(
                f"token id {i} contains a literal newline; "
                "it cannot be represented in the one-token-per-line format"
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    # newline="\n" keeps the output LF-only on every platform.
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for token in tokens:
            f.write(token + "\n")

    cr_tokens = sum(1 for token in tokens if "\r" in token)
    print(
        f"wrote {os.path.abspath(out_path)}: "
        f"{len(tokens)} tokens, {cr_tokens} contain a literal '\\r'"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "tokenizer", help="SentencePiece .model or HF tokenizer.json"
    )
    ap.add_argument(
        "-o", "--output", required=True, help="destination vocab.txt"
    )
    ap.add_argument(
        "--format",
        choices=("sentencepiece", "hf-json"),
        help="tokenizer format (default: infer from the file extension)",
    )
    args = ap.parse_args()

    fmt = args.format or detect_format(args.tokenizer)
    if fmt == "sentencepiece":
        tokens = tokens_from_sentencepiece(args.tokenizer)
    else:
        tokens = tokens_from_hf_json(args.tokenizer)

    write_vocab(tokens, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
