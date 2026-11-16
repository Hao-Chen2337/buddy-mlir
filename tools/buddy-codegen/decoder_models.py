"""Single source of truth for the decoder-only LLM families buddy-mlir builds.

Every decoder-only model in ``models/<family>/`` runs the *same* generic
runtime (``runtime/models/DecoderRunner.cpp`` + ``DecoderResidentModel.cpp``)
and differs only by the compile-time constants that ``gen_config.py`` emits
into the generated ``ModelSession.h``.  Those constants are, in turn, derived
from the family-level facts recorded here plus the variant spec and the
HuggingFace ``config.json``.

This table is consumed by:

* ``gen_config.py``          -- derives the ``BUDDY_DECODER_*`` token/display
                                 macros (``derive_tokens``);
* ``gen_model_cmake.py``     -- renders the per-family ``CMakeLists.txt`` (or
                                 the loop in ``models/CMakeLists.txt``) so that
                                 *adding a decoder model is adding one entry
                                 here plus one ``models/<family>/specs/*.json``*.

Fields
------
display      : Human-readable name used in the buddy-cli banner.
tokenizer    : ``BUDDY_DECODER_TOKENIZER`` dispatch tag.  Names a
               ``TextContainer`` tokenizer *method family*, not the model
               family: ``qwen25`` and ``qwen3`` both use the Qwen3 byte-level
               BPE codec, so both carry ``"qwen3"``.  ``DecoderCodec.h`` maps
               this tag to the concrete ``tokenize*``/``revert*`` method.
eos          : End-of-sequence token id.  Non-``None`` pins the value even when
               the HF ``config.json`` disagrees (DeepSeek-R1 reports the Qwen
               base's ``<|im_end|>`` 151645 while it actually stops on
               ``<|end▁of▁sentence|>`` 151643).  ``None`` means "trust the HF
               ``config.json`` ``eos_token_id``" (unverified families).
pad          : Benign pad token id for fixed-length benchmark prompts.  ``None``
               means "trust the HF ``config.json`` ``pad_token_id``".
extra_stop   : Additional stop-token ids appended after ``eos``.
default_spec : The variant spec used when the model is configured with no
               explicit ``-DBUDDY_<FAMILY>_SPEC=`` (relative to the family dir).
cmake_prefix : Cache-variable prefix ``gen_model_cmake.py`` emits as the
               ``PREFIX`` argument of ``buddy_add_decoder_model``.  The build
               reads/writes ``BUDDY_<cmake_prefix>_SPEC``, ``..._HF_CONFIG``,
               ``..._LOCAL_MODEL``, ``..._COMPILE_JOBS``, ``..._TIERED_KV_CACHE``
               and ``..._TIERED_CACHE_SIZES``.
cmake_upper  : Mode A/B dir prefix, emitted as the ``UPPER`` argument.  The
               build reads/writes ``<cmake_upper>_BUILD_DIR`` and
               ``<cmake_upper>_MLIR_DIR``.
               Kept separate from ``cmake_prefix`` because deepseek_r1
               historically uses ``BUDDY_DSR1_*`` cache vars but
               ``DEEPSEEKR1_*`` dirs (``build_model.py`` / ``import_model.py``
               depend on those exact spellings).
build        : Whether the family is wired into the top-level build
               (``models/CMakeLists.txt``).  ``False`` keeps an unvalidated
               scaffold out of the default build.
vocab        : Path, relative to the repo root, of the family's tokenizer
               ``vocab.txt`` copied next to the built ``.rax``.  ``gen_model_cmake.py``
               renders it as the ``VOCAB`` argument of ``buddy_add_decoder_model``,
               which passes it through to ``buddy_add_model`` as ``ASSET_FILES``.
               ``None`` (or omitted) keeps the shared fallback
               ``examples/BuddyDeepSeekR1/vocab.txt`` -- the Qwen2.5 vocab -- for
               families whose per-family vocab.txt is not generated yet.
note         : What still has to happen before this family builds end-to-end.
"""

DECODER_MODELS = {
    # ── Validated references (round-trip byte-for-byte) ─────────────────────
    "qwen25": {
        "display": "Qwen2.5",
        "tokenizer": "qwen3",  # Qwen2.5 shares the Qwen3 byte-level BPE codec
        "eos": 151645,  # <|im_end|>, verified against Qwen/Qwen2.5-0.5B-Instruct
        "pad": 198,  # newline; benign pad for fixed-length benchmark prompts
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "QWEN25",
        "cmake_upper": "QWEN25",
        # Qwen2.5 shares the DeepSeek-R1 vocab file (the Qwen2.5 tokenizer),
        # which is also the fallback buddy_model.cmake used before per-family
        # vocab wiring, so pointing here keeps the copied file unchanged.
        "vocab": "examples/BuddyDeepSeekR1/vocab.txt",
        "build": True,
        "note": "reference model",
    },
    "deepseek_r1": {
        "display": "DeepSeekR1",
        "tokenizer": "deepseek",
        # DeepSeek-R1 ends a reply with <|end▁of▁sentence|> (151643); the Qwen2.5
        # base's <|im_end|> (151645) is the assistant token, not the R1 EOS, so
        # pin 151643 instead of trusting the eos_token_id transformers reports.
        "eos": 151643,
        "pad": 198,
        "extra_stop": [151647],  # <|EOT|> (end of thinking)
        "default_spec": "f32.json",
        "cmake_prefix": "DSR1",
        "cmake_upper": "DEEPSEEKR1",
        "vocab": "examples/BuddyDeepSeekR1/vocab.txt",
        "build": True,
        "note": "reference model",
    },
    # ── Target coverage (scaffolds present, NOT yet end-to-end validated) ──
    "qwen3": {
        "display": "Qwen3",
        "tokenizer": "qwen3",
        # <|im_end|>, verified against Qwen/Qwen3-0.6B config.json eos_token_id.
        "eos": 151645,
        # <|endoftext|>; config.json has no pad_token_id, generation_config.json
        # and tokenizer_config.json both use 151643.
        "pad": 151643,
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "QWEN3",
        "cmake_upper": "QWEN3",
        "vocab": "examples/BuddyQwen3/vocab.txt",
        "build": False,
        "note": "bias=false linears lowered via mm_op; QK-norm reuses the existing RMSNorm elementwise lowering; import reaches MLIR; build=True pending",
    },
    "llama": {
        "display": "Llama",
        "tokenizer": "llama",  # tokenizeLlama already in TextContainer.h
        # SmolLM-135M-Instruct config.json: eos_token_id=2 (</s>), pad_token_id=2.
        "eos": 2,
        "pad": 2,
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "LLAMA",
        "cmake_upper": "LLAMA",
        "vocab": "examples/BuddyLlama/vocab.txt",
        "build": True,
        "note": "end-to-end validated on SmolLM-135M-Instruct (bias=false linears via mm_op; vocab from HF tokenizer)",
    },
    "mistral": {
        "display": "Mistral",
        "tokenizer": "mistral",  # tokenizeMistral/revertMistral in TextContainer.h
        # </s>, verified against mistralai/Mistral-7B-v0.1 config.json (eos=2, bos=1).
        "eos": 2,
        # config.json has no pad_token_id; Mistral convention pads with </s> = eos
        # (matches tokenizeLlama's this->pad = 2).
        "pad": 2,
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "MISTRAL",
        "cmake_upper": "MISTRAL",
        "vocab": "examples/BuddyMistral/vocab.txt",
        "build": False,
        "note": "tokenizer ready; needs sliding-window mask in tosa.py",
    },
    "gemma": {
        "display": "Gemma",
        "tokenizer": "gemma",  # tokenizeGemma4 already in TextContainer.h
        "eos": None,  # Gemma uses <eos> = 1; verify config.json
        "pad": None,
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "GEMMA",
        "cmake_upper": "GEMMA",
        "build": False,
        "note": "tokenizer ready; needs GEGLU + norm paths in tosa.py + eos; TODO: vocab.txt",
    },
    "phi": {
        "display": "Phi",
        "tokenizer": "phi",  # tokenizePhi/revertPhi in TextContainer.h (Llama-style BPE)
        # <|endoftext|>, verified against microsoft/Phi-3-mini-4k-instruct config.json.
        "eos": 32000,
        # config.json sets pad_token_id = eos_token_id = 32000.
        "pad": 32000,
        "extra_stop": [],
        "default_spec": "f16.json",
        "cmake_prefix": "PHI",
        "cmake_upper": "PHI",
        "vocab": "examples/BuddyPhi/vocab.txt",
        "build": False,
        "note": "tokenizer ready; needs partial-RoPE/no-bias path",
    },
}


def families(build_only: bool = False) -> list[str]:
    """Return the family keys, optionally limited to build-enabled families."""
    return [
        f
        for f, e in DECODER_MODELS.items()
        if (not build_only or e.get("build"))
    ]
