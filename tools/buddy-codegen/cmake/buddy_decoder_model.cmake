# ===- buddy_decoder_model.cmake - registry-driven decoder model build -----===//
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
# buddy_add_decoder_model() is the *only* thing a decoder-only LLM family
# needs in its models/<family>/CMakeLists.txt.  Every decoder family runs the
# same generic runtime (runtime/models/Decoder*.cpp) and differs only by the
# family identifiers below and its variant spec.  The per-family facts live in
# tools/buddy-codegen/decoder_models.py, which gen_model_cmake.py renders into
# the one-line buddy_add_decoder_model() call in each family's CMakeLists.txt.
#
#   buddy_add_decoder_model(
#     NAME qwen25 PREFIX QWEN25 UPPER QWEN25
#     DEFAULT_SPEC f16.json DISPLAY "Qwen2.5"
#     VOCAB examples/BuddyDeepSeekR1/vocab.txt)
#
#   NAME          lower-case family / target name (also models/<NAME>/)
#   PREFIX        cache-variable prefix  -> BUDDY_<PREFIX>_SPEC etc.
#   UPPER         Mode A/B variable prefix -> <UPPER>_BUILD_DIR / <UPPER>_MLIR_DIR
#   DEFAULT_SPEC  variant spec under models/<NAME>/specs/ used when
#                 -DBUDDY_<PREFIX>_SPEC is left empty
#   DISPLAY       human-readable name for cache-variable help strings
#   VOCAB         repo-root-relative path of the family's vocab.txt, passed
#                 through to buddy_add_model as ASSET_FILES.  Omit it to keep
#                 the shared fallback examples/BuddyDeepSeekR1/vocab.txt.
#
# PREFIX and UPPER are kept separate because deepseek_r1 historically uses
# BUDDY_DSR1_* for its cache variables but DEEPSEEKR1_* for its Mode A/B dirs
# (and build_model.py / import_model.py depend on those exact spellings).
#
# ===----------------------------------------------------------------------===//

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

macro(buddy_add_decoder_model)
  cmake_parse_arguments(
    BDM "" "NAME;PREFIX;UPPER;DEFAULT_SPEC;DISPLAY;VOCAB" "" ${ARGN})

  if(NOT BDM_NAME OR NOT BDM_PREFIX OR NOT BDM_UPPER OR NOT BDM_DEFAULT_SPEC)
    message(FATAL_ERROR
      "buddy_add_decoder_model: NAME, PREFIX, UPPER and DEFAULT_SPEC are required")
  endif()
  if(NOT BDM_DISPLAY)
    set(BDM_DISPLAY "${BDM_NAME}")
  endif()

  ProcessorCount(_BDM_DEFAULT_COMPILE_JOBS)
  if(_BDM_DEFAULT_COMPILE_JOBS EQUAL 0)
    set(_BDM_DEFAULT_COMPILE_JOBS 1)
  endif()

  # ── Per-family configure knobs (cache vars) ────────────────────────────────
  set(BUDDY_${BDM_PREFIX}_SPEC
    "${CMAKE_CURRENT_SOURCE_DIR}/specs/${BDM_DEFAULT_SPEC}"
    CACHE FILEPATH "Variant spec JSON, e.g. specs/${BDM_DEFAULT_SPEC}")
  set(BUDDY_${BDM_PREFIX}_HF_CONFIG ""
    CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
  set(BUDDY_${BDM_PREFIX}_LOCAL_MODEL ""
    CACHE PATH "Optional: local HF model directory (weights import; sets ${BDM_UPPER}_MODEL_PATH)")
  set(BUDDY_${BDM_PREFIX}_COMPILE_JOBS "${_BDM_DEFAULT_COMPILE_JOBS}"
    CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")
  set(BUDDY_${BDM_PREFIX}_TIERED_KV_CACHE ""
    CACHE STRING "Auto/ON/OFF: build ${BDM_DISPLAY} with tiered KV cache entrypoints")
  set(BUDDY_${BDM_PREFIX}_TIERED_CACHE_SIZES ""
    CACHE STRING "Semicolon-separated tiered KV cache sizes; auto-detected from spec if empty")

  # Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
  set(${BDM_UPPER}_BUILD_DIR "" CACHE PATH
    "Mode A: dir containing pre-built forward/subgraph .o files")
  set(${BDM_UPPER}_MLIR_DIR "" CACHE PATH
    "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

  if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()

  # Read tiered-KV-cache intent from the variant spec so an empty cache var
  # follows the spec, not a hard-coded default.
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import json,sys; s=json.load(open(sys.argv[1])); raw=s.get('tiered_kv_cache', False); enabled=False; sizes=[]\nif isinstance(raw, dict):\n  enabled=bool(raw.get('enabled', True)); sizes=raw.get('cache_sizes', s.get('cache_sizes', []))\nelse:\n  enabled=bool(raw); sizes=s.get('cache_sizes', [])\nif isinstance(sizes, str): sizes=[x.strip() for x in sizes.split(',') if x.strip()]\nif enabled and not sizes: sizes=[32,64,128,256,512,1024]\nprint(('ON' if enabled else 'OFF') + '|' + ','.join(str(x) for x in sizes))"
            "${BUDDY_${BDM_PREFIX}_SPEC}"
    OUTPUT_VARIABLE _BDM_TIERED_FROM_SPEC
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _BDM_TIERED_PARSE_RESULT)

  set(_BDM_TIERED_ENABLED_FROM_SPEC OFF)
  set(_BDM_TIERED_SIZES_FROM_SPEC "")
  if(_BDM_TIERED_PARSE_RESULT EQUAL 0)
    string(REPLACE "|" ";" _BDM_TIERED_FIELDS "${_BDM_TIERED_FROM_SPEC}")
    list(GET _BDM_TIERED_FIELDS 0 _BDM_TIERED_ENABLED_FROM_SPEC)
    list(LENGTH _BDM_TIERED_FIELDS _BDM_TIERED_FIELD_COUNT)
    if(_BDM_TIERED_FIELD_COUNT GREATER 1)
      list(GET _BDM_TIERED_FIELDS 1 _BDM_TIERED_SIZES_CSV)
      string(REPLACE "," ";" _BDM_TIERED_SIZES_FROM_SPEC "${_BDM_TIERED_SIZES_CSV}")
    endif()
  endif()

  if(BUDDY_${BDM_PREFIX}_TIERED_KV_CACHE STREQUAL "")
    set(_BDM_TIERED_ENABLED "${_BDM_TIERED_ENABLED_FROM_SPEC}")
  else()
    set(_BDM_TIERED_ENABLED "${BUDDY_${BDM_PREFIX}_TIERED_KV_CACHE}")
  endif()

  if(BUDDY_${BDM_PREFIX}_TIERED_CACHE_SIZES STREQUAL "")
    set(_BDM_TIERED_CACHE_SIZES "${_BDM_TIERED_SIZES_FROM_SPEC}")
  else()
    set(_BDM_TIERED_CACHE_SIZES "${BUDDY_${BDM_PREFIX}_TIERED_CACHE_SIZES}")
  endif()

  set(_BDM_ASSET_ARGS)
  if(BDM_VOCAB)
    list(APPEND _BDM_ASSET_ARGS ASSET_FILES "${CMAKE_SOURCE_DIR}/${BDM_VOCAB}")
  endif()

  buddy_add_model(
    NAME "${BDM_NAME}"
    SPEC "${BUDDY_${BDM_PREFIX}_SPEC}"
    RUNNER_SRC DecoderRunner.cpp
    RUNNER_PLUGIN_SRC DecoderRunnerPlugin.cpp
    MODEL_PLUGIN_SRC DecoderResidentModelPlugin.cpp
    EXTRA_SRCS DecoderResidentModel.cpp
    HF_CONFIG "${BUDDY_${BDM_PREFIX}_HF_CONFIG}"
    LOCAL_MODEL "${BUDDY_${BDM_PREFIX}_LOCAL_MODEL}"
    BUILD_DIR "${${BDM_UPPER}_BUILD_DIR}"
    MLIR_DIR "${${BDM_UPPER}_MLIR_DIR}"
    COMPILE_JOBS "${BUDDY_${BDM_PREFIX}_COMPILE_JOBS}"
    TIERED_KV_CACHE "${_BDM_TIERED_ENABLED}"
    TIERED_CACHE_SIZES "${_BDM_TIERED_CACHE_SIZES}"
    ${_BDM_ASSET_ARGS}
  )
endmacro()
