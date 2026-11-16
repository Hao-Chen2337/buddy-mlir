//===- DecoderRunner.cpp - generic decoder-only inference loop ------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Implements DecoderRunner::run().  This file owns session setup and
// dispatch:
//   - weight file loading
//   - chat template loading
//   - sampler creation
//   - dispatch to single-shot generation or interactive REPL
//
// buddy-cli calls this through the generic InferenceRunner interface.
//
//===----------------------------------------------------------------------===//

#include "buddy/runtime/models/DecoderRunner.h"
#include "buddy/LLM/ChatTemplate.h"
#include "buddy/LLM/ConversationManager.h"
#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/llm/BenchLoop.h"
#include "buddy/runtime/llm/InteractiveSession.h"
#include "buddy/runtime/llm/Perplexity.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/DecoderCodec.h"
#include "buddy/runtime/models/ModelSession.h"

#include "buddy/Core/Container.h"

using buddy::Text;

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

namespace {
/// Create a session for @p cfg (Mode A: .rax manifest, or Mode B: explicit
/// paths) and load its weights. Shared by run() and runBench().
std::unique_ptr<ModelSession>
createSession(const RunConfig &cfg, std::vector<std::string> &weightPaths,
              std::string &vocabPath, bool suppress) {
  std::unique_ptr<ModelSession> session;

  if (!cfg.raxPath.empty()) {
    printLog("Manifest: " + cfg.raxPath, suppress);
    ModelManifest manifest;
    session = ModelSession::createFromRax(cfg.raxPath, manifest);

    weightPaths = manifest.weightPaths;
    vocabPath = manifest.vocabPath.empty()
                    ? (std::filesystem::path(manifest.soPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : manifest.vocabPath;

    printLog("  .so     = " + manifest.soPath, suppress);
    for (const auto &wp : weightPaths)
      printLog("  weights = " + wp, suppress);
    printLog("  vocab   = " + vocabPath, suppress);
  } else {
    if (cfg.modelSoPath.empty())
      throw std::runtime_error("Mode B requires modelSoPath (--model-so).");
    if (cfg.weightsPath.empty())
      throw std::runtime_error("Mode B requires weightsPath (--weights).");

    weightPaths.push_back(cfg.weightsPath);
    vocabPath = cfg.vocabPath.empty()
                    ? (std::filesystem::path(cfg.modelSoPath).parent_path() /
                       "vocab.txt")
                          .string()
                    : cfg.vocabPath;

    ModelSession::Config mcfg;
    mcfg.modelSoPath = cfg.modelSoPath;
    printLog("Loading model: " + cfg.modelSoPath, suppress);
    session = ModelSession::create(mcfg);
  }

  // Reads weight files from disk into session-owned MemRefs (layout and
  // element types match the compiled variant; see manifest constant order).
  session->loadWeights(weightPaths);
  printLog("Weights loaded.", suppress);

  return session;
}

/// Benign newline token (already used by the per-model scaffolds) for padding
/// a fixed-length prompt; avoid EOS so the first sampled token is not a stop
/// token that would end decode immediately.
static constexpr int kPadToken = BUDDY_DECODER_PAD_TOKEN_ID;
} // namespace

//===----------------------------------------------------------------------===//
// DecoderRunner::run
//===----------------------------------------------------------------------===//

void DecoderRunner::run(const RunConfig &cfgIn) {
  RunConfig cfg = cfgIn;

  const bool suppress = cfg.suppressStats || cfg.streamJsonl;

  if (!suppress)
    std::cerr << "\033[33;1m" BUDDY_DECODER_DISPLAY_NAME
              << " Inference (buddy-cli / BuddyRuntime)\033[0m\n";

  // ── Chat template: load if provided ─────────────────────────────────────
  std::vector<long long> stopTokenIds = BUDDY_DECODER_STOP_TOKEN_IDS;
  std::unique_ptr<buddy::ChatTemplate> chatTmpl;

  if (!cfg.chatTemplatePath.empty()) {
    chatTmpl = std::make_unique<buddy::ChatTemplate>(
        buddy::ChatTemplate::fromFile(cfg.chatTemplatePath));
    for (int id : chatTmpl->stopTokenIds()) {
      if (std::find(stopTokenIds.begin(), stopTokenIds.end(), id) ==
          stopTokenIds.end()) {
        stopTokenIds.push_back(static_cast<long long>(id));
      }
    }
    printLog("Chat template loaded: " + cfg.chatTemplatePath, suppress);
  }

  // ── Create session and load weights ──────────────────────────────────────
  std::vector<std::string> weightPaths;
  std::string vocabPath;
  std::unique_ptr<ModelSession> session =
      createSession(cfg, weightPaths, vocabPath, suppress);

  // ── Perplexity mode: score a token stream instead of generating ─────────
  if (!cfg.perplexityIdsPath.empty()) {
    if (cfg.perplexityContext > BUDDY_DECODER_MAX_TOKEN_LEN)
      throw std::runtime_error(
          "--ppl-context " + std::to_string(cfg.perplexityContext) +
          " exceeds the KV cache (" +
          std::to_string(BUDDY_DECODER_MAX_TOKEN_LEN) + " tokens)");
    PerplexityOptions opts;
    opts.context = cfg.perplexityContext;
    opts.maxChunks = cfg.perplexityChunks;
    opts.quiet = suppress;
    runPerplexity(*session, readTokenIds(cfg.perplexityIdsPath), opts);
    return;
  }

  printLog("Vocab: " + vocabPath, suppress);
  printLog("KV cache: " + std::to_string(BUDDY_DECODER_KV_LAYERS) + " x {1," +
               std::to_string(BUDDY_DECODER_HEAD_NUM) + "," +
               std::to_string(BUDDY_DECODER_MAX_TOKEN_LEN) + "," +
               std::to_string(BUDDY_DECODER_HIDDEN_SIZE) + "} f32",
           suppress);

  // ── Model-specific text codec ───────────────────────────────────────────
  TextCodec codec = makeDecoderCodec();

  // ── Create Sampler ───────────────────────────────────────────────────────
  buddy::Sampler sampler(cfg.samplerConfig);

  // ── Interactive or single-shot mode ──────────────────────────────────────
  if (cfg.interactive) {
    if (!chatTmpl) {
      throw std::runtime_error(
          "--interactive requires --chat-template <path.json>");
    }
    buddy::ConversationManager conv(
        std::move(*chatTmpl), [&vocabPath](const std::string &text) -> size_t {
          Text<size_t, 2> tmp(text);
          tokenizeDecoder(tmp, vocabPath);
          return tmp.getTokenCnt();
        });
    if (!cfg.prompt.empty())
      conv.setSystemPrompt(cfg.prompt);

    runInteractiveSession(*session, vocabPath, cfg, stopTokenIds, conv, codec,
                          sampler);
  } else {
    // Single-shot: format prompt with chat template if available.
    std::string finalPrompt = cfg.prompt;
    if (chatTmpl) {
      std::vector<buddy::Message> msgs = {{"user", cfg.prompt}};
      finalPrompt = chatTmpl->apply(msgs);
    }

    GenerationResult result =
        runGeneration(finalPrompt, *session, vocabPath, cfg.maxNewTokens,
                      stopTokenIds, sampler, codec, suppress, cfg.streamJsonl);

    if (!suppress)
      printStats(result, /*verbose=*/true);
  }
}

//===----------------------------------------------------------------------===//
// DecoderRunner::runBench
//===----------------------------------------------------------------------===//

DecoderRunner::DecoderRunner() = default;
DecoderRunner::~DecoderRunner() = default;

bool DecoderRunner::runBench(const RunConfig &cfgIn, int warmupReps, int reps,
                             std::vector<BenchmarkResult> &out) {
  RunConfig cfg = cfgIn;
  cfg.suppressStats = true;
  const bool suppress = true;

  // A fixed prompt length is padded or truncated to exactly that many tokens;
  // beyond the KV cache the padded token buffer would overrun the prefill
  // kernel's expected shape.
  const TextCodec codec = makeDecoderCodec();
  if (cfg.promptLength > codec.maxTokenLen) {
    std::fprintf(stderr,
                 "DecoderRunner: promptLength %d exceeds the KV cache (%d "
                 "tokens)\n",
                 cfg.promptLength, codec.maxTokenLen);
    return false;
  }

  // buddy-bench calls runBench() once per (promptLen, nGen) configuration.
  // Keep the loaded session alive across those calls: reloading the weights
  // for every configuration would dominate the benchmark and change nothing
  // being measured. Different model files invalidate the cached session.
  const std::string sessionKey = cfg.raxPath + '\x01' + cfg.modelSoPath +
                                 '\x01' + cfg.weightsPath + '\x01' +
                                 std::to_string(cfg.batchSize);
  if (!benchSession_ || sessionKey != benchSessionKey_) {
    benchSession_ =
        createSession(cfg, benchWeightPaths_, benchVocabPath_, suppress);
    benchSessionKey_ = sessionKey;
  }
  ModelSession &session = *benchSession_;
  const std::string &vocabPath = benchVocabPath_;

  buddy::Sampler sampler(cfg.samplerConfig);
  const std::vector<long long> stopTokenIds = BUDDY_DECODER_STOP_TOKEN_IDS;

  // The tiered KV cache rounds the prompt up to a cache tier; report the
  // positions the prefill actually computed.
  runBenchLoop(
      session, vocabPath, cfg, sampler, stopTokenIds, codec, kPadToken,
      [&session] { return session.lastPrefillWorkLen(); }, warmupReps, reps,
      out);
  return true;
}

} // namespace runtime
} // namespace buddy
