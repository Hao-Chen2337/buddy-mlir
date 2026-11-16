//===- DecoderRunner.h - generic decoder-only inference runner ------------===//
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

#ifndef BUDDY_RUNTIME_MODELS_DECODERRUNNER_H
#define BUDDY_RUNTIME_MODELS_DECODERRUNNER_H

#include "buddy/runtime/core/InferenceRunner.h"

#include <memory>
#include <string>
#include <vector>

namespace buddy {
namespace runtime {

class ModelSession;

/// Full inference runner for decoder-only models (Qwen2.5, DeepSeek R1, ...).
///
/// The compile-time shape and token constants come from the generated
/// ModelSession.h (BUDDY_DECODER_*), so this same source is compiled once per
/// model family against that family's generated header.
///
/// Implements the complete loop: load weights → tokenize → prefill → decode.
/// Handles both Mode A (.rax manifest) and Mode B (explicit paths).
class DecoderRunner : public InferenceRunner {
public:
  DecoderRunner();
  ~DecoderRunner() override;
  void run(const RunConfig &cfg) override;
  bool runBench(const RunConfig &cfg, int warmupReps, int reps,
                std::vector<BenchmarkResult> &out) override;

private:
  // Session used by runBench(), kept across calls so a multi-configuration
  // benchmark loads the weights once. Recreated when the model files change.
  std::unique_ptr<ModelSession> benchSession_;
  std::string benchSessionKey_;
  std::vector<std::string> benchWeightPaths_;
  std::string benchVocabPath_;
};

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DECODERRUNNER_H
