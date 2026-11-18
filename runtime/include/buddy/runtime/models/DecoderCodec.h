//===- DecoderCodec.h - tokenizer dispatch for decoder models -------------===//
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
// Maps the compile-time BUDDY_DECODER_TOKENIZER tag (see
// gen_config._TOKENIZER_METHOD) to the right TextContainer tokenize/revert
// method. Kept in a header so the Runner and the ResidentModel share one
// dispatch table.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_MODELS_DECODERCODEC_H
#define BUDDY_RUNTIME_MODELS_DECODERCODEC_H

#include "buddy/LLM/TextContainer.h"
#include "buddy/runtime/llm/TextGeneration.h"
#include "buddy/runtime/models/ModelSession.h"

#include <string_view>

namespace buddy {
namespace runtime {

namespace {

// Tokenizer method family tag; qwen25/qwen3 share the Qwen3 codec.
constexpr std::string_view kDecoderTokenizer = BUDDY_DECODER_TOKENIZER;

inline void tokenizeDecoder(Text<size_t, 2> &t, const std::string &vocab) {
  if (kDecoderTokenizer == "deepseek")
    t.tokenizeDeepSeekR1(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else if (kDecoderTokenizer == "llama")
    t.tokenizeLlama(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else if (kDecoderTokenizer == "gemma")
    t.tokenizeGemma4(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else if (kDecoderTokenizer == "mistral")
    t.tokenizeMistral(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else if (kDecoderTokenizer == "phi")
    t.tokenizePhi(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else if (kDecoderTokenizer == "gpt2")
    t.tokenizeGpt2(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
  else
    t.tokenizeQwen3(vocab, BUDDY_DECODER_MAX_TOKEN_LEN);
}

inline std::string detokenizeDecoder(Text<size_t, 2> &t) {
  if (kDecoderTokenizer == "deepseek")
    return t.revertDeepSeekR1();
  if (kDecoderTokenizer == "llama")
    return t.revertLlama();
  if (kDecoderTokenizer == "gemma")
    return t.revertGemma4();
  if (kDecoderTokenizer == "mistral")
    return t.revertMistral();
  if (kDecoderTokenizer == "phi")
    return t.revertPhi();
  if (kDecoderTokenizer == "gpt2")
    return t.revertGpt2();
  return t.revertQwen3();
}

inline TextCodec makeDecoderCodec() {
  TextCodec codec;
  codec.tokenize = [](Text<size_t, 2> &t, const std::string &vocab) {
    tokenizeDecoder(t, vocab);
  };
  codec.detokenize = [](Text<size_t, 2> &t) { return detokenizeDecoder(t); };
  codec.maxTokenLen = BUDDY_DECODER_MAX_TOKEN_LEN;
  return codec;
}

} // namespace

} // namespace runtime
} // namespace buddy

#endif // BUDDY_RUNTIME_MODELS_DECODERCODEC_H
