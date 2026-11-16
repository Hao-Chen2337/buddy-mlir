//===- BenchUtils.cpp - Config parsing and aggregation for buddy-bench ----===//
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

#include "BenchUtils.h"
#include "Fields.h"
#include "Stats.h"

#include <cstdint>
#include <stdexcept>

namespace buddy {
namespace bench {

namespace {

int parseNonNegativeInt(const std::string &text, const std::string &spec) {
  int value = 0;
  try {
    size_t consumed = 0;
    value = std::stoi(text, &consumed);
    if (consumed != text.size())
      throw std::invalid_argument("trailing characters");
  } catch (const std::exception &) {
    throw CLI::ValidationError("invalid integer '" + text + "' in " + spec);
  }
  if (value < 0)
    throw CLI::ValidationError("negative value " + text + " in " + spec);
  return value;
}

} // namespace

std::vector<int> parseIntRange(const std::string &spec) {
  std::vector<int> out;
  const auto dotdot = spec.find("..");
  if (dotdot == std::string::npos) {
    out.push_back(parseNonNegativeInt(spec, spec));
    return out;
  }
  const int lo = parseNonNegativeInt(spec.substr(0, dotdot), spec);
  const std::string rest = spec.substr(dotdot + 2);
  int hi = 0;
  int step = 1;
  const auto colon = rest.find(':');
  if (colon != std::string::npos) {
    hi = parseNonNegativeInt(rest.substr(0, colon), spec);
    step = parseNonNegativeInt(rest.substr(colon + 1), spec);
  } else {
    hi = parseNonNegativeInt(rest, spec);
  }
  if (step <= 0)
    throw CLI::ValidationError("range step must be positive: " + spec);
  if (lo > hi)
    throw CLI::ValidationError("range start exceeds end: " + spec);
  // Cap the expansion so a stray wide range fails instead of bad_alloc.
  constexpr int64_t kMaxRangeElements = 1 << 20;
  const int64_t count = (int64_t{hi} - lo) / step + 1;
  if (count > kMaxRangeElements)
    throw CLI::ValidationError("range " + spec + " expands to " +
                               std::to_string(count) + " values (limit " +
                               std::to_string(kMaxRangeElements) + ")");
  out.reserve(out.size() + static_cast<size_t>(count));
  for (int64_t v = lo; v <= hi; v += step)
    out.push_back(static_cast<int>(v));
  return out;
}

std::vector<int> expandIntList(const std::vector<std::string> &specs) {
  std::vector<int> out;
  for (const auto &s : specs)
    for (int v : parseIntRange(s))
      out.push_back(v);
  return out;
}

std::vector<BenchConfig>
buildConfigs(bool forwardMode, const std::vector<std::string> &nGenSpecs,
             const std::vector<std::string> &nPromptSpecs,
             const std::vector<std::string> &pgPairs) {
  std::vector<BenchConfig> configs;
  if (forwardMode) {
    // Forward: one pass per -p length (default: --prompt text); -n/-pg ignored.
    const std::vector<int> promptLens = nPromptSpecs.empty()
                                            ? std::vector<int>{0}
                                            : expandIntList(nPromptSpecs);
    for (int p : promptLens)
      configs.push_back({p, 0});
  } else {
    // -p, -n, --pg are independent selectors; concatenated; defaults 512/128.
    const std::vector<int> nPrompt = expandIntList(
        nPromptSpecs.empty() ? std::vector<std::string>{"512"} : nPromptSpecs);
    const std::vector<int> nGen = expandIntList(
        nGenSpecs.empty() ? std::vector<std::string>{"128"} : nGenSpecs);

    // pp: prompt processing only; -p 0 is skipped (llama-bench does too).
    for (int p : nPrompt)
      if (p > 0)
        configs.push_back({p, 0});
    // tg: decode only, with no prompt.
    for (int n : nGen)
      if (n > 0)
        configs.push_back({-1, n});

    // pg: explicit p,n pairs (CLI11 flattens them); (0,0) selects none.
    if (pgPairs.size() % 2 != 0)
      throw CLI::ValidationError(
          "-pg expects p,n pairs (even number of values), got " +
          std::to_string(pgPairs.size()) + " values");
    for (size_t i = 0; i < pgPairs.size(); i += 2) {
      const int p = parseNonNegativeInt(pgPairs[i], "-pg");
      const int n = parseNonNegativeInt(pgPairs[i + 1], "-pg");
      if (p == 0 && n == 0)
        continue;
      configs.push_back({p == 0 ? -1 : p, n});
    }
  }

  if (configs.empty())
    throw CLI::ValidationError("no benchmark configuration selected");
  return configs;
}

namespace {

/// The decoder field table: every numeric column a decoder model reports, in
/// report order. Each entry derives one number from one raw sample; `summarize`
/// aggregates the entries across reps. This keeps the LLM numerics identical to
/// the previous hardcoded columns while making them data-driven.
std::vector<Field> decoderFields(bool decodeOnly) {
  SampleGetter prefillTokens = [](const buddy::runtime::BenchmarkResult &s) {
    return s.prefillWorkLen > 0 ? static_cast<double>(s.prefillWorkLen)
                                : static_cast<double>(s.nPrompt);
  };
  SampleGetter genTokens = [](const buddy::runtime::BenchmarkResult &s) {
    return static_cast<double>(s.nGen);
  };
  SampleGetter prefillSecs = [](const buddy::runtime::BenchmarkResult &s) {
    return s.prefillSecs;
  };
  SampleGetter decodeSecs = [](const buddy::runtime::BenchmarkResult &s) {
    return s.decodeSecs;
  };
  SampleGetter totalSecs = [](const buddy::runtime::BenchmarkResult &s) {
    return s.prefillSecs + s.decodeSecs;
  };

  std::vector<Field> fields;
  if (decodeOnly)
    fields.push_back(
        Field{"prefill_tps",
              [](const buddy::runtime::BenchmarkResult &) { return 0.0; }});
  else
    fields.push_back(Throughput("prefill_tps", prefillTokens, prefillSecs));
  fields.push_back(Throughput("decode_tps", genTokens, decodeSecs));
  fields.push_back(Latency("avg_s", decodeOnly ? decodeSecs : totalSecs));
  return fields;
}

} // namespace

BenchmarkRecord
summarize(const std::string &model,
          const std::vector<buddy::runtime::BenchmarkResult> &samples,
          bool forwardMode) {
  const int nPrompt = samples.empty() ? 0 : samples.front().nPrompt;
  const int nGen = samples.empty() ? 0 : samples.front().nGen;

  // tg times decode alone: the seed prefill is setup, so report no prefill tps.
  const bool decodeOnly = !forwardMode && nPrompt == 0;

  BenchmarkRecord rec;
  rec.model = model;
  rec.nPrompt = nPrompt;
  rec.nGen = nGen;
  rec.reps = static_cast<int>(samples.size());

  // Derive and aggregate the decoder field table, then any model-specific
  // extra metrics; both become first-class named metrics in report order.
  for (const Field &f : decoderFields(decodeOnly)) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto &s : samples)
      values.push_back(f.getter(s));
    rec.metrics.push_back(Metric{f.name, avg(values), stdev(values)});
  }
  if (!samples.empty()) {
    for (const auto &first : samples.front().extraMetrics) {
      std::vector<double> values;
      values.reserve(samples.size());
      for (const auto &s : samples)
        for (const auto &m : s.extraMetrics)
          if (m.first == first.first)
            values.push_back(m.second);
      if (values.empty())
        continue;
      rec.metrics.push_back(Metric{first.first, avg(values), stdev(values)});
    }
  }

  if (forwardMode)
    rec.test = "forward";
  else if (nGen == 0)
    rec.test = "pp";
  else if (nPrompt == 0)
    rec.test = "tg";
  else
    rec.test = "pg";
  return rec;
}

} // namespace bench
} // namespace buddy
