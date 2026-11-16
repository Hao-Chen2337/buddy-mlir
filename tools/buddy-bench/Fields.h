//===- Fields.h - Named benchmark fields and their derivation -------------===//
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
// A benchmark reports a set of named numeric fields. Each field is a pure
// per-sample getter: it reads one BenchmarkResult and returns one number.
// Fields are aggregated (mean, sample stddev) across the repetitions of one
// configuration.
//
// There is no expression engine. Every field is either one of the three
// patterns below (throughput / latency / count) or a hand-written getter for a
// model-specific metric; the number of fields is whatever the model's field
// table lists, so the report adapts to the model instead of hardcoding columns.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_BENCH_FIELDS_H
#define BUDDY_BENCH_FIELDS_H

#include "buddy/runtime/core/InferenceRunner.h"

#include <functional>
#include <string>

namespace buddy {
namespace bench {

/// Maps one raw benchmark sample to a single number.
using SampleGetter =
    std::function<double(const buddy::runtime::BenchmarkResult &)>;

/// A named field: `getter` is evaluated per sample, then aggregated.
struct Field {
  std::string name;
  SampleGetter getter;
};

/// Throughput: `count` / `secs` per sample (0 when secs <= 0).
Field Throughput(std::string name, SampleGetter count, SampleGetter secs);

/// Latency: `secs` scaled by `scale` (default 1) per sample.
Field Latency(std::string name, SampleGetter secs, double scale = 1.0);

/// Count: a per-sample quantity promoted to a metric.
Field Count(std::string name, SampleGetter count);

} // namespace bench
} // namespace buddy

#endif // BUDDY_BENCH_FIELDS_H
