//===- Fields.cpp - Named benchmark fields and their derivation -----------===//
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

#include "Fields.h"

#include <utility>

namespace buddy {
namespace bench {

Field Throughput(std::string name, SampleGetter count, SampleGetter secs) {
  return Field{std::move(name),
               [count = std::move(count), secs = std::move(secs)](
                   const buddy::runtime::BenchmarkResult &s) {
                 const double t = secs(s);
                 return t > 0.0 ? count(s) / t : 0.0;
               }};
}

Field Latency(std::string name, SampleGetter secs, double scale) {
  return Field{std::move(name), [secs = std::move(secs), scale](
                                    const buddy::runtime::BenchmarkResult &s) {
                 return secs(s) * scale;
               }};
}

Field Count(std::string name, SampleGetter count) {
  return Field{std::move(name), std::move(count)};
}

} // namespace bench
} // namespace buddy
