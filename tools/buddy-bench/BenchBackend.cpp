//===- BenchBackend.cpp - Unified facade for buddy-bench ------------------===//
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

#include "BenchBackend.h"
#include "Affinity.h"

#include "buddy/runtime/core/ModelManifest.h"

#include <cstdio>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <thread>

namespace buddy {
namespace bench {

namespace {

std::string resolvePathRelativeToRax(const std::string &path,
                                     const std::string &raxPath) {
  namespace fs = std::filesystem;
  if (path.empty())
    return "";
  fs::path p(path);
  if (p.is_absolute() || raxPath.empty())
    return p.string();
  return (fs::absolute(fs::path(raxPath)).parent_path() / p).string();
}

} // namespace

BenchBackend::~BenchBackend() {
  if (runner_ && destroy_)
    destroy_(runner_);
  if (runnerHandle_)
    dlclose(runnerHandle_);
}

bool BenchBackend::load(const std::string &raxPath) {
  buddy::runtime::ModelManifest manifest;
  try {
    manifest = buddy::runtime::ModelManifest::loadFromRax(raxPath);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "buddy-bench: reading manifest: %s\n", e.what());
    return false;
  }

  modelName_ = manifest.modelName;
  if (modelName_.empty())
    modelName_ = std::filesystem::path(raxPath).stem().string();

  // Record which package produced the results (path, size, mtime).
  raxPath_ = raxPath;
  struct stat st;
  if (::stat(raxPath.c_str(), &st) == 0) {
    raxSizeBytes_ = static_cast<long long>(st.st_size);
    std::tm tm{};
    if (::gmtime_r(&st.st_mtime, &tm)) {
      char buf[32];
      if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm))
        raxMtime_ = buf;
    }
  }

  precision_ = manifest.weightDtype;
  if (precision_.empty() && !manifest.weightPaths.empty()) {
    std::string base =
        std::filesystem::path(manifest.weightPaths[0]).filename().string();
    const auto dot = base.rfind('.');
    if (dot != std::string::npos)
      base = base.substr(0, dot);
    const auto dash = base.rfind('-');
    if (dash != std::string::npos)
      base = base.substr(dash + 1);
    precision_ = base;
  }
  if (precision_.empty())
    precision_ = "unknown";

  // Execution backend from the first code object (e.g. "cpu", "rvv").
  for (const auto &co : manifest.codeObjects) {
    if (!co.backend.empty()) {
      backend_ = co.backend;
      break;
    }
  }
  if (backend_.empty())
    backend_ = "cpu";
  device_ = backend_;

  // Keep RAX dependencies loaded while the runner is alive.
  if (!manifest.dependentSoPaths.empty()) {
    try {
      raxDeps_ = std::make_unique<buddy::runtime::RaxDependenciesHandle>(
          manifest.dependentSoPaths, "buddy-bench");
    } catch (const std::exception &e) {
      std::fprintf(stderr, "buddy-bench: loading RAX dependencies: %s\n",
                   e.what());
      return false;
    }
  }

  std::string runnerPath =
      resolvePathRelativeToRax(manifest.runnerLibraryPath, raxPath);
  if (runnerPath.empty()) {
    std::fprintf(stderr,
                 "buddy-bench: runner plugin not specified in manifest\n");
    return false;
  }

  runnerHandle_ = dlopen(runnerPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!runnerHandle_) {
    std::fprintf(stderr, "buddy-bench: dlopen runner failed: %s: %s\n",
                 runnerPath.c_str(), dlerror());
    return false;
  }

  dlerror();
  auto *capsSym =
      dlsym(runnerHandle_, "buddy_inference_runner_capabilities_v1");
  const char *capsErr = dlerror();
  if (capsErr || !capsSym) {
    std::fprintf(stderr,
                 "buddy-bench: runner plugin predates buddy-bench (missing "
                 "buddy_inference_runner_capabilities_v1); rebuild and repack "
                 "the .rax with the current toolchain\n");
    return false;
  }
  capabilities_ =
      reinterpret_cast<buddy::runtime::InferenceRunnerCapabilitiesFn>(
          capsSym)();

  dlerror();
  auto *createSym = dlsym(runnerHandle_, "buddy_create_inference_runner_v1");
  const char *createErr = dlerror();
  if (createErr) {
    std::fprintf(stderr,
                 "buddy-bench: runner missing "
                 "buddy_create_inference_runner_v1: %s\n",
                 createErr);
    return false;
  }

  dlerror();
  auto *destroySym = dlsym(runnerHandle_, "buddy_destroy_inference_runner_v1");
  const char *destroyErr = dlerror();
  if (destroyErr) {
    std::fprintf(stderr,
                 "buddy-bench: runner missing "
                 "buddy_destroy_inference_runner_v1: %s\n",
                 destroyErr);
    return false;
  }

  auto create =
      reinterpret_cast<buddy::runtime::CreateInferenceRunnerFn>(createSym);
  destroy_ =
      reinterpret_cast<buddy::runtime::DestroyInferenceRunnerFn>(destroySym);
  runner_ = create();
  if (!runner_) {
    std::fprintf(stderr, "buddy-bench: runner plugin returned null for %s\n",
                 modelName_.c_str());
    return false;
  }

  return true;
}

const std::string &BenchBackend::modelName() const { return modelName_; }

bool BenchBackend::runBench(const buddy::runtime::RunConfig &cfg,
                            int warmupReps, int reps,
                            std::vector<buddy::runtime::BenchmarkResult> &out) {
  if (!runner_) {
    std::fprintf(stderr, "buddy-bench: model not loaded\n");
    return false;
  }
  if (!supportsBench()) {
    std::fprintf(stderr,
                 "buddy-bench: model %s does not implement runBench()\n",
                 modelName_.c_str());
    return false;
  }
  return runner_->runBench(cfg, warmupReps, reps, out);
}

EnvInfo BenchBackend::getEnvInfo() const {
  EnvInfo env;

  // ── Shared table: host + build attributes, identical for every model ──
  const int numCores = static_cast<int>(std::thread::hardware_concurrency());

  // Effective CPU affinity and the number of cores it leaves available.
  const std::string affinity = effectiveCpuAffinity();
  int usedCores = effectiveCpuCount();
  if (usedCores == 0)
    usedCores = numCores; // affinity read failed; assume unrestricted

  // CPU model: first "model name" line of /proc/cpuinfo (Linux).
  std::string cpuModel;
  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
      const std::string key = "model name";
      if (line.compare(0, key.size(), key) != 0)
        continue;
      auto colon = line.find(':');
      if (colon == std::string::npos)
        break;
      std::string val = line.substr(colon + 1);
      auto begin = val.find_first_not_of(" \t");
      auto end = val.find_last_not_of(" \t");
      if (begin != std::string::npos)
        cpuModel = val.substr(begin, end - begin + 1);
      break;
    }
  }

  env.shared.push_back(EnvField::s("cpu_model", cpuModel));
  env.shared.push_back(EnvField::i("cores", numCores));
  env.shared.push_back(EnvField::i("used_cores", usedCores));
  env.shared.push_back(EnvField::s("cpu_affinity", affinity));
  env.shared.push_back(EnvField::s("numa", numaConfig()));
  env.shared.push_back(EnvField::s("isa", cpuIsa()));
#ifdef BUDDY_VERSION
  env.shared.push_back(EnvField::s("version", BUDDY_VERSION));
#else
  env.shared.push_back(EnvField::s("version", "unknown"));
#endif

  // ── Category table: package + backend attributes ──
  env.category.push_back(EnvField::s("model_path", raxPath_));
  env.category.push_back(EnvField::i("model_size", raxSizeBytes_));
  env.category.push_back(EnvField::s("model_mtime", raxMtime_));
  env.category.push_back(EnvField::s("precision", precision_));
  env.category.push_back(EnvField::s("backend", backend_));
  env.category.push_back(EnvField::s("device", device_));
  // A cpu-backend model runs on the host CPU; report its architecture.
  env.category.push_back(EnvField::s(
      "device_arch",
      (backend_ == "cpu" && !cpuModel.empty()) ? cpuModel : backend_));
  env.category.push_back(EnvField::i("device_count", 1));
  env.category.push_back(EnvField::i("batch_size", batchSize_));

  return env;
}

} // namespace bench
} // namespace buddy
