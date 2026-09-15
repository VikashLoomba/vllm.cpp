#pragma once

#include <string_view>

#include "vt/rocm/rocm_arch.h"

namespace vt::rocm {

// Attention admission is separate from quantized WMMA admission. Every admitted
// target must compile the attention body as well as pass this host predicate.
constexpr bool GcnArchNameHasSharedKAttentionWmma(std::string_view arch) {
  return arch == "gfx1100" || arch.starts_with("gfx1100:") ||
         GcnArchNameIsGfx12PrefillWmma(arch);
}

// gfx1100 remains opt-in until the broader Gemma token gate passes. Preserve
// the existing gfx12 default and the environment knob's first-character rule.
constexpr bool SharedKAttentionWmmaEnabled(std::string_view arch, const char* override_value) {
  return GcnArchNameHasSharedKAttentionWmma(arch) &&
         (override_value ? override_value[0] != '0' : GcnArchNameIsGfx12PrefillWmma(arch));
}

}  // namespace vt::rocm
