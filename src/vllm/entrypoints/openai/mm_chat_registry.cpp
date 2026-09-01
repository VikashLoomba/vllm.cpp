// The per-architecture multimodal chat seam registry. See
// include/vllm/entrypoints/openai/mm_chat_registry.h for the upstream anchors
// and the design argument.
#include "vllm/entrypoints/openai/mm_chat_registry.h"

#include <algorithm>
#include <exception>
#include <ostream>
#include <stdexcept>
#include <utility>

#include "vllm/v1/engine/input_processor.h"  // InputValidationError -> HTTP 400

namespace vllm::entrypoints::openai {

namespace {

// Function-local static: constructed on the first RegisterMultiModalChat call,
// which is safely before any registrar's body runs. Same shape as
// model_registry.cpp:123-126.
std::vector<MultiModalChatRegistration>& RegistryStorage() {
  static std::vector<MultiModalChatRegistration> storage;
  return storage;
}

// Sorted once, on first query, after all static init. Registration arrival
// order across translation units is unspecified; lookup is by exact
// architecture and is therefore order-independent, so this only makes
// SupportedArchs() and the refusal message deterministic
// (model_registry.cpp:134-144).
const std::vector<MultiModalChatRegistration>& OrderedRegistry() {
  [[maybe_unused]] static const bool sorted = [] {
    std::vector<MultiModalChatRegistration>& storage = RegistryStorage();
    std::stable_sort(storage.begin(), storage.end(),
                     [](const MultiModalChatRegistration& a,
                        const MultiModalChatRegistration& b) {
                       return a.architecture < b.architecture;
                     });
    return true;
  }();
  return RegistryStorage();
}

std::string SupportedArchsList() {
  std::string out;
  for (const MultiModalChatRegistration& r : OrderedRegistry()) {
    if (!out.empty()) out += ", ";
    out += std::string(r.architecture);
  }
  return out.empty() ? std::string("(none)") : out;
}

}  // namespace

void RegisterMultiModalChat(const MultiModalChatRegistration& registration) {
  RegistryStorage().push_back(registration);
}

std::span<const MultiModalChatRegistration>
MultiModalChatRegistry::Registrations() {
  return OrderedRegistry();
}

std::vector<std::string_view> MultiModalChatRegistry::SupportedArchs() {
  std::vector<std::string_view> archs;
  for (const MultiModalChatRegistration& r : OrderedRegistry()) {
    archs.push_back(r.architecture);
  }
  return archs;
}

const MultiModalChatRegistration* MultiModalChatRegistry::Find(
    std::string_view architecture) {
  const std::vector<MultiModalChatRegistration>& registry = OrderedRegistry();
  const auto it =
      std::find_if(registry.begin(), registry.end(),
                   [&](const MultiModalChatRegistration& r) {
                     return r.architecture == architecture;
                   });
  return it == registry.end() ? nullptr : &*it;
}

void MultiModalChatRegistry::RaiseForUnregistered(
    std::string_view architecture) {
  // Mirrors registry.py:182-185 — name the model and the missing part, never
  // substitute another architecture's factory. The supported list is appended
  // because the realistic reader is porting the second multimodal model and
  // needs to know what a registration looks like.
  throw std::runtime_error(
      "architecture '" + std::string(architecture) +
      "' declares multimodal support but no multimodal CHAT seam is registered "
      "for it; its processor and chat function are the missing part. Register "
      "one with REGISTER_VLLM_MM_CHAT from that architecture's own translation "
      "unit (include/vllm/entrypoints/openai/mm_chat_registry.h). Registered: " +
      SupportedArchsList() + ". ENG-MM-INPUT-PIPELINE (#2475).");
}

MultiModalChatSeam MultiModalChatRegistry::MakeSeam(
    const MultiModalChatContext& ctx) {
  const MultiModalChatRegistration* reg = Find(ctx.architecture);
  if (reg == nullptr || reg->make_seam == nullptr) {
    RaiseForUnregistered(ctx.architecture);
  }
  return reg->make_seam(ctx);
}

MultiModalChatFn MakeRefusingMultiModalChatFn(std::string architecture,
                                              std::string reason) {
  // Composed ONCE, not per request. The reason arrives from a thrown
  // `what()` and usually ends in a full stop of its own, so trim before
  // joining rather than emit "..".
  while (!reason.empty() && (reason.back() == '.' || reason.back() == ' ')) {
    reason.pop_back();
  }
  const std::string message =
      "multimodal input is not available for architecture '" + architecture +
      "': " + reason +
      ". The request was REFUSED rather than answered from the text path, "
      "because an image request answered as text looks like a working server.";
  return [message](const std::vector<ChatMessage>& messages)
             -> std::optional<multimodal::MultiModalInputs> {
    // Defensive: serving_chat.cpp:699-710 only calls the seam when a message
    // carries a multimodal part, but the seam's own contract is "nullopt when
    // nothing does", and a refusal that fired on a text request would take the
    // text path down with it.
    const bool has_mm =
        std::any_of(messages.begin(), messages.end(),
                    [](const ChatMessage& m) { return HasMultiModalParts(m); });
    if (!has_mm) return std::nullopt;
    throw vllm::v1::InputValidationError(message);
  };
}

MultiModalChatInstall InstallMultiModalChatSeam(
    OpenAIServingChat& chat, bool is_multimodal_model,
    const MultiModalChatContext& ctx, std::ostream& log) {
  // registry.py:109-110: a model that is not multimodal has no processor and
  // needs none. Nothing is installed, `mm_chat_fn_` stays null, and every chat
  // request takes the byte-identical text path. No diagnostic, because this is
  // the ordinary text-only server and not a failure.
  if (!is_multimodal_model) return MultiModalChatInstall::kTextOnlyModel;

  MultiModalChatSeam seam;
  try {
    seam = MultiModalChatRegistry::MakeSeam(ctx);
  } catch (const std::exception& e) {
    // Both failure modes land here: nothing registered for this architecture
    // (RaiseForUnregistered) and a registered factory that could not build —
    // a GGUF checkpoint with no `preprocessor_config.json` beside it, say.
    //
    // The previous code printed one line and left the seam UNSET, so an image
    // request was answered as text. It now installs a seam that REFUSES, which
    // keeps the text path serving (the arm a startup abort would have taken
    // away, and the arm a staged model port needs) while making a silent wrong
    // answer impossible: the only chat function present throws.
    log << "server: multimodal chat seam UNAVAILABLE for architecture '"
        << ctx.architecture << "' (" << e.what()
        << "); multimodal requests are REFUSED with HTTP 400 and the text path "
           "is unaffected\n";
    chat.set_multimodal_chat_fn(
        MakeRefusingMultiModalChatFn(std::string(ctx.architecture), e.what()));
    return MultiModalChatInstall::kRefusing;
  }

  chat.set_multimodal_chat_fn(seam.chat_fn);
  for (const auto& [modality, limit] : seam.allowed_limits) {
    log << "server: multimodal limit " << modality << "=" << limit
        << " (over the request limit for this seam)\n";
  }
  log << "server: multimodal chat seam wired for architecture '"
      << ctx.architecture << "' (" << seam.detail << ")\n";
  return MultiModalChatInstall::kInstalled;
}

}  // namespace vllm::entrypoints::openai
