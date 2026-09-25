#!/usr/bin/env python3
"""Generate the portable Vulkan search shader from the certified CUDA logic."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_vulkan_timeline_shader import (  # noqa: E402
    CONCRETE_TYPE_PARAMETERS,
    DEVICE_FILES,
    PRELUDE,
    ReferenceTranslator,
    SEARCH_TEMPLATE_VALUES,
    translate_common,
)


SEARCH_FILES = DEVICE_FILES + (
    "cuda_modifier_event_ops.cuh",
    "cuda_candidate_events.cuh",
    "cuda_search_branch_state.cuh",
    "cuda_search_winner_selection.cuh",
    "cuda_search_executor.cu",
)


SEARCH_CONCRETE_TYPE_PARAMETERS = dict(CONCRETE_TYPE_PARAMETERS)
SEARCH_CONCRETE_TYPE_PARAMETERS["Scratch"] = (
    "cuda__collision__VulkanCollisionSearchScratch"
)
for parameter in ("Collision", "Left", "Right"):
    SEARCH_CONCRETE_TYPE_PARAMETERS[parameter] = (
        "cuda__collision__CudaCollisionSearchReference"
    )
SEARCH_TEMPLATE_VALUES = dict(SEARCH_TEMPLATE_VALUES)
SEARCH_TEMPLATE_VALUES.update({
    "TrackDiagnostics": False,
    "TrackCollisionDiagnostics": False,
    "TrustedInputs": True,
    "CompactReplacements": True,
    "EightOrderedEllipsoids": True,
    "WriteOutputSnapshots": False,
    "ReuseWheelPassInvariants": True,
    "UnifiedBounds": True,
    "UseMeshCellCache": True,
    # CUDA's throughput kernel uses the non-warp-coherent traversal. The tail
    # kernels use the alternate path, but Vulkan keeps one staged throughput
    # pipeline so pipeline creation and behavior stay stable across vendors.
    "WarpCoherentAcceleration": False,
})


def isolate_search_mesh_cell_cache(text: str) -> str:
    """Keep NVIDIA from inlining the large persistent mesh-cache builder."""
    text, count = re.subn(
        r"inline void BuildMeshCellCache<",
        "[noinline]\ninline void BuildMeshCellCache<",
        text,
    )
    if count != 1:
        raise ValueError(f"expected one mesh-cache builder, found {count}")
    return text


def between(text: str, begin: str, end: str) -> str:
    first = text.index(begin)
    last = text.index(end, first)
    return text[first:last]


def dereference_pointer_calls(text: str, name: str) -> str:
    """Dereference translated ref-return calls not rewritten by Clang edits."""
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:detail::)?{name}\s*\(")
    matches = list(pattern.finditer(text))
    for match in reversed(matches):
        if text[max(0, match.start() - 2):match.start()] == "(*":
            continue
        opening = text.find("(", match.start(), match.end())
        depth = 1
        cursor = opening + 1
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth:
            raise RuntimeError(f"unbalanced call to {name}")
        following = cursor
        while following < len(text) and text[following].isspace():
            following += 1
        line_start = text.rfind("\n", 0, match.start()) + 1
        if (following < len(text) and text[following] == "{" and
                "inline" in text[line_start:match.start()]):
            continue
        text = text[:match.start()] + "(*" + text[match.start():cursor] + ")" + text[cursor:]
    return text


def eliminate_local_address_aliases(text: str) -> str:
    """Expand local pointers to subobjects so Slang retains address spaces."""
    declaration = re.compile(
        r"(?m)^[ \t]*(?:const[ \t]+)?"
        r"[A-Za-z_]\w*(?:::\w+)*(?:<[^;\n]*>)?[ \t]*\*"
        r"(?P<name>\w+)[ \t]*=\s*&(?P<expr>[^;\n]+);",
    )
    cursor = 0
    replacements = 0
    while True:
        match = declaration.search(text, cursor)
        if match is None:
            break
        replacements += 1
        if replacements > 128:
            raise RuntimeError("local address-alias expansion did not converge")
        depth = text.count("{", 0, match.start()) - text.count(
            "}", 0, match.start()
        )
        scan_depth = depth
        scope_end = len(text)
        for index in range(match.end(), len(text)):
            if text[index] == "{":
                scan_depth += 1
            elif text[index] == "}":
                scan_depth -= 1
                if scan_depth < depth:
                    scope_end = index
                    break
        name = match.group("name")
        expression = re.sub(r"\s+", " ", match.group("expr").strip())
        body = text[match.end():scope_end]
        body = re.sub(
            rf"\(\s*\*\s*{re.escape(name)}\s*\)",
            f"({expression})",
            body,
        )
        body = re.sub(
            rf"\b{re.escape(name)}\s*->", f"({expression}).", body
        )
        body = re.sub(
            rf"\b{re.escape(name)}\s*\.", f"({expression}).", body
        )
        body = re.sub(
            rf"(?<![A-Za-z0-9_])\*\s*{re.escape(name)}\b",
            f"({expression})",
            body,
        )
        body = re.sub(
            rf"(?<![A-Za-z0-9_.:]){re.escape(name)}\b",
            f"&({expression})",
            body,
        )
        text = text[:match.start()] + body + text[scope_end:]
        cursor = match.start()
    text = re.sub(
        r"forevervalidator::simulation::cuda::facts::WheelCount\("
        r"&\(([^()]*)\)\)",
        r"(\1).wheels.count",
        text,
    )
    return text


def rewrite_search_collision_slang(text: str) -> str:
    text, count = re.subn(
        r"const CudaCandidatePhysicsState\s*&candidate,\s*"
        r"const GmIso4\s*&bodyPose",
        "__constref CudaCandidatePhysicsState candidate,\n"
        "        __constref GmIso4 bodyPose",
        text,
    )
    if count != 1:
        raise ValueError(
            f"expected one root-shape pose reference signature, found {count}"
        )
    for name in (
        "ShapeWorldAt", "MovingBoundsAt", "MeshRangeAt", "MeshCellAt"
    ):
        text = dereference_pointer_calls(text, name)
    text = re.sub(
        r"(CudaCollisionMeshRange \*range\s*=\s*)"
        r"\(\*MeshRangeAt\((.*?)\)\);",
        r"\1MeshRangeAt(\2);",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"inline GmBoxAligned UnifiedMovingBoundsAt\(.*?\n\}",
        "inline GmBoxAligned UnifiedMovingBoundsAt(\n"
        "        __constref cuda__collision__VulkanCollisionSearchScratch scratch) {\n"
        "    return scratch.unifiedMovingBoundsStorage[scratch.slot];\n}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"inline void StoreUnifiedMovingBounds\(.*?\n\}",
        "inline void StoreUnifiedMovingBounds(\n"
        "        inout cuda__collision__VulkanCollisionSearchScratch scratch,\n"
        "        __constref GmBoxAligned value) {\n"
        "    scratch.unifiedMovingBoundsStorage[scratch.slot] = value;\n}",
        text,
        count=1,
        flags=re.DOTALL,
    )

    text = remove_unused_full_scratch_overloads(text)
    return rewrite_search_collision_references(text)


def rewrite_search_collision_references(text: str) -> str:
    """Use value-like Slang property proxies for CUDA search references."""
    # The CUDA search layout exposes collision fields through reference
    # proxies into a candidate-interleaved structure-of-arrays tile. Slang
    # cannot represent C++ reference members, so use an equivalent property
    # proxy and remove the pointer syntax introduced by the AST translator.
    text = text.replace(
        "cuda__collision__CudaCollision *",
        "cuda__collision__CudaCollisionSearchReference ",
    )
    text = text.replace("(*collision).", "collision.")
    text = text.replace("(*primary).", "primary.")
    text = text.replace("(*destination).", "destination.")
    text = text.replace("(*contact).", "contact.")
    text = text.replace("(*collision)", "collision")
    text = text.replace("(*primary)", "primary")
    text = text.replace("(*destination)", "destination")
    text = text.replace("(*contact)", "contact")
    for name in ("CollisionAt", "ShapeCollisionAt", "OrderedCollisionAt"):
        text = undereference_proxy_calls(text, name)
    text = re.sub(
        r"\(\*detail::ReplacementOverflowAt\("
        r"scratch, scratch\.replacementOverflowCount\+\+\)\) = replacement;",
        "detail::StoreReplacementOverflowAt(\n"
        "            scratch, scratch.replacementOverflowCount++, replacement);",
        text,
    )
    return text


def remove_unused_full_scratch_overloads(text: str) -> str:
    """Drop AoS-only accessor overloads from the search specialization."""
    marker = "cuda__collision__CudaCollisionScratch"
    while True:
        parameter = text.find(marker)
        if parameter < 0:
            return text
        start = text.rfind(" inline ", 0, parameter)
        opening = text.find("{", parameter)
        if start < 0 or opening < 0:
            raise RuntimeError("could not isolate full collision scratch overload")
        depth = 1
        cursor = opening + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise RuntimeError("unbalanced full collision scratch overload")
        text = text[:start] + "\n" + text[cursor:]


def undereference_proxy_calls(text: str, name: str) -> str:
    """Turn ``(*Name(args))`` back into a value-like property proxy."""
    pattern = re.compile(rf"\(\*(?:(?:detail::)?{name})\s*\(")
    matches = list(pattern.finditer(text))
    for match in reversed(matches):
        call_open = text.find("(", match.start() + 2, match.end())
        depth = 1
        cursor = call_open + 1
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth or cursor >= len(text) or text[cursor] != ")":
            raise RuntimeError(f"malformed proxy dereference for {name}")
        text = text[:match.start()] + text[match.start() + 2:cursor] + text[cursor + 1:]
    return text


def rewrite_search_dynamics_slang(text: str) -> str:
    return re.sub(
        r"const GmVec3 replacement = \[&\]\(\) \{.*?\}\(\);",
        "const GmVec3 replacement = scratch.replacementOverflowCount == 0u\n"
        "            ? GmVec3() : FinalizeReplacement(\n"
        "                  scratch.replacementSumX,\n"
        "                  scratch.replacementSumY,\n"
        "                  scratch.replacementSumZ);",
        text,
        flags=re.DOTALL,
    )


def replace_specialization_fragment(
        text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"steady-velocity specialization {label} matched {count} times")
    return text.replace(old, new, 1)


def apply_unified_patch(text: str, patch_path: Path) -> str:
    """Apply a checked-in exact-context patch to generated shader text."""
    source = text.splitlines(keepends=True)
    patch = patch_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output: list[str] = []
    source_index = 0
    patch_index = 0
    hunk_header = re.compile(
        r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    while patch_index < len(patch):
        match = hunk_header.match(patch[patch_index])
        if match is None:
            patch_index += 1
            continue
        expected_index = int(match.group(1)) - 1
        patch_index += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        while patch_index < len(patch) and not patch[patch_index].startswith("@@ "):
            line = patch[patch_index]
            patch_index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in " +-":
                continue
            payload = line[1:]
            if line[0] in " -":
                old_lines.append(payload)
            if line[0] in " +":
                new_lines.append(payload)

        matches = [index for index in range(source_index,
                                            len(source) - len(old_lines) + 1)
                   if source[index:index + len(old_lines)] == old_lines]
        if not matches:
            raise RuntimeError(
                f"{patch_path.name} context not found near source line "
                f"{expected_index + 1}")
        hunk_source_index = min(matches,
                                key=lambda index: abs(index - expected_index))
        output.extend(source[source_index:hunk_source_index])
        output.extend(new_lines)
        source_index = hunk_source_index + len(old_lines)

    output.extend(source[source_index:])
    return "".join(output)


def specialize_steady_velocity(text: str, handling: str) -> str:
    fact_replacements = (
        (
            "uint WheelAxle(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return Wheel(configuration, index).axle;\n}",
            "uint WheelAxle(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return index < 2u ? VehicleWheelAxle_Front : VehicleWheelAxle_Rear;\n}",
            "wheel axle fact",
        ),
        (
            "bool WheelKillsLateralSpeed(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return Wheel(configuration, index).killsLateralSpeedOnContact != 0u;\n}",
            "bool WheelKillsLateralSpeed(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return true;\n}",
            "wheel lateral-speed fact",
        ),
        (
            "float WheelRollingRadius(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return Wheel(configuration, index).rollingRadius;\n}",
            "float WheelRollingRadius(CudaPackedStaticConfigurationHeader* configuration, uint index)\n"
            "{\n    return Wheel(configuration, 0u).rollingRadius;\n}",
            "wheel rolling-radius fact",
        ),
        (
            "uint WheelForceMode(CudaPackedStaticConfigurationHeader* configuration)\n"
            "{\n    return configuration->tuning.wheelForceMode;\n}",
            "uint WheelForceMode(CudaPackedStaticConfigurationHeader* configuration)\n"
            "{\n    return 2u;\n}",
            "wheel force-mode fact",
        ),
    )
    for old, new, label in fact_replacements:
        text = replace_specialization_fragment(text, old, new, label)
    # @@ -2270,11 +2270,11 @@
    old = '\n}  // namespace detail\n\n inline float FromDouble(double value) {\n    if (isnan(value)) {\n        return detail::QuietNaN(signbit_portable(value));\n    }\n    return float(value);\n}\n\n inline float FromUnsignedInteger(uint value) {\n'
    new = '\n}  // namespace detail\n\ninline float FromDouble(double value) {\n    const uint nanMask = isnan(value) ? 0xffffffffu : 0u;\n    const uint converted = asuint(float(value));\n    const uint nan = asuint(detail::QuietNaN(signbit_portable(value)));\n    return asfloat((converted & ~nanMask) | (nan & nanMask));\n}\n\n inline float FromUnsignedInteger(uint value) {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 0")
    # @@ -2282,7 +2282,9 @@
    old = '}\n\n inline float Divide(float numerator, float denominator) {\n    return FromDouble(double(numerator) / double(denominator));\n}\n\n inline uint TruncateToUint32Modulo(float value) {\n'
    new = '}\n\n inline float Divide(float numerator, float denominator) {\n    const float quotient = numerator / denominator;\n    const float residual = fma(-quotient, denominator, numerator);\n    return quotient + residual / denominator;\n}\n\n inline uint TruncateToUint32Modulo(float value) {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 1")
    # @@ -2298,10 +2300,7 @@
    old = '}\n\n inline float Sqrt(float value) {\n    if (value >= 0.0f) {\n        return FromDouble(sqrt(double(value)));\n    }\n    return FromDouble(sqrt(double(value)));\n}\n\n inline float Atan2(float y, float x) {\n'
    new = '}\n\n inline float Sqrt(float value) {\n    return sqrt(value);\n}\n\n inline float Atan2(float y, float x) {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 2")
    # @@ -3037,6 +3036,20 @@
    old = '    return result;\n}\n\n inline GmVec3 TransformDirection(\n        __constref GmMat3 matrix, __constref GmVec3 direction) {\n    return {\n'
    new = '    return result;\n}\n\n\n inline GmVec3 NormalizeEdgeBranchless(\n        __constref GmVec3 value, float epsilonSquared) {\n    const float lengthSquared = Dot(value, value);\n    const bool normalize = epsilonSquared < lengthSquared;\n    const float safeLengthSquared = normalize ? lengthSquared : 1.0f;\n    const float normalizedScale =\n            forevervalidator::simulation::cuda::exact::Divide(\n                    1.0f,\n                    forevervalidator::simulation::cuda::exact::Sqrt(\n                            safeLengthSquared));\n    const float scale = normalize ? normalizedScale : 1.0f;\n    return Scale(value, scale);\n}\n inline GmVec3 TransformDirection(\n        __constref GmMat3 matrix, __constref GmVec3 direction) {\n    return {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 3")
    # @@ -3390,37 +3403,35 @@
    old = '        const GmVec3 projected = Add(\n                center,\n                Scale(triangleNormal, -planeDistance));\n        for (uint edge = 0u; edge < 3u; ++edge) {\n            const uint next =\n                    edge == 2u ? 0u : edge + 1u;\n            const GmVec3 start = vertices[edge];\n            const GmVec3 end = vertices[next];\n            const GmVec3 direction = Normalize(\n                    Subtract(end, start),\n                    DirectionEpsilonSquared);\n            const GmVec3 edgeNormal =\n                    Cross(direction, triangleNormal);\n            const float edgeDistance = Dot(\n                    Subtract(projected, start), edgeNormal);\n            if (edgeReach < edgeDistance) return 0;\n            if (edgeDistance > 0.0f) {\n                const float fromStart = Dot(\n                        Subtract(projected, start), direction);\n                if (fromStart < 0.0f) {\n                    return EmitFeature(\n                            start, DirectionEpsilonSquared, true);\n                }\n                const float fromEnd = Dot(\n                        Subtract(projected, end), direction);\n                if (!(0.0f < fromEnd)) {\n                    return EmitFeature(\n                            Add(projected,\n                                Scale(edgeNormal, -edgeDistance)),\n                            CollisionDistance, false);\n                }\n                return EmitEndpointB(\n                        end, DirectionEpsilonSquared);\n            }\n        }\n        if (planeDistance > 0.0f) {\n            const uint collisionIndex = AddCollision();\n'
    new = '        const GmVec3 projected = Add(\n                center,\n                Scale(triangleNormal, -planeDistance));\n        const GmVec3 d0 = NormalizeEdgeBranchless(Subtract(vertices[1], vertices[0]), DirectionEpsilonSquared);\n        const GmVec3 d1 = NormalizeEdgeBranchless(Subtract(vertices[2], vertices[1]), DirectionEpsilonSquared);\n        const GmVec3 d2 = NormalizeEdgeBranchless(Subtract(vertices[0], vertices[2]), DirectionEpsilonSquared);\n        const float e0 = Dot(Subtract(projected, vertices[0]), Cross(d0, triangleNormal));\n        const float e1 = Dot(Subtract(projected, vertices[1]), Cross(d1, triangleNormal));\n        const float e2 = Dot(Subtract(projected, vertices[2]), Cross(d2, triangleNormal));\n        const uint r0 = edgeReach < e0 ? 1u : 0u;\n        const uint c0 = (1u - r0) * (e0 > 0.0f ? 1u : 0u);\n        const uint active1 = 1u - c0;\n        const uint r1 = active1 * (edgeReach < e1 ? 1u : 0u);\n        const uint c1 = active1 * (1u - r1) * (e1 > 0.0f ? 1u : 0u);\n        const uint active2 = active1 * (1u - r1) * (1u - c1);\n        const uint r2 = active2 * (edgeReach < e2 ? 1u : 0u);\n        const uint c2 = active2 * (1u - r2) * (e2 > 0.0f ? 1u : 0u);\n        if ((r0 | r1 | r2) != 0u) return 0;\n        const uint selected = c0 != 0u ? 0u : (c1 != 0u ? 1u : (c2 != 0u ? 2u : 3u));\n        if (selected < 3u) {\n            const uint next = selected == 2u ? 0u : selected + 1u;\n            const GmVec3 start = vertices[selected];\n            const GmVec3 end = vertices[next];\n            const GmVec3 direction = NormalizeEdgeBranchless(Subtract(end, start), DirectionEpsilonSquared);\n            const GmVec3 edgeNormal = Cross(direction, triangleNormal);\n            const float edgeDistance = Dot(Subtract(projected, start), edgeNormal);\n            const float fromStart = Dot(Subtract(projected, start), direction);\n            if (fromStart < 0.0f) return EmitFeature(start, DirectionEpsilonSquared, true);\n            const float fromEnd = Dot(Subtract(projected, end), direction);\n            if (!(0.0f < fromEnd))\n                return EmitFeature(Add(projected, Scale(edgeNormal, -edgeDistance)), CollisionDistance, false);\n            return EmitEndpointB(end, DirectionEpsilonSquared);\n        }\n        if (planeDistance > 0.0f) {\n            const uint collisionIndex = AddCollision();\n'
    text = replace_specialization_fragment(text, old, new, "fragment 4")
    # @@ -4209,37 +4220,33 @@
    old = '            shapeIndex, shapes, candidate, bodyPose);\n}\n\ninline int CompareForResponse(\n        __constref cuda__collision__CudaCollisionSearchReference left,\n        __constref cuda__collision__CudaCollisionSearchReference right) {\n    const float leftValues[] = {\n            left.contactPoint.x,\n            left.contactPoint.y,\n            left.contactPoint.z,\n            left.impulseNormal.x,\n            left.impulseNormal.y,\n            left.impulseNormal.z,\n            left.separation.x,\n            left.separation.y,\n            left.separation.z,\n    };\n    const float rightValues[] = {\n            right.contactPoint.x,\n            right.contactPoint.y,\n            right.contactPoint.z,\n            right.impulseNormal.x,\n            right.impulseNormal.y,\n            right.impulseNormal.z,\n            right.separation.x,\n            right.separation.y,\n            right.separation.z,\n    };\n    for (uint index = 0u; index < 9u; ++index) {\n        const float leftValue = leftValues[index];\n        const float rightValue = rightValues[index];\n        if (!(rightValue <= leftValue)) return 1;\n        if (rightValue < leftValue) return -1;\n    }\n    if (!left.sphereMergePrimary &&\n        right.sphereMergePrimary) {\n        return -1;\n'
    new = '            shapeIndex, shapes, candidate, bodyPose);\n}\n\ninline int CompareResponseFloat(float leftValue, float rightValue) {\n    if (!(rightValue <= leftValue)) return 1;\n    if (rightValue < leftValue) return -1;\n    return 0;\n}\n\ninline int CompareForResponse(\n        __constref cuda__collision__CudaCollisionSearchReference left,\n        __constref cuda__collision__CudaCollisionSearchReference right) {\n    int order = CompareResponseFloat(left.contactPoint.x, right.contactPoint.x);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.contactPoint.y, right.contactPoint.y);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.contactPoint.z, right.contactPoint.z);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.impulseNormal.x, right.impulseNormal.x);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.impulseNormal.y, right.impulseNormal.y);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.impulseNormal.z, right.impulseNormal.z);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.separation.x, right.separation.x);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.separation.y, right.separation.y);\n    if (order != 0) return order;\n    order = CompareResponseFloat(left.separation.z, right.separation.z);\n    if (order != 0) return order;\n    if (!left.sphereMergePrimary &&\n        right.sphereMergePrimary) {\n        return -1;\n'
    text = replace_specialization_fragment(text, old, new, "fragment 5")
    # @@ -4247,8 +4254,10 @@
    old = '    return 1;\n}\n\ninline void SortForResponse(\n        inout cuda__collision__VulkanCollisionSearchScratch scratch) {\n    static const uint Cutoff = 8u;\n    static const uint StackSize = 30u;\n    InitializeResponseOrder(scratch);\n'
    new = '    return 1;\n}\n\n[noRefInline]\n[noinline]\ninline void SortForResponse(\n        __ref cuda__collision__VulkanCollisionSearchScratch scratch) {\n    static const uint Cutoff = 8u;\n    static const uint StackSize = 30u;\n    InitializeResponseOrder(scratch);\n'
    text = replace_specialization_fragment(text, old, new, "fragment 6")
    # @@ -7762,15 +7771,15 @@
    old = '    values = {};\n    present = false;\n    uint count = 0u;\n    CudaVehicleState vehicle = candidate.vehicle;\n    for (uint index = 0u;\n         index < forevervalidator::simulation::cuda::facts::WheelCount(vehicle); ++index) {\n        if (!vehicle.wheels.values[index].realTime.contactPresent) {\n            continue;\n        }\n        const uint firstMaterial =\n                uint(\n                        vehicle.wheels.values[0u].\n                                realTime.contactMaterial);\n        VehicleMaterialDefinition *material =\n                Material(configuration, firstMaterial);\n'
    new = '    values = {};\n    present = false;\n    uint count = 0u;\n    for (uint index = 0u;\n         index < forevervalidator::simulation::cuda::facts::WheelCount(\n                 candidate.vehicle); ++index) {\n        if (!candidate.vehicle.wheels.values[index].realTime.contactPresent) {\n            continue;\n        }\n        const uint firstMaterial =\n                uint(\n                        candidate.vehicle.wheels.values[0u].\n                                realTime.contactMaterial);\n        VehicleMaterialDefinition *material =\n                Material(configuration, firstMaterial);\n'
    text = replace_specialization_fragment(text, old, new, "fragment 7")
    # @@ -9030,17 +9039,20 @@
    old = '    GmVec3 normalSum = GmVec3();\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        CudaWheelState wheel = (candidate.vehicle).wheels.values[index];\n        if (!wheel.realTime.contactPresent) continue;\n        normalSum.x =\n                normalSum.x +\n                wheel.realTime.accumulatedContactNormal.x;\n        normalSum.y =\n                wheel.realTime.accumulatedContactNormal.y +\n                normalSum.y;\n        normalSum.z =\n                normalSum.z +\n                wheel.realTime.accumulatedContactNormal.z;\n    }\n    const float normalLengthSquared =\n            (normalSum.x * normalSum.x +\n'
    new = '    GmVec3 normalSum = GmVec3();\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        if (!(candidate.vehicle).wheels.values[index].realTime.contactPresent)\n            continue;\n        normalSum.x =\n                normalSum.x +\n                (candidate.vehicle).wheels.values[index].realTime.\n                    accumulatedContactNormal.x;\n        normalSum.y =\n                (candidate.vehicle).wheels.values[index].realTime.\n                    accumulatedContactNormal.y +\n                normalSum.y;\n        normalSum.z =\n                normalSum.z +\n                (candidate.vehicle).wheels.values[index].realTime.\n                    accumulatedContactNormal.z;\n    }\n    const float normalLengthSquared =\n            (normalSum.x * normalSum.x +\n'
    text = replace_specialization_fragment(text, old, new, "fragment 8")
    # @@ -9457,9 +9469,8 @@
    old = '    };\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        CudaWheelState wheel =\n                (candidate.vehicle).wheels.values[index];\n        if (!wheel.realTime.slipping) continue;\n        if (index <= 1u) {\n            AddForceAtPoint(\n                    candidate, front,\n'
    new = '    };\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        if (!(candidate.vehicle).wheels.values[index].realTime.slipping)\n            continue;\n        if (index <= 1u) {\n            AddForceAtPoint(\n                    candidate, front,\n'
    text = replace_specialization_fragment(text, old, new, "fragment 9")
    # @@ -10199,9 +10210,7 @@
    old = '    bool anyContact = false;\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        CudaWheelState wheel =\n                (candidate.vehicle).wheels.values[index];\n        if (wheel.realTime.contactPresent) {\n            anyContact = true;\n            sideKill |= forevervalidator::simulation::cuda::facts::WheelKillsLateralSpeed(\n                    configuration, index);\n'
    new = '    bool anyContact = false;\n    for (uint index = 0u;\n         index < (candidate.vehicle).wheels.count; ++index) {\n        if ((candidate.vehicle).wheels.values[index].realTime.contactPresent) {\n            anyContact = true;\n            sideKill |= forevervalidator::simulation::cuda::facts::WheelKillsLateralSpeed(\n                    configuration, index);\n'
    text = replace_specialization_fragment(text, old, new, "fragment 10")
    # @@ -14719,6 +14728,35 @@
    old = '    return result;\n}\n\n  bool MaximizesScore(\n        uint kind) {\n    return kind == CudaSearchEvaluatorKind_Velocity ||\n'
    new = '    return result;\n}\n\ncuda_search_detail__DeviceSample EvaluateVelocityState(\n        __constref CudaCandidatePhysicsState state,\n        double currentTimeMs) {\n    cuda_search_detail__DeviceSample result;\n    result.score = 0.0;\n    result.timeMs = 0.0;\n    result.detail0 = 0.0;\n    result.detail1 = 0.0;\n    result.candidateId = 0u;\n    result.logicalOrder = 0xffffffffffffffffull;\n    result.candidateSlot = 0xffffffffu;\n    result.evaluationTick = 0u;\n    result.eventCount = 0xffffffffu;\n    result.valid = false;\n    result.mutation = false;\n    result.preciseFinish = false;\n    result.timeMs = currentTimeMs;\n    GmVec3 velocity = state.body.current.linearSpeed;\n    const double x = velocity.x;\n    const double y = velocity.y;\n    const double z = velocity.z;\n    const double speed = sqrt((x * x + y * y) + z * z);\n    result.score = speed;\n    result.detail0 = speed;\n    result.detail1 = 1.0;\n    result.valid = true;\n    return result;\n}\n\n  bool MaximizesScore(\n        uint kind) {\n    return kind == CudaSearchEvaluatorKind_Velocity ||\n'
    text = replace_specialization_fragment(text, old, new, "fragment 11")
    # @@ -15258,7 +15296,7 @@
    old = '    if (!state.vehicle.mobil.physicsUpdatesEnabled)\n        return forevervalidator::simulation::cuda::vehicle::ForceStatus::Success;\n    return forevervalidator::simulation::cuda::vehicle::ComputeForcesModel6<\n        true, CudaHandlingSpecialization_Generic>(\n            state, configuration, dt);\n}\n\n'
    handling_name = {
        "water": "CudaHandlingSpecialization_GearedDriveWater",
    }[handling]
    new = '    if (!state.vehicle.mobil.physicsUpdatesEnabled)\n        return forevervalidator::simulation::cuda::vehicle::ForceStatus::Success;\n    return forevervalidator::simulation::cuda::vehicle::ComputeForcesModel6<\n        true, ' + handling_name + '>(\n            state, configuration, dt);\n}\n\n'
    text = replace_specialization_fragment(text, old, new, "fragment 12")
    # @@ -15289,12 +15327,13 @@
    old = '            scene, configuration, state, *scratch);\n}\n\n[noinline]\nforevervalidator::simulation::cuda::physics::Status\nExecuteSearchCollisionSubstep(\n    CudaPackedSceneHeader* scene,\n    CudaPackedStaticConfigurationHeader* configuration,\n    inout CudaCandidatePhysicsState state,\n    cuda__collision__VulkanCollisionSearchScratch* scratch,\n    float dt)\n{\n'
    new = '            scene, configuration, state, *scratch);\n}\n\n[noRefInline]\n[noinline]\nforevervalidator::simulation::cuda::physics::Status\nExecuteSearchCollisionSubstep(\n    CudaPackedSceneHeader* scene,\n    CudaPackedStaticConfigurationHeader* configuration,\n    __ref CudaCandidatePhysicsState state,\n    cuda__collision__VulkanCollisionSearchScratch* scratch,\n    float dt)\n{\n'
    text = replace_specialization_fragment(text, old, new, "fragment 13")
    # @@ -15538,7 +15577,7 @@
    old = '\n    CudaSearchEvaluatorConfiguration* evaluator =\n        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);\n    bool maximize = MaximizesScore(evaluator->kind);\n    cuda_search_detail__DeviceSample incumbent = samples[0];\n    if (parameters->baseline != 0u)\n    {\n'
    new = '\n    CudaSearchEvaluatorConfiguration* evaluator =\n        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);\n    bool maximize = true;\n    cuda_search_detail__DeviceSample incumbent = samples[0];\n    if (parameters->baseline != 0u)\n    {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 14")
    # @@ -15661,7 +15700,7 @@
    old = '    uint8_t* evaluatorReportedValues =\n        reinterpret<uint8_t*>(parameters->evaluatorReported);\n    bool evaluatorReported = evaluatorReportedValues[slot] != 0u;\n    bool maximize = MaximizesScore(evaluator->kind);\n\n    uint tickEnd = min(\n        parameters->timelineTickCount,\n'
    new = '    uint8_t* evaluatorReportedValues =\n        reinterpret<uint8_t*>(parameters->evaluatorReported);\n    bool evaluatorReported = evaluatorReportedValues[slot] != 0u;\n    bool maximize = true;\n\n    uint tickEnd = min(\n        parameters->timelineTickCount,\n'
    text = replace_specialization_fragment(text, old, new, "fragment 15")
    # @@ -15702,29 +15741,10 @@
    old = '        forevervalidator::simulation::ApplyControlAndTimingPrefix(\n            localState, tick, false);\n        if (localState.firstStep == 0u)\n            forevervalidator::simulation::cuda::transition::PrepareStep(\n                localState, tick, configuration);\n        localState.vehicle.mobil.absorbContactEnabled = 1u;\n        localState.vehicle.mobil.physicsUpdatesEnabled =\n            (tick.actionFlags & 2u) == 0u;\n        for (uint respawn = 0u;\n             respawn < tick.respawnAtCheckpointCount; ++respawn)\n        {\n            if (forevervalidator::simulation::cuda::transition::Respawn(\n                    localState, configuration))\n            {\n                ++results[slot].executedRespawnCount;\n                ++localState.incrementalRespawnCount;\n                if (parameters->simulateStunts != 0u)\n                {\n                    *deviceState = localState;\n                    CudaCandidateState* fullState =\n                        reinterpret<CudaCandidateState*>(deviceState);\n                    forevervalidator::simulation::cuda::stunts::\n                        ApplyRespawnPenalty(fullState->stunts);\n                }\n            }\n        }\n\n        forevervalidator::simulation::cuda::physics::Status physicsStatus =\n            ExecuteSearchPhysicsStep(\n'
    new = '        forevervalidator::simulation::ApplyControlAndTimingPrefix(\n            localState, tick, false);\n        if (localState.firstStep == 0u)\n            forevervalidator::simulation::cuda::transition::PrepareSteadyStep(\n                localState, tick);\n        localState.vehicle.mobil.absorbContactEnabled = 1u;\n        localState.vehicle.mobil.physicsUpdatesEnabled = 1u;\n\n        forevervalidator::simulation::cuda::physics::Status physicsStatus =\n            ExecuteSearchPhysicsStep(\n'
    text = replace_specialization_fragment(text, old, new, "fragment 16")
    # @@ -15743,7 +15763,7 @@
    old = '            return;\n        }\n\n        if (parameters->simulateStunts != 0u)\n        {\n            CudaCandidateState* fullState =\n                reinterpret<CudaCandidateState*>(deviceState);\n'
    new = '            return;\n        }\n\n        if (false)\n        {\n            CudaCandidateState* fullState =\n                reinterpret<CudaCandidateState*>(deviceState);\n'
    text = replace_specialization_fragment(text, old, new, "fragment 17")
    # @@ -15774,7 +15794,7 @@
    old = '        results[slot].failureTick = UINT32_MAX;\n\n        if (publicTime < parameters->evaluationStartTimeMs) continue;\n        if (parameters->conditionInstructionCount != 0u &&\n            !EvaluateCondition(\n                condition, parameters->conditionInstructionCount,\n                localState,\n'
    new = '        results[slot].failureTick = UINT32_MAX;\n\n        if (publicTime < parameters->evaluationStartTimeMs) continue;\n        if (false &&\n            !EvaluateCondition(\n                condition, parameters->conditionInstructionCount,\n                localState,\n'
    text = replace_specialization_fragment(text, old, new, "fragment 18")
    # @@ -15793,16 +15813,14 @@
    old = '        }\n\n        uint stuntsScore = 0u;\n        if (parameters->simulateStunts != 0u)\n        {\n            CudaCandidateState* fullState =\n                reinterpret<CudaCandidateState*>(deviceState);\n            stuntsScore = fullState->stunts.stuntsScore;\n        }\n        cuda_search_detail__DeviceSample sample = EvaluateState(\n            *evaluator, localState, previousPosition,\n            double(publicTime - parameters->tickDurationMs),\n            double(publicTime), stuntsScore, &evaluatorReported);\n        sample.candidateId = candidateId;\n        sample.candidateSlot = slot;\n        sample.evaluationTick = evaluationIndex;\n'
    new = '        }\n\n        uint stuntsScore = 0u;\n        if (false)\n        {\n            CudaCandidateState* fullState =\n                reinterpret<CudaCandidateState*>(deviceState);\n            stuntsScore = fullState->stunts.stuntsScore;\n        }\n        cuda_search_detail__DeviceSample sample = EvaluateVelocityState(\n            localState, double(publicTime));\n        sample.candidateId = candidateId;\n        sample.candidateSlot = slot;\n        sample.evaluationTick = evaluationIndex;\n'
    text = replace_specialization_fragment(text, old, new, "fragment 19")
    # @@ -15814,13 +15832,13 @@
    old = '            localBest = sample;\n        ++evaluationIndex;\n\n        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&\n            evaluatorReported)\n        {\n            completed = true;\n            break;\n        }\n        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&\n            incumbent.preciseFinish != 0u &&\n            double(publicTime) >= incumbent.timeMs)\n        {\n'
    new = '            localBest = sample;\n        ++evaluationIndex;\n\n        if (false &&\n            evaluatorReported)\n        {\n            completed = true;\n            break;\n        }\n        if (false &&\n            incumbent.preciseFinish != 0u &&\n            double(publicTime) >= incumbent.timeMs)\n        {\n'
    text = replace_specialization_fragment(text, old, new, "fragment 20")
    patch_path = Path(__file__).with_name(
        "vulkan_steady_velocity_cooperative.patch")
    return apply_unified_patch(text, patch_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--types", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--specialization",
        choices=("generic", "steady-velocity"),
        default="generic",
    )
    parser.add_argument(
        "--handling",
        choices=("water",),
        default="water",
    )
    parser.add_argument(
        "--split-phases",
        action="store_true",
        help="emit the four-lane water collision pipeline entries",
    )
    return parser.parse_args()


SEARCH_ENTRY = r"""

namespace forevervalidator::simulation::vulkan_search {

enum class DeviceCandidateStatus : uint
{
    Success,
    CapacityExceeded,
    Cancelled,
    SimulationFailed,
    UnsupportedPhysicsTransition,
};

struct VulkanSearchParameters
{
    uint64_t scene;
    uint64_t configuration;
    uint64_t branchState;
    uint64_t baselineTicks;
    uint64_t evaluator;
    uint64_t condition;
    uint64_t candidateStates;
    uint64_t candidateBestStates;
    uint64_t collisionScratch;
    uint64_t collisionStorage;
    uint64_t shapeCollisionStorage;
    uint64_t shapeWorldStorage;
    uint64_t movingBoundsStorage;
    uint64_t unifiedMovingBoundsStorage;
    uint64_t surfaceHitStorage;
    uint64_t meshRangeStorage;
    uint64_t meshCellStorage;
    uint64_t responseOrderStorage;
    uint64_t controlStates;
    uint64_t eventCursors;
    uint64_t simulationActive;
    uint64_t evaluatorReported;
    uint64_t previousPositions;
    uint64_t evaluationIndices;
    uint64_t timelineDescriptors;
    uint64_t candidateTicks;
    uint64_t timelineResults;
    uint64_t baselineInputs;
    uint64_t modifiers;
    uint64_t smoothWeights;
    uint64_t mutableBoundaryControls;
    uint64_t candidateBestSamples;
    uint64_t randomStateWords;
    uint64_t candidateEvents;
    uint64_t temporaryEvents;
    uint64_t passBaselineEvents;
    uint64_t eligibleIndices;
    uint64_t eventCounts;
    uint64_t mutationCounts;
    uint64_t statuses;
    uint64_t activeCandidates;
    uint64_t cancellation;
    uint64_t firstCandidateId;
    int64_t branchTimeMs;
    int64_t mutableFromTimeMs;
    int64_t evaluationStartTimeMs;
    double lastImprovementTimeSeconds;
    double lastRestartTimeSeconds;
    double currentTimeSeconds;
    uint baselineInputCount;
    uint immutableTailInputCount;
    uint modifierCount;
    uint tickDurationMs;
    uint candidateCount;
    uint eventCapacity;
    uint baseline;
    uint legacyMutationPipeline;
    uint baselineInputsCanonical;
    uint timelineTickCount;
    uint conditionInstructionCount;
    uint prestartDurationMs;
    uint evaluationTickCount;
    uint simulateStunts;
    uint stateWordCount;
    uint scratchStride;
    uint shapeCapacity;
    uint _pad0;
    uint64_t winnerSummary;
    uint64_t winnerEvents;
};

struct VulkanSearchBatchSummary
{
    cuda_search_detail__DeviceSample winner;
    uint64_t totalMutationCount;
    uint64_t mutationImprovementCount;
    uint winnerSlot;
    uint winnerEventCount;
    uint winnerMutationCount;
    uint evaluatedCandidateCount;
    uint status;
    uint bestChanged;
};

struct VulkanSearchPushConstants
{
    uint64_t parameters;
    uint tickIndex;
    uint candidateBase;
    uint tickCount;
};

[[vk::push_constant]] ConstantBuffer<VulkanSearchPushConstants> searchPush;

[shader("compute")]
[numthreads(32, 1, 1)]
void GenerateVulkanSearchCandidates(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = searchPush.candidateBase + dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;

    CudaSearchInputEvent* baselineInputs =
        reinterpret<CudaSearchInputEvent*>(parameters->baselineInputs);
    CudaSearchModifierConfiguration* modifiers =
        reinterpret<CudaSearchModifierConfiguration*>(parameters->modifiers);
    double* smoothWeights =
        reinterpret<double*>(parameters->smoothWeights);
    cuda_search_detail__DeviceControlState* mutableBoundaryControls =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->mutableBoundaryControls);
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    uint* randomStateWords =
        reinterpret<uint*>(parameters->randomStateWords);
    CudaSearchInputEvent* candidateEvents =
        reinterpret<CudaSearchInputEvent*>(parameters->candidateEvents);
    CudaSearchInputEvent* temporaryEvents =
        reinterpret<CudaSearchInputEvent*>(parameters->temporaryEvents);
    CudaSearchInputEvent* passBaselineEvents =
        reinterpret<CudaSearchInputEvent*>(parameters->passBaselineEvents);
    uint* eligibleIndices =
        reinterpret<uint*>(parameters->eligibleIndices);
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    uint* mutationCounts = reinterpret<uint*>(parameters->mutationCounts);
    uint* statuses = reinterpret<uint*>(parameters->statuses);
    uint8_t* activeCandidates =
        reinterpret<uint8_t*>(parameters->activeCandidates);
    uint* cancellation = reinterpret<uint*>(parameters->cancellation);

    uint64_t eventOffset = uint64_t(slot) * parameters->eventCapacity;
    CudaSearchInputEvent* events = candidateEvents + eventOffset;
    CudaSearchInputEvent* temporary = temporaryEvents + eventOffset;
    CudaSearchInputEvent* passBaseline = passBaselineEvents + eventOffset;
    uint* eligible = eligibleIndices + eventOffset;
    uint eventCount = parameters->baselineInputCount;
    for (uint index = 0u; index < eventCount; ++index)
        events[index] = baselineInputs[index];
    statuses[slot] = uint(DeviceCandidateStatus::Success);
    samples[slot + 1u] = {};
    if (*cancellation != 0u)
    {
        statuses[slot] = uint(DeviceCandidateStatus::Cancelled);
        activeCandidates[slot] = 0u;
        eventCounts[slot] = eventCount;
        mutationCounts[slot] = 0u;
        return;
    }

    uint64_t candidateId = parameters->firstCandidateId + slot;
    DeviceMt19937 random;
    random.Initialize(randomStateWords, slot, parameters->candidateCount);
    if (parameters->baseline == 0u)
    {
        bool normalized = parameters->baselineInputsCanonical != 0u;
        for (uint pass = 0u; pass < parameters->modifierCount; ++pass)
        {
            if (!ApplyModifier(
                    modifiers[pass], pass, candidateId, random,
                    parameters->tickDurationMs, 0,
                    *mutableBoundaryControls,
                    baselineInputs, parameters->baselineInputCount,
                    events, &eventCount, parameters->eventCapacity,
                    temporary, passBaseline, eligible,
                    smoothWeights,
                    parameters->legacyMutationPipeline != 0u,
                    &normalized))
            {
                statuses[slot] =
                    uint(DeviceCandidateStatus::CapacityExceeded);
                activeCandidates[slot] = 0u;
                eventCounts[slot] = eventCount;
                mutationCounts[slot] = 0u;
                return;
            }
        }
        if (parameters->legacyMutationPipeline != 0u || !normalized)
        {
            for (uint index = 0u;
                 index < parameters->baselineInputCount; ++index)
                passBaseline[index] = baselineInputs[index];
            eventCount = NormalizeEvents(
                events, eventCount, temporary,
                passBaseline, parameters->baselineInputCount, 0,
                parameters->legacyMutationPipeline != 0u,
                parameters->eventCapacity);
            if (eventCount == UINT32_MAX)
            {
                statuses[slot] =
                    uint(DeviceCandidateStatus::CapacityExceeded);
                activeCandidates[slot] = 0u;
                eventCounts[slot] = parameters->eventCapacity;
                mutationCounts[slot] = 0u;
                return;
            }
        }
    }
    uint mutationCount = parameters->baseline != 0u
        ? 0u
        : EffectiveChangeCount(
            baselineInputs, parameters->baselineInputCount,
            events, eventCount);
    if (parameters->baseline == 0u &&
        eventCount != parameters->baselineInputCount)
        mutationCount += parameters->immutableTailInputCount;
    eventCounts[slot] = eventCount;
    mutationCounts[slot] = mutationCount;
    activeCandidates[slot] =
        (parameters->baseline != 0u || mutationCount != 0u) ? 1u : 0u;
}

}
"""


SEARCH_SCRATCH_TYPE = r"""

struct cuda__collision__CudaCollisionSearchReference
{
    cuda__collision__CudaCollisionSearchTile* tile;
    uint lane;

    property separation : GmVec3 {
        get { return {tile->separationX[lane], tile->separationY[lane],
                      tile->separationZ[lane]}; }
        [nonmutating] set {
            tile->separationX[lane] = newValue.x;
            tile->separationY[lane] = newValue.y;
            tile->separationZ[lane] = newValue.z;
        }
    }
    property impulseNormal : GmVec3 {
        get { return {tile->impulseNormalX[lane], tile->impulseNormalY[lane],
                      tile->impulseNormalZ[lane]}; }
        [nonmutating] set {
            tile->impulseNormalX[lane] = newValue.x;
            tile->impulseNormalY[lane] = newValue.y;
            tile->impulseNormalZ[lane] = newValue.z;
        }
    }
    property contactPoint : GmVec3 {
        get { return {tile->contactPointX[lane], tile->contactPointY[lane],
                      tile->contactPointZ[lane]}; }
        [nonmutating] set {
            tile->contactPointX[lane] = newValue.x;
            tile->contactPointY[lane] = newValue.y;
            tile->contactPointZ[lane] = newValue.z;
        }
    }
    property extraNegated : GmVec3 {
        get { return {tile->extraNegatedX[lane], tile->extraNegatedY[lane],
                      tile->extraNegatedZ[lane]}; }
        [nonmutating] set {
            tile->extraNegatedX[lane] = newValue.x;
            tile->extraNegatedY[lane] = newValue.y;
            tile->extraNegatedZ[lane] = newValue.z;
        }
    }
    property materialA : uint {
        get { return tile->materialA[lane]; }
        [nonmutating] set { tile->materialA[lane] = newValue; }
    }
    property materialB : uint {
        get { return tile->materialB[lane]; }
        [nonmutating] set { tile->materialB[lane] = newValue; }
    }
    property sphereMergePrimary : bool {
        get { return tile->sphereMergePrimary[lane] != 0u; }
        [nonmutating] set {
            tile->sphereMergePrimary[lane] = newValue ? 1u : 0u;
        }
    }
    property movingShapeIndex : uint {
        get { return tile->movingShapeIndex[lane]; }
        [nonmutating] set { tile->movingShapeIndex[lane] = newValue; }
    }
    property staticSurfaceIndex : uint {
        get { return tile->staticSurfaceIndex[lane]; }
        [nonmutating] set { tile->staticSurfaceIndex[lane] = newValue; }
    }
    property staticActorIndex : uint {
        get { return tile->staticActorIndex[lane]; }
        [nonmutating] set { tile->staticActorIndex[lane] = newValue; }
    }
};

struct cuda__collision__VulkanCollisionSearchScratch
{
    uint collisionCount;
    uint shapeCollisionCount;
    uint surfaceHitCount;
    uint8_t overflow;
    uint8_t surfaceCacheEnabled;
    uint8_t _pad0[2];
    cuda__collision__CudaCollisionSearchTile* collisionStorage;
    cuda__collision__CudaCollisionSearchTile* shapeCollisionStorage;
    GmIso4* shapeWorldStorage;
    GmBoxAligned* movingBoundsStorage;
    GmBoxAligned* unifiedMovingBoundsStorage;
    cuda__collision__CudaCollisionSurfaceHit* surfaceHitStorage;
    cuda__collision__CudaCollisionMeshRange* meshRangeStorage;
    uint* meshCellStorage;
    uint slot;
    uint stride;
    uint shapeCapacity;
    uint overflowReason;
    uint8_t surfaceCacheValid;
    uint8_t _pad1[3];
    uint meshCellCount;
    uint8_t meshCacheValid;
    uint8_t _pad2[3];
    uint replacementOverflowCount;
    float replacementSumX;
    float replacementSumY;
    float replacementSumZ;
    uint8_t _pad3[4];
    uint16_t* responseOrderStorage;
};
"""


SEARCH_SCRATCH_ACCESSORS = r"""

inline cuda__collision__CudaCollisionSearchReference SearchCollisionAt(
    cuda__collision__CudaCollisionSearchTile* storage,
    inout cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index)
{
    uint tileStride = (scratch.stride + CudaCollisionSearchTileWidth - 1u) /
        CudaCollisionSearchTileWidth;
    cuda__collision__CudaCollisionSearchTile* tile = storage +
        uint64_t(index) * tileStride +
        scratch.slot / CudaCollisionSearchTileWidth;
    return {tile, scratch.slot % CudaCollisionSearchTileWidth};
}

inline cuda__collision__CudaCollisionSearchReference CollisionAt(
    inout cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index)
{
    return SearchCollisionAt(scratch.collisionStorage, scratch, index);
}

inline cuda__collision__CudaCollisionSearchReference ShapeCollisionAt(
    inout cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index)
{
    return SearchCollisionAt(scratch.shapeCollisionStorage, scratch, index);
}

inline GmVec3 ReplacementOverflowAtConst(
    __constref cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index)
{
    cuda__collision__VulkanCollisionSearchScratch mutableScratch = scratch;
    cuda__collision__CudaCollisionSearchReference storage =
        ShapeCollisionAt(mutableScratch, index >> 1u);
    return (index & 1u) == 0u
        ? storage.extraNegated : storage.contactPoint;
}

inline void StoreReplacementOverflowAt(
    inout cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index,
    __constref GmVec3 value)
{
    cuda__collision__CudaCollisionSearchReference storage =
        ShapeCollisionAt(scratch, index >> 1u);
    if ((index & 1u) == 0u) storage.extraNegated = value;
    else storage.contactPoint = value;
}

inline cuda__collision__CudaCollisionSearchReference OrderedCollisionAt(
    inout cuda__collision__VulkanCollisionSearchScratch scratch,
    uint index)
{
    return CollisionAt(
        scratch,
        scratch.responseOrderStorage[
            uint64_t(index) * scratch.stride + scratch.slot]);
}

"""


SEARCH_SIMULATION_ENTRY = r"""

namespace forevervalidator::simulation::vulkan_search {

bool SearchStrictlyBetter(
    __constref cuda_search_detail__DeviceSample candidate,
    __constref cuda_search_detail__DeviceSample incumbent,
    bool maximize)
{
    if (candidate.valid == 0u) return false;
    if (incumbent.valid == 0u) return true;
    if (candidate.score != incumbent.score)
        return maximize ? candidate.score > incumbent.score
                        : candidate.score < incumbent.score;
    return candidate.eventCount < incumbent.eventCount;
}

bool SearchScriptedBetter(
    __constref cuda_search_detail__DeviceSample candidate,
    __constref cuda_search_detail__DeviceSample incumbent)
{
    if (candidate.valid == 0u) return false;
    if (incumbent.valid == 0u) return true;
    if (candidate.scriptedObjectiveCount == 0u ||
        candidate.scriptedObjectiveCount != incumbent.scriptedObjectiveCount)
        return false;
    bool better = false;
    for (uint i = 0u; i < candidate.scriptedObjectiveCount; ++i)
    {
        if (candidate.objectiveScores[i] < incumbent.objectiveScores[i])
            return false;
        better = better ||
            candidate.objectiveScores[i] > incumbent.objectiveScores[i];
    }
    return better;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void SimulateVulkanSearchCandidates(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = searchPush.candidateBase + dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;

    uint8_t* activeCandidates =
        reinterpret<uint8_t*>(parameters->activeCandidates);
    uint* statuses = reinterpret<uint*>(parameters->statuses);
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    if (activeCandidates[slot] == 0u || statuses[slot] != 0u)
    {
        samples[slot + 1u] = {};
        return;
    }

    CudaPackedSceneHeader* scene =
        reinterpret<CudaPackedSceneHeader*>(parameters->scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(
            parameters->configuration);
    CudaCandidateState* branchState =
        reinterpret<CudaCandidateState*>(parameters->branchState);
    CudaCandidateState* states =
        reinterpret<CudaCandidateState*>(parameters->candidateStates);
    CudaCandidateState* bestStates =
        reinterpret<CudaCandidateState*>(parameters->candidateBestStates);
    CudaControlTick* baselineTicks =
        reinterpret<CudaControlTick*>(parameters->baselineTicks);
    CudaSearchEvaluatorConfiguration* evaluator =
        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);
    CudaSearchConditionInstruction* condition =
        reinterpret<CudaSearchConditionInstruction*>(parameters->condition);
    cuda__collision__CudaCollisionScratch* scratch =
        reinterpret<cuda__collision__CudaCollisionScratch*>(
            parameters->collisionScratch);
    cuda_search_detail__DeviceControlState* boundaryControls =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->mutableBoundaryControls);
    CudaSearchInputEvent* candidateEvents =
        reinterpret<CudaSearchInputEvent*>(parameters->candidateEvents);
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    uint* cancellation = reinterpret<uint*>(parameters->cancellation);

    if (!forevervalidator::simulation::ValidPackedInputs(scene, configuration) ||
        branchState->schemaVersion != CudaCandidateState_SchemaVersion)
    {
        statuses[slot] = 4u;
        samples[slot + 1u] = {};
        return;
    }

    states[slot] = *branchState;
    states[slot].candidateId = uint(parameters->firstCandidateId + slot);
    cuda_search_detail__DeviceControlState controlState = *boundaryControls;
    uint64_t eventOffset = uint64_t(slot) * parameters->eventCapacity;
    CudaSearchInputEvent* events = candidateEvents + eventOffset;
    uint eventCount = eventCounts[slot];
    uint nextEventIndex = 0u;
    bool evaluatorReported = false;
    bool maximize = MaximizesScore(evaluator->kind);
    cuda_search_detail__DeviceSample incumbent = samples[0];
    cuda_search_detail__DeviceSample localBest = {};
    uint evaluationIndex = 0u;
    uint64_t candidateId = parameters->firstCandidateId + slot;

    for (uint tickIndex = 0u;
         tickIndex < parameters->timelineTickCount; ++tickIndex)
    {
        if ((tickIndex & 63u) == 0u && *cancellation != 0u)
        {
            statuses[slot] = 2u;
            samples[slot + 1u] = localBest;
            return;
        }
        int64_t publicTime = parameters->branchTimeMs +
            int64_t(tickIndex + 1u) * parameters->tickDurationMs;
        int64_t suffixTime = publicTime - parameters->mutableFromTimeMs;
        while (nextEventIndex < eventCount &&
               events[nextEventIndex].timeMs <= suffixTime)
        {
            forevervalidator::simulation::cuda_search_detail::ApplyControlEvent(
                controlState, events[nextEventIndex],
                parameters->mutableFromTimeMs);
            ++nextEventIndex;
        }

        CudaControlTick tick = baselineTicks[tickIndex];
        tick.controls = forevervalidator::simulation::cuda_search_detail::ControlsFromState(controlState);
        tick.stuntsInput = forevervalidator::simulation::cuda_search_detail::StuntsFromState(
            controlState, parameters->prestartDurationMs);
        GmVec3 previousPosition = states[slot].body.current.position;
        ApplyControlPrefix(states[slot], tick);
        if (states[slot].firstStep == 0u)
            forevervalidator::simulation::cuda::transition::PrepareStep(
                states[slot], tick, configuration);
        states[slot].vehicle.mobil.absorbContactEnabled = 1u;
        states[slot].vehicle.mobil.physicsUpdatesEnabled =
            (tick.actionFlags & 2u) == 0u ? 1u : 0u;
        for (uint respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn)
        {
            if (forevervalidator::simulation::cuda::transition::Respawn(
                    states[slot], configuration))
            {
                ++states[slot].incrementalRespawnCount;
                if (parameters->simulateStunts != 0u)
                    forevervalidator::simulation::cuda::stunts::ApplyRespawnPenalty(
                        states[slot].stunts);
            }
        }
        forevervalidator::simulation::cuda::physics::Status physicsStatus =
            forevervalidator::simulation::ExecuteTimelinePhysicsStep(
            scene, configuration, states + slot, scratch + slot);
        if (physicsStatus != forevervalidator::simulation::cuda::physics::Status::Success)
        {
            statuses[slot] = 4u;
            samples[slot + 1u] = localBest;
            return;
        }
        if (parameters->simulateStunts != 0u)
        {
            forevervalidator::simulation::cuda::stunts::Status stuntStatus =
                forevervalidator::simulation::cuda::stunts::Update(
                    states[slot], tick);
            if (stuntStatus != forevervalidator::simulation::cuda::stunts::Status::Success)
            {
                statuses[slot] = 1u;
                samples[slot + 1u] = localBest;
                return;
            }
        }
        states[slot].firstStep = 0u;
        ++states[slot].controlCursor;
        if (publicTime < parameters->evaluationStartTimeMs) continue;
        if (parameters->conditionInstructionCount != 0u &&
            !EvaluateCondition(
                condition, parameters->conditionInstructionCount,
                states[slot],
                parameters->baseline != 0u ? 0u : candidateId + 1u,
                parameters->lastImprovementTimeSeconds,
                parameters->lastRestartTimeSeconds,
                parameters->currentTimeSeconds))
        {
            ++evaluationIndex;
            if (states[slot].race.progress.raceCompleted != 0u) break;
            continue;
        }
        if (evaluator->kind == CudaSearchEvaluatorKind_Scripted)
        {
            if (UpdateScriptedSample(
                    *evaluator, states[slot],
                    parameters->baseline != 0u ? 0u : candidateId + 1u,
                    parameters->lastImprovementTimeSeconds,
                    parameters->lastRestartTimeSeconds,
                    parameters->currentTimeSeconds,
                    double(publicTime), localBest))
            {
                localBest.candidateId = candidateId;
                localBest.candidateSlot = slot;
                localBest.evaluationTick = evaluationIndex;
                localBest.eventCount = eventCount;
                localBest.logicalOrder = 1u + uint64_t(slot) *
                    parameters->evaluationTickCount + evaluationIndex;
                localBest.mutation = parameters->baseline == 0u ? 1u : 0u;
                bestStates[slot] = states[slot];
            }
            ++evaluationIndex;
            if (states[slot].race.progress.raceCompleted != 0u) break;
            continue;
        }
        uint stuntsScore = parameters->simulateStunts != 0u
            ? states[slot].stunts.stuntsScore : 0u;
        cuda_search_detail__DeviceSample sample = EvaluateState(
            *evaluator, states[slot], previousPosition,
            double(publicTime - parameters->tickDurationMs),
            double(publicTime), stuntsScore, &evaluatorReported);
        sample.candidateId = candidateId;
        sample.candidateSlot = slot;
        sample.evaluationTick = evaluationIndex;
        sample.eventCount = eventCount;
        sample.logicalOrder = 1u + uint64_t(slot) *
            parameters->evaluationTickCount + evaluationIndex;
        sample.mutation = parameters->baseline == 0u ? 1u : 0u;
        if (SearchStrictlyBetter(sample, localBest, maximize))
        {
            localBest = sample;
            bestStates[slot] = states[slot];
        }
        ++evaluationIndex;
        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
            evaluatorReported)
            break;
        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
            incumbent.preciseFinish != 0u &&
            double(publicTime) >= incumbent.timeMs)
        {
            localBest = {};
            break;
        }
    }
    samples[slot + 1u] = localBest;
}

}
"""


SEARCH_STAGED_ENTRIES = r"""

namespace forevervalidator::simulation::vulkan_search {

bool SearchStrictlyBetter(
    __constref cuda_search_detail__DeviceSample candidate,
    __constref cuda_search_detail__DeviceSample incumbent,
    bool maximize)
{
    if (candidate.valid == 0u) return false;
    if (incumbent.valid == 0u) return true;
    if (candidate.score != incumbent.score)
        return maximize ? candidate.score > incumbent.score
                        : candidate.score < incumbent.score;
    return candidate.eventCount < incumbent.eventCount;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void InitializeVulkanSearchCandidates(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = searchPush.candidateBase + dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;

    uint* sourceState = reinterpret<uint*>(parameters->branchState);
    uint* candidateStateWords =
        reinterpret<uint*>(parameters->candidateStates);
    uint stateOffset = slot * parameters->stateWordCount;
    for (uint index = 0u; index < parameters->stateWordCount; ++index)
        candidateStateWords[stateOffset + index] = sourceState[index];

    uint8_t* stateBytes = reinterpret<uint8_t*>(parameters->candidateStates);
    CudaCandidatePhysicsState* state =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(stateOffset) * 4u);
    state->candidateId = uint(parameters->firstCandidateId + slot);
    cuda__collision__VulkanCollisionSearchScratch* collisionScratch =
        reinterpret<cuda__collision__VulkanCollisionSearchScratch*>(
            parameters->collisionScratch);
    collisionScratch[slot] = {};
    collisionScratch[slot].surfaceCacheEnabled = 1u;
    collisionScratch[slot].collisionStorage =
        reinterpret<cuda__collision__CudaCollisionSearchTile*>(
            parameters->collisionStorage);
    collisionScratch[slot].shapeCollisionStorage =
        reinterpret<cuda__collision__CudaCollisionSearchTile*>(
            parameters->shapeCollisionStorage);
    collisionScratch[slot].shapeWorldStorage =
        reinterpret<GmIso4*>(parameters->shapeWorldStorage);
    collisionScratch[slot].movingBoundsStorage =
        reinterpret<GmBoxAligned*>(parameters->movingBoundsStorage);
    collisionScratch[slot].unifiedMovingBoundsStorage =
        reinterpret<GmBoxAligned*>(parameters->unifiedMovingBoundsStorage);
    collisionScratch[slot].surfaceHitStorage =
        reinterpret<cuda__collision__CudaCollisionSurfaceHit*>(
            parameters->surfaceHitStorage);
    collisionScratch[slot].meshRangeStorage =
        reinterpret<cuda__collision__CudaCollisionMeshRange*>(
            parameters->meshRangeStorage);
    collisionScratch[slot].meshCellStorage =
        reinterpret<uint*>(parameters->meshCellStorage);
    collisionScratch[slot].responseOrderStorage =
        reinterpret<uint16_t*>(parameters->responseOrderStorage);
    collisionScratch[slot].slot = slot;
    collisionScratch[slot].stride = parameters->scratchStride;
    collisionScratch[slot].shapeCapacity = parameters->shapeCapacity;
    cuda_search_detail__DeviceControlState* boundary =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->mutableBoundaryControls);
    cuda_search_detail__DeviceControlState* controls =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->controlStates);
    controls[slot] = *boundary;
    uint* eventCursors = reinterpret<uint*>(parameters->eventCursors);
    eventCursors[slot] = 0u;
    uint8_t* active = reinterpret<uint8_t*>(parameters->activeCandidates);
    uint8_t* simulationActive =
        reinterpret<uint8_t*>(parameters->simulationActive);
    simulationActive[slot] = active[slot];
    uint8_t* reported =
        reinterpret<uint8_t*>(parameters->evaluatorReported);
    reported[slot] = 0u;
    uint* evaluationIndices =
        reinterpret<uint*>(parameters->evaluationIndices);
    evaluationIndices[slot] = 0u;
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    samples[slot + 1u] = {};

    forevervalidator::simulation::VulkanTimelineDescriptor* descriptors =
        reinterpret<forevervalidator::simulation::VulkanTimelineDescriptor*>(
            parameters->timelineDescriptors);
    descriptors[slot].firstTick = slot;
    descriptors[slot].tickCount = active[slot] != 0u ? 1u : 0u;
    descriptors[slot]._pad0 = 0u;
    descriptors[slot].firstObservation = 0u;
    descriptors[slot].observationCapacity = 0u;
    descriptors[slot]._pad1 = 0u;
    forevervalidator::simulation::VulkanTimelineResult* results =
        reinterpret<forevervalidator::simulation::VulkanTimelineResult*>(
            parameters->timelineResults);
    results[slot] = {};
    results[slot].failureTick = UINT32_MAX;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void PrepareVulkanSearchTick(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = searchPush.candidateBase + dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;
    uint8_t* simulationActive =
        reinterpret<uint8_t*>(parameters->simulationActive);
    forevervalidator::simulation::VulkanTimelineDescriptor* descriptors =
        reinterpret<forevervalidator::simulation::VulkanTimelineDescriptor*>(
            parameters->timelineDescriptors);
    if (simulationActive[slot] == 0u)
    {
        descriptors[slot].tickCount = 0u;
        return;
    }
    descriptors[slot].tickCount = 1u;

    CudaSearchInputEvent* events =
        reinterpret<CudaSearchInputEvent*>(parameters->candidateEvents) +
        uint64_t(slot) * parameters->eventCapacity;
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    uint* eventCursors = reinterpret<uint*>(parameters->eventCursors);
    cuda_search_detail__DeviceControlState* controls =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->controlStates);
    int64_t publicTime = parameters->branchTimeMs +
        int64_t(searchPush.tickIndex + 1u) * parameters->tickDurationMs;
    int64_t suffixTime = publicTime - parameters->mutableFromTimeMs;
    uint cursor = eventCursors[slot];
    while (cursor < eventCounts[slot] &&
           events[cursor].timeMs <= suffixTime)
    {
        forevervalidator::simulation::cuda_search_detail::ApplyControlEvent(
            controls[slot], events[cursor], parameters->mutableFromTimeMs);
        ++cursor;
    }
    eventCursors[slot] = cursor;

    CudaControlTick* baselineTicks =
        reinterpret<CudaControlTick*>(parameters->baselineTicks);
    CudaControlTick* candidateTicks =
        reinterpret<CudaControlTick*>(parameters->candidateTicks);
    CudaControlTick tick = baselineTicks[searchPush.tickIndex];
    tick.observe = false;
    tick.controls =
        forevervalidator::simulation::cuda_search_detail::ControlsFromState(
            controls[slot]);
    tick.stuntsInput =
        forevervalidator::simulation::cuda_search_detail::StuntsFromState(
            controls[slot], parameters->prestartDurationMs);
    candidateTicks[slot] = tick;

    uint8_t* stateBytes = reinterpret<uint8_t*>(parameters->candidateStates);
    CudaCandidatePhysicsState* state =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(slot) * parameters->stateWordCount * 4u);
    GmVec3* previousPositions =
        reinterpret<GmVec3*>(parameters->previousPositions);
    previousPositions[slot] = state->body.current.position;
    forevervalidator::simulation::VulkanTimelineResult* results =
        reinterpret<forevervalidator::simulation::VulkanTimelineResult*>(
            parameters->timelineResults);
    results[slot] = {};
    results[slot].failureTick = UINT32_MAX;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void EvaluateVulkanSearchTick(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;
    uint8_t* simulationActive =
        reinterpret<uint8_t*>(parameters->simulationActive);
    if (simulationActive[slot] == 0u) return;

    forevervalidator::simulation::VulkanTimelineResult* timelineResults =
        reinterpret<forevervalidator::simulation::VulkanTimelineResult*>(
            parameters->timelineResults);
    uint* statuses = reinterpret<uint*>(parameters->statuses);
    if (timelineResults[slot].status != 0u)
    {
        statuses[slot] = timelineResults[slot].status == 3u ? 1u
            : timelineResults[slot].status == 4u ? 2u : 4u;
        simulationActive[slot] = 0u;
        return;
    }

    int64_t publicTime = parameters->branchTimeMs +
        int64_t(searchPush.tickIndex + 1u) * parameters->tickDurationMs;
    if (publicTime < parameters->evaluationStartTimeMs) return;
    uint8_t* stateBytes = reinterpret<uint8_t*>(parameters->candidateStates);
    CudaCandidatePhysicsState* state =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(slot) * parameters->stateWordCount * 4u);
    uint* evaluationIndices =
        reinterpret<uint*>(parameters->evaluationIndices);
    uint evaluationIndex = evaluationIndices[slot];
    CudaSearchConditionInstruction* condition =
        reinterpret<CudaSearchConditionInstruction*>(parameters->condition);
    uint64_t candidateId = parameters->firstCandidateId + slot;
    if (parameters->conditionInstructionCount != 0u &&
        !EvaluateCondition(
            condition, parameters->conditionInstructionCount,
            *state,
            parameters->baseline != 0u ? 0u : candidateId + 1u,
            parameters->lastImprovementTimeSeconds,
            parameters->lastRestartTimeSeconds,
            parameters->currentTimeSeconds))
    {
        evaluationIndices[slot] = evaluationIndex + 1u;
        if (state->race.progress.raceCompleted != 0u)
            simulationActive[slot] = 0u;
        return;
    }

    CudaSearchEvaluatorConfiguration* evaluator =
        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);
    GmVec3* previousPositions =
        reinterpret<GmVec3*>(parameters->previousPositions);
    uint8_t* reported =
        reinterpret<uint8_t*>(parameters->evaluatorReported);
    bool evaluatorReported = reported[slot] != 0u;
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    if (evaluator->kind == CudaSearchEvaluatorKind_Scripted)
    {
        cuda_search_detail__DeviceSample sample = samples[slot + 1u];
        if (UpdateScriptedSample(
                *evaluator, *state,
                parameters->baseline != 0u ? 0u : candidateId + 1u,
                parameters->lastImprovementTimeSeconds,
                parameters->lastRestartTimeSeconds,
                parameters->currentTimeSeconds,
                double(publicTime), sample))
        {
            sample.candidateId = candidateId;
            sample.candidateSlot = slot;
            sample.evaluationTick = evaluationIndex;
            sample.eventCount = eventCounts[slot];
            sample.logicalOrder = 1u + uint64_t(slot) *
                parameters->evaluationTickCount + evaluationIndex;
            sample.mutation = parameters->baseline == 0u ? 1u : 0u;
            samples[slot + 1u] = sample;
        }
        evaluationIndices[slot] = evaluationIndex + 1u;
        return;
    }
    uint stuntsScore = 0u;
    if (parameters->simulateStunts != 0u)
    {
        CudaCandidateState* fullState =
            reinterpret<CudaCandidateState*>(state);
        stuntsScore = fullState->stunts.stuntsScore;
    }
    cuda_search_detail__DeviceSample sample = EvaluateState(
        *evaluator, *state, previousPositions[slot],
        double(publicTime - parameters->tickDurationMs),
        double(publicTime), stuntsScore,
        &evaluatorReported);
    reported[slot] = evaluatorReported ? 1u : 0u;
    sample.candidateId = candidateId;
    sample.candidateSlot = slot;
    sample.evaluationTick = evaluationIndex;
    sample.eventCount = eventCounts[slot];
    sample.logicalOrder = 1u + uint64_t(slot) *
        parameters->evaluationTickCount + evaluationIndex;
    sample.mutation = parameters->baseline == 0u ? 1u : 0u;
    if (SearchStrictlyBetter(
            sample, samples[slot + 1u], MaximizesScore(evaluator->kind)))
        samples[slot + 1u] = sample;
    evaluationIndices[slot] = evaluationIndex + 1u;

    if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
        evaluatorReported)
    {
        simulationActive[slot] = 0u;
        return;
    }
    cuda_search_detail__DeviceSample incumbent = samples[0];
    if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
        incumbent.preciseFinish != 0u &&
        double(publicTime) >= incumbent.timeMs)
    {
        samples[slot + 1u] = {};
        simulationActive[slot] = 0u;
    }
}

}
"""


SEARCH_PHYSICS_SUPPORT = r"""

namespace forevervalidator::simulation {

struct VulkanTimelineDescriptor
{
    uint64_t firstTick;
    uint tickCount;
    uint _pad0;
    uint64_t firstObservation;
    uint observationCapacity;
    uint _pad1;
};

struct VulkanTimelineResult
{
    uint status;
    uint failureTick;
    uint failureDetail;
    uint executedTickCount;
    uint executedRespawnCount;
    uint observationCount;
};

bool ValidPackedInputs(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration)
{
    return scene != nullptr && configuration != nullptr &&
        scene->magic == CudaPackedSceneHeader_Magic &&
        scene->schemaVersion == CudaPackedSceneHeader_SchemaVersion &&
        configuration->magic == CudaPackedStaticConfigurationHeader_Magic &&
        configuration->schemaVersion == CudaPackedStaticConfigurationHeader_SchemaVersion;
}

void ApplyControlAndTimingPrefix(
    inout CudaCandidatePhysicsState state,
    __constref CudaControlTick tick,
    bool applyControls)
{
    state.world.schemePeriodMs = tick.periodMs;
    state.world.tickTimeMs = tick.timeMs;
    if (!applyControls) return;
    state.vehicle.controls.lowSpeedGateA = tick.controls.lowSpeedGateA;
    state.vehicle.controls.lowSpeedGateB = tick.controls.lowSpeedGateB;
    state.vehicle.controls.steeringControl = tick.controls.steering;
    state.vehicle.frameHistory.physicsCurrent.lowSpeedGateA = tick.controls.lowSpeedGateA;
    state.vehicle.frameHistory.physicsCurrent.lowSpeedGateB = tick.controls.lowSpeedGateB;
    state.vehicle.frameHistory.physicsCurrent.steeringControl = tick.controls.steering;
}

}
"""


SEARCH_PHYSICS_ENTRY = r"""

namespace forevervalidator::simulation::vulkan_search {

[noinline]
forevervalidator::simulation::cuda::vehicle::ForceStatus
ExecuteSearchForcePass(
    CudaPackedStaticConfigurationHeader* configuration,
    inout CudaCandidatePhysicsState state,
    float dt)
{
    forevervalidator::simulation::cuda::environment::BeginForcePass(
        state.body, configuration);
    if (!state.vehicle.mobil.physicsUpdatesEnabled)
        return forevervalidator::simulation::cuda::vehicle::ForceStatus::Success;
    return forevervalidator::simulation::cuda::vehicle::ComputeForcesModel6<
        true, CudaHandlingSpecialization_Generic>(
            state, configuration, dt);
}

[noinline]
uint ExecuteSearchCollisionDetection(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    inout CudaCandidatePhysicsState state,
    cuda__collision__VulkanCollisionSearchScratch* scratch,
    float dt)
{
    forevervalidator::simulation::cuda::dynamics::PreCollision<true>(
        state.body, *scratch, dt);
    return forevervalidator::simulation::cuda::collision::Detect<
        false, true, true, false>(
            scene, configuration, state, *scratch);
}

[noinline]
uint ExecuteSearchCollisionResponse(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    inout CudaCandidatePhysicsState state,
    cuda__collision__VulkanCollisionSearchScratch* scratch)
{
    return forevervalidator::simulation::cuda::collision::Respond<
        false, true, true>(
            scene, configuration, state, *scratch);
}

[noinline]
forevervalidator::simulation::cuda::physics::Status
ExecuteSearchCollisionSubstep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    inout CudaCandidatePhysicsState state,
    cuda__collision__VulkanCollisionSearchScratch* scratch,
    float dt)
{
    forevervalidator::simulation::cuda::vehicle::ForceStatus forceStatus =
        ExecuteSearchForcePass(configuration, state, dt);
    if (forceStatus !=
            forevervalidator::simulation::cuda::vehicle::ForceStatus::Success)
        return forevervalidator::simulation::cuda::physics::Status(
            uint(forevervalidator::simulation::cuda::physics::Status::
                UnsupportedForceBase) + uint(forceStatus));
    uint collisionStatus = ExecuteSearchCollisionDetection(
        scene, configuration, state, scratch, dt);
    if (collisionStatus == cuda__collision__Status_Success)
        collisionStatus = ExecuteSearchCollisionResponse(
            scene, configuration, state, scratch);
    if (collisionStatus != cuda__collision__Status_Success)
        return forevervalidator::simulation::cuda::physics::Status(
            uint(forevervalidator::simulation::cuda::physics::Status::
                CollisionFailureBase) + collisionStatus);
    forevervalidator::simulation::cuda::dynamics::PostCollision<true>(
        state.body, *scratch);
    return forevervalidator::simulation::cuda::physics::Status::Success;
}

[noinline]
forevervalidator::simulation::cuda::physics::Status ExecuteSearchPhysicsStep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    inout CudaCandidatePhysicsState state,
    cuda__collision__VulkanCollisionSearchScratch* scratch)
{
    const float dt = float(int(state.world.schemePeriodMs)) * 0.001f;
    if (state.body.dynamicActive)
    {
        state.body.temporary = state.body.current;
        GmVec3 linear = state.body.current.linearSpeed;
        GmVec3 angular = state.body.current.angularSpeed;
        const float linearLength =
            forevervalidator::simulation::cuda::exact::Sqrt(
                (linear.y * linear.y + linear.x * linear.x) +
                linear.z * linear.z);
        const float angularLength =
            forevervalidator::simulation::cuda::exact::Sqrt(
                (angular.x * angular.x + angular.y * angular.y) +
                angular.z * angular.z);
        const float scaled =
            forevervalidator::simulation::cuda::exact::Divide(
                (linearLength + angularLength) * dt,
                state.body.parameters.maxStepDistance);
        uint substeps =
            forevervalidator::simulation::cuda::exact::
                TruncateToUint32Modulo(scaled) + 1u;
        if (substeps > 1000u) substeps = 1000u;
        float remaining = dt;
        if (substeps > 1u)
        {
            const float split =
                forevervalidator::simulation::cuda::exact::Divide(
                    dt,
                    forevervalidator::simulation::cuda::exact::
                        FromUnsignedInteger(substeps));
            for (uint count = substeps - 1u; count != 0u; --count)
            {
                forevervalidator::simulation::cuda::physics::Status status =
                    ExecuteSearchCollisionSubstep(
                        scene, configuration, state, scratch, split);
                if (status != forevervalidator::simulation::cuda::physics::
                                      Status::Success)
                    return status;
                remaining -= split;
            }
        }
        forevervalidator::simulation::cuda::physics::Status finalStatus =
            ExecuteSearchCollisionSubstep(
                scene, configuration, state, scratch, remaining);
        if (finalStatus != forevervalidator::simulation::cuda::physics::
                                   Status::Success)
            return finalStatus;
        state.body.write = state.body.temporary;
    }
    if (state.vehicle.mobil.physicsUpdatesEnabled)
        forevervalidator::simulation::cuda::vehicle::
            AfterContactsWithoutSnapshots(state);
    return forevervalidator::simulation::cuda::physics::Status::Success;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ExecuteVulkanSearchPhysics(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    uint slot = dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;

    uint8_t* simulationActive =
        reinterpret<uint8_t*>(parameters->simulationActive);
    if (simulationActive[slot] == 0u) return;

    forevervalidator::simulation::VulkanTimelineResult* results =
        reinterpret<forevervalidator::simulation::VulkanTimelineResult*>(
            parameters->timelineResults);
    uint* cancellation = reinterpret<uint*>(parameters->cancellation);
    if (*cancellation != 0u)
    {
        results[slot].status = 4u;
        results[slot].failureTick = searchPush.tickIndex;
        return;
    }

    CudaPackedSceneHeader* scene =
        reinterpret<CudaPackedSceneHeader*>(parameters->scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(
            parameters->configuration);
    uint8_t* stateBytes = reinterpret<uint8_t*>(parameters->candidateStates);
    CudaCandidatePhysicsState* state =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(slot) * parameters->stateWordCount * 4u);
    if (!forevervalidator::simulation::ValidPackedInputs(
            scene, configuration))
    {
        results[slot].status = 1u;
        return;
    }
    if (state->schemaVersion != CudaCandidateState_SchemaVersion)
    {
        results[slot].status = 2u;
        return;
    }

    CudaControlTick* ticks =
        reinterpret<CudaControlTick*>(parameters->candidateTicks);
    CudaControlTick tick = ticks[slot];
    forevervalidator::simulation::ApplyControlAndTimingPrefix(
        *state, tick, false);
    if (state->firstStep == 0u)
        forevervalidator::simulation::cuda::transition::PrepareStep(
            *state, tick, configuration);
    state->vehicle.mobil.absorbContactEnabled = 1u;
    state->vehicle.mobil.physicsUpdatesEnabled =
        (tick.actionFlags & 2u) == 0u;
    for (uint respawn = 0u;
         respawn < tick.respawnAtCheckpointCount; ++respawn)
    {
        if (forevervalidator::simulation::cuda::transition::Respawn(
                *state, configuration))
        {
            ++results[slot].executedRespawnCount;
            ++state->incrementalRespawnCount;
            if (parameters->simulateStunts != 0u)
            {
                CudaCandidateState* fullState =
                    reinterpret<CudaCandidateState*>(state);
                forevervalidator::simulation::cuda::stunts::
                    ApplyRespawnPenalty(fullState->stunts);
            }
        }
    }

    cuda__collision__VulkanCollisionSearchScratch* scratch =
        reinterpret<cuda__collision__VulkanCollisionSearchScratch*>(
            parameters->collisionScratch) + slot;
    forevervalidator::simulation::cuda::physics::Status physicsStatus =
        ExecuteSearchPhysicsStep(
        scene, configuration, *state, scratch);
    if (physicsStatus !=
            forevervalidator::simulation::cuda::physics::Status::Success)
    {
        results[slot].status = 6u;
        results[slot].failureTick = searchPush.tickIndex;
        results[slot].failureDetail = uint(physicsStatus) +
            1000u * uint(scratch->overflowReason) +
            100000u * scratch->collisionCount;
        return;
    }

    if (parameters->simulateStunts != 0u)
    {
        CudaCandidateState* fullState =
            reinterpret<CudaCandidateState*>(state);
        forevervalidator::simulation::cuda::collision::detail::
            CaptureReplacementOverflow(
            *scratch, fullState->collisionReplacementOverflow);
        if (state->stuntsEnabled != 0u)
        {
            forevervalidator::simulation::cuda::stunts::Status stuntStatus =
                forevervalidator::simulation::cuda::stunts::Update(
                    *fullState, tick);
            if (stuntStatus != forevervalidator::simulation::cuda::stunts::
                    Status::Success)
            {
                results[slot].status = 3u;
                results[slot].failureTick = searchPush.tickIndex;
                results[slot].failureDetail = uint(stuntStatus);
                return;
            }
        }
    }
    state->firstStep = 0u;
    ++state->controlCursor;
    ++results[slot].executedTickCount;
    results[slot].status = 0u;
    results[slot].failureTick = UINT32_MAX;
}

}
"""


SEARCH_COMBINED_ENTRY = r"""

namespace forevervalidator::simulation::vulkan_search {

bool SearchScriptedBetter(
    __constref cuda_search_detail__DeviceSample candidate,
    __constref cuda_search_detail__DeviceSample incumbent)
{
    if (candidate.valid == 0u) return false;
    if (incumbent.valid == 0u) return true;
    if (candidate.scriptedObjectiveCount == 0u ||
        candidate.scriptedObjectiveCount != incumbent.scriptedObjectiveCount)
        return false;
    bool better = false;
    for (uint i = 0u; i < candidate.scriptedObjectiveCount; ++i)
    {
        if (candidate.objectiveScores[i] < incumbent.objectiveScores[i])
            return false;
        better = better ||
            candidate.objectiveScores[i] > incumbent.objectiveScores[i];
    }
    return better;
}

[noinline]
void FinalizeVulkanSearchBatch(VulkanSearchParameters* parameters)
{
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    uint* statuses = reinterpret<uint*>(parameters->statuses);
    uint8_t* active =
        reinterpret<uint8_t*>(parameters->activeCandidates);
    uint* mutationCounts =
        reinterpret<uint*>(parameters->mutationCounts);
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    VulkanSearchBatchSummary result = {};
    result.winnerSlot = UINT32_MAX;
    result.status = 0u;

    for (uint slot = 0u; slot < parameters->candidateCount; ++slot)
    {
        if (statuses[slot] == 2u)
            result.status = 4u;
        else if (statuses[slot] == 1u)
            result.status = 3u;
        else if (statuses[slot] == 4u)
            result.status = 6u;
        if (active[slot] != 0u)
            ++result.evaluatedCandidateCount;
        result.totalMutationCount += uint64_t(mutationCounts[slot]);
    }

    CudaSearchEvaluatorConfiguration* evaluator =
        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);
    bool maximize = MaximizesScore(evaluator->kind);
    cuda_search_detail__DeviceSample incumbent = samples[0];
    if (parameters->baseline != 0u)
    {
        if (evaluator->kind == CudaSearchEvaluatorKind_Scripted
                ? SearchScriptedBetter(samples[1], incumbent)
                : samples[1].valid != 0u)
        {
            incumbent = samples[1];
            result.winnerSlot = 0u;
        }
    }
    else
    {
        for (uint slot = 0u; slot < parameters->candidateCount; ++slot)
        {
            cuda_search_detail__DeviceSample sample = samples[slot + 1u];
            if (evaluator->kind == CudaSearchEvaluatorKind_Scripted
                    ? SearchScriptedBetter(sample, incumbent)
                    : SearchStrictlyBetter(sample, incumbent, maximize))
            {
                ++result.mutationImprovementCount;
                incumbent = sample;
                result.winnerSlot = slot;
            }
        }
    }
    result.winner = incumbent;

    if (result.winnerSlot != UINT32_MAX && incumbent.valid != 0u &&
        (evaluator->kind == CudaSearchEvaluatorKind_Scripted ||
         parameters->baseline != 0u ||
         SearchStrictlyBetter(incumbent, samples[0], maximize)))
    {
        uint slot = result.winnerSlot;
        result.winnerEventCount = eventCounts[slot];
        result.winnerMutationCount = mutationCounts[slot];
        CudaSearchInputEvent* source =
            reinterpret<CudaSearchInputEvent*>(parameters->candidateEvents) +
            uint64_t(slot) * parameters->eventCapacity;
        CudaSearchInputEvent* destination =
            reinterpret<CudaSearchInputEvent*>(parameters->winnerEvents);
        for (uint index = 0u; index < result.winnerEventCount; ++index)
            destination[index] = source[index];
        result.bestChanged = 1u;
    }

    VulkanSearchBatchSummary* summary =
        reinterpret<VulkanSearchBatchSummary*>(parameters->winnerSummary);
    *summary = result;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ExecuteVulkanSearchCandidates(
    uint3 dispatchThreadId : SV_DispatchThreadID)
{
    VulkanSearchParameters* parameters =
        reinterpret<VulkanSearchParameters*>(searchPush.parameters);
    if (searchPush.tickIndex == UINT32_MAX)
    {
        if (dispatchThreadId.x == 0u) FinalizeVulkanSearchBatch(parameters);
        return;
    }
    uint slot = searchPush.candidateBase + dispatchThreadId.x;
    if (slot >= parameters->candidateCount) return;

    uint8_t* simulationActive =
        reinterpret<uint8_t*>(parameters->simulationActive);
    if (simulationActive[slot] == 0u) return;

    CudaPackedSceneHeader* scene =
        reinterpret<CudaPackedSceneHeader*>(parameters->scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(
            parameters->configuration);
    uint8_t* stateBytes = reinterpret<uint8_t*>(parameters->candidateStates);
    CudaCandidatePhysicsState* deviceState =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(slot) * parameters->stateWordCount * 4u);
    CudaCandidatePhysicsState localState = *deviceState;
    uint* statuses = reinterpret<uint*>(parameters->statuses);
    cuda_search_detail__DeviceSample* samples =
        reinterpret<cuda_search_detail__DeviceSample*>(
            parameters->candidateBestSamples);
    cuda_search_detail__DeviceSample localBest = samples[slot + 1u];
    if (!forevervalidator::simulation::ValidPackedInputs(
            scene, configuration) ||
        localState.schemaVersion != CudaCandidateState_SchemaVersion)
    {
        statuses[slot] = 4u;
        samples[slot + 1u] = localBest;
        return;
    }

    uint* cancellation = reinterpret<uint*>(parameters->cancellation);
    forevervalidator::simulation::VulkanTimelineResult* results =
        reinterpret<forevervalidator::simulation::VulkanTimelineResult*>(
            parameters->timelineResults);
    cuda__collision__VulkanCollisionSearchScratch* scratch =
        reinterpret<cuda__collision__VulkanCollisionSearchScratch*>(
            parameters->collisionScratch) + slot;
    CudaSearchInputEvent* events =
        reinterpret<CudaSearchInputEvent*>(parameters->candidateEvents) +
        uint64_t(slot) * parameters->eventCapacity;
    uint* eventCounts = reinterpret<uint*>(parameters->eventCounts);
    uint eventCount = eventCounts[slot];
    uint* eventCursors = reinterpret<uint*>(parameters->eventCursors);
    uint eventCursor = eventCursors[slot];
    cuda_search_detail__DeviceControlState* controlStates =
        reinterpret<cuda_search_detail__DeviceControlState*>(
            parameters->controlStates);
    cuda_search_detail__DeviceControlState controlState =
        controlStates[slot];
    CudaControlTick* baselineTicks =
        reinterpret<CudaControlTick*>(parameters->baselineTicks);
    CudaSearchConditionInstruction* condition =
        reinterpret<CudaSearchConditionInstruction*>(parameters->condition);
    CudaSearchEvaluatorConfiguration* evaluator =
        reinterpret<CudaSearchEvaluatorConfiguration*>(parameters->evaluator);
    cuda_search_detail__DeviceSample incumbent = samples[0];
    uint64_t candidateId = parameters->firstCandidateId + slot;
    uint* evaluationIndices =
        reinterpret<uint*>(parameters->evaluationIndices);
    uint evaluationIndex = evaluationIndices[slot];
    uint8_t* evaluatorReportedValues =
        reinterpret<uint8_t*>(parameters->evaluatorReported);
    bool evaluatorReported = evaluatorReportedValues[slot] != 0u;
    bool maximize = MaximizesScore(evaluator->kind);

    uint tickEnd = min(
        parameters->timelineTickCount,
        searchPush.tickIndex + searchPush.tickCount);
    bool completed = false;
    for (uint tickIndex = searchPush.tickIndex;
         tickIndex < tickEnd; ++tickIndex)
    {
        if ((tickIndex & 63u) == 0u && *cancellation != 0u)
        {
            statuses[slot] = 2u;
            samples[slot + 1u] = localBest;
            *deviceState = localState;
            return;
        }

        int64_t publicTime = parameters->branchTimeMs +
            int64_t(tickIndex + 1u) * parameters->tickDurationMs;
        int64_t suffixTime = publicTime - parameters->mutableFromTimeMs;
        while (eventCursor < eventCount &&
               events[eventCursor].timeMs <= suffixTime)
        {
            forevervalidator::simulation::cuda_search_detail::
                ApplyControlEvent(
                    controlState, events[eventCursor],
                    parameters->mutableFromTimeMs);
            ++eventCursor;
        }

        CudaControlTick tick = baselineTicks[tickIndex];
        tick.observe = false;
        tick.controls = forevervalidator::simulation::cuda_search_detail::
            ControlsFromState(controlState);
        tick.stuntsInput = forevervalidator::simulation::cuda_search_detail::
            StuntsFromState(controlState, parameters->prestartDurationMs);
        GmVec3 previousPosition = localState.body.current.position;

        forevervalidator::simulation::ApplyControlAndTimingPrefix(
            localState, tick, false);
        if (localState.firstStep == 0u)
            forevervalidator::simulation::cuda::transition::PrepareStep(
                localState, tick, configuration);
        localState.vehicle.mobil.absorbContactEnabled = 1u;
        localState.vehicle.mobil.physicsUpdatesEnabled =
            (tick.actionFlags & 2u) == 0u;
        for (uint respawn = 0u;
             respawn < tick.respawnAtCheckpointCount; ++respawn)
        {
            if (forevervalidator::simulation::cuda::transition::Respawn(
                    localState, configuration))
            {
                ++results[slot].executedRespawnCount;
                ++localState.incrementalRespawnCount;
                if (parameters->simulateStunts != 0u)
                {
                    *deviceState = localState;
                    CudaCandidateState* fullState =
                        reinterpret<CudaCandidateState*>(deviceState);
                    forevervalidator::simulation::cuda::stunts::
                        ApplyRespawnPenalty(fullState->stunts);
                }
            }
        }

        forevervalidator::simulation::cuda::physics::Status physicsStatus =
            ExecuteSearchPhysicsStep(
                scene, configuration, localState, scratch);
        if (physicsStatus !=
                forevervalidator::simulation::cuda::physics::Status::Success)
        {
            results[slot].status = 6u;
            results[slot].failureTick = tickIndex;
            results[slot].failureDetail = uint(physicsStatus) +
                1000u * uint(scratch->overflowReason) +
                100000u * scratch->collisionCount;
            statuses[slot] = 4u;
            samples[slot + 1u] = localBest;
            *deviceState = localState;
            return;
        }

        if (parameters->simulateStunts != 0u)
        {
            CudaCandidateState* fullState =
                reinterpret<CudaCandidateState*>(deviceState);
            forevervalidator::simulation::cuda::collision::detail::
                CaptureReplacementOverflow(
                    *scratch, fullState->collisionReplacementOverflow);
            if (localState.stuntsEnabled != 0u)
            {
                *deviceState = localState;
                forevervalidator::simulation::cuda::stunts::Status stuntStatus =
                    forevervalidator::simulation::cuda::stunts::Update(
                        *fullState, tick);
                if (stuntStatus !=
                        forevervalidator::simulation::cuda::stunts::
                            Status::Success)
                {
                    statuses[slot] = 1u;
                    samples[slot + 1u] = localBest;
                    return;
                }
                localState = *deviceState;
            }
        }
        localState.firstStep = 0u;
        ++localState.controlCursor;
        ++results[slot].executedTickCount;
        results[slot].status = 0u;
        results[slot].failureTick = UINT32_MAX;

        if (publicTime < parameters->evaluationStartTimeMs) continue;
        if (parameters->conditionInstructionCount != 0u &&
            !EvaluateCondition(
                condition, parameters->conditionInstructionCount,
                localState,
                parameters->baseline != 0u ? 0u : candidateId + 1u,
                parameters->lastImprovementTimeSeconds,
                parameters->lastRestartTimeSeconds,
                parameters->currentTimeSeconds))
        {
            ++evaluationIndex;
            if (localState.race.progress.raceCompleted != 0u)
            {
                completed = true;
                break;
            }
            continue;
        }

        if (evaluator->kind == CudaSearchEvaluatorKind_Scripted)
        {
            if (UpdateScriptedSample(
                    *evaluator, localState,
                    parameters->baseline != 0u ? 0u : candidateId + 1u,
                    parameters->lastImprovementTimeSeconds,
                    parameters->lastRestartTimeSeconds,
                    parameters->currentTimeSeconds,
                    double(publicTime), localBest))
            {
                localBest.candidateId = candidateId;
                localBest.candidateSlot = slot;
                localBest.evaluationTick = evaluationIndex;
                localBest.eventCount = eventCount;
                localBest.logicalOrder = 1u + uint64_t(slot) *
                    parameters->evaluationTickCount + evaluationIndex;
                localBest.mutation = parameters->baseline == 0u ? 1u : 0u;
            }
            ++evaluationIndex;
            if (localState.race.progress.raceCompleted != 0u)
            {
                completed = true;
                break;
            }
            continue;
        }

        uint stuntsScore = 0u;
        if (parameters->simulateStunts != 0u)
        {
            CudaCandidateState* fullState =
                reinterpret<CudaCandidateState*>(deviceState);
            stuntsScore = fullState->stunts.stuntsScore;
        }
        cuda_search_detail__DeviceSample sample = EvaluateState(
            *evaluator, localState, previousPosition,
            double(publicTime - parameters->tickDurationMs),
            double(publicTime), stuntsScore, &evaluatorReported);
        sample.candidateId = candidateId;
        sample.candidateSlot = slot;
        sample.evaluationTick = evaluationIndex;
        sample.eventCount = eventCount;
        sample.logicalOrder = 1u + uint64_t(slot) *
            parameters->evaluationTickCount + evaluationIndex;
        sample.mutation = parameters->baseline == 0u ? 1u : 0u;
        if (SearchStrictlyBetter(sample, localBest, maximize))
            localBest = sample;
        ++evaluationIndex;

        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
            evaluatorReported)
        {
            completed = true;
            break;
        }
        if (evaluator->kind == CudaSearchEvaluatorKind_FinishTime &&
            incumbent.preciseFinish != 0u &&
            double(publicTime) >= incumbent.timeMs)
        {
            localBest = {};
            completed = true;
            break;
        }
    }
    if (completed) simulationActive[slot] = 0u;
    controlStates[slot] = controlState;
    eventCursors[slot] = eventCursor;
    evaluationIndices[slot] = evaluationIndex;
    evaluatorReportedValues[slot] = evaluatorReported ? 1u : 0u;
    samples[slot + 1u] = localBest;
    *deviceState = localState;
}

}
"""




MUTATION_HELPERS = r"""
int64_t RoundAwayFromZero(double value)
{
    return value >= 0.0
        ? int64_t(floor(value + 0.5))
        : int64_t(ceil(value - 0.5));
}

int64_t RandomModifierTime(
    __constref CudaSearchModifierConfiguration modifier,
    uint tickDurationMs,
    inout DeviceMt19937 random)
{
    return random.UniformS64(
        modifier.window.minimumTimeMs / tickDurationMs,
        modifier.window.maximumTimeMs / tickDurationMs) * tickDurationMs;
}

int64_t RandomModifierHold(
    int64_t maximum,
    uint tickDurationMs,
    inout DeviceMt19937 random)
{
    return maximum <= 0 ? 0 :
        random.UniformS64(0, maximum / tickDurationMs) * tickDurationMs;
}

bool InsertSwitchChannel(
    __constref CudaSearchChannel channel,
    uint action,
    __constref CudaSearchModifierConfiguration modifier,
    uint tickDurationMs,
    __constref cuda_search_detail__DeviceControlState initialControls,
    CudaSearchInputEvent* events,
    uint* eventCount,
    uint eventCapacity,
    CudaSearchInputEvent* passBaseline,
    uint passBaselineCount,
    bool passBaselineCanonical,
    inout DeviceMt19937 random)
{
    if (channel.enabled == 0u) return true;
    uint count = random.UniformU32(
        channel.minimumCount, channel.maximumCount);
    for (uint index = 0u; index < count; ++index)
    {
        int64_t start = RandomModifierTime(
            modifier, tickDurationMs, random);
        int64_t end = start + RandomModifierHold(
            channel.maximumHoldMs, tickDurationMs, random);
        if (end > modifier.window.maximumTimeMs)
            end = modifier.window.maximumTimeMs;
        bool previous = forevervalidator::simulation::
            cuda_search_modifier_detail::RemoveActionRangeAndReadState(
                events, eventCount, action, 1u, start, end,
                action == 1u ? initialControls.accelerate
                             : initialControls.brake) != 0;
        if (!PushEvent(
                events, eventCount, eventCapacity,
                SwitchEvent(start, action, !previous)))
            return false;
        if (end > start &&
            !PushEvent(
                events, eventCount, eventCapacity,
                SwitchEvent(
                    end, action,
                    SwitchStateAt(
                        passBaseline, passBaselineCount, action, end,
                        action == 1u ? initialControls.accelerate
                                     : initialControls.brake,
                        passBaselineCanonical))))
            return false;
    }
    return true;
}

void DeleteChannel(
    __constref CudaSearchChannel channel,
    uint kind,
    __constref CudaSearchModifierConfiguration modifier,
    CudaSearchInputEvent* events,
    uint* eventCount,
    uint* eligible,
    bool normalized,
    inout DeviceMt19937 random)
{
    if (channel.enabled == 0u) return;
    uint requested = random.UniformU32(0u, channel.maximumCount);
    if (requested == 0u) return;
    uint initialEligibleCount = forevervalidator::simulation::
        cuda_search_modifier_detail::CollectDeletionEligible(
            events, *eventCount, eligible,
            modifier.window.minimumTimeMs,
            modifier.window.maximumTimeMs, kind, normalized);
    uint eligibleCount = initialEligibleCount;
    for (uint removal = 0u; removal < requested; ++removal)
    {
        if (eligibleCount == 0u) break;
        forevervalidator::simulation::cuda_search_modifier_detail::
            SelectDeletionRank(
                eligible, &eligibleCount,
                random.UniformU32(0u, eligibleCount - 1u));
    }
    if (eligibleCount != initialEligibleCount)
        forevervalidator::simulation::cuda_search_modifier_detail::
            CompactSelectedDeletionTail(
                events, eventCount, eligible,
                eligibleCount, initialEligibleCount);
}

"""


def rewrite_executor_slang(text: str) -> str:
    text = re.sub(
        r"\s*let randomTime = \[&\]\(\) \{.*?\n\s*\};",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"\s*let randomHold = \[&\]\(int64_t maximum\) \{.*?\n\s*\};",
        "",
        text,
        flags=re.DOTALL,
    )
    text = text.replace(
        "randomTime()",
        "RandomModifierTime(modifier, tickDurationMs, random)",
    )
    text = text.replace(
        "randomHold(\n                                        modifier.steering.maximumHoldMs)",
        "RandomModifierHold(modifier.steering.maximumHoldMs, "
        "tickDurationMs, random)",
    )
    insertion = text.index("        let insertSwitch =")
    insertion_end = text.index("        if (!insertSwitch", insertion)
    text = text[:insertion] + text[insertion_end:]
    text = text.replace(
        "if (!insertSwitch(modifier.accelerate, 1u) ||\n"
        "            !insertSwitch(modifier.brake, 3u))",
        "if (!InsertSwitchChannel(\n"
        "                modifier.accelerate, 1u, modifier, tickDurationMs,\n"
        "                initialControls, events, eventCount, eventCapacity,\n"
        "                passBaseline, passBaselineCount,\n"
        "                passBaselineCanonical, random) ||\n"
        "            !InsertSwitchChannel(\n"
        "                modifier.brake, 3u, modifier, tickDurationMs,\n"
        "                initialControls, events, eventCount, eventCapacity,\n"
        "                passBaseline, passBaselineCount,\n"
        "                passBaselineCanonical, random))",
    )
    deletion = text.index("        let deleteChannel =")
    deletion_end = text.index(
        "        deleteChannel(modifier.steering", deletion
    )
    text = text[:deletion] + text[deletion_end:]
    text = text.replace(
        "deleteChannel(modifier.steering, 0u);\n"
        "        deleteChannel(modifier.accelerate, 1u);\n"
        "        deleteChannel(modifier.brake, 2u);",
        "DeleteChannel(modifier.steering, 0u, modifier, events, "
        "eventCount, eligible, *normalized, random);\n"
        "        DeleteChannel(modifier.accelerate, 1u, modifier, events, "
        "eventCount, eligible, *normalized, random);\n"
        "        DeleteChannel(modifier.brake, 2u, modifier, events, "
        "eventCount, eligible, *normalized, random);",
    )
    text = text.replace(
        " bool ApplyModifier(", MUTATION_HELPERS + "\nbool ApplyModifier("
    )
    return text


def rewrite_evaluation_slang(text: str) -> str:
    text = text.replace("bool vector = false;", "uint vector = 0u;")
    packed_begin = text.index("__device__ bool ValidPackedInputs(")
    packed_end = text.index("__device__ bool ContainsVolume(", packed_begin)
    text = text[:packed_begin] + text[packed_end:]
    text = text.replace(
        "const double fromValues[3]{", "const double fromValues[3] = {")
    text = text.replace(
        "const double toValues[3]{", "const double toValues[3] = {")
    text = text.replace(
        "atan2(sinYaw, cosYaw)",
        "forevervalidator::simulation::cuda::exact::detail::Atan2("
        "sinYaw, cosYaw)",
    )
    text = text.replace(
        "atan2(sinRoll, cosRoll)",
        "forevervalidator::simulation::cuda::exact::detail::Atan2("
        "sinRoll, cosRoll)",
    )
    text = text.replace(
        "asin(sinPitch)",
        "forevervalidator::simulation::cuda::exact::detail::Atan2("
        "sinPitch, sqrt((1.0 - sinPitch) * (1.0 + sinPitch)))")
    text = text.replace(
        "acos(dot)",
        "forevervalidator::simulation::cuda::exact::detail::Atan2("
        "sqrt((1.0 - dot) * (1.0 + dot)), dot)")
    dot_begin = text.index("    const auto dot =")
    dot_end = text.index("    return {", dot_begin)
    text = text[:dot_begin] + text[dot_end:]
    text = text.replace(
        "dot(body.linearSpeed, body.rotation.basisX)",
        "body.linearSpeed.x * body.rotation.basisX.x + "
        "body.linearSpeed.y * body.rotation.basisX.y + "
        "body.linearSpeed.z * body.rotation.basisX.z",
    )
    text = text.replace(
        "dot(body.linearSpeed, body.rotation.basisY)",
        "body.linearSpeed.x * body.rotation.basisY.x + "
        "body.linearSpeed.y * body.rotation.basisY.y + "
        "body.linearSpeed.z * body.rotation.basisY.z",
    )
    text = text.replace(
        "dot(body.linearSpeed, body.rotation.basisZ)",
        "body.linearSpeed.x * body.rotation.basisZ.x + "
        "body.linearSpeed.y * body.rotation.basisZ.y + "
        "body.linearSpeed.z * body.rotation.basisZ.z",
    )
    vector_begin = text.index("    const auto vector =")
    vector_end = text.index("    switch (source)", vector_begin)
    text = text[:vector_begin] + text[vector_end:]
    text = text.replace("vector(", "ConditionVector(")
    scripted_begin = text.index(" bool UpdateScriptedSample(")
    scripted_end = text.rfind(
        "\n", scripted_begin, text.index(" EvaluateState(", scripted_begin))
    scripted = text[scripted_begin:scripted_end]
    scripted = scripted.replace(
        "DeviceSample *sample",
        "inout DeviceSample sample")
    scripted = scripted.replace("sample->", "sample.")
    text = text[:scripted_begin] + scripted + text[scripted_end:]
    text = text.replace(
        "struct DeviceConditionValue {",
        "struct DeviceConditionValue {",
    )
    helper_position = text.index("__device__ DeviceConditionValue ConditionSource(")
    helper = """__device__ DeviceConditionValue ConditionVector(
        __constref GmVec3 value) {
    return {value.x, value.y, value.z, true};
}

"""
    return text[:helper_position] + helper + text[helper_position:]


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    cuda_root = source_root / "src/simulation/backends/cuda"
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    translator = ReferenceTranslator(
        source_root, SEARCH_FILES, include_search_helpers=True)

    device_parts = [
        args.types.read_text(encoding="utf-8"), PRELUDE,
        SEARCH_SCRATCH_TYPE,
    ]
    for filename in DEVICE_FILES:
        source_path = cuda_root / filename
        source = translator.translate(
            source_path, source_path.read_text(encoding="utf-8"))
        file_mapping = dict(mapping)
        if filename in ("cuda_collision.cuh", "cuda_collision_response.cuh"):
            prefix = "forevervalidator::simulation::cuda::collision::"
            for key, value in mapping.items():
                if key.startswith(prefix):
                    file_mapping[key.removeprefix(prefix)] = value
        translated_device = translate_common(
            source, file_mapping, SEARCH_TEMPLATE_VALUES,
            SEARCH_CONCRETE_TYPE_PARAMETERS)
        translated_device = translated_device.replace(
            "cuda__collision__CudaCollisionSearchScratch",
            "cuda__collision__VulkanCollisionSearchScratch")
        if filename == "cuda_collision.cuh":
            insertion = translated_device.index(
                "static const float SphereNormalAlignment")
            insertion = translated_device.index(";", insertion) + 1
            translated_device = (
                translated_device[:insertion] +
                SEARCH_SCRATCH_ACCESSORS +
                translated_device[insertion:]
            )
            translated_device = rewrite_search_collision_slang(
                translated_device)
            translated_device = isolate_search_mesh_cell_cache(
                translated_device)
        elif filename == "cuda_dynamics.cuh":
            translated_device = rewrite_search_dynamics_slang(
                translated_device)
        elif filename in (
            "cuda_collision_response.cuh", "cuda_physics_step.cuh"
        ):
            translated_device = rewrite_search_collision_references(
                translated_device)
        if filename == "cuda_physics_step.cuh":
            translated_device = translated_device.replace(
                "inline Status Step<", "[noinline]\ninline Status Step<")
        if filename == "cuda_finish_time_refinement.cuh":
            translated_device = (
                "#if 0\n" + translated_device + "\n#endif"
            )
        device_parts.append(
            f"\n// Search specialization translated from {filename}.\n")
        device_parts.append(translated_device)
    device_parts.append(SEARCH_PHYSICS_SUPPORT)

    modifier_path = cuda_root / "cuda_modifier_event_ops.cuh"
    modifier_source = translator.translate(
        modifier_path, modifier_path.read_text(encoding="utf-8"))
    modifier_source = between(
        modifier_source,
        "namespace forevervalidator::simulation::cuda_search_modifier_detail {",
        "#undef FOREVERVALIDATOR_CUDA_HD",
    )
    modifier_source += "\n}\n"

    candidate_path = cuda_root / "cuda_candidate_events.cuh"
    candidate_source = translator.translate(
        candidate_path, candidate_path.read_text(encoding="utf-8"))
    candidate_source = between(
        candidate_source,
        "FOREVERVALIDATOR_CANDIDATE_HD inline std::int32_t SaturateValue(",
        "FOREVERVALIDATOR_CANDIDATE_HD inline bool Materialize(",
    )
    candidate_source = re.sub(r"\bEvent\b", "CudaSearchInputEvent", candidate_source)
    candidate_source = (
        "namespace forevervalidator::simulation::cuda::candidate_events {\n"
        + candidate_source
        + "\n}\n"
    )

    branch_path = cuda_root / "cuda_search_branch_state.cuh"
    branch_source = translator.translate(
        branch_path, branch_path.read_text(encoding="utf-8"))
    branch_source = between(
        branch_source,
        "FOREVERVALIDATOR_BRANCH_HD inline int StuntActionIndex(",
        "struct SearchInputPartition {",
    )
    branch_source = (
        "namespace forevervalidator::simulation::cuda_search_detail {\n"
        + branch_source
        + "\n}\n"
    )

    executor_path = cuda_root / "cuda_search_executor.cu"
    executor_source = translator.translate(
        executor_path, executor_path.read_text(encoding="utf-8"))
    executor_source = between(
        executor_source,
        "__device__ bool IsAnalog(",
        "__device__ bool SparseExistingEventEligible(",
    )
    cursor_begin = executor_source.index(
        "__device__ CudaSearchInputEvent CandidateInputAt("
    )
    cursor_end = executor_source.index("class DeviceMt19937", cursor_begin)
    executor_source = (
        executor_source[:cursor_begin] + executor_source[cursor_end:]
    )
    edit_begin = executor_source.index("__device__ bool EncodeCandidateEdits(")
    edit_end = executor_source.index(
        "__device__ std::int32_t SteeringStateAt(", edit_begin
    )
    executor_source = executor_source[:edit_begin] + executor_source[edit_end:]
    executor_source = executor_source.replace(
        "__device__ DeviceMt19937(std::uint32_t *stateWords,\n"
        "                             std::uint32_t slot,\n"
        "                             std::uint32_t stride)\n"
        "        : stateWords_(stateWords), slot_(slot), stride_(stride) {}",
        "__device__ void Initialize(std::uint32_t *stateWords,\n"
        "                           std::uint32_t slot,\n"
        "                           std::uint32_t stride) {\n"
        "        stateWords_ = stateWords; slot_ = slot; stride_ = stride;\n"
        "        cursor_ = 624u;\n"
        "    }",
    )
    executor_source = re.sub(r"\bState\(", "RandomWord(", executor_source)
    executor_source = (
        "namespace forevervalidator::simulation::vulkan_search {\n"
        "using namespace forevervalidator::simulation::cuda_search_detail;\n"
        + executor_source
        + "\n}\n"
    )

    evaluation_source = translator.translate(
        executor_path, executor_path.read_text(encoding="utf-8"))
    evaluation_source = between(
        evaluation_source,
        "__device__ void ApplyControlPrefix(",
        "__global__ void SeedCandidateBestSamplesKernel(",
    )
    evaluation_source = rewrite_evaluation_slang(evaluation_source)
    evaluation_source = (
        "namespace forevervalidator::simulation::vulkan_search {\n"
        "using namespace forevervalidator::simulation::cuda_search_detail;\n"
        + evaluation_source
        + "\n}\n"
    )

    translated = []
    for source_index, source in enumerate((
        modifier_source,
        candidate_source,
        branch_source,
        executor_source,
        evaluation_source,
    )):
        text = translate_common(
            source, mapping, SEARCH_TEMPLATE_VALUES,
            SEARCH_CONCRETE_TYPE_PARAMETERS)
        text = text.replace("FOREVERVALIDATOR_CUDA_HD", "")
        text = text.replace("FOREVERVALIDATOR_CANDIDATE_HD", "")
        text = text.replace("FOREVERVALIDATOR_BRANCH_HD", "")
        text = text.replace("INT64_MIN", "(int64_t(-9223372036854775807) - 1)")
        text = text.replace(
            "modifier_ops::",
            "forevervalidator::simulation::cuda_search_modifier_detail::",
        )
        text = re.sub(
            r"uint\s+&RandomWord\(uint index\)\s*\{\s*"
            r"return stateWords_\[\s*"
            r"uint64_t\(index\) \* stride_ \+ slot_\];\s*\}",
            "uint* RandomWordPointer(uint index) {\n"
            "        return &(stateWords_[uint64_t(index) * stride_ + slot_]);\n"
            "    }",
            text,
        )
        if source_index == 2:
            text = re.sub(
                r"\s*uint \*lastChangeTimeMs =\s*"
                r"reinterpret<uint \*>\(&result\.lastChangeTimeMs\);",
                "",
                text,
            )
            text = text.replace(
                "lastChangeTimeMs[index]", "result.lastChangeTimeMs[index]")
        text = text.replace(
            "struct DeviceMt19937 {",
            "#define RandomWord(index) (*RandomWordPointer(index))\n"
            "struct DeviceMt19937 {",
        )
        if source_index == 3:
            text = rewrite_executor_slang(text)
            text = text.replace("const uint seeds[4]{", "const uint seeds[4] = {")
            text = text.replace(
                "     uint64_t UniformUnsigned(uint64_t minimum,",
                """     [mutating]
     uint64_t UniformFromZeroThroughU32(uint64_t maximum) {
        if (maximum == UINT32_MAX) return Next();
        const uint extendedRange = uint(maximum + 1u);
        uint64_t product = uint64_t(Next()) * extendedRange;
        uint low = uint(product);
        if (low < extendedRange) {
            const uint threshold = uint(-extendedRange) % extendedRange;
            while (low < threshold) {
                product = uint64_t(Next()) * extendedRange;
                low = uint(product);
            }
        }
        return product >> 32u;
     }

     uint64_t UniformUnsigned(uint64_t minimum,""",
            )
            text = text.replace(
                "const uint64_t high = UniformUnsigned(\n"
                "                        0u, range / generatorRange);",
                "const uint64_t high = UniformFromZeroThroughU32(\n"
                "                        range / generatorRange);",
            )
            text = text.replace(
                "llround(double(amplitude) *\n"
                "                                smoothWeights[weightIndex])",
                "RoundAwayFromZero(double(amplitude) *\n"
                "                                smoothWeights[weightIndex])",
            )
            for method in (
                "Initialize", "Seed", "Next", "UniformUnsigned",
                "UniformU32", "UniformS32", "UniformS64", "Twist",
            ):
                text = re.sub(
                    rf"(\n\s*)(void|uint|uint64_t|int|int64_t) {method}\(",
                    rf"\1[mutating]\n\1\2 {method}(",
                    text,
                )
        if source_index == 4:
            text = text.replace(
                "cuda_search_detail__DeviceSample result;\n"
                "    result.timeMs = currentTimeMs;",
                "cuda_search_detail__DeviceSample result;\n"
                "    result.score = 0.0;\n"
                "    result.timeMs = 0.0;\n"
                "    result.detail0 = 0.0;\n"
                "    result.detail1 = 0.0;\n"
                "    result.candidateId = 0u;\n"
                "    result.logicalOrder = 0xffffffffffffffffull;\n"
                "    result.candidateSlot = 0xffffffffu;\n"
                "    result.evaluationTick = 0u;\n"
                "    result.eventCount = 0xffffffffu;\n"
                "    result.valid = false;\n"
                "    result.mutation = false;\n"
                "    result.preciseFinish = false;\n"
                "    result.timeMs = currentTimeMs;",
            )
        translated.append(text)

    output = "\n".join(device_parts)
    output += (
        "\n" + "\n".join(translated) + SEARCH_ENTRY +
        SEARCH_STAGED_ENTRIES + SEARCH_PHYSICS_ENTRY + SEARCH_COMBINED_ENTRY
    )
    output = eliminate_local_address_aliases(output)
    if args.specialization == "steady-velocity":
        output = specialize_steady_velocity(output, args.handling)
    if args.split_phases:
        if (args.specialization, args.handling) != ("steady-velocity", "water"):
            raise ValueError(
                "split phases require the steady-velocity water specialization"
            )
        output = apply_unified_patch(
            output,
            Path(__file__).with_name(
                "vulkan_steady_velocity_split_phases.patch"
            ),
        )
    # Slang does not honor C++ default member initializers for this translated
    # local. A non-scripted sample must never publish an undefined count.
    initializer = (
        "    result.preciseFinish = false;\n"
        "    result.timeMs = currentTimeMs;"
    )
    if initializer not in output:
        raise RuntimeError("sample initializer translation was not found")
    output = output.replace(
        initializer,
        "    result.preciseFinish = false;\n"
        "    result.scriptedObjectiveCount = 0u;\n"
        "    result.timeMs = currentTimeMs;",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
