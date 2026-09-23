#!/usr/bin/env python3
"""Translate the certified CUDA device implementation into Slang.

This deliberately consumes only device-side source.  CUDA remains an oracle,
while Vulkan executes the same ordered arithmetic and control flow after a
small, deterministic syntax translation.  The generated Slang file is checked
in so release consumers do not need Python or Clang.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path

from clang import cindex


DEVICE_FILES = (
    "cuda_exact_math.cuh",
    "cuda_memory.cuh",
    "cuda_tuning.cuh",
    "cuda_collision.cuh",
    "cuda_dynamics.cuh",
    "cuda_environment.cuh",
    "cuda_race.cuh",
    "cuda_collision_response.cuh",
    "cuda_vehicle_after_contacts.cuh",
    "cuda_vehicle_forces.cuh",
    "cuda_vehicle_powertrain.cuh",
    "cuda_vehicle_wheels.cuh",
    "cuda_physics_step.cuh",
    "cuda_finish_time_origin.cuh",
    "cuda_finish_time_refinement.cuh",
    "cuda_stunts.cuh",
    "cuda_vehicle_transitions.cuh",
)


class ReferenceTranslator:
    """Lower the C++ reference subset to Slang parameters and pointers.

    Slang has first-class ``inout`` parameters, but local ``Ref<T>`` values
    currently ICE when they refer into a physical storage buffer.  Clang gives
    us exact declarations and uses, allowing mutable local/return references to
    become pointers without brittle whole-file regular expressions.  Const
    locals and const returns become values; all of the GPU ABI records involved
    here are small enough that Slang inlines those copies.
    """

    def __init__(self, source_root: Path,
                 device_files: tuple[str, ...] = DEVICE_FILES,
                 include_search_helpers: bool = False):
        self.source_root = source_root
        self.include_search_helpers = include_search_helpers
        probe = str(source_root / "vulkan_slang_reference_probe.cu")
        source = "\n".join(
            f'#include "simulation/backends/cuda/{filename}"'
            for filename in device_files
        )
        resource_dir = os.popen("clang -print-resource-dir").read().strip()
        arguments = [
            "-x", "cuda", "--cuda-host-only", "-std=c++17",
            "-nocudainc", "-nocudalib", "-ferror-limit=0",
            f"-resource-dir={resource_dir}",
            f"-I{source_root / 'src'}", f"-I{source_root / 'include'}",
            "-I/opt/cuda/include",
            "-D__device__=", "-D__host__=", "-D__forceinline__=",
            "-D__global__=", "-D__constant__=",
        ]
        self.tu = cindex.Index.create().parse(
            probe, args=arguments, unsaved_files=[(probe, source)])
        self.cursors_by_file: dict[str, list[tuple[cindex.Cursor, cindex.Cursor | None]]] = {}
        self._walk(self.tu.cursor, None)

    def _walk(self, cursor: cindex.Cursor, parent: cindex.Cursor | None) -> None:
        if cursor.location.file:
            path = str(cursor.location.file)
            if path.startswith(str(self.source_root)):
                self.cursors_by_file.setdefault(path, []).append((cursor, parent))
        for child in cursor.get_children():
            self._walk(child, cursor)

    @staticmethod
    def _is_const_reference(type_: cindex.Type) -> bool:
        return type_.get_pointee().is_const_qualified()

    @staticmethod
    def _ampersand(text: str, cursor: cindex.Cursor) -> int:
        start = cursor.extent.start.offset
        end = cursor.location.offset + len(cursor.spelling)
        position = text.rfind("&", start, end)
        if position < 0:
            raise RuntimeError(f"reference token not found for {cursor.spelling}")
        return position

    @staticmethod
    def _add(edits: dict[tuple[int, int], str], start: int, end: int,
             replacement: str) -> None:
        key = (start, end)
        previous = edits.get(key)
        if previous is not None and previous != replacement:
            raise RuntimeError(f"conflicting source edits at {key}")
        edits[key] = replacement

    @staticmethod
    def _inside(cursor: cindex.Cursor, outer: cindex.Cursor) -> bool:
        return (outer.extent.start.offset <= cursor.extent.start.offset and
                cursor.extent.end.offset <= outer.extent.end.offset)

    def translate(self, path: Path, text: str) -> str:
        entries = self.cursors_by_file.get(str(path.resolve()), [])
        edits: dict[tuple[int, int], str] = {}
        mutable_locals: list[cindex.Cursor] = []
        mutable_auto_locals: list[cindex.Cursor] = []
        pointer_return_functions: set[str] = set()
        value_return_functions: set[str] = set()

        search_only_names = (
            (
                "CudaCollisionSearchTile",
                "CudaCollisionSearchReference",
                "CudaCollisionSearchConstReference",
                "CudaCollisionSearchVectorReference",
                "CudaCollisionSearchConstVectorReference",
            )
            if self.include_search_helpers
            else (
                "CudaCollisionSearch", "CudaCollisionSurfaceHit",
                "CudaCollisionMeshRange",
            )
        )
        removed_ranges: list[tuple[int, int]] = []
        for cursor, _ in entries:
            if cursor.kind not in (cindex.CursorKind.FUNCTION_DECL,
                                   cindex.CursorKind.FUNCTION_TEMPLATE):
                continue
            opening_brace = text.find("{", cursor.extent.start.offset,
                                      cursor.extent.end.offset)
            signature_end = opening_brace if opening_brace >= 0 else cursor.extent.end.offset
            signature = text[cursor.extent.start.offset:signature_end]
            if (any(name in signature for name in search_only_names) or
                    cursor.spelling == "EllipsoidMeshPairCached" or
                    (not self.include_search_helpers and
                     cursor.spelling == "CachedShapeWorldPose")):
                removed_ranges.append((cursor.extent.start.offset,
                                       cursor.extent.end.offset))

        def removed(cursor: cindex.Cursor) -> bool:
            return any(start <= cursor.extent.start.offset and
                       cursor.extent.end.offset <= end
                       for start, end in removed_ranges)

        for start, end in removed_ranges:
            self._add(edits, start, end, "")

        for cursor, _ in entries:
            if removed(cursor):
                continue
            if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                               cindex.CursorKind.FUNCTION_TEMPLATE):
                if cursor.result_type.kind == cindex.TypeKind.LVALUEREFERENCE:
                    ampersand = self._ampersand(text, cursor)
                    if self._is_const_reference(cursor.result_type):
                        self._add(edits, ampersand, ampersand + 1, "")
                        value_return_functions.add(cursor.get_usr())
                    else:
                        self._add(edits, ampersand, ampersand + 1, "*")
                        pointer_return_functions.add(cursor.get_usr())

        for cursor, _ in entries:
            if removed(cursor):
                continue
            if cursor.type.kind != cindex.TypeKind.LVALUEREFERENCE:
                continue
            if cursor.kind == cindex.CursorKind.PARM_DECL:
                # Parenthesized array references need reconstruction rather
                # than just deleting '&'.
                pointee = cursor.type.get_pointee()
                if pointee.kind == cindex.TypeKind.CONSTANTARRAY:
                    element = pointee.element_type.spelling.removeprefix("const ")
                    qualifier = "__constref" if pointee.element_type.is_const_qualified() else "inout"
                    replacement = f"{qualifier} {element} {cursor.spelling}[{pointee.element_count}]"
                    self._add(edits, cursor.extent.start.offset,
                              cursor.extent.end.offset, replacement)
                else:
                    ampersand = self._ampersand(text, cursor)
                    self._add(edits, ampersand, ampersand + 1, "")
                    qualifier = "__constref " if self._is_const_reference(cursor.type) else "inout "
                    self._add(edits, cursor.extent.start.offset,
                              cursor.extent.start.offset, qualifier)
            elif cursor.kind == cindex.CursorKind.VAR_DECL:
                pointee = cursor.type.get_pointee()
                type_name = pointee.spelling.removeprefix("const ")
                declaration_end = cursor.location.offset + len(cursor.spelling)
                if self._is_const_reference(cursor.type):
                    self._add(edits, cursor.extent.start.offset,
                              declaration_end,
                              f"{type_name} {cursor.spelling}")
                else:
                    self._add(edits, cursor.extent.start.offset,
                              declaration_end,
                              f"{type_name} *{cursor.spelling}")
                    mutable_locals.append(cursor)
                    # A ref-returning call is already a pointer after lowering;
                    # other lvalue initializers need their address taken.
                    expressions = [child for child in cursor.get_children()
                                   if child.kind not in (cindex.CursorKind.TYPE_REF,
                                                         cindex.CursorKind.TEMPLATE_REF,
                                                         cindex.CursorKind.NAMESPACE_REF)]
                    initializer = expressions[-1] if expressions else None
                    referenced = initializer.referenced if initializer is not None else None
                    if not (initializer is not None and
                            initializer.kind == cindex.CursorKind.CALL_EXPR and
                            referenced is not None and
                            referenced.get_usr() in pointer_return_functions):
                        if initializer is not None:
                            self._add(edits, initializer.extent.start.offset,
                                      initializer.extent.start.offset, "&")
            elif cursor.kind == cindex.CursorKind.FIELD_DECL:
                ampersand = self._ampersand(text, cursor)
                self._add(edits, ampersand, ampersand + 1, "*")

        # Slang considers inout and __constref overloads equally viable for an
        # lvalue. Give const-ref-returning accessors distinct internal names.
        for cursor, _ in entries:
            if removed(cursor):
                continue
            if (cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                                cindex.CursorKind.FUNCTION_TEMPLATE) and
                    cursor.get_usr() in value_return_functions):
                self._add(edits, cursor.location.offset,
                          cursor.location.offset + len(cursor.spelling),
                          cursor.spelling + "Const")
            elif (cursor.kind == cindex.CursorKind.DECL_REF_EXPR and
                  cursor.referenced is not None and
                  cursor.referenced.get_usr() in value_return_functions):
                self._add(edits, cursor.extent.start.offset,
                          cursor.extent.end.offset,
                          cursor.spelling + "Const")

        # Clang exposes decltype(auto) declarations as TypeKind.AUTO even when
        # their initializer is a mutable reference-returning accessor.
        for cursor, _ in entries:
            if removed(cursor) or cursor.kind != cindex.CursorKind.VAR_DECL:
                continue
            declaration_prefix = text[cursor.extent.start.offset:
                                      cursor.location.offset]
            children = [child for child in cursor.get_children()
                        if child.kind not in (cindex.CursorKind.TYPE_REF,
                                              cindex.CursorKind.TEMPLATE_REF,
                                              cindex.CursorKind.NAMESPACE_REF)]
            initializer = children[-1] if children else None
            referenced = initializer.referenced if initializer is not None else None
            if ("decltype(auto)" in declaration_prefix and
                    initializer is not None and
                    initializer.kind == cindex.CursorKind.CALL_EXPR and
                    referenced is not None and
                    referenced.get_usr() in pointer_return_functions):
                pointee = referenced.result_type.get_pointee().spelling.removeprefix("const ")
                declaration_end = cursor.location.offset + len(cursor.spelling)
                self._add(edits, cursor.extent.start.offset, declaration_end,
                          f"{pointee} *{cursor.spelling}")
                mutable_auto_locals.append(cursor)
            elif "decltype(auto)" in declaration_prefix and initializer is not None:
                initializer_text = text[initializer.extent.start.offset:
                                        initializer.extent.end.offset]
                if any(name in initializer_text for name in (
                        "CollisionAt(", "ShapeCollisionAt(",
                        "OrderedCollisionAt(")):
                    declaration_end = cursor.location.offset + len(cursor.spelling)
                    self._add(edits, cursor.extent.start.offset, declaration_end,
                              f"CudaCollision *{cursor.spelling}")
                    mutable_auto_locals.append(cursor)

            if (cursor.type.kind == cindex.TypeKind.POINTER and
                    re.search(r"\bauto\s*\*\s*$", declaration_prefix)):
                pointer_type = cursor.type
                if ("auto" in pointer_type.spelling and initializer is not None and
                        initializer.type.kind == cindex.TypeKind.POINTER):
                    pointer_type = initializer.type
                pointee = pointer_type.get_pointee().spelling.removeprefix("const ")
                declaration_end = cursor.location.offset + len(cursor.spelling)
                self._add(edits, cursor.extent.start.offset, declaration_end,
                          f"{pointee} *{cursor.spelling}")

        # Mutable local reference uses become explicit pointer dereferences.
        for local in mutable_locals + mutable_auto_locals:
            for cursor, _ in entries:
                if removed(cursor):
                    continue
                if (cursor.kind == cindex.CursorKind.DECL_REF_EXPR and
                        cursor.referenced is not None and
                        cursor.referenced.get_usr() == local.get_usr()):
                    self._add(edits, cursor.extent.start.offset,
                              cursor.extent.end.offset,
                              f"(*{local.spelling})")

        # Calls to mutable ref-returning accessors are dereferenced except when
        # they initialize another pointer or are forwarded by a pointer-return.
        for cursor, parent in entries:
            if removed(cursor):
                continue
            if cursor.kind != cindex.CursorKind.CALL_EXPR or cursor.referenced is None:
                continue
            if cursor.referenced.get_usr() not in pointer_return_functions:
                continue
            forwarded = (parent is not None and parent.kind == cindex.CursorKind.RETURN_STMT)
            initialized_pointer = any(
                self._inside(cursor, local)
                for local in mutable_locals + mutable_auto_locals)
            if not forwarded and not initialized_pointer:
                self._add(edits, cursor.extent.start.offset,
                          cursor.extent.start.offset, "(*")
                self._add(edits, cursor.extent.end.offset,
                          cursor.extent.end.offset, ")")

        # Pointer-return functions must take the address of plain lvalue return
        # expressions. Forwarded pointer-return calls remain unchanged.
        for function, _ in entries:
            if removed(function):
                continue
            if function.get_usr() not in pointer_return_functions:
                continue
            for cursor, _ in entries:
                if cursor.kind != cindex.CursorKind.RETURN_STMT or not self._inside(cursor, function):
                    continue
                expressions = list(cursor.get_children())
                expression = expressions[-1] if expressions else None
                if expression is None:
                    continue
                referenced = expression.referenced
                if not (expression.kind == cindex.CursorKind.CALL_EXPR and
                        referenced is not None and
                        referenced.get_usr() in pointer_return_functions):
                    self._add(edits, expression.extent.start.offset,
                              expression.extent.start.offset, "&(")
                    self._add(edits, expression.extent.end.offset,
                              expression.extent.end.offset, ")")

        # Reference-valued struct fields are lowered to pointers too. Member
        # expression extents cover only the field token, making this exact.
        pointer_fields = [cursor for cursor, _ in entries
                          if cursor.kind == cindex.CursorKind.FIELD_DECL and
                          cursor.type.kind == cindex.TypeKind.LVALUEREFERENCE and
                          not self._is_const_reference(cursor.type)]
        for field in pointer_fields:
            for cursor, _ in entries:
                if (cursor.kind == cindex.CursorKind.MEMBER_REF_EXPR and
                        cursor.referenced is not None and
                        cursor.referenced.get_usr() == field.get_usr()):
                    self._add(edits, cursor.extent.start.offset,
                              cursor.extent.end.offset,
                              f"(*{field.spelling})")

        # Apply from the back so Clang's original byte offsets stay valid.
        # Suppress nested edits covered by a removed function range.
        filtered = {
            key: value for key, value in edits.items()
            if key in removed_ranges or not any(
                start <= key[0] and key[1] <= end
                for start, end in removed_ranges)
        }
        for (start, end), replacement in sorted(
                filtered.items(), key=lambda item: (item[0][0], item[0][1]), reverse=True):
            text = text[:start] + replacement + text[end:]
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--types", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def replace_cpp_casts(text: str) -> str:
    for source, target in (
        ("static_cast", ""),
        ("reinterpret_cast", "reinterpret"),
        ("const_cast", "reinterpret"),
    ):
        cursor = 0
        while True:
            start = text.find(source + "<", cursor)
            if start < 0:
                break
            angle = start + len(source)
            depth = 0
            end = angle
            while end < len(text):
                if text[end] == "<":
                    depth += 1
                elif text[end] == ">":
                    depth -= 1
                    if depth == 0:
                        break
                end += 1
            if end >= len(text) or end + 1 >= len(text) or text[end + 1] != "(":
                cursor = end + 1
                continue
            type_text = re.sub(r"\s+", " ", text[angle + 1 : end]).strip()
            replacement = (
                f"reinterpret<{type_text}>" if target else type_text
            )
            text = text[:start] + replacement + text[end + 1 :]
            cursor = start + len(replacement)
    return text


def split_template_parameters(parameters: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(parameters):
        if character in "<({[":
            depth += 1
        elif character in ">)}]":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(parameters[start:index].strip())
            start = index + 1
    result.append(parameters[start:].strip())
    return [parameter for parameter in result if parameter]


CONCRETE_TYPE_PARAMETERS: dict[str, str] = {
    "Scratch": "cuda__collision__CudaCollisionScratch",
    "Collision": "cuda__collision__CudaCollision",
    "Left": "cuda__collision__CudaCollision",
    "Right": "cuda__collision__CudaCollision",
    "State": "CSceneVehicleCar__SSimulationWheel__SState",
    "Candidate": "CudaCandidatePhysicsState",
}


def translate_templates(
    text: str, concrete_type_parameters: dict[str, str] | None = None
) -> str:
    concrete_types = (
        CONCRETE_TYPE_PARAMETERS
        if concrete_type_parameters is None
        else concrete_type_parameters
    )
    pattern = re.compile(r"template\s*<(?P<params>.*?)>\s*", re.DOTALL)
    cursor = 0
    pieces: list[str] = []
    while True:
        match = pattern.search(text, cursor)
        if not match:
            pieces.append(text[cursor:])
            break
        pieces.append(text[cursor : match.start()])
        parameters: list[str] = []
        for parameter in split_template_parameters(match.group("params")):
            concrete = re.fullmatch(
                r"typename\s+(\w+)(?:\s*=.*)?", parameter, re.DOTALL
            )
            if concrete and concrete.group(1) in concrete_types:
                continue
            typename = re.fullmatch(r"typename\s+(\w+)(?:\s*=\s*(.*))?",
                                    parameter, re.DOTALL)
            if typename:
                translated = typename.group(1)
                if typename.group(2):
                    translated += " = " + typename.group(2)
                parameters.append(translated)
                continue
            value = re.fullmatch(
                r"(?:std::)?(bool|uint|uint8_t|uint16_t|uint32_t|uint64_t|size_t|CudaHandlingSpecialization)\s+"
                r"(\w+)(?:\s*=\s*(.*))?",
                re.sub(r"\s+", " ", parameter),
            )
            if not value:
                raise RuntimeError(f"unsupported template parameter: {parameter}")
            type_name = {
                "bool": "bool",
                "uint": "uint",
                "uint8_t": "uint8_t",
                "uint16_t": "uint16_t",
                "uint32_t": "uint",
                "uint64_t": "uint64_t",
                "size_t": "uint64_t",
                "CudaHandlingSpecialization": "uint",
            }[value.group(1)]
            translated = f"let {value.group(2)} : {type_name}"
            if value.group(3):
                translated += " = " + value.group(3)
            parameters.append(translated)

        declaration_start = match.end()
        if not parameters:
            cursor = declaration_start
            continue
        brace = text.find("{", declaration_start)
        semicolon = text.find(";", declaration_start)
        limit_candidates = [position for position in (brace, semicolon) if position >= 0]
        limit = min(limit_candidates) if limit_candidates else len(text)
        declaration = text[declaration_start:limit]
        struct_match = re.search(r"\b(struct|class)\s+(\w+)", declaration)
        if struct_match:
            name_start = declaration_start + struct_match.start(2)
            name_end = declaration_start + struct_match.end(2)
        else:
            paren = text.find("(", declaration_start, limit)
            if paren < 0:
                raise RuntimeError(
                    "could not locate generic declaration after "
                    + match.group(0)
                )
            prefix = text[declaration_start:paren]
            function = re.search(r"([A-Za-z_]\w*)\s*$", prefix)
            if not function:
                raise RuntimeError(f"could not locate generic function in {prefix}")
            name_start = declaration_start + function.start(1)
            name_end = declaration_start + function.end(1)
        pieces.append(text[declaration_start:name_end])
        pieces.append("<" + ", ".join(parameters) + ">")
        cursor = name_end
    result = "".join(pieces)
    for old, new in concrete_types.items():
        result = re.sub(rf"\b{old}\b", new, result)
    return result


def strip_source_file(text: str) -> str:
    text = re.sub(r"^\s*#include[^\n]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^\s*#(?:ifndef|define)\s+FOREVERVALIDATOR_CUDA_[A-Z0-9_]+\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )
    # Header-guard closing directives are the final non-whitespace token.
    text = re.sub(r"\n#endif\s*$", "\n", text)
    return text


TIMELINE_TEMPLATE_VALUES = {
        "TrackDiagnostics": True,
        "TrackCollisionDiagnostics": True,
        "TrustedInputs": False,
        "CompactReplacements": False,
        "EightOrderedEllipsoids": False,
        "WarpCoherentAcceleration": False,
        "WriteOutputSnapshots": True,
        "TriggerOnly": False,
        "UseMeshCellCache": False,
        "UnifiedBounds": False,
        "ReuseWorldCenter": False,
        "ReuseFrontInvariants": False,
        "ReuseWheelPassInvariants": False,
}


SEARCH_TEMPLATE_VALUES = {
        "TrackDiagnostics": False,
        "TrackCollisionDiagnostics": False,
        "TrustedInputs": True,
        "CompactReplacements": True,
        "EightOrderedEllipsoids": True,
        "WarpCoherentAcceleration": False,
        "WriteOutputSnapshots": False,
        "TriggerOnly": False,
        "UseMeshCellCache": False,
        "UnifiedBounds": False,
        "ReuseWorldCenter": False,
        "ReuseFrontInvariants": False,
        "ReuseWheelPassInvariants": True,
}


def fold_boolean_template_branches(
    text: str, values: dict[str, bool] | None = None
) -> str:
    """Discard C++ ``if constexpr`` branches for one GPU specialization."""
    if values is None:
        values = TIMELINE_TEMPLATE_VALUES

    def matching_brace(opening: int) -> int:
        depth = 1
        cursor = opening + 1
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            raise RuntimeError("unbalanced constexpr branch")
        return cursor - 1

    changed = True
    while changed:
        changed = False
        for name, value in values.items():
            pattern = re.compile(
                rf"\bif\s+(?:constexpr\s+)?\(\s*(?P<not>!\s*)?{name}\s*\)\s*\{{")
            match = pattern.search(text)
            if not match:
                continue
            first_open = match.end() - 1
            first_close = matching_brace(first_open)
            after = first_close + 1
            whitespace = re.match(r"\s*", text[after:]).group(0)
            else_start = after + len(whitespace)
            else_match = re.match(r"else\s*\{", text[else_start:])
            else_if_match = re.match(r"else\s+if\s*\(", text[else_start:])
            else_body = ""
            whole_end = after
            if else_match:
                else_open = else_start + else_match.end() - 1
                else_close = matching_brace(else_open)
                else_body = text[else_open + 1:else_close]
                whole_end = else_close + 1
            elif else_if_match:
                conditional_start = else_start + re.match(
                    r"else\s+", text[else_start:]
                ).end()
                chain_cursor = conditional_start
                while True:
                    chain_open = text.find("{", chain_cursor)
                    if chain_open < 0:
                        raise RuntimeError("malformed constexpr else-if chain")
                    chain_close = matching_brace(chain_open)
                    chain_after = chain_close + 1
                    chain_space = re.match(
                        r"\s*", text[chain_after:]
                    ).group(0)
                    next_else = chain_after + len(chain_space)
                    if re.match(r"else\s+if\s*\(", text[next_else:]):
                        chain_cursor = next_else + re.match(
                            r"else\s+", text[next_else:]
                        ).end()
                        continue
                    final_else = re.match(r"else\s*\{", text[next_else:])
                    if final_else:
                        final_open = next_else + final_else.end() - 1
                        whole_end = matching_brace(final_open) + 1
                    else:
                        whole_end = chain_after
                    break
                else_body = text[conditional_start:whole_end]
            condition = value != bool(match.group("not"))
            selected = text[first_open + 1:first_close] if condition else else_body
            text = (
                text[:match.start()] + "{\n" + selected + "\n}" +
                text[whole_end:]
            )
            changed = True
            break
    return text


def translate_common(
    text: str,
    mapping: dict[str, str],
    template_values: dict[str, bool] | None = None,
    concrete_type_parameters: dict[str, str] | None = None,
) -> str:
    text = strip_source_file(text)
    text = re.sub(r"\bstatic_assert\s*\(.*?\)\s*;", "", text, flags=re.DOTALL)
    text = fold_boolean_template_branches(text, template_values)
    text = text.replace("if constexpr", "if")
    text = text.replace(".Value()", ".value_")
    text = text.replace(".Index()", ".index_")
    text = text.replace("CHmsCorpusId::FromValue(", "CHmsCorpusId(")
    text = re.sub(r"(?<![A-Za-z0-9_:])cuda::",
                  "forevervalidator::simulation::cuda::", text)
    for namespace in (
        "exact", "memory", "tuning", "collision", "dynamics",
        "environment", "race", "response", "vehicle", "physics",
        "finish", "stunts", "transition", "facts",
    ):
        text = re.sub(
            rf"(?<![A-Za-z0-9_:]){namespace}::",
            f"forevervalidator::simulation::cuda::{namespace}::",
            text,
        )
    text = re.sub(r"::\s+", "::", text)
    for source in sorted(mapping, key=len, reverse=True):
        text = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(source)}(?![A-Za-z0-9_])",
            mapping[source],
            text,
        )
    replacements = {
        "std::byte": "uint8_t",
        "unsigned char": "uint8_t",
        "unsigned long long": "uint64_t",
        "unsigned long": "uint64_t",
        "unsigned": "uint",
        "std::": "",
        "__device__": "",
        "__host__": "",
        "__forceinline__": "",
        "__noinline__": "[ForceInline]",
        "__restrict__": "",
        "constexpr": "static const",
        "UINT64_C(1)": "uint64_t(1)",
        "__uint_as_float": "asfloat",
        "__float_as_uint": "asuint",
        "__longlong_as_double": "double_from_bits",
        "__double_as_longlong": "double_bits",
        "__double2float_rn": "float",
        "fabsf": "abs",
        "fabs": "abs",
        "sqrtf": "sqrt",
        "frexpf": "frexp",
        "fmodf": "fmod",
        "fminf": "min",
        "fmaxf": "max",
        "__uint2float_rn": "float",
        "__int2float_rn": "float",
        "signbit": "signbit_portable",
        "isfinite": "isfinite_portable",
        "nextafterf": "nextafter_portable",
        "noexcept": "",
    }
    for source, destination in replacements.items():
        text = text.replace(source, destination)
    text = re.sub(r"\bconst auto\b", "let", text)
    text = re.sub(r"\bdecltype\(auto\)", "var", text)
    text = re.sub(r"\bauto\b", "var", text)
    text = re.sub(r"\bclass\s+", "struct ", text)
    text = text.replace("enum struct ", "enum class ")
    text = re.sub(r"^\s*(public|private|protected):\s*$", "", text, flags=re.MULTILINE)
    text = replace_cpp_casts(text)
    text = translate_templates(text, concrete_type_parameters)
    text = re.sub(r"\b__constref\s+const\b", "__constref", text)
    # C-style physical pointers in Slang do not accept C++ pointee const. The
    # shader only writes through pointers originating from mutable ABI inputs.
    text = re.sub(r"\bconst\s+([A-Za-z_]\w*(?:::\w+)*(?:<[^;()]+>)?)\s*\*", r"\1 *", text)
    text = re.sub(r"\bfrexp\(([^,]+),\s*&\s*(\w+)\)", r"frexp(\1, \2)", text)
    text = re.sub(
        r"\s*let\s+lowerBound\s*=\s*\[\]\(float value\)\s*\{\s*return value - KeyEpsilon;\s*\};",
        "", text)
    text = re.sub(
        r"\s*let\s+upperBound\s*=\s*\[\]\(float value\)\s*\{\s*return value \+ KeyEpsilon;\s*\};",
        "", text)
    text = re.sub(r"\blowerBound\(([^()\n]+)\)", r"(\1 - KeyEpsilon)", text)
    text = re.sub(r"\bupperBound\(([^()\n]+)\)", r"(\1 + KeyEpsilon)", text)
    text = re.sub(
        r"const\s+GmVec3\s+replacement\s*=\s*\[&\]\(\)\s*\{\s*"
        r"if static const \(CompactReplacements\)\s*\{\s*"
        r"return scratch\.replacementOverflowCount == 0u\s*"
        r"\? GmVec3(?:\{\}|\(\))\s*:\s*FinalizeReplacement\(\s*"
        r"scratch\.replacementSumX,\s*scratch\.replacementSumY,\s*"
        r"scratch\.replacementSumZ\);\s*\}\s*else\s*\{\s*"
        r"return SynthesizeReplacement\(\s*body\.collisionReplacements, scratch\);\s*"
        r"\}\s*\}\(\);",
        "GmVec3 replacement;\n    if static const (CompactReplacements) {\n"
        "        replacement = scratch.replacementOverflowCount == 0u\n"
        "            ? GmVec3() : FinalizeReplacement(scratch.replacementSumX,\n"
        "                scratch.replacementSumY, scratch.replacementSumZ);\n"
        "    } else {\n        replacement = SynthesizeReplacement(\n"
        "            body.collisionReplacements, scratch);\n    }",
        text,
    )
    text = re.sub(
        r"\s*let\s+reject\s*=\s*\[&\]\(\)\s*\{.*?return true;\s*\};",
        "", text, flags=re.DOTALL)
    text = text.replace("reject()", "finish_reject(rejected)")
    text = re.sub(
        r"(UnitSphereTriangleQuery<[^>]+>\s+\w+\s*\{\s*)scratch\s*,",
        r"\1&scratch,", text)
    text = re.sub(r"\bUnitSphereTriangleQuery<[^>]+>",
                  "UnitSphereTriangleQuery", text)
    text = re.sub(r"\b(UnitSphereTriangleQuery\s+\w+)\s*\{",
                  r"\1 = {", text)
    text = re.sub(r"\b([A-Za-z_]\w*(?:::\w+)*(?:<[^;{}]+>)?)\s+(\w+)\s*"
                  r"(\[[^\]]+\])\s*\{\s*\}",
                  r"\1 \2\3 = {}", text)
    text = re.sub(
        r"return\s+&\(\(index\s*&\s*1u\)\s*==\s*0u\s*"
        r"\?\s*\(\*storage\)\.extraNegated\s*:\s*"
        r"\(\*storage\)\.contactPoint\);",
        "if ((index & 1u) == 0u) return &(*storage).extraNegated;\n"
        "    return &(*storage).contactPoint;", text)
    text = re.sub(
        r"(destination\.values\[index\]\s*=\s*)ReplacementOverflowAt\(",
        r"\1ReplacementOverflowAtConst(", text)
    text = re.sub(
        r"AddMain\(scratch,\s*ShapeCollisionAt\(([^;]+?)\)\);",
        r"AddMain(scratch, (*ShapeCollisionAt(\1)));", text)
    text = re.sub(
        r"ShapeCollisionAt\(\s*(scratch|\(\*scratch\)),\s*([^()]+?)\)\s*\.",
        r"(*ShapeCollisionAt(\1, \2)).", text)
    text = re.sub(
        r"(?<!\*)OrderedCollisionAt\(\s*scratch\s*,\s*([^()]+?)\)",
        r"(*OrderedCollisionAt(scratch, \1))", text)
    text = re.sub(
        r"(CudaCollision\s*\*\s*\w+\s*=\s*)\(\*OrderedCollisionAt\((.*?)\)\)",
        r"\1OrderedCollisionAt(\2)", text, flags=re.DOTALL)
    text = text.replace(
        "forevervalidator::simulation::cuda::collision::detail::ReplacementOverflowAt(",
        "forevervalidator::simulation::cuda::collision::detail::ReplacementOverflowAtConst(")
    text = re.sub(
        r"(?<!Const)detail::ReplacementOverflowAt\(\s*scratch\s*,\s*([^()]+?)\)",
        r"(*detail::ReplacementOverflowAt(scratch, \1))", text)
    text = re.sub(
        r"const\s+GmVec3\s+replacement\s*=\s*\[&\]\(\)\s*\{\s*"
        r"return\s+SynthesizeReplacement\(\s*body\.collisionReplacements,\s*scratch\);\s*"
        r"\}\(\);",
        "const GmVec3 replacement = SynthesizeReplacement(\n"
        "            body.collisionReplacements, scratch);", text)
    text = text.replace("detail::(*OrderedCollisionAt(",
                        "(*detail::OrderedCollisionAt(")
    text = text.replace(
        "forevervalidator::simulation::cuda::collision::detail::(*OrderedCollisionAt(",
        "(*forevervalidator::simulation::cuda::collision::detail::OrderedCollisionAt(")
    text = text.replace(
        "forevervalidator::simulation::cuda::facts::WheelCount(vehicle)",
        "forevervalidator::simulation::cuda::facts::WheelCount(vehicle)")
    text = text.replace("SlippingWheelScale(vehicle, configuration)",
                        "SlippingWheelScale((*vehicle), configuration)")
    text = text.replace(
        "forevervalidator::simulation::cuda::exact::SinCosResult frontSteerSinCos = SinCosResult()",
        "forevervalidator::simulation::cuda::exact::SinCosResult frontSteerSinCos = forevervalidator::simulation::cuda::exact::SinCosResult()")
    text = text.replace(
        "forevervalidator::simulation::cuda::exact::SinCosResult steeringSinCos = SinCosResult()",
        "forevervalidator::simulation::cuda::exact::SinCosResult steeringSinCos = forevervalidator::simulation::cuda::exact::SinCosResult()")
    text = text.replace("= SinCosResult()",
                        "= forevervalidator::simulation::cuda::exact::SinCosResult()")
    text = re.sub(
        r"return index < configuration->materials\.count\s*"
        r"\? materials \+ index\s*:\s*nullptr;",
        "if (index < configuration->materials.count) return materials + index;\n"
        "    return reinterpret<VehicleMaterialDefinition*>(uint64_t(0));", text)
    text = re.sub(
        r"\(\*race\)\.checkpointSlotsPassed\.Get\(([^)]+)\)",
        r"CheckpointSlotsGet((*race).checkpointSlotsPassed, \1)", text)
    text = re.sub(
        r"\(\*race\)\.checkpointSlotsPassed\.Set\(([^)]+)\);",
        r"CheckpointSlotsSet((*race).checkpointSlotsPassed, \1);", text)
    text = text.replace("(*race).checkpointSlotsPassed.Clear();",
                        "CheckpointSlotsClear((*race).checkpointSlotsPassed);")
    text = re.sub(
        r"for \(GmSpring_float \*spring &: \(\*vehicle\)\.dynaPartSprings\) \{\s*"
        r"\(\*spring\)\.value = 0\.0f;\s*"
        r"\(\*spring\)\.target = 0\.0f;\s*"
        r"\(\*spring\)\.velocity = 0\.0f;\s*\}",
        "for (uint spring = 0u; spring < 4u; ++spring) {\n"
        "        (*vehicle).dynaPartSprings[spring].value = 0.0f;\n"
        "        (*vehicle).dynaPartSprings[spring].target = 0.0f;\n"
        "        (*vehicle).dynaPartSprings[spring].velocity = 0.0f;\n    }", text)
    text = re.sub(
        r"(cuda__collision__CudaCollision\s*\*\s*\w+\s*=\s*)"
        r"\(\*detail::OrderedCollisionAt\(([^)]*)\)\);",
        r"\1detail::OrderedCollisionAt(\2);", text)
    text = re.sub(
        r"(cuda__collision__CudaCollision\s*\*\s*\w+\s*=\s*)"
        r"forevervalidator::simulation::cuda::collision::\(\*detail::OrderedCollisionAt\(([^)]*)\)\);",
        r"\1forevervalidator::simulation::cuda::collision::detail::OrderedCollisionAt(\2);",
        text)
    text = text.replace("material->", "(*material).")
    text = text.replace("wheelMaterial->", "(*wheelMaterial).")
    text = text.replace("estimate->", "(*estimate).")
    text = re.sub(
        r"\b((?:u?int(?:8|16|32|64)?_t|[A-Z]\w*))\{([^{}]*)\}",
        r"\1(\2)",
        text,
    )
    text = re.sub(
        r"\b([A-Z]\w*(?:::\w+)*)\s+(\w+)\s*\{\s*\}",
        r"\1 \2 = \1()",
        text,
    )
    text = re.sub(
        r"struct\s+cuda__finish__Refinement\s*\{.*?\};",
        "",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"inline\s+void\s+ResetVehiclePassthrough\s*\(\s*inout\s+CudaCandidateState\s*\)\s*\{\s*\}",
        "",
        text,
    )
    text = text.replace(
        "= SinCosResult()",
        "= forevervalidator::simulation::cuda::exact::SinCosResult()")
    # SPIR-V's single-precision extended sqrt has implementation-dependent
    # accuracy. Evaluate it in binary64 and round once so every Vulkan vendor
    # reaches the same correctly-rounded binary32 value as precise CUDA.
    text = text.replace(
        "if (value >= 0.0f) {\n        return sqrt(value);\n    }",
        "if (value >= 0.0f) {\n"
        "        return FromDouble(sqrt(double(value)));\n"
        "    }")
    # Vulkan float division is allowed to be implementation-dependent. Route
    # parity-sensitive divisions through binary64 and round once to binary32.
    text = text.replace(
        "return numerator / denominator;",
        "return FromDouble(double(numerator) / double(denominator));")
    # Slang otherwise lowers even compile-time rational constants to OpFDiv.
    # Vulkan does not require correctly rounded binary64 division, so materialize
    # the host/CUDA constant-folded IEEE value by bits instead.
    def replace_rational_constant(match: re.Match[str]) -> str:
        sign = -1.0 if match.group("sign") else 1.0
        denominator = float(match.group("denominator"))
        bits = struct.unpack("<Q", struct.pack("<d", sign / denominator))[0]
        return f"double_from_bits(0x{bits:016x}ull)"

    text = re.sub(
        r"(?<![0-9.])(?P<sign>-?)1\.0\s*/\s*"
        r"(?P<denominator>[0-9]+(?:\.[0-9]*)?)",
        replace_rational_constant,
        text,
    )
    # Slang 2026.14 also parses unsuffixed decimal literals through binary32
    # before widening them in a double context. Preserve C++ binary64 literal
    # semantics whenever that intermediate narrowing would change the value.
    def replace_double_literal(match: re.Match[str]) -> str:
        literal = match.group("literal")
        value = float(literal)
        try:
            narrowed = struct.unpack("<f", struct.pack("<f", value))[0]
        except OverflowError:
            narrowed = float("inf")
        if value == narrowed:
            return literal
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return f"double_from_bits(0x{bits:016x}ull)"

    text = re.sub(
        r"(?<![A-Za-z0-9_.])"
        r"(?P<literal>(?:[0-9]+\.[0-9]*|\.[0-9]+)"
        r"(?:[eE][+-]?[0-9]+)?)"
        r"(?![fF0-9A-Za-z_])",
        replace_double_literal,
        text,
    )
    # A bit reinterpretation is not accepted as a Slang global constant
    # initializer. Turn affected namespace constants into tiny inline accessors
    # and update their uses; the accessor inlines to one OpBitcast.
    global_double = re.compile(
        r"static const double (?P<name>[A-Za-z_]\w*) = "
        r"double_from_bits\((?P<bits>0x[0-9a-f]+ull)\);")
    for match in list(global_double.finditer(text)):
        name = match.group("name")
        text = re.sub(
            rf"\b{re.escape(name)}\b(?!\s*=)",
            f"{name}()",
            text,
        )
        text = text.replace(
            f"static const double {name} = "
            f"double_from_bits({match.group('bits')});",
            f"inline double {name}() {{ return "
            f"double_from_bits({match.group('bits')}); }}",
        )
    specialization = template_values or TIMELINE_TEMPLATE_VALUES
    if specialization["CompactReplacements"]:
        replacement = (
            "const GmVec3 replacement = scratch.replacementOverflowCount == 0u\n"
            "            ? GmVec3() : FinalizeReplacement(\n"
            "                  scratch.replacementSumX,\n"
            "                  scratch.replacementSumY,\n"
            "                  scratch.replacementSumZ);"
        )
    else:
        replacement = (
            "const GmVec3 replacement = SynthesizeReplacement(\n"
            "            body.collisionReplacements, scratch);"
        )
    text = re.sub(
        r"const GmVec3 replacement = \[&\]\(\) \{.*?\}\(\);",
        replacement,
        text,
        flags=re.DOTALL,
    )
    return text


PRELUDE = r"""
// Vulkan/Slang compatibility intrinsics. The runtime requires IEEE float
// controls and compiles every arithmetic instruction with NoContraction.
bool signbit_portable(float value) { return (asuint(value) & 0x80000000u) != 0u; }
bool signbit_portable(double value) { return (reinterpret<uint64_t>(value) & 0x8000000000000000ull) != 0ull; }
bool isfinite_portable(float value) { return (asuint(value) & 0x7f800000u) != 0x7f800000u; }
bool isfinite_portable(double value) { return (reinterpret<uint64_t>(value) & 0x7ff0000000000000ull) != 0x7ff0000000000000ull; }
uint64_t double_bits(double value) { return reinterpret<uint64_t>(value); }
double double_from_bits(uint64_t value) { return reinterpret<double>(value); }

static const uint CudaCollisionReplacementInlineCapacity = 1u;
static const uint CudaCollisionReplacementOverflowCapacity = 511u;
static const uint CudaCollisionReplacementCapacity = 512u;
static const uint CollisionCapacity = 512u;
static const uint ShapeCollisionCapacity = 256u;
static const uint SurfaceHitCapacity = 128u;
static const uint MeshCellHitCapacity = 1024u;
static const uint CudaCollisionSearchTileWidth = 32u;
static const uint CudaTuningCurvePositionsNondecreasing = 1u;
static const uint VehicleFakeContactTextureWidth = 128u;
static const uint VehicleFakeContactTextureHeight = 128u;
static const uint VehicleFakeContactTextureBytesPerPixel = 3u;

bool finish_reject(bool* rejected)
{
    if (rejected != nullptr) *rejected = true;
    return true;
}

bool CheckpointSlotsGet(__constref CudaCheckpointSlots slots, uint slot)
{
    return slot < slots.count &&
        (slots.words[slot / 32u] & (1u << (slot % 32u))) != 0u;
}
void CheckpointSlotsSet(inout CudaCheckpointSlots slots, uint slot)
{
    if (slot < slots.count) slots.words[slot / 32u] |= 1u << (slot % 32u);
}
void CheckpointSlotsClear(inout CudaCheckpointSlots slots)
{
    for (uint index = 0u; index < 32u; ++index) slots.words[index] = 0u;
}

float nextafter_portable(float from, float toward)
{
    if (isnan(from) || isnan(toward)) return asfloat(0x7fc00000u);
    if (from == toward) return toward;
    uint bits = asuint(from);
    if ((bits & 0x7fffffffu) == 0u)
        return asfloat((asuint(toward) & 0x80000000u) | 1u);
    bool increment = (from < toward) == !signbit_portable(from);
    return asfloat(increment ? bits + 1u : bits - 1u);
}

namespace forevervalidator::simulation::cuda::facts {
GmBoxAligned ShapeLocalBounds(uint, __constref CudaVehicleCollisionShape shape)
{
    return shape.localBounds;
}
GmIso4 ShapeBodyPose(uint, __constref CudaVehicleCollisionShape shape)
{
    return shape.bodyPose;
}
uint ShapeWheelIndex(uint, __constref CudaVehicleCollisionShape shape)
{
    return shape.wheelIndex;
}
uint ShapeSurfaceMaterial(uint, __constref CudaVehicleCollisionShape shape)
{
    return shape.surfaceMaterial;
}
CudaVehicleTuning Tuning(CudaPackedStaticConfigurationHeader* configuration)
{
    return configuration->tuning;
}
VehicleWheelDefinition Wheel(CudaPackedStaticConfigurationHeader* configuration, uint index)
{
    VehicleWheelDefinition* wheels = reinterpret<VehicleWheelDefinition*>(&configuration->wheels.wheels);
    return wheels[index];
}
uint WheelAxle(CudaPackedStaticConfigurationHeader* configuration, uint index)
{
    return Wheel(configuration, index).axle;
}
bool WheelKillsLateralSpeed(CudaPackedStaticConfigurationHeader* configuration, uint index)
{
    return Wheel(configuration, index).killsLateralSpeedOnContact != 0u;
}
float WheelRollingRadius(CudaPackedStaticConfigurationHeader* configuration, uint index)
{
    return Wheel(configuration, index).rollingRadius;
}
uint WheelForceMode(CudaPackedStaticConfigurationHeader* configuration)
{
    return configuration->tuning.wheelForceMode;
}
uint WheelCount(__constref CudaVehicleState vehicle)
{
    return min(vehicle.wheels.count, 4u);
}
uint WheelCount(CudaVehicleState* vehicle)
{
    return min((*vehicle).wheels.count, 4u);
}
}
"""


TIMELINE_KERNEL = r"""
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

struct VulkanTimelinePushConstants
{
    uint64_t scene;
    uint64_t configuration;
    uint64_t states;
    uint64_t descriptors;
    uint64_t ticks;
    uint64_t observations;
    uint64_t results;
    uint64_t scratch;
    uint64_t cancellation;
    uint candidateCount;
    uint stateStride;
    uint fullState;
};

[[vk::push_constant]] ConstantBuffer<VulkanTimelinePushConstants> timelinePush;

bool ValidPackedInputs(CudaPackedSceneHeader* scene, CudaPackedStaticConfigurationHeader* configuration)
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

void RecordObservation(
    __constref CudaCandidatePhysicsState state,
    __constref CudaControlTick tick,
    inout CudaTimelineObservation observation)
{
    observation.simulatedPosition = state.body.current.position;
    observation.writePosition = state.body.write.position;
    observation.hasComparison = tick.hasComparisonTarget;
    observation.comparisonTarget = tick.comparisonTarget;
    if (tick.hasComparisonTarget != 0u)
    {
        observation.comparisonDelta = GmVec3(
            state.body.write.position.x - tick.comparisonTarget.x,
            state.body.write.position.y - tick.comparisonTarget.y,
            state.body.write.position.z - tick.comparisonTarget.z);
        GmVec3 delta = observation.comparisonDelta;
        observation.comparisonDistance = cuda::exact::Sqrt(
            (delta.x * delta.x + delta.y * delta.y) + delta.z * delta.z);
    }
    observation.hasFinishTick = state.race.progress.raceCompleted;
    observation.finishTickMs = state.race.progress.lastPrepareTimeMs;
}

[noinline]
cuda::vehicle::ForceStatus ExecuteTimelineForcePass(
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    float dt)
{
    cuda::environment::BeginForcePass(state->body, configuration);
    if (!state->vehicle.mobil.physicsUpdatesEnabled)
        return cuda::vehicle::ForceStatus::Success;
    return cuda::vehicle::ComputeForcesModel6<
        false, CudaHandlingSpecialization_Generic>(
            *state, configuration, dt);
}

[noinline]
uint ExecuteTimelineCollisionDetection(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    cuda__collision__CudaCollisionScratch* scratch,
    float dt)
{
    cuda::dynamics::PreCollision<false>(state->body, *scratch, dt);
    return cuda::collision::Detect<true, false, false, false>(
        scene, configuration, *state, *scratch);
}

[noinline]
uint ExecuteTimelineCollisionResponse(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    cuda__collision__CudaCollisionScratch* scratch)
{
    return cuda::collision::Respond<true, false, false>(
        scene, configuration, *state, *scratch);
}

[noinline]
void ExecuteTimelinePostCollision(
    CudaCandidatePhysicsState* state,
    cuda__collision__CudaCollisionScratch* scratch)
{
    cuda::dynamics::PostCollision<false>(state->body, *scratch);
}

[noinline]
cuda::physics::Status ExecuteTimelineCollisionSubstep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    cuda__collision__CudaCollisionScratch* scratch,
    float dt)
{
    cuda::vehicle::ForceStatus forceStatus = ExecuteTimelineForcePass(
        configuration, state, dt);
    if (forceStatus != cuda::vehicle::ForceStatus::Success)
        return cuda::physics::Status(
            uint(cuda::physics::Status::UnsupportedForceBase) +
            uint(forceStatus));
    uint collisionStatus = ExecuteTimelineCollisionDetection(
        scene, configuration, state, scratch, dt);
    if (collisionStatus == cuda__collision__Status_Success)
        collisionStatus = ExecuteTimelineCollisionResponse(
            scene, configuration, state, scratch);
    if (collisionStatus != cuda__collision__Status_Success)
        return cuda::physics::Status(
            uint(cuda::physics::Status::CollisionFailureBase) +
            collisionStatus);
    ExecuteTimelinePostCollision(state, scratch);
    return cuda::physics::Status::Success;
}

[noinline]
void ExecuteTimelineAfterContacts(
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state)
{
    cuda::vehicle::AfterContacts(*state, configuration);
}

[noinline]
cuda::physics::Status ExecuteTimelinePhysicsStep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    cuda__collision__CudaCollisionScratch* scratch)
{
    float dt = float(int(state->world.schemePeriodMs)) * 0.001f;
    if (state->body.dynamicActive)
    {
        state->body.temporary = state->body.current;
        GmVec3 linear = state->body.current.linearSpeed;
        GmVec3 angular = state->body.current.angularSpeed;
        float linearLength = cuda::exact::Sqrt(
            (linear.y * linear.y + linear.x * linear.x) +
            linear.z * linear.z);
        float angularLength = cuda::exact::Sqrt(
            (angular.x * angular.x + angular.y * angular.y) +
            angular.z * angular.z);
        float scaled = cuda::exact::Divide(
            (linearLength + angularLength) * dt,
            state->body.parameters.maxStepDistance);
        uint substeps = cuda::exact::TruncateToUint32Modulo(scaled) + 1u;
        if (substeps > 1000u) substeps = 1000u;
        float remaining = dt;
        if (substeps > 1u)
        {
            float split = cuda::exact::Divide(
                dt, cuda::exact::FromUnsignedInteger(substeps));
            for (uint count = substeps - 1u; count != 0u; --count)
            {
                cuda::physics::Status status =
                    ExecuteTimelineCollisionSubstep(
                        scene, configuration, state, scratch, split);
                if (status != cuda::physics::Status::Success)
                    return status;
                remaining -= split;
            }
        }
        cuda::physics::Status finalStatus =
            ExecuteTimelineCollisionSubstep(
                scene, configuration, state, scratch, remaining);
        if (finalStatus != cuda::physics::Status::Success)
            return finalStatus;
        state->body.write = state->body.temporary;
    }
    if (state->vehicle.mobil.physicsUpdatesEnabled)
        ExecuteTimelineAfterContacts(configuration, state);
    return cuda::physics::Status::Success;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ExecuteVulkanTimeline(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint candidate = dispatchThreadId.x;
    if (candidate >= timelinePush.candidateCount) return;
    CudaPackedSceneHeader* scene = reinterpret<CudaPackedSceneHeader*>(timelinePush.scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(timelinePush.configuration);
    uint8_t* stateBytes = reinterpret<uint8_t*>(timelinePush.states);
    CudaCandidatePhysicsState* state =
        reinterpret<CudaCandidatePhysicsState*>(
            stateBytes + uint64_t(candidate) * timelinePush.stateStride);
    VulkanTimelineDescriptor* descriptors = reinterpret<VulkanTimelineDescriptor*>(timelinePush.descriptors);
    CudaControlTick* ticks = reinterpret<CudaControlTick*>(timelinePush.ticks);
    CudaTimelineObservation* observations = reinterpret<CudaTimelineObservation*>(timelinePush.observations);
    VulkanTimelineResult* results = reinterpret<VulkanTimelineResult*>(timelinePush.results);
    cuda__collision__CudaCollisionScratch* scratch =
        reinterpret<cuda__collision__CudaCollisionScratch*>(timelinePush.scratch);
    uint* cancellation = reinterpret<uint*>(timelinePush.cancellation);

    if (!ValidPackedInputs(scene, configuration))
    {
        results[candidate].status = 1u;
        return;
    }
    if (state->schemaVersion != CudaCandidateState_SchemaVersion)
    {
        results[candidate].status = 2u;
        return;
    }
    VulkanTimelineDescriptor descriptor = descriptors[candidate];
    if (descriptor.tickCount == 0u)
    {
        results[candidate].status = 0u;
        results[candidate].failureTick = UINT32_MAX;
        return;
    }
    for (uint index = 0u; index < descriptor.tickCount; ++index)
    {
        if (*cancellation != 0u)
        {
            results[candidate].status = 4u;
            results[candidate].failureTick = index;
            return;
        }
        CudaControlTick tick = ticks[descriptor.firstTick + index];
        ApplyControlAndTimingPrefix(*state, tick, false);
        if (state->firstStep == 0u)
            cuda::transition::PrepareStep(*state, tick, configuration);
        state->vehicle.mobil.absorbContactEnabled = 1u;
        state->vehicle.mobil.physicsUpdatesEnabled =
            (tick.actionFlags & 2u) == 0u;
        for (uint respawn = 0u; respawn < tick.respawnAtCheckpointCount; ++respawn)
        {
            if (cuda::transition::Respawn(*state, configuration))
            {
                ++results[candidate].executedRespawnCount;
                ++state->incrementalRespawnCount;
                if (timelinePush.fullState != 0u)
                {
                    CudaCandidateState* fullState =
                        reinterpret<CudaCandidateState*>(state);
                    cuda::stunts::ApplyRespawnPenalty(fullState->stunts);
                }
            }
        }
        cuda::physics::Status physicsStatus = ExecuteTimelinePhysicsStep(
            scene, configuration, state, scratch + candidate);
        if (physicsStatus != cuda::physics::Status::Success)
        {
            results[candidate].status = 6u;
            results[candidate].failureTick = index;
            results[candidate].failureDetail = uint(physicsStatus) +
                1000u * uint(scratch[candidate].overflowReason) +
                100000u * scratch[candidate].collisionCount;
            return;
        }
        if (timelinePush.fullState != 0u)
        {
            CudaCandidateState* fullState =
                reinterpret<CudaCandidateState*>(state);
            cuda::collision::detail::CaptureReplacementOverflow(
                scratch[candidate], fullState->collisionReplacementOverflow);
            if (state->stuntsEnabled != 0u)
            {
                cuda::stunts::Status stuntStatus =
                    cuda::stunts::Update(*fullState, tick);
                if (stuntStatus != cuda::stunts::Status::Success)
                {
                    results[candidate].status = 3u;
                    results[candidate].failureTick = index;
                    results[candidate].failureDetail = uint(stuntStatus);
                    return;
                }
            }
        }
        state->firstStep = 0u;
        ++state->controlCursor;
        ++results[candidate].executedTickCount;
        if (tick.observe != 0u)
        {
            if (results[candidate].observationCount >= descriptor.observationCapacity)
            {
                results[candidate].status = 3u;
                results[candidate].failureTick = index;
                return;
            }
            RecordObservation(
                *state, tick,
                observations[descriptor.firstObservation + results[candidate].observationCount]);
            ++results[candidate].observationCount;
        }
    }
    results[candidate].status = 0u;
    results[candidate].failureTick = UINT32_MAX;
}

#if !defined(FOREVERVALIDATOR_VULKAN_TIMELINE_ONLY)
struct VulkanFinishRefinementWork
{
    uint status;
    float fullDt;
    double substepStartNs;
    FinishTimeEstimate estimate;
};

[noinline]
cuda::physics::Status ExecuteVulkanFinishProbeSubstep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    float dt,
    cuda__collision__CudaCollisionScratch* scratch)
{
    return cuda::finish::ProbeFinishSubstep<
        true, false, false, false, false, false>(
            scene, configuration, *state, dt, *scratch);
}

[noinline]
cuda::physics::Status ExecuteVulkanFinishCollisionSubstep(
    CudaPackedSceneHeader* scene,
    CudaPackedStaticConfigurationHeader* configuration,
    CudaCandidatePhysicsState* state,
    float dt,
    cuda__collision__CudaCollisionScratch* scratch)
{
    return cuda::physics::CollisionSubstep<
        true, false, false, false, false, false,
        CudaHandlingSpecialization_Generic>(
            scene, configuration, *state, dt, *scratch);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ExecuteVulkanFinishLocate(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint candidate = dispatchThreadId.x;
    if (candidate >= timelinePush.candidateCount) return;
    CudaPackedSceneHeader* scene = reinterpret<CudaPackedSceneHeader*>(timelinePush.scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(timelinePush.configuration);
    CudaCandidateState* states = reinterpret<CudaCandidateState*>(timelinePush.states);
    VulkanTimelineDescriptor* descriptors = reinterpret<VulkanTimelineDescriptor*>(timelinePush.descriptors);
    CudaControlTick* ticks = reinterpret<CudaControlTick*>(timelinePush.ticks);
    VulkanTimelineResult* results = reinterpret<VulkanTimelineResult*>(timelinePush.results);
    cuda__collision__CudaCollisionScratch* scratch =
        reinterpret<cuda__collision__CudaCollisionScratch*>(timelinePush.scratch);
    VulkanFinishRefinementWork* work =
        reinterpret<VulkanFinishRefinementWork*>(timelinePush.observations);
    uint* cancellation = reinterpret<uint*>(timelinePush.cancellation);

    if (!ValidPackedInputs(scene, configuration))
    {
        results[candidate].status = 1u;
        return;
    }
    if (states[candidate].schemaVersion != CudaCandidateState_SchemaVersion)
    {
        results[candidate].status = 2u;
        return;
    }
    VulkanTimelineDescriptor descriptor = descriptors[candidate];
    if (descriptor.tickCount == 0u)
    {
        results[candidate].status = 0u;
        results[candidate].failureTick = UINT32_MAX;
        return;
    }
    work[candidate].status = 0u;
    for (uint index = 0u; index < descriptor.tickCount; ++index)
    {
        if (*cancellation != 0u)
        {
            results[candidate].status = 4u;
            results[candidate].failureTick = index;
            return;
        }
        CudaControlTick tick = ticks[descriptor.firstTick + index];
        CudaCandidatePhysicsState* state =
            reinterpret<CudaCandidatePhysicsState*>(states + candidate);
        ApplyControlAndTimingPrefix(*state, tick, false);
        if (states[candidate].firstStep == 0u)
            cuda::transition::PrepareStep(*state, tick, configuration);
        states[candidate].vehicle.mobil.absorbContactEnabled = 1u;
        states[candidate].vehicle.mobil.physicsUpdatesEnabled =
            (tick.actionFlags & 2u) == 0u;
        for (uint respawn = 0u; respawn < tick.respawnAtCheckpointCount; ++respawn)
        {
            if (cuda::transition::Respawn(*state, configuration))
                ++states[candidate].incrementalRespawnCount;
        }
        float dt = float(int(states[candidate].world.schemePeriodMs)) * 0.001f;
        if (states[candidate].body.dynamicActive)
        {
            states[candidate].body.temporary = states[candidate].body.current;
            GmVec3 linear = states[candidate].body.current.linearSpeed;
            GmVec3 angular = states[candidate].body.current.angularSpeed;
            float linearLength = cuda::exact::Sqrt(
                (linear.y * linear.y + linear.x * linear.x) + linear.z * linear.z);
            float angularLength = cuda::exact::Sqrt(
                (angular.x * angular.x + angular.y * angular.y) + angular.z * angular.z);
            float scaled = cuda::exact::Divide(
                (linearLength + angularLength) * dt,
                states[candidate].body.parameters.maxStepDistance);
            uint substeps = cuda::exact::TruncateToUint32Modulo(scaled) + 1u;
            if (substeps > 1000u) substeps = 1000u;
            float remaining = dt;
            double elapsed = 0.0;
            uint64_t tickStartNs = cuda::finish::TickStartNanoseconds(tick.timeMs);
            for (uint substep = 0u; substep < substeps; ++substep)
            {
                float substepDt = substep + 1u < substeps
                    ? cuda::exact::Divide(
                        dt, cuda::exact::FromUnsignedInteger(substeps))
                    : remaining;
                uint preSubstepIndex = timelinePush.candidateCount + candidate;
                states[preSubstepIndex] = states[candidate];
                bool wasFinished =
                    states[preSubstepIndex].race.progress.raceCompleted;
                cuda::physics::Status physicsStatus =
                    ExecuteVulkanFinishCollisionSubstep(
                        scene, configuration, state,
                        substepDt, &(scratch[candidate]));
                if (physicsStatus != cuda::physics::Status::Success)
                {
                    results[candidate].status = 6u;
                    results[candidate].failureTick = index;
                    results[candidate].failureDetail = uint(physicsStatus) +
                        1000u * uint(scratch[candidate].overflowReason) +
                        100000u * scratch[candidate].collisionCount;
                    return;
                }
                if (!wasFinished &&
                    states[candidate].race.progress.raceCompleted)
                {
                    work[candidate].status = 1u;
                    work[candidate].fullDt = substepDt;
                    work[candidate].substepStartNs = double(tickStartNs) +
                        elapsed * 1000000000.0;
                    ++results[candidate].executedTickCount;
                    results[candidate].status = 0u;
                    results[candidate].failureTick = UINT32_MAX;
                    return;
                }
                elapsed += double(substepDt);
                remaining -= substepDt;
            }
            states[candidate].body.write = states[candidate].body.temporary;
        }
        if (states[candidate].vehicle.mobil.physicsUpdatesEnabled)
            cuda::vehicle::AfterContacts(*state, configuration);
        ++results[candidate].executedTickCount;
        states[candidate].firstStep = 0u;
    }
    results[candidate].status = 6u;
    results[candidate].failureTick = UINT32_MAX;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void ExecuteVulkanFinishProbe(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint candidate = dispatchThreadId.x;
    if (candidate >= timelinePush.candidateCount) return;
    CudaPackedSceneHeader* scene = reinterpret<CudaPackedSceneHeader*>(timelinePush.scene);
    CudaPackedStaticConfigurationHeader* configuration =
        reinterpret<CudaPackedStaticConfigurationHeader*>(timelinePush.configuration);
    CudaCandidateState* states = reinterpret<CudaCandidateState*>(timelinePush.states);
    VulkanTimelineResult* results = reinterpret<VulkanTimelineResult*>(timelinePush.results);
    VulkanFinishRefinementWork* work =
        reinterpret<VulkanFinishRefinementWork*>(timelinePush.observations);
    cuda__collision__CudaCollisionScratch* scratch =
        reinterpret<cuda__collision__CudaCollisionScratch*>(timelinePush.scratch);
    if (work[candidate].status != 1u) return;

    double lower = work[candidate].substepStartNs;
    double upper = lower + double(work[candidate].fullDt) * 1000000000.0;
    for (;;)
    {
        uint64_t intervalLower = uint64_t(floor(lower));
        uint64_t intervalUpper = uint64_t(ceil(upper));
        if (intervalUpper >= intervalLower &&
            intervalUpper - intervalLower <= 1u)
            break;
        uint64_t firstInterior = uint64_t(floor(lower)) + 1u;
        uint64_t upperCeiling = uint64_t(ceil(upper));
        if (upperCeiling == 0u || firstInterior >= upperCeiling) break;
        uint64_t lastInterior = upperCeiling - 1u;
        uint64_t candidateNs = firstInterior +
            (lastInterior - firstInterior) / 2u;
        float partialDt = float(
            (double(candidateNs) - work[candidate].substepStartNs) /
                1000000000.0);
        if (!(partialDt > 0.0f))
        {
            lower = double(candidateNs);
            continue;
        }
        uint preSubstepIndex = timelinePush.candidateCount + candidate;
        uint probeIndex = timelinePush.candidateCount * 2u + candidate;
        states[probeIndex] = states[preSubstepIndex];
        cuda::physics::Status physicsStatus = ExecuteVulkanFinishProbeSubstep(
            scene, configuration,
            reinterpret<CudaCandidatePhysicsState*>(states + probeIndex),
            partialDt, &(scratch[candidate]));
        if (physicsStatus != cuda::physics::Status::Success)
        {
            work[candidate].status = 2u;
            results[candidate].status = 6u;
            results[candidate].failureDetail = uint(physicsStatus) +
                1000u * uint(scratch[candidate].overflowReason) +
                100000u * scratch[candidate].collisionCount;
            return;
        }
        if (states[probeIndex].race.progress.raceCompleted)
            upper = double(candidateNs);
        else
            lower = double(candidateNs);
    }
    work[candidate].estimate.lowerBoundNs = uint64_t(floor(lower));
    work[candidate].estimate.upperBoundNs = uint64_t(ceil(upper));
    work[candidate].estimate.estimatedNs =
        work[candidate].estimate.upperBoundNs;
    if (work[candidate].estimate.lowerBoundNs >=
            work[candidate].estimate.upperBoundNs ||
        work[candidate].estimate.upperBoundNs -
            work[candidate].estimate.lowerBoundNs > 1u)
    {
        work[candidate].status = 2u;
        results[candidate].status = 6u;
        results[candidate].failureDetail = 900000002u;
        return;
    }
    states[candidate].finishTime.present = true;
    states[candidate].finishTime.value = work[candidate].estimate;
    work[candidate].status = 3u;
    results[candidate].status = 0u;
    results[candidate].failureTick = UINT32_MAX;
}
#endif

}
"""


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    # C++ inheritance makes the full candidate implicitly bind to its physics
    # base. Generated scalar-layout records are flattened, so device functions
    # operate directly on the ABI-identical full candidate record.
    reference_translator = ReferenceTranslator(source_root)
    parts = [args.types.read_text(encoding="utf-8"), PRELUDE]
    cuda_root = source_root / "src/simulation/backends/cuda"
    for filename in DEVICE_FILES:
        source_path = cuda_root / filename
        source = source_path.read_text(encoding="utf-8")
        source = reference_translator.translate(source_path, source)
        parts.append(f"\n// Translated from {filename}.\n")
        file_mapping = dict(mapping)
        if filename in ("cuda_collision.cuh", "cuda_collision_response.cuh"):
            prefix = "forevervalidator::simulation::cuda::collision::"
            for key, value in mapping.items():
                if key.startswith(prefix):
                    file_mapping[key.removeprefix(prefix)] = value
        translated = translate_common(source, file_mapping)
        if filename == "cuda_physics_step.cuh":
            translated = translated.replace(
                "inline Status Step<", "[noinline]\ninline Status Step<")
        if filename == "cuda_finish_time_refinement.cuh":
            translated = (
                "#if !defined(FOREVERVALIDATOR_VULKAN_TIMELINE_ONLY)\n"
                + translated
                + "\n#endif"
            )
        parts.append(translated)
    parts.append(TIMELINE_KERNEL)
    output = "\n".join(parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
