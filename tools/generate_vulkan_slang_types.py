#!/usr/bin/env python3
"""Generate scalar-layout Slang declarations for the GPU ABI.

The host-side CUDA packing structures are the current ABI reference while the
Vulkan backend is brought up.  This generator uses Clang's record-layout data
instead of attempting to duplicate thousands of fields and padding rules by
hand.  Generated declarations use scalar members (including byte-sized bools)
so physical-storage-buffer pointers see exactly the same bytes as C++/CUDA.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from pathlib import Path

from clang import cindex


ROOT_TYPES = (
    "GmVec2",
    "GmVec3",
    "GmVec4",
    "GmQuat",
    "GmMat3",
    "GmIso4",
    "GmBoxAligned",
    "GmNat2",
    "GmLocalMaterialIndex",
    "CudaCandidateState",
    "CudaControlTick",
    "CudaTimelineObservation",
    "CudaPackedStaticConfigurationHeader",
    "CudaPackedSceneHeader",
    "CudaTuningCurveKey",
    "CudaVehicleCollisionShape",
    "CudaForceField",
    "CudaSceneActor",
    "CudaSceneSurface",
    "CudaSceneTriangle",
    "CudaSceneOctreeCell",
    "CudaSceneAccelerationCell",
    "CudaCollision",
    "CudaCollisionScratch",
    "CudaCollisionSearchTile",
    "CudaCollisionSearchScratch",
    "CudaRaceState",
    "VehicleMaterialBlendValues",
    "VehicleMaterialDefinition",
    "Refinement",
    "CudaSearchWindow",
    "CudaSearchChannel",
    "CudaSearchModifierConfiguration",
    "CudaSearchEvaluatorConfiguration",
    "CudaSearchConditionInstruction",
    "CudaSearchInputEvent",
    "CudaSearchIncumbent",
    "DeviceControlState",
    "DeviceSample",
)

ROOT_ENUMS = (
    "CudaHandlingSpecialization",
    "CudaTuningCurveId",
    "CudaTransmissionArrayId",
    "ReplayTuningCurveInterpolation",
    "CSceneVehicleCarWheelForceMode",
    "CSceneVehicleCarHandlingModel",
    "BlockRaceRole",
    "EChallengePlayMode",
    "CudaControlActionFlag",
    "GmSurf::EGmSurfType",
    "CHmsDyna::EDynamicType",
    "CSceneVehicle::EVehicleEvent",
    "forevervalidator::simulation::cuda::collision::Status",
    "forevervalidator::simulation::CudaSearchModifierKind",
    "forevervalidator::simulation::CudaSearchEvaluatorKind",
    "forevervalidator::simulation::CudaSearchConditionOpcode",
    "forevervalidator::simulation::CudaSearchConditionValue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path)
    return parser.parse_args()


def qualified_name(cursor: cindex.Cursor) -> str:
    pieces: list[str] = []
    current = cursor
    while current and current.kind != cindex.CursorKind.TRANSLATION_UNIT:
        if current.spelling:
            pieces.append(current.spelling)
        current = current.semantic_parent
    return "::".join(reversed(pieces))


def emitted_name(cursor: cindex.Cursor) -> str:
    name = qualified_name(cursor)
    name = name.removeprefix("forevervalidator::simulation::")
    name = name.removeprefix("forevervalidator::")
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def enum_value_name(cursor: cindex.Cursor) -> str:
    parent = cursor.semantic_parent
    return f"{emitted_name(parent)}_{cursor.spelling}"


class Generator:
    def __init__(self, source_root: Path):
        self.source_root = source_root
        self.records_by_short_name: dict[str, list[cindex.Cursor]] = {}
        self.enums_by_qualified_name: dict[str, cindex.Cursor] = {}
        self.record_dependencies: dict[str, set[str]] = {}
        self.records: OrderedDict[str, cindex.Cursor] = OrderedDict()
        self.enum_constants: OrderedDict[str, tuple[str, str, int]] = OrderedDict()
        self.replacements: dict[str, str] = {
            "std::uint8_t": "uint8_t",
            "std::int8_t": "int8_t",
            "std::uint16_t": "uint16_t",
            "std::int16_t": "int16_t",
            "std::uint32_t": "uint",
            "std::int32_t": "int",
            "std::uint64_t": "uint64_t",
            "std::int64_t": "int64_t",
            "std::size_t": "uint64_t",
        }
        self.tu = self._parse_translation_unit()
        self._index_declarations()

    def _parse_translation_unit(self) -> cindex.TranslationUnit:
        source = "\n".join(
            (
                '#include "simulation/backends/cuda/cuda_timeline_executor.h"',
                '#include "simulation/backends/cuda/cuda_static_configuration_storage.h"',
                '#include "simulation/backends/cuda/cuda_scene_storage.h"',
                '#include "simulation/backends/cuda/cuda_collision_layout.h"',
                '#include "simulation/backends/cuda/cuda_search_branch_state.cuh"',
                '#include "simulation/backends/cuda/cuda_search_winner_selection.cuh"',
                '#include <forevervalidator/finish_time.h>',
                "namespace forevervalidator::simulation::cuda::finish {",
                "struct Refinement {",
                "  bool present = false; bool failed = false; bool rejected = false;",
                "  forevervalidator::FinishTimeEstimate estimate{};",
                "};",
                "}",
            )
        )
        probe = str(self.source_root / "vulkan_slang_type_probe.cpp")
        resource_dir = os.popen("clang -print-resource-dir").read().strip()
        args = [
            "-x",
            "c++",
            "-std=c++17",
            f"-resource-dir={resource_dir}",
            f"-I{self.source_root / 'src'}",
            f"-I{self.source_root / 'include'}",
            "-I/opt/cuda/include",
            "-D__device__=",
            "-D__host__=",
            "-D__forceinline__=",
            "-D__global__=",
            "-D__constant__=",
        ]
        return cindex.Index.create().parse(
            probe, args=args, unsaved_files=[(probe, source)])

    def _index_declarations(self) -> None:
        errors = [
            diagnostic
            for diagnostic in self.tu.diagnostics
            if diagnostic.severity >= cindex.Diagnostic.Fatal
        ]
        if errors:
            raise RuntimeError("; ".join(error.spelling for error in errors))
        for cursor in self.tu.cursor.walk_preorder():
            if cursor.kind in (
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.CLASS_DECL,
            ) and cursor.is_definition():
                self.records_by_short_name.setdefault(cursor.spelling, []).append(cursor)
            elif cursor.kind == cindex.CursorKind.ENUM_DECL and cursor.is_definition():
                self.enums_by_qualified_name[qualified_name(cursor)] = cursor

    def find_record(self, short_name: str) -> cindex.Cursor:
        matches = self.records_by_short_name.get(short_name, [])
        project_matches = [
            match
            for match in matches
            if match.location.file
            and str(match.location.file).startswith(str(self.source_root))
        ]
        if len(project_matches) != 1:
            locations = ", ".join(str(match.location) for match in project_matches)
            raise RuntimeError(f"expected one project record {short_name}, got {locations}")
        return project_matches[0]

    @staticmethod
    def _strip_elaborated(type_: cindex.Type) -> cindex.Type:
        while type_.kind in (cindex.TypeKind.ELABORATED, cindex.TypeKind.TYPEDEF):
            canonical = type_.get_canonical()
            if canonical.kind == type_.kind and canonical.spelling == type_.spelling:
                break
            type_ = canonical
        return type_

    def _template_type(self, type_: cindex.Type) -> str | None:
        spelling = type_.spelling
        if "CudaFixedArray<" in spelling:
            element = self.type_name(type_.get_template_argument_type(0))
            instantiated = type_.get_declaration().displayname or spelling
            match = re.search(r",\s*([0-9]+)U?>$", instantiated)
            if not match:
                raise RuntimeError(f"unresolved fixed-array capacity: {spelling}")
            count = int(match.group(1))
            return f"CudaFixedArray<{element}, {count}>"
        if "CudaOptional<" in spelling:
            element = self.type_name(type_.get_template_argument_type(0))
            return f"CudaOptional<{element}>"
        if spelling.startswith("std::array<"):
            element = self.type_name(type_.get_template_argument_type(0))
            instantiated = type_.get_declaration().displayname or spelling
            match = re.search(r",\s*([0-9]+)>$", instantiated)
            if not match:
                raise RuntimeError(f"unresolved std::array capacity: {spelling}")
            count = int(match.group(1))
            return f"{element}[{count}]"
        spring = re.fullmatch(r"GmSpring<float>", spelling)
        if spring:
            return "GmSpring_float"
        return None

    def type_name(self, type_: cindex.Type) -> str:
        direct = self.replacements.get(type_.spelling)
        if direct:
            return direct
        template = self._template_type(type_)
        if template:
            return template
        type_ = self._strip_elaborated(type_)
        template = self._template_type(type_)
        if template:
            return template
        builtin = {
            cindex.TypeKind.BOOL: "uint8_t",
            cindex.TypeKind.CHAR_U: "uint8_t",
            cindex.TypeKind.UCHAR: "uint8_t",
            cindex.TypeKind.CHAR_S: "int8_t",
            cindex.TypeKind.SCHAR: "int8_t",
            cindex.TypeKind.USHORT: "uint16_t",
            cindex.TypeKind.SHORT: "int16_t",
            cindex.TypeKind.UINT: "uint",
            cindex.TypeKind.INT: "int",
            cindex.TypeKind.ULONG: "uint64_t" if type_.get_size() == 8 else "uint",
            cindex.TypeKind.LONG: "int64_t" if type_.get_size() == 8 else "int",
            cindex.TypeKind.ULONGLONG: "uint64_t",
            cindex.TypeKind.LONGLONG: "int64_t",
            cindex.TypeKind.FLOAT: "float",
            cindex.TypeKind.DOUBLE: "double",
        }
        if type_.kind in builtin:
            return builtin[type_.kind]
        if type_.kind == cindex.TypeKind.CONSTANTARRAY:
            return f"{self.type_name(type_.element_type)}[{type_.element_count}]"
        if type_.kind == cindex.TypeKind.POINTER:
            return f"{self.type_name(type_.get_pointee())}*"
        if type_.kind == cindex.TypeKind.ENUM:
            declaration = type_.get_declaration()
            self._add_enum(declaration)
            return self.type_name(declaration.enum_type)
        if type_.kind in (cindex.TypeKind.RECORD, cindex.TypeKind.UNEXPOSED):
            declaration = type_.get_declaration()
            if declaration and declaration.is_definition():
                self._add_record(declaration)
                return emitted_name(declaration)
        raise RuntimeError(
            f"unsupported type {type_.kind}: {type_.spelling} ({type_.get_size()} bytes)")

    def _add_enum(self, cursor: cindex.Cursor) -> None:
        name = qualified_name(cursor)
        if name in self.enum_constants:
            return
        underlying = self.type_name(cursor.enum_type)
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                replacement = enum_value_name(child)
                qualified = f"{name}::{child.spelling}"
                self.enum_constants[qualified] = (
                    replacement,
                    underlying,
                    child.enum_value,
                )
                self.replacements[qualified] = replacement
                parent_name = qualified_name(cursor.semantic_parent)
                if parent_name and not cursor.is_scoped_enum():
                    old = f"{parent_name}::{child.spelling}"
                    self.replacements[old] = replacement
        self.replacements[name] = underlying
        if cursor.spelling != "Status":
            self.replacements.setdefault(cursor.spelling, underlying)
            for child in cursor.get_children():
                if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                    replacement = enum_value_name(child)
                    if not cursor.is_scoped_enum():
                        self.replacements.setdefault(child.spelling, replacement)
                    self.replacements.setdefault(
                        f"{cursor.spelling}::{child.spelling}", replacement)
                    parent = cursor.semantic_parent
                    if parent is not None and parent.spelling:
                        self.replacements.setdefault(
                            f"{parent.spelling}::{child.spelling}", replacement)
                        self.replacements.setdefault(
                            f"{parent.spelling}::{cursor.spelling}::{child.spelling}",
                            replacement)

    def _add_record(self, cursor: cindex.Cursor) -> None:
        name = emitted_name(cursor)
        if cursor.spelling in ("CudaFixedArray", "CudaOptional", "GmSpring"):
            return
        if name in self.records:
            return
        self.records[name] = cursor
        qualified = qualified_name(cursor)
        self.replacements[qualified] = name
        self.replacements.setdefault(cursor.spelling, name)
        self.record_dependencies[name] = set()
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                base = child.type.get_declaration()
                if base and base.is_definition():
                    self._add_record(base)
                    self.record_dependencies[name].add(emitted_name(base))
            elif child.kind == cindex.CursorKind.FIELD_DECL:
                before = set(self.records)
                self.type_name(child.type)
                self.record_dependencies[name].update(set(self.records) - before)

    def add_roots(self) -> None:
        for root in ROOT_TYPES:
            self._add_record(self.find_record(root))
        for root in ROOT_ENUMS:
            matches = [cursor for name, cursor in self.enums_by_qualified_name.items()
                       if name == root or ("::" not in root and name.endswith("::" + root))]
            project_matches = [cursor for cursor in matches if cursor.location.file and
                               str(cursor.location.file).startswith(str(self.source_root))]
            if len(project_matches) != 1:
                raise RuntimeError(f"expected one project enum {root}, got {len(project_matches)}")
            cursor = project_matches[0]
            self._add_enum(cursor)
            if "::" not in root:
                self.replacements[cursor.spelling] = self.type_name(cursor.enum_type)
                qualified = qualified_name(cursor)
                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                        self.replacements[f"{cursor.spelling}::{child.spelling}"] = (
                            self.replacements[f"{qualified}::{child.spelling}"])

    def _flattened_fields(
        self, cursor: cindex.Cursor, base_offset: int = 0
    ) -> list[tuple[int, cindex.Cursor]]:
        fields: list[tuple[int, cindex.Cursor]] = []
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                base = child.type.get_declaration()
                fields.extend(self._flattened_fields(base, base_offset))
            elif child.kind == cindex.CursorKind.FIELD_DECL:
                bit_offset = child.get_field_offsetof()
                if bit_offset < 0 or bit_offset % 8:
                    raise RuntimeError(f"unsupported bit field {qualified_name(child)}")
                fields.append((base_offset + bit_offset // 8, child))
        return fields

    def _record_text(self, name: str, cursor: cindex.Cursor) -> str:
        size = cursor.type.get_size()
        align = cursor.type.get_align()
        if size < 0 or align < 0:
            raise RuntimeError(f"incomplete record {qualified_name(cursor)}")
        lines = [f"// C++ ABI: sizeof={size}, align={align}", f"struct {name}", "{"]
        offset = 0
        pad_index = 0
        fields = self._flattened_fields(cursor)
        for field_offset, field in fields:
            if field_offset < offset:
                raise RuntimeError(f"overlapping field {qualified_name(field)}")
            if field_offset > offset:
                lines.append(f"    uint8_t _abiPad{pad_index}[{field_offset - offset}];")
                pad_index += 1
                offset = field_offset
            field_type = self.type_name(field.type)
            array = re.fullmatch(r"(.+)\[([0-9]+)\]", field_type)
            if array:
                lines.append(f"    {array.group(1)} {field.spelling}[{array.group(2)}];")
            else:
                lines.append(f"    {field_type} {field.spelling};")
            field_size = field.type.get_size()
            if field_size < 0:
                raise RuntimeError(f"incomplete field {qualified_name(field)}")
            offset = field_offset + field_size
        if not fields:
            lines.append("    uint8_t _empty;")
            offset = 1
        if offset < size:
            lines.append(f"    uint8_t _abiPad{pad_index}[{size - offset}];")
        lines.extend(("};", ""))
        return "\n".join(lines)

    def _ordered_records(self) -> list[str]:
        ordered: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise RuntimeError(f"recursive value record {name}")
            visiting.add(name)
            for dependency in sorted(self.record_dependencies.get(name, ())):
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            ordered.append(name)

        for name in self.records:
            visit(name)
        return ordered

    def render(self) -> str:
        self.add_roots()
        constants = [
            "// Generated by tools/generate_vulkan_slang_types.py. Do not edit.",
            "// Host-compatible scalar layout for Vulkan physical storage buffers.",
            "",
            "static const uint UINT32_MAX = 0xffffffffu;",
            "static const uint64_t UINT64_MAX = 0xffffffffffffffffull;",
            "",
            "struct CudaFixedArray<T, let Capacity : int>",
            "{",
            "    uint count;",
            "    T values[Capacity];",
            "};",
            "",
            "struct CudaOptional<T>",
            "{",
            "    uint8_t present;",
            "    T value;",
            "};",
            "",
            "// C++ ABI: sizeof=20, align=4",
            "struct GmSpring_float",
            "{",
            "    float stiffness;",
            "    float damping;",
            "    float value;",
            "    float target;",
            "    float velocity;",
            "};",
            "",
        ]
        for _, (name, underlying, value) in self.enum_constants.items():
            suffix = "u" if value >= 0 else ""
            constants.append(
                f"static const {underlying} {name} = {value}{suffix};"
            )
        constants.append("")
        for name in self._ordered_records():
            constants.append(self._record_text(name, self.records[name]))
        # Static ABI constants referenced by the device implementation.
        candidate_schema = self._static_integer(
            "CudaCandidatePhysicsState", "SchemaVersion")
        configuration_schema = self._static_integer(
            "CudaPackedStaticConfigurationHeader", "SchemaVersion")
        configuration_magic = self._static_integer(
            "CudaPackedStaticConfigurationHeader", "Magic")
        scene_schema = self._static_integer(
            "CudaPackedSceneHeader", "SchemaVersion")
        scene_magic = self._static_integer(
            "CudaPackedSceneHeader", "Magic")
        constants.extend(
            (
                f"static const uint CudaCandidateState_SchemaVersion = {candidate_schema}u;",
                f"static const uint CudaPackedStaticConfigurationHeader_SchemaVersion = {configuration_schema}u;",
                f"static const uint64_t CudaPackedStaticConfigurationHeader_Magic = 0x{configuration_magic:x}ull;",
                f"static const uint CudaPackedSceneHeader_SchemaVersion = {scene_schema}u;",
                f"static const uint64_t CudaPackedSceneHeader_Magic = 0x{scene_magic:x}ull;",
                "",
            )
        )
        self.replacements.update(
            {
                "CudaCandidateState::SchemaVersion": "CudaCandidateState_SchemaVersion",
                "CudaPackedStaticConfigurationHeader::SchemaVersion": "CudaPackedStaticConfigurationHeader_SchemaVersion",
                "CudaPackedStaticConfigurationHeader::Magic": "CudaPackedStaticConfigurationHeader_Magic",
                "CudaPackedSceneHeader::SchemaVersion": "CudaPackedSceneHeader_SchemaVersion",
                "CudaPackedSceneHeader::Magic": "CudaPackedSceneHeader_Magic",
                "GmSpring<float>": "GmSpring_float",
            }
        )
        return "\n".join(constants)

    def _static_integer(self, record_name: str, member_name: str) -> int:
        for cursor in self.records_by_short_name.get(record_name, []):
            for child in cursor.get_children():
                if (child.spelling != member_name):
                    continue
                tokens = [token.spelling for token in child.get_tokens()]
                if "=" not in tokens:
                    continue
                value = "".join(tokens[tokens.index("=") + 1:])
                value = re.sub(r"[uUlL]+$", "", value)
                return int(value, 0)
        raise RuntimeError(
            f"missing static integer {record_name}::{member_name}")


def main() -> None:
    args = parse_args()
    generator = Generator(args.source_root.resolve())
    output = generator.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    if args.mapping:
        args.mapping.parent.mkdir(parents=True, exist_ok=True)
        args.mapping.write_text(
            json.dumps(generator.replacements, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
