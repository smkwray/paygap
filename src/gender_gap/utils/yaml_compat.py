"""Small YAML compatibility shim.

Uses PyYAML when available and falls back to a tiny indentation-based parser
that supports the subset of YAML used by this repo's configs and tests.
"""

from __future__ import annotations

import ast


def load_yaml(text: str):
    """Load YAML text with a PyYAML fallback."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        parser = _MiniYamlParser(text)
        return parser.parse()


class _MiniYamlParser:
    def __init__(self, text: str):
        self.lines = [
            line.rstrip("\n")
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def parse(self):
        if not self.lines:
            return {}
        value, _ = self._parse_block(0, self._indent(self.lines[0]))
        return value

    def _parse_block(self, index: int, indent: int):
        if self.lines[index].lstrip().startswith("- "):
            return self._parse_list(index, indent)
        return self._parse_dict(index, indent)

    def _parse_dict(self, index: int, indent: int):
        result = {}
        i = index
        while i < len(self.lines):
            line = self.lines[i]
            current = self._indent(line)
            if current < indent:
                break
            if current > indent:
                raise ValueError(f"Unexpected indent in YAML near: {line}")
            stripped = line.strip()
            if stripped.startswith("- "):
                break
            key, _, raw_value = stripped.partition(":")
            key = key.strip().strip("'").strip('"')
            raw_value = raw_value.strip()
            if raw_value:
                result[key] = _parse_scalar(raw_value)
                i += 1
                continue
            if i + 1 < len(self.lines) and self._indent(self.lines[i + 1]) > indent:
                value, i = self._parse_block(i + 1, self._indent(self.lines[i + 1]))
                result[key] = value
            else:
                result[key] = None
                i += 1
        return result, i

    def _parse_list(self, index: int, indent: int):
        result = []
        i = index
        while i < len(self.lines):
            line = self.lines[i]
            current = self._indent(line)
            if current < indent:
                break
            if current > indent:
                raise ValueError(f"Unexpected indent in YAML near: {line}")
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            item = stripped[2:].strip()
            if not item:
                if i + 1 < len(self.lines) and self._indent(self.lines[i + 1]) > indent:
                    value, i = self._parse_block(i + 1, self._indent(self.lines[i + 1]))
                    result.append(value)
                else:
                    result.append(None)
                    i += 1
                continue
            if ":" in item:
                key, _, raw_value = item.partition(":")
                entry = {key.strip().strip("'").strip('"'): _parse_scalar(raw_value.strip()) if raw_value.strip() else None}
                i += 1
                while i < len(self.lines) and self._indent(self.lines[i]) > indent:
                    nested, i = self._parse_dict(i, self._indent(self.lines[i]))
                    entry.update(nested)
                result.append(entry)
                continue
            result.append(_parse_scalar(item))
            i += 1
        return result, i

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))


def _parse_scalar(value: str):
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
