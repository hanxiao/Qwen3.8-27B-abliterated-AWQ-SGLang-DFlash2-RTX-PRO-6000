import json
import logging
import re
from typing import Any, List, Optional

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import (
    get_schema_properties,
    infer_type_from_json_schema,
    safe_literal_eval,
)

logger = logging.getLogger(__name__)


class Qwen3CoderDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()

        # Sentinel tokens
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

        # Regex for non-streaming fallback
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        # Streaming State
        # Base class already initializes _buffer, we just use it directly
        # No need to check with hasattr - we control the lifecycle through inheritance

        # Index pointing to the next character to be processed in buffer
        self.parsed_pos: int = 0
        # Parameter count inside the current tool being processed, used to determine whether to add comma
        self.current_tool_param_count: int = 0
        # Flag indicating whether current tool has already sent '{'
        self.json_started: bool = False

        # [FIX] New state flag: mark whether inside tool_call structure block
        self.is_inside_tool_call: bool = False

        # Initialize attributes that were missing in the original PR
        self.current_func_name: Optional[str] = None

        self.current_streaming_param_name: Optional[str] = None
        self.current_streaming_param_prefix_sent: bool = False
        self.current_streaming_param_lead_checked: bool = False
        self.current_streaming_param_pending: str = ""

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def _get_arguments_config(
        self, func_name: str, tools: Optional[list[Tool]]
    ) -> dict:
        """Extract argument configuration for a function."""
        if tools is None:
            return {}
        for config in tools:
            try:
                config_type = config.type
                config_function = config.function
                config_function_name = config_function.name
            except AttributeError:
                continue

            if config_type == "function" and config_function_name == func_name:
                try:
                    params = config_function.parameters
                except AttributeError:
                    return {}

                if isinstance(params, dict):
                    properties = get_schema_properties(params)
                    if properties or "properties" in params:
                        return properties
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _get_param_type(self, param_schema: Any) -> str:
        """Infer the parser conversion type from a JSON schema parameter."""
        inferred_type = infer_type_from_json_schema(param_schema)
        if inferred_type is None:
            return "string"
        return str(inferred_type).strip().lower()

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        # Handle null value for any type
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        param_type = self._get_param_type(param_config[param_name])
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an integer in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                maybe_convert = (
                    False if "." in param_value or "e" in param_value.lower() else True
                )
                param_value: float = float(param_value)
                if maybe_convert and param_value.is_integer():
                    param_value = int(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float in tool "
                    f"'{func_name}', degenerating to string."
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean (`true` of `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except Exception:
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be parsed with json.loads in tool "
                        f"'{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = safe_literal_eval(param_value)
            except Exception:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value

    def _is_streamable_string_param(self, param_name: str, param_config: dict) -> bool:
        # Typed as _convert_param_value types it, so the streamed value equals the parsed
        # one; a parameter missing from the schema is kept as its raw string there too.
        if param_name not in param_config:
            return True
        param_type = self._get_param_type(param_config[param_name])
        return param_type in ["string", "str", "text", "varchar", "char", "enum"]

    def _json_string_fragment(self, value: str) -> str:
        return json.dumps(value, ensure_ascii=False)[1:-1]

    def _ensure_streaming_param_prefix(self, calls: list[ToolCallItem]) -> None:
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self.json_started = True

        if self.current_streaming_param_prefix_sent:
            return

        param_name = self.current_streaming_param_name
        json_key = json.dumps(param_name, ensure_ascii=False)
        prefix = f'{json_key}: "'
        if self.current_tool_param_count > 0:
            prefix = ", " + prefix

        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=prefix,
            )
        )
        self.current_streaming_param_prefix_sent = True

    def _find_streaming_param_terminator(self, text: str) -> Optional[tuple[int, int]]:
        candidates = []
        for token, token_len in [
            (self.parameter_end_token, len(self.parameter_end_token)),
            (self.parameter_prefix, 0),
            (self.function_end_token, 0),
            (self.tool_call_end_token, 0),
        ]:
            idx = text.find(token)
            if idx != -1:
                candidates.append((idx, token_len))

        if not candidates:
            return None
        return min(candidates, key=lambda x: x[0])

    def _held_streaming_param_suffix_len(self, text: str) -> int:
        held = 1 if text else 0
        for token in [
            self.parameter_end_token,
            self.parameter_prefix,
            self.function_end_token,
            self.tool_call_end_token,
        ]:
            max_len = min(len(token) - 1, len(text))
            for suffix_len in range(max_len, 0, -1):
                if token.startswith(text[-suffix_len:]):
                    held = max(held, min(len(text), suffix_len + 1))
                    break
        return held

    def _split_function_bodies(self, region: str) -> list[str]:
        # One body per <function=...>, cut at its </function>, or at the next <function=... when
        # the model omitted </function> (and maybe </tool_call>). Tool-call tags left at the end of
        # a body by such an omission are dropped. The regex approach merged two calls in that case.
        bodies = []
        for piece in region.split(self.tool_call_prefix)[1:]:
            body = piece.split(self.function_end_token, 1)[0].rstrip()
            stripped = True
            while stripped:
                stripped = False
                for tok in (self.tool_call_end_token, self.tool_call_start_token):
                    if body.endswith(tok):
                        body = body[: -len(tok)].rstrip()
                        stripped = True
            bodies.append(body)
        return bodies

    def _close_open_function(self, calls: list[ToolCallItem]) -> None:
        # The model started a new <tool_call> or <function=...> while the previous function was
        # still open (it omitted both </function> and </tool_call>): close the previous object.
        if self.current_func_name is None:
            return
        if not self.json_started:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="{"))
            self.json_started = True
        calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters="}"))
        self.current_func_name = None
        self._reset_streaming_param_state()

    def _reset_streaming_param_state(self) -> None:
        self.current_streaming_param_name = None
        self.current_streaming_param_prefix_sent = False
        self.current_streaming_param_lead_checked = False
        self.current_streaming_param_pending = ""

    def _emit_streaming_param_value(
        self, current_slice: str, calls: list[ToolCallItem]
    ) -> int:
        text = self.current_streaming_param_pending + current_slice
        self.current_streaming_param_pending = ""

        # One newline after <parameter=...> is layout, as in the non-streaming path:
        # dropped once, however the value is chunked.
        if not self.current_streaming_param_lead_checked and text:
            if text.startswith("\n"):
                text = text[1:]
            self.current_streaming_param_lead_checked = True

        terminator = self._find_streaming_param_terminator(text)
        if terminator is not None:
            end_pos, end_token_len = terminator
            value = text[:end_pos]
            if value.endswith("\n"):
                value = value[:-1]

            if not self.current_streaming_param_prefix_sent and value.lower() == "null":
                # _convert_param_value turns "null" into None for every type.
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True
                key = json.dumps(self.current_streaming_param_name, ensure_ascii=False)
                sep = ", " if self.current_tool_param_count > 0 else ""
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id, parameters=f"{sep}{key}: null"
                    )
                )
            else:
                self._ensure_streaming_param_prefix(calls)
                if value:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            parameters=self._json_string_fragment(value),
                        )
                    )
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters='"')
                )
            self.current_tool_param_count += 1
            self._reset_streaming_param_state()
            # A zero-length terminator (</function>, </tool_call>, <parameter=) may sit in the
            # held-back pending text, which the buffer no longer holds once an earlier increment
            # trimmed it. Put whatever follows the value back at the cursor so the main loop
            # sees the terminator; a negative consumed length would point before the buffer.
            self._buffer = self._buffer[: self.parsed_pos] + text[end_pos + end_token_len :]
            return 0

        held_len = self._held_streaming_param_suffix_len(text)
        emit_len = max(0, len(text) - held_len)
        if emit_len == 0 or (
            not self.current_streaming_param_prefix_sent
            and "null".startswith(text[:emit_len].lower())
        ):
            # Nothing to send yet, or the value may still turn out to be null.
            self.current_streaming_param_pending = text
            return len(current_slice)

        value = text[:emit_len]
        self.current_streaming_param_pending = text[emit_len:]
        self._ensure_streaming_param_prefix(calls)
        calls.append(
            ToolCallItem(
                tool_index=self.current_tool_id,
                parameters=self._json_string_fragment(value),
            )
        )
        return len(current_slice)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parsing for non-streaming scenarios."""
        if self.tool_call_start_token not in text:
            return StreamingParseResult(normal_text=text)

        calls = []
        try:
            # Simple cleanup of the text to find tool calls
            # Note: This is a simplified regex approach consistent with vLLM
            # Everything from the first <tool_call> on, split per function (see
            # _split_function_bodies); robust to omitted </function> and </tool_call>.
            raw_tool_calls = [text[text.find(self.tool_call_start_token) :]]

            tool_idx = 0
            for tool_content in raw_tool_calls:
                for func_body in self._split_function_bodies(tool_content):
                    if ">" not in func_body:
                        continue

                    name_end = func_body.index(">")
                    func_name = func_body[:name_end]
                    params_str = func_body[name_end + 1 :]

                    param_config = self._get_arguments_config(func_name, tools)
                    parsed_params = {}

                    for p_match in self.tool_call_parameter_regex.findall(params_str):
                        if ">" not in p_match:
                            continue
                        p_idx = p_match.index(">")
                        p_name = p_match[:p_idx]
                        p_val = p_match[p_idx + 1 :]
                        # Remove prefixing and trailing \n
                        if p_val.startswith("\n"):
                            p_val = p_val[1:]
                        if p_val.endswith("\n"):
                            p_val = p_val[:-1]

                        parsed_params[p_name] = self._convert_param_value(
                            p_val, p_name, param_config, func_name
                        )

                    calls.append(
                        ToolCallItem(
                            tool_index=tool_idx,
                            name=func_name,
                            parameters=json.dumps(parsed_params, ensure_ascii=False),
                        )
                    )
                    tool_idx += 1

            # Determine normal text (text before the first tool call)
            start_idx = text.find(self.tool_call_start_token)
            if start_idx == -1:
                start_idx = text.find(self.tool_call_prefix)
            normal_text = text[:start_idx] if start_idx > 0 else ""

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        """
        Robust cursor-based streaming parser.
        """
        self._buffer += new_text

        # Guard against empty buffer
        if not self._buffer:
            return StreamingParseResult()

        calls = []
        normal_text_chunks = []

        while True:
            # Working text slice
            current_slice = self._buffer[self.parsed_pos :]

            # Optimization: If almost empty, wait for more
            if not current_slice:
                break

            if self.current_streaming_param_name is not None:
                consumed_len = self._emit_streaming_param_value(current_slice, calls)
                self.parsed_pos += consumed_len
                continue

            # -------------------------------------------------------
            # 1. Priority detection: check if it's the start of Tool Call
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_start_token):
                self._close_open_function(calls)
                self.parsed_pos += len(self.tool_call_start_token)
                self.is_inside_tool_call = True
                continue

            # -------------------------------------------------------
            # 2. Function Name: <function=name>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_prefix):
                end_angle = current_slice.find(">")
                if end_angle != -1:
                    func_name = current_slice[len(self.tool_call_prefix) : end_angle]

                    self._close_open_function(calls)
                    self.current_tool_id += 1
                    self.current_tool_name_sent = True
                    self.current_tool_param_count = 0
                    self.json_started = False
                    self.current_func_name = func_name

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    self.parsed_pos += end_angle + 1
                    continue
                else:
                    # Incomplete tag
                    break

            # -------------------------------------------------------
            # 3. Parameter: <parameter=name>value...
            # -------------------------------------------------------
            if current_slice.startswith(self.parameter_prefix):
                name_end = current_slice.find(">")
                if name_end != -1:
                    param_name = current_slice[len(self.parameter_prefix) : name_end]
                    param_config = self._get_arguments_config(
                        self.current_func_name, tools
                    )
                    if self._is_streamable_string_param(param_name, param_config):
                        self._reset_streaming_param_state()
                        self.current_streaming_param_name = param_name
                        self.parsed_pos += name_end + 1
                        continue

                    value_start_idx = name_end + 1
                    rest_of_slice = current_slice[value_start_idx:]

                    # A parameter can end in multiple ways:
                    # 1. [Normal] Encounter </parameter>
                    # 2. [Abnormal] Encounter next <parameter=
                    # 3. [Abnormal] Encounter </function>
                    # 4. [Abnormal] Encounter </tool_call> (model omitted </function>, sglang#43150)
                    # So we need to find the smallest one as the parameter end position.
                    cand_end_param = rest_of_slice.find(self.parameter_end_token)
                    cand_next_param = rest_of_slice.find(self.parameter_prefix)
                    cand_end_func = rest_of_slice.find(self.function_end_token)
                    cand_end_tool = rest_of_slice.find(self.tool_call_end_token)

                    candidates = []
                    if cand_end_param != -1:
                        candidates.append(
                            (cand_end_param, len(self.parameter_end_token))
                        )
                    if cand_next_param != -1:
                        candidates.append((cand_next_param, 0))
                    if cand_end_func != -1:
                        candidates.append((cand_end_func, 0))
                    if cand_end_tool != -1:
                        candidates.append((cand_end_tool, 0))

                    if candidates:
                        best_cand = min(candidates, key=lambda x: x[0])
                        end_pos = best_cand[0]
                        end_token_len = best_cand[1]

                        raw_value = rest_of_slice[:end_pos]

                        # Cleanup value
                        if raw_value.startswith("\n"):
                            raw_value = raw_value[1:]
                        if raw_value.endswith("\n"):
                            raw_value = raw_value[:-1]

                        # JSON Construction
                        if not self.json_started:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id, parameters="{"
                                )
                            )
                            self.json_started = True

                        converted_val = self._convert_param_value(
                            raw_value, param_name, param_config, self.current_func_name
                        )

                        # Construct JSON fragment: "key": value
                        # Note: We must be careful with json.dumps to ensure valid JSON streaming
                        json_key_val = f"{json.dumps(param_name)}: {json.dumps(converted_val, ensure_ascii=False)}"

                        if self.current_tool_param_count > 0:
                            fragment = f", {json_key_val}"
                        else:
                            fragment = json_key_val

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id, parameters=fragment
                            )
                        )
                        self.current_tool_param_count += 1

                        # Advance cursor
                        total_len = (name_end + 1) + end_pos + end_token_len
                        self.parsed_pos += total_len
                        continue

                # Incomplete parameter tag or value
                break

            # -------------------------------------------------------
            # 4. Function End: </function>, or </tool_call> while a function is still
            #    open (the model omitted </function>; sglang#43150, fix from PR #43467).
            #    On </tool_call> only the object is closed; section 5 consumes the tag.
            # -------------------------------------------------------
            closes_func = current_slice.startswith(self.function_end_token)
            if closes_func or (
                self.current_func_name is not None
                and current_slice.startswith(self.tool_call_end_token)
            ):
                if not self.json_started:
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, parameters="{")
                    )
                    self.json_started = True

                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, parameters="}")
                )
                if closes_func:
                    self.parsed_pos += len(self.function_end_token)
                self.current_func_name = None
                self._reset_streaming_param_state()
                continue

            # -------------------------------------------------------
            # 5. Tool Call End: </tool_call>
            # -------------------------------------------------------
            if current_slice.startswith(self.tool_call_end_token):
                self.parsed_pos += len(self.tool_call_end_token)
                self.is_inside_tool_call = False  # [FIX] Exit tool call region
                continue

            # -------------------------------------------------------
            # 6. Handling content / whitespace / normal text
            # -------------------------------------------------------
            # If current position is not the start of a tag (i.e., doesn't start with <), it might be plain text,
            # or a newline between two tags.
            # But we need to be careful not to output truncated tags like "<fun" as text.

            next_open_angle = current_slice.find("<")

            if next_open_angle == -1:
                # This entire segment is plain text
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(current_slice)
                # [FIX] If inside tool call, discard this text (usually \n), don't append
                self.parsed_pos += len(current_slice)
                continue

            elif next_open_angle == 0:
                # Looks like a Tag, but doesn't match any known Tag above

                possible_tags = [
                    self.tool_call_start_token,
                    self.tool_call_end_token,
                    self.tool_call_prefix,
                    self.function_end_token,
                    self.parameter_prefix,
                    self.parameter_end_token,
                ]

                is_potential_tag = False
                for tag in possible_tags:
                    if tag.startswith(current_slice):
                        is_potential_tag = True
                        break

                if is_potential_tag:
                    break  # Wait for more
                else:
                    # Just a plain '<' symbol
                    if not self.is_inside_tool_call:
                        normal_text_chunks.append("<")
                    self.parsed_pos += 1
                    continue

            else:
                # '<' is in the middle
                text_segment = current_slice[:next_open_angle]
                if not self.is_inside_tool_call:
                    normal_text_chunks.append(text_segment)
                # [FIX] If inside tool call, discard whitespace/text before Tag
                self.parsed_pos += next_open_angle
                continue

        # Memory Cleanup: Slice the buffer
        # Keep unparsed part, discard parsed part
        if self.parsed_pos > 0:
            self._buffer = self._buffer[self.parsed_pos :]
            self.parsed_pos = 0

        normal_text = "".join(normal_text_chunks) if normal_text_chunks else ""
        # serving_chat yields an increment's normal text before its tool-call items. On
        # /v1/messages a text delta closes the open tool_use block, so a '}' that follows in the
        # same increment (e.g. "</function>\n</tool_call>\n" under speculative decoding) is dropped
        # with "Dropping tool_call argument delta with no open tool_use block". The newline
        # between calls carries nothing; drop whitespace-only text that rides with call items.
        if calls and normal_text and not normal_text.strip():
            normal_text = ""
        return StreamingParseResult(calls=calls, normal_text=normal_text)

    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def get_structural_tag_name(self) -> str:
        return "qwen_3_coder"
