"""
Interactive configuration editor for interaction-finder.
Provides a menu-driven interface for editing configuration files.
"""

# ============================================================================
# SECTION 1: Imports and Constants
# ============================================================================
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_origin,
    get_args,
)
from datetime import datetime
import copy
import sys
import tty
import termios
import inspect

try:
    import tomli
    import tomli_w
except ImportError:
    raise ImportError("tomli and tomli_w are required for configuration editing")

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, IntPrompt, Confirm, FloatPrompt
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.tree import Tree
from pydantic import ValidationError, BaseModel
from pydantic.fields import FieldInfo
from .settings import IfetcherConfig


# ============================================================================
# SECTION 2: Model Inspector for Dynamic Configuration
# ============================================================================
class ModelInspector:
    """Inspects Pydantic models to extract field metadata dynamically."""

    @staticmethod
    def get_field_type(field_info: FieldInfo, annotation: type) -> str:
        """Extract field type from annotation and field info."""
        # Handle Optional types
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Check for Union[T, None] (Optional[T])
        if origin is Union:
            # Filter out NoneType to get the actual type
            non_none_types = [arg for arg in args if arg is not type(None)]
            if len(non_none_types) == 1:
                annotation = non_none_types[0]
                origin = get_origin(annotation)
                args = get_args(annotation)

        # Handle generic types like List, Dict
        if origin is list or annotation is list:
            return "list"
        elif origin is dict or annotation is dict:
            # Check if this is a Dict[str, BaseModel] (structured dict)
            if args and len(args) == 2:
                key_type, value_type = args
                if (
                    key_type is str
                    and inspect.isclass(value_type)
                    and issubclass(value_type, BaseModel)
                ):
                    return "structured_dict"
            return "dict"
        elif annotation is bool:
            return "bool"
        elif annotation is int:
            return "int"
        elif annotation is float:
            return "float"
        elif annotation is str:
            return "string"
        else:
            return "string"  # Default fallback

    @staticmethod
    def get_field_constraints(field_info: FieldInfo) -> Dict[str, Any]:
        """Extract constraints from FieldInfo metadata."""
        constraints = {}

        # Check metadata list for constraint objects
        if hasattr(field_info, "metadata") and field_info.metadata:
            for constraint_obj in field_info.metadata:
                # Handle Ge (greater than or equal) constraints
                if hasattr(constraint_obj, "ge") and constraint_obj.ge is not None:
                    constraints["min"] = constraint_obj.ge
                # Handle Le (less than or equal) constraints
                elif hasattr(constraint_obj, "le") and constraint_obj.le is not None:
                    constraints["max"] = constraint_obj.le
                # Handle Gt (greater than) constraints
                elif hasattr(constraint_obj, "gt") and constraint_obj.gt is not None:
                    constraints["min"] = constraint_obj.gt + (
                        1 if isinstance(constraint_obj.gt, int) else 0.1
                    )
                # Handle Lt (less than) constraints
                elif hasattr(constraint_obj, "lt") and constraint_obj.lt is not None:
                    constraints["max"] = constraint_obj.lt - (
                        1 if isinstance(constraint_obj.lt, int) else 0.1
                    )
                # Handle MinLen constraints
                elif (
                    hasattr(constraint_obj, "min_length")
                    and constraint_obj.min_length is not None
                ):
                    constraints["min_length"] = constraint_obj.min_length
                # Handle MaxLen constraints
                elif (
                    hasattr(constraint_obj, "max_length")
                    and constraint_obj.max_length is not None
                ):
                    constraints["max_length"] = constraint_obj.max_length

        return constraints

    @staticmethod
    def extract_choices_from_validator(
        model: type, field_name: str
    ) -> Optional[List[str]]:
        """Extract choices from field validators."""
        # Look for methods that validate this specific field
        validator_method_name = f"validate_{field_name}"
        if hasattr(model, validator_method_name):
            try:
                validator_func = getattr(model, validator_method_name)
                source = inspect.getsource(validator_func)

                # Look for common patterns that define choices
                import re

                # Pattern for "not in (" followed by quoted values
                match = re.search(r"not in \(([^)]+)\)", source)
                if match:
                    choices_str = match.group(1)
                    # Parse quoted strings
                    choices = re.findall(r'["\']([^"\']+)["\']', choices_str)
                    return choices

                # Pattern for "not in [" followed by quoted values
                match = re.search(r"not in \[([^\]]+)\]", source)
                if match:
                    choices_str = match.group(1)
                    choices = re.findall(r'["\']([^"\']+)["\']', choices_str)
                    return choices

                # Pattern for variable assignment followed by "not in variable"
                # Look for: valid_modes = [...] followed by if v not in valid_modes
                var_assignment = re.search(r"(\w+)\s*=\s*\[([^\]]+)\]", source)
                if var_assignment:
                    var_name = var_assignment.group(1)
                    choices_str = var_assignment.group(2)
                    # Check if this variable is used in not in check
                    if f"not in {var_name}" in source:
                        choices = re.findall(r'["\']([^"\']+)["\']', choices_str)
                        return choices

            except Exception:
                pass
        return None

    @classmethod
    def inspect_model(cls, model: type, prefix: str = "") -> Dict[str, Dict[str, Any]]:
        """Recursively inspect a Pydantic model and return field metadata."""
        metadata = {}

        if not issubclass(model, BaseModel):
            return metadata

        for field_name, field_info in model.model_fields.items():
            field_path = f"{prefix}.{field_name}" if prefix else field_name
            annotation = model.model_fields[field_name].annotation

            # Handle Optional types first
            origin = get_origin(annotation)
            args = get_args(annotation)

            # Handle Optional[NestedModel]
            if origin is Union:
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    annotation = non_none_types[0]
                    origin = get_origin(annotation)

            # Check if this is a BaseModel subclass - if so, recurse but don't create metadata for it
            if (
                inspect.isclass(annotation)
                and issubclass(annotation, BaseModel)
                and annotation is not BaseModel
            ):
                # Recursively inspect nested BaseModel fields (don't create metadata for the container)
                nested_meta = cls.inspect_model(annotation, field_path)
                metadata.update(nested_meta)
            else:
                # This is a leaf field - create metadata for it
                field_type = cls.get_field_type(field_info, annotation)
                constraints = cls.get_field_constraints(field_info)

                field_meta = {
                    "type": field_type,
                    "help": field_info.description or f"Configuration for {field_name}",
                    **constraints,
                }

                # For structured dicts, store the value model type for better formatting
                origin = get_origin(annotation)
                if field_type == "structured_dict" and origin is dict:
                    args = get_args(annotation)
                    if len(args) == 2:
                        _, value_type = args
                        field_meta["value_model"] = value_type.__name__

                # Check if field is nullable
                if get_origin(annotation) is Union:
                    args = get_args(annotation)
                    if type(None) in args:
                        field_meta["nullable"] = True

                # Look for choices in validators
                choices = cls.extract_choices_from_validator(model, field_name)
                if choices:
                    field_meta["type"] = "choice"
                    field_meta["choices"] = choices

                metadata[field_path] = field_meta

        return metadata

    @classmethod
    def build_menu_structure(cls, model: type) -> Dict[str, Dict[str, Any]]:
        """Build menu structure dynamically from model hierarchy."""
        structure = {}
        counter = 1

        # First, add sections for nested BaseModel classes
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation

            # Handle Optional types
            origin = get_origin(annotation)
            if origin is Union:
                args = get_args(annotation)
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    annotation = non_none_types[0]

            # Only create sections for nested BaseModel classes
            if (
                inspect.isclass(annotation)
                and issubclass(annotation, BaseModel)
                and annotation is not BaseModel
            ):
                # Get the class docstring for the section title
                title = (
                    getattr(annotation, "__doc__", "").strip().split("\n")[0]
                    if annotation.__doc__
                    else field_name.title()
                )
                if not title:
                    title = field_name.replace("_", " ").title()

                # Clean up redundant prefixes for better readability
                title = cls._clean_title(title)

                structure[str(counter)] = {
                    "title": title,
                    "section": field_name,
                    "subsections": cls._build_subsections(annotation, field_name),
                }
                counter += 1

        # Check if there are any root-level fields (non-BaseModel fields)
        root_fields = []
        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation
            origin = get_origin(annotation)
            if origin is Union:
                args = get_args(annotation)
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    annotation = non_none_types[0]

            # If not a nested BaseModel, it's a root field
            if not (
                inspect.isclass(annotation)
                and issubclass(annotation, BaseModel)
                and annotation is not BaseModel
            ):
                root_fields.append(field_name)

        # If there are root-level fields, add a "General Settings" section
        if root_fields:
            structure[str(counter)] = {
                "title": "General Settings",
                "section": "general",
                "subsections": {"1": {"title": "General Configuration", "prefix": ""}},
            }

        return structure

    @staticmethod
    def _clean_title(title: str) -> str:
        """Clean up redundant prefixes and make titles more concise."""
        # Remove common redundant prefixes
        prefixes_to_remove = ["Configuration for ", "Settings for ", "Config for "]

        for prefix in prefixes_to_remove:
            if title.startswith(prefix):
                title = title[len(prefix) :]
                break

        # Remove trailing periods
        title = title.rstrip(".")

        # Simplify some common patterns
        replacements = {
            "web scraping tool": "Web Scraping",
            "search engine integration": "Search Engine",
            "file paths": "File Paths",
            "research tools and commands": "External Research",
            "task definition and parameters": "Task Definition",
            "document handling": "Document Processing",
            "document summarization parameters": "Summarization",
            "document grouping within workflow": "Document Grouping",
            "file paths and caching": "Output & Caching",
        }

        for old, new in replacements.items():
            if old in title.lower():
                title = new
                break

        # Capitalize first letter if not already done
        if title:
            title = title[0].upper() + title[1:]

        return title

    @classmethod
    def _build_subsections(
        cls, model: type, parent_prefix: str
    ) -> Dict[str, Dict[str, str]]:
        """Build subsections for nested models."""
        subsections = {}
        counter = 1

        for field_name, field_info in model.model_fields.items():
            annotation = field_info.annotation

            # Handle Optional types
            origin = get_origin(annotation)
            if origin is Union:
                args = get_args(annotation)
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    annotation = non_none_types[0]

            if (
                inspect.isclass(annotation)
                and issubclass(annotation, BaseModel)
                and annotation is not BaseModel
            ):
                # Get class docstring for title
                title = (
                    getattr(annotation, "__doc__", "").strip().split("\n")[0]
                    if annotation.__doc__
                    else field_name.title()
                )
                if not title:
                    title = field_name.replace("_", " ").title()

                # Clean up redundant prefixes
                title = cls._clean_title(title)

                subsections[str(counter)] = {
                    "title": title,
                    "prefix": f"{parent_prefix}.{field_name}",
                }
                counter += 1

        # If no nested models, create a generic subsection for the fields
        if not subsections:
            # For subsections without nested models, use a simplified title
            # This avoids redundant titles like "Task Definition -> Task Definition"
            subsections["1"] = {"title": "Settings", "prefix": parent_prefix}

        return subsections


class ConfigValueFormatter:
    """Smart formatters for different types of configuration values."""

    @staticmethod
    def format_value(
        value: Any, field_type: str, field_name: str = "", metadata: dict = None
    ) -> str:
        """Format a configuration value for display in the tree."""
        metadata = metadata or {}

        if value is None:
            return "[dim]None[/dim]"

        if field_type == "bool":
            return "[green]true[/green]" if value else "[red]false[/red]"

        elif field_type in ("int", "float"):
            return f"[cyan]{value}[/cyan]"

        elif field_type == "string":
            return ConfigValueFormatter._format_string(value, field_name)

        elif field_type == "list":
            return ConfigValueFormatter._format_list(value, field_name)

        elif field_type == "structured_dict":
            return ConfigValueFormatter._format_structured_dict(
                value, metadata.get("value_model", "")
            )

        elif field_type == "dict":
            return ConfigValueFormatter._format_dict(value, field_name)

        else:
            return str(value)

    @staticmethod
    def _format_string(value: str, field_name: str) -> str:
        """Format string values with smart truncation."""
        if not value:
            return '[dim]""[/dim]'

        # Show path templates nicely
        if "{" in value and "}" in value:
            if len(value) > 25:
                return f"[bright_green]{value[:22]}…[/bright_green]"
            return f"[bright_green]{value}[/bright_green]"

        # Regular strings
        if len(value) > 20:
            return f"[bright_green]{value[:17]}…[/bright_green]"
        return f"[bright_green]{value}[/bright_green]"

    @staticmethod
    def _format_list(value: list, field_name: str) -> str:
        """Format list values with smart summaries."""
        if not value:
            return "[dim][][/dim]"

        count = len(value)
        if count == 0:
            return "[dim][][/dim]"

        # For short lists of strings, show them
        if count <= 3 and all(isinstance(item, str) for item in value):
            items_str = ", ".join(str(item) for item in value)
            if len(items_str) <= 25:
                return f"[magenta][{items_str}][/magenta]"

        # For longer lists or complex items, show count with context
        if "site" in field_name.lower():
            return f"[magenta][{count} sites][/magenta]"
        elif "tool" in field_name.lower():
            return f"[magenta][{count} tools][/magenta]"
        else:
            return f"[magenta][{count} items][/magenta]"

    @staticmethod
    def _format_structured_dict(value: dict, value_model: str) -> str:
        """Format structured dictionaries like agents, kinds."""
        if not value:
            return "[dim]{}[/dim]"

        count = len(value)
        if count == 0:
            return "[dim]{}[/dim]"

        keys = list(value.keys())

        # Show configured items with their names
        if count <= 3:
            keys_str = ", ".join(keys)
            if value_model:
                return f"[yellow]{{{keys_str}}}[/yellow] [dim]({count} {value_model.lower()}s)[/dim]"
            else:
                return f"[yellow]{{{keys_str}}}[/yellow]"
        else:
            if value_model:
                return f"[yellow]{{{count} {value_model.lower()}s configured}}[/yellow]"
            else:
                return f"[yellow]{{{count} items}}[/yellow]"

    @staticmethod
    def _format_dict(value: dict, field_name: str) -> str:
        """Format generic dictionaries."""
        if not value:
            return "[dim]{}[/dim]"

        count = len(value)
        if count == 0:
            return "[dim]{}[/dim]"

        # For mode overrides or similar
        if "mode" in field_name.lower():
            return f"[yellow]{{{count} modes}}[/yellow]"
        else:
            return f"[yellow]{{{count} keys}}[/yellow]"


# ============================================================================
# SECTION 3: Keyboard Input Handler
# ============================================================================
class KeyboardInput:
    """Handle keyboard input for navigation."""

    @staticmethod
    def get_key():
        """Get a single key press without requiring Enter."""
        if sys.platform == "win32":
            import msvcrt

            key = msvcrt.getch()
            if key == b"\xe0":  # Special key prefix on Windows
                key = msvcrt.getch()
                key_map = {
                    b"H": "UP",
                    b"P": "DOWN",
                    b"K": "LEFT",
                    b"M": "RIGHT",
                    b"S": "DELETE",
                }
                return key_map.get(key, key.decode("utf-8", errors="ignore"))
            elif key == b"\x08":  # Backspace on Windows
                return "BACKSPACE"
            elif key == b"\r":  # Enter on Windows
                return "ENTER"
            return key.decode("utf-8", errors="ignore")
        else:
            # Unix/Linux/Mac
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)

                # Handle escape sequences for arrow keys and special keys
                if key == "\x1b":  # ESC sequence
                    next_chars = sys.stdin.read(2)
                    key += next_chars
                    if key == "\x1b[A":
                        return "UP"
                    elif key == "\x1b[B":
                        return "DOWN"
                    elif key == "\x1b[C":
                        return "RIGHT"
                    elif key == "\x1b[D":
                        return "LEFT"
                    elif key == "\x1b[1":
                        # Check for Alt+Arrow sequences: \x1b[1;3A, \x1b[1;3B, etc.
                        extra_chars = sys.stdin.read(3)  # Read ";3X"
                        full_seq = key + extra_chars
                        if full_seq == "\x1b[1;3A":
                            return "ALT+UP"
                        elif full_seq == "\x1b[1;3B":
                            return "ALT+DOWN"
                        elif full_seq == "\x1b[1;3C":
                            return "ALT+RIGHT"
                        elif full_seq == "\x1b[1;3D":
                            return "ALT+LEFT"
                        # Check for Shift+Arrow sequences: \x1b[1;2A, \x1b[1;2B, etc.
                        elif full_seq == "\x1b[1;2A":
                            return "SHIFT+UP"
                        elif full_seq == "\x1b[1;2B":
                            return "SHIFT+DOWN"
                        elif full_seq == "\x1b[1;2C":
                            return "SHIFT+RIGHT"
                        elif full_seq == "\x1b[1;2D":
                            return "SHIFT+LEFT"
                        # Ctrl+Arrow sequences: \x1b[1;5X
                        elif full_seq == "\x1b[1;5A":
                            return "CTRL+UP"
                        elif full_seq == "\x1b[1;5B":
                            return "CTRL+DOWN"
                        elif full_seq == "\x1b[1;5C":
                            return "CTRL+RIGHT"
                        elif full_seq == "\x1b[1;5D":
                            return "CTRL+LEFT"
                        # If not a recognized modifier sequence, treat as ESC
                        return "ESC"
                    elif next_chars == "[3":
                        # Delete key sequence \x1b[3~ - need to read the ~
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            return "DELETE"
                    elif next_chars == "[H":
                        # Home key sequence \x1b[H
                        return "HOME"
                    elif next_chars == "[F":
                        # End key sequence \x1b[F
                        return "END"
                    elif next_chars == "[1":
                        # Home/End keys can also be \x1b[1~ and \x1b[4~
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            return "HOME"
                    elif next_chars == "[4":
                        # End key sequence \x1b[4~ - need to read the ~
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            return "END"
                    elif next_chars == "[5":
                        # Page Up key sequence \x1b[5~ - need to read the ~
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            return "PAGE_UP"
                    elif next_chars == "[6":
                        # Page Down key sequence \x1b[6~ - need to read the ~
                        tilde = sys.stdin.read(1)
                        if tilde == "~":
                            return "PAGE_DOWN"
                    elif next_chars == "[Z":
                        # Shift+Tab sequence \x1b[Z
                        return "SHIFT+TAB"
                    return "ESC"
                elif key == "\r":
                    return "ENTER"
                elif key == "\t":
                    return "TAB"
                elif key == "\x7f" or key == "\x08":  # Backspace/Delete
                    return "BACKSPACE"
                elif key == " ":
                    return "SPACE"
                elif ord(key) == 3:  # ⌃C
                    raise KeyboardInterrupt
                elif ord(key) == 4:  # ⌃D (EOF)
                    return "CTRL+D"

                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ============================================================================
# SECTION 4: Interactive Menu System
# ============================================================================
class InteractiveMenu:
    """Interactive menu that can be navigated with arrow keys."""

    def __init__(self, console: Console, title: str = "Menu"):
        self.console = console
        self.title = title
        self.items = []
        self.selected_index = 0
        self.show_help = True

    def add_item(
        self, label: str, value: Any = None, description: str = "", style: str = ""
    ):
        """Add an item to the menu."""
        self.items.append(
            {
                "label": label,
                "value": value if value is not None else label,
                "description": description,
                "style": style,
            }
        )

    def add_separator(self):
        """Add a visual separator."""
        self.items.append({"separator": True})

    def render(self) -> None:
        """Render the menu."""
        self.console.clear()

        # Create menu table
        table = Table(title=self.title, show_header=False, box=None, padding=(0, 2))
        table.add_column("Selector", width=3)
        table.add_column("Item", min_width=20)
        table.add_column("Description", style="dim", min_width=25)

        for i, item in enumerate(self.items):
            if item.get("separator"):
                table.add_row("", "", "")
                continue

            # Selection indicator
            if i == self.selected_index:
                selector = "▶"
                label_style = "bold cyan"
            else:
                selector = " "
                label_style = item.get("style", "")

            # Format label with style
            label = item["label"]
            if label_style:
                label = f"[{label_style}]{label}[/{label_style}]"

            table.add_row(selector, label, item.get("description", ""))

        self.console.print(table)

        # Show help
        if self.show_help:
            help_text = Text()
            help_text.append("Navigation: ", style="dim")
            help_text.append("↑↓ ", style="bold")
            help_text.append("move ", style="dim")
            help_text.append("Enter ", style="bold")
            help_text.append("select ", style="dim")
            help_text.append("← ", style="bold")
            help_text.append("back ", style="dim")
            help_text.append("Del/Backspace ", style="bold")
            help_text.append("back ", style="dim")
            help_text.append("q ", style="bold")
            help_text.append("quit", style="dim")

            self.console.print(Panel(help_text, style="dim"))

    def navigate(self) -> Optional[Any]:
        """Handle navigation and return selected value."""
        # Skip separators when setting initial position
        while self.selected_index < len(self.items) and self.items[
            self.selected_index
        ].get("separator", False):
            self.selected_index += 1

        while True:
            self.render()

            try:
                key = KeyboardInput.get_key()

                if key == "UP":
                    self.selected_index = max(0, self.selected_index - 1)
                    # Skip separators
                    while self.selected_index >= 0 and self.items[
                        self.selected_index
                    ].get("separator", False):
                        self.selected_index -= 1
                    if self.selected_index < 0:
                        self.selected_index = (
                            len(
                                [
                                    item
                                    for item in self.items
                                    if not item.get("separator", False)
                                ]
                            )
                            - 1
                        )

                elif key == "DOWN":
                    self.selected_index = min(
                        len(self.items) - 1, self.selected_index + 1
                    )
                    # Skip separators
                    while self.selected_index < len(self.items) and self.items[
                        self.selected_index
                    ].get("separator", False):
                        self.selected_index += 1
                    if self.selected_index >= len(self.items):
                        # Find first non-separator
                        self.selected_index = 0
                        while self.selected_index < len(self.items) and self.items[
                            self.selected_index
                        ].get("separator", False):
                            self.selected_index += 1

                elif key in ("ENTER", "\r", "\n"):
                    if self.selected_index < len(self.items):
                        return self.items[self.selected_index]["value"]

                elif key in ("LEFT", "BACKSPACE", "DELETE", "ESC", "q", "Q"):
                    return None

            except KeyboardInterrupt:
                return None


# ============================================================================
# SECTION 5: Value Editors
# ============================================================================
class ValueEditor:
    """Type-specific value editing functions."""

    @staticmethod
    def edit_string(
        current: str, field_name: str, help_text: str = ""
    ) -> Optional[str]:
        """Edit a string value."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")

        prompt_text = f"Enter value for {field_name.split('.')[-1]}"
        if current:
            prompt_text += f" (current: {current})"

        result = Prompt.ask(prompt_text, default=current or "")
        return result if result != current else None

    @staticmethod
    def edit_int(
        current: int,
        field_name: str,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        help_text: str = "",
    ) -> Optional[int]:
        """Edit an integer value."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")

        range_text = ""
        if min_val is not None and max_val is not None:
            range_text = f" (range: {min_val}-{max_val})"
        elif min_val is not None:
            range_text = f" (min: {min_val})"
        elif max_val is not None:
            range_text = f" (max: {max_val})"

        prompt_text = f"Enter {field_name.split('.')[-1]}{range_text}"

        while True:
            try:
                result = IntPrompt.ask(prompt_text, default=current)

                if min_val is not None and result < min_val:
                    console.print(f"[red]Value must be >= {min_val}[/red]")
                    continue

                if max_val is not None and result > max_val:
                    console.print(f"[red]Value must be <= {max_val}[/red]")
                    continue

                return result if result != current else None

            except KeyboardInterrupt:
                return None

    @staticmethod
    def edit_float(
        current: float,
        field_name: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        help_text: str = "",
    ) -> Optional[float]:
        """Edit a float value."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")

        range_text = ""
        if min_val is not None and max_val is not None:
            range_text = f" (range: {min_val}-{max_val})"
        elif min_val is not None:
            range_text = f" (min: {min_val})"
        elif max_val is not None:
            range_text = f" (max: {max_val})"

        prompt_text = f"Enter {field_name.split('.')[-1]}{range_text}"

        while True:
            try:
                result = FloatPrompt.ask(prompt_text, default=current)

                if min_val is not None and result < min_val:
                    console.print(f"[red]Value must be >= {min_val}[/red]")
                    continue

                if max_val is not None and result > max_val:
                    console.print(f"[red]Value must be <= {max_val}[/red]")
                    continue

                return result if result != current else None

            except KeyboardInterrupt:
                return None

    @staticmethod
    def edit_bool(
        current: bool, field_name: str, help_text: str = ""
    ) -> Optional[bool]:
        """Edit a boolean value."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")

        current_text = "enabled" if current else "disabled"
        prompt_text = (
            f"{field_name.split('.')[-1]} is currently {current_text}. Enable?"
        )

        result = Confirm.ask(prompt_text, default=current)
        return result if result != current else None

    @staticmethod
    def edit_choice(
        current: str, field_name: str, choices: List[str], help_text: str = ""
    ) -> Optional[str]:
        """Edit a choice value from predefined options."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")
            console.print()

        # Create interactive menu for choices
        menu = InteractiveMenu(console, title=f"Choose {field_name.split('.')[-1]}")

        for choice in choices:
            marker = " (current)" if choice == current else ""
            style = "green" if choice == current else "default"
            menu.add_item(f"{choice}{marker}", choice, style=style)

        menu.add_separator()
        menu.add_item("← Cancel", None, "Keep current value", "yellow")

        result = menu.navigate()
        return result if result != current else None

    @staticmethod
    def edit_list(
        current: List, field_name: str, help_text: str = ""
    ) -> Optional[List]:
        """Edit a list value."""
        console = Console()
        if help_text:
            console.print(f"[dim]{help_text}[/dim]")
            console.print()

        working_list = list(current) if current else []

        while True:
            # Display current list
            if working_list:
                list_table = Table(title=f"Current {field_name.split('.')[-1]}")
                list_table.add_column("Index", style="cyan")
                list_table.add_column("Value")

                for i, item in enumerate(working_list):
                    list_table.add_row(str(i + 1), str(item))
                console.print(list_table)
            else:
                console.print(f"[dim]{field_name.split('.')[-1]} is empty[/dim]")

            # Create interactive menu for list actions
            menu = InteractiveMenu(console, title="List Actions")

            menu.add_item("Add item", "add", "Add a new item to the list", "green")
            if working_list:
                menu.add_item(
                    "Remove item", "remove", "Remove an item from the list", "red"
                )
                menu.add_item("Clear all", "clear", "Remove all items", "red")
            menu.add_item("Finish editing", "finish", "Save changes and exit", "cyan")

            action = menu.navigate()

            if action == "add":
                item = Prompt.ask("Enter new item")
                if item and item not in working_list:
                    working_list.append(item)
                    console.print(f"[green]Added: {item}[/green]")
                elif item in working_list:
                    console.print(f"[yellow]Item '{item}' already exists[/yellow]")
            elif action == "remove" and working_list:
                # Create menu to select item to remove
                remove_menu = InteractiveMenu(console, title="Select item to remove")
                for i, item in enumerate(working_list):
                    remove_menu.add_item(f"{i + 1}. {item}", i, str(item))
                remove_menu.add_separator()
                remove_menu.add_item(
                    "← Cancel", None, "Don't remove anything", "yellow"
                )

                remove_idx = remove_menu.navigate()
                if remove_idx is not None:
                    removed = working_list.pop(remove_idx)
                    console.print(f"[green]Removed: {removed}[/green]")
            elif action == "clear":
                if Confirm.ask("Clear all items?"):
                    working_list.clear()
                    console.print("[green]All items cleared[/green]")
            elif action == "finish" or action is None:
                break

        return working_list if working_list != current else None


# ============================================================================
# SECTION 6: Structured Dict Editor
# ============================================================================
class StructuredDictEditor:
    """Interactive editor for Dict[str, BaseModel] configurations with tab navigation."""

    def __init__(
        self,
        current_value: dict,
        value_model: type,
        field_path: str,
        help_text: str = "",
    ):
        self.entries = current_value.copy() if current_value else {}
        self.value_model = value_model
        self.field_path = field_path
        self.help_text = help_text
        self.console = Console()

        # Navigation state
        self.selected_entry_index = 0
        self.focused_field = 0  # 0 = key, 1+ = value fields
        self.mode = "browse"  # browse | edit_field | add_entry

        # Get model field information for form layout
        self.model_fields = self._get_model_fields()

        # Current editing state
        self.edit_key = ""
        self.edit_value = None
        self.original_key = ""

    def _get_model_fields(self) -> list:
        """Extract field information from the value model."""
        fields = [("_key", "string", "Entry identifier")]  # Field 0: the key

        # Add fields from the BaseModel
        if hasattr(self.value_model, "model_fields"):
            for name, field_info in self.value_model.model_fields.items():
                field_type = ModelInspector.get_field_type(
                    field_info, field_info.annotation
                )
                description = field_info.description or f"Configuration for {name}"
                fields.append((name, field_type, description))

        return fields

    def edit_interactive(self) -> dict:
        """Main interactive editing loop with tab navigation."""
        if not self.entries:
            # If empty, start by adding first entry
            return self._prompt_for_first_entry()

        entry_keys = list(self.entries.keys())
        self.selected_entry_index = 0

        while True:
            self._render_display()
            key = self._get_key()

            if self.mode == "browse":
                if key == "UP":
                    self.selected_entry_index = max(0, self.selected_entry_index - 1)
                elif key == "DOWN":
                    self.selected_entry_index = min(
                        len(entry_keys) - 1, self.selected_entry_index + 1
                    )
                elif key == "ENTER":
                    self.mode = "edit_field"
                    self.focused_field = 0
                elif key == "+":
                    self._add_new_entry()
                    entry_keys = list(self.entries.keys())
                elif key == "DELETE":
                    self._delete_current_entry()
                    entry_keys = list(self.entries.keys())
                    if not entry_keys:
                        break
                elif key == "ESCAPE":
                    return self.entries

            elif self.mode == "edit_field":
                if key == "TAB":
                    self.focused_field = (self.focused_field + 1) % len(
                        self.model_fields
                    )
                elif key == "SHIFT+TAB":
                    self.focused_field = (self.focused_field - 1) % len(
                        self.model_fields
                    )
                elif key == "ENTER":
                    self._edit_current_field()
                elif key == "ESCAPE":
                    self.mode = "browse"

        return self.entries

    def _prompt_for_first_entry(self) -> dict:
        """Prompt to add the first entry when dict is empty."""
        # TODO: Implement proper input handling
        return self.entries

    def _add_new_entry(self):
        """Add a new entry to the dict."""
        # TODO: Implement key input and value creation
        pass

    def _delete_current_entry(self):
        """Delete the currently selected entry."""
        # TODO: Implement deletion with confirmation
        pass

    def _edit_current_field(self):
        """Edit the currently focused field."""
        # TODO: Implement field editing based on field type
        pass

    def _render_display(self):
        """Render the current state."""
        # TODO: Implement Rich panel display
        pass

    def _get_key(self) -> str:
        """Get keyboard input."""
        return KeyboardInput.get_key()


# SECTION 7: Configuration Navigation and Manipulation
# ============================================================================
class ConfigNavigator:
    """Navigate and manipulate nested configuration dictionaries."""

    @staticmethod
    def get_value(config_dict: dict, path: str) -> Any:
        """Get value at dotted path like 'workflow.grouping.enabled'."""
        if not path:
            return config_dict

        keys = path.split(".")
        current = config_dict

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    @staticmethod
    def set_value(config_dict: dict, path: str, value: Any) -> None:
        """Set value at dotted path."""
        if not path:
            return

        keys = path.split(".")
        current = config_dict

        # Navigate to parent of target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    @staticmethod
    def get_section_fields(
        config_dict: dict, prefix: str, field_metadata: Dict[str, dict]
    ) -> List[Tuple[str, Any, dict]]:
        """Get all fields in a section with their metadata."""
        fields = []

        for field_path, metadata in field_metadata.items():
            if field_path.startswith(prefix):
                # For exact prefix matches or direct children
                remaining = field_path[len(prefix) :].lstrip(".")
                if not remaining or ("." not in remaining):
                    current_value = ConfigNavigator.get_value(config_dict, field_path)
                    fields.append((field_path, current_value, metadata))

        return sorted(fields)

    @staticmethod
    def has_changes(original: dict, modified: dict) -> bool:
        """Check if configuration has been modified."""
        return original != modified

    @staticmethod
    def get_changes(
        original: dict, modified: dict, prefix: str = ""
    ) -> Dict[str, Tuple[Any, Any]]:
        """Get dictionary of changes between original and modified configs."""
        changes = {}

        def compare_dict(orig, mod, path=""):
            if isinstance(orig, dict) and isinstance(mod, dict):
                # Check all keys from both dicts
                all_keys = set(orig.keys()) | set(mod.keys())
                for key in all_keys:
                    new_path = f"{path}.{key}" if path else key
                    orig_val = orig.get(key)
                    mod_val = mod.get(key)

                    if orig_val != mod_val:
                        if isinstance(orig_val, dict) and isinstance(mod_val, dict):
                            compare_dict(orig_val, mod_val, new_path)
                        else:
                            changes[new_path] = (orig_val, mod_val)
            elif orig != mod:
                changes[path] = (orig, mod)

        compare_dict(original, modified, prefix)
        return changes


# ============================================================================
# SECTION 7: Menu Display Components
# ============================================================================
class MenuDisplay:
    """Rich UI components for menu display."""

    @staticmethod
    def create_header(config_path: Path, modified: bool) -> Panel:
        """Create header panel with config info."""
        status = "[red]Modified[/red]" if modified else "[green]Saved[/green]"
        content = f"Config: [cyan]{config_path}[/cyan] | Status: {status}"
        return Panel(content, title="Configuration Editor", border_style="blue")

    @staticmethod
    def create_main_menu(menu_structure: Dict[str, Dict[str, Any]]) -> Table:
        """Create the main menu table."""
        menu = Table(title="Select Configuration Section", show_header=False, box=None)
        menu.add_column("Option", style="cyan", width=8)
        menu.add_column("Description", min_width=30)

        for key, section in menu_structure.items():
            menu.add_row(f"  {key}", section["title"])

        menu.add_row("", "")
        menu.add_row("  S", "[green]Save and Exit[/green]")
        menu.add_row("  Q", "[red]Quit without Saving[/red]")

        return menu

    @staticmethod
    def create_section_menu(
        section_key: str, menu_structure: Dict[str, Dict[str, Any]]
    ) -> Table:
        """Create menu for a configuration section."""
        section = menu_structure.get(section_key, {})
        menu = Table(
            title=f"{section.get('title', 'Section')} - Choose Subsection",
            show_header=False,
            box=None,
        )
        menu.add_column("Option", style="cyan", width=8)
        menu.add_column("Description", min_width=30)

        subsections = section.get("subsections", {})
        for key, subsection in subsections.items():
            menu.add_row(f"  {key}", subsection["title"])

        menu.add_row("", "")
        menu.add_row("  0", "[yellow]← Back to Main Menu[/yellow]")

        return menu

    @staticmethod
    def create_field_menu(title: str, fields: List[Tuple[str, Any, dict]]) -> Table:
        """Create menu for editing fields in a subsection."""
        menu = Table(title=f"{title} - Edit Values", show_header=True, box=None)
        menu.add_column("Option", style="cyan", width=8)
        menu.add_column("Setting", style="white", min_width=20)
        menu.add_column("Current Value", style="green", min_width=15)
        menu.add_column("Description", style="dim", min_width=25)

        for i, (field_path, current_value, metadata) in enumerate(fields, 1):
            field_name = field_path.split(".")[-1]
            value_display = (
                str(current_value)
                if current_value is not None
                else "[dim]not set[/dim]"
            )
            if isinstance(current_value, bool):
                value_display = "enabled" if current_value else "disabled"
            elif isinstance(current_value, list):
                value_display = (
                    f"[{len(current_value)} items]" if current_value else "[empty]"
                )

            help_text = (
                metadata.get("help", "")[:40] + "…"
                if len(metadata.get("help", "")) > 40
                else metadata.get("help", "")
            )

            menu.add_row(f"  {i}", field_name, value_display, help_text)

        menu.add_row("", "", "", "")
        menu.add_row("  0", "[yellow]← Back[/yellow]", "", "")

        return menu

    @staticmethod
    def create_breadcrumb(path: List[str]) -> Text:
        """Create navigation breadcrumb."""
        if not path:
            return Text("Main Menu", style="bold cyan")

        breadcrumb = Text()
        breadcrumb.append("Main Menu", style="cyan")

        for item in path:
            breadcrumb.append(" > ", style="dim")
            breadcrumb.append(item, style="cyan")

        return breadcrumb

    @staticmethod
    def show_changes_summary(changes: Dict[str, Tuple[Any, Any]]) -> Panel:
        """Show summary of changes before saving."""
        if not changes:
            return Panel("No changes to save", style="dim")

        content = []
        for field_path, (old_val, new_val) in changes.items():
            old_str = str(old_val) if old_val is not None else "None"
            new_str = str(new_val) if new_val is not None else "None"
            content.append(
                f"[cyan]{field_path}[/cyan]: [red]{old_str}[/red] → [green]{new_str}[/green]"
            )

        return Panel(
            "\n".join(content), title="Changes to be Saved", border_style="yellow"
        )


# ============================================================================
# SECTION 8: File Operations
# ============================================================================
class ConfigFileManager:
    """Handle config file reading, writing, and backups."""

    @staticmethod
    def find_config_file() -> Path:
        """Find existing config file or return default location."""
        candidates = [
            Path("config.toml"),
            Path("interaction_finder.toml"),
            Path(".interaction_finder.toml"),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # Return first candidate as default for new config
        return candidates[0]

    @staticmethod
    def load_config(path: Path) -> Tuple[dict, IfetcherConfig]:
        """Load config from TOML file."""
        if path.exists():
            with open(path, "rb") as f:
                config_dict = tomli.load(f)

            # Validate config
            config_obj = IfetcherConfig.model_validate(config_dict)
        else:
            # Create default config
            config_obj = IfetcherConfig()
            config_dict = config_obj.model_dump(exclude_unset=False)

        return config_dict, config_obj

    @staticmethod
    def save_config(config_dict: dict, path: Path, backup: bool = True) -> None:
        """Save config to TOML with optional backup."""
        if backup and path.exists():
            ConfigFileManager.create_backup(path)

        # Generate TOML with comments
        toml_content = ConfigFileManager.generate_commented_toml(config_dict)

        with open(path, "w", encoding="utf-8") as f:
            f.write(toml_content)

    @staticmethod
    def create_backup(path: Path) -> Path:
        """Create timestamped backup of config file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_suffix(f".{timestamp}.backup")

        if path.exists():
            backup_path.write_bytes(path.read_bytes())

        return backup_path

    @staticmethod
    def generate_commented_toml(config_dict: dict) -> str:
        """Generate TOML with helpful comments."""

        def clean_dict_for_toml(obj):
            """Recursively clean dictionary for TOML serialization."""
            if isinstance(obj, dict):
                cleaned = {}
                for k, v in obj.items():
                    if v is not None:  # Skip None values
                        cleaned[k] = clean_dict_for_toml(v)
                return cleaned
            elif isinstance(obj, list):
                return [clean_dict_for_toml(item) for item in obj if item is not None]
            else:
                return obj

        # Clean the config dict
        clean_config = clean_dict_for_toml(config_dict)

        # Add header comment
        header = [
            "# Interaction Finder Configuration",
            f"# Generated: {datetime.now().isoformat()}",
            "",
        ]

        try:
            # Use tomli_w to generate basic TOML
            basic_toml = tomli_w.dumps(clean_config)

            # Combine header and content
            lines = header + basic_toml.split("\n")
            return "\n".join(lines)

        except Exception as e:
            # Fallback to minimal TOML
            return f"# Error generating TOML: {e}\n# Basic config:\n{str(clean_config)}"


# ============================================================================
# SECTION 9: Main Configuration Editor Class
# ============================================================================
class ConfigEditor:
    """Main interactive configuration editor."""

    def __init__(self, config_path: Optional[Path] = None):
        self.console = Console()
        self.config_path = config_path or ConfigFileManager.find_config_file()
        self.original_config, self.config_obj = ConfigFileManager.load_config(
            self.config_path
        )
        self.working_config = copy.deepcopy(self.original_config)
        self.modified = False
        self.navigation_stack = []  # Track menu navigation

        # Generate dynamic metadata from model inspection
        self.field_metadata = ModelInspector.inspect_model(IfetcherConfig)
        self.menu_structure = ModelInspector.build_menu_structure(IfetcherConfig)

    def run(self) -> None:
        """Main menu loop."""
        try:
            while True:
                self.console.clear()
                self._show_header()

                if len(self.navigation_stack) == 0:
                    # Main menu
                    choice = self._show_main_menu()
                elif len(self.navigation_stack) == 1:
                    # Section menu
                    choice = self._show_section_menu()
                else:
                    # Field editing menu
                    choice = self._show_field_menu()

                if choice == "save":
                    if self._save_and_exit():
                        break
                elif choice == "quit":
                    if self._confirm_quit():
                        break
                elif choice == "back":
                    if self.navigation_stack:
                        self.navigation_stack.pop()
                elif (
                    choice and isinstance(choice, str) and choice in self.menu_structure
                ):
                    # Navigation to a main section
                    self.navigation_stack.append(choice)
                elif (
                    choice
                    and isinstance(choice, str)
                    and len(self.navigation_stack) == 1
                    and choice
                    in self.menu_structure[self.navigation_stack[0]].get(
                        "subsections", {}
                    )
                ):
                    # Navigation to a subsection
                    self.navigation_stack.append(choice)

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Editor cancelled by user[/yellow]")

    def _show_header(self) -> None:
        """Display header with config info."""
        header = MenuDisplay.create_header(self.config_path, self.modified)
        self.console.print(header)
        self.console.print()

        # Show breadcrumb
        breadcrumb = MenuDisplay.create_breadcrumb(self.navigation_stack)
        self.console.print(breadcrumb)
        self.console.print()

    def _show_main_menu(self) -> str:
        """Display main menu and get user choice."""
        # Show header first
        header = MenuDisplay.create_header(self.config_path, self.modified)
        self.console.print(header)
        self.console.print()

        # Show breadcrumb
        breadcrumb = MenuDisplay.create_breadcrumb(self.navigation_stack)
        self.console.print(breadcrumb)
        self.console.print()

        menu = InteractiveMenu(self.console, "Select Configuration Section")

        # Add main menu items
        for key, section in self.menu_structure.items():
            menu.add_item(
                section["title"], key, "Configure " + section["title"].lower()
            )

        menu.add_separator()
        menu.add_item("Save and Exit", "save", "Save changes and exit", "green")
        menu.add_item(
            "Quit without Saving", "quit", "Exit without saving changes", "red"
        )

        result = menu.navigate()
        return result or "quit"

    def _show_section_menu(self) -> str:
        """Display section menu and get user choice."""
        # Show header first
        header = MenuDisplay.create_header(self.config_path, self.modified)
        self.console.print(header)
        self.console.print()

        # Show breadcrumb
        breadcrumb = MenuDisplay.create_breadcrumb(self.navigation_stack)
        self.console.print(breadcrumb)
        self.console.print()

        section_key = self.navigation_stack[0]
        section = self.menu_structure[section_key]

        menu = InteractiveMenu(self.console, f"{section['title']} - Choose Subsection")

        # Add subsection items
        for key, subsection in section.get("subsections", {}).items():
            menu.add_item(
                subsection["title"], key, f"Edit {subsection['title'].lower()}"
            )

        menu.add_separator()
        menu.add_item("← Back to Main Menu", "back", "Return to main menu", "yellow")

        result = menu.navigate()
        return result or "back"

    def _show_field_menu(self) -> str:
        """Display field editing menu and get user choice."""
        if len(self.navigation_stack) < 2:
            return "back"

        section_key = self.navigation_stack[0]
        subsection_key = self.navigation_stack[1]

        section = self.menu_structure[section_key]
        subsection = section.get("subsections", {}).get(subsection_key, {})
        prefix = subsection.get("prefix", "")

        fields = ConfigNavigator.get_section_fields(
            self.working_config, prefix, self.field_metadata
        )

        if not fields:
            self.console.print("[red]No editable fields found in this section[/red]")
            Prompt.ask("Press Enter to continue")
            return "back"

        # Create interactive menu for fields
        menu = InteractiveMenu(self.console, title=subsection.get("title", "Settings"))

        # Add field items
        for field_path, current_value, metadata in fields:
            field_name = field_path.split(".")[-1]
            field_type = metadata.get("type", "string")

            # Format current value for display
            if current_value is None:
                value_display = "[dim]None[/dim]"
            elif isinstance(current_value, bool):
                value_display = (
                    f"[{'green' if current_value else 'red'}]{current_value}[/]"
                )
            elif isinstance(current_value, (int, float)):
                value_display = f"[cyan]{current_value}[/cyan]"
            elif isinstance(current_value, str):
                if len(str(current_value)) > 30:
                    value_display = (
                        f"[bright_green]{str(current_value)[:27]}…[/bright_green]"
                    )
                else:
                    value_display = f"[bright_green]{current_value}[/bright_green]"
            else:
                value_display = f"[magenta]{current_value}[/magenta]"

            description = f"{field_name} ({field_type}) = {value_display}"
            help_text = metadata.get("help", "")
            if help_text:
                description += (
                    f" - {help_text[:60]}{'…' if len(help_text) > 60 else ''}"
                )

            menu.add_item(description, (field_path, current_value, metadata))

        menu.add_separator()
        menu.add_item("← Back", "back", "Return to previous menu", "yellow")

        result = menu.navigate()

        if result == "back" or result is None:
            return "back"
        elif isinstance(result, tuple):
            # Edit the selected field
            self._edit_field(result)
            return ""  # Stay in current menu

        return ""

    # _handle_numeric_choice method removed - no longer needed with arrow key navigation

    def _edit_field(self, field_info: Tuple[str, Any, dict]) -> None:
        """Edit a single configuration field."""
        field_path, current_value, metadata = field_info

        self.console.print(f"\n[bold]Editing: {field_path}[/bold]")
        display_value = (
            "[dim]None[/dim]"
            if current_value is None
            else f"[green]{current_value}[/green]"
        )
        self.console.print(f"Current value: {display_value}")

        field_type = metadata.get("type", "string")
        help_text = metadata.get("help", "")

        new_value = None

        try:
            if field_type == "int":
                new_value = ValueEditor.edit_int(
                    current_value or 0,
                    field_path,
                    metadata.get("min"),
                    metadata.get("max"),
                    help_text,
                )
            elif field_type == "float":
                new_value = ValueEditor.edit_float(
                    current_value or 0.0,
                    field_path,
                    metadata.get("min"),
                    metadata.get("max"),
                    help_text,
                )
            elif field_type == "bool":
                new_value = ValueEditor.edit_bool(
                    current_value if current_value is not None else False,
                    field_path,
                    help_text,
                )
            elif field_type == "choice":
                new_value = ValueEditor.edit_choice(
                    current_value or metadata.get("choices", [""])[0],
                    field_path,
                    metadata.get("choices", []),
                    help_text,
                )
            elif field_type == "list":
                new_value = ValueEditor.edit_list(
                    current_value or [], field_path, help_text
                )
            elif field_type == "structured_dict":
                # Get value model from metadata
                value_model_name = metadata.get("value_model", "")
                value_model = self._get_value_model_class(value_model_name)

                if value_model:
                    editor = StructuredDictEditor(
                        current_value or {}, value_model, field_path, help_text
                    )
                    new_value = editor.edit_interactive()
                else:
                    # Fallback to dict editing if model not found
                    new_value = ValueEditor.edit_dict(
                        current_value or {}, field_path, help_text
                    )
            else:  # string
                new_value = ValueEditor.edit_string(
                    current_value or "", field_path, help_text
                )

            if new_value is not None:
                # Validate the change
                if self._validate_change(field_path, new_value):
                    ConfigNavigator.set_value(
                        self.working_config, field_path, new_value
                    )
                    self.modified = True
                    self.console.print(
                        f"[green]Updated {field_path.split('.')[-1]}[/green]"
                    )
                else:
                    self.console.print(
                        "[red]Change not saved due to validation error[/red]"
                    )

                Prompt.ask("\nPress Enter to continue")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Edit cancelled[/yellow]")

    def _validate_change(self, field_path: str, new_value: Any) -> bool:
        """Validate a configuration change using Pydantic."""
        try:
            # Create a copy of working config with the change
            temp_config = copy.deepcopy(self.working_config)
            ConfigNavigator.set_value(temp_config, field_path, new_value)

            # Validate with Pydantic
            IfetcherConfig.model_validate(temp_config)
            return True

        except ValidationError as e:
            self.console.print("[red]Validation Error:[/red]")
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                self.console.print(f"  {field}: {msg}")
            return False
        except Exception as e:
            self.console.print(f"[red]Unexpected error: {e}[/red]")
            return False

    def _save_and_exit(self) -> bool:
        """Save changes and exit."""
        if not self.modified:
            self.console.print("[dim]No changes to save[/dim]")
            return True

        # Show changes summary
        changes = ConfigNavigator.get_changes(self.original_config, self.working_config)
        if changes:
            summary = MenuDisplay.show_changes_summary(changes)
            self.console.print(summary)
            self.console.print()

        if Confirm.ask("Save these changes?", default=True):
            try:
                ConfigFileManager.save_config(self.working_config, self.config_path)
                self.console.print(
                    f"[green]Configuration saved to {self.config_path}[/green]"
                )
                return True
            except Exception as e:
                self.console.print(f"[red]Error saving configuration: {e}[/red]")
                return False

        return False

    def _confirm_quit(self) -> bool:
        """Confirm quitting without saving."""
        if not self.modified:
            return True

        self.console.print("[yellow]You have unsaved changes.[/yellow]")
        return Confirm.ask("Quit without saving?", default=False)


# ============================================================================
# SECTION 10: Reusable Text Input Widget
# ============================================================================
class TextInputWidget:
    """Reusable text input widget with cursor navigation and visual feedback."""

    def __init__(
        self, initial_text: str = "", title: str = "Input", placeholder: str = ""
    ):
        self.text = initial_text
        self.cursor_position = len(initial_text)
        self.title = title
        self.placeholder = placeholder

    def get_cursor_display_text(self) -> str:
        """Get text with inverted cursor at current position and spaces shown as dim cyan dots."""

        def format_text(text: str) -> str:
            """Replace spaces with dim cyan dots."""
            return text.replace(" ", "[dim cyan]⋅[/dim cyan]")

        if not self.text:
            # Empty input - show inverted space as cursor
            return "[reverse] [/reverse]"
        else:
            # Split text at cursor position
            before = self.text[: self.cursor_position]
            after = self.text[self.cursor_position :]

            if self.cursor_position >= len(self.text):
                # Cursor at end - add inverted space
                return f"{format_text(before)}[reverse] [/reverse]"
            else:
                # Cursor in middle - invert character at cursor position
                char_at_cursor = self.text[self.cursor_position]
                if char_at_cursor == " ":
                    # Special handling for space at cursor - show normal inverted space
                    return f"{format_text(before)}[reverse] [/reverse]{format_text(after[1:])}"
                else:
                    return f"{format_text(before)}[reverse]{char_at_cursor}[/reverse]{format_text(after[1:])}"

    def render_panel(
        self,
        help_text: str = "",
        border_style: str = "bright_cyan",
        extra_display_text: str = "",
    ) -> Panel:
        """Render the input field with cursor and borders."""
        from rich.text import Text

        display_text = self.get_cursor_display_text()

        # Use Rich's text measurement for accurate width calculation
        rich_text = Text.from_markup(display_text)
        visual_width = rich_text.cell_len
        extra_width = (
            Text.from_markup(extra_display_text).cell_len if extra_display_text else 0
        )

        # Dynamic width calculation: minimum 48, expand if content is longer
        min_width = 48
        content_width = visual_width + extra_width
        field_width = max(min_width, content_width + 4)  # +4 for padding and margins

        # Dashed grey box drawing characters for input field
        top_border = f"[dim]┌{'╌' * field_width}┐[/dim]"
        bottom_border = f"[dim]└{'╌' * field_width}┘[/dim]"

        # Calculate padding to make total line width match borders
        # Format: "╎ " + display_text + padding + extra_display_text + " ╎"
        padding_needed = max(0, field_width + 2 - 4 - visual_width - extra_width)

        if extra_display_text:
            input_line = f"[dim]╎[/dim] {display_text}{' ' * padding_needed}{extra_display_text} [dim]╎[/dim]"
        else:
            input_line = (
                f"[dim]╎[/dim] {display_text}{' ' * padding_needed} [dim]╎[/dim]"
            )

        # Build content with optional help text
        content_parts = [
            "[bold]New Value:[/bold]",
            "",
            top_border,
            input_line,
            bottom_border,
        ]
        if help_text:
            content_parts.extend(["", f"[dim]{help_text}[/dim]"])

        content = "\n".join(content_parts)

        return Panel(
            content,
            title=f"[bold cyan]✎ {self.title.upper()}[/bold cyan]",
            border_style=border_style,
        )

    def _find_word_boundary(self, text: str, pos: int, direction: int) -> int:
        """Find next word boundary. Direction: -1 for left, 1 for right."""
        original_pos = pos
        if direction < 0:  # Moving left
            # Always move at least one position first (unless at start)
            if pos > 0:
                pos -= 1
            # Skip current word characters
            while pos > 0 and text[pos - 1].isalnum():
                pos -= 1
            # Skip whitespace
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
        else:  # Moving right
            # Always move at least one position first (unless at end)
            if pos < len(text):
                pos += 1
            # Skip current word characters
            while pos < len(text) and text[pos].isalnum():
                pos += 1
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
        return max(0, min(len(text), pos))

    def handle_input(self, key: str) -> tuple[bool, str]:
        """
        Handle keyboard input for text editing.

        Returns:
            (continue_editing, final_text):
            - continue_editing: True if editing should continue, False if done/cancelled
            - final_text: The current text value (useful when editing completes)
        """
        if key == "ENTER":
            # Complete editing - return final text
            return False, self.text
        elif key in ("ESC", "CTRL+D"):
            # Cancel editing - return empty to indicate cancellation
            return False, ""
        elif key == "LEFT":
            # Move cursor left
            if self.cursor_position > 0:
                self.cursor_position -= 1
        elif key == "RIGHT":
            # Move cursor right
            if self.cursor_position < len(self.text):
                self.cursor_position += 1
        elif key == "HOME":
            # Move to start of line
            self.cursor_position = 0
        elif key == "END":
            # Move to end of line
            self.cursor_position = len(self.text)
        elif key == "CTRL+LEFT":
            # Move to previous word boundary
            self.cursor_position = self._find_word_boundary(
                self.text, self.cursor_position, -1
            )
        elif key == "CTRL+RIGHT":
            # Move to next word boundary
            self.cursor_position = self._find_word_boundary(
                self.text, self.cursor_position, 1
            )
        elif key == "BACKSPACE":
            # Remove character before cursor, or signal empty for cancellation
            if self.cursor_position > 0:
                # Remove character at cursor-1 position
                before = self.text[: self.cursor_position - 1]
                after = self.text[self.cursor_position :]
                self.text = before + after
                self.cursor_position -= 1
            elif not self.text:
                # Input is empty, signal cancellation
                return False, ""
        elif key == "DELETE":
            # Remove character at cursor position
            if self.cursor_position < len(self.text):
                before = self.text[: self.cursor_position]
                after = self.text[self.cursor_position + 1 :]
                self.text = before + after
        elif key == "SPACE":
            # Insert space character
            before = self.text[: self.cursor_position]
            after = self.text[self.cursor_position :]
            self.text = before + " " + after
            self.cursor_position += 1
        elif len(key) == 1 and key.isprintable():
            # Insert character at cursor position
            before = self.text[: self.cursor_position]
            after = self.text[self.cursor_position :]
            self.text = before + key + after
            self.cursor_position += 1

        # Continue editing
        return True, self.text


# ============================================================================
# SECTION 11: Tree-Based Configuration Editor
# ============================================================================
class TreeNode:
    """Represents a node in the configuration tree."""

    def __init__(
        self,
        name: str,
        path: str = "",
        node_type: str = "section",
        value: Any = None,
        metadata: dict = None,
        is_explicit: bool = False,
        is_dict_entry: bool = False,
        is_action_node: bool = False,
        dict_key: str = None,
    ):
        self.name = name
        self.path = path  # dotted path like "workflow.grouping.enabled"
        self.node_type = node_type  # "section", "field", "root"
        self.value = value
        self.metadata = metadata or {}
        self.is_explicit = (
            is_explicit  # True if explicitly set in TOML, False if default
        )
        self.children = []
        self.parent = None
        self.expanded = True
        self.modified = False  # Changed during current editing session
        self.differs_from_default = (
            False  # Value differs from default (for view filtering)
        )
        self.visible = True  # For tree view state filtering
        self.is_dict_entry = is_dict_entry  # True if this node represents a dict key
        self.is_action_node = is_action_node  # True for action nodes like "[+ Add]"
        self.dict_key = dict_key  # Store original dict key for dict entries

    def add_child(self, child: "TreeNode") -> "TreeNode":
        """Add a child node and return it."""
        child.parent = self
        self.children.append(child)
        return child

    def is_expandable(self) -> bool:
        """Check if this node can be expanded to show children."""
        # Action nodes are not expandable
        if self.is_action_node:
            return False

        # Structured dict nodes are expandable if they have entries
        if self.node_type == "field" and self.metadata.get("type") == "structured_dict":
            return isinstance(self.value, dict) and len(self.value) > 0

        # Dict entry nodes are expandable if they represent BaseModel instances
        if self.is_dict_entry and hasattr(self.value, "model_fields"):
            return True

        # Regular section nodes are expandable if they have children
        return len(self.children) > 0

    def get_display_value(self) -> str:
        """Get formatted display value for this node."""
        # Action nodes show specialized text
        if self.is_action_node:
            return "[dim][bright_green]+[/bright_green] Click to add[/dim]"

        # Dict entry nodes should not show values - they're containers for their fields
        if self.is_dict_entry:
            return ""

        if self.node_type != "field":
            return ""

        field_type = self.metadata.get("type", "string")
        field_name = self.path.split(".")[-1] if self.path else self.name

        # Show unset and default values in dim grey
        if not self.is_explicit:
            if self.value is None:
                return "[dim]None[/dim]"
            # Use formatter but make it dim for default values
            formatted = ConfigValueFormatter.format_value(
                self.value, field_type, field_name, self.metadata
            )
            # Remove existing color markup and make it dim
            import re

            clean_value = re.sub(r"\[/?[^\]]*\]", "", formatted)
            return f"[dim]{clean_value}[/dim]"

        # Explicitly configured values use full formatting
        return ConfigValueFormatter.format_value(
            self.value, field_type, field_name, self.metadata
        )

    def get_all_visible_nodes(self) -> List["TreeNode"]:
        """Get all visible nodes in traversal order."""
        nodes = []
        if self.visible:
            nodes.append(self)
            if self.expanded:
                for child in self.children:
                    nodes.extend(child.get_all_visible_nodes())
        return nodes

    def has_explicit_values(self) -> bool:
        """Check if this node or any of its descendants have explicit values."""
        # If this is a field node, check if it's explicit
        if self.node_type == "field":
            return self.is_explicit

        # For section nodes, check all children recursively
        for child in self.children:
            if child.has_explicit_values():
                return True
        return False


class TreeConfigEditor:
    """Configuration editor using Rich Tree + Layout + Live display."""

    def __init__(self, config_path: Optional[Path] = None):
        self.console = Console()
        self.config_path = config_path or ConfigFileManager.find_config_file()
        self.working_config, self.config_obj = ConfigFileManager.load_config(
            self.config_path
        )
        self.original_config = copy.deepcopy(self.working_config)

        # Load raw TOML data to distinguish defaults from explicit values
        self.explicit_config = self._load_explicit_config()

        # Generate dynamic metadata from model inspection
        self.field_metadata = ModelInspector.inspect_model(IfetcherConfig)
        self.menu_structure = ModelInspector.build_menu_structure(IfetcherConfig)

        # Build tree structure
        self.root_node = self._build_tree_structure()
        self.selected_node = (
            self.root_node.children[0] if self.root_node.children else self.root_node
        )
        self.modified = False

        # Scrolling state for keeping selected item visible
        self.scroll_offset = 0
        self.visible_height = 20

        # Editing state
        self.editing_mode = False
        self.edit_input_value = ""
        self.edit_field_info = None
        self.edit_field_type = "string"  # Track the field type for specialized input
        self.edit_choices = []  # For choice fields
        self.edit_list_items = []  # For list fields
        self.edit_selected_choice = 0  # For choice field navigation
        # List widget specific state
        self.edit_list_selected = 0  # Currently selected list item
        self.edit_list_mode = "browse"  # "browse", "add", "edit"
        self.edit_list_input = ""  # Input for add/edit mode

        # Cursor position tracking for text editing
        self.edit_cursor_position = 0  # Position of cursor within edit_input_value

        # Tree view state cycling (Shift+Tab)
        self.tree_view_states = ["modified_only", "all_expanded", "all_collapsed"]
        self.current_tree_state = 0  # Index into tree_view_states
        self.input_field_row = 0  # Terminal row of input field (calculated at render)
        self.input_field_col = 0  # Terminal column where input starts
        self.cursor_visible = False  # Track cursor visibility state

        # Search/filter state
        self.search_mode = False  # Whether we're in search mode
        self.search_term = ""  # Current search/filter term
        self.filtered_nodes = []  # Cached list of nodes matching search
        self.original_selected_node = None  # Store selection before filtering

        # Create layout
        self.layout = self._create_layout()

        # Initialize default tree view state
        self._apply_modified_only_view()

    def _load_explicit_config(self) -> dict:
        """Load only the explicitly configured values from TOML file."""
        if self.config_path.exists():
            with open(self.config_path, "rb") as f:
                return tomli.load(f)
        return {}

    def _is_explicit_value(self, field_path: str) -> bool:
        """Check if a value is explicitly set in the TOML file."""
        return ConfigNavigator.get_value(self.explicit_config, field_path) is not None

    def _create_layout(self) -> Layout:
        """Create the main layout with tree and editor panels."""
        layout = Layout()

        # Split into three main areas
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )

        # Split main area into tree and editor
        layout["main"].split_row(
            Layout(name="tree", ratio=1), Layout(name="editor", ratio=2)
        )

        # Editor can be split vertically when in edit mode
        # Initially just has the info panel
        layout["editor"].split_column(Layout(name="field_info", ratio=1))

        return layout

    def _update_editor_layout(self):
        """Update the editor layout based on current editing state."""
        if self.editing_mode:
            # Split editor into field info and input area
            if len(self.layout["editor"].children) == 1:
                # Add input area if not already split
                self.layout["editor"].split_column(
                    Layout(name="field_info", ratio=2),
                    Layout(name="input_area", ratio=1),
                )

            # Update both panels
            self.layout["field_info"].update(self._get_field_info_panel())
            self.layout["input_area"].update(self._get_input_panel())
        else:
            # Single panel mode - recreate editor with just field info
            self.layout["editor"].split_column(Layout(name="field_info", ratio=1))
            self.layout["field_info"].update(self._get_editor_content())

    def _get_field_info_panel(self) -> Panel:
        """Get field information panel when in editing mode."""
        if not self.edit_field_info:
            return Panel("No field selected", title="Field Info", border_style="yellow")

        field_path, current_value, metadata = self.edit_field_info
        field_type = metadata.get("type", "string")
        help_text = metadata.get("help", "No description available")

        if field_type == "list":
            # For list fields, show the list items in the field info panel
            return self._get_list_field_info_panel(field_path, current_value, help_text)

        # Get default value for comparison
        default_config = IfetcherConfig()
        default_value = ConfigNavigator.get_value(default_config.__dict__, field_path)

        display_value = "[dim]None[/dim]" if current_value is None else current_value
        display_default = "[dim]None[/dim]" if default_value is None else default_value

        content = [
            f"[bold]Field:[/bold] {field_path}",
            f"[bold]Type:[/bold] {field_type}",
            f"[bold]Current Value:[/bold] {display_value}",
            f"[bold]Default Value:[/bold] {display_default}",
            "",
            f"[dim]{help_text}[/dim]",
            "",
        ]

        # Show different instructions based on context
        has_changes = (
            current_value != self.edit_original_value
            if hasattr(self, "edit_original_value")
            else False
        )
        differs_from_default = current_value != default_value

        instructions = "[yellow]Type new value and press Enter to save"
        if differs_from_default:
            instructions += " • [bold]R[/bold]eset to default"
        if has_changes:
            instructions += " • [bold]U[/bold]ndo"
        instructions += " • Esc to cancel[/yellow]"

        content.append(instructions)

        return Panel("\n".join(content), title="Field Info", border_style="green")

    def _get_list_field_info_panel(
        self, field_path: str, current_value, help_text: str
    ) -> Panel:
        """Get list field info panel showing the list items."""
        content_lines = [
            f"[bold]Field:[/bold] {field_path}",
            "[bold]Type:[/bold] list",
            "",
            f"[dim]{help_text}[/dim]",
            "",
            "[bold]List Items:[/bold]",
            "",
        ]

        if not self.edit_list_items:
            content_lines.append("[dim]No items - Press [+] to add one[/dim]")
        else:
            for i, item in enumerate(self.edit_list_items):
                if i == self.edit_list_selected:
                    # Selected item - highlighted
                    content_lines.append(f"[bright_cyan]• {item}[/bright_cyan]")
                else:
                    # Unselected item - normal brightness
                    content_lines.append(f"• {item}")

        return Panel(
            "\n".join(content_lines), title="List Editor", border_style="green"
        )

    def _get_list_field_view_panel(self) -> Panel:
        """Get list field view panel for non-editing mode."""
        field_path = self.selected_node.path
        help_text = self.selected_node.metadata.get("help", "No description available")
        current_value = self.selected_node.value

        content_lines = [
            f"[bold]Field:[/bold] {field_path}",
            "[bold]Type:[/bold] list",
            "",
            f"[dim]{help_text}[/dim]",
            "",
            "[bold]List Items:[/bold]",
            "",
        ]

        if self.selected_node.modified:
            content_lines.insert(0, "[yellow]* Modified (unsaved)[/yellow]")
            content_lines.insert(1, "")

        if not current_value:
            content_lines.append("[dim]No items[/dim]")
        else:
            for item in current_value:
                content_lines.append(f"• {item}")

        content_lines.extend(["", "[dim]Press Enter to edit this list[/dim]"])

        return Panel("\n".join(content_lines), title="Field Info", border_style="green")

    def _get_input_panel(self) -> Panel:
        """Get input panel for editing - routes to appropriate widget."""
        if self.edit_field_type == "bool":
            return self._get_boolean_input_panel()
        elif self.edit_field_type == "choice":
            return self._get_choice_input_panel()
        elif self.edit_field_type == "list":
            return self._get_list_item_input_panel()
        elif self.edit_field_type in ("int", "float"):
            return self._get_numeric_input_panel()
        else:
            # Default string input using TextInputWidget
            widget = TextInputWidget(
                initial_text=self.edit_input_value or "",
                title="Text Input",
                placeholder="Enter value…",
            )
            widget.cursor_position = self.edit_cursor_position

            return widget.render_panel(
                help_text="Use Home/End, ⌃←→ for navigation • ESC/⌃D to cancel",
                border_style="bright_cyan",
            )

    def _get_boolean_input_panel(self) -> Panel:
        """Get boolean toggle input panel."""
        current_value = self.edit_input_value

        # Show toggle switches
        false_indicator = (
            "[dim]○[/dim]" if current_value else "[bright_cyan]●[/bright_cyan]"
        )
        true_indicator = (
            "[bright_cyan]●[/bright_cyan]" if current_value else "[dim]○[/dim]"
        )

        false_label = (
            "[bright_white]False[/bright_white]"
            if not current_value
            else "[dim]False[/dim]"
        )
        true_label = (
            "[bright_white]True[/bright_white]" if current_value else "[dim]True[/dim]"
        )

        content = f"[bold]Boolean Value:[/bold]\n\n{false_indicator} {false_label}    {true_indicator} {true_label}\n\n[dim]Use ←→ arrows or Space to toggle[/dim]"

        return Panel(
            content,
            title="[bold cyan]✎ BOOLEAN[/bold cyan]",
            border_style="bright_cyan",
        )

    def _get_choice_input_panel(self) -> Panel:
        """Get choice selection input panel."""
        if not self.edit_choices:
            content = "[red]No choices available[/red]"
            return Panel(
                content,
                title="[bold cyan]✎ CHOICE[/bold cyan]",
                border_style="bright_cyan",
            )

        content_lines = ["[bold]Select Option:[/bold]", ""]

        for i, choice in enumerate(self.edit_choices):
            if i == self.edit_selected_choice:
                # Selected choice
                content_lines.append(f"[bright_cyan]● {choice}[/bright_cyan]")
            else:
                # Unselected choice
                content_lines.append(f"[dim]○ {choice}[/dim]")

        content_lines.extend(
            ["", "[dim]Use ↑↓ arrows to navigate, Enter to select[/dim]"]
        )
        content = "\n".join(content_lines)

        return Panel(
            content, title="[bold cyan]✎ CHOICE[/bold cyan]", border_style="bright_cyan"
        )

    def _get_list_item_input_panel(self) -> Panel:
        """Get input panel for individual list item editing (add/edit mode only)."""
        if self.edit_list_mode in ("add", "edit"):
            # Show input for individual item using TextInputWidget
            if self.edit_list_mode == "add":
                title = "ADD NEW ITEM"
                placeholder = "Enter new item value…"
            else:  # edit
                title = f"EDIT ITEM {self.edit_list_selected + 1}"
                placeholder = "Modify item value…"

            widget = TextInputWidget(
                initial_text=self.edit_list_input or "",
                title=title,
                placeholder=placeholder,
            )
            widget.cursor_position = self.edit_cursor_position

            help_text = f"{placeholder}\n[Enter] Save | ESC/⌃D Cancel | Use Home/End, ⌃←→ for navigation"

            return widget.render_panel(help_text=help_text, border_style="bright_cyan")
        else:
            # Browse mode - hide input panel by showing a minimal message
            content = "[dim]Use ↑↓ to navigate items, + to add, - to remove, Enter to edit, or Enter to save the list[/dim]"
            return Panel(
                content,
                title="[bold green]List Commands[/bold green]",
                border_style="green",
            )

    def _get_numeric_input_panel(self) -> Panel:
        """Get numeric input panel with increment/decrement arrows."""
        # Create custom numeric display with dimmed trailing zeros
        display_text = self._get_numeric_display_with_cursor()

        # Manual rendering since we need custom display text
        from rich.text import Text

        # Use Rich's text measurement for accurate width calculation
        rich_text = Text.from_markup(display_text)
        visual_width = rich_text.cell_len
        extra_width = 3  # Width of "[dim]↑↓[/dim]"

        # Dynamic width calculation: minimum 48, expand if content is longer
        min_width = 48
        field_width = max(min_width, visual_width + extra_width + 4)

        # Dashed grey box drawing characters for input field
        top_border = f"[dim]┌{'╌' * field_width}┐[/dim]"
        bottom_border = f"[dim]└{'╌' * field_width}┘[/dim]"
        # Calculate padding: need to account for all components in the line
        # Format: "╎ " + display_text + padding + " ↑↓ ╎"
        # Components: left_border(1) + left_space(1) + content + padding + space_before_arrows(1) + arrows(2) + space_after_arrows(1) + right_border(1)
        # Total should equal field_width + 2 (for the border width)
        total_fixed_width = (
            1 + 1 + visual_width + 1 + extra_width + 1 + 1
        )  # = visual_width + extra_width + 6
        padding_needed = max(0, (field_width + 2) - total_fixed_width)
        input_line = f"[dim]╎[/dim] {display_text}{' ' * padding_needed} [dim]↑↓[/dim] [dim]╎[/dim]"

        help_text = (
            "↑↓ Increment/Decrement • Home/End, ⌃←→ navigation • ESC Cancel • ⌃D Exit"
        )

        content_parts = [
            "[bold]New Value:[/bold]",
            "",
            top_border,
            input_line,
            bottom_border,
            "",
            f"[dim]{help_text}[/dim]",
        ]

        return Panel(
            "\n".join(content_parts),
            title=f"[bold cyan]✎ NUMERIC ({self.edit_field_type.upper()})[/bold cyan]",
            border_style="bright_cyan",
        )

    def _get_tree_panel(self) -> Panel:
        """Get the tree panel with appropriate styling based on editing mode."""
        tree = self._build_rich_tree()

        if self.editing_mode:
            # Fade out the tree when editing - dim border and muted title
            # Apply dim style to the Panel itself, not the tree content
            return Panel(
                tree,
                title="[dim]Configuration Tree[/dim]",
                border_style="dim",
                style="dim",  # This dims the entire panel content
            )
        else:
            # Normal styling
            return Panel(
                tree,
                title="Configuration Tree",
                border_style="blue",
            )

    def _build_tree_structure(self) -> TreeNode:
        """Build the tree structure from dynamic model inspection."""
        root = TreeNode("Configuration", node_type="root")

        for section_key, section_info in self.menu_structure.items():
            section_node = root.add_child(
                TreeNode(
                    name=section_info["title"],
                    path=section_info["section"],
                    node_type="section",
                )
            )

            # Add subsections
            for subsection_key, subsection_info in section_info.get(
                "subsections", {}
            ).items():
                subsection_node = section_node.add_child(
                    TreeNode(
                        name=subsection_info["title"],
                        path=subsection_info["prefix"],
                        node_type="section",
                    )
                )

                # Add fields for this subsection
                prefix = subsection_info["prefix"]
                fields = ConfigNavigator.get_section_fields(
                    self.working_config, prefix, self.field_metadata
                )

                for field_path, current_value, metadata in fields:
                    field_name = field_path.split(".")[-1]
                    is_explicit = self._is_explicit_value(field_path)

                    # Check if this field differs from defaults
                    default_config = IfetcherConfig()
                    default_value = ConfigNavigator.get_value(
                        default_config.__dict__, field_path
                    )
                    differs_from_default = current_value != default_value

                    field_node = subsection_node.add_child(
                        TreeNode(
                            name=field_name,
                            path=field_path,
                            node_type="field",
                            value=current_value,
                            metadata=metadata,
                            is_explicit=is_explicit,
                        )
                    )

                    # Set flag for values that differ from defaults (for view filtering)
                    field_node.differs_from_default = differs_from_default

                    # Expand structured_dict fields to show their entries
                    if metadata.get("type") == "structured_dict" and isinstance(
                        current_value, dict
                    ):
                        self._expand_structured_dict(
                            field_node, current_value, metadata
                        )

                # Collapse subsection if it has no explicit values
                if not subsection_node.has_explicit_values():
                    subsection_node.expanded = False

            # Collapse main section if it has no explicit values
            if not section_node.has_explicit_values():
                section_node.expanded = False

        # Compact single-child branches to flatten unnecessary hierarchy
        self._compact_single_child_branches(root)

        return root

    def _expand_structured_dict(
        self, parent_node: TreeNode, dict_value: dict, metadata: dict
    ):
        """Expand a structured dict into tree nodes for each entry and their fields."""
        value_model_name = metadata.get("value_model", "")

        # Get the actual BaseModel instances from the config object instead of dumped dict
        if "." not in parent_node.path:
            # Top-level field like 'agents'
            actual_dict_value = getattr(self.config_obj, parent_node.path, dict_value)
        else:
            # Nested field like 'task.kinds'
            parts = parent_node.path.split(".")
            current = self.config_obj
            for part in parts:
                current = getattr(current, part, None)
                if current is None:
                    break
            actual_dict_value = current if current is not None else dict_value

        # Add each dict entry as a child node
        for key, value in actual_dict_value.items():
            # Create entry node for the dict key
            entry_node = parent_node.add_child(
                TreeNode(
                    name=key,
                    path=f"{parent_node.path}.{key}",
                    node_type="field",
                    value=value,
                    metadata={"type": "dict_entry", "value_model": value_model_name},
                    is_explicit=True,  # Dict entries are always explicit
                    is_dict_entry=True,
                    dict_key=key,
                )
            )

            # Dict entries are always non-default (they represent user configuration)
            entry_node.differs_from_default = True

            # If the value is a BaseModel, expand its fields as children
            is_basemodel = hasattr(value, "model_dump") and hasattr(
                type(value), "model_fields"
            )
            if is_basemodel:
                value_dict = value.model_dump()
                value_class = type(value)

                for field_name, field_value in value_dict.items():
                    # Get field metadata for this specific field
                    field_info = value_class.model_fields.get(field_name)
                    if field_info:
                        # Build metadata manually using ModelInspector methods
                        field_type = ModelInspector.get_field_type(
                            field_info, field_info.annotation
                        )
                        constraints = ModelInspector.get_field_constraints(field_info)

                        field_metadata = {
                            "type": field_type,
                            "help": field_info.description
                            or f"Configuration for {field_name}",
                            **constraints,
                        }

                        field_node = entry_node.add_child(
                            TreeNode(
                                name=field_name,
                                path=f"{entry_node.path}.{field_name}",
                                node_type="field",
                                value=field_value,
                                metadata=field_metadata,
                                is_explicit=True,
                            )
                        )

                        # Dict entry fields are considered non-default if they have non-empty values
                        field_node.differs_from_default = (
                            field_value is not None
                            and field_value != []
                            and field_value != {}
                            and field_value != ""
                        )

        # Add "[+ Add]" action node for adding new entries
        parent_node.add_child(
            TreeNode(
                name=f"[+ Add {value_model_name}]",
                path=f"{parent_node.path}._add",
                node_type="action",
                value=None,
                metadata={
                    "type": "action",
                    "action": "add_entry",
                    "value_model": value_model_name,
                },
                is_action_node=True,
            )
        )

    def _compact_single_child_branches(self, node: TreeNode):
        """Recursively compact single-child branches to flatten unnecessary hierarchy.

        When a section node has exactly one section child, absorb the parent name
        into the child and promote the child to replace the parent.
        """
        # Process children first (bottom-up)
        for child in list(
            node.children
        ):  # Use list() to avoid modification during iteration
            self._compact_single_child_branches(child)

        # Check if this node can be compacted
        if (
            node.node_type == "section"
            and len(node.children) == 1
            and node.children[0].node_type == "section"
        ):
            child = node.children[0]
            parent = node.parent

            if parent is not None:  # Don't compact root node
                # Create a new combined name that shows the absorbed hierarchy
                combined_name = f"{node.name}"  # Keep the parent name as primary

                # Update child properties
                child.name = combined_name
                # Keep child's path, metadata, etc. as they are more specific

                # Replace node with child in parent's children list
                parent_index = parent.children.index(node)
                parent.children[parent_index] = child
                child.parent = parent

                # Update the child's expanded state to match the parent
                child.expanded = node.expanded or child.expanded

    def _build_rich_tree(self) -> Tree:
        """Build a Rich Tree from our tree structure with scrolling."""
        # Update visible height based on layout:
        # total height - header(3) - footer(3) = main area
        # Tree gets 1/3 of main area (shared with editor at ratio 1:2)
        main_height = max(6, self.console.size.height - 6)  # header + footer = 6
        self.visible_height = max(5, main_height - 4)  # Account for panel borders

        # Get all visible nodes in order
        all_nodes = self._get_all_nodes()
        if not all_nodes:
            return Tree("Configuration")

        # Determine visible window
        start_idx = self.scroll_offset
        end_idx = min(len(all_nodes), self.scroll_offset + self.visible_height)
        visible_nodes = all_nodes[start_idx:end_idx]

        # Build a tree that maintains hierarchy but only shows visible nodes
        tree = Tree("Configuration")
        self._add_windowed_nodes_to_tree(
            tree, visible_nodes, all_nodes, start_idx, end_idx
        )

        return tree

    def _add_windowed_nodes_to_tree(
        self,
        tree: Tree,
        visible_nodes: List[TreeNode],
        all_nodes: List[TreeNode],
        start_idx: int,
        end_idx: int,
    ):
        """Add visible nodes to tree while maintaining proper hierarchy."""
        # Add scroll indicators
        if start_idx > 0:
            tree.add(f"[dim]↑ {start_idx} items above[/dim]")

        # Group visible nodes by their actual parent relationships
        tree_map = {}  # Maps TreeNode -> rich tree node
        tree_map[self.root_node] = tree  # Root level

        for node in visible_nodes:
            # Build display text
            display_text = node.name
            value_display = node.get_display_value()
            if value_display:
                display_text += f": {value_display}"

            # Add style based on selection and modification
            style = ""
            if node == self.selected_node:
                style = "bold cyan"
            elif node.modified:
                style = "yellow"
            elif node.node_type == "field":
                style = "green" if node.is_explicit else "dim"

            # Add expand/collapse indicator for sections
            if node.node_type == "section" and node.children:
                expand_icon = "▼" if node.expanded else "▶"
                display_text = f"{expand_icon} {display_text}"

            # Apply styling
            if style:
                display_text = f"[{style}]{display_text}[/{style}]"

            # Find the right parent using actual TreeNode parent relationship
            parent_node = node.parent if node.parent else self.root_node
            parent_rich_node = tree_map.get(parent_node, tree)

            # If parent is not in tree_map, it means the parent is not visible
            # In this case, add to root level with hierarchy indicator
            if parent_rich_node is None:
                parent_rich_node = tree
                # Add hierarchy indicator to show this is nested deeper
                display_text = f"… {display_text}"

            # Add this node to the tree
            rich_node = parent_rich_node.add(display_text)

            # Store this node for its children to find
            tree_map[node] = rich_node

        # Add scroll indicator for content below
        if end_idx < len(all_nodes):
            remaining = len(all_nodes) - end_idx
            tree.add(f"[dim]↓ {remaining} items below[/dim]")

    def _add_nodes_to_tree(self, rich_tree: Tree, node: TreeNode, rich_parent):
        """Recursively add nodes to Rich Tree."""
        for child in node.children:
            # Build the display text
            display_text = child.name
            value_display = child.get_display_value()

            if value_display:
                display_text += f": {value_display}"

            # Add style based on selection and modification
            style = ""
            if child == self.selected_node:
                style = "bold cyan"
            elif child.modified:
                style = "yellow"
            elif child.node_type == "field":
                style = "green" if child.value is not None else "dim"

            # Add expand/collapse indicator for sections
            if child.node_type == "section" and child.children:
                expand_icon = "▼" if child.expanded else "▶"
                display_text = f"{expand_icon} {display_text}"

            # Create the rich tree node
            if style:
                display_text = f"[{style}]{display_text}[/{style}]"

            if child.node_type == "section" and child.expanded and child.children:
                rich_node = rich_parent.add(display_text)
                self._add_nodes_to_tree(rich_tree, child, rich_node)
            elif child.node_type == "section":
                rich_parent.add(display_text)
            else:
                rich_parent.add(display_text)

    def _get_all_nodes(self) -> List[TreeNode]:
        """Get all visible nodes in the tree."""
        # If in search mode with filtered results, return only filtered nodes
        if self.search_mode and self.filtered_nodes:
            return self.filtered_nodes

        return self.root_node.get_all_visible_nodes()[1:]  # Skip root node

    def _move_selection(self, direction: int):
        """Move selection up (-1) or down (1)."""
        nodes = self._get_all_nodes()
        if not nodes:
            return

        try:
            current_index = nodes.index(self.selected_node)
            new_index = (current_index + direction) % len(nodes)
            self.selected_node = nodes[new_index]
        except ValueError:
            self.selected_node = nodes[0]

        self._ensure_selected_visible()

    def _ensure_selected_visible(self):
        """Ensure the selected node is visible by adjusting scroll offset."""
        nodes = self._get_all_nodes()
        if not nodes:
            return

        try:
            selected_index = nodes.index(self.selected_node)

            # If selected item is above visible area, scroll up
            if selected_index < self.scroll_offset:
                self.scroll_offset = selected_index

            # If selected item is below visible area, scroll down
            elif selected_index >= self.scroll_offset + self.visible_height:
                self.scroll_offset = selected_index - self.visible_height + 1

            # Keep scroll offset within bounds
            max_scroll = max(0, len(nodes) - self.visible_height)
            self.scroll_offset = max(0, min(self.scroll_offset, max_scroll))

        except ValueError:
            pass

    def _toggle_expand(self):
        """Toggle expansion of the selected node."""
        if self.selected_node.node_type == "section" and self.selected_node.children:
            self.selected_node.expanded = not self.selected_node.expanded

    def _expand_collapse_horizontal(self, expand: bool):
        """Expand or collapse selected node based on left/right arrow."""
        if self.selected_node.node_type == "section" and self.selected_node.children:
            self.selected_node.expanded = expand

    def _cycle_tree_view_state(self):
        """Cycle through tree view states: modified_only -> all_expanded -> all_collapsed."""
        self.current_tree_state = (self.current_tree_state + 1) % len(
            self.tree_view_states
        )
        state = self.tree_view_states[self.current_tree_state]

        if state == "modified_only":
            self._apply_modified_only_view()
        elif state == "all_expanded":
            self._apply_all_expanded_view()
        elif state == "all_collapsed":
            self._apply_all_collapsed_view()

    def _apply_modified_only_view(self):
        """Show only non-default settings (default view)."""
        # First check if there are ANY non-default values in the tree
        has_any_non_defaults = self._has_non_default_descendants(self.root_node)

        def set_visibility_modified_only(node):
            if node.node_type == "root":
                # Root node is always visible
                node.visible = True
            elif node.node_type == "field":
                if has_any_non_defaults:
                    # Dict entries and action nodes should be visible if they have non-default children
                    if node.is_dict_entry or node.is_action_node:
                        node.visible = (
                            node.is_action_node
                            or self._has_non_default_descendants(node)
                            or any(
                                child.differs_from_default for child in node.children
                            )
                        )
                    else:
                        # Show only non-default fields when they exist
                        node.visible = node.differs_from_default
                else:
                    # Show all fields when no non-defaults exist
                    node.visible = True
            elif node.node_type == "section":
                if has_any_non_defaults:
                    # Show sections that contain non-default fields
                    has_non_default_children = any(
                        child.differs_from_default
                        if child.node_type == "field"
                        else self._has_non_default_descendants(child)
                        for child in node.children
                    )
                    node.visible = has_non_default_children
                    node.expanded = has_non_default_children  # Auto-expand sections with non-default fields
                else:
                    # Show all sections when no non-defaults exist, but collapsed
                    node.visible = True
                    node.expanded = False

            for child in node.children:
                set_visibility_modified_only(child)

        set_visibility_modified_only(self.root_node)

    def _apply_all_expanded_view(self):
        """Show everything expanded."""

        def expand_all(node):
            node.visible = True
            if node.node_type == "section":
                node.expanded = True
            for child in node.children:
                expand_all(child)

        expand_all(self.root_node)

    def _apply_all_collapsed_view(self):
        """Show everything but collapsed."""

        def collapse_all(node):
            node.visible = True
            if node.node_type == "section":
                node.expanded = False
            for child in node.children:
                collapse_all(child)

        collapse_all(self.root_node)

    def _has_modified_descendants(self, node):
        """Check if a node has any modified descendants."""
        for child in node.children:
            if child.node_type == "field" and child.modified:
                return True
            elif child.node_type == "section" and self._has_modified_descendants(child):
                return True
        return False

    def _has_non_default_descendants(self, node):
        """Check if a node has any non-default descendants."""
        for child in node.children:
            if child.node_type == "field" and child.differs_from_default:
                return True
            elif child.node_type == "section" and self._has_non_default_descendants(
                child
            ):
                return True
        return False

    def _edit_selected_field(self):
        """Edit the selected field if it's editable."""
        if self.selected_node.node_type != "field":
            return

        # Enter editing mode
        field_path = self.selected_node.path
        current_value = self.selected_node.value
        metadata = self.selected_node.metadata
        field_type = metadata.get("type", "string")

        self.editing_mode = True
        self.edit_field_info = (field_path, current_value, metadata)
        self.edit_field_type = field_type
        # Track if field started empty/unset (for backspace-to-exit behavior)
        self.edit_started_empty = current_value is None or (
            isinstance(current_value, str) and current_value.strip() == ""
        )
        # Track original value for undo functionality
        self.edit_original_value = current_value

        # Initialize cursor state for text fields
        if field_type in ("string", "int", "float"):
            self.edit_cursor_position = (
                len(str(current_value)) if current_value is not None else 0
            )
        else:
            # Reset cursor position for non-text fields
            self.edit_cursor_position = 0

        # Initialize widget-specific state based on field type
        self._initialize_widget_state(field_type, current_value, metadata)

    def _initialize_widget_state(self, field_type: str, current_value, metadata: dict):
        """Initialize widget state based on field type."""
        if field_type == "bool":
            # Boolean: track current boolean value
            self.edit_input_value = (
                bool(current_value) if current_value is not None else False
            )
        elif field_type == "choice":
            # Choice: track available choices and selected index
            self.edit_choices = metadata.get("choices", [])
            current_str = str(current_value) if current_value is not None else ""
            try:
                self.edit_selected_choice = self.edit_choices.index(current_str)
            except ValueError:
                self.edit_selected_choice = 0
        elif field_type == "list":
            # List: track current list items
            self.edit_list_items = list(current_value) if current_value else []
        elif field_type in ("int", "float"):
            # Numeric: track current value as string for editing
            self.edit_input_value = (
                str(current_value) if current_value is not None else "0"
            )
        else:
            # String and others: free-form text
            self.edit_input_value = (
                str(current_value) if current_value is not None else ""
            )

    def _handle_action_node(self):
        """Handle action nodes like [+ Add] entries."""
        if not self.selected_node.is_action_node:
            return

        action = self.selected_node.metadata.get("action")
        if action == "add_entry":
            self._add_dict_entry()

    def _add_dict_entry(self):
        """Add a new entry to a structured dict."""
        # Find the parent structured dict node
        parent_node = self.selected_node.parent
        if not parent_node or parent_node.metadata.get("type") != "structured_dict":
            return

        # Get value model info
        value_model_name = self.selected_node.metadata.get("value_model", "")

        # Simple implementation: prompt for key name
        # TODO: This should be replaced with a proper input widget
        self.console.print(f"\n[bold]Add new {value_model_name}[/bold]")
        self.console.print("Enter key name (or ESC to cancel):")

        # For now, just show placeholder - will implement proper input later
        self.console.print("[dim]TODO: Implement key input widget[/dim]")

    def _get_value_model_class(self, value_model_name: str) -> type:
        """Get the BaseModel class from the value model name."""
        if value_model_name == "AgentSpec":
            return IfetcherConfig.AgentSpec
        elif value_model_name == "Kind":
            return IfetcherConfig.Task.Kind
        # Add more mappings as needed for other structured dict types
        return None

    def _handle_edit_input(self, key: str) -> bool:
        """Handle keyboard input during editing mode. Returns True if editing should continue."""
        # Handle widget-specific input first (some widgets need custom ENTER/ESC handling)
        if self.edit_field_type == "bool":
            return self._handle_boolean_input(key)
        elif self.edit_field_type == "choice":
            return self._handle_choice_input(key)
        elif self.edit_field_type == "list":
            return self._handle_list_input(key)
        elif self.edit_field_type in ("int", "float"):
            return self._handle_numeric_input(key)

        # Default handling for other widgets (string)
        if key == "ENTER":
            # Save the value
            self._save_edit_value()
            return False
        elif key in ("ESC", "CTRL+D"):
            # Cancel editing
            self._cancel_editing()
            return False
        elif key == "CTRL+C":
            # Cancel just this input (don't exit entire editing session)
            return False
        else:
            # Advanced string input handling with cursor movement
            if key == "LEFT":
                # Move cursor left
                if self.edit_cursor_position > 0:
                    self.edit_cursor_position -= 1
            elif key == "RIGHT":
                # Move cursor right
                if self.edit_cursor_position < len(self.edit_input_value):
                    self.edit_cursor_position += 1
            elif key == "HOME":
                # Move to start of line
                self.edit_cursor_position = 0
            elif key == "END":
                # Move to end of line
                self.edit_cursor_position = len(self.edit_input_value)
            elif key == "CTRL+LEFT":
                # Move to previous word boundary
                self.edit_cursor_position = self._find_word_boundary(
                    self.edit_input_value, self.edit_cursor_position, -1
                )
            elif key == "CTRL+RIGHT":
                # Move to next word boundary
                self.edit_cursor_position = self._find_word_boundary(
                    self.edit_input_value, self.edit_cursor_position, 1
                )
            elif key == "BACKSPACE":
                # Remove character before cursor, or cancel editing if empty
                if self.edit_cursor_position > 0:
                    # Remove character at cursor-1 position
                    before = self.edit_input_value[: self.edit_cursor_position - 1]
                    after = self.edit_input_value[self.edit_cursor_position :]
                    self.edit_input_value = before + after
                    self.edit_cursor_position -= 1

                    # Visual cursor will update on next refresh
                elif not self.edit_input_value and self.edit_started_empty:
                    # Input is empty and we started with empty field, cancel editing
                    self._cancel_editing()
                    return False
            elif key == "DELETE":
                # Remove character at cursor position
                if self.edit_cursor_position < len(self.edit_input_value):
                    before = self.edit_input_value[: self.edit_cursor_position]
                    after = self.edit_input_value[self.edit_cursor_position + 1 :]
                    self.edit_input_value = before + after

                    # Visual cursor will update on next refresh
            elif key == "SPACE":
                # Insert space character
                before = self.edit_input_value[: self.edit_cursor_position]
                after = self.edit_input_value[self.edit_cursor_position :]
                self.edit_input_value = before + " " + after
                self.edit_cursor_position += 1
            elif len(key) == 1 and key.isprintable():
                # Insert character at cursor position
                before = self.edit_input_value[: self.edit_cursor_position]
                after = self.edit_input_value[self.edit_cursor_position :]
                self.edit_input_value = before + key + after
                self.edit_cursor_position += 1

                # Visual cursor will update on next refresh

        return True

    def _handle_boolean_input(self, key: str) -> bool:
        """Handle keyboard input for boolean widget."""
        if key in ("LEFT", "RIGHT", " ", "SPACE"):
            # Toggle the boolean value
            self.edit_input_value = not self.edit_input_value
        elif key in ("BACKSPACE", "CTRL+D", "CTRL+C"):
            # Cancel editing for non-textual input
            self._cancel_editing()
            return False
        return True

    def _handle_choice_input(self, key: str) -> bool:
        """Handle keyboard input for choice widget."""
        if key == "UP":
            # Move selection up
            self.edit_selected_choice = max(0, self.edit_selected_choice - 1)
        elif key == "DOWN":
            # Move selection down
            self.edit_selected_choice = min(
                len(self.edit_choices) - 1, self.edit_selected_choice + 1
            )
        elif key in ("BACKSPACE", "CTRL+D", "CTRL+C"):
            # Cancel editing for non-textual input
            self._cancel_editing()
            return False
        return True

    def _handle_list_input(self, key: str) -> bool:
        """Handle keyboard input for list widget."""
        if self.edit_list_mode == "browse":
            # Browse mode - navigate list and execute commands
            if key == "UP":
                # Move selection up
                if self.edit_list_items:
                    self.edit_list_selected = max(0, self.edit_list_selected - 1)
            elif key == "DOWN":
                # Move selection down
                if self.edit_list_items:
                    self.edit_list_selected = min(
                        len(self.edit_list_items) - 1, self.edit_list_selected + 1
                    )
            elif key == "ALT+UP":
                # Move selected item up in the list
                if self.edit_list_items and self.edit_list_selected > 0:
                    # Swap with item above
                    current_idx = self.edit_list_selected
                    (
                        self.edit_list_items[current_idx],
                        self.edit_list_items[current_idx - 1],
                    ) = (
                        self.edit_list_items[current_idx - 1],
                        self.edit_list_items[current_idx],
                    )
                    # Move selection up to follow the moved item
                    self.edit_list_selected = current_idx - 1
            elif key == "ALT+DOWN":
                # Move selected item down in the list
                if (
                    self.edit_list_items
                    and self.edit_list_selected < len(self.edit_list_items) - 1
                ):
                    # Swap with item below
                    current_idx = self.edit_list_selected
                    (
                        self.edit_list_items[current_idx],
                        self.edit_list_items[current_idx + 1],
                    ) = (
                        self.edit_list_items[current_idx + 1],
                        self.edit_list_items[current_idx],
                    )
                    # Move selection down to follow the moved item
                    self.edit_list_selected = current_idx + 1
            elif key == "+":
                # Enter add mode
                self.edit_list_mode = "add"
                self.edit_list_input = ""
                self.edit_cursor_position = 0
            elif key == "-":
                # Remove selected item
                if self.edit_list_items and 0 <= self.edit_list_selected < len(
                    self.edit_list_items
                ):
                    self.edit_list_items.pop(self.edit_list_selected)
                    # Adjust selection if needed
                    if self.edit_list_selected >= len(self.edit_list_items):
                        self.edit_list_selected = max(0, len(self.edit_list_items) - 1)
            elif key in ("ENTER", "\r", "\n"):
                if self.edit_list_items:
                    # Enter edit mode for selected item
                    self.edit_list_mode = "edit"
                    self.edit_list_input = self.edit_list_items[self.edit_list_selected]
                    self.edit_cursor_position = len(self.edit_list_input)
                else:
                    # No items - save the list and exit editing
                    self._save_edit_value()
                    return False
            elif key in ("CTRL+D", "CTRL+C"):
                # Save current list and exit editing
                self._save_edit_value()
                return False
            elif key == "ESC":
                # Cancel editing
                self._cancel_editing()
                return False

        elif self.edit_list_mode in ("add", "edit"):
            # Input mode - handle text input
            if key == "ENTER":
                # Save the add/edit
                if self.edit_list_input.strip():
                    if self.edit_list_mode == "add":
                        # Add new item
                        self.edit_list_items.append(self.edit_list_input.strip())
                        self.edit_list_selected = len(self.edit_list_items) - 1
                    else:  # edit
                        # Update existing item
                        if 0 <= self.edit_list_selected < len(self.edit_list_items):
                            self.edit_list_items[self.edit_list_selected] = (
                                self.edit_list_input.strip()
                            )

                # Return to browse mode
                self.edit_list_mode = "browse"
                self.edit_list_input = ""
                self.edit_cursor_position = 0

            elif key == "ESC":
                # Cancel add/edit
                self.edit_list_mode = "browse"
                self.edit_list_input = ""
                self.edit_cursor_position = 0
            elif key in ("CTRL+D", "CTRL+C"):
                # Save current list and exit editing entirely
                self._save_edit_value()
                return False

            # Enhanced cursor-based input handling for list items
            elif key == "LEFT":
                # Move cursor left
                if self.edit_cursor_position > 0:
                    self.edit_cursor_position -= 1
            elif key == "RIGHT":
                # Move cursor right
                if self.edit_cursor_position < len(self.edit_list_input):
                    self.edit_cursor_position += 1
            elif key == "HOME":
                # Move to start of line
                self.edit_cursor_position = 0
            elif key == "END":
                # Move to end of line
                self.edit_cursor_position = len(self.edit_list_input)
            elif key == "CTRL+LEFT":
                # Move to previous word boundary
                self.edit_cursor_position = self._find_word_boundary(
                    self.edit_list_input, self.edit_cursor_position, -1
                )
            elif key == "CTRL+RIGHT":
                # Move to next word boundary
                self.edit_cursor_position = self._find_word_boundary(
                    self.edit_list_input, self.edit_cursor_position, 1
                )
            elif key == "BACKSPACE":
                # Remove character before cursor, or cancel editing if empty
                if self.edit_cursor_position > 0:
                    # Remove character at cursor-1 position
                    before = self.edit_list_input[: self.edit_cursor_position - 1]
                    after = self.edit_list_input[self.edit_cursor_position :]
                    self.edit_list_input = before + after
                    self.edit_cursor_position -= 1
                elif not self.edit_list_input:
                    # Input is empty, cancel editing
                    self.edit_list_mode = "browse"
                    self.edit_list_input = ""
                    self.edit_cursor_position = 0
            elif key == "DELETE":
                # Remove character at cursor position
                if self.edit_cursor_position < len(self.edit_list_input):
                    before = self.edit_list_input[: self.edit_cursor_position]
                    after = self.edit_list_input[self.edit_cursor_position + 1 :]
                    self.edit_list_input = before + after
            elif key == "SPACE":
                # Insert space character
                before = self.edit_list_input[: self.edit_cursor_position]
                after = self.edit_list_input[self.edit_cursor_position :]
                self.edit_list_input = before + " " + after
                self.edit_cursor_position += 1
            elif len(key) == 1 and key.isprintable():
                # Insert character at cursor position
                before = self.edit_list_input[: self.edit_cursor_position]
                after = self.edit_list_input[self.edit_cursor_position :]
                self.edit_list_input = before + key + after
                self.edit_cursor_position += 1

        return True

    def _handle_numeric_input(self, key: str) -> bool:
        """Handle keyboard input for numeric fields (int/float)."""
        if key == "ENTER":
            # Save the value
            self._save_edit_value()
            return False
        elif key == "ESC":
            # Cancel editing
            self._cancel_editing()
            return False
        elif key == "CTRL+D":
            # Save and exit
            self._save_edit_value()
            return False
        elif key.upper() == "R":
            # Reset to default value
            self._reset_to_default()
            return False
        elif key.upper() == "U":
            # Undo changes (restore original value)
            self._undo_changes()
            return False
        elif key == "UP":
            # Increment value with smart increment size
            try:
                current = float(self.edit_input_value) if self.edit_input_value else 0
                if self.edit_field_type == "int":
                    increment = self._calculate_smart_increment(self.edit_input_value)
                    self.edit_input_value = str(int(current) + increment)
                else:  # float
                    self.edit_input_value = str(round(current + 0.1, 1))
                self.edit_cursor_position = len(self.edit_input_value)
            except ValueError:
                pass  # Invalid number, ignore
        elif key == "DOWN":
            # Decrement value with smart increment size
            try:
                current = float(self.edit_input_value) if self.edit_input_value else 0
                if self.edit_field_type == "int":
                    increment = self._calculate_smart_increment(self.edit_input_value)
                    self.edit_input_value = str(max(0, int(current) - increment))
                else:  # float
                    self.edit_input_value = str(round(max(0.0, current - 0.1), 1))
                self.edit_cursor_position = len(self.edit_input_value)
            except ValueError:
                pass  # Invalid number, ignore
        elif key == "LEFT":
            # Move cursor left
            if self.edit_cursor_position > 0:
                self.edit_cursor_position -= 1
        elif key == "RIGHT":
            # Move cursor right
            if self.edit_cursor_position < len(self.edit_input_value):
                self.edit_cursor_position += 1
        elif key == "HOME":
            # Move to start of line
            self.edit_cursor_position = 0
        elif key == "END":
            # Move to end of line
            self.edit_cursor_position = len(self.edit_input_value)
        elif key == "CTRL+LEFT":
            # Move to previous word boundary
            self.edit_cursor_position = self._find_word_boundary(
                self.edit_input_value, self.edit_cursor_position, -1
            )
        elif key == "CTRL+RIGHT":
            # Move to next word boundary
            self.edit_cursor_position = self._find_word_boundary(
                self.edit_input_value, self.edit_cursor_position, 1
            )
        elif key == "BACKSPACE":
            # Remove character before cursor
            if self.edit_cursor_position > 0:
                before = self.edit_input_value[: self.edit_cursor_position - 1]
                after = self.edit_input_value[self.edit_cursor_position :]
                self.edit_input_value = before + after
                self.edit_cursor_position -= 1
            elif not self.edit_input_value and self.edit_started_empty:
                # Input is empty and we started with empty field, cancel editing
                self._cancel_editing()
                return False
        elif key == "DELETE":
            # Remove character at cursor position
            if self.edit_cursor_position < len(self.edit_input_value):
                before = self.edit_input_value[: self.edit_cursor_position]
                after = self.edit_input_value[self.edit_cursor_position + 1 :]
                self.edit_input_value = before + after
        elif key == "SPACE":
            # Ignore spaces in numeric input
            pass
        elif len(key) == 1 and key.isprintable():
            # Only allow numeric characters, decimal point, and minus sign
            if (
                key.isdigit()
                or (
                    key == "."
                    and self.edit_field_type == "float"
                    and "." not in self.edit_input_value
                )
                or (key == "-" and self.edit_cursor_position == 0)
            ):
                # Insert character at cursor position
                before = self.edit_input_value[: self.edit_cursor_position]
                after = self.edit_input_value[self.edit_cursor_position :]
                self.edit_input_value = before + key + after
                self.edit_cursor_position += 1

        return True

    def _get_numeric_display_with_cursor(self) -> str:
        """Get numeric display with cursor and dimmed trailing zeros."""
        text = self.edit_input_value or "0"

        # Find trailing zeros to dim them
        if text.isdigit() and len(text) > 1:
            # Count trailing zeros
            trailing_zeros = 0
            for i in range(len(text) - 1, -1, -1):
                if text[i] == "0":
                    trailing_zeros += 1
                else:
                    break

            if trailing_zeros > 0:
                # Split into significant digits and trailing zeros
                significant_part = text[:-trailing_zeros]
                zero_part = text[-trailing_zeros:]

                # Apply cursor to the appropriate part
                if self.edit_cursor_position < len(significant_part):
                    # Cursor in significant digits
                    before = significant_part[: self.edit_cursor_position]
                    if self.edit_cursor_position >= len(significant_part):
                        cursor_display = f"{significant_part}[reverse] [/reverse][dim]{zero_part}[/dim]"
                    else:
                        char_at_cursor = significant_part[self.edit_cursor_position]
                        after = significant_part[self.edit_cursor_position + 1 :]
                        cursor_display = f"{before}[reverse]{char_at_cursor}[/reverse]{after}[dim]{zero_part}[/dim]"
                    return cursor_display
                else:
                    # Cursor in trailing zeros
                    zero_cursor_pos = self.edit_cursor_position - len(significant_part)
                    before_zeros = zero_part[:zero_cursor_pos]
                    if zero_cursor_pos >= len(zero_part):
                        cursor_display = f"{significant_part}[dim]{zero_part}[/dim][reverse] [/reverse]"
                    else:
                        char_at_cursor = zero_part[zero_cursor_pos]
                        after_zeros = zero_part[zero_cursor_pos + 1 :]
                        cursor_display = f"{significant_part}[dim]{before_zeros}[reverse]{char_at_cursor}[/reverse]{after_zeros}[/dim]"
                    return cursor_display

        # Fallback to normal cursor display for non-pattern numbers
        if not text:
            return "[reverse] [/reverse]"
        else:
            before = text[: self.edit_cursor_position]
            if self.edit_cursor_position >= len(text):
                return f"{before}[reverse] [/reverse]"
            else:
                char_at_cursor = text[self.edit_cursor_position]
                after = text[self.edit_cursor_position + 1 :]
                return f"{before}[reverse]{char_at_cursor}[/reverse]{after}"

    def _calculate_smart_increment(self, value_str: str) -> int:
        """Calculate smart increment size based on value pattern (X00000 -> 10^N)."""
        if not value_str or not value_str.isdigit():
            return 1

        # Count trailing zeros
        trailing_zeros = 0
        for i in range(len(value_str) - 1, -1, -1):
            if value_str[i] == "0":
                trailing_zeros += 1
            else:
                break

        if trailing_zeros == 0:
            return 1

        # Get the first digit(s) before the zeros
        prefix = value_str[:-trailing_zeros] if trailing_zeros < len(value_str) else "1"

        # If prefix is "1", use 10^(N-1), otherwise use 10^N
        if prefix == "1" and trailing_zeros > 0:
            return 10 ** (trailing_zeros - 1)
        else:
            return 10**trailing_zeros

    def _save_edit_value(self):
        """Save the edited value."""
        if not self.edit_field_info:
            return

        field_path, current_value, metadata = self.edit_field_info
        field_type = metadata.get("type", "string")

        try:
            # Convert widget state to appropriate value type
            if field_type == "bool":
                # Boolean widget stores the boolean value directly
                new_value = self.edit_input_value
            elif field_type == "choice":
                # Choice widget stores the selected choice
                if self.edit_choices and 0 <= self.edit_selected_choice < len(
                    self.edit_choices
                ):
                    new_value = self.edit_choices[self.edit_selected_choice]
                else:
                    new_value = (
                        current_value  # Keep current value if selection is invalid
                    )
            elif field_type == "list":
                # List widget stores the list items
                new_value = list(self.edit_list_items) if self.edit_list_items else []
            elif field_type == "int":
                new_value_str = str(self.edit_input_value).strip()
                new_value = int(new_value_str) if new_value_str else 0
            elif field_type == "float":
                new_value_str = str(self.edit_input_value).strip()
                new_value = float(new_value_str) if new_value_str else 0.0
            else:  # string and others
                new_value = str(self.edit_input_value).strip()

            # Only update if the value actually changed
            if new_value != current_value:
                # Update the selected node
                self.selected_node.value = new_value
                self.selected_node.modified = True
                self.selected_node.is_explicit = True

                # Update configurations
                ConfigNavigator.set_value(self.working_config, field_path, new_value)
                ConfigNavigator.set_value(self.explicit_config, field_path, new_value)
                self.modified = True

        except (ValueError, TypeError):
            # Invalid input - could show error message
            pass

        self._cancel_editing()

    def _cancel_editing(self):
        """Cancel editing mode."""
        # Hide cursor when exiting editing (clean up)
        self._hide_cursor()

        self.editing_mode = False
        self.edit_field_info = None
        self.edit_field_type = "string"
        self.edit_input_value = ""
        self.edit_choices = []
        self.edit_list_items = []
        self.edit_selected_choice = 0
        self.edit_list_selected = 0
        self.edit_list_mode = "browse"
        self.edit_list_input = ""

        # Reset cursor position tracking
        self.edit_cursor_position = 0
        self.input_field_row = 0
        self.input_field_col = 0

    def _reset_to_default(self):
        """Reset the current field to its default value."""
        if self.selected_node.node_type != "field":
            return

        field_path = self.selected_node.path

        # Get default value
        default_config = IfetcherConfig()
        default_value = ConfigNavigator.get_value(default_config.__dict__, field_path)

        # Always update the field to default value (regardless of current editor state)
        self.selected_node.value = default_value
        self.selected_node.modified = True
        self.selected_node.differs_from_default = False  # Now matches default

        # Update configurations
        ConfigNavigator.set_value(self.working_config, field_path, default_value)
        ConfigNavigator.set_value(self.explicit_config, field_path, default_value)
        self.modified = True

        # Exit editing mode if we're currently editing
        if self.editing_mode:
            self._cancel_editing()

    def _undo_changes(self):
        """Undo changes and restore the original value when the session started."""
        if self.selected_node.node_type != "field":
            return

        field_path = self.selected_node.path

        # Get original value from session start
        original_value = ConfigNavigator.get_value(self.original_config, field_path)

        # Always restore the original value (regardless of current editor state)
        self.selected_node.value = original_value
        # Check if original value differs from default to set differs_from_default correctly
        default_config = IfetcherConfig()
        default_value = ConfigNavigator.get_value(default_config.__dict__, field_path)
        self.selected_node.differs_from_default = original_value != default_value
        self.selected_node.modified = (
            False  # No longer modified since we're back to original
        )

        # Update configurations
        ConfigNavigator.set_value(self.working_config, field_path, original_value)
        ConfigNavigator.set_value(self.explicit_config, field_path, original_value)

        # Exit editing mode if we're currently editing
        if self.editing_mode:
            self._cancel_editing()

    def _get_editor_content(self) -> Panel:
        """Get content for the editor panel."""
        if self.selected_node.node_type == "field":
            # Show field editor with special handling for list fields
            field_type = self.selected_node.metadata.get("type", "string")

            if field_type == "list":
                return self._get_list_field_view_panel()

            # Regular field display
            field_info = [
                f"[bold]Field:[/bold] {self.selected_node.path}",
                f"[bold]Type:[/bold] {field_type}",
                f"[bold]Current Value:[/bold] {self.selected_node.get_display_value()}",
                "",
                f"[dim]{self.selected_node.metadata.get('help', 'No description available')}[/dim]",
            ]

            if self.selected_node.modified:
                field_info.insert(0, "[yellow]* Modified (unsaved)[/yellow]")

            content = "\n".join(field_info)
            return Panel(content, title="Field Info", border_style="green")

        elif self.selected_node.node_type == "section":
            # Show section info
            child_count = len(self.selected_node.children)
            expanded_status = "Expanded" if self.selected_node.expanded else "Collapsed"

            content = f"[bold]{self.selected_node.name}[/bold]\n\n"
            content += f"Children: {child_count}\n"
            content += f"Status: {expanded_status}\n\n"
            content += "[dim]Navigate with arrow keys\n"
            content += "Press Tab to expand/collapse\n"
            content += "Press Enter to edit fields[/dim]"

            return Panel(content, title="Section Info", border_style="blue")

        else:
            return Panel("Select a field to edit", title="Editor", border_style="dim")

    def _get_header_content(self) -> Panel:
        """Get content for the header."""
        status = "Modified" if self.modified else "Unmodified"
        status_color = "yellow" if self.modified else "green"

        header_text = f"[bold]Configuration Editor[/bold] - {self.config_path.name}"
        header_text += f" | Status: [{status_color}]{status}[/{status_color}]"

        if self.editing_mode:
            # Dim the header when editing
            return Panel(header_text, style="dim")
        else:
            return Panel(header_text, style="blue")

    def _get_list_footer_text(self) -> str:
        """Get context-sensitive footer text for list editing."""
        if self.edit_list_mode == "browse":
            return (
                "[bold yellow]List Mode:[/bold yellow] [bold magenta]↑↓[/bold magenta] Navigate | [bold magenta]Alt+↑↓[/bold magenta] Reorder | "
                "[bold cyan]+[/bold cyan] Add | [bold red]-[/bold red] Remove | "
                "[bold green]Enter[/bold green] Edit | [bold red]⌃D[/bold red] Exit"
            )
        elif self.edit_list_mode == "add":
            return (
                "[bold yellow]Add Item:[/bold yellow] Type new value | "
                "[bold green]Enter[/bold green] Save | [bold red]ESC[/bold red] Cancel | [bold red]⌃D[/bold red] Exit"
            )
        else:  # edit
            return (
                "[bold yellow]Edit Item:[/bold yellow] Modify value | "
                "[bold green]Enter[/bold green] Save | [bold red]ESC[/bold red] Cancel | [bold red]⌃D[/bold red] Exit"
            )

    def _build_responsive_footer(
        self, keybindings_with_priority: list, prefix: str = ""
    ) -> str:
        """Build responsive footer text that fits available width.

        Args:
            keybindings_with_priority: List of (binding_text, priority, display_order) tuples
            prefix: Optional prefix text
        """
        from rich.text import Text

        # Calculate available width (account for panel borders and padding)
        available_width = max(40, self.console.size.width - 6)  # Leave some margin

        # Start with prefix if provided
        current_text = (
            f"[bold bright_black]{prefix}[/bold bright_black] " if prefix else ""
        )

        # Sort by priority (higher priority first) to determine which to include
        priority_sorted = sorted(
            keybindings_with_priority, key=lambda x: x[1], reverse=True
        )

        # Select bindings that fit, maintaining priority order
        selected_bindings = []
        for binding_text, priority, display_order in priority_sorted:
            # Test if adding this binding would fit
            test_line = current_text
            if selected_bindings:
                test_line += " [dim]•[/dim] ".join([b[0] for b in selected_bindings])
                test_line += " [dim]•[/dim] " + binding_text
            else:
                test_line += binding_text

            # Measure visual width (Rich handles markup)
            visual_width = Text.from_markup(test_line).cell_len

            if visual_width <= available_width:
                selected_bindings.append((binding_text, priority, display_order))
            # If it doesn't fit, skip this binding (try next priority)

        # Now arrange selected bindings in display order (left to right)
        display_sorted = sorted(selected_bindings, key=lambda x: x[2])

        # Build final text in display order
        if display_sorted:
            binding_texts = [binding[0] for binding in display_sorted]
            current_text += " [dim]•[/dim] ".join(binding_texts)

        return current_text

    def _get_footer_content(self) -> Panel:
        """Get content for the footer."""
        if self.editing_mode:
            # Show editing-specific help with more emphasis and context-sensitive commands
            if self.edit_field_type == "list":
                help_text = self._get_list_footer_text()
            else:
                # Priority-ordered editing keybindings: navigation left, save/quit right, actions between
                editing_bindings = [
                    (
                        "[bold green]Enter[/bold green] Save",
                        10,
                        1,
                    ),  # Save (highest priority)
                    (
                        "[bold red]⌃D[/bold red] Cancel",
                        9,
                        2,
                    ),  # Cancel (high priority)
                ]
                help_text = self._build_responsive_footer(
                    editing_bindings,
                    "[yellow]Editing Mode:[/yellow] Type new value [dim]•[/dim]",
                )
            return Panel(help_text, style="bold", border_style="yellow")
        elif self.search_mode:
            # Show search mode help and status
            search_status = self._get_search_status_text()
            # Priority-ordered search keybindings
            search_bindings = [
                ("[bold cyan]↑↓[/bold cyan] Navigate results", 10, 1),
                ("[bold green]Enter[/bold green] Exit search (keep filter)", 9, 2),
            ]
            help_text = f"{search_status}\n" + self._build_responsive_footer(
                search_bindings, "Keys: Type to search"
            )
            return Panel(help_text, style="bold", border_style="green")
        else:
            # Normal navigation help with current view state
            current_state = (
                self.tree_view_states[self.current_tree_state].replace("_", " ").title()
            )
            # (text, priority, display_order) - priority determines inclusion, display_order determines left-to-right position
            nav_bindings = [
                (
                    "[cyan]↑↓→←[/cyan] Navigate",
                    10,
                    1,
                ),  # Navigation on left (highest priority)
                ("[cyan]Tab[/cyan] Toggle", 8, 2),  # Navigation on left
                (
                    f"[cyan]Shift+Tab[/cyan] View ({current_state})",
                    2,
                    3,
                ),  # Low priority navigation, grouped with Tab
                ("[bold green]/[/bold green] Search", 5, 4),  # Search functionality
                ("[green]Enter[/green] Edit", 9, 5),  # Edit in middle
                ("[bold magenta]R[/bold magenta]eset", 7, 6),  # Reset action
                ("[bold magenta]U[/bold magenta]ndo", 6, 7),  # Undo action
                ("[yellow]⌃S[/yellow] Save", 4, 8),  # Save on right
                (
                    "[red]⌃D[/red]/[bold red]Q[/bold red]uit",
                    3,
                    9,
                ),  # Quit on far right
            ]

            help_text = self._build_responsive_footer(nav_bindings, "Keys:")
            return Panel(help_text, border_style="bright_black")

    def _save_config(self) -> bool:
        """Save the configuration."""
        try:
            # Validate first
            IfetcherConfig.model_validate(self.working_config)

            # Save to file
            ConfigFileManager.save_config(self.working_config, self.config_path)

            # Update state
            self.original_config = copy.deepcopy(self.working_config)
            self.modified = False

            # Mark all nodes as unmodified
            def clear_modified(node: TreeNode):
                node.modified = False
                for child in node.children:
                    clear_modified(child)

            clear_modified(self.root_node)

            return True

        except ValidationError as e:
            self.console.print("[red]Validation Error:[/red]")
            for error in e.errors():
                field = ".".join(str(loc) for loc in error["loc"])
                msg = error["msg"]
                self.console.print(f"  {field}: {msg}")
            return False
        except Exception as e:
            self.console.print(f"[red]Save failed: {e}[/red]")
            return False

    def run(self):
        """Main run loop with Live display."""
        try:
            with Live(
                self.layout, console=self.console, screen=True, auto_refresh=False
            ) as live:
                while True:
                    # Update layout components with editing mode styling
                    self.layout["header"].update(self._get_header_content())
                    self.layout["tree"].update(self._get_tree_panel())
                    # Update editor layout based on editing state
                    self._update_editor_layout()
                    self.layout["footer"].update(self._get_footer_content())

                    live.refresh()

                    # Handle keyboard input
                    key = KeyboardInput.get_key()

                    if self.editing_mode:
                        # Handle input during editing mode
                        if not self._handle_edit_input(key):
                            # Editing was cancelled or completed, continue with normal loop
                            pass
                    elif self.search_mode:
                        # Search mode - handle search input
                        if not self._handle_search_input(key):
                            # Search was cancelled, continue with normal loop
                            pass
                    else:
                        # Normal navigation mode
                        if key == "/":
                            self._enter_search_mode()
                        elif key == "UP":
                            self._move_selection(-1)
                        elif key == "DOWN":
                            self._move_selection(1)
                        elif key == "PAGE_UP":
                            self._move_selection(-10)
                        elif key == "PAGE_DOWN":
                            self._move_selection(10)
                        elif key == "LEFT":
                            self._expand_collapse_horizontal(False)  # Collapse
                        elif key == "RIGHT":
                            self._expand_collapse_horizontal(True)  # Expand
                        elif key == "TAB":
                            self._toggle_expand()
                        elif key == "SHIFT+TAB":
                            self._cycle_tree_view_state()
                        elif key in ("ENTER", "\r", "\n"):
                            if self.selected_node.is_action_node:
                                self._handle_action_node()
                            elif self.selected_node.node_type == "field":
                                self._edit_selected_field()
                        elif key.upper() == "R":
                            # Reset field to default value
                            if self.selected_node.node_type == "field":
                                self._reset_to_default()
                        elif key.upper() == "U":
                            # Undo field changes
                            if self.selected_node.node_type == "field":
                                self._undo_changes()
                        elif key == "\x13":  # ⌃S
                            live.stop()
                            if self._save_config():
                                self.console.print(
                                    "[green]Configuration saved successfully![/green]"
                                )
                            Prompt.ask("Press Enter to continue")
                            live.start()
                        elif key in ("CTRL+D", "q", "Q"):
                            if self.modified:
                                live.stop()
                                if not Confirm.ask(
                                    "You have unsaved changes. Quit anyway?"
                                ):
                                    live.start()
                                    continue
                            break

        except KeyboardInterrupt:
            if self.modified:
                self.console.print("\n[yellow]You have unsaved changes.[/yellow]")
                if not Confirm.ask("Quit without saving?"):
                    return

            self.console.print("\n[yellow]Editor cancelled by user[/yellow]")

    def _enter_search_mode(self):
        """Enter search/filter mode."""
        self.search_mode = True
        self.search_term = ""
        # Store current selection to restore if search is cancelled
        self.original_selected_node = self.selected_node

    def _exit_search_mode(self, restore_selection: bool = False):
        """Exit search/filter mode."""
        self.search_mode = False
        self.search_term = ""
        self.filtered_nodes = []

        # Optionally restore original selection
        if restore_selection and self.original_selected_node:
            self.selected_node = self.original_selected_node
        self.original_selected_node = None

    def _handle_search_input(self, key: str) -> bool:
        """Handle keyboard input during search mode. Returns True to continue searching."""
        if key == "ESC":
            # Cancel search and restore original selection
            self._exit_search_mode(restore_selection=True)
            return False
        elif key == "ENTER":
            # Accept search and keep filtered results
            self.search_mode = False
            self.original_selected_node = None
            return False
        elif key == "UP":
            # Navigate through filtered results
            self._move_selection(-1)
        elif key == "DOWN":
            # Navigate through filtered results
            self._move_selection(1)
        elif key == "PAGE_UP":
            # Navigate through filtered results with larger jump
            self._move_selection(-10)
        elif key == "PAGE_DOWN":
            # Navigate through filtered results with larger jump
            self._move_selection(10)
        elif key == "BACKSPACE":
            # Remove last character from search term, or exit search if empty
            if self.search_term:
                self.search_term = self.search_term[:-1]
                self._update_search_filter()
            else:
                # Search term is empty, exit search with cancel behavior
                self._exit_search_mode(restore_selection=True)
                return False
        elif len(key) == 1 and key.isprintable():
            # Add character to search term
            self.search_term += key
            self._update_search_filter()

        return True

    def _update_search_filter(self):
        """Update the filtered nodes based on current search term."""
        if not self.search_term:
            self.filtered_nodes = []
            return

        # Get all nodes from the full tree (not filtered) for searching
        all_nodes = self.root_node.get_all_visible_nodes()[1:]  # Skip root node
        self.filtered_nodes = []

        search_lower = self.search_term.lower()

        for node in all_nodes:
            # Check if node name matches
            if search_lower in node.name.lower():
                self.filtered_nodes.append(node)
                continue

            # For field nodes, also check the path and value
            if node.node_type == "field":
                if search_lower in node.path.lower() or (
                    node.value and search_lower in str(node.value).lower()
                ):
                    self.filtered_nodes.append(node)
                    continue

        # If we have filtered results, select the first one and reset scroll
        if self.filtered_nodes:
            self.selected_node = self.filtered_nodes[0]
            self.scroll_offset = 0

    def _get_search_status_text(self) -> str:
        """Get status text for search mode."""
        if not self.search_mode:
            return ""

        if not self.search_term:
            return (
                "[yellow]Search: /[/yellow] [dim](type to filter, Enter to exit)[/dim]"
            )

        count = len(self.filtered_nodes)
        if count == 0:
            return f"[red]Search: /{self.search_term}[/red] [dim](no matches - Enter to exit, ESC to cancel)[/dim]"
        else:
            return f"[green]Search: /{self.search_term}[/green] [dim]({count} matches - Enter to exit, ESC to cancel)[/dim]"

    # Cursor control methods
    def _show_cursor(self):
        """Show the terminal cursor."""
        if not self.cursor_visible:
            print("\033[?25h", end="", flush=True)
            self.cursor_visible = True

    def _hide_cursor(self):
        """Hide the terminal cursor."""
        if self.cursor_visible:
            print("\033[?25l", end="", flush=True)
            self.cursor_visible = False

    def _move_cursor_to_input_position(self):
        """Move terminal cursor to current input position."""
        if self.input_field_row > 0 and self.input_field_col > 0:
            actual_col = self.input_field_col + self.edit_cursor_position
            print(f"\033[{self.input_field_row};{actual_col}H", end="", flush=True)

    def _move_cursor_relative(self, columns: int):
        """Move cursor relative to current position."""
        if columns > 0:
            print(f"\033[{columns}C", end="", flush=True)
        elif columns < 0:
            print(f"\033[{abs(columns)}D", end="", flush=True)

    def _find_word_boundary(self, text: str, pos: int, direction: int) -> int:
        """Find next word boundary. Direction: -1 for left, 1 for right."""
        original_pos = pos
        if direction < 0:  # Moving left
            # Always move at least one position first (unless at start)
            if pos > 0:
                pos -= 1
            # Skip current word characters
            while pos > 0 and text[pos - 1].isalnum():
                pos -= 1
            # Skip whitespace
            while pos > 0 and text[pos - 1].isspace():
                pos -= 1
        else:  # Moving right
            # Always move at least one position first (unless at end)
            if pos < len(text):
                pos += 1
            # Skip current word characters
            while pos < len(text) and text[pos].isalnum():
                pos += 1
            # Skip whitespace
            while pos < len(text) and text[pos].isspace():
                pos += 1
        return max(0, min(len(text), pos))


# ============================================================================
# SECTION 11: CLI Integration Function
# ============================================================================
def run_config_editor(config_path: Optional[str] = None, use_tree: bool = True) -> None:
    """
    Entry point for the configuration editor.

    Can be called from cli.py as a new subcommand.

    Args:
        config_path: Optional path to config file to edit
        use_tree: Whether to use the new tree-based editor (default: True)
    """
    try:
        path = Path(config_path) if config_path else None

        if use_tree:
            editor = TreeConfigEditor(path)
        else:
            editor = ConfigEditor(path)

        editor.run()
    except KeyboardInterrupt:
        Console().print("\n[yellow]Editor cancelled by user[/yellow]")
    except Exception as e:
        console = Console()
        console.print(f"[red]Error starting configuration editor: {e}[/red]")
        console.print_exception()
        raise


def run_tree_config_editor(config_path: Optional[str] = None) -> None:
    """
    Entry point for the tree-based configuration editor.

    Args:
        config_path: Optional path to config file to edit
    """
    run_config_editor(config_path, use_tree=True)


def run_legacy_config_editor(config_path: Optional[str] = None) -> None:
    """
    Entry point for the legacy (non-tree) configuration editor.

    Args:
        config_path: Optional path to config file to edit
    """
    run_config_editor(config_path, use_tree=False)
