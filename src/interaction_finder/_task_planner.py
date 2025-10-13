"""
TaskPlanner: A three-stage task execution system with configuration, planning, and execution.

This module implements a clean separation of concerns:
1. Configuration: Define rules and dependencies once
2. Planning: Compute the minimal work DAG for given outputs
3. Execution: Execute the plan with serial/parallel options

The design allows for optimal task scheduling while keeping the API simple.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Set, Callable, Any, Optional, Union, Tuple
from enum import Enum
from pathlib import Path
import time


class TaskStatus(Enum):
    """Status of a task execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


@dataclass
class TaskRule:
    """
    Defines how to execute a task.

    This is the configuration stage - rules are defined once and reused.
    """

    name: str
    dependencies: List[str] = field(default_factory=list)
    executor: Optional[Callable] = None
    output_pattern: Optional[str] = None  # Pattern for expected output file
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"Execute {self.name}"


@dataclass
class TaskNode:
    """
    A node in the computed execution DAG.

    This is created during the planning stage.
    """

    rule: TaskRule
    dependencies: Set["TaskNode"] = field(default_factory=set)
    dependents: Set["TaskNode"] = field(default_factory=set)
    level: int = 0  # Execution level (0 = no deps, 1 = depends on level 0, etc.)
    status: TaskStatus = TaskStatus.PENDING
    path: Optional[Path] = None  # Path to cached result file
    error: Optional[str] = None
    execution_time: float = 0.0

    def __hash__(self) -> int:
        """Make TaskNode hashable based on rule name."""
        return hash(self.rule.name)

    def __eq__(self, other) -> bool:
        """TaskNodes are equal if they have the same rule name."""
        if not isinstance(other, TaskNode):
            return False
        return self.rule.name == other.rule.name

    @property
    def name(self) -> str:
        return self.rule.name

    def add_dependency(self, dep_node: "TaskNode") -> None:
        """Add a dependency relationship between nodes."""
        self.dependencies.add(dep_node)
        dep_node.dependents.add(self)

    def get_dependency_names(self) -> Set[str]:
        """Get names of dependency nodes."""
        return {node.name for node in self.dependencies}

    def get_dependent_names(self) -> Set[str]:
        """Get names of dependent nodes."""
        return {node.name for node in self.dependents}

    def is_ready(self, completed_nodes: Set["TaskNode"]) -> bool:
        """Check if this node is ready to execute (all dependencies completed)."""
        return all(dep in completed_nodes for dep in self.dependencies)

    def get_dependency_paths(self) -> Dict[str, Optional[Path]]:
        """Get file paths from all dependency nodes that have completed."""
        return {
            dep.name: dep.path
            for dep in self.dependencies
            if dep.status in (TaskStatus.COMPLETED, TaskStatus.CACHED)
        }

    async def load_result(self) -> Optional[Any]:
        """Load the actual result content from the cached file."""
        if self.path is None or not self.path.exists():
            return None

        try:
            # For JSON files (like chunks), parse as JSON
            if self.path.suffix == ".json":
                import json

                return json.loads(self.path.read_text())
            else:
                # For text files, return as string
                return self.path.read_text()
        except Exception:
            return None

    def has_failed_dependencies(self) -> bool:
        """Check if any dependencies have failed."""
        return any(dep.status == TaskStatus.FAILED for dep in self.dependencies)


class TaskPlanner:
    """
    Configuration stage: Define rules and dependencies for task execution.

    This class holds the "recipe book" - all the rules for how tasks relate
    to each other and how to execute them.
    """

    def __init__(self, max_concurrent: int = 5):
        self.max_concurrent = max_concurrent
        self.rules: Dict[str, TaskRule] = {}

    def add_rule(
        self,
        name: str,
        dependencies: Optional[List[str]] = None,
        executor: Optional[Callable] = None,
        output_pattern: Optional[str] = None,
        description: str = "",
    ) -> None:
        """
        Add a task rule to the planner.

        Args:
            name: Unique task identifier
            dependencies: List of task names this task depends on
            executor: Async function that executes the task (signature: async func(context, deps))
            output_pattern: Pattern for expected output file path (e.g., "cache/{url_hash}.chunks")
            description: Human-readable description
        """
        if dependencies is None:
            dependencies = []

        self.rules[name] = TaskRule(
            name=name,
            dependencies=dependencies,
            executor=executor,
            output_pattern=output_pattern,
            description=description,
        )

    def get_rule(self, name: str) -> TaskRule:
        """Get a task rule by name."""
        if name not in self.rules:
            available = ", ".join(self.rules.keys())
            raise ValueError(f"Unknown task '{name}'. Available: {available}")
        return self.rules[name]

    def list_rules(self) -> List[str]:
        """Get list of all defined task names."""
        return list(self.rules.keys())

    def validate_dependencies(self) -> None:
        """Validate that all dependencies reference existing tasks."""
        for rule in self.rules.values():
            for dep in rule.dependencies:
                if dep not in self.rules:
                    raise ValueError(
                        f"Task '{rule.name}' depends on unknown task '{dep}'"
                    )

    def create_task_list(
        self, target_tasks: List[str], context: Optional[Dict[str, Any]] = None
    ) -> "TaskList":
        """
        Planning stage: Create an optimized execution plan for the given targets.

        Args:
            target_tasks: List of task names to execute
            context: Shared context that will be passed to all executors

        Returns:
            TaskList containing the computed execution DAG
        """
        if context is None:
            context = {}

        # Validate targets exist
        for target in target_tasks:
            if target not in self.rules:
                available = ", ".join(self.rules.keys())
                raise ValueError(
                    f"Unknown target task '{target}'. Available: {available}"
                )

        # Validate all dependencies are satisfied
        self.validate_dependencies()

        # Build the complete dependency graph
        required_tasks = self._compute_required_tasks(target_tasks)
        nodes = self._create_task_nodes(required_tasks)
        execution_levels = self._compute_execution_levels(nodes)

        return TaskList(
            nodes=nodes,
            execution_levels=execution_levels,
            target_tasks=target_tasks,
            context=context,
            max_concurrent=self.max_concurrent,
        )

    def _compute_required_tasks(self, targets: List[str]) -> Set[str]:
        """Compute all tasks needed to satisfy the targets (including transitive deps)."""
        required = set()
        visited = set()

        def visit(task_name: str):
            if task_name in visited:
                return
            if task_name not in self.rules:
                raise ValueError(f"Unknown task: {task_name}")

            visited.add(task_name)
            required.add(task_name)

            # Visit all dependencies
            rule = self.rules[task_name]
            for dep in rule.dependencies:
                visit(dep)

        for target in targets:
            visit(target)

        return required

    def _create_task_nodes(self, task_names: Set[str]) -> Dict[str, TaskNode]:
        """Create TaskNode objects with dependency relationships."""
        nodes = {}

        # Create nodes
        for name in task_names:
            rule = self.rules[name]
            nodes[name] = TaskNode(rule=rule)

        # Set up dependency relationships
        for name, node in nodes.items():
            for dep_name in node.rule.dependencies:
                if dep_name in nodes:
                    node.add_dependency(nodes[dep_name])

        return nodes

    def _compute_execution_levels(self, nodes: Dict[str, TaskNode]) -> List[List[str]]:
        """
        Compute execution levels using topological sort.

        Returns:
            List of lists, where each inner list contains tasks that can run in parallel
        """
        # Calculate in-degrees (number of dependencies)
        remaining_nodes = set(nodes.values())
        levels = []

        while remaining_nodes:
            # Find all nodes with no remaining dependencies (among remaining nodes)
            ready_nodes = [
                node
                for node in remaining_nodes
                if all(dep not in remaining_nodes for dep in node.dependencies)
            ]

            if not ready_nodes:
                remaining_names = [node.name for node in remaining_nodes]
                raise ValueError(
                    f"Circular dependency detected among tasks: {remaining_names}"
                )

            ready_names = [node.name for node in ready_nodes]
            levels.append(ready_names)

            # Update node levels and remove completed nodes
            for node in ready_nodes:
                node.level = len(levels) - 1
                remaining_nodes.remove(node)

        return levels


class TaskList:
    """
    Execution stage: Contains a computed execution plan that can be run.

    This class represents the output of the planning stage and provides
    methods for actually executing the work.
    """

    def __init__(
        self,
        nodes: Dict[str, TaskNode],
        execution_levels: List[List[str]],
        target_tasks: List[str],
        context: Dict[str, Any],
        max_concurrent: int = 5,
    ):
        self.nodes = nodes
        self.execution_levels = execution_levels
        self.target_tasks = target_tasks
        self.context = context
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def get_node(self, name: str) -> TaskNode:
        """Get a task node by name."""
        if name not in self.nodes:
            available = ", ".join(self.nodes.keys())
            raise ValueError(f"Unknown task '{name}'. Available: {available}")
        return self.nodes[name]

    async def get_target_results(self) -> Dict[str, Any]:
        """Get results for the originally requested target tasks."""
        results = {}
        for name in self.target_tasks:
            node = self.nodes[name]
            if node.status == TaskStatus.COMPLETED:
                results[name] = await node.load_result()
        return results

    async def get_all_results(self) -> Dict[str, Any]:
        """Get results for all completed tasks."""
        results = {}
        for name, node in self.nodes.items():
            if node.status == TaskStatus.COMPLETED:
                results[name] = await node.load_result()
        return results

    def get_failed_tasks(self) -> Dict[str, Optional[str]]:
        """Get mapping of failed task names to error messages."""
        return {
            name: node.error
            for name, node in self.nodes.items()
            if node.status == TaskStatus.FAILED
        }

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution statistics."""
        status_counts = {}
        total_time = 0.0

        for node in self.nodes.values():
            status = node.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_time += node.execution_time

        return {
            "total_tasks": len(self.nodes),
            "status_counts": status_counts,
            "total_execution_time": total_time,
            "levels": len(self.execution_levels),
            "max_parallel_tasks": max(len(level) for level in self.execution_levels),
        }

    async def execute_serial(
        self, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute all tasks serially (one at a time).

        Args:
            progress_callback: Optional callback(task_name, status, completed, total)

        Returns:
            Results for target tasks
        """
        total_tasks = len(self.nodes)
        completed = 0

        # Execute level by level, but serially within each level
        for level in self.execution_levels:
            for task_name in level:
                node = self.get_node(task_name)

                # Skip if dependencies failed
                if node.has_failed_dependencies():
                    node.status = TaskStatus.FAILED
                    node.error = "Dependencies failed"
                    completed += 1
                    if progress_callback:
                        await self._safe_call(
                            progress_callback,
                            task_name,
                            "failed",
                            completed,
                            total_tasks,
                        )
                    continue

                # Check cache first
                if await self._check_cache(node):
                    completed += 1
                    if progress_callback:
                        await self._safe_call(
                            progress_callback,
                            task_name,
                            "cached",
                            completed,
                            total_tasks,
                        )
                    continue

                # Execute task
                await self._execute_single_task(node)
                completed += 1

                if progress_callback:
                    status = (
                        "completed" if node.status == TaskStatus.COMPLETED else "failed"
                    )
                    await self._safe_call(
                        progress_callback, task_name, status, completed, total_tasks
                    )

        return await self.get_target_results()

    async def execute_parallel(
        self, progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel where possible.

        Tasks in the same execution level can run concurrently.

        Args:
            progress_callback: Optional callback(task_name, status, completed, total)

        Returns:
            Results for target tasks
        """
        total_tasks = len(self.nodes)
        completed = 0

        # Execute level by level, with parallelism within each level
        for level in self.execution_levels:
            level_tasks = []

            # Check cache and dependencies for all tasks in this level
            for task_name in level:
                node = self.get_node(task_name)

                # Skip if dependencies failed
                if node.has_failed_dependencies():
                    node.status = TaskStatus.FAILED
                    node.error = "Dependencies failed"
                    completed += 1
                    if progress_callback:
                        await self._safe_call(
                            progress_callback,
                            task_name,
                            "failed",
                            completed,
                            total_tasks,
                        )
                    continue

                if await self._check_cache(node):
                    completed += 1
                    if progress_callback:
                        await self._safe_call(
                            progress_callback,
                            task_name,
                            "cached",
                            completed,
                            total_tasks,
                        )
                else:
                    level_tasks.append(task_name)

            if level_tasks:
                # Execute non-cached tasks in parallel
                tasks = [
                    self._execute_single_task(self.get_node(name))
                    for name in level_tasks
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Update progress
                for task_name in level_tasks:
                    completed += 1
                    if progress_callback:
                        node = self.get_node(task_name)
                        status = (
                            "completed"
                            if node.status == TaskStatus.COMPLETED
                            else "failed"
                        )
                        await self._safe_call(
                            progress_callback, task_name, status, completed, total_tasks
                        )

        return await self.get_target_results()

    async def _check_cache(self, node: TaskNode) -> bool:
        """Check if task result is cached by checking expected output file."""
        if node.rule.output_pattern is None:
            return False

        try:
            # Format the output pattern with context
            expected_path = Path(node.rule.output_pattern.format(**self.context))

            if expected_path.exists():
                node.status = TaskStatus.CACHED
                node.path = expected_path
                return True

        except Exception:
            pass  # Pattern formatting or file check failed

        return False

    async def _execute_single_task(self, node: TaskNode) -> None:
        """Execute a single task with concurrency control."""
        async with self._semaphore:
            start_time = time.time()
            node.status = TaskStatus.RUNNING

            try:
                # Prepare context with dependency paths
                task_context = self.context.copy()
                task_context["dependency_paths"] = node.get_dependency_paths()

                # Also provide loaded dependency results for backward compatibility
                dependency_results = {}
                for dep_name, dep_node in zip(
                    node.get_dependency_names(), node.dependencies
                ):
                    if dep_node.status in (TaskStatus.COMPLETED, TaskStatus.CACHED):
                        dependency_results[dep_name] = await dep_node.load_result()
                task_context["dependencies"] = dependency_results

                # Execute the task - should return file path
                if node.rule.executor:
                    result_path = await node.rule.executor(task_context)
                    if isinstance(result_path, (str, Path)):
                        node.path = Path(result_path)
                    else:
                        # Executor returned content directly - this is deprecated
                        node.path = None
                else:
                    raise ValueError(f"No executor defined for task '{node.name}'")

                node.status = TaskStatus.COMPLETED

            except Exception as e:
                node.status = TaskStatus.FAILED
                node.error = str(e)
                node.path = None

            finally:
                node.execution_time = time.time() - start_time

    async def _safe_call(self, func: Callable, *args, **kwargs) -> Any:
        """Safely call a function, handling both sync and async."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception:
            return None

    def visualize(self) -> str:
        """Generate a text visualization of the execution plan."""
        lines = ["Execution Plan:"]

        # Show execution levels
        for i, level in enumerate(self.execution_levels):
            level_names = []
            for name in level:
                node = self.nodes[name]
                status_symbol = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.FAILED: "❌",
                    TaskStatus.CACHED: "💾",
                }.get(node.status, "❓")
                level_names.append(f"{status_symbol} {name}")

            parallel_note = " (parallel)" if len(level) > 1 else ""
            lines.append(f"  Level {i + 1}{parallel_note}: {' | '.join(level_names)}")

        # Show dependencies
        lines.append("\nDependencies:")
        for name, node in self.nodes.items():
            if node.dependencies:
                deps = ", ".join(sorted(node.get_dependency_names()))
                lines.append(f"  {name} ← {deps}")
            else:
                lines.append(f"  {name} (no dependencies)")

        # Show targets
        lines.append(f"\nTargets: {', '.join(self.target_tasks)}")

        # Show any validation issues
        failed_nodes = [
            node for node in self.nodes.values() if node.status == TaskStatus.FAILED
        ]
        if failed_nodes:
            lines.append(
                f"\nFailed tasks: {', '.join(node.name for node in failed_nodes)}"
            )

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export task list state as dictionary for serialization."""
        return {
            "nodes": {
                name: {
                    "name": node.name,
                    "dependencies": list(node.get_dependency_names()),
                    "level": node.level,
                    "status": node.status.value,
                    "execution_time": node.execution_time,
                    "has_result": node.path is not None and node.path.exists()
                    if node.path
                    else False,
                    "result_path": str(node.path) if node.path else None,
                    "error": node.error,
                }
                for name, node in self.nodes.items()
            },
            "execution_levels": self.execution_levels,
            "target_tasks": self.target_tasks,
            "summary": self.get_execution_summary(),
        }


# =============================================================================
# Helper Functions and Utilities
# =============================================================================

# TaskPlanner is now fully generic - no content-type specific helpers needed


# =============================================================================
# Example Usage
# =============================================================================


async def example_usage():
    """Example of how to use the three-stage system."""

    # Stage 1: Configuration - Define the rules
    planner = TaskPlanner(max_concurrent=3)

    # Add rules for a content processing pipeline
    planner.add_rule(
        name="html",
        dependencies=[],
        executor=lambda ctx: f"cache/{ctx['url_hash']}.html",
        output_pattern="cache/{url_hash}.html",
        description="Fetch HTML content",
    )

    planner.add_rule(
        name="markdown",
        dependencies=["html"],
        executor=lambda ctx: f"cache/{ctx['url_hash']}.md",
        output_pattern="cache/{url_hash}.md",
        description="Convert HTML to markdown",
    )

    planner.add_rule(
        name="chunks",
        dependencies=["markdown"],
        executor=lambda ctx: f"cache/{ctx['url_hash']}.json",
        output_pattern="cache/{url_hash}.json",
        description="Split markdown into chunks",
    )

    # Stage 2: Planning - Create execution plan
    context = {"url": "https://example.com", "url_hash": "abc123"}
    task_list = planner.create_task_list(["chunks"], context)

    # Visualize the plan
    print(task_list.visualize())

    # Stage 3: Execution - Run the plan
    results = await task_list.execute_parallel(
        progress_callback=lambda name, status, completed, total: print(
            f"Progress: {name} {status} ({completed}/{total})"
        )
    )

    print("Results:", results)
    print("Summary:", task_list.get_execution_summary())


if __name__ == "__main__":
    asyncio.run(example_usage())
