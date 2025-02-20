"""
Team module for collaborative agent-based tasks.
"""

from .team import Team
from .roles import Role, RoleRegistry
from .task import Task, TaskQueue
from .coordinator import TeamCoordinator

__all__ = ['Team', 'Role', 'RoleRegistry', 'Task', 'TaskQueue', 'TeamCoordinator']
