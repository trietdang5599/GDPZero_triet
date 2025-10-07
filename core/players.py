"""
Compatibility wrapper for legacy imports.

The original `core.players` module grew large, so the concrete classes now live in
dedicated modules:
  * core.P4GSystemPlanner
  * core.PersuaderModel
  * core.PersuadeeModel

Existing code that imports from `core.players` will continue to work because this
module re-exports the public classes.
"""

from core.P4GSystemPlanner import DialogPlanner, P4GSystemPlanner, P4GChatSystemPlanner
from core.PersuaderModel import PersuaderModel, PersuaderChatModel
from core.PersuadeeModel import PersuadeeModel, PersuadeeChatModel


__all__ = [
	"DialogPlanner",
	"P4GSystemPlanner",
	"P4GChatSystemPlanner",
	"PersuaderModel",
	"PersuaderChatModel",
	"PersuadeeModel",
	"PersuadeeChatModel",
]

