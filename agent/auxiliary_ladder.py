"""Synchronous and asynchronous drivers for auxiliary recovery ladders."""
from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from agent.auxiliary_client import _LadderStep


def _drive_ladder(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Run a ladder generator, feeding each step result or exception back in."""
    try:
        step = next(ladder)
        while True:
            try:
                result = perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value


async def _drive_ladder_async(ladder, perform: Callable[[_LadderStep], Any]) -> Any:
    """Async twin of _drive_ladder; perform is awaited."""
    try:
        step = next(ladder)
        while True:
            try:
                result = await perform(step)
            except Exception as exc:
                step = ladder.throw(exc)
            else:
                step = ladder.send(result)
    except StopIteration as stop:
        return stop.value
