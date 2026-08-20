"""Rule-based router: picks a specialist for a given input query."""

import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from specialists.registry import SpecialistRegistry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

_TAMIL_SCRIPT_PATTERN = re.compile(r"[\u0B80-\u0BFF]")

_CODE_SIGNALS = re.compile(
    r"\b(def|class|import|function|const|let|var|return|for\s*\(|while\s*\()\b"
    r"|[{};]|=>|->|::"
)

_MATH_SCIENCE_SIGNALS = re.compile(
    r"\b(equation|integral|derivative|theorem|hypothesis|molecule|"
    r"velocity|acceleration|calculate|solve for|physics|chemistry)\b",
    re.IGNORECASE,
)

_GAMEDEV_SIGNALS = re.compile(
    r"\b(game engine|sprite|collision|shader|unity|godot|pixel art|"
    r"game loop|frame rate|hitbox)\b",
    re.IGNORECASE,
)


@dataclass
class RoutingRule:
    specialist_name: str
    matcher: Callable[[str], bool]
    description: str = ""


class Router:
    def __init__(self, registry: SpecialistRegistry, default_specialist: str = "general", rules=None):
        self.registry = registry
        self.default_specialist = default_specialist
        self.rules = rules if rules is not None else self._default_rules()

    def _default_rules(self) -> List[RoutingRule]:
        return [
            RoutingRule("tamil", lambda text: bool(_TAMIL_SCRIPT_PATTERN.search(text)), "Contains Tamil script characters"),
            RoutingRule("coding", lambda text: bool(_CODE_SIGNALS.search(text)), "Contains code-like keywords/symbols"),
            RoutingRule("math_science", lambda text: bool(_MATH_SCIENCE_SIGNALS.search(text)), "Contains math/science terminology"),
            RoutingRule("gamedev", lambda text: bool(_GAMEDEV_SIGNALS.search(text)), "Contains game-development terminology"),
        ]

    def route(self, query: str) -> str:
        registered = self.registry.list_specialists()

        for rule in self.rules:
            if rule.matcher(query):
                if rule.specialist_name in registered:
                    logger.info("Routed to '%s' (matched rule: %s)", rule.specialist_name, rule.description)
                    return rule.specialist_name
                else:
                    logger.warning(
                        "Rule matched '%s' (%s) but it isn't registered yet -- falling through.",
                        rule.specialist_name, rule.description,
                    )
                    continue

        if self.default_specialist in registered:
            logger.info("No rule matched -- using default specialist '%s'", self.default_specialist)
            return self.default_specialist

        if registered:
            fallback = next(iter(registered))
            logger.warning(
                "Default specialist '%s' not registered. Falling back to '%s' (first available).",
                self.default_specialist, fallback,
            )
            return fallback

        raise RuntimeError("No specialists are registered at all -- nothing to route to.")
