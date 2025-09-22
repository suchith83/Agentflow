import secrets


class AIThreadNameGenerator:
    """
    Simple AI thread name generator that creates meaningful, varied names
    for AI conversations using different patterns and themes.
    """

    # Enhanced adjectives grouped by semantic meaning
    ADJECTIVES = [
        # Intellectual
        "thoughtful",
        "insightful",
        "analytical",
        "logical",
        "strategic",
        "methodical",
        "systematic",
        "comprehensive",
        "detailed",
        "precise",
        # Creative
        "creative",
        "imaginative",
        "innovative",
        "artistic",
        "expressive",
        "original",
        "inventive",
        "inspired",
        "visionary",
        "whimsical",
        # Emotional/Social
        "engaging",
        "collaborative",
        "meaningful",
        "productive",
        "harmonious",
        "enlightening",
        "empathetic",
        "supportive",
        "encouraging",
        "uplifting",
        # Dynamic
        "dynamic",
        "energetic",
        "vibrant",
        "lively",
        "spirited",
        "active",
        "flowing",
        "adaptive",
        "responsive",
        "interactive",
        # Quality-focused
        "focused",
        "dedicated",
        "thorough",
        "meticulous",
        "careful",
        "patient",
        "persistent",
        "resilient",
        "determined",
        "ambitious",
    ]

    # Enhanced nouns with more conversational context
    NOUNS = [
        # Conversation-related
        "dialogue",
        "conversation",
        "discussion",
        "exchange",
        "chat",
        "consultation",
        "session",
        "meeting",
        "interaction",
        "communication",
        # Journey/Process
        "journey",
        "exploration",
        "adventure",
        "quest",
        "voyage",
        "expedition",
        "discovery",
        "investigation",
        "research",
        "study",
        # Conceptual
        "insight",
        "vision",
        "perspective",
        "understanding",
        "wisdom",
        "knowledge",
        "learning",
        "growth",
        "development",
        "progress",
        # Solution-oriented
        "solution",
        "approach",
        "strategy",
        "method",
        "framework",
        "plan",
        "blueprint",
        "pathway",
        "route",
        "direction",
        # Creative/Abstract
        "canvas",
        "story",
        "narrative",
        "symphony",
        "composition",
        "creation",
        "masterpiece",
        "design",
        "pattern",
        "concept",
        # Collaborative
        "partnership",
        "collaboration",
        "alliance",
        "connection",
        "bond",
        "synergy",
        "harmony",
        "unity",
        "cooperation",
        "teamwork",
    ]

    # Action-based patterns for more dynamic names
    ACTION_PATTERNS = {
        "exploring": ["ideas", "concepts", "possibilities", "mysteries", "frontiers", "depths"],
        "building": ["solutions", "understanding", "connections", "frameworks", "bridges"],
        "discovering": ["insights", "patterns", "answers", "truths", "secrets", "wisdom"],
        "crafting": ["responses", "solutions", "stories", "strategies", "experiences"],
        "navigating": ["challenges", "questions", "complexities", "territories", "paths"],
        "unlocking": ["potential", "mysteries", "possibilities", "creativity", "knowledge"],
        "weaving": ["ideas", "stories", "connections", "patterns", "narratives"],
        "illuminating": ["concepts", "mysteries", "paths", "truths", "possibilities"],
    }

    # Descriptive compound patterns
    COMPOUND_PATTERNS = [
        ("deep", ["dive", "thought", "reflection", "analysis", "exploration"]),
        ("bright", ["spark", "idea", "insight", "moment", "flash"]),
        ("fresh", ["perspective", "approach", "start", "take", "view"]),
        ("open", ["dialogue", "discussion", "conversation", "exchange", "forum"]),
        ("creative", ["flow", "spark", "burst", "stream", "wave"]),
        ("mindful", ["moment", "pause", "reflection", "consideration", "thought"]),
        ("collaborative", ["effort", "venture", "journey", "exploration", "creation"]),
    ]

    def generate_simple_name(self, separator: str = "-") -> str:
        """
        Generate a simple adjective-noun combination.

        Returns:
            Names like "thoughtful-dialogue" or "creative-exploration"
        """
        adj = secrets.choice(self.ADJECTIVES)
        noun = secrets.choice(self.NOUNS)
        return f"{adj}{separator}{noun}"

    def generate_action_name(self, separator: str = "-") -> str:
        """
        Generate an action-based name for more dynamic feel.

        Returns:
            Names like "exploring-ideas" or "building-understanding"
        """
        action = secrets.choice(list(self.ACTION_PATTERNS.keys()))
        target = secrets.choice(self.ACTION_PATTERNS[action])
        return f"{action}{separator}{target}"

    def generate_compound_name(self, separator: str = "-") -> str:
        """
        Generate a compound descriptive name.

        Returns:
            Names like "deep-dive" or "bright-spark"
        """
        base, options = secrets.choice(self.COMPOUND_PATTERNS)
        complement = secrets.choice(options)
        return f"{base}{separator}{complement}"

    def generate_name(self, separator: str = "-") -> str:
        """
        Generate a meaningful thread name using random pattern selection.

        Args:
            separator: String to separate words (default: "-")

        Returns:
            A meaningful thread name from various patterns
        """
        # Randomly choose between different naming patterns
        pattern = secrets.choice(["simple", "action", "compound"])

        if pattern == "simple":
            return self.generate_simple_name(separator)
        if pattern == "action":
            return self.generate_action_name(separator)
        # compound
        return self.generate_compound_name(separator)


# Convenience function to maintain backward compatibility
def generate_dummy_thread_name(separator: str = "-") -> str:
    """
    Generate a meaningful English name for an AI chat thread.

    Args:
        separator: String to separate words (default: "-")

    Returns:
        A meaningful thread name like 'thoughtful-dialogue', 'exploring-ideas', or 'deep-dive'
    """
    generator = AIThreadNameGenerator()
    return generator.generate_name(separator)
