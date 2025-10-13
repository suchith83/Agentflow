"""
Comprehensive tests for the thread_name_generator module.

This module tests the AIThreadNameGenerator class and the generate_dummy_thread_name
convenience function, ensuring they generate meaningful, varied thread names with
proper formats and patterns.
"""

import re
from unittest.mock import patch

import pytest

from taf.utils.thread_name_generator import (
    AIThreadNameGenerator,
    generate_dummy_thread_name,
)


class TestAIThreadNameGenerator:
    """Test the AIThreadNameGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = AIThreadNameGenerator()

    def test_initialization(self):
        """Test that the generator initializes correctly."""
        assert hasattr(self.generator, "ADJECTIVES")  # noqa: S101
        assert hasattr(self.generator, "NOUNS")  # noqa: S101
        assert hasattr(self.generator, "ACTION_PATTERNS")  # noqa: S101
        assert hasattr(self.generator, "COMPOUND_PATTERNS")  # noqa: S101
        
        # Verify lists are not empty
        assert len(self.generator.ADJECTIVES) > 0  # noqa: S101
        assert len(self.generator.NOUNS) > 0  # noqa: S101
        assert len(self.generator.ACTION_PATTERNS) > 0  # noqa: S101
        assert len(self.generator.COMPOUND_PATTERNS) > 0  # noqa: S101

    def test_generate_simple_name_format(self):
        """Test that generate_simple_name produces correct format."""
        name = self.generator.generate_simple_name()
        
        # Should contain exactly one separator
        assert name.count("-") == 1  # noqa: S101
        
        # Should have two parts
        parts = name.split("-")
        assert len(parts) == 2  # noqa: S101
        
        # Both parts should be non-empty strings
        assert all(len(part) > 0 for part in parts)  # noqa: S101
        assert all(part.isalpha() for part in parts)  # noqa: S101

    def test_generate_simple_name_custom_separator(self):
        """Test generate_simple_name with custom separator."""
        separators = ["_", ".", " ", "::"]
        
        for sep in separators:
            name = self.generator.generate_simple_name(separator=sep)
            assert sep in name  # noqa: S101
            parts = name.split(sep)
            assert len(parts) == 2  # noqa: S101

    def test_generate_simple_name_uses_valid_words(self):
        """Test that generate_simple_name uses words from the predefined lists."""
        name = self.generator.generate_simple_name()
        adj, noun = name.split("-")
        
        assert adj in self.generator.ADJECTIVES  # noqa: S101
        assert noun in self.generator.NOUNS  # noqa: S101

    def test_generate_action_name_format(self):
        """Test that generate_action_name produces correct format."""
        name = self.generator.generate_action_name()
        
        # Should contain exactly one separator
        assert name.count("-") == 1  # noqa: S101
        
        # Should have two parts
        parts = name.split("-")
        assert len(parts) == 2  # noqa: S101
        
        # Both parts should be non-empty strings
        assert all(len(part) > 0 for part in parts)  # noqa: S101

    def test_generate_action_name_uses_valid_patterns(self):
        """Test that generate_action_name uses valid action patterns."""
        name = self.generator.generate_action_name()
        action, target = name.split("-")
        
        # Action should be a key in ACTION_PATTERNS
        assert action in self.generator.ACTION_PATTERNS  # noqa: S101
        
        # Target should be in the corresponding list
        valid_targets = self.generator.ACTION_PATTERNS[action]
        assert target in valid_targets  # noqa: S101

    def test_generate_action_name_custom_separator(self):
        """Test generate_action_name with custom separator."""
        name = self.generator.generate_action_name(separator="_")
        assert "_" in name  # noqa: S101
        parts = name.split("_")
        assert len(parts) == 2  # noqa: S101

    def test_generate_compound_name_format(self):
        """Test that generate_compound_name produces correct format."""
        name = self.generator.generate_compound_name()
        
        # Should contain exactly one separator
        assert name.count("-") == 1  # noqa: S101
        
        # Should have two parts
        parts = name.split("-")
        assert len(parts) == 2  # noqa: S101
        
        # Both parts should be non-empty strings
        assert all(len(part) > 0 for part in parts)  # noqa: S101

    def test_generate_compound_name_uses_valid_patterns(self):
        """Test that generate_compound_name uses valid compound patterns."""
        name = self.generator.generate_compound_name()
        base, complement = name.split("-")
        
        # Find the matching pattern
        found_pattern = False
        for pattern_base, options in self.generator.COMPOUND_PATTERNS:
            if base == pattern_base and complement in options:
                found_pattern = True
                break
        
        assert found_pattern  # noqa: S101

    def test_generate_compound_name_custom_separator(self):
        """Test generate_compound_name with custom separator."""
        name = self.generator.generate_compound_name(separator=":")
        assert ":" in name  # noqa: S101
        parts = name.split(":")
        assert len(parts) == 2  # noqa: S101

    def test_generate_name_returns_valid_format(self):
        """Test that generate_name always returns valid format."""
        for _ in range(20):  # Generate multiple names to test randomness
            name = self.generator.generate_name()
            
            # Should contain exactly one separator
            assert name.count("-") == 1  # noqa: S101
            
            # Should have two parts
            parts = name.split("-")
            assert len(parts) == 2  # noqa: S101
            
            # Both parts should be non-empty strings
            assert all(len(part) > 0 for part in parts)  # noqa: S101

    def test_generate_name_custom_separator(self):
        """Test generate_name with custom separator."""
        name = self.generator.generate_name(separator="__")
        assert "__" in name  # noqa: S101
        parts = name.split("__")
        assert len(parts) == 2  # noqa: S101

    def test_generate_name_uses_different_patterns(self):
        """Test that generate_name uses different patterns over multiple calls."""
        names = [self.generator.generate_name() for _ in range(50)]
        
        # With 50 names, we should see some variation in patterns
        # This test verifies the randomization works
        assert len(set(names)) > 20  # Should have reasonable variety  # noqa: S101

    @patch("secrets.choice")
    def test_generate_name_pattern_selection(self, mock_choice):
        """Test that generate_name properly selects between patterns."""
        # Mock the pattern selection to always return "simple"
        mock_choice.side_effect = ["simple", "thoughtful", "dialogue"]
        
        name = self.generator.generate_name()
        
        # Should call secrets.choice at least once for pattern selection
        assert mock_choice.call_count >= 1  # noqa: S101
        assert isinstance(name, str)  # noqa: S101
        assert "-" in name  # noqa: S101

    def test_all_adjectives_are_strings(self):
        """Test that all adjectives are valid strings."""
        for adj in self.generator.ADJECTIVES:
            assert isinstance(adj, str)  # noqa: S101
            assert len(adj) > 0  # noqa: S101
            assert adj.isalpha()  # noqa: S101
            assert adj.islower()  # noqa: S101

    def test_all_nouns_are_strings(self):
        """Test that all nouns are valid strings."""
        for noun in self.generator.NOUNS:
            assert isinstance(noun, str)  # noqa: S101
            assert len(noun) > 0  # noqa: S101
            assert noun.replace("-", "").isalpha()  # Allow hyphens in compound nouns  # noqa: S101
            assert noun.islower()  # noqa: S101

    def test_action_patterns_structure(self):
        """Test that ACTION_PATTERNS has correct structure."""
        for action, targets in self.generator.ACTION_PATTERNS.items():
            assert isinstance(action, str)  # noqa: S101
            assert len(action) > 0  # noqa: S101
            assert isinstance(targets, list)  # noqa: S101
            assert len(targets) > 0  # noqa: S101
            
            for target in targets:
                assert isinstance(target, str)  # noqa: S101
                assert len(target) > 0  # noqa: S101

    def test_compound_patterns_structure(self):
        """Test that COMPOUND_PATTERNS has correct structure."""
        for pattern in self.generator.COMPOUND_PATTERNS:
            assert isinstance(pattern, tuple)  # noqa: S101
            assert len(pattern) == 2  # noqa: S101
            
            base, options = pattern
            assert isinstance(base, str)  # noqa: S101
            assert len(base) > 0  # noqa: S101
            assert isinstance(options, list)  # noqa: S101
            assert len(options) > 0  # noqa: S101
            
            for option in options:
                assert isinstance(option, str)  # noqa: S101
                assert len(option) > 0  # noqa: S101


class TestGenerateDummyThreadName:
    """Test the generate_dummy_thread_name convenience function."""

    def test_function_returns_string(self):
        """Test that the function returns a string."""
        name = generate_dummy_thread_name()
        assert isinstance(name, str)  # noqa: S101

    def test_function_default_separator(self):
        """Test that the function uses default separator."""
        name = generate_dummy_thread_name()
        assert "-" in name  # noqa: S101
        parts = name.split("-")
        assert len(parts) == 2  # noqa: S101

    def test_function_custom_separator(self):
        """Test that the function accepts custom separator."""
        separators = ["_", ".", " ", "::"]
        
        for sep in separators:
            name = generate_dummy_thread_name(separator=sep)
            assert sep in name  # noqa: S101
            parts = name.split(sep)
            assert len(parts) == 2  # noqa: S101

    def test_function_generates_varied_names(self):
        """Test that the function generates varied names."""
        names = [generate_dummy_thread_name() for _ in range(30)]
        
        # Should have good variety
        assert len(set(names)) > 15  # noqa: S101

    def test_function_name_format_consistency(self):
        """Test that all generated names follow consistent format."""
        for _ in range(20):
            name = generate_dummy_thread_name()
            
            # Should match pattern: word-word
            pattern = r"^[a-z]+(-[a-z]+)+$"
            assert re.match(pattern, name)  # noqa: S101

    def test_function_uses_generator_class(self):
        """Test that the function uses the AIThreadNameGenerator class."""
        # This is more of an integration test to ensure the function
        # properly delegates to the class
        name = generate_dummy_thread_name()
        
        # Create a generator and verify it can produce similar names
        generator = AIThreadNameGenerator()
        generator_name = generator.generate_name()
        
        # Both should follow the same pattern
        pattern = r"^[a-z]+(-[a-z]+)+$"
        assert re.match(pattern, name)  # noqa: S101
        assert re.match(pattern, generator_name)  # noqa: S101

    def test_function_empty_separator_edge_case(self):
        """Test function behavior with empty separator."""
        name = generate_dummy_thread_name(separator="")
        
        # Should still be a valid string with no separator
        assert isinstance(name, str)  # noqa: S101
        assert len(name) > 0  # noqa: S101
        # Should be concatenated words without separator
        assert name.isalpha()  # noqa: S101

    def test_function_long_separator(self):
        """Test function with unusually long separator."""
        long_sep = "---SEPARATOR---"
        name = generate_dummy_thread_name(separator=long_sep)
        
        assert long_sep in name  # noqa: S101
        parts = name.split(long_sep)
        assert len(parts) == 2  # noqa: S101
        assert all(len(part) > 0 for part in parts)  # noqa: S101

    def test_function_special_character_separator(self):
        """Test function with special character separators."""
        special_seps = ["@", "#", "$", "%", "&", "*"]
        
        for sep in special_seps:
            name = generate_dummy_thread_name(separator=sep)
            assert sep in name  # noqa: S101
            parts = name.split(sep)
            assert len(parts) == 2  # noqa: S101


class TestThreadNameGeneratorIntegration:
    """Integration tests for the thread name generator module."""

    def test_generator_and_function_consistency(self):
        """Test that the class and function produce consistent results."""
        generator = AIThreadNameGenerator()
        
        # Generate multiple names from both
        class_names = [generator.generate_name() for _ in range(10)]
        function_names = [generate_dummy_thread_name() for _ in range(10)]
        
        # All should follow the same pattern
        pattern = r"^[a-z]+(-[a-z]+)+$"
        
        for name in class_names:
            assert re.match(pattern, name)  # noqa: S101
            
        for name in function_names:
            assert re.match(pattern, name)  # noqa: S101

    def test_all_pattern_types_work(self):
        """Test that all pattern types can be generated successfully."""
        generator = AIThreadNameGenerator()
        
        # Test each pattern type individually
        simple_name = generator.generate_simple_name()
        action_name = generator.generate_action_name()
        compound_name = generator.generate_compound_name()
        
        # All should be valid
        for name in [simple_name, action_name, compound_name]:
            assert isinstance(name, str)  # noqa: S101
            assert len(name) > 0  # noqa: S101
            assert "-" in name  # noqa: S101
            parts = name.split("-")
            assert len(parts) == 2  # noqa: S101

    def test_thread_names_are_meaningful(self):
        """Test that generated names appear meaningful and readable."""
        # This is a more subjective test, but we can check for basic qualities
        names = [generate_dummy_thread_name() for _ in range(20)]
        
        for name in names:
            # Should not be too short (meaningful words should be longer)
            assert len(name) >= 7  # At least 3-char word + sep + 3-char word  # noqa: S101
            
            # Should not be excessively long
            assert len(name) <= 50  # Reasonable upper bound  # noqa: S101
            
            # Should only contain letters, hyphens, and no consecutive hyphens
            assert re.match(r"^[a-z]+([-][a-z]+)*$", name)  # noqa: S101

    def test_no_offensive_or_inappropriate_words(self):
        """Test that generated names don't contain inappropriate content."""
        # This is a basic sanity check - the word lists should be curated
        names = [generate_dummy_thread_name() for _ in range(50)]
        
        # Check that names don't contain common inappropriate patterns
        inappropriate_patterns = [
            r".*damn.*", r".*hell.*", r".*stupid.*", r".*idiot.*"
        ]
        
        for name in names:
            for pattern in inappropriate_patterns:
                assert not re.match(pattern, name, re.IGNORECASE)  # noqa: S101