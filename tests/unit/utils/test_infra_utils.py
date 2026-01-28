"""
Tests for bluebox/utils/infra_utils.py

Tests for resolve_glob_patterns function.
"""

import tempfile
from pathlib import Path

import pytest

from bluebox.utils.infra_utils import resolve_glob_patterns


class TestResolveGlobPatterns:
    """Tests for resolve_glob_patterns function."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create directory structure:
            # tmpdir/
            #   docs/
            #     readme.md
            #     guide.md
            #     internal.md
            #   src/
            #     main.py
            #     utils.py
            #     tests/
            #       test_main.py
            #   data/
            #     config.json
            #     config.yaml

            (root / "docs").mkdir()
            (root / "docs" / "readme.md").write_text("# README")
            (root / "docs" / "guide.md").write_text("# Guide")
            (root / "docs" / "internal.md").write_text("# Internal")

            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("# main")
            (root / "src" / "utils.py").write_text("# utils")
            (root / "src" / "tests").mkdir()
            (root / "src" / "tests" / "test_main.py").write_text("# test")

            (root / "data").mkdir()
            (root / "data" / "config.json").write_text("{}")
            (root / "data" / "config.yaml").write_text("key: value")

            yield root

    def test_single_file(self, temp_dir: Path) -> None:
        """Test resolving a single file path."""
        file_path = str(temp_dir / "docs" / "readme.md")
        result = resolve_glob_patterns([file_path])

        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_single_file_with_extension_filter(self, temp_dir: Path) -> None:
        """Test single file with extension filter."""
        file_path = str(temp_dir / "docs" / "readme.md")

        # Should match
        result = resolve_glob_patterns([file_path], extensions={".md"})
        assert len(result) == 1

        # Should not match
        result = resolve_glob_patterns([file_path], extensions={".py"})
        assert len(result) == 0

    def test_directory_non_recursive(self, temp_dir: Path) -> None:
        """Test resolving a directory non-recursively."""
        dir_path = str(temp_dir / "docs")
        result = resolve_glob_patterns([dir_path], recursive=False)

        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"readme.md", "guide.md", "internal.md"}

    def test_directory_recursive(self, temp_dir: Path) -> None:
        """Test resolving a directory recursively."""
        dir_path = str(temp_dir / "src")
        result = resolve_glob_patterns([dir_path], extensions={".py"}, recursive=True)

        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"main.py", "utils.py", "test_main.py"}

    def test_glob_pattern_star(self, temp_dir: Path) -> None:
        """Test glob pattern with single star."""
        pattern = str(temp_dir / "docs" / "*.md")
        result = resolve_glob_patterns([pattern])

        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"readme.md", "guide.md", "internal.md"}

    def test_glob_pattern_double_star(self, temp_dir: Path) -> None:
        """Test glob pattern with double star (recursive)."""
        pattern = str(temp_dir / "src" / "**" / "*.py")
        result = resolve_glob_patterns([pattern])

        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"main.py", "utils.py", "test_main.py"}

    def test_exclusion_single_file(self, temp_dir: Path) -> None:
        """Test excluding a single file."""
        dir_path = str(temp_dir / "docs")
        exclude_path = "!" + str(temp_dir / "docs" / "internal.md")

        result = resolve_glob_patterns([dir_path, exclude_path], recursive=False)

        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"readme.md", "guide.md"}

    def test_exclusion_directory(self, temp_dir: Path) -> None:
        """Test excluding an entire directory."""
        dir_path = str(temp_dir / "src")
        exclude_path = "!" + str(temp_dir / "src" / "tests")

        result = resolve_glob_patterns(
            [dir_path, exclude_path],
            extensions={".py"},
            recursive=True
        )

        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"main.py", "utils.py"}

    def test_exclusion_glob_pattern(self, temp_dir: Path) -> None:
        """Test excluding with a glob pattern."""
        dir_path = str(temp_dir / "src")
        exclude_pattern = "!" + str(temp_dir / "src" / "**" / "test_*.py")

        result = resolve_glob_patterns(
            [dir_path, exclude_pattern],
            extensions={".py"},
            recursive=True
        )

        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"main.py", "utils.py"}

    def test_multiple_patterns(self, temp_dir: Path) -> None:
        """Test multiple include patterns."""
        patterns = [
            str(temp_dir / "docs" / "readme.md"),
            str(temp_dir / "src" / "main.py"),
        ]
        result = resolve_glob_patterns(patterns)

        assert len(result) == 2
        names = {f.name for f in result}
        assert names == {"readme.md", "main.py"}

    def test_mixed_files_and_dirs(self, temp_dir: Path) -> None:
        """Test mixing single files and directories."""
        patterns = [
            str(temp_dir / "docs" / "readme.md"),  # single file
            str(temp_dir / "data"),  # directory
        ]
        result = resolve_glob_patterns(patterns, recursive=False)

        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"readme.md", "config.json", "config.yaml"}

    def test_nonexistent_path_silent(self, temp_dir: Path) -> None:
        """Test that non-existent paths are silently skipped by default."""
        patterns = [
            str(temp_dir / "docs" / "readme.md"),
            str(temp_dir / "nonexistent" / "file.md"),
        ]
        result = resolve_glob_patterns(patterns, raise_on_missing=False)

        assert len(result) == 1
        assert result[0].name == "readme.md"

    def test_nonexistent_path_raises(self, temp_dir: Path) -> None:
        """Test that non-existent paths raise ValueError when requested."""
        patterns = [str(temp_dir / "nonexistent" / "file.md")]

        with pytest.raises(ValueError, match="Path does not exist"):
            resolve_glob_patterns(patterns, raise_on_missing=True)

    def test_empty_patterns(self) -> None:
        """Test empty patterns list."""
        result = resolve_glob_patterns([])
        assert result == []

    def test_results_are_sorted(self, temp_dir: Path) -> None:
        """Test that results are sorted."""
        dir_path = str(temp_dir / "docs")
        result = resolve_glob_patterns([dir_path], recursive=False)

        # Results should be sorted
        assert result == sorted(result)

    def test_no_duplicates(self, temp_dir: Path) -> None:
        """Test that duplicate files are not included."""
        patterns = [
            str(temp_dir / "docs"),
            str(temp_dir / "docs" / "readme.md"),  # already included via dir
        ]
        result = resolve_glob_patterns(patterns, recursive=False)

        # Should have 3 files, not 4
        assert len(result) == 3
        names = {f.name for f in result}
        assert names == {"readme.md", "guide.md", "internal.md"}

    def test_extension_filter_case_insensitive(self, temp_dir: Path) -> None:
        """Test that extension filtering is case-insensitive."""
        # Create a file with uppercase extension
        (temp_dir / "docs" / "UPPER.MD").write_text("# UPPER")

        dir_path = str(temp_dir / "docs")
        result = resolve_glob_patterns([dir_path], extensions={".md"}, recursive=False)

        names = {f.name for f in result}
        assert "UPPER.MD" in names
