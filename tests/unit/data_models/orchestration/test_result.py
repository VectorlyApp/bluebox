"""
tests/unit/data_models/orchestration/test_result.py

Tests for SpecialistResultWrapper.
"""

from bluebox.data_models.orchestration.result import SpecialistResultWrapper


class TestSpecialistResultWrapper:
    """Tests for SpecialistResultWrapper."""

    def test_defaults(self) -> None:
        wrapper = SpecialistResultWrapper()
        assert wrapper.output is None
        assert wrapper.success is True
        assert wrapper.notes == []
        assert wrapper.failure_reason is None

    def test_success_with_output(self) -> None:
        wrapper = SpecialistResultWrapper(
            output={"endpoints": ["/api/search"], "auth": "Bearer"},
            success=True,
            notes=["Found 3 endpoints, filtered to 1"],
        )
        assert wrapper.output["endpoints"] == ["/api/search"]
        assert len(wrapper.notes) == 1

    def test_failure(self) -> None:
        wrapper = SpecialistResultWrapper(
            success=False,
            failure_reason="Timeout after 60 seconds",
            notes=["Started processing", "Reached max iterations"],
        )
        assert wrapper.success is False
        assert "Timeout" in wrapper.failure_reason
        assert len(wrapper.notes) == 2

    def test_roundtrip(self) -> None:
        wrapper = SpecialistResultWrapper(
            output={"key": "value"},
            notes=["note1"],
        )
        data = wrapper.model_dump()
        wrapper2 = SpecialistResultWrapper.model_validate(data)
        assert wrapper2.output == wrapper.output
        assert wrapper2.notes == wrapper.notes
