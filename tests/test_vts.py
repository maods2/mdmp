"""
Tests for Virtual Typical Subject (VTS) module.
"""

import numpy as np
import pandas as pd
import pytest

from mdmp import MDM, compute_vts, validate_multi_subject_data, VTSResult
from mdmp.group_analysis.vts import align_subjects, prepare_multi_subject_data


@pytest.fixture
def multi_subject_data():
    """Variable-length multi-subject data: 3 subjects with 50, 60, 55 time points."""
    np.random.seed(42)
    return [
        np.random.randn(50, 4),
        np.random.randn(60, 4),
        np.random.randn(55, 4),
    ]


@pytest.fixture
def aligned_multi_subject():
    """Aligned multi-subject data: 3 subjects x 50 time points x 4 variables."""
    np.random.seed(42)
    return np.random.randn(3, 50, 4)


class TestValidateMultiSubjectData:
    """Tests for validate_multi_subject_data."""

    def test_list_of_arrays_valid(self, multi_subject_data):
        arrays, meta = validate_multi_subject_data(multi_subject_data)
        assert len(arrays) == 3
        assert meta["n_subjects"] == 3
        assert meta["subject_lengths"] == [50, 60, 55]
        assert len(meta["node_names"]) == 4

    def test_3d_array_valid(self, aligned_multi_subject):
        arrays, meta = validate_multi_subject_data(aligned_multi_subject)
        assert len(arrays) == 3
        assert meta["n_subjects"] == 3
        assert meta["subject_lengths"] == [50, 50, 50]
        assert all(a.shape == (50, 4) for a in arrays)

    def test_dataframe_valid(self):
        np.random.seed(42)
        rows = []
        for s in range(3):
            for t in range(50):
                rows.append({
                    "subject_id": s,
                    "V1": np.random.randn(),
                    "V2": np.random.randn(),
                    "V3": np.random.randn(),
                })
        df = pd.DataFrame(rows)
        arrays, meta = validate_multi_subject_data(df)
        assert len(arrays) == 3
        assert meta["node_names"] == ["V1", "V2", "V3"]

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            validate_multi_subject_data([])

    def test_3d_array_wrong_dims_raises(self):
        with pytest.raises(ValueError, match="3D array"):
            validate_multi_subject_data(np.random.randn(3, 50))

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="list of arrays"):
            validate_multi_subject_data("not valid")


class TestPrepareMultiSubjectData:
    """Tests for prepare_multi_subject_data."""

    def test_delegates_to_validate(self, multi_subject_data):
        arrays, meta = prepare_multi_subject_data(multi_subject_data)
        assert len(arrays) == 3
        assert meta["n_subjects"] == 3


class TestAlignSubjects:
    """Tests for align_subjects."""

    def test_truncate(self, multi_subject_data):
        aligned = align_subjects(multi_subject_data, method="truncate")
        assert all(a.shape[0] == 50 for a in aligned)
        assert all(a.shape[1] == 4 for a in aligned)

    def test_already_aligned(self, aligned_multi_subject):
        arrays = [aligned_multi_subject[i] for i in range(3)]
        aligned = align_subjects(arrays, method="truncate")
        assert len(aligned) == 3
        assert all(a.shape == (50, 4) for a in aligned)

    def test_interpolate(self, multi_subject_data):
        aligned = align_subjects(multi_subject_data, method="interpolate")
        assert all(a.shape[0] == 60 for a in aligned)
        assert all(a.shape[1] == 4 for a in aligned)

    def test_invalid_method_raises(self, multi_subject_data):
        with pytest.raises(ValueError, match="method must be"):
            align_subjects(multi_subject_data, method="invalid")


class TestComputeVTS:
    """Tests for compute_vts."""

    def test_mean_method(self, aligned_multi_subject):
        result = compute_vts(aligned_multi_subject, method="mean")
        assert isinstance(result, VTSResult)
        assert result.method == "mean"
        assert result.n_subjects == 3
        assert result.vts_data.shape == (50, 4)

    def test_concatenation_method(self, aligned_multi_subject):
        result = compute_vts(aligned_multi_subject, method="concatenation")
        assert result.method == "concatenation"
        assert result.vts_data.shape == (150, 4)

    def test_concatenation_return_series_false(self, aligned_multi_subject):
        result = compute_vts(
            aligned_multi_subject,
            method="concatenation",
            return_series=False,
        )
        assert result.vts_data.shape == (4,)

    def test_invalid_method_raises(self, aligned_multi_subject):
        with pytest.raises(ValueError, match="method must be"):
            compute_vts(aligned_multi_subject, method="invalid")


class TestVTSWithMDM:
    """Smoke test: VTS output as MDM input."""

    def test_mean_vts_fits_mdm(self, aligned_multi_subject):
        vts_result = compute_vts(aligned_multi_subject, method="mean")
        model = MDM(vts_result.vts_data, method="hc", verbose=False)
        assert model.adj_mat is not None
        assert model.data.shape == vts_result.vts_data.shape

    def test_concatenation_vts_fits_mdm(self, aligned_multi_subject):
        vts_result = compute_vts(aligned_multi_subject, method="concatenation")
        model = MDM(vts_result.vts_data, method="hc", verbose=False)
        assert model.adj_mat is not None
        assert model.data.shape == vts_result.vts_data.shape
