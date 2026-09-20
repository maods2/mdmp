"""Guard the slim package root: helpers live on purpose submodules."""

import pytest


def test_root_does_not_export_mdm():
    with pytest.raises(ImportError):
        from mdmp import MDM  # noqa: F401


def test_root_does_not_export_plot_dag():
    with pytest.raises(ImportError):
        from mdmp import plot_dag  # noqa: F401
