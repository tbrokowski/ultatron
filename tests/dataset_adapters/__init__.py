"""
tests.dataset_adapters
======================

Exploratory scripts for working with real ultrasound datasets on CSCS.

These modules are intended for **manual** execution on systems where the
full datasets are available (e.g. Capstor Store/Scratch). They are not
part of the automated pytest suite and may assume:

- Access to /capstor/store/cscs/swissai/a127/ultrasound
- BUSI / COVIDx-US and other datasets are already downloaded
- Optional GPU / heavy dependencies (torch, decord, cv2, Pillow)

Typical usage from the project root:

    python -m tests.dataset_adapters.busi_explore
    python -m tests.dataset_adapters.covidx_us_explore
"""

