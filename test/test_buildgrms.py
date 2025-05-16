"""
This file contains unit tests for the buildgrms module. It's recommended to run them sequentially to avoid
missing files.
"""

import time
from pathlib import Path

import pytest
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler

from buildgrms.grm_tools.grm_toolkit import ingest_resources, download_genetic_data, merge_plink_files, \
    calculate_relatedness, get_individuals, calculate_missingness, filter_plink, make_grm

test_data_dir = Path(__file__).parent / 'test_data'


@pytest.mark.parametrize(
    "genetic_data_file, sample_ids_file, ancestry_file",
    [
        ("input_coords.txt", "sampleids.txt", "ancestry_file.txt")
    ]
)
def test_ingest_resources(genetic_data_file, sample_ids_file, ancestry_file):
    """
    Test the ingest_resources function.
    """
    genetic_data_file = test_data_dir / genetic_data_file
    sample_ids_file = test_data_dir / sample_ids_file
    ancestry_file = test_data_dir / ancestry_file

    # Assuming ingest_resources is a function that takes these parameters
    genetic_files, sample_ids_file, ancestry_file = ingest_resources(genetic_data_file, sample_ids_file, ancestry_file)

    assert genetic_files is not None
    assert sample_ids_file is not None
    assert ancestry_file is not None


@pytest.mark.parametrize(
    "genetic_data_file",
    [
        "input_coords.txt",
    ]
)
def test_download_genetic_data(genetic_data_file):
    """
    Test the download_genetic_data function.
    """
    genetic_data_file = test_data_dir / genetic_data_file
    download_genetic_data(genetic_data_file)


@pytest.mark.parametrize(
    "genetic_files",
    [
        "sim_chromosome"
    ]
)
def test_merge_plink_files(genetic_files):
    """
    Test the merge_plink_files function.
    """

    genetic_files = test_data_dir / genetic_files
    output_stub = merge_plink_files(genetic_files)

    output_stub = Path(output_stub)  # Convert it to a Path object
    assert (output_stub.with_suffix(".pgen")).exists()
    assert (output_stub.with_suffix(".psam")).exists()
    assert (output_stub.with_suffix(".pvar")).exists()

    time.sleep(3)


@pytest.mark.parametrize(
    "merged_file",
    [
        "Autosomes"
    ]
)
def test_calculate_relatedness(merged_file):
    """
    Test the calculate_relatedness function.
    """

    relatedness_out = calculate_relatedness(merged_file)

    assert relatedness_out is not None
    assert relatedness_out.exists()
    assert relatedness_out.is_file()


@pytest.mark.parametrize(
    ["sample_ids_file", "ancestry_file", "relatedness_file"],
    [
        ["sampleids.txt", "ancestry_file.txt", "relatedness_table_processed.kin0"]
    ]
)
def test_get_individuals(sample_ids_file, ancestry_file, relatedness_file):
    sample_ids_file = test_data_dir / sample_ids_file
    ancestry_file = test_data_dir / ancestry_file
    relatedness_file = test_data_dir / relatedness_file

    # Assuming get_individuals is a function that takes these parameters
    samples, include_files = get_individuals(sample_ids_file, ancestry_file, relatedness_file)

    assert samples is not None
    assert include_files is not None


@pytest.mark.parametrize(
    "merged_filename",
    [
        "Autosomes"
    ]
)
def test_calculate_missingness(merged_filename):
    """
    Test the calculate_missingness function.
    """
    merged_filename = test_data_dir / merged_filename

    # Assuming calculate_missingness is a function that takes this parameter
    missingness = calculate_missingness(merged_filename)

    assert missingness is not None


@pytest.mark.parametrize(
    ["filepath", "snp_qc", "sample_qc"],
    [
        ("Autosomes", "output_variants.snplist", "sampleids.txt")
    ]
)
def test_filter_plink(filepath, snp_qc, sample_qc):
    """
    Test the filter_plink function.
    """

    snp_qc = test_data_dir / snp_qc
    sample_qc = test_data_dir / sample_qc

    snp_qc = InputFileHandler(snp_qc).get_file_handle()
    sample_qc = InputFileHandler(sample_qc).get_file_handle()

    # Assuming filter_plink is a function that takes these parameters
    final_genetic_file, snplist = filter_plink(filepath, snp_qc, sample_qc)

    assert final_genetic_file is not None
    assert (final_genetic_file.with_suffix(".bed")).exists()
    assert (final_genetic_file.with_suffix(".bim")).exists()
    assert (final_genetic_file.with_suffix(".fam")).exists()


@pytest.mark.parametrize(
    ["sample_ids_file", "ancestry_file", "relatedness_file"],
    [
        ["sampleids.txt", "ancestry_file.txt", "relatedness_table_processed.kin0"]
    ]
)
def test_make_grm(sample_ids_file, ancestry_file, relatedness_file):
    """
    Test the make_grm function.
    """

    sample_ids_file = test_data_dir / sample_ids_file
    ancestry_file = test_data_dir / ancestry_file
    relatedness_file = test_data_dir / relatedness_file

    samples, include_files = get_individuals(sample_ids_file, ancestry_file, relatedness_file)

    # Assuming make_GRM is a function that takes these parameters
    grm, grm_sample = make_grm(samples, relatedness_file)

    assert grm is not None
    assert grm_sample is not None

    assert grm.exists()
    assert grm_sample.exists()
