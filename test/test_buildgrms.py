"""
This file contains unit tests for the buildgrms module. It's recommended to run them sequentially to avoid
missing files.
"""

import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler

from buildgrms.grm_tools.grm_toolkit import ingest_resources, download_genetic_data, merge_plink_files, \
    calculate_relatedness, get_individuals, calculate_missingness, filter_plink, select_related_individual, \
    column_swap, check_qc_ukb, load_ancestry_dict, load_samples, load_relatedness, get_relateds_to_remove

test_data_dir = Path(__file__).parent / 'test_data'


@pytest.mark.parametrize(
    "col1, col2, expected",
    [
        ("A", "B", ("B", "A")),  # col1 < col2
        ("B", "A", ("B", "A")),  # col1 > col2
        ("A", "A", ("A", "A")),  # col1 == col2
    ]
)
def test_column_swap(col1: str, col2: str, expected: Tuple[str, str]):
    assert column_swap(col1, col2) == expected


@pytest.mark.parametrize(
    "genetic_data_file, sample_ids_file, ancestry_file, relatedness_file",
    [
        ("input_coords.txt", "sampleids.txt", "ancestry_file.txt", None)
    ]
)
def test_ingest_resources(genetic_data_file, sample_ids_file, ancestry_file, relatedness_file):
    """
    Test the ingest_resources function.
    """
    genetic_data_file = test_data_dir / genetic_data_file
    sample_ids_file = test_data_dir / sample_ids_file
    ancestry_file = test_data_dir / ancestry_file

    # Assuming ingest_resources is a function that takes these parameters
    genetic_files, sample_ids_file, ancestry_file, relatedness = ingest_resources(genetic_data_file, sample_ids_file,
                                                                                  ancestry_file,
                                                                                  relatedness_file)

    # ensure the genetic files contain data
    assert (Path(genetic_files).with_suffix(".bed")).stat().st_size > 0
    assert (Path(genetic_files).with_suffix(".bim")).stat().st_size > 0
    assert (Path(genetic_files).with_suffix(".fam")).stat().st_size > 0

    sample_ids = pd.read_csv(sample_ids_file, sep="\t", header=None)
    assert len(sample_ids) == 10000
    # ensure your sample ids range from 0000000 to 0009999
    assert sample_ids[0].min() == 0 and sample_ids[0].max() == 9999

    ancestry = pd.read_csv(ancestry_file, sep="\t")
    assert len(ancestry) == 10000
    # ensure your ancestry file has the expected column names
    assert list(ancestry.columns) == ['n_eid', 'ancestry']
    # ensure the first column ranges from 1 to 9999
    assert ancestry['n_eid'].min() == 0 and ancestry['n_eid'].max() == 9999
    # ensure the second column contains "eur"
    assert all(ancestry['ancestry'].str.contains("eur", case=False, na=False))


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
    assert (output_stub.with_suffix(".pgen")).exists() and (output_stub.with_suffix(".pgen")).stat().st_size > 0
    assert (output_stub.with_suffix(".psam")).exists() and (output_stub.with_suffix(".psam")).stat().st_size > 0
    assert (output_stub.with_suffix(".pvar")).exists() and (output_stub.with_suffix(".pvar")).stat().st_size > 0

    # assert length of pvar file
    pvar_file = output_stub.with_suffix(".pvar")
    pvar_df = pd.read_csv(pvar_file, sep="\t", header=None)
    assert len(pvar_df) == 678206, "The pvar file should not be empty."

    # assert the length of psam file
    psam_file = output_stub.with_suffix(".psam")
    psam_df = pd.read_csv(psam_file, sep="\t")
    assert len(psam_df) == 10000, "The psam file should contain 10000 samples."

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

    # read in the table
    relatedness_df = pd.read_csv(relatedness_out, sep="\t")
    assert len(relatedness_df) == 49995000, "The relatedness file should not be empty."
    # ensure column names are correct
    assert list(relatedness_df.columns) == ['ID1', 'ID2', 'HetHet', 'IBS0',
                                            'Kinship'], "The relatedness file should have the correct columns."


@pytest.mark.parametrize(
    "file_content, expected_output",
    [
        # Test case 1: Valid input with multiple ancestries
        (
                "n_eid\tancestry\n1\tEuropean\n2\tAsian\n3\tNA\n",
                {'all': {'1', '2', '3'}, 'European': {'1'}, 'Asian': {'2'}}
        ),
        # Test case 2: Empty file
        (
                "",
                {'all': set()}
        ),
        # Test case 3: File with only 'NA' ancestry (with failsafe)
        (
                "n_eid\tancestry\n1\tNA\n2\tNA\n",
                {'all': {'1', '2'}}
        ),
        # Test case 4: Missing 'ancestry' column
        (
                "n_eid\n1\n2\n",
                {'all': {'1', '2'}}
        ),
    ]
)
def test_load_ancestry_dict(file_content, expected_output, tmp_path):
    # Create a temporary file with the provided content
    temp_file = tmp_path / "ancestry_file.txt"
    temp_file.write_text(file_content)

    # Call the function and assert the output
    try:
        result = load_ancestry_dict(temp_file)
        assert result == expected_output
    except KeyError as e:
        # Failsafe: Handle missing 'ancestry' column
        assert "ancestry" in str(e)


@pytest.mark.parametrize(
    "file_content, expected_output",
    [
        # Test case 1: Valid input with multiple sample IDs
        (
                "sample1\nsample2\nsample3\n",
                {"sample1", "sample2", "sample3"}
        ),
        # Test case 2: Empty file
        (
                "",
                set()
        ),
        # Test case 3: File with extra whitespace
        (
                "sample1 \n sample2\n\nsample3\n",
                {"sample1", "sample2", "sample3"}
        ),
    ]
)
def test_load_samples(file_content, expected_output, tmp_path):
    # Create a temporary file with the provided content
    temp_file = tmp_path / "sample_ids.txt"
    temp_file.write_text(file_content)

    # Call the function and assert the output
    result = load_samples(temp_file)
    assert result == expected_output


@pytest.mark.parametrize(
    "relatedness_content, wes_samples, expected_output",
    [
        # Test case 1: Valid input with matching WES samples
        (
                "ID1 ID2 Kinship\nsample1 sample2 0.1\nsample3 sample4 0.2\n",
                {"sample1", "sample2", "sample3"},
                pd.DataFrame({"ID1": ["sample1"], "ID2": ["sample2"], "Kinship": [0.1]})
        ),
        # Test case 2: No matching WES samples
        (
                "ID1 ID2 Kinship\nsample1 sample2 0.1\nsample3 sample4 0.2\n",
                {"sample5", "sample6"},
                pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})
        ),
        # Test case 3: Empty relatedness file
        (
                "",
                {"sample1", "sample2"},
                pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})
        ),
    ]
)
def test_load_relatedness(relatedness_content, wes_samples, expected_output, tmp_path):
    # Create a temporary relatedness file
    relatedness_file = tmp_path / "relatedness.txt"
    relatedness_file.write_text(relatedness_content)

    # Call the function
    result = load_relatedness(relatedness_file, wes_samples)

    # Assert the output matches the expected DataFrame
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected_output.reset_index(drop=True))


@pytest.mark.parametrize(
    "rel_data, expected_output",
    [
        # Test case 1: Simple relatedness data
        (
                pd.DataFrame({
                    "ID1": ["sample1", "sample2", "sample3"],
                    "ID2": ["sample2", "sample3", "sample4"],
                    "Kinship": [0.1, 0.2, 0.3]
                }),
                {"sample3", "sample2"}
        ),
        # Test case 2: No related individuals
        (
                pd.DataFrame(columns=["ID1", "ID2", "Kinship"]),
                set()
        ),
        # Test case 3: Circular relatedness
        (
                pd.DataFrame({
                    "ID1": ["sample1", "sample2", "sample3"],
                    "ID2": ["sample2", "sample3", "sample1"],
                    "Kinship": [0.1, 0.2, 0.3]
                }),
                {"sample2", "sample3"}
        ),
    ]
)
def test_get_relateds_to_remove(rel_data, expected_output):
    result = get_relateds_to_remove(rel_data)
    assert result == expected_output


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

    assert len(include_files) == 4
    assert len(samples) == 10000


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

    assert len(missingness) == 678205


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

    assert len(final_genetic_file.with_suffix(".fam").read_text().splitlines()) == 10000
    assert len(final_genetic_file.with_suffix(".bim").read_text().splitlines()) == 17919


@pytest.mark.parametrize(
    "rel_data, samples_to_exclude, expected_rel_ids, expected_totals",
    [
        # Test case 1: No exclusions, simple relatedness
        (
                pd.DataFrame({
                    'ID1': ['A', 'B', 'C', 'A'],
                    'ID2': ['B', 'C', 'D', 'D']
                }),
                [],
                ['A', 'B', 'C', 'D'],
                {'A': 2, 'B': 2, 'C': 2, 'D': 2}
        ),
        # Test case 2: Exclude one individual
        (
                pd.DataFrame({
                    'ID1': ['A', 'B', 'C', 'A'],
                    'ID2': ['B', 'C', 'D', 'D']
                }),
                ['A'],
                ['B', 'C', 'D'],
                {'B': 1, 'C': 2, 'D': 1}
        ),
        # Test case 3: Empty relatedness file
        (
                pd.DataFrame(columns=['ID1', 'ID2']),
                [],
                [],
                {}
        ),
        # Test case 4: Exclude all individuals
        (
                pd.DataFrame({
                    'ID1': ['A', 'B', 'C'],
                    'ID2': ['B', 'C', 'D']
                }),
                ['A', 'B', 'C', 'D'],
                [],
                {}
        ),
    ]
)
def test_select_related_individual(rel_data, samples_to_exclude, expected_rel_ids, expected_totals):
    result = select_related_individual(rel_data, samples_to_exclude)
    rel, rel_totals = result['rel'], result['rel_totals']

    # Check the remaining relatedness DataFrame
    assert set(rel['ID1']).union(set(rel['ID2'])) == set(expected_rel_ids)

    # Check the totals
    assert rel_totals.to_dict()['total'] == expected_totals


@pytest.mark.parametrize(
    "wes_samples, missingness, ukb_snp_qc, ukb_snps_qc_v2, expected_snps_file, expected_samples_file",
    [
        (
                {"INCLUDEFOR_ALL_Unrelated.txt"},  # WES samples
                {"1:55545:C:T": 0.01, "1:546802:G:C": 0.02, "1:569933:G:A": 0.01},  # Missingness dictionary
                Path("test_data/ukb_snp_qc.txt"),  # SNP QC file
                Path("test_data/ukb_snp_qc_v2.txt"),  # SNP QC v2 file
                Path("pass_snps.txt"),  # Expected SNPs file
                Path("pass_samples.txt"),  # Expected samples file
        ),
    ],
)
def test_check_qc_ukb(wes_samples, missingness, ukb_snp_qc, ukb_snps_qc_v2, expected_snps_file, expected_samples_file,
                      tmp_path):
    # Arrange: Create temporary files for testing
    pass_snps_file, pass_samples_file = check_qc_ukb(wes_samples, missingness, ukb_snp_qc, ukb_snps_qc_v2)

    # Assert: Compare the output files with the expected files
    with open(pass_snps_file, "r") as actual_snps, open(expected_snps_file, "r") as expected_snps:
        assert actual_snps.read() == expected_snps.read()

    with open(pass_samples_file, "r") as actual_samples, open(expected_samples_file, "r") as expected_samples:
        assert actual_samples.read() == expected_samples.read()
