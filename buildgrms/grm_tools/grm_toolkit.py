import csv
import gc
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import pandas as pd
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor, CommandExecutor
from general_utilities.mrc_logger import MRCLogger
from buildgrms.grm_tools.relatedness_calculator import RelatednessCalculator

# Initialize shared utilities
CMD_EXECUTOR = build_default_command_executor()
LOGGER = MRCLogger().get_logger()


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, relatedness_file: dict) -> \
        Tuple[Set[str], Path, Path, Optional[Path]]:
    """Download and prepare all necessary genetic and metadata files.

    :param genetic_data_file: A DNAnexus file-like object (dict) for the genetic data coordinate list.
    :param sample_ids_file: A DNAnexus file-like object (dict) for the sample IDs file.
    :param ancestry_file: A DNAnexus file-like object (dict) for the ancestry file.
    :param relatedness_file: A DNAnexus file-like object (dict) for the relatedness file, or None.
    :return: A tuple containing:
             - A set of genetic file stems (e.g., {'arrays', 'ukb_c1'}).
             - Path to the sample IDs file.
             - Path to the ancestry file.
             - Path to the relatedness file (or None).
    """
    LOGGER.info("Starting resource ingestion...")

    genetic_data_handle = InputFileHandler(genetic_data_file, download_now=True).get_file_handle()
    genetic_files = download_genetic_data(genetic_data_handle)

    sample_ids_handle = InputFileHandler(sample_ids_file, download_now=True).get_file_handle()
    ancestry_handle = InputFileHandler(ancestry_file, download_now=True).get_file_handle()

    relatedness_handle = None
    if relatedness_file is not None:
        relatedness_handle = InputFileHandler(relatedness_file, download_now=True).get_file_handle()
        LOGGER.info(f"Relatedness file provided: {relatedness_handle.name}")
    else:
        LOGGER.info("No relatedness file provided; will calculate from scratch.")

    return genetic_files, sample_ids_handle, ancestry_handle, relatedness_handle


def download_genetic_data(input_file_list: Path) -> Set[str]:
    """Parse a coordinate list file and download associated PLINK binaries.

    :param input_file_list: Path to a text file containing [filename, path] columns.
    :return: A set of unique genetic file stems that were downloaded.
    """
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()

    LOGGER.info(f"Parsing genetic coordinates from {input_file_list}")
    with open(input_file_list, 'r') as file:
        for line in file:
            columns = line.strip().split()
            filename, file_id = columns

            # check that we are indeed working with PLINK files
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension: {filename}")

            InputFileHandler(file_id, download_now=True).get_file_handle()
            stems.add(Path(filename).stem)

    LOGGER.info(f"Downloaded {len(stems)} unique genetic file sets.")
    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor: CommandExecutor = CMD_EXECUTOR) -> str:
    """Merge multiple PLINK binary files into a single dataset.

    This method skips merging if only one file set is present.

    :param genetic_files: A set of PLINK file stems to merge.
    :param cmd_executor: An executor to run shell commands.
    :return: The file stem of the merged PLINK dataset.
    """
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        output_stub = single_file
    else:
        output_stub = "Autosomes"
        LOGGER.info(f"Merging {len(genetic_files)} PLINK files into '{output_stub}'...")

        with open('merge_list.txt', 'w') as merge_list:
            for base_name in sorted(genetic_files):
                merge_list.write(f"{base_name}\n")

        # force BED/BIM/FAM output
        cmd = f"plink2 --pmerge-list merge_list.txt bfile --make-bed --out {output_stub}"
        cmd_executor.run_cmd_on_docker(cmd)

    return output_stub


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    """Load ancestry information into a dictionary mapping Population to a Set of IDs.

    This method handles dynamic column detection for individual IDs and ancestry labels.

    :param ancestry_file: Path to the ancestry file.
    :return: A dictionary mapping population strings to sets of sample IDs.
    """
    ancestry_dict = {'all': set()}
    ancestry_counts = {}

    with ancestry_file.open(mode='r') as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames
        header_set = set(header)

        # Dynamic column detection
        id_candidates = ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id', 'FID']
        anc_candidates = ['POP', 'ancestry_pred', 'ancestry', 'predicted_ancestry']

        id_col = next((c for c in id_candidates if c in header_set), header[0])
        anc_col = next((c for c in anc_candidates if c in header_set), header[1] if len(header) > 1 else None)

        for indv in reader:
            eid = str(indv[id_col])
            pop = indv.get(anc_col)

            ancestry_dict['all'].add(eid)
            ancestry_dict.setdefault(pop, set()).add(eid)
            ancestry_counts[pop] = ancestry_counts.get(pop, 0) + 1

    LOGGER.info("Ancestry Data Loaded.")
    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    """Load valid sample IDs into a set.

    :param sample_ids_file: Path to the file containing sample IDs.
    :return: A set of sample IDs.
    """
    with sample_ids_file.open('r') as f:
        samps = {line.strip().split()[0] for line in f if line.strip()}
    return samps


def write_and_upload_ancestry_files(wes_samples: Set[str], ancestry_dict: Dict[str, Set[str]],
                                    relateds_to_remove: Set[str]) -> List[dict]:
    """Write and upload ancestry-specific sample inclusion lists.

    This function generates files of samples for each ancestry group, both with
    and without related individuals removed. It then uploads these files.
    It also prepends '0' to sample IDs for PLINK compatibility.

    :param wes_samples: Set of all samples.
    :param ancestry_dict: Dictionary mapping ancestry to a set of sample IDs.
    :param relateds_to_remove: Set of related sample IDs to exclude.
    :return: A list of DNAnexus file-like objects (dicts) for the uploaded files.
    """
    include_files = []
    exporter = ExportFileHandler(delete_on_upload=False)

    for ancestry in ancestry_dict:
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry]).difference(relateds_to_remove)

        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        with unrelated_path.open('w') as unrelated_file, related_path.open('w') as related_file:
            for samp in wes_samples:
                if samp in pass_samples:
                    unrelated_file.write(f"0 {samp}\n")  # FID 0
                if samp in ancestry_dict[ancestry]:
                    related_file.write(f"0 {samp}\n")  # FID 0

        include_files.append(exporter.export_files(unrelated_path.name))
        include_files.append(exporter.export_files(related_path.name))

    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[Set[str], List[dict]]:
    """Orchestrate sample filtering and inclusion list generation.

    This function loads sample IDs, ancestry information, and relatedness data,
    identifies related individuals to remove, and generates inclusion files for
    each ancestry group. Includes explicit memory management.

    :param sample_ids_file: Path to the sample IDs file.
    :param ancestry_file: Path to the ancestry file.
    :param relatedness: Path to the relatedness file.
    :return: A tuple containing:
             - A set of all WES sample IDs.
             - A list of DNAnexus file-like objects for the uploaded inclusion files.
    """
    LOGGER.info("Processing ancestry and relatedness filtering...")

    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)

    relatedness_calculator = RelatednessCalculator(next(iter(wes_samples)))
    relateds_to_remove = relatedness_calculator.get_relateds_to_remove(relatedness, wes_samples)
    LOGGER.info(f"Identified {len(relateds_to_remove)} related individuals to exclude.")

    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)

    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor: CommandExecutor = CMD_EXECUTOR) -> Dict[str, float]:
    """Calculate per-variant missingness statistics using PLINK2.

    :param merged_filename: The file stem of the merged PLINK dataset.
    :param cmd_executor: An executor to run shell commands.
    :return: A dictionary mapping variant ID to missingness fraction.
    """
    merged_data_file = Path.cwd() / merged_filename

    # Validation Check
    extensions = ['.bed', '.bim', '.fam']
    missing_files = [ext for ext in extensions if not Path(f"{merged_data_file}{ext}").exists()]

    if missing_files:
        raise FileNotFoundError(
            f"PLINK files missing for prefix '{merged_filename}': {missing_files}. "
            f"Current directory content: {list(Path.cwd().glob('*'))}"
        )

    missingness_db = "missingness_out"

    LOGGER.info(f"Calculating variant missingness for {merged_filename}...")
    cmd = f"plink2 --missing 'variant-only' --bfile {merged_filename} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    missingness = {}
    vmiss_path = Path(f"{missingness_db}.vmiss")
    if not vmiss_path.exists():
        raise RuntimeError(f"PLINK failed to generate {vmiss_path}")

    with open(vmiss_path, 'r') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for snp in reader:
            missingness[snp['ID']] = float(snp['F_MISS'])

    return missingness


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None,
                 output_prefix: str = "Filtered_Data", cmd_executor: CommandExecutor = CMD_EXECUTOR) -> Tuple[
    Path, Path]:
    """Apply QC filters to PLINK files.

    This function filters a PLINK dataset based on provided SNP and sample lists,
    and also generates a list of rare variants. Includes disk and memory safeguards.

    :param merged_filename: The file stem of the PLINK dataset to filter.
    :param pass_snps: Path to a file of SNPs to keep.
    :param pass_samples: Path to a file of samples to keep.
    :param output_prefix: Filename prefix for output files.
    :param cmd_executor: An executor to run shell commands.
    :return: A tuple containing:
             - Path object for the filtered PLINK data prefix.
             - Path to the list of low minor allele count (MAC) variants.
    """
    merged_data_file = Path.cwd() / merged_filename
    LOGGER.info(f"Filtering genotype data. Input: {merged_data_file.name}, Output Prefix: {output_prefix}")

    # Pre-flight Check: Ensure we have disk space before creating massive files
    sys.stdout.flush()

    snplist = Path(f"{output_prefix}.low_MAC.snplist")

    cmd = (f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed "
           f"--extract {pass_snps.name} --keep {pass_samples.name} "
           f"--out {output_prefix}")
    cmd_executor.run_cmd_on_docker(cmd)

    # Generate Rare Variant list
    cmd = f"plink2 --bfile {output_prefix} --max-mac 100 --write-snplist --out {output_prefix}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)

    if not snplist.exists():
        LOGGER.warning("No low MAC variants found. Creating empty SNPLIST.")
        snplist.touch()

    return Path(output_prefix), snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    """Ensure matrix coordinates are ordered for lower-left triangle storage.

    :param col1: First coordinate.
    :param col2: Second coordinate.
    :return: A tuple of the coordinates, ordered.
    """
    # Convert to float first, then int, to safely handle strings that might represent floats (e.g., '123.0')
    return (col2, col1) if int(float(col1)) < int(float(col2)) else (col1, col2)


def read_and_clean_relatedness(relatedness: Path) -> pd.DataFrame:
    """Read and standardize a relatedness matrix from a file.

    This function handles various header formats (UKB/PLINK) and ensures
    the output DataFrame has columns ['ID1', 'ID2', 'Kinship']. It uses a
    defined conditional path to parse files to avoid unexpected overwrites.

    :param relatedness: Path to the relatedness file.
    :return: A pandas DataFrame with standardized relatedness information.
    """
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    with relatedness.open('r') as f:
        header_line = f.readline().strip()

    header = header_line.split()
    header_set = set(header)

    # Use a clear conditional path to handle different file formats
    # Case 1: Raw format
    if {'i.s', 'j.s', 'kin'}.issubset(header_set):
        rel = pd.read_csv(relatedness, delim_whitespace=True)
        rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})
    # Case 2: Standard PLINK format (with possible '#' prefix)
    elif {'IID1', 'IID2', 'KINSHIP'}.issubset(header_set) or {'#IID1', 'IID2', 'KINSHIP'}.issubset(header_set):
        rel = pd.read_csv(relatedness, delim_whitespace=True)
        rel.columns = rel.columns.str.replace('^#', '', regex=True)
        rel = rel.rename(columns={'IID1': 'ID1', 'IID2': 'ID2', 'KINSHIP': 'Kinship'})
    # Case 3: Already in the desired format
    elif {'ID1', 'ID2', 'Kinship'}.issubset(header_set):
        rel = pd.read_csv(relatedness, delim_whitespace=True)
    # Case 4: Fallback for headerless or unknown format, taking the first two columns as IDs
    else:
        LOGGER.warning("Standard headers not found. Attempting to parse as headerless, skipping first row.")
        # Parse as headerless, treating the first two columns as IDs and replicating original skip-row behavior.
        rel = pd.read_csv(relatedness, delim_whitespace=True, header=None, skiprows=1)
        rel = rel.rename(columns={0: 'ID1', 1: 'ID2', rel.columns[-1]: 'Kinship'})
    rel['ID1'] = rel['ID1'].astype(str)
    rel['ID2'] = rel['ID2'].astype(str)
    return rel[['ID1', 'ID2', 'Kinship']]


def make_grm(wes_samples: Set[str], rel_mtx: Path) -> Tuple[Path, Path]:
    """Generate the sparse GRM in MatrixMarket format.

    :param wes_samples: A set of sample IDs to include in the GRM.
    :param rel_mtx: Path to the relatedness matrix file.
    :return: A tuple containing paths to the GRM file and the sample ID file.
    """
    LOGGER.info("Generating Sparse GRM matrix...")
    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')

    ws_sorted = sorted(list(wes_samples))
    ws_df = pd.DataFrame(data={'ID1': ws_sorted, 'ID2': ws_sorted, 'Kinship': 0.5})
    ws_df['column1'] = ws_df['column2'] = range(1, len(ws_sorted) + 1)

    gt_matrix = read_and_clean_relatedness(rel_mtx)
    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples) & gt_matrix['ID2'].isin(wes_samples)]

    gt_matrix = pd.merge(gt_matrix, ws_df[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, ws_df[['ID2', 'column2']], on='ID2', how="left")

    gt_matrix = pd.concat([gt_matrix, ws_df])[['column1', 'column2', 'Kinship']]
    gt_matrix.dropna(inplace=True)
    gt_matrix['column1'] = gt_matrix['column1'].astype(int)
    gt_matrix['column2'] = gt_matrix['column2'].astype(int)
    gt_matrix[['column1', 'column2']] = gt_matrix.apply(
        lambda row: column_swap(str(row['column1']), str(row['column2'])), axis=1, result_type='expand'
    )
    gt_matrix = gt_matrix.sort_values(['column1', 'column2'])

    LOGGER.info(f"Writing GRM for {len(ws_sorted)} samples and {len(gt_matrix)} non-zero entries.")

    with open(grm, 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write(f'{len(ws_sorted)} {len(ws_sorted)} {len(gt_matrix)}\n')
        for row in gt_matrix.itertuples(index=False):
            matrix.write(f'{int(float(row.column1))} {int(float(row.column2))} {row.Kinship}\n')

    with open(grm_samples, 'w') as f:
        for s in ws_sorted:
            f.write(f"{s}\n")

    return grm, grm_samples


def ld_prune_plink_fileset(input_prefix: str, cmd_executor: CommandExecutor = CMD_EXECUTOR):
    """Performs LD pruning on a given PLINK fileset, overwriting the input files.

    Uses a window size of 50kb, a step of 5 variants, and an r^2 threshold of 0.1.

    :param input_prefix: The prefix of the PLINK fileset to prune in-place.
    :param cmd_executor: An executor to run shell commands.
    """
    LOGGER.info(f"Performing in-place LD pruning on '{input_prefix}' with r2 threshold of 0.1...")

    prune_calc_prefix = f"{input_prefix}_prune_calc"
    pruned_snps_file = Path(f"{prune_calc_prefix}.prune.in")

    # Step 1: Calculate SNPs to keep after pruning
    cmd = (f"plink2 --bfile {input_prefix} "
           f"--indep-pairwise 50 5 0.1 "
           f"--out {prune_calc_prefix}")
    cmd_executor.run_cmd_on_docker(cmd)

    if not pruned_snps_file.exists():
        raise RuntimeError(f"LD pruning SNP list generation failed for {input_prefix}")

    # Step 2: Extract the pruned SNPs and overwrite the original fileset
    # PLINK automatically handles creating temporary files and renaming them, making this safe.
    cmd = (f"plink2 --bfile {input_prefix} "
           f"--extract {pruned_snps_file} "
           f"--make-bed --out {input_prefix}")
    cmd_executor.run_cmd_on_docker(cmd)

    LOGGER.info(f"In-place LD pruning complete for '{input_prefix}'.")
