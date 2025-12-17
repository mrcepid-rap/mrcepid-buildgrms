import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import pandas as pd
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor
from general_utilities.mrc_logger import MRCLogger

# Standardized Command Executor and Logger
CMD_EXECUTOR = build_default_command_executor()
LOGGER = MRCLogger().get_logger()


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, relatedness_file: dict) -> \
        Tuple[set, Path, Path, Optional[Path]]:
    """
    Downloads and prepares all necessary genetic and metadata files for the pipeline.

    :param genetic_data_file: A dictionary containing the genetic data file names and IDs.
    :param sample_ids_file: A dictionary representing the file containing valid sample IDs.
    :param ancestry_file: A dictionary representing the file containing ancestry assignments.
    :param relatedness_file: A dictionary representing the relatedness matrix table.
    :return: A tuple containing (set of genetic file stems, path to sample IDs, path to ancestry, path to relatedness).
    """
    # Ingest the genetic plink files
    genetic_data_file = InputFileHandler(genetic_data_file, download_now=True).get_file_handle()
    genetic_files = download_genetic_data(genetic_data_file)

    # Download metadata files
    sample_ids_file = InputFileHandler(sample_ids_file, download_now=True).get_file_handle()
    ancestry_file = InputFileHandler(ancestry_file, download_now=True).get_file_handle()

    # Download the relatedness file if provided as an input
    if relatedness_file is not None:
        relatedness_file = InputFileHandler(relatedness_file, download_now=True).get_file_handle()

    return genetic_files, sample_ids_file, ancestry_file, relatedness_file


def download_genetic_data(input_file_list: Path) -> set:
    """
    Parses an input list of file coordinates and downloads the associated genetic files.

    :param input_file_list: A file containing two columns: [filename, file_id/path].
    :return: A set of unique stems (prefixes) of the files downloaded for downstream PLINK processing.
    """
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()
    with open(input_file_list, 'r') as file:
        for line in file:
            if not line.strip():
                continue

            columns = line.strip().split()
            if len(columns) != 2:
                raise ValueError(f"Each line must have exactly two columns. Invalid line: {line.strip()}")

            filename, file_id = columns
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension in filename: {filename}")

            # Download the file via InputFileHandler
            InputFileHandler(file_id, download_now=True).get_file_handle()
            stems.add(Path(filename).stem)

    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor=CMD_EXECUTOR) -> str:
    """
    Merges multiple PLINK binary files into a single autosomal dataset.

    :param genetic_files: A set of file stems (prefixes) to be merged.
    :param cmd_executor: A command executor object for running commands on the docker instance.
    :return: The name of the merged file prefix.
    """
    # Skip merge if only one chromosome/file is present (e.g., All of Us array data)
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    output_stub = "Autosomes"

    # Create the merge list required by PLINK2
    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            merge_list.write(f"{base_name}\n")

    cmd = f"plink2 --pmerge-list merge_list.txt bfile --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)

    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    """
    Calculates sample relatedness from genetic data using the KING-table algorithm.

    :param genetic_data_file: The prefix path to the merged genetic data file.
    :param cmd_executor: A command executor object.
    :return: Path to the generated .kin0 relatedness file.
    """
    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    # Step 1: Calculate Principal Components for variant filtering
    if not Path(f"{genetic_data_file.name}.eigenvec.allele").exists():
        cmd = f"plink2 --bfile {genetic_data_file.name} --pca 3 allele-wts --out {genetic_data_file.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    # Step 2: Filter for weak SNPs based on PC loadings
    eigen_df = pd.read_csv(f"{genetic_data_file.name}.eigenvec.allele", sep='\t')
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()

    # Save variant whitelist
    pd.Series(weak_snps).to_csv(f"{genetic_data_file.name}_eigen_filtered.txt", index=False, header=False)

    # Step 3: Extract variants and calculate kinship matrix
    cmd = (
        f"plink2 --bfile {genetic_data_file.name} "
        f"--extract {genetic_data_file.name}_eigen_filtered.txt "
        f"--make-bed --out {genetic_data_file.name}_filtered_for_kinship"
    )
    cmd_executor.run_cmd_on_docker(cmd)

    cmd = f"plink2 --bfile {genetic_data_file.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    return Path(f"{relatedness_db}.kin0")


def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:
    """
    Helper function to iteratively identify related individuals for removal.

    :param rel: DataFrame containing related pairs.
    :param samples_to_exclude: List of individuals already flagged for removal.
    :return: Dictionary containing the updated relatedness DataFrame and counts of occurrences.
    """
    rel = rel[~rel['ID1'].isin(samples_to_exclude)]
    rel = rel[~rel['ID2'].isin(samples_to_exclude)]

    # Count how many times each individual appears in the relatedness list
    rel_ids = pd.DataFrame(data=pd.concat([rel['ID1'], rel['ID2']]), columns=['ID'])
    rel_ids['dummy'] = 1
    rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum')).sort_values(by='total')

    return {'rel': rel, 'rel_totals': rel_totals}


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    """
    Loads ancestry information and groups sample IDs by population.
    Supports both UKB and AoU header formats.

    :param ancestry_file: Path to the ancestry mapping file.
    :return: Dictionary mapping ancestry groups to sets of sample IDs.
    """
    ancestry_dict: Dict[str, Set[str]] = {'all': set()}
    ancestry_counts = {}

    with ancestry_file.open(mode='r') as ancestry_info:
        header = ancestry_info.readline().strip().split('\t')
        ancestry_info.seek(0)

        # Detect columns dynamically
        id_col = next((c for c in ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id'] if c in header), header[0])
        anc_col = next((c for c in ['POP', 'ancestry_pred', 'ancestry'] if c in header),
                       header[1] if len(header) > 1 else None)

        if not anc_col:
            raise ValueError("Ancestry file is missing a population data column.")

        LOGGER.info(f"Loading ancestry using ID: '{id_col}' and Population: '{anc_col}'")
        reader = csv.DictReader(ancestry_info, delimiter="\t")

        for indv in reader:
            eid = str(indv[id_col])
            pop = indv[anc_col]
            ancestry_dict['all'].add(eid)
            if pop and pop != "NA":
                ancestry_dict.setdefault(pop, set()).add(eid)
                ancestry_counts[pop] = ancestry_counts.get(pop, 0) + 1

    # Print summary of ancestry groups
    LOGGER.info("\n" + "=" * 40 + f"\n{'ANCESTRY GROUP':<25} | {'COUNT':<10}\n" + "-" * 40)
    for anc, count in sorted(ancestry_counts.items()):
        LOGGER.info(f"{anc:<25} | {count:<10}")
    LOGGER.info("-" * 40 + f"\n{'TOTAL SAMPLES':<25} | {len(ancestry_dict['all']):<10}\n" + "=" * 40 + "\n")

    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    """Loads a set of unique sample IDs from a file."""
    with sample_ids_file.open('r') as f:
        return {line.strip().split()[0] for line in f if line.strip()}


def _read_and_clean_relatedness(relatedness: Path) -> pd.DataFrame:
    """
    Helper to normalize relatedness file headers and format for cross-platform data (AoU/UKB).

    :param relatedness: Path to the raw relatedness matrix.
    :return: DataFrame with normalized columns ['ID1', 'ID2', 'Kinship'].
    """
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    rel = pd.read_csv(relatedness, delim_whitespace=True)

    # Normalize All of Us headers
    if 'i.s' in rel.columns and 'j.s' in rel.columns and 'kin' in rel.columns:
        rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})

    # Normalize standard PLINK/KING headers
    rel.columns = rel.columns.str.replace('^#', '', regex=True)
    rel = rel.rename(columns={'IID1': 'ID1', 'IID2': 'ID2', 'KINSHIP': 'Kinship'})

    # Robust index-based fallback if headers are completely non-standard
    if not {'ID1', 'ID2', 'Kinship'}.issubset(rel.columns):
        LOGGER.warning(f"Standard headers not found. Found: {rel.columns.tolist()}. Using index fallback.")
        rel = pd.read_csv(relatedness, delim_whitespace=True, header=None, skiprows=1)
        rel = rel.rename(columns={0: 'ID1', 1: 'ID2', rel.columns[-1]: 'Kinship'})

    rel['ID1'] = rel['ID1'].astype(str)
    rel['ID2'] = rel['ID2'].astype(str)

    return rel[['ID1', 'ID2', 'Kinship']]


def load_relatedness(relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
    """Reads the relatedness file and filters for valid samples."""
    rel = _read_and_clean_relatedness(relatedness)
    return rel[(rel["ID1"].isin(wes_samples)) & (rel["ID2"].isin(wes_samples))]


def get_relateds_to_remove(rel: pd.DataFrame) -> Set[str]:
    """Identifies the minimum set of individuals to remove to eliminate all related pairs."""
    relateds_to_remove = set()
    returned = select_related_individual(rel, [])
    rel, rel_totals = returned['rel'], returned['rel_totals']
    while len(rel_totals) > 0:
        samp_to_remove = rel_totals.iloc[-1].name
        relateds_to_remove.add(samp_to_remove)
        returned = select_related_individual(rel, [samp_to_remove])
        rel, rel_totals = returned['rel'], returned['rel_totals']
    return relateds_to_remove


def write_and_upload_ancestry_files(wes_samples: Set[str], ancestry_dict: Dict[str, Set[str]],
                                    relateds_to_remove: Set[str]) -> List:
    """
    Generates and uploads population-specific inclusion lists.
    Uses delete_on_upload=False to ensure files are available for the final pipeline export.
    """
    include_files = []
    exporter = ExportFileHandler(delete_on_upload=False)

    for ancestry in ancestry_dict:
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry]).difference(relateds_to_remove)

        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        with unrelated_path.open('w') as unrelated_f, related_path.open('w') as related_f:
            for samp in wes_samples:
                if samp in pass_samples:
                    unrelated_f.write(f"{samp}\n")
                if samp in ancestry_dict[ancestry]:
                    related_f.write(f"{samp}\n")

        include_files.append(exporter.export_files(unrelated_path.name))
        include_files.append(exporter.export_files(related_path.name))

    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[Set[str], List]:
    """Orchestrates ancestry and relatedness processing to generate sample inclusion lists."""
    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)
    rel = load_relatedness(relatedness, wes_samples)
    relateds_to_remove = get_relateds_to_remove(rel)
    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)
    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor=CMD_EXECUTOR) -> dict:
    """Calculates variant missingness rates."""
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"
    cmd = f"plink2 --missing 'variant-only' --bfile {merged_data_file.name} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    missingness = {}
    with open(f"{missingness_db}.vmiss", 'r') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for snp in reader:
            missingness[snp['ID']] = float(snp['F_MISS'])
    return missingness


def check_qc_ukb(wes_samples: set, missingness: dict, ukb_snp_qc: Path, ukb_snps_qc_v2: Path,
                 cmd_executor=CMD_EXECUTOR) -> Tuple[Path, Path]:
    """Performs UKBiobank-specific QC using provided batch and array quality metrics."""
    pass_snps_file = Path("pass_snps.txt")
    pass_samples = Path("pass_samples.txt")

    # SNP QC
    with open(ukb_snp_qc, 'r') as f_in, pass_snps_file.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ")
        array_names = [f"Batch_b{x:03d}_qc" for x in range(1, 96)] + [f"UKBiLEVEAX_b{x}_qc" for x in range(1, 12)]

        for snp in reader:
            if snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness[snp['rs_id']] < 0.05:
                if all(snp[arr] == "1" for arr in array_names):
                    f_out.write(snp['rs_id'] + "\n")

    # Sample QC
    ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")
    subprocess.run(f'paste -d " " {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}', shell=True)

    header = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1', 'affyID1', 'affyID2', 'array', 'batch2', 'plate',
              'well', 'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender', 'x.int', 'y.int', 'plate.sub',
              'well.sub', 'missing.rate', 'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship',
              'excl.kinship', 'excess.relatives', 'in.wba', 'used.pc']
    header.extend([f"PC{x}" for x in range(1, 41)])
    header.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])

    with open(ukb_sqc_v2_with_fam, 'r') as f_in, pass_samples.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ", fieldnames=header)
        for sample in reader:
            if sample['ID1'] in wes_samples and sample['het.missing.outliers'] == "0" and sample[
                'in.phasing.auto'] == "1":
                f_out.write(sample['ID1'] + "\n")

    return pass_snps_file, pass_samples


def check_qc_other(wes_samples: set, snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
    """Filters samples and SNPs for general datasets (treats sample_qc as blacklist, snp_qc as whitelist)."""
    output_snps = Path("pass_snps.txt")
    output_samples = Path("pass_samples.txt")

    with open(snp_qc_file, 'r') as f_in, output_snps.open('w') as f_out:
        for line in f_in:
            if line.strip(): f_out.write(line.strip().split()[0] + "\n")

    flagged_samples = set()
    try:
        with open(sample_qc_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                flagged_samples.add(row['s'] if 's' in row else list(row.values())[0])
    except Exception:
        with open(sample_qc_file, 'r') as f:
            for line in f: flagged_samples.add(line.strip().split()[0])

    final_samples = wes_samples - flagged_samples
    with output_samples.open('w') as f_out:
        for s in final_samples: f_out.write(f"{s}\n")

    return output_snps, output_samples


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None, cmd_executor=CMD_EXECUTOR) -> Tuple[
    Path, Path]:
    """
    Applies QC filters to PLINK files.
    Gracefully handles cases with 0 rare variants (common in array data).
    """
    merged_data_file = Path.cwd() / merged_filename
    snplist = Path(merged_data_file.name + ".low_MAC.snplist")

    # Main Filter
    cmd = f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed --extract {pass_snps.name} --keep-fam {pass_samples.name} --out {merged_data_file.name}"
    cmd_executor.run_cmd_on_docker(cmd)

    # Rare variant list (ignore error if none found)
    cmd = f"plink2 --bfile {merged_data_file.name} --max-mac 100 --write-snplist --out {merged_data_file.name}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)

    if not snplist.exists():
        LOGGER.warning("No low MAC variants found. Creating empty SNPLIST.")
        snplist.touch()

    return merged_data_file, snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    """Ensures matrix coordinates are ordered for the lower-left triangle."""
    return (col2, col1) if col1 < col2 else (col1, col2)


def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    """Generates a sparse GRM in MatrixMarket format from the relatedness matrix."""
    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')

    wes_samples_sorted = sorted(wes_samples)
    wes_samples_df = pd.DataFrame(data={'ID1': wes_samples_sorted, 'ID2': wes_samples_sorted, 'Kinship': 0.5})
    wes_samples_df['column1'] = wes_samples_df['column2'] = range(1, len(wes_samples_sorted) + 1)

    gt_matrix = _read_and_clean_relatedness(rel_mtx)
    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples) & gt_matrix['ID2'].isin(wes_samples)]

    gt_matrix = pd.merge(gt_matrix, wes_samples_df[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, wes_samples_df[['ID2', 'column2']], on='ID2', how="left")
    gt_matrix = pd.concat([gt_matrix, wes_samples_df])[['column1', 'column2', 'Kinship']]

    gt_matrix[['column1', 'column2']] = gt_matrix.apply(lambda row: column_swap(row['column1'], row['column2']), axis=1,
                                                        result_type='expand')
    gt_matrix = gt_matrix.sort_values(['column1', 'column2'])

    with open(grm, 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write(f'{len(wes_samples_sorted)} {len(wes_samples_sorted)} {len(gt_matrix)}\n')
        for row in gt_matrix.itertuples(index=False):
            matrix.write(f'{int(row.column1)} {int(row.column2)} {row.Kinship}\n')

    with open(grm_samples, 'w') as f:
        for s in wes_samples_sorted: f.write(f"{s}\n")

    return grm, grm_samples
