import csv
import os
import subprocess
import gc
import shutil  # Added for disk usage checks
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union

import pandas as pd
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor
from general_utilities.mrc_logger import MRCLogger

# Initialize shared utilities
CMD_EXECUTOR = build_default_command_executor()
LOGGER = MRCLogger().get_logger()


def check_disk_usage(path: str = ".") -> None:
    """Debug helper: Prints disk usage statistics for the current volume."""
    total, used, free = shutil.disk_usage(path)
    LOGGER.info(
        f"DISK USAGE ({path}): Total: {total // (2 ** 30)}GB | Used: {used // (2 ** 30)}GB | Free: {free // (2 ** 30)}GB")
    # Also run df -h for good measure to see all mounts
    subprocess.run("df -h", shell=True)


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, relatedness_file: dict) -> \
        Tuple[set, Path, Path, Optional[Path]]:
    """Downloads and prepares all necessary genetic and metadata files."""
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

    check_disk_usage()
    return genetic_files, sample_ids_handle, ancestry_handle, relatedness_handle


def download_genetic_data(input_file_list: Path) -> Set[str]:
    """Parses a coordinate list file and downloads associated PLINK binaries."""
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()

    LOGGER.info(f"Parsing genetic coordinates from {input_file_list}")
    with open(input_file_list, 'r') as file:
        for line in file:
            if not line.strip(): continue
            columns = line.strip().split()
            if len(columns) != 2: raise ValueError(f"Invalid line format: {line.strip()}")
            filename, file_id = columns
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension: {filename}")

            InputFileHandler(file_id, download_now=True).get_file_handle()
            stems.add(Path(filename).stem)

    LOGGER.info(f"Downloaded {len(stems)} unique genetic file sets.")
    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor=CMD_EXECUTOR) -> str:
    """Merges multiple PLINK binary files into a single dataset."""
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    output_stub = "Autosomes"
    LOGGER.info(f"Merging {len(genetic_files)} PLINK files into '{output_stub}'...")
    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            merge_list.write(f"{base_name}\n")

    cmd = f"plink2 --pmerge-list merge_list.txt bfile --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)
    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    """Calculates the relatedness matrix using the KING-table algorithm."""
    genetic_data_path = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    LOGGER.info("Calculating relatedness matrix (KING)...")
    if not Path(f"{genetic_data_path.name}.eigenvec.allele").exists():
        cmd = f"plink2 --bfile {genetic_data_path.name} --pca 3 allele-wts --out {genetic_data_path.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    eigen_df = pd.read_csv(f"{genetic_data_path.name}.eigenvec.allele", sep='\t')
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()
    pd.Series(weak_snps).to_csv(f"{genetic_data_path.name}_eigen_filtered.txt", index=False, header=False)

    cmd = (f"plink2 --bfile {genetic_data_path.name} --extract {genetic_data_path.name}_eigen_filtered.txt "
           f"--make-bed --out {genetic_data_path.name}_filtered_for_kinship")
    cmd_executor.run_cmd_on_docker(cmd)

    cmd = f"plink2 --bfile {genetic_data_path.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    return Path(f"{relatedness_db}.kin0")


def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:
    """Helper to identify related individuals."""
    rel = rel[~rel['ID1'].isin(samples_to_exclude)]
    rel = rel[~rel['ID2'].isin(samples_to_exclude)]
    rel_ids = pd.DataFrame(data=pd.concat([rel['ID1'], rel['ID2']]), columns=['ID'])
    rel_ids['dummy'] = 1
    rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum')).sort_values(by='total')
    return {'rel': rel, 'rel_totals': rel_totals}


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    """Loads ancestry information."""
    ancestry_dict = {'all': set()}
    ancestry_counts = {}
    with ancestry_file.open(mode='r') as f:
        header = f.readline().strip().split('\t')
        f.seek(0)
        id_col = next((c for c in ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id'] if c in header), header[0])
        anc_col = next((c for c in ['POP', 'ancestry_pred', 'ancestry'] if c in header),
                       header[1] if len(header) > 1 else None)
        reader = csv.DictReader(f, delimiter="\t")
        for indv in reader:
            eid = str(indv[id_col])
            pop = indv[anc_col]
            ancestry_dict['all'].add(eid)
            if pop and pop != "NA":
                ancestry_dict.setdefault(pop, set()).add(eid)
                ancestry_counts[pop] = ancestry_counts.get(pop, 0) + 1

    LOGGER.info("Ancestry Data Loaded.")
    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    with sample_ids_file.open('r') as f:
        samps = {line.strip().split()[0] for line in f if line.strip()}
    return samps


def _read_and_clean_relatedness(relatedness: Path) -> pd.DataFrame:
    """Standardizes relatedness headers."""
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})
    rel = pd.read_csv(relatedness, delim_whitespace=True)
    if 'i.s' in rel.columns and 'j.s' in rel.columns and 'kin' in rel.columns:
        rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})
    rel.columns = rel.columns.str.replace('^#', '', regex=True)
    rel = rel.rename(columns={'IID1': 'ID1', 'IID2': 'ID2', 'KINSHIP': 'Kinship'})
    if not {'ID1', 'ID2', 'Kinship'}.issubset(rel.columns):
        LOGGER.warning("Standard headers not found. Using index fallback.")
        rel = pd.read_csv(relatedness, delim_whitespace=True, header=None, skiprows=1)
        rel = rel.rename(columns={0: 'ID1', 1: 'ID2', rel.columns[-1]: 'Kinship'})
    rel['ID1'] = rel['ID1'].astype(str)
    rel['ID2'] = rel['ID2'].astype(str)
    return rel[['ID1', 'ID2', 'Kinship']]


def load_relatedness(relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
    rel = _read_and_clean_relatedness(relatedness)
    return rel[(rel["ID1"].isin(wes_samples)) & (rel["ID2"].isin(wes_samples))]


def get_relateds_to_remove(rel: pd.DataFrame) -> Set[str]:
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
    include_files = []
    exporter = ExportFileHandler(delete_on_upload=False)
    for ancestry in ancestry_dict:
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry]).difference(relateds_to_remove)
        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')
        with unrelated_path.open('w') as u_f, related_path.open('w') as r_f:
            for samp in wes_samples:
                if samp in pass_samples: u_f.write(f"0 {samp}\n")
                if samp in ancestry_dict[ancestry]: r_f.write(f"0 {samp}\n")
        include_files.append(exporter.export_files(unrelated_path.name))
        include_files.append(exporter.export_files(related_path.name))
    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[Set[str], List]:
    """Orchestrator for sample filtering and inclusion list generation."""
    LOGGER.info("Processing ancestry and relatedness filtering...")
    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)
    rel = load_relatedness(relatedness, wes_samples)
    relateds_to_remove = get_relateds_to_remove(rel)
    LOGGER.info(f"Identified {len(relateds_to_remove)} related individuals to exclude.")
    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)

    # Force garbage collection to free GBs of RAM before PLINK starts
    LOGGER.info("Starting memory cleanup...")
    del rel
    del ancestry_dict
    del relateds_to_remove
    gc.collect()
    LOGGER.info("Memory cleanup complete.")

    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor=CMD_EXECUTOR) -> dict:
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"
    LOGGER.info("Calculating variant missingness...")
    cmd = f"plink2 --missing 'variant-only' --bfile {merged_data_file.name} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)
    missingness = {}
    with open(f"{missingness_db}.vmiss", 'r') as f:
        reader = csv.DictReader(f, delimiter="\t")
        for snp in reader: missingness[snp['ID']] = float(snp['F_MISS'])
    return missingness


def check_qc_ukb(wes_samples: set, missingness: dict, ukb_snp_qc: Path, ukb_snps_qc_v2: Path,
                 cmd_executor=CMD_EXECUTOR) -> Tuple[Path, Path]:
    pass_snps_file = Path("pass_snps.txt")
    pass_samples = Path("pass_samples.txt")
    with open(ukb_snp_qc, 'r') as f_in, pass_snps_file.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ")
        arrs = [f"Batch_b{x:03d}_qc" for x in range(1, 96)] + [f"UKBiLEVEAX_b{x}_qc" for x in range(1, 12)]
        for snp in reader:
            if snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness[snp['rs_id']] < 0.05:
                if all(snp[a] == "1" for a in arrs): f_out.write(snp['rs_id'] + "\n")
    ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")
    subprocess.run(f'paste -d " " {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}', shell=True)
    h = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1', 'affyID1', 'affyID2', 'array', 'batch2', 'plate',
         'well', 'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender', 'x.int', 'y.int', 'plate.sub', 'well.sub',
         'missing.rate', 'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship', 'excl.kinship',
         'excess.relatives', 'in.wba', 'used.pc']
    h.extend([f"PC{x}" for x in range(1, 41)])
    h.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])
    with open(ukb_sqc_v2_with_fam, 'r') as f_in, pass_samples.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ", fieldnames=h)
        for s in reader:
            if s['ID1'] in wes_samples and s['het.missing.outliers'] == "0" and s['in.phasing.auto'] == "1":
                f_out.write(f"0 {s['ID1']}\n")
    return pass_snps_file, pass_samples


def check_qc_other(wes_samples: set, snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
    LOGGER.info("Starting QC Check for non-UKB (AoU) data...")
    output_snps = Path("pass_snps.txt")
    output_samples = Path("pass_samples.txt")
    with open(snp_qc_file, 'r') as f_in, output_snps.open('w') as f_out:
        for line in f_in:
            if line.strip(): f_out.write(line.strip().split()[0] + "\n")
    flagged = set()
    try:
        with open(sample_qc_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for r in reader: flagged.add(r['s'] if 's' in r else list(r.values())[0])
    except Exception:
        with open(sample_qc_file, 'r') as f:
            for line in f: flagged.add(line.strip().split()[0])
    final_samples = wes_samples - flagged
    LOGGER.info(f"Retaining {len(final_samples)} samples after QC filtering (Removed {len(flagged)} flagged).")
    with output_samples.open('w') as f_out:
        for s in final_samples: f_out.write(f"0 {s}\n")
    return output_snps, output_samples


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None,
                 output_prefix: str = "Filtered_Data", cmd_executor=CMD_EXECUTOR) -> Tuple[Path, Path]:
    """Applies QC filters to PLINK files."""
    merged_data_file = Path.cwd() / merged_filename
    LOGGER.info(f"Filtering genotype data. Input: {merged_data_file.name}, Output Prefix: {output_prefix}")

    # Pre-flight Check
    check_disk_usage()
    sys.stdout.flush()  # Ensure logs are written before any potential crash

    snplist = Path(f"{output_prefix}.low_MAC.snplist")

    # Limit PLINK to 32GB approx
    cmd = (f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed "
           f"--extract {pass_snps.name} --keep-fam {pass_samples.name} --out {output_prefix} --memory 32000")
    cmd_executor.run_cmd_on_docker(cmd)

    cmd = f"plink2 --bfile {output_prefix} --max-mac 100 --write-snplist --out {output_prefix}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)

    if not snplist.exists():
        LOGGER.warning("No low MAC variants found. Creating empty SNPLIST.")
        snplist.touch()
    return Path(output_prefix), snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    return (col2, col1) if col1 < col2 else (col1, col2)


def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    LOGGER.info("Generating Sparse GRM matrix...")
    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')
    ws_sorted = sorted(wes_samples)
    ws_df = pd.DataFrame(data={'ID1': ws_sorted, 'ID2': ws_sorted, 'Kinship': 0.5})
    ws_df['column1'] = ws_df['column2'] = range(1, len(ws_sorted) + 1)

    gt_matrix = _read_and_clean_relatedness(rel_mtx)
    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples) & gt_matrix['ID2'].isin(wes_samples)]
    gt_matrix = pd.merge(gt_matrix, ws_df[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, ws_df[['ID2', 'column2']], on='ID2', how="left")
    gt_matrix = pd.concat([gt_matrix, ws_df])[['column1', 'column2', 'Kinship']]
    gt_matrix[['column1', 'column2']] = gt_matrix.apply(lambda row: column_swap(row['column1'], row['column2']), axis=1,
                                                        result_type='expand')
    gt_matrix = gt_matrix.sort_values(['column1', 'column2'])

    LOGGER.info(f"Writing GRM for {len(ws_sorted)} samples and {len(gt_matrix)} non-zero entries.")
    with open(grm, 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write(f'{len(ws_sorted)} {len(ws_sorted)} {len(gt_matrix)}\n')
        for row in gt_matrix.itertuples(index=False):
            matrix.write(f'{int(row.column1)} {int(row.column2)} {row.Kinship}\n')
    with open(grm_samples, 'w') as f:
        for s in ws_sorted: f.write(f"{s}\n")
    return grm, grm_samples