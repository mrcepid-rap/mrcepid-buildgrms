import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import dxpy
import numpy as np
import pandas as pd
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor, CommandExecutor
from general_utilities.mrc_logger import MRCLogger

CMD_EXECUTOR = build_default_command_executor()
LOGGER = MRCLogger().get_logger()


def run_debug_cmd(desc: str, cmd: str):
    """Helper to run a shell command and print output nicely for debugging."""
    print(f"\n--- DEBUG [{desc}] ---")
    print(f"CMD: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=False)
    except Exception as e:
        print(f"Debug command failed: {e}")
    print("----------------------\n")


def check_disk_usage(path: str = ".") -> None:
    """Print disk usage statistics."""
    total, used, free = shutil.disk_usage(path)
    LOGGER.info(
        f"DISK USAGE ({path}): "
        f"Total: {total // (2 ** 30)}GB | "
        f"Used: {used // (2 ** 30)}GB | "
        f"Free: {free // (2 ** 30)}GB"
    )
    subprocess.run("df -h", shell=True)


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, relatedness_file: dict) -> \
        Tuple[Set[str], Path, Path, Optional[Path]]:
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

    # DEBUG: Check downloaded files
    run_debug_cmd("Ingested Files", "ls -lh")

    return genetic_files, sample_ids_handle, ancestry_handle, relatedness_handle


def download_genetic_data(input_file_list: Path) -> Set[str]:
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()

    LOGGER.info(f"Parsing genetic coordinates from {input_file_list}")
    with open(input_file_list, 'r') as file:
        for line in file:
            if not line.strip():
                continue
            columns = line.strip().split()
            if len(columns) != 2:
                raise ValueError(f"Invalid line format in coords file: {line.strip()}")
            filename, file_id = columns
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension: {filename}")
            InputFileHandler(file_id, download_now=True).get_file_handle()
            stems.add(Path(filename).stem)

    LOGGER.info(f"Downloaded {len(stems)} unique genetic file sets.")
    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor: CommandExecutor = CMD_EXECUTOR) -> str:
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    output_stub = "Autosomes"
    LOGGER.info(f"Merging {len(genetic_files)} PLINK files into '{output_stub}'...")

    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            merge_list.write(f"{base_name}\n")

    # DEBUG: Check merge list content
    run_debug_cmd("Merge List Content", "head merge_list.txt")

    cmd = f"plink2 --pmerge-list merge_list.txt bfile --make-bed --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)

    # DEBUG: Verify merge success
    run_debug_cmd("Check Merged FAM", f"head -n 5 {output_stub}.fam")

    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    if not Path(f"{genetic_data_file.name}.eigenvec.allele").exists():
        cmd = f"plink2 -pfile {genetic_data_file.name} --pca 3 allele-wts --out {genetic_data_file.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    eigen_df = pd.read_csv(f"{genetic_data_file.name}.eigenvec.allele", sep='\t')
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()
    pd.Series(weak_snps).to_csv(f"{genetic_data_file.name}_eigen_filtered.txt", index=False, header=False)

    cmd = (
        f"plink2 --pfile {genetic_data_file.name} "
        f"--extract {genetic_data_file.name}_eigen_filtered.txt "
        f"--make-bed "
        f"--out {genetic_data_file.name}_filtered_for_kinship"
    )
    cmd_executor.run_cmd_on_docker(cmd)

    cmd = f"plink2 --bfile {genetic_data_file.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    with open(f"{relatedness_db}.kin0", 'r') as kin0_file:
        kin0_data = pd.read_csv(kin0_file, delim_whitespace=True)
        kin0_data = kin0_data[['IID1', 'IID2', 'HETHET', 'IBS0', 'KINSHIP']].rename(
            columns={'IID1': 'ID1', 'IID2': 'ID2', 'HETHET': 'HetHet', 'IBS0': 'IBS0', 'KINSHIP': 'Kinship'}
        )
        kin0_data.to_csv(f"{relatedness_db}_processed.kin0", sep='\t', index=False)

    return Path(f"{relatedness_db}_processed.kin0")


def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:
    rel = rel[rel['ID1'].isin(samples_to_exclude) == False]
    rel = rel[rel['ID2'].isin(samples_to_exclude) == False]

    rel_ids = [rel['ID1'], rel['ID2']]
    rel_ids = pd.DataFrame(data=pd.concat(rel_ids), columns=['ID'])
    rel_ids['dummy'] = [1] * len(rel_ids)
    rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum'))
    rel_totals = rel_totals.sort_values(by='total')

    return {'rel': rel, 'rel_totals': rel_totals}


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    ancestry_dict = {'all': set()}
    with ancestry_file.open(mode='r') as f:
        header = f.readline().strip().split('\t')
        f.seek(0)
        id_candidates = ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id']
        anc_candidates = ['POP', 'ancestry_pred', 'ancestry', 'predicted_ancestry']

        id_col = next((c for c in id_candidates if c in header), header[0])
        anc_col = next((c for c in anc_candidates if c in header), header[1] if len(header) > 1 else None)
        LOGGER.info(f"Ancestry column detected: {anc_col}")

        reader = csv.DictReader(f, delimiter="\t")
        for indv in reader:
            eid = str(indv[id_col])
            pop = indv[anc_col]
            ancestry_dict['all'].add(eid)
            if pop and pop != "NA":
                ancestry_dict.setdefault(pop, set()).add(eid)
    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    LOGGER.info(f"Loading samples from {sample_ids_file.name}...")
    # DEBUG: Check sample file format
    run_debug_cmd(f"Head of {sample_ids_file.name}", f"head -n 5 {sample_ids_file.name}")

    with sample_ids_file.open('r') as wes_samp_file:
        samples = {line.strip().split()[0] for line in wes_samp_file if line.strip()}

    LOGGER.info(f"Loaded {len(samples)} samples.")
    if len(samples) > 0:
        print(f"DEBUG: First 5 loaded IDs: {list(samples)[:5]}")
    return samples


def load_relatedness(relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    rel = pd.read_csv(relatedness, delim_whitespace=True, names=["ID1", "ID2", "Kinship"], skiprows=1)
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
                                    relateds_to_remove: Set[str]) -> List[dxpy.DXFile]:
    include_files = []
    for ancestry in ancestry_dict:
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry])
        pass_samples = pass_samples.difference(relateds_to_remove)

        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        with unrelated_path.open('w') as unrelated_f, related_path.open('w') as related_f:
            for samp in wes_samples:
                if samp in pass_samples:
                    unrelated_f.write(f"{samp}\n")
                if samp in ancestry_dict[ancestry]:
                    related_f.write(f"{samp}\n")

        include_files.extend([
            dxpy.upload_local_file(unrelated_path.name),
            dxpy.upload_local_file(related_path.name)
        ])
    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[
    Set[str], List[dxpy.DXFile]]:
    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)
    rel = load_relatedness(relatedness, wes_samples)
    relateds_to_remove = get_relateds_to_remove(rel)
    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)
    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor=CMD_EXECUTOR) -> dict:
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"
    cmd = f"plink2 --missing 'variant-only' --pfile {merged_data_file.name} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    missingness_qc = csv.DictReader(open(f"{missingness_db}.vmiss", 'r'), delimiter="\t")
    missingness = dict()
    for snp in missingness_qc:
        missingness[snp['ID']] = float(snp['F_MISS'])
    return missingness


def check_qc_ukb(wes_samples: Set[str], missingness: Dict[str, float], ukb_snp_qc: Path, ukb_snps_qc_v2: Path) \
        -> Tuple[Path, Path]:
    """Perform QC checks specific to UK Biobank datasets with strict filtering."""

    pass_snps_file = Path("pass_snps.txt")
    pass_samples = Path("pass_samples.txt")

    LOGGER.info("Starting SNP QC...")
    with open(ukb_snp_qc, 'r') as f_in, pass_snps_file.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ")
        arrs = [f"Batch_b{x:03d}_qc" for x in range(1, 96)] + [f"UKBiLEVEAX_b{x}_qc" for x in range(1, 12)]
        for snp in reader:
            if snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness[snp['rs_id']] < 0.05:
                if all(snp[a] == "1" for a in arrs):
                    f_out.write(snp['rs_id'] + "\n")

    LOGGER.info("Starting Sample QC...")
    ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")

    # --- CRITICAL FIX: Use source FAM for alignment ---
    # Find any original UKB fam file (ukb*.fam) to ensure ID match
    try:
        fam_file = list(Path('.').glob('ukb*.fam'))[0]
        LOGGER.info(f"Using source FAM for QC alignment: {fam_file.name}")
    except IndexError:
        LOGGER.warning("Could not find 'ukb*.fam'. Falling back to 'Autosomes.fam' (Risk of ID mismatch!)")
        fam_file = Path('Autosomes.fam')

    # DEBUG: Check the FAM file we picked
    run_debug_cmd("FAM for Alignment", f"head -n 5 {fam_file.name}")

    subprocess.run(f'paste -d " " {fam_file.name} {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}', shell=True)

    # DEBUG: Check Paste Result
    run_debug_cmd("Paste Result Head", f"head -n 5 {ukb_sqc_v2_with_fam}")

    h = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1', 'affyID1', 'affyID2', 'array', 'batch2', 'plate',
         'well', 'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender', 'x.int', 'y.int', 'plate.sub', 'well.sub',
         'missing.rate', 'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship', 'excl.kinship',
         'excess.relatives', 'in.wba', 'used.pc']
    h.extend([f"PC{x}" for x in range(1, 41)])
    h.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])

    kept_count = 0
    total_count = 0

    with open(ukb_sqc_v2_with_fam, 'r') as f_in, pass_samples.open('w') as f_out:
        reader = csv.DictReader(f_in, delimiter=" ", fieldnames=h)
        for s in reader:
            total_count += 1
            # DEBUG: Diagnose first rejection
            if total_count == 1:
                print(f"DEBUG: Processing first sample ID2={s.get('ID2')}")
                if s['ID2'] not in wes_samples:
                    print(f"DEBUG: First sample REJECTED. Reason: Not in WES samples list.")
                elif not (s['het.missing.outliers'] == "0" and s['in.phasing.auto'] == "1"):
                    print(
                        f"DEBUG: First sample REJECTED. Reason: QC flags (het={s['het.missing.outliers']}, phase={s['in.phasing.auto']})")

            if s['ID2'] in wes_samples:
                if (s['het.missing.outliers'] == "0"
                        and s['in.phasing.auto'] == "1"
                        and s['in.phasing.x'] == "1"
                        and s['in.phasing.xy'] == "1"):
                    f_out.write(f"{s['ID1']} {s['ID2']}\n")
                    kept_count += 1

    LOGGER.info(f"Sample QC Summary: Processed {total_count}, Kept {kept_count}.")

    # DEBUG: Check if empty
    if kept_count == 0:
        print("!!! ALARM: 0 samples passed QC. Check if WES IDs match QC IDs.")

    return pass_snps_file, pass_samples


def check_qc_other(snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
    output_snps = Path("pass_snps.txt")
    output_samples = Path("pass_samples.txt")

    snp_qc = csv.DictReader(open(snp_qc_file, 'r'), delimiter=" ")
    with output_snps.open('w') as snps_file:
        for snp in snp_qc:
            snps_file.write(snp['ID'] + "\n")

    sample_qc = csv.DictReader(open(sample_qc_file, 'r'), delimiter=" ")
    with output_samples.open('w') as samples_file:
        for sample in sample_qc:
            samples_file.write(sample['ID'] + "\n")

    return output_snps, output_samples


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None,
                 output_prefix: str = "Filtered_Data", cmd_executor: CommandExecutor = CMD_EXECUTOR) -> Tuple[
    Path, Path]:
    merged_data_file = Path.cwd() / merged_filename
    LOGGER.info(f"Filtering genotype data. Input: {merged_data_file.name}, Output Prefix: {output_prefix}")

    check_disk_usage()
    snplist = Path(f"{output_prefix}.low_MAC.snplist")

    # --- DEBUG PRE-FLIGHT ---
    run_debug_cmd("Pre-Filter Check: SNP List", f"wc -l {pass_snps.name} && head -n 3 {pass_snps.name}")
    run_debug_cmd("Pre-Filter Check: Sample List", f"wc -l {pass_samples.name} && head -n 3 {pass_samples.name}")
    # ------------------------

    cmd = (f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed "
           f"--extract {pass_snps.name} --keep {pass_samples.name} "
           f"--out {output_prefix} --memory 32000")
    cmd_executor.run_cmd_on_docker(cmd)

    # Fixed typo: --max-m1ac to --max-mac
    cmd = f"plink2 --bfile {output_prefix} --max-mac 100 --write-snplist --out {output_prefix}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)

    if not snplist.exists():
        LOGGER.warning("No low MAC variants found. Creating empty SNPLIST.")
        snplist.touch()

    return Path(output_prefix), snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    if col1 < col2:
        return col2, col1
    else:
        return col1, col2


def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')

    wes_samples_sorted = sorted(wes_samples)
    wes_samples_sorted = pd.DataFrame(data={'ID1': wes_samples_sorted,
                                            'ID2': wes_samples_sorted,
                                            'Kinship': [0.5] * len(wes_samples_sorted)})
    wes_samples_sorted['column1'] = wes_samples_sorted.index + 1
    wes_samples_sorted['column2'] = wes_samples_sorted.index + 1

    gt_matrix = pd.read_csv(rel_mtx, sep="\t", dtype={'ID1': str, 'ID2': str})
    gt_matrix = gt_matrix.drop(columns=['HetHet', 'IBS0'])

    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples)]
    gt_matrix = gt_matrix[gt_matrix['ID2'].isin(wes_samples)]

    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID2', 'column2']], on='ID2', how="left")

    gt_matrix = pd.concat([gt_matrix, wes_samples_sorted])
    gt_matrix = gt_matrix[['column1', 'column2', 'Kinship']]

    gt_matrix[['column1', 'column2']] = gt_matrix.apply(lambda row: column_swap(row['column1'], row['column2']),
                                                        axis=1,
                                                        result_type='expand')

    gt_matrix = gt_matrix.sort_values(['column1', 'column2'])
    gt_matrix['column1'] = gt_matrix.apply(lambda row: '%i' % row['column1'], axis=1)
    gt_matrix['column2'] = gt_matrix.apply(lambda row: '%i' % row['column2'], axis=1)

    with open(grm, 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write('{n_samps} {n_samps} {n_rows}\n'.format(n_samps=len(wes_samples_sorted), n_rows=len(gt_matrix)))
        for row in gt_matrix.iterrows():
            matrix.write(
                '{col1} {col2} {kin}\n'.format(col1=row[1]['column1'], col2=row[1]['column2'], kin=row[1]['Kinship']))

    with open(grm_samples, 'w') as matrix_samples:
        for row in wes_samples_sorted.iterrows():
            matrix_samples.write('{samp}\n'.format(samp=row[1]['ID1']))

    return grm, grm_samples