import csv
import subprocess
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

import pandas as pd
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor
from general_utilities.mrc_logger import MRCLogger

CMD_EXECUTOR = build_default_command_executor()

LOGGER = MRCLogger().get_logger()


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, relatedness_file: dict) -> \
        Tuple[set, Path, Path, Optional[Path]]:
    """
    This function downloads the data we will need to run this module
    """
    genetic_data_file = InputFileHandler(genetic_data_file, download_now=True).get_file_handle()
    genetic_files = download_genetic_data(genetic_data_file)

    sample_ids_file = InputFileHandler(sample_ids_file, download_now=True).get_file_handle()
    ancestry_file = InputFileHandler(ancestry_file, download_now=True).get_file_handle()

    if relatedness_file is not None:
        relatedness_file = InputFileHandler(relatedness_file, download_now=True).get_file_handle()

    return genetic_files, sample_ids_file, ancestry_file, relatedness_file


def download_genetic_data(input_file_list: Path) -> set:
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()
    with open(input_file_list, 'r') as file:
        for line in file:
            if not line.strip(): continue
            columns = line.strip().split()
            if len(columns) != 2:
                raise ValueError(f"Each line must have exactly two columns. Invalid line: {line.strip()}")
            filename, file_id = columns
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension in filename: {filename}")

            InputFileHandler(file_id, download_now=True).get_file_handle()
            stems.add(Path(filename).stem)
    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor=CMD_EXECUTOR) -> str:
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    output_stub = "Autosomes"
    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            merge_list.write(f"{base_name}\n")
    cmd = f"plink2 --pmerge-list merge_list.txt bfile --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)
    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    if not Path(f"{genetic_data_file.name}.eigenvec.allele").exists():
        cmd = f"plink2 --bfile {genetic_data_file.name} --pca 3 allele-wts --out {genetic_data_file.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    eigen_df = pd.read_csv(f"{genetic_data_file.name}.eigenvec.allele", sep='\t')
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()
    pd.Series(weak_snps).to_csv(f"{genetic_data_file.name}_eigen_filtered.txt", index=False, header=False)

    cmd = (
        f"plink2 --bfile {genetic_data_file.name} "
        f"--extract {genetic_data_file.name}_eigen_filtered.txt "
        f"--make-bed "
        f"--out {genetic_data_file.name}_filtered_for_kinship"
    )
    cmd_executor.run_cmd_on_docker(cmd)

    cmd = f"plink2 --bfile {genetic_data_file.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    relatedness_output = f"{relatedness_db}.kin0"
    return Path(relatedness_output)


def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:
    rel = rel[rel['ID1'].isin(samples_to_exclude) == False]
    rel = rel[rel['ID2'].isin(samples_to_exclude) == False]
    rel_ids = pd.DataFrame(data=pd.concat([rel['ID1'], rel['ID2']]), columns=['ID'])
    rel_ids['dummy'] = [1] * len(rel_ids)
    rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum')).sort_values(by='total')
    return {'rel': rel, 'rel_totals': rel_totals}


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    ancestry_dict: Dict[str, Set[str]] = {'all': set()}
    ancestry_counts = {}

    with ancestry_file.open(mode='r') as ancestry_info:
        header_line = ancestry_info.readline()
        header = header_line.strip().split('\t')
        ancestry_info.seek(0)

        id_col = None
        possible_id_cols = ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id']
        for candidate in possible_id_cols:
            if candidate in header:
                id_col = candidate
                break
        if not id_col:
            id_col = header[0]
            LOGGER.warning(f"Could not detect standard ID column. Defaulting to: '{id_col}'")

        anc_col = None
        possible_anc_cols = ['POP', 'ancestry_pred', 'ancestry', 'predicted_ancestry']
        for candidate in possible_anc_cols:
            if candidate in header:
                anc_col = candidate
                break
        if not anc_col:
            if len(header) > 1:
                anc_col = header[1]
                LOGGER.warning(f"Could not detect standard ancestry column. Defaulting to: '{anc_col}'")
            else:
                raise ValueError("Ancestry file missing second column.")

        LOGGER.info(f"Loading ancestry using ID: '{id_col}' and Ancestry: '{anc_col}'")
        ancestry_reader = csv.DictReader(ancestry_info, delimiter="\t")

        for indv in ancestry_reader:
            eid = str(indv[id_col])
            ancestry_val = indv[anc_col]
            ancestry_dict['all'].add(eid)
            if ancestry_val != "NA" and ancestry_val is not None:
                ancestry_dict.setdefault(ancestry_val, set()).add(eid)
                ancestry_counts[ancestry_val] = ancestry_counts.get(ancestry_val, 0) + 1

    LOGGER.info("\n" + "=" * 40)
    LOGGER.info(f"{'ANCESTRY GROUP':<25} | {'COUNT':<10}")
    LOGGER.info("-" * 40)
    for anc, count in sorted(ancestry_counts.items()):
        LOGGER.info(f"{anc:<25} | {count:<10}")
    LOGGER.info("-" * 40)
    LOGGER.info(f"{'TOTAL SAMPLES':<25} | {len(ancestry_dict['all']):<10}")
    LOGGER.info("=" * 40 + "\n")
    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    with sample_ids_file.open('r') as wes_samp_file:
        return {line.strip().split()[0] for line in wes_samp_file if line.strip()}


def _read_and_clean_relatedness(relatedness: Path) -> pd.DataFrame:
    """
    Helper to read and normalize relatedness file.
    Centralizes logic for load_relatedness and make_grm.
    """
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    rel = pd.read_csv(relatedness, delim_whitespace=True)

    # 1. Rename AoU headers (i.s -> ID1)
    if 'i.s' in rel.columns and 'j.s' in rel.columns and 'kin' in rel.columns:
        rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})

    # 2. Rename PLINK headers (strip leading #)
    rel.columns = rel.columns.str.replace('^#', '', regex=True)
    rel = rel.rename(columns={'IID1': 'ID1', 'IID2': 'ID2', 'KINSHIP': 'Kinship'})

    # 3. Fallback for unexpected headers (Index-based rename)
    if not {'ID1', 'ID2', 'Kinship'}.issubset(rel.columns):
        LOGGER.warning(f"Standard headers not found. Found: {rel.columns.tolist()}. Trying index-based fallback.")
        # Re-read without header to safely access by index
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
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry])
        pass_samples = pass_samples.difference(relateds_to_remove)

        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        with unrelated_path.open('w') as unrelated_f, related_path.open('w') as related_f:
            for samp in wes_samples:
                if samp in pass_samples: unrelated_f.write(f"{samp}\n")
                if samp in ancestry_dict[ancestry]: related_f.write(f"{samp}\n")

        include_files.append(exporter.export_files(unrelated_path.name))
        include_files.append(exporter.export_files(related_path.name))

    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[
    Set[str], List]:
    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)
    rel = load_relatedness(relatedness, wes_samples)
    relateds_to_remove = get_relateds_to_remove(rel)
    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)
    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor=CMD_EXECUTOR) -> dict:
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"
    cmd = f"plink2 --missing 'variant-only' --bfile {merged_data_file.name} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)
    missingness_qc = csv.DictReader(open(f"{missingness_db}.vmiss", 'r'), delimiter="\t")
    missingness = dict()
    for snp in missingness_qc:
        missingness[snp['ID']] = float(snp['F_MISS'])
    return missingness


def check_qc_ukb(wes_samples: set, missingness: dict, ukb_snp_qc: Path, ukb_snps_qc_v2: Path,
                 cmd_executor=CMD_EXECUTOR) -> Tuple[Path, Path]:
    pass_snps_file = Path("pass_snps.txt")
    snp_qc = csv.DictReader(open(ukb_snp_qc, 'r'), delimiter=" ")
    pass_snps = open(pass_snps_file, 'w')
    array_names = []
    for x in range(1, 96): array_names.append("Batch_b%03d_qc" % x)
    for x in range(1, 12): array_names.append("UKBiLEVEAX_b%d_qc" % x)
    for snp in snp_qc:
        if snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness[snp['rs_id']] < 0.05:
            pass_batch_qc = True
            for array_ID in array_names:
                if snp[array_ID] != "1": pass_batch_qc = False
            if pass_batch_qc: pass_snps.write(snp['rs_id'] + "\n")
    pass_snps.close()
    ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")
    cmd = f'paste -d " " {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}'
    subprocess.run(cmd, shell=True)
    snp_qc_header = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1',
                     'affyID1', 'affyID2', 'array', 'batch2', 'plate', 'well',
                     'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender',
                     'x.int', 'y.int', 'plate.sub', 'well.sub', 'missing.rate',
                     'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship',
                     'excl.kinship', 'excess.relatives', 'in.wba', 'used.pc']
    snp_qc_header.extend(["PC%d" % item for item in range(1, 41)])
    snp_qc_header.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])
    snp_qc = csv.DictReader(open(ukb_sqc_v2_with_fam, 'r'), delimiter=" ", fieldnames=snp_qc_header)
    pass_samples = Path("pass_samples.txt")
    wr_file = open(pass_samples, 'w')
    for sample in snp_qc:
        if sample['ID1'] in wes_samples and sample['het.missing.outliers'] == "0" and sample[
            'in.phasing.auto'] == "1" and sample['in.phasing.x'] == "1" and sample['in.phasing.xy'] == "1":
            wr_file.write(sample['ID1'] + "\n")
    wr_file.close()
    return pass_snps_file, pass_samples


def check_qc_other(wes_samples: set, snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
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
                if 's' in row:
                    flagged_samples.add(row['s'])
                else:
                    flagged_samples.add(list(row.values())[0])
    except Exception as e:
        LOGGER.warning(f"Could not parse flagged samples file as TSV: {e}. Trying simple list.")
        with open(sample_qc_file, 'r') as f:
            for line in f: flagged_samples.add(line.strip().split()[0])

    final_samples = wes_samples - flagged_samples
    LOGGER.info(f"Retaining {len(final_samples)} samples after QC filtering.")

    with output_samples.open('w') as f_out:
        for sample in final_samples: f_out.write(f"{sample}\n")
    return output_snps, output_samples


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None, cmd_executor=CMD_EXECUTOR) -> Tuple[
    Path, Path]:
    merged_data_file = Path.cwd() / merged_filename
    snplist = Path(merged_data_file.name + ".low_MAC.snplist")
    cmd = f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed --extract {pass_snps.name} --keep-fam {pass_samples.name} --out {merged_data_file.name}"
    cmd_executor.run_cmd_on_docker(cmd)
    cmd = f"plink2 --bfile {merged_data_file.name} --max-mac 100 --write-snplist --out {merged_data_file.name}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)
    if not snplist.exists():
        LOGGER.warning(f"No low MAC variants found. Creating empty SNPLIST at {snplist}")
        snplist.touch()
    return merged_data_file, snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    if col1 < col2:
        return col2, col1
    else:
        return col1, col2


def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')

    wes_samples_sorted = sorted(wes_samples)
    wes_samples_sorted = pd.DataFrame(
        data={'ID1': wes_samples_sorted, 'ID2': wes_samples_sorted, 'Kinship': [0.5] * len(wes_samples_sorted)})
    wes_samples_sorted['column1'] = wes_samples_sorted.index + 1
    wes_samples_sorted['column2'] = wes_samples_sorted.index + 1

    # RE-USE ROBUST LOAD LOGIC
    gt_matrix = _read_and_clean_relatedness(rel_mtx)

    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples)]
    gt_matrix = gt_matrix[gt_matrix['ID2'].isin(wes_samples)]

    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID2', 'column2']], on='ID2', how="left")

    gt_matrix = pd.concat([gt_matrix, wes_samples_sorted])
    gt_matrix = gt_matrix[['column1', 'column2', 'Kinship']]

    gt_matrix[['column1', 'column2']] = gt_matrix.apply(lambda row: column_swap(row['column1'], row['column2']), axis=1,
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
        matrix.close()

    with open(grm_samples, 'w') as matrix_samples:
        for row in wes_samples_sorted.iterrows():
            matrix_samples.write('{samp}\n'.format(samp=row[1]['ID1']))
        matrix_samples.close()

    return grm, grm_samples
