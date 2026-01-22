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


def check_disk_usage(path: str = ".") -> None:
    """Print disk usage statistics for the given path.

    This is a debugging helper useful for diagnosing "No space left on device"
    errors in large pipelines.

    :param path: The path to check disk usage for. Defaults to current directory.
    """
    total, used, free = shutil.disk_usage(path)
    LOGGER.info(
        f"DISK USAGE ({path}): "
        f"Total: {total // (2 ** 30)}GB | "
        f"Used: {used // (2 ** 30)}GB | "
        f"Free: {free // (2 ** 30)}GB"
    )
    # Run df -h for detailed mount visibility
    subprocess.run("df -h", shell=True)


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

    check_disk_usage()
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
    """Merge multiple PLINK binary files into a single dataset.

    This method skips merging if only one file set is present.

    :param genetic_files: A set of PLINK file stems to merge.
    :param cmd_executor: An executor to run shell commands.
    :return: The file stem of the merged PLINK dataset.
    """
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    output_stub = "Autosomes"
    LOGGER.info(f"Merging {len(genetic_files)} PLINK files into '{output_stub}'...")

    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            merge_list.write(f"{base_name}\n")

    # force BED/BIM/FAM output
    cmd = f"plink2 --pmerge-list merge_list.txt bfile --make-bed --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)

    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    """
    This function calculates the relatedness of the samples in the genetic data file

    :param genetic_data_file: a path to the genetic data file
    :param run_king: a boolean indicating whether to run KING for relatedness calculation
    :param cmd_executor: a command executor object to run commands on the docker instance
    :return: a path to the relatedness file matrix
    """

    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    # first we need to calculate the PCs
    # as it takes a long time let's only do this if the file does not already exist
    if not Path(f"{genetic_data_file.name}.eigenvec.allele").exists():
        cmd = f"plink2 -pfile {genetic_data_file.name} --pca 3 allele-wts --out {genetic_data_file.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    eigen_df = pd.read_csv(f"{genetic_data_file.name}.eigenvec.allele", sep='\t')
    # print(eigen_df.head())
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()
    # Save list
    pd.Series(weak_snps).to_csv(f"{genetic_data_file.name}_eigen_filtered.txt", index=False, header=False)

    # Filter variants for kinship analysis using PLINK2
    cmd = (
        f"plink2 --pfile {genetic_data_file.name} "
        f"--extract {genetic_data_file.name}_eigen_filtered.txt "
        f"--make-bed "
        f"--out {genetic_data_file.name}_filtered_for_kinship"
    )
    cmd_executor.run_cmd_on_docker(cmd)

    # # Calculate relatedness using KING:
    cmd = f"plink2 --bfile {genetic_data_file.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    with open(f"{relatedness_db}.kin0", 'r') as kin0_file:
        kin0_data = pd.read_csv(kin0_file, delim_whitespace=True)
        # Keep and rename specific columns
        kin0_data = kin0_data[['IID1', 'IID2', 'HETHET', 'IBS0', 'KINSHIP']].rename(
            columns={'IID1': 'ID1',
                     'IID2': 'ID2',
                     'HETHET': 'HetHet',
                     'IBS0': 'IBS0',
                     'KINSHIP': 'Kinship'}
        )
        # Write the processed DataFrame to a file
        kin0_data.to_csv(f"{relatedness_db}_processed.kin0", sep='\t', index=False)

    relatedness_output = f"{relatedness_db}_processed.kin0"

    return Path(relatedness_output)


def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:
    """
    This function selects related individuals from the relatedness file and returns a dictionary with the
    relatedness file and the total number of times each individual occurs in the relatedness file

    :param rel: a pandas DataFrame containing the relatedness file
    :param samples_to_exclude: a list of individuals to exclude from the relatedness file
    :return: a dictionary with the relatedness file and the total number of times each individual occurs in the relatedness file
    """
    # Remove individuals not in samples_to_exclude:
    rel = rel[rel['ID1'].isin(samples_to_exclude) == False]
    rel = rel[rel['ID2'].isin(samples_to_exclude) == False]

    # Get a list of related individuals:
    # This first bit makes one column of ID1 and ID2 so we can total the amount of times each individual occurs in rel
    rel_ids = [rel['ID1'], rel['ID2']]
    rel_ids = pd.DataFrame(data=pd.concat(rel_ids), columns=['ID'])  # and convert back into a DataFrame

    # This makes a dummy variable for each individual so that we can...
    rel_ids['dummy'] = [1] * len(rel_ids)
    # ... sum it together to count the number of times that individual appears in the list ...
    rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum'))
    # ... and then we sort it by that value
    rel_totals = rel_totals.sort_values(by='total')

    return {'rel': rel, 'rel_totals': rel_totals}


def load_ancestry_dict(ancestry_file: Path) -> Dict[str, Set[str]]:
    """
    This function loads the ancestry file and returns a dictionary
     with ancestry as keys and sets of individual IDs as values.
    :param ancestry_file: a file containing the ancestry sample IDs and ancestry information
    :return: a dictionary with ancestry as keys and sets of individual IDs as values
    """
    # Get WBA individuals
    # - Need to be able to generate some separate list of individuals
    # - This is the list generated by Felix to be a bit more inclusive
    ancestry_dict: Dict[str, Set[str]] = {'all': set()}
    with ancestry_file.open(mode='r') as ancestry_info:
        ancestry_reader = csv.DictReader(ancestry_info, delimiter="\t")
        ancestry_col = None
        # Find the column containing ancestry values
        for col in ancestry_reader.fieldnames:
            ancestry_info.seek(0)
            next(ancestry_info)  # skip header
            for row in ancestry_reader:
                val = row[col].lower()
                if val in {'eur', 'sas', 'afr'}:
                    ancestry_col = col
                    # print the column we found so we can check it's correct
                    LOGGER.info(f"Ancestry column detected: {ancestry_col}")
                    break
            if ancestry_col:
                break
        ancestry_info.seek(0)
        ancestry_reader = csv.DictReader(ancestry_info, delimiter="\t")
        for indv in ancestry_reader:
            eid = str(list(indv.values())[0])
            ancestry_dict['all'].add(eid)
            if ancestry_col and indv[ancestry_col] != "NA":
                ancestry_dict.setdefault(indv[ancestry_col], set()).add(eid)
    return ancestry_dict


def load_samples(sample_ids_file: Path) -> Set[str]:
    """
    This function loads the sample IDs file and returns a set of individual IDs.
    :param sample_ids_file: a file containing the sample IDs
    :return: a set of individual IDs
    """
    # Read overall list of individuals with data so we can subset the genetic data.
    with sample_ids_file.open('r') as wes_samp_file:
        return {line.strip() for line in wes_samp_file if line.strip()}


def load_relatedness(relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
    """
    This function loads the relatedness file and returns a DataFrame containing only the related individuals
    :param relatedness:  a file containing the relatedness matrix table
    :param wes_samples: a set of individual IDs that are WES samples
    :return: a DataFrame containing only the related individuals
    """
    # Calculate relateds:
    # Read the relatedness file in as a pandas DataFrame
    # dtype sets eids as characters
    # ID1 and ID2 are two spearate individuals that are related according to some kinship value
    # Check if the file is empty
    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    # Read the file and filter based on WES samples
    rel = pd.read_csv(
        relatedness,
        delim_whitespace=True,
        names=["ID1", "ID2", "Kinship"],
        skiprows=1
    )
    return rel[(rel["ID1"].isin(wes_samples)) & (rel["ID2"].isin(wes_samples))]


def get_relateds_to_remove(rel: pd.DataFrame) -> Set[str]:
    """
    This function identifies individuals to remove from the relatedness DataFrame based on their relatedness pairs.
    :param rel: a pandas DataFrame containing the relatedness file
    :return: a set of individual IDs to remove from the relatedness DataFrame
    """
    # The relatedenss list is passed to the select_related_individuals() function for the first time so that we can
    # just calculate the number of times each individual occurs in the rel file after we limit to WES samples
    # parameter 1 is a pandas DataFrame
    # parameter 2 is a list of individuals we want to remove from parameter 1
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
    """
    This function writes ancestry-specific inclusion files for samples and uploads them to DNANexus.
    :param wes_samples: a set of sample IDs
    :param ancestry_dict: a dictionary with ancestry as keys and sets of individual IDs as values
    :param relateds_to_remove: a set of individual IDs to remove from the relatedness DataFrame
    :return: a list of DXFile objects representing the uploaded inclusion files
    """

    # Write ancestry-specific exclusion lists, relatedness, and combo of the two:
    # 1. list of WES non-ancestry or related individuals
    # 2. list of ancestry-specific individuals with WES
    # 3. list of related individuals with WES

    # Get lists of samples to include specific to certain ancestries:
    include_files = []

    for ancestry in ancestry_dict:
        pass_samples = wes_samples.intersection(ancestry_dict[ancestry])
        pass_samples = pass_samples.difference(relateds_to_remove)

        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        # This writes to each list based on a set of requirements
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
    """
    Generates a list of unrelated individuals for each ancestry and uploads inclusion files to DNANexus.

    :param sample_ids_file: a file containing the sample IDs
    :param ancestry_file: a file containing ancestry sample IDs and ancestry info
    :param relatedness: a file with relatedness matrix
    :return: a tuple of samples and uploaded DXFiles
    """
    ancestry_dict = load_ancestry_dict(ancestry_file)
    wes_samples = load_samples(sample_ids_file)
    rel = load_relatedness(relatedness, wes_samples)
    relateds_to_remove = get_relateds_to_remove(rel)
    include_files = write_and_upload_ancestry_files(wes_samples, ancestry_dict, relateds_to_remove)
    return wes_samples, include_files


def calculate_missingness(merged_filename: str, cmd_executor=CMD_EXECUTOR) -> dict:
    """
    This function calculates the missingness of the SNPs in the merged plink file
    :param merged_filename: a file containing the merged plink file
    :param cmd_executor: a command executor object to run commands on the docker instance
    :return: a dictionary with SNP IDs as keys and their missingness as values
    """
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"

    # First generate missingness information for all SNPs:
    cmd = f"plink2 --missing 'variant-only' --pfile {merged_data_file.name} --out {missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    # Then read as a pandas DataFrame:
    missingness_qc = csv.DictReader(open(f"{missingness_db}.vmiss", 'r'),
                                    delimiter="\t")
    # And convert to a dictionary with format SNP ID : missingness
    missingness = dict()
    for snp in missingness_qc:
        missingness[snp['ID']] = float(snp['F_MISS'])

    return missingness


def check_qc_ukb(wes_samples: set, missingness: dict, ukb_snp_qc: Path, ukb_snps_qc_v2: Path,
                 cmd_executor=CMD_EXECUTOR) -> Tuple[Path, Path]:
    """
    This function checks the quality control of the SNPs and samples in the genetic data file

    :param wes_samples: a set of WES samples
    :param missingness: a dictionary of SNP IDs and their missingness values
    :param ukb_snp_qc: a file containing the SNP QC information
    :param ukb_snps_qc_v2: a file containing the SNP QC information version 2
    :param cmd_executor: Command Executor for running commands on Docker
    :return: a tuple of the pass SNPs file and the pass samples file
    """

    pass_snps_file = Path("pass_snps.txt")

    # Read in UKBiobank provided quality control for SNPs
    snp_qc = csv.DictReader(open(ukb_snp_qc, 'r'), delimiter=" ")
    # Create a simple list of SNPs that pass our QC
    pass_snps = open(pass_snps_file, 'w')

    # Generate list of the names of arrays so we can iterate through them programmatically below
    array_names = []
    for x in range(1, 96):  # Why is range zero-based... but not?
        array_names.append("Batch_b%03d_qc" % x)
    for x in range(1, 12):
        array_names.append("UKBiLEVEAX_b%d_qc" % x)

    # And then check each SNP to make sure it is on both arrays, an autosome and has missingness < 0.05%,
    for snp in snp_qc:
        if snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness[snp['rs_id']] < 0.05:
            pass_batch_qc = True
            # Now iterate through each individual array and make sure the SNP passes there
            for array_ID in array_names:
                if snp[array_ID] != "1":
                    pass_batch_qc = False

            if pass_batch_qc:
                pass_snps.write(snp['rs_id'] + "\n")

    pass_snps.close()

    # Have to generate a pasted version of the sample QC file with the fam file to get useable sample IDs:
    ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")
    cmd = f'paste -d " " {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}'
    subprocess.run(cmd, shell=True)
    # Check sample QC files:
    # Here generating a header that mashes together the two files above
    snp_qc_header = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1',
                     'affyID1', 'affyID2', 'array', 'batch2', 'plate', 'well',
                     'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender',
                     'x.int', 'y.int', 'plate.sub', 'well.sub', 'missing.rate',
                     'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship',
                     'excl.kinship', 'excess.relatives', 'in.wba', 'used.pc']
    snp_qc_header.extend(["PC%d" % item for item in range(1, 41)])
    snp_qc_header.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])

    snp_qc = csv.DictReader(open(ukb_sqc_v2_with_fam, 'r'), delimiter=" ", fieldnames=snp_qc_header)
    # write pass IDs as a file:
    pass_samples = Path("pass_samples.txt")
    wr_file = open(pass_samples, 'w')

    # Retain samples that are:
    # 1. In the WES samples
    # 2. Are not missingness outliers
    # 3. Are included in autosomal phasing
    for sample in snp_qc:
        if sample['ID1'] in wes_samples \
                and sample['het.missing.outliers'] == "0" \
                and sample['in.phasing.auto'] == "1" \
                and sample['in.phasing.x'] == "1" \
                and sample['in.phasing.xy'] == "1":
            wr_file.write(sample['ID1'] + "\n")

    wr_file.close()

    return pass_snps_file, pass_samples


def check_qc_other(snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
    """
    When working with non-DNA Nexus files, we may still have some QC files that we need to check. This is a
    placeholder function to do that.

    :param snp_qc_file: a file containing the SNP QC information
    :param sample_qc_file: a file containing the sample QC information
    :return: a tuple of the pass SNPs file and the pass samples file
    """

    output_snps = Path("pass_snps.txt")
    output_samples = Path("pass_samples.txt")

    # Read in the SNP QC file
    snp_qc = csv.DictReader(open(snp_qc_file, 'r'), delimiter=" ")
    # Create a simple list of SNPs that pass our QC
    with output_snps.open('w') as snps_file:
        for snp in snp_qc:
            snps_file.write(snp['ID'] + "\n")  # Assuming 'ID' is the column name for SNP IDs

    # Read in the sample QC file
    sample_qc = csv.DictReader(open(sample_qc_file, 'r'), delimiter=" ")
    # Create a simple list of samples that pass our QC
    with output_samples.open('w') as samples_file:
        for sample in sample_qc:
            samples_file.write(sample['ID'] + "\n")  # Assuming 'ID' is the column name for sample IDs

    return output_snps, output_samples


def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None, cmd_executor=CMD_EXECUTOR) -> Tuple[
    Path, Path]:
    """
    This function filters the merged plink file based on the pass SNPs and pass samples files

    :param merged_filename: a file containing the merged plink file
    :param pass_snps: a file containing the pass SNPs
    :param pass_samples: a file containing the pass samples
    :param cmd_executor: a command executor object to run commands on the docker instance
    :return: a path to the filtered merged plink file and a path to the low MAC SNPs file
    """

    merged_data_file = Path.cwd() / merged_filename
    snplist = Path(merged_data_file.name).with_suffix(".low_MAC.snplist")

    # CRITICAL: Cap PLINK memory usage (~32GB) to leave room for Python/OS
    cmd = (f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed "
           f"--extract {pass_snps.name} --keep {pass_samples.name} "
           f"--out {merged_data_file.name} --memory 32000")
    cmd_executor.run_cmd_on_docker(cmd)

    # Generate a list of low MAC sites for BOLT
    cmd = f"plink2 --bfile {merged_data_file.name} --max-mac 100 --write-snplist " \
          f"--out {merged_data_file.name}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd)

    return merged_data_file, snplist


def column_swap(col1: str, col2: str) -> Tuple[str, str]:
    """
    This function swaps the columns of a matrix to ensure that the resulting matrix is lower-left

    :param col1: is the first column
    :param col2: is the second column
    :return: a tuple of the swapped columns
    """
    if col1 < col2:
        return col2, col1
    else:
        return col1, col2


def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    """
    This function generates a GRM from the WES samples and the relatedness matrix
    We use the KING-relate derived relatedness information for our GRM. Just need to convert it into a format that
    SAIGE and STAAR can use...

    :param wes_samples: a set of WES samples
    :param rel_mtx: a file containing the relatedness matrix table
    :return: None
    """

    grm = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')
    grm_samples = Path('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')

    # Construct a pd.DataFrame of wes_samples for merging purposes
    wes_samples_sorted = sorted(wes_samples)
    wes_samples_sorted = pd.DataFrame(data={'ID1': wes_samples_sorted,
                                            'ID2': wes_samples_sorted,
                                            'Kinship': [0.5] * len(wes_samples_sorted)})
    wes_samples_sorted['column1'] = wes_samples_sorted.index + 1
    wes_samples_sorted['column2'] = wes_samples_sorted.index + 1

    # import UKBB KING matrix
    gt_matrix = pd.read_csv(rel_mtx, sep="\t", dtype={'ID1': str, 'ID2': str})
    gt_matrix = gt_matrix.drop(columns=['HetHet', 'IBS0'])

    # Filter to individuals that have WES data...
    gt_matrix = gt_matrix[gt_matrix['ID1'].isin(wes_samples)]
    gt_matrix = gt_matrix[gt_matrix['ID2'].isin(wes_samples)]

    # Get column incidies from the wes_samples for the gt matrix
    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID1', 'column1']], on='ID1', how="left")
    gt_matrix = pd.merge(gt_matrix, wes_samples_sorted[['ID2', 'column2']], on='ID2', how="left")

    # Add all samples to complete the matrix diagonal and drop EIDs
    gt_matrix = pd.concat([gt_matrix, wes_samples_sorted])
    gt_matrix = gt_matrix[['column1', 'column2', 'Kinship']]

    # And ensure that the matrix is lower left and eids are in integer format:
    gt_matrix[['column1', 'column2']] = gt_matrix.apply(lambda row: column_swap(row['column1'], row['column2']),
                                                        axis=1,
                                                        result_type='expand')

    # And sort...
    gt_matrix = gt_matrix.sort_values(['column1', 'column2'])

    # and ensure columns #s are in integer format:
    gt_matrix['column1'] = gt_matrix.apply(lambda row: '%i' % row['column1'], axis=1)
    gt_matrix['column2'] = gt_matrix.apply(lambda row: '%i' % row['column2'], axis=1)

    # And print outputs:
    with open(grm, 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write('{n_samps} {n_samps} {n_rows}\n'.format(n_samps=len(wes_samples_sorted), n_rows=len(gt_matrix)))
        for row in gt_matrix.iterrows():
            ret = matrix.write(
                '{col1} {col2} {kin}\n'.format(col1=row[1]['column1'], col2=row[1]['column2'], kin=row[1]['Kinship']))
        matrix.close()

    with open(grm_samples, 'w') as matrix_samples:
        for row in wes_samples_sorted.iterrows():
            ret = matrix_samples.write('{samp}\n'.format(samp=row[1]['ID1']))
        matrix_samples.close()

    return grm, grm_samples
