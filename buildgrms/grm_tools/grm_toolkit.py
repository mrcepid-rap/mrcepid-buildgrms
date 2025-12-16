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

    :param genetic_data_file: a file containing the genetic data file names and IDs
    :param sample_ids_file: a file containing the sample IDs
    :param ancestry_file: a file containing the ancestry sample IDs
    :param relatedness_file: a file containing the relatedness matrix table
    :return: a tuple of the genetic data file, sample IDs file, and ancestry file
    """
    # Ingest the UKBB plink files (this also includes relatedness and snp/sample QC files)
    genetic_data_file = InputFileHandler(genetic_data_file, download_now=True).get_file_handle()
    genetic_files = download_genetic_data(genetic_data_file)

    # Download a pre-computed sample IDs file.
    sample_ids_file = InputFileHandler(sample_ids_file, download_now=True).get_file_handle()

    # Download wba file:
    ancestry_file = InputFileHandler(ancestry_file, download_now=True).get_file_handle()

    # Download the relatedness files:
    if relatedness_file is not None:
        relatedness_file = InputFileHandler(relatedness_file, download_now=True).get_file_handle()

    return genetic_files, sample_ids_file, ancestry_file, relatedness_file


def download_genetic_data(input_file_list: Path) -> set:
    """
    Downloads genetic data files using input coordinates.

    :param input_file_list: a file containing the genetic data file names and IDs
    :return: a set of unique stems (prefixes) of the files for downstream use
    """
    valid_extensions = {'.bed', '.bim', '.fam'}
    stems = set()
    with open(input_file_list, 'r') as file:
        for line in file:
            # Skip empty lines if any
            if not line.strip():
                continue

            columns = line.strip().split()
            if len(columns) != 2:
                raise ValueError(f"Each line must have exactly two columns. Invalid line: {line.strip()}")
            filename, file_id = columns
            if not any(filename.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file extension in filename: {filename}")

            # Download the file
            # InputFileHandler now automatically detects 'gs://' and handles it
            InputFileHandler(file_id, download_now=True).get_file_handle()

            # Add stem for downstream use
            stems.add(Path(filename).stem)
    return stems


def merge_plink_files(genetic_files: Set[str], cmd_executor=CMD_EXECUTOR) -> str:
    """
    This function merges all autosomal files together.
    It uses the set of file stems provided by download_genetic_data.

    If only one file is present (e.g. All of Us Array data), it skips the merge step.

    :param genetic_files: a set of file stems (prefixes) to be merged
    :param cmd_executor: a command executor object to run commands on the docker instance
    :return: the name of the merged file
    """

    # If we only have one file, we don't need to merge anything.
    if len(genetic_files) == 1:
        single_file = list(genetic_files)[0]
        LOGGER.info(f"Only one genetic file detected ({single_file}). Skipping merge step.")
        return single_file

    # OUTPUT STUB
    output_stub = "Autosomes"

    # Merge autosomal PLINK files together:
    # We iterate through the set of stems we already know we downloaded.
    # We sort them to ensure the order is deterministic.
    with open('merge_list.txt', 'w') as merge_list:
        for base_name in sorted(genetic_files):
            # We assume the docker container maps the current directory to
            # and that PLINK expects the prefix without extension
            merge_list.write(f"{base_name}\n")

    # Run PLINK merge command
    # Note: --pmerge-list takes a file containing a list of plink file stems
    cmd = f"plink2 --pmerge-list merge_list.txt bfile --out {output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)

    return output_stub


def calculate_relatedness(genetic_data_file: str, cmd_executor=CMD_EXECUTOR) -> Path:
    """
    This function calculates the relatedness of the samples in the genetic data file

    :param genetic_data_file: a path to the genetic data file
    :param cmd_executor: a command executor object to run commands on the docker instance
    :return: a path to the relatedness file matrix
    """

    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    # first we need to calculate the PCs
    # as it takes a long time let's only do this if the file does not already exist
    if not Path(f"{genetic_data_file.name}.eigenvec.allele").exists():
        # FIX: Changed -pfile to --bfile for compatibility with PLINK binary format
        cmd = f"plink2 --bfile {genetic_data_file.name} --pca 3 allele-wts --out {genetic_data_file.name}"
        cmd_executor.run_cmd_on_docker(cmd)

    eigen_df = pd.read_csv(f"{genetic_data_file.name}.eigenvec.allele", sep='\t')
    # print(eigen_df.head())
    filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
    weak_snps = filtered_eigen_df['ID'].unique()
    # Save list
    pd.Series(weak_snps).to_csv(f"{genetic_data_file.name}_eigen_filtered.txt", index=False, header=False)

    # Filter variants for kinship analysis using PLINK2
    # FIX: Changed --pfile to --bfile
    cmd = (
        f"plink2 --bfile {genetic_data_file.name} "
        f"--extract {genetic_data_file.name}_eigen_filtered.txt "
        f"--make-bed "
        f"--out {genetic_data_file.name}_filtered_for_kinship"
    )
    cmd_executor.run_cmd_on_docker(cmd)

    # # Calculate relatedness using KING:
    cmd = f"plink2 --bfile {genetic_data_file.name}_filtered_for_kinship --make-king-table --out {relatedness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    # Note: PLINK 2 king-table output headers are typically: #IID1 IID2 KINSHIP
    # The file extension is .kin0
    relatedness_output = f"{relatedness_db}.kin0"

    # We return the raw output path. load_relatedness() handles standardization.
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

    It supports multiple header formats (research_id, n_eid, person_id, IID)
    and (POP, ancestry_pred). It also logs a summary table of found ancestries.

    :param ancestry_file: a file containing the ancestry sample IDs and ancestry information
    :return: a dictionary with ancestry as keys and sets of individual IDs as values
    """
    ancestry_dict: Dict[str, Set[str]] = {'all': set()}
    ancestry_counts = {}

    with ancestry_file.open(mode='r') as ancestry_info:
        # Read headers to find ID and Ancestry columns
        header_line = ancestry_info.readline()
        header = header_line.strip().split('\t')

        # Reset file pointer
        ancestry_info.seek(0)

        # 1. Detect ID Column
        id_col = None
        possible_id_cols = ['research_id', 'n_eid', 'person_id', 'IID', 'sample_id']
        for candidate in possible_id_cols:
            if candidate in header:
                id_col = candidate
                break

        if not id_col:
            id_col = header[0]
            LOGGER.warning(
                f"Could not detect standard ID column ({possible_id_cols}). Defaulting to first column: '{id_col}'")

        # 2. Detect Ancestry Column
        anc_col = None
        possible_anc_cols = ['POP', 'ancestry_pred', 'ancestry', 'predicted_ancestry']
        for candidate in possible_anc_cols:
            if candidate in header:
                anc_col = candidate
                break

        if not anc_col:
            if len(header) > 1:
                anc_col = header[1]
                LOGGER.warning(
                    f"Could not detect standard ancestry column ({possible_anc_cols}). Defaulting to second column: '{anc_col}'")
            else:
                raise ValueError("Ancestry file does not appear to have a second column for population data.")

        LOGGER.info(f"Loading ancestry using ID column: '{id_col}' and Ancestry column: '{anc_col}'")

        ancestry_reader = csv.DictReader(ancestry_info, delimiter="\t")

        for indv in ancestry_reader:
            eid = str(indv[id_col])
            ancestry_val = indv[anc_col]

            ancestry_dict['all'].add(eid)
            if ancestry_val != "NA" and ancestry_val is not None:
                ancestry_dict.setdefault(ancestry_val, set()).add(eid)
                ancestry_counts[ancestry_val] = ancestry_counts.get(ancestry_val, 0) + 1

    # 3. Print Summary Table
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
    """
    This function loads the sample IDs file and returns a set of individual IDs.
    :param sample_ids_file: a file containing the sample IDs
    :return: a set of individual IDs
    """
    # Read overall list of individuals with data so we can subset the genetic data.
    with sample_ids_file.open('r') as wes_samp_file:
        # split() splits on any whitespace (tabs/spaces)
        # [0] grabs the first column (the Sample ID)
        return {line.strip().split()[0] for line in wes_samp_file if line.strip()}


def load_relatedness(relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
    """
    This function loads the relatedness file and returns a DataFrame containing only the related individuals.
    It normalizes the columns to 'ID1', 'ID2', 'Kinship' regardless of input format.

    :param relatedness:  a file containing the relatedness matrix table
    :param wes_samples: a set of individual IDs that are WES samples
    :return: a DataFrame containing only the related individuals with standardized columns
    """

    if relatedness.stat().st_size == 0:
        return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

    # Read the file
    # We use delim_whitespace to handle both tabs and spaces
    rel = pd.read_csv(relatedness, delim_whitespace=True)

    # Standardize Column Names
    # 1. Check for All of Us format (i.s, j.s, kin)
    if 'i.s' in rel.columns and 'j.s' in rel.columns and 'kin' in rel.columns:
        rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})

    # 2. Check for Standard KING/PLINK format (IID1, IID2, KINSHIP) or (ID1, ID2, Kinship)
    # Note: PLINK output often has #IID1
    rel.columns = rel.columns.str.replace('^#', '', regex=True)  # Remove leading # if present
    rel = rel.rename(columns={
        'IID1': 'ID1',
        'IID2': 'ID2',
        'KINSHIP': 'Kinship'
    })

    # Ensure IDs are strings to match wes_samples
    rel['ID1'] = rel['ID1'].astype(str)
    rel['ID2'] = rel['ID2'].astype(str)

    # Filter to only keep required columns
    if not {'ID1', 'ID2', 'Kinship'}.issubset(rel.columns):
        # Fallback: Assume the file is headerless and columns 0, 1, and the last column are what we want
        # This handles cases where headers are completely missing
        LOGGER.warning(f"Could not detect standard headers in relatedness file. Assuming columns 0, 1, and last.")
        rel = pd.read_csv(relatedness, delim_whitespace=True, header=None, skiprows=1)
        rel = rel.rename(columns={0: 'ID1', 1: 'ID2', rel.columns[-1]: 'Kinship'})
        rel['ID1'] = rel['ID1'].astype(str)
        rel['ID2'] = rel['ID2'].astype(str)

    # Keep only the standardized columns
    rel = rel[['ID1', 'ID2', 'Kinship']]

    # Filter based on WES samples
    return rel[(rel["ID1"].isin(wes_samples)) & (rel["ID2"].isin(wes_samples))]


def get_relateds_to_remove(rel: pd.DataFrame) -> Set[str]:
    """
    This function identifies individuals to remove from the relatedness DataFrame based on their relatedness pairs.
    :param rel: a pandas DataFrame containing the relatedness file
    :return: a set of individual IDs to remove from the relatedness DataFrame
    """
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
    This function writes ancestry-specific inclusion files for samples and uploads them to DNANexus.
    :param wes_samples: a set of sample IDs
    :param ancestry_dict: a dictionary with ancestry as keys and sets of individual IDs as values
    :param relateds_to_remove: a set of individual IDs to remove from the relatedness DataFrame
    :return: a list of DXFile objects representing the uploaded inclusion files
    """

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

        exporter = ExportFileHandler()

        # Manually export files using the ExportFileHandler (supports GCP upload)
        include_files.append(exporter.export_files(unrelated_path.name))
        include_files.append(exporter.export_files(related_path.name))

    return include_files


def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[
    Set[str], List]:
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
    # FIX: Changed -pfile to --bfile
    cmd = f"plink2 --missing 'variant-only' --bfile {merged_data_file.name} --out {missingness_db}"
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


def check_qc_other(wes_samples: set, snp_qc_file: Path, sample_qc_file: Path) -> Tuple[Path, Path]:
    """
    Filters samples and SNPs for non-DNAnexus datasets.

    For SAMPLES: treats 'sample_qc_file' as a BLACKLIST (samples to remove).
    For SNPS: treats 'snp_qc_file' as a WHITELIST (variants to keep).

    :param wes_samples: The set of all available samples (from sample_ids_file)
    :param snp_qc_file: A file containing the SNP QC information (Whitelist)
    :param sample_qc_file: A file containing the flagged samples (Blacklist)
    :return: a tuple of the pass SNPs file and the pass samples file
    """

    output_snps = Path("pass_snps.txt")
    output_samples = Path("pass_samples.txt")

    # 1. Handle SNP QC (Whitelist approach)
    with open(snp_qc_file, 'r') as f_in, output_snps.open('w') as f_out:
        for line in f_in:
            if line.strip():
                # Take first column if multiple exist
                f_out.write(line.strip().split()[0] + "\n")

    # 2. Handle Sample QC (Blacklist approach)
    # Read the flagged samples file (All of Us format: header with 's' column)
    flagged_samples = set()
    try:
        with open(sample_qc_file, 'r') as f:
            # Your file is TSV and has a header starting with 's'
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                if 's' in row:
                    flagged_samples.add(row['s'])
                else:
                    # Fallback if header is missing/different: assume first column
                    flagged_samples.add(list(row.values())[0])
    except Exception as e:
        LOGGER.warning(f"Could not parse flagged samples file as TSV: {e}. Trying simple list.")
        with open(sample_qc_file, 'r') as f:
            for line in f:
                flagged_samples.add(line.strip().split()[0])

    LOGGER.info(f"Identified {len(flagged_samples)} flagged samples to remove.")

    # 3. Subtract Blacklist from Whitelist
    final_samples = wes_samples - flagged_samples
    LOGGER.info(f"Retaining {len(final_samples)} samples after QC filtering.")

    # 4. Write the final pass_samples.txt
    with output_samples.open('w') as f_out:
        for sample in final_samples:
            f_out.write(f"{sample}\n")

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

    # Retain pass samples and pass SNPs
    # FIX: Changed -pfile to --bfile
    cmd = f"plink2 --mac 1 --bfile {merged_data_file.name} --make-bed --extract {pass_snps.name} " \
          f"--keep-fam {pass_samples.name} --out {merged_data_file.name}"
    cmd_executor.run_cmd_on_docker(cmd)
    # Generate a list of low MAC sites for BOLT
    # fail silently if no SNPs are found
    cmd = f"plink2 --bfile {merged_data_file.name} --max-mac 100 --write-snplist " \
          f"--out {merged_data_file.name}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd, ignore_error=True)

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

    # import UKBB KING matrix (or AoU Normalized matrix)
    # Note: load_relatedness has already normalized the columns to ID1, ID2, Kinship
    gt_matrix = pd.read_csv(rel_mtx, sep="\t", dtype={'ID1': str, 'ID2': str})

    # We no longer drop HetHet/IBS0 because they might not exist in AoU data.
    # We select only the columns we need.
    gt_matrix = gt_matrix[['ID1', 'ID2', 'Kinship']]

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
