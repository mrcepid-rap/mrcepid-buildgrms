import csv
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple, Union

import dxpy
import numpy as np
import pandas as pd
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler
from general_utilities.job_management.command_executor import build_default_command_executor


def ingest_resources(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict) -> Tuple[str, Path, Path]:
    """
    This function downloads the data we will need to run this module

    :param genetic_data_file: a file containing the genetic data file names and IDs
    :param sample_ids_file: a file containing the sample IDs
    :param ancestry_file: a file containing the ancestry sample IDs
    :return: a tuple of the genetic data file, sample IDs file, and ancestry file
    """
    # Ingest the UKBB plink files (this also includes relatedness and snp/sample QC files)
    genetic_data_file = InputFileHandler(genetic_data_file, download_now=True).get_file_handle()
    genetic_files = download_genetic_data(genetic_data_file)

    # Download a pre-computed sample IDs file.
    sample_ids_file = InputFileHandler(sample_ids_file, download_now=True).get_file_handle()

    # Download wba file:
    ancestry_file = InputFileHandler(ancestry_file, download_now=True).get_file_handle()

    return genetic_files, sample_ids_file, ancestry_file


# Function simply merges all autosomal files together
def merge_plink_files(genetic_files: str) -> str:
    """
    This function merges all autosomal files together. It is assumed that the files are named in a way that
    they can be matched by the prefix 'genetic_files.name' and that they are all in the same directory.

    :param genetic_files: the name of the genetic files to be merged
    :return: the name of the merged file
    """

    genetic_files = Path.cwd() / genetic_files
    output_stub = "Autosomes"
    cmd_executor = build_default_command_executor()

    # Merge autosomal PLINK files together:
    # List all files matching the prefix
    matching_files = list(Path('.').glob(f"{genetic_files.name}*"))

    # Write the matching base names (with 'test/' prepended and without extensions) to merge_list.txt
    with open('merge_list.txt', 'w') as merge_list:
        # Use a set to remove duplicates
        unique_base_names = {os.path.splitext(file.name)[0] for file in matching_files}
        for base_name in sorted(unique_base_names):
            merge_list.write(f"test/{base_name}\n")
        # remove duplicates
    cmd = f"plink2 --pmerge-list /test/merge_list.txt bfile --out /test/{output_stub}"
    cmd_executor.run_cmd_on_docker(cmd)

    return output_stub


def download_genetic_data(genetic_data_file: Path) -> str:
    """
    This function downloads the genetic data files using input coordinates

    :param genetic_data_file: a file containing the genetic data file names and IDs
    :return: stem of one of the files for downstream use
    """
    # we should have two columns in the genetic input file
    # the first column should be the filename
    # the second column should be the file ID
    # in total there should be 22 sets of .bed .bim .fam files
    with open(genetic_data_file, 'r') as file:
        lines = file.readlines()
        # check we have 22 chromosomes in sets of 3
        if len(lines) != 66:
            raise ValueError("The file must contain exactly 66 lines, 3 for each chromosome.")

        for line in lines:
            columns = line.strip().split()
            # check we have two columns, one being the filename and one being the file ID
            if len(columns) != 2:
                raise ValueError(f"Each line must have exactly two columns. Invalid line: {line.strip()}")

            # Check that the second column contains valid file extensions
            valid_extensions = {'.bed', '.bim', '.fam'}
            if not any(columns[0].endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Invalid file format in second column: {columns[0]}")

            # download the files
            file = line.strip().split()[1]
            output = InputFileHandler(file, download_now=True).get_file_handle()

    # return the stem of one of the files so we can use it downstream
    return output.stem


def calculate_relatedness(genetic_data_file: str) -> Path:
    """
    This function calculates the relatedness of the samples in the genetic data file

    :param genetic_data_file: a path to the genetic data file
    :return: a path to the relatedness file matrix
    """

    cmd_executor = build_default_command_executor()
    genetic_data_file = Path.cwd() / genetic_data_file
    relatedness_db = "relatedness_table"

    # Calculate relatedness using KING:
    cmd = f"plink2 --pfile /test/{genetic_data_file.name} --make-king-table --out /test/{relatedness_db}"
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


# This is a helper function to get_individuals()
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


# This function generates a list of related individuals and then generates various exclusions lists based on
# relatedness and ancestry
def get_individuals(sample_ids_file: Path, ancestry_file: Path, relatedness: Path) -> Tuple[
    Set[str], List[dxpy.DXFile]]:
    """
    This function generates a list of related individuals and then generates various exclusions lists based on
    relatedness and ancestry

    :param sample_ids_file: a file containing the sample IDs
    :param ancestry_file: a file containing the ancestry sample IDs and ancestry information
    :param relatedness: a file containing the relatedness matrix table
    :return: a tuple of the WES samples and a list of files to be uploaded to DNANexus
    """
    # Get WBA individuals
    # - Need to be able to generate some separate list of individuals
    # - This is the list generated by Felix to be a bit more inclusive
    with Path(ancestry_file).open(mode='r') as ancestry_file:
        ancestry_dict: Dict[str, Set[str]] = {'all': set()}
        ancestry_reader = csv.DictReader(ancestry_file, delimiter="\t")
        for indv in ancestry_reader:
            if indv['ancestry'] != "NA":
                if indv['ancestry'] in ancestry_dict:
                    ancestry_dict[indv['ancestry']].add(str(indv['n_eid']))
                else:
                    ancestry_dict[indv['ancestry']] = {str(indv['n_eid'])}

            # Create an all category as well:
            ancestry_dict['all'].add(str(indv['n_eid']))

        ancestry_file.close()

    # Read overall list of individuals with WES data so we can subset the genetic data.
    wes_samp_file = open(sample_ids_file, 'r')
    wes_samples = set()
    for eid in wes_samp_file:
        eid = eid.rstrip()
        wes_samples.add(str(eid))

    # Calculate relateds:
    # Read the relatedness file in as a pandas DataFrame
    # dtype sets eids as characters
    # ID1 and ID2 are two spearate individuals that are related according to some kinship value
    rel = pd.read_csv(relatedness.name,
                      delim_whitespace=True,
                      dtype={'ID1': np.str_, 'ID2': np.str_})

    # Remove individuals from the relatedness DataFrame not in WES data:
    rel = rel[rel['ID1'].isin(wes_samples)]
    rel = rel[rel['ID2'].isin(wes_samples)]

    # The relatedenss list is passed to the select_related_individuals() function for the first time so that we can
    # just calculate the number of times each individual occurs in the rel file after we limit to WES samples
    # parameter 1 is a pandas DataFrame
    # parameter 2 is a list of individuals we want to remove from parameter 1
    returned = select_related_individual(rel, [])  # use an empty list first since we are just calculating totals
    rel = returned['rel']
    rel_totals = returned['rel_totals']

    relateds_to_remove = set()

    # We now iterate until we have no more related pairs in the relatedness file (rel)
    while len(rel_totals) > 0:
        # Remove the individual with the most relatedness pairs
        samp_to_remove = rel_totals.iloc[len(rel_totals) - 1].name
        relateds_to_remove.add(samp_to_remove)  # and add them to the list of related individuals we want to exclude

        # And then remove that person from rel and recalculate per-individual totals and loop again
        returned = select_related_individual(rel, [samp_to_remove])
        rel = returned['rel']
        rel_totals = returned['rel_totals']

    # Get lists of WES samples to include specific to certain ancestries:
    include_files = []

    for ancestry in ancestry_dict.keys():

        pass_samples = wes_samples.intersection(ancestry_dict[ancestry])  # gets WES samples that are ancestry-specific
        pass_samples = pass_samples.difference(relateds_to_remove)  # gets unrelated samples

        # Write ancestry-specific exclusion lists, relatedness, and combo of the two:
        # 1. list of WES non-ancestry or related individuals
        # 2. list of ancestry-specific individuals with WES
        # 3. list of related individuals with WES
        unrelated_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Unrelated.txt')
        related_path = Path(f'INCLUDEFOR_{ancestry.upper()}_Related.txt')

        with unrelated_path.open('w') as ancestry_unrelated_inclusion_list, \
                related_path.open('w') as ancestry_inclusion_list:

            # This writes to each list based on a set of requirements
            for samp in wes_samples:
                if samp in pass_samples:
                    ancestry_unrelated_inclusion_list.write(samp + "\n")
                if samp in ancestry_dict[ancestry]:
                    ancestry_inclusion_list.write(samp + "\n")

            ancestry_unrelated_inclusion_list.close()
            ancestry_inclusion_list.close()

            # Upload to DNANexus
            include_files.extend(
                [dxpy.upload_local_file(unrelated_path.name), dxpy.upload_local_file(related_path.name)])

    return wes_samples, include_files


# Calculate per-snp missingness for filtering purposes:
def calculate_missingness(merged_filename: str) -> dict:
    cmd_executor = build_default_command_executor()
    merged_data_file = Path.cwd() / merged_filename
    missingness_db = "missingness_out"

    # First generate missingness information for all SNPs:
    cmd = f"plink2 --missing 'variant-only' --pfile /test/{merged_data_file.name} --out /test/{missingness_db}"
    cmd_executor.run_cmd_on_docker(cmd)

    # Then read as a pandas DataFrame:
    missingness_qc = csv.DictReader(open(f"{missingness_db}.vmiss", 'r'),
                                    delimiter="\t")
    # And convert to a dictionary with format SNP ID : missingness
    missingness = dict()
    for snp in missingness_qc:
        missingness[snp['ID']] = float(snp['F_MISS'])

    return missingness


# Check per-SNP and per-sample quality control
def check_qc_ukb(wes_samples: set, missingness: dict, ukb_snp_qc: Path, ukb_snps_qc_v2: Path) -> Tuple[Path, Path]:
    """
    This function checks the quality control of the SNPs and samples in the genetic data file

    :param wes_samples: a set of WES samples
    :param missingness: a dictionary of SNP IDs and their missingness values
    :param ukb_snp_qc: a file containing the SNP QC information
    :param ukb_snps_qc_v2: a file containing the SNP QC information version 2
    :return: a tuple of the pass SNPs file and the pass samples file
    """

    cmd_executor = build_default_command_executor()
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
    cmd = f'paste -d " " {ukb_snps_qc_v2} > ukb_sqc_v2.with_fam.txt'
    cmd_executor.run_cmd_on_docker(cmd)

    # Check sample QC files:
    # Here generating a header that mashes together the two files above
    smp_qc_header = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1',
                     'affyID1', 'affyID2', 'array', 'batch2', 'plate', 'well',
                     'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender',
                     'x.int', 'y.int', 'plate.sub', 'well.sub', 'missing.rate',
                     'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship',
                     'excl.kinship', 'excess.relatives', 'in.wba', 'used.pc']
    smp_qc_header.extend(["PC%d" % item for item in range(1, 41)])
    smp_qc_header.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])

    smp_qc = csv.DictReader(open('ukb_sqc_v2.with_fam.txt', 'r'), delimiter=" ", fieldnames=smp_qc_header)
    # write pass IDs as a file:
    pass_samples = Path("pass_samples.txt")
    wr_file = open(pass_samples, 'w')

    # Retain samples that are:
    # 1. In the WES samples
    # 2. Are not missingness outliers
    # 3. Are included in autosomal phasing
    for sample in smp_qc:
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


# Now apply filtering based on lists that we made above
def filter_plink(merged_filename: str, pass_snps: Path, pass_samples: Path = None) -> Tuple[Path, Path]:
    """
    This function filters the merged plink file based on the pass SNPs and pass samples files

    :param merged_filename: a file containing the merged plink file
    :param pass_snps: a file containing the pass SNPs
    :param pass_samples: a file containing the pass samples
    :return: a path to the filtered merged plink file and a path to the low MAC SNPs file
    """

    cmd_executor = build_default_command_executor()
    merged_data_file = Path.cwd() / merged_filename
    snplist = Path(merged_data_file.name).with_suffix(".low_MAC.snplist")

    # Retain pass samples and pass SNPs
    cmd = f"plink2 --mac 1 --pfile /test/{merged_data_file.name} --make-bed --extract /test/{pass_snps.name} " \
          f"--keep-fam /test/{pass_samples.name} --out /test/{merged_data_file.name}"
    cmd_executor.run_cmd_on_docker(cmd)
    # Generate a list of low MAC sites for BOLT
    cmd = f"plink2 --bfile /test/{merged_data_file.name} --max-mac 100 --write-snplist " \
          f"--out /test/{merged_data_file.name}.low_MAC"
    cmd_executor.run_cmd_on_docker(cmd)

    return merged_data_file, snplist


# This is a helper function for building GRMs to ensure that the resulting matrix is lower-left
def column_swap(col1, col2):
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


# We use the KING-relate derived relatedness information for our GRM. Just need to convert it into a format that
# SAIGE and STAAR can use...
def make_grm(wes_samples: set, rel_mtx: Path) -> Tuple[Path, Path]:
    """
    This function generates a GRM from the WES samples and the relatedness matrix

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
