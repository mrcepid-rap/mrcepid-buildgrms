#!/usr/bin/env python
# mrcepid-collecthsmetrics 0.0.1
# Generated by dx-app-wizard.
#
# Basic execution pattern: Your app will run on a single machine from
# beginning to end.
#
# See https://documentation.dnanexus.com/developer for documentation and
# tutorials on how to modify this file.
#
# DNAnexus Python Bindings (dxpy) documentation:
#   http://autodoc.dnanexus.com/bindings/python/current/
from pathlib import Path
from typing import List, Dict, Set, Tuple

import dxpy
import subprocess
import csv
import pandas as pd
import numpy as np


# This function runs a command on an instance, either with or without calling the docker instance we downloaded
# By default, commands are not run via Docker, but can be changed by setting is_docker = True
def run_cmd(cmd: str, is_docker: bool = False, stdout_file: str = None, print_cmd = False) -> None:

    # -v here mounts a local directory on an instance (in this case the home dir) to a directory internal to the
    # Docker instance named /test/. This allows us to run commands on files stored on the AWS instance within Docker.
    # This looks slightly different from other versions of this command I have written as I needed to write a custom
    # R script to run STAAR. That means we have multiple mounts here to enable this code to find the script.
    if is_docker:
        cmd = "docker run " \
              "-v /home/dnanexus:/test " \
              "-v /usr/bin/:/prog " \
              "egardner413/mrcepid-burdentesting " + cmd

    if print_cmd:
        print(cmd)

    # Standard python calling external commands protocol
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if stdout_file is not None:
        with open(stdout_file, 'w') as stdout_writer:
            stdout_writer.write(stdout.decode('utf-8'))
        stdout_writer.close()

    # If the command doesn't work, print the error stream and close the AWS instance out with 'dxpy.AppError'
    if proc.returncode != 0:
        print("The following cmd failed:")
        print(cmd)
        print("STDOUT follows\n")
        print(stdout.decode('utf-8'))
        print("STDERR follows\n")
        print(stderr.decode('utf-8'))
        raise dxpy.AppError("Failed to run properly...")


# This is just to compartmentalise the collection of all the resources I need for this task and
# get them into the right place
def ingest_resources(genetic_data_folder: str, sample_ids_file: dict, ancestry_file: dict) -> None:

    # Bring our docker image into our environment so that we can run commands we need:
    cmd = "docker pull egardner413/mrcepid-burdentesting:latest"
    run_cmd(cmd)

    # Infer the current project ID:
    project_id = dxpy.PROJECT_CONTEXT_ID

    # Ingest the UKBB plink files (this also includes relatedness and snp/sample QC files)
    # This is done __ASSUMING__ the file path to the bulk genotype data HAS NOT CHANGED. This can be changed from the
    # default with input settings.
    dxpy.download_folder(project=project_id,
                         destdir='genotypes/',
                         folder=genetic_data_folder)

    # Download a pre-computed sample IDs file.
    dxpy.download_dxfile(dxid=dxpy.DXFile(sample_ids_file).get_id(),
                         filename="wes_samples.txt")

    # Download wba file:
    dxpy.download_dxfile(dxid=dxpy.DXFile(ancestry_file).get_id(),
                         filename="ancestry.txt")


# Function simply merges all autosomal files together
def merge_plink_files() -> None:
    # Merge autosomal PLINK files together:
    with open('merge_list.txt', 'w') as merge_list:
        for chrom in range(1, 23):
            merge_list.write("/test/genotypes/ukb22418_c%d_b0_v2\n" % (chrom))
        merge_list.close()
    cmd = "plink2 --pmerge-list /test/merge_list.txt bfile --out /test/UKBB_500K_Autosomes"
    run_cmd(cmd, True)


# This is a helper function to get_individuals()
def select_related_individual(rel: pd.DataFrame, samples_to_exclude: list) -> dict:

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
    rel_totals = rel_ids.groupby('ID').agg(total = ('dummy','sum'))
    # ... and then we sort it by that value
    rel_totals = rel_totals.sort_values(by = 'total')

    return {'rel': rel, 'rel_totals': rel_totals}


# This function generates a list of related individuals and then generates various exclusions lists based on
# relatedness and ancestry
def get_individuals() -> Tuple[Set[str], List[dxpy.DXFile]]:

    # Get WBA individuals
    # - Need to be able to generate some separate list of individuals
    # - This is the list generated by Felix to be a bit more inclusive
    with Path('ancestry.txt').open(mode='r') as ancestry_file:
        ancestry_dict: Dict[str, Set[str]] = {'all': set()}
        ancestry_reader = csv.DictReader(ancestry_file, delimiter=" ")
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
    wes_samp_file = open('wes_samples.txt', 'r')
    wes_samples = set()
    for eid in wes_samp_file:
        eid = eid.rstrip()
        wes_samples.add(str(eid))

    # Calculate relateds:
    # Read the relatedness file in as a pandas DataFrame
    # dtype sets eids as characters
    # ID1 and ID2 are two spearate individuals that are related according to some kinship value
    rel = pd.read_csv("genotypes/ukb_rel.dat",
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
        include_files.extend([dxpy.upload_local_file(unrelated_path.name), dxpy.upload_local_file(related_path.name)])

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

    return wes_samples, include_files  # Return a memory-stored set of samples for later


# Calculate per-snp missingness for filtering purposes:
def calculate_missingness() -> dict:

    # First generate missingness information for all SNPs:
    cmd = "plink2 --missing 'variant-only' --pfile /test/UKBB_500K_Autosomes --out /test/UKBB_500K_Autosomes"
    run_cmd(cmd, True)

    # Then read as a pandas DataFrame:
    missingness_qc = csv.DictReader(open('UKBB_500K_Autosomes.vmiss'),
                                    delimiter="\t")
    # And convert to a dictionary with format SNP ID : missingness
    missingness = dict()
    for snp in missingness_qc:
        missingness[snp['ID']] = float(snp['F_MISS'])

    return missingness


# Check per-SNP and per-sample quality control
def check_QC(wes_samples: set, missingness: dict) -> None:

    # Read in UKBiobank provided quality control for SNPs
    snp_qc = csv.DictReader(open('genotypes/ukb_snp_qc.txt', 'r'), delimiter=" ")
    # Create a simple list of SNPs that pass our QC
    pass_snps = open('pass_snps.txt', 'w')

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
    cmd = 'paste -d " " genotypes/ukb22418_c22_b0_v2.fam genotypes/ukb_sqc_v2.txt > genotypes/ukb_sqc_v2.with_fam.txt'
    run_cmd(cmd)

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

    smp_qc = csv.DictReader(open('genotypes/ukb_sqc_v2.with_fam.txt', 'r'), delimiter=" ", fieldnames=smp_qc_header)
    # write pass IDs as a file:
    wr_file = open('samp_pass_gt_qc.txt', 'w')

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


# Now apply filtering based on lists that we made above
def filter_plink() -> None:

    # Retain pass samples and pass SNPs
    cmd = "plink2 --mac 1 --pfile /test/UKBB_500K_Autosomes --make-bed --extract /test/pass_snps.txt " \
          "--keep-fam /test/samp_pass_gt_qc.txt --out /test/UKBB_470K_Autosomes_QCd"
    run_cmd(cmd, True)
    # Generate a list of low MAC sites for BOLT
    cmd = "plink2 --bfile /test/UKBB_470K_Autosomes_QCd --max-mac 100 --write-snplist " \
          "--out /test/UKBB_470K_Autosomes_QCd.low_MAC"
    run_cmd(cmd, True)


# This is a helper function for building GRMs to ensure that the resulting matrix is lower-left
def column_swap(col1, col2):
    if col1 < col2:
        return col2, col1
    else:
        return col1, col2


# We use the KING-relate derived relatedness information for our GRM. Just need to convert it into a format that
# SAIGE and STAAR can use...
def make_GRM(wes_samples: set) -> None:

    # Construct a pd.DataFrame of wes_samples for merging purposes
    wes_samples_sorted = sorted(wes_samples)
    wes_samples_sorted = pd.DataFrame(data={'ID1': wes_samples_sorted,
                                            'ID2': wes_samples_sorted,
                                            'Kinship': [0.5] * len(wes_samples_sorted)})
    wes_samples_sorted['column1'] = wes_samples_sorted.index + 1
    wes_samples_sorted['column2'] = wes_samples_sorted.index + 1

    # import UKBB KING matrix
    gt_matrix = pd.read_csv("genotypes/ukb_rel.dat", sep=" ", dtype={'ID1': str, 'ID2': str})
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
    with open('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx', 'w') as matrix:
        matrix.write('%%MatrixMarket matrix coordinate real symmetric\n')
        matrix.write('{n_samps} {n_samps} {n_rows}\n'.format(n_samps=len(wes_samples_sorted), n_rows=len(gt_matrix)))
        for row in gt_matrix.iterrows():
            ret = matrix.write('{col1} {col2} {kin}\n'.format(col1=row[1]['column1'], col2=row[1]['column2'], kin=row[1]['Kinship']))
        matrix.close()

    with open('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt', 'w') as matrix_samples:
        for row in wes_samples_sorted.iterrows():
            ret = matrix_samples.write('{samp}\n'.format(samp=row[1]['ID1']))
        matrix_samples.close()


@dxpy.entry_point('main')
def main(genetic_data_folder: str, sample_ids_file: dict, ancestry_file: dict):

    # Grab plink files and sample exclusion lists
    ingest_resources(genetic_data_folder, sample_ids_file, ancestry_file)

    # merge autosomal plink files together
    merge_plink_files()

    # Decide on a set of individuals to extract from plink files and get per-SNP missingness:
    wes_samples, include_files = get_individuals()
    missingness = calculate_missingness()

    # Check UKB Internal SNP and Sample QC:
    check_QC(wes_samples, missingness)

    # Filter plink files to something we can use for BOLT and making GRMs
    # do not filter for WBA/Relateds, only for QC fail samples and SNPs
    filter_plink()

    # Now here we generate GRMs for tools that require it (SAIGE & STAAR):
    # BOLT and REGENIE use raw PLINK files, so do not need it here:
    make_GRM(wes_samples)

    # Have to do 'upload_local_file' to make sure the new file is registered with dna nexus
    output = {'output_pgen': dxpy.dxlink(dxpy.upload_local_file('UKBB_470K_Autosomes_QCd.bed')),
              'output_psam': dxpy.dxlink(dxpy.upload_local_file('UKBB_470K_Autosomes_QCd.fam')),
              'output_pvar': dxpy.dxlink(dxpy.upload_local_file('UKBB_470K_Autosomes_QCd.bim')),
              'inclusion_lists': [dxpy.dxlink(item) for item in include_files],
              'grm': dxpy.dxlink(dxpy.upload_local_file('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx')),
              'grm_samp': dxpy.dxlink(dxpy.upload_local_file('sparseGRM_470K_Autosomes_QCd.sparseGRM.mtx.sampleIDs.txt')),
              'snp_list': dxpy.dxlink(dxpy.upload_local_file('UKBB_470K_Autosomes_QCd.low_MAC.snplist'))}

    return output

dxpy.run()