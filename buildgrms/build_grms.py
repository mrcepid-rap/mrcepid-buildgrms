#!/usr/bin/env python
import dxpy
import subprocess  # <--- ADD THIS IMPORT
import os  # <--- ADD THIS IMPORT
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler, FileType
from general_utilities.mrc_logger import MRCLogger

from buildgrms.grm_tools.grm_toolkit import ingest_resources, merge_plink_files, get_individuals, calculate_missingness, \
    check_qc_ukb, filter_plink, make_grm, calculate_relatedness, check_qc_other

LOGGER = MRCLogger().get_logger()


@dxpy.entry_point('main')
def main(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict, snp_qc: dict = None,
         sample_qc: dict = None, ukb_snp_qc: dict = None, ukb_snps_qc_v2: dict = None, relatedness_file: dict = None):
    # Grab plink files and sample exclusion lists
    genetic_files, sample_ids_file, ancestry_file, relatedness = ingest_resources(genetic_data_file,
                                                                                  sample_ids_file,
                                                                                  ancestry_file,
                                                                                  relatedness_file)

    if not ((snp_qc and sample_qc) or (ukb_snp_qc and ukb_snps_qc_v2)):
        raise ValueError(
            "Either both SNP QC and Sample QC files, or both UKB SNP QC files must be provided. Please check your input files.")

    # merge autosomal plink files together
    merged_filename = merge_plink_files(genetic_files)

    # get the relatedness matrix
    if relatedness_file is None:
        # if it has not been provided, calculate it
        LOGGER.info(f"The relatedness file was not provided, calculating it from the merged plink files.")
        relatedness = calculate_relatedness(merged_filename)

    # Decide on a set of individuals to extract from plink files and get per-SNP missingness:
    samples, include_files = get_individuals(sample_ids_file, ancestry_file, relatedness)
    missingness = calculate_missingness(merged_filename)

    # Check UKB Internal SNP and Sample QC:
    if InputFileHandler(genetic_data_file).get_file_type() == FileType.DNA_NEXUS_FILE:
        ukb_snp_qc = InputFileHandler(ukb_snp_qc, download_now=True).get_file_handle()
        ukb_snps_qc_v2 = InputFileHandler(ukb_snps_qc_v2, download_now=True).get_file_handle()
        pass_snps, pass_samples = check_qc_ukb(samples, missingness, ukb_snp_qc, ukb_snps_qc_v2)
    elif InputFileHandler(genetic_data_file).get_file_type() == FileType.LOCAL_PATH:
        # Check SNP and Sample QC for other datasets:
        snp_qc = InputFileHandler(snp_qc, download_now=True).get_file_handle()
        sample_qc = InputFileHandler(sample_qc, download_now=True).get_file_handle()
        pass_snps, pass_samples = check_qc_other(snp_qc_file=snp_qc, sample_qc_file=sample_qc)
    else:
        raise ValueError("Could not perform variant and/or sample QC on the input data. Check your input files.")

    # ========================== DEBUGGING BLOCK START ==========================
    # PASTE THIS RIGHT HERE (After the QC checks, before filter_plink)

    print("\n--- DEBUGGING ID MISMATCH ---")

    print(f"1. Checking 'pass_samples.txt' path: {pass_samples}")
    if not pass_samples.exists() or pass_samples.stat().st_size == 0:
        print("!!! ALARM: pass_samples.txt is EMPTY. check_qc_ukb failed to write any samples.")
    else:
        print("First 10 lines of pass_samples.txt:")
        subprocess.run(f"head -n 10 {pass_samples}", shell=True)

    print(f"\n2. Checking '{merged_filename}.fam' (PLINK input)")
    # This checks the actual FAM file PLINK reads
    subprocess.run(f"head -n 10 {merged_filename}.fam", shell=True)

    print("\n3. Checking 'samples' set (Filter List)")
    print(f"Total WES samples loaded: {len(samples)}")
    print(f"Sample of first 5 WES IDs in memory: {list(samples)[:5]}")

    # Check the intermediate QC paste file (only relevant if we ran UKB QC)
    if os.path.exists("ukb_sqc_v2_with_fam.txt"):
        print("\n4. Checking intermediate file 'ukb_sqc_v2_with_fam.txt'")
        subprocess.run("head -n 10 ukb_sqc_v2_with_fam.txt", shell=True)

    print("--- END DEBUGGING ---\n")
    # ========================== DEBUGGING BLOCK END ============================

    # Filter plink files to something we can use for BOLT and making GRMs
    # do not filter for WBA/Relateds, only for QC fail samples and SNPs
    final_genetic_file, snplist = filter_plink(merged_filename=merged_filename, pass_snps=pass_snps,
                                               pass_samples=pass_samples)

    # Now here we generate GRMs for tools that require it (SAIGE & STAAR):
    # BOLT and REGENIE use raw PLINK files, so do not need it here:
    grm, grm_sample = make_grm(samples, relatedness)

    # Have to do 'upload_local_file' to make sure the new file is registered with dna nexus
    output = {'output_pgen': dxpy.dxlink(dxpy.upload_local_file(f'{final_genetic_file.name}.bed')),
              'output_psam': dxpy.dxlink(dxpy.upload_local_file(f'{final_genetic_file.name}.fam')),
              'output_pvar': dxpy.dxlink(dxpy.upload_local_file(f'{final_genetic_file.name}.bim')),
              'inclusion_lists': [dxpy.dxlink(item) for item in include_files],
              'grm': dxpy.dxlink(dxpy.upload_local_file(grm.name)),
              'grm_samp': dxpy.dxlink(
                  dxpy.upload_local_file(grm_sample.name)),
              'snp_list': dxpy.dxlink(dxpy.upload_local_file(snplist.name))}

    return output


dxpy.run()