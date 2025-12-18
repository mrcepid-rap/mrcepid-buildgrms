import dxpy
from general_utilities.import_utils.file_handlers.export_file_handler import ExportFileHandler
from general_utilities.import_utils.file_handlers.input_file_handler import InputFileHandler, FileType
from general_utilities.mrc_logger import MRCLogger

from buildgrms.grm_tools.grm_toolkit import (
    ingest_resources, merge_plink_files, get_individuals, calculate_missingness,
    check_qc_ukb, filter_plink, make_grm, calculate_relatedness, check_qc_other
)

LOGGER = MRCLogger().get_logger()


@dxpy.entry_point('main')
def main(genetic_data_file: dict, sample_ids_file: dict, ancestry_file: dict,
         snp_qc: dict = None, sample_qc: dict = None, ukb_snp_qc: dict = None,
         ukb_snps_qc_v2: dict = None, relatedness_file: dict = None):
    """
    Execution entry point for building GRMs.
    """
    LOGGER.info("Starting BuildGRMs Pipeline.")

    # 1. Ingest Data
    genetic_files, sample_ids_handle, ancestry_handle, relatedness_handle = ingest_resources(
        genetic_data_file, sample_ids_file, ancestry_file, relatedness_file
    )

    if (snp_qc and sample_qc) is None and (ukb_snp_qc and ukb_snps_qc_v2) is None:
        raise ValueError("Insufficient QC files provided. Check your inputs.")

    # 2. Merge Genotypes
    merged_filename = merge_plink_files(genetic_files)

    # 3. Handle Relatedness
    if relatedness_handle is None:
        relatedness_handle = calculate_relatedness(merged_filename)

    # 4. Generate Inclusion Lists
    # Returns list of GCS paths to uploaded inclusion files
    samples, include_files = get_individuals(sample_ids_handle, ancestry_handle, relatedness_handle)

    missingness = calculate_missingness(merged_filename)

    # 5. Execute QC
    if InputFileHandler(genetic_data_file).get_file_type() == FileType.DNA_NEXUS_FILE:
        ukb_snp_qc_handle = InputFileHandler(ukb_snp_qc, download_now=True).get_file_handle()
        ukb_sqc_v2_handle = InputFileHandler(ukb_snps_qc_v2, download_now=True).get_file_handle()
        pass_snps, pass_samples = check_qc_ukb(samples, missingness, ukb_snp_qc_handle, ukb_sqc_v2_handle)
    else:
        snp_qc_handle = InputFileHandler(snp_qc, download_now=True).get_file_handle()
        sample_qc_handle = InputFileHandler(sample_qc, download_now=True).get_file_handle()
        pass_snps, pass_samples = check_qc_other(wes_samples=samples, snp_qc_file=snp_qc_handle,
                                                 sample_qc_file=sample_qc_handle)

    # 6. Filter PLINK Data
    # 'Arrays_QCd' is used as the prefix to prevent overwriting the input 'arrays.bed'
    filtered_prefix = "Arrays_QCd"
    final_genetic_file, snplist = filter_plink(
        merged_filename=merged_filename,
        pass_snps=pass_snps,
        pass_samples=pass_samples,
        output_prefix=filtered_prefix
    )

    # 7. Generate GRM
    grm, grm_sample = make_grm(samples, relatedness_handle)

    # 8. Export Results
    exporter = ExportFileHandler()
    output = {
        'output_pgen': exporter.export_files(f'{filtered_prefix}.bed'),
        'output_psam': exporter.export_files(f'{filtered_prefix}.fam'),
        'output_pvar': exporter.export_files(f'{filtered_prefix}.bim'),
        'inclusion_lists': include_files,  # Direct pass-through; do not re-export
        'grm': exporter.export_files(grm.name),
        'grm_samp': exporter.export_files(grm_sample.name),
        'snp_list': exporter.export_files(snplist.name)
    }

    LOGGER.info("Pipeline completed successfully. Output registered.")
    return output


if __name__ == "__main__":
    dxpy.run()