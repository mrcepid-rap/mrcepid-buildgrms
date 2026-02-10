import csv
import subprocess
from pathlib import Path
from typing import Set, Dict, Tuple

from general_utilities.mrc_logger import MRCLogger

LOGGER = MRCLogger().get_logger()


class PlatformQC:
    """A class to encapsulate platform-specific QC steps."""

    @staticmethod
    def run_ukb_qc(wes_samples: Set[str], missingness: Dict[str, float], ukb_snp_qc: Path, ukb_snps_qc_v2: Path) \
            -> Tuple[Path, Path]:
        """Performs the QC process specific to UK Biobank datasets.

        This involves SNP quality control based on array presence, chromosome, missingness,
        and batch QC, followed by sample quality control using UKB's sample QC file.

        :param wes_samples: A set of WES sample IDs.
        :param missingness: A dictionary of variant missingness rates.
        :param ukb_snp_qc: Path to the UKB SNP QC file.
        :param ukb_snps_qc_v2: Path to the UKB sample QC file (version 2).
        :return: A tuple containing paths to the passing SNPs file and passing samples file.
        """
        pass_snps_file = Path("pass_snps.txt")
        pass_samples = Path("pass_samples.txt")

        # 1. SNP Quality Control
        with open(ukb_snp_qc, 'r') as snp_qc_source_file, pass_snps_file.open('w') as passing_snps_output_file:
            reader = csv.DictReader(snp_qc_source_file, delimiter=" ")
            batch_qc_columns = [f"Batch_b{x:03d}_qc" for x in range(1, 96)] + [f"UKBiLEVEAX_b{x}_qc" for x in
                                                                               range(1, 12)]
            for snp in reader:
                # Apply filters: SNP must be on array '2', be autosomal (chr <= 22), and have <5% missingness
                missingness_value = missingness.get(snp['rs_id'])
                if missingness_value is not None and snp['array'] == "2" and int(snp['chromosome']) <= 22 and missingness_value < 0.05:
                    # Additionally, check if the SNP passes all batch-specific QC flags
                    if all(snp[a] == "1" for a in batch_qc_columns):
                        passing_snps_output_file.write(snp['rs_id'] + "\n")

        # 2. Sample Quality Control
        ukb_sqc_v2_with_fam = Path("ukb_sqc_v2_with_fam.txt")

        # Find a .fam file in the current directory, which is necessary to align QC data
        try:
            fam_file = list(Path('.').glob('*.fam'))[0]
        except IndexError:
            raise FileNotFoundError("No .fam file found to align QC data! This is required for UKB.")

        LOGGER.info(f"Aligning QC metadata using FAM file: {fam_file.name}")

        # Paste FAM file and UKB sample QC file together to align sample IDs
        subprocess.run(f'paste -d " " {fam_file.name} {ukb_snps_qc_v2} > {ukb_sqc_v2_with_fam}', shell=True)

        # Read the aligned file using a predefined header
        combined_header = ['ID1', 'ID2', 'null1', 'null2', 'fam.gender', 'batch1', 'affyID1', 'affyID2', 'array',
                           'batch2', 'plate',
                           'well', 'call.rate', 'dQC', 'dna.conc', 'sub.gender', 'inf.gender', 'x.int', 'y.int',
                           'plate.sub', 'well.sub',
                           'missing.rate', 'het', 'het.pc.corr', 'het.missing.outliers', 'aneuploidy', 'in.kinship',
                           'excl.kinship',
                           'excess.relatives', 'in.wba', 'used.pc']
        combined_header.extend([f"PC{x}" for x in range(1, 41)])
        combined_header.extend(['in.phasing.auto', 'in.phasing.x', 'in.phasing.xy'])

        with open(ukb_sqc_v2_with_fam, 'r') as pasted_qc_source_file, pass_samples.open(
                'w') as passing_samples_output_file:
            reader = csv.DictReader(pasted_qc_source_file, delimiter=" ", fieldnames=combined_header)
            for sample_row in reader:
                # Samples must be in the WES samples set and pass various QC flags
                if sample_row['ID2'] in wes_samples:  # Match by Individual ID (IID)
                    if (sample_row['het.missing.outliers'] == "0"
                            and sample_row['in.phasing.auto'] == "1"
                            and sample_row['in.phasing.x'] == "1"
                            and sample_row['in.phasing.xy'] == "1"):
                        # Write Family ID and Individual ID for PLINK '--keep' flag
                        passing_samples_output_file.write(f"{sample_row['ID1']} {sample_row['ID2']}\n")

        return pass_snps_file, pass_samples