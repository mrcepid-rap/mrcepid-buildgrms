from pathlib import Path
from typing import List, Dict, Set

import pandas as pd

from general_utilities.job_management.command_executor import build_default_command_executor, CommandExecutor
from general_utilities.mrc_logger import MRCLogger

CMD_EXECUTOR = build_default_command_executor()
LOGGER = MRCLogger().get_logger()


class RelatednessCalculator:

    def __init__(self, genetic_data_file: str, cmd_executor: CommandExecutor = CMD_EXECUTOR):
        self._genetic_data_file = genetic_data_file
        self._cmd_executor = cmd_executor
        self._relatedness_db = "relatedness_table"
        self._genetic_data_path = Path.cwd() / self._genetic_data_file

    def calculate_relatedness(self) -> Path:
        """Calculate the relatedness matrix using the KING-table algorithm via PLINK2.

        :return: Path to the generated KING-table kinship file.
        """
        LOGGER.info("Calculating relatedness matrix (KING)...")

        # Step 1: PCA for variant filtering
        pca_output_prefix = f"{self._genetic_data_path.name}"
        pca_eigenvec_allele_path = Path(f"{pca_output_prefix}.eigenvec.allele")
        if not pca_eigenvec_allele_path.exists():
            LOGGER.info("Step 1/3: Calculating PCA...")
            cmd = f"plink2 --bfile {self._genetic_data_path.name} --pca 3 allele-wts --out {pca_output_prefix}"
            self._cmd_executor.run_cmd_on_docker(cmd)

        # Step 2: Filter for high-quality variants (low loading on PCs)
        LOGGER.info("Step 2/3: Filtering for high-quality variants...")
        eigen_df = pd.read_csv(pca_eigenvec_allele_path, sep='\t')
        filtered_eigen_df = eigen_df[(eigen_df[['PC1', 'PC2', 'PC3']].abs() < 0.003).all(axis=1)]
        weak_snps = filtered_eigen_df['ID'].unique()

        eigen_filtered_txt_path = Path(f"{pca_output_prefix}_eigen_filtered.txt")
        pd.Series(weak_snps).to_csv(eigen_filtered_txt_path, index=False, header=False)

        cmd = (
               f"plink2 --bfile {self._genetic_data_path.name} "
               f"--extract {eigen_filtered_txt_path} "
               f"--make-bed --out {self._genetic_data_path.name}_filtered_for_kinship"
        )
        self._cmd_executor.run_cmd_on_docker(cmd)

        # Step 3: Generate KING table
        LOGGER.info("Step 3/3: Generating KING table...")
        cmd = f"plink2 --bfile {self._genetic_data_path.name}_filtered_for_kinship --make-king-table --out {self._relatedness_db}"
        self._cmd_executor.run_cmd_on_docker(cmd)

        return Path(f'{self._relatedness_db}.king')

    def get_relateds_to_remove(self, relatedness_file: Path, wes_samples: Set[str]) -> Set[str]:
        """The main method for this class. Returns a list of samples to remove based on a relatedness report.

        :param relatedness_file: Path to a file containing a relatedness report (e.g., a KING table)
        :param wes_samples: A set of samples to consider.
        :return: A set of sample IDs to remove.
        """
        rel = self._load_relatedness(relatedness_file, wes_samples)
        return self._get_relateds_to_remove(rel)

    def _select_related_individual(self, rel: pd.DataFrame, samples_to_exclude: List[str]) -> Dict[str, pd.DataFrame]:
        """Identify individuals causing the most relatedness connections.

        :param rel: DataFrame with relatedness information.
        :param samples_to_exclude: A list of sample IDs to exclude from consideration.
        :return: A dictionary containing two DataFrames: 'rel' (filtered relatedness) and
                 'rel_totals' (counts of remaining related connections per individual).
        """
        rel = rel[~rel['ID1'].isin(samples_to_exclude)]
        rel = rel[~rel['ID2'].isin(samples_to_exclude)]

        # To find the most connected individuals, we first combine all IDs from the
        # 'ID1' and 'ID2' columns into a single series.
        rel_ids = pd.DataFrame(data=pd.concat([rel['ID1'], rel['ID2']]), columns=['ID'])
        
        # We then add a 'dummy' column of 1s, which allows us to count the occurrences of each ID.
        rel_ids['dummy'] = 1
        
        # Finally, we group by individual ID, sum the dummy column to get the total number of
        # connections, and sort to find the most connected individuals.
        rel_totals = rel_ids.groupby('ID').agg(total=('dummy', 'sum')).sort_values(by='total')

        return {'rel': rel, 'rel_totals': rel_totals}

    def _read_and_clean_relatedness(self, relatedness: Path) -> pd.DataFrame:
        """Read and standardize a relatedness matrix from a file.

        This function handles various header formats (UKB/PLINK) and ensures
        the output DataFrame has columns ['ID1', 'ID2', 'Kinship'].

        :param relatedness: Path to the relatedness file.
        :return: A pandas DataFrame with standardized relatedness information.
        """
        if relatedness.stat().st_size == 0:
            return pd.DataFrame(columns=["ID1", "ID2", "Kinship"]).astype({"Kinship": "float64"})

        rel = pd.read_csv(relatedness, delim_whitespace=True)

        # Normalize headers
        if 'i.s' in rel.columns and 'j.s' in rel.columns and 'kin' in rel.columns:
            rel = rel.rename(columns={'i.s': 'ID1', 'j.s': 'ID2', 'kin': 'Kinship'})

        # Normalize standard PLINK headers
        rel.columns = rel.columns.str.replace('^#', '', regex=True)
        rel = rel.rename(columns={'IID1': 'ID1', 'IID2': 'ID2', 'KINSHIP': 'Kinship'})

        # Fallback for headerless files
        if not {'ID1', 'ID2', 'Kinship'}.issubset(rel.columns):
            LOGGER.warning("Standard headers not found. Using index fallback.")
            rel = pd.read_csv(relatedness, delim_whitespace=True, header=None, skiprows=1)
            rel = rel.rename(columns={0: 'ID1', 1: 'ID2', rel.columns[-1]: 'Kinship'})

        rel['ID1'] = rel['ID1'].astype(str)
        rel['ID2'] = rel['ID2'].astype(str)
        return rel[['ID1', 'ID2', 'Kinship']]

    def _load_relatedness(self, relatedness: Path, wes_samples: Set[str]) -> pd.DataFrame:
        """Load and filter a relatedness matrix to include only specified samples.

        :param relatedness: Path to the relatedness file.
        :param wes_samples: A set of sample IDs to keep.
        :return: A pandas DataFrame with relatedness information for the given samples.
        """
        rel = self._read_and_clean_relatedness(relatedness)
        return rel[(rel["ID1"].isin(wes_samples)) & (rel["ID2"].isin(wes_samples))]

    def _get_relateds_to_remove(self, rel: pd.DataFrame) -> Set[str]:
        """Iteratively identify individuals to remove to break all related pairs.

        :param rel: A DataFrame of related pairs.
        :return: A set of sample IDs to remove.
        """
        relateds_to_remove = set()

        selection_result = self._select_related_individual(rel, [])
        rel, rel_totals = selection_result['rel'], selection_result['rel_totals']

        while len(rel_totals) > 0:
            samp_to_remove = rel_totals.iloc[-1].name
            relateds_to_remove.add(samp_to_remove)

            selection_result = self._select_related_individual(rel, list(relateds_to_remove))
            rel, rel_totals = selection_result['rel'], selection_result['rel_totals']

        return relateds_to_remove
    
