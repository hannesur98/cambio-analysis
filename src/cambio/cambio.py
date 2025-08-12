
from pathlib import Path
import pandas as pd
import os
import typing
import numpy as np
from scipy import stats
import re
import numpy as np
import pandas as pd
from scipy import stats

# Regular expression to identify numbers
NUMERIC_RE = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$')

class Cambio:
    """
    A helper class to organize and perform preliminary statistical analysis
    on CAMBIO dataset files.

    This class is designed to:
      - Load CAMBIO data from a CSV file.
      - Clean and preprocess the dataset for analysis.
      - Classify columns by data type (e.g., numeric vs categorical).
      - Prepare and store group information for statistical testing.

    Parameters
    ----------
    cambio_path : Path
        Path to the CAMBIO CSV file. The file must exist, otherwise a
        FileNotFoundError is raised. The CSV is expected to be separated by semicolons (`;`).
    group_names : list of str, optional
        Names of the two groups to be used in group comparisons
        (default is ["AMD", "Control"]).

    Attributes
    ----------
    _df : pandas.DataFrame
        Raw dataframe loaded directly from the CSV file.
    _df_clean : pandas.DataFrame
        Cleaned dataframe produced by `get_clean_dataframe()`, ready for analysis.
    _col_type : dict
        Dictionary mapping column names to their detected data types
        (e.g., "numeric", "categorical").
    _group_names : list of str
        Names of the two groups used for t-tests or other comparisons.

    Methods
    -------
    get_clean_dataframe():
        Cleans the raw dataframe by removing or replacing invalid/missing entries.
    define_column_types():
        Detects and assigns a data type for each column in the dataframe.
    perform_t_test_on_numerical_cols():
        Runs independent t-tests for numeric columns between the specified groups.
    """
    def __init__(self, 
                 cambio_path: Path,
                 group_names = ["AMD", "Control"]
                 ) -> None:
        if not os.path.exists(cambio_path):
            raise FileNotFoundError(f"Could not find file at {cambio_path}")
        self._df = pd.read_csv(cambio_path, sep=";")
        self._df_clean = self.get_clean_dataframe()
        self._col_type = self.define_column_types()
        self._group_names = group_names        
        



    def get_clean_dataframe(self) -> pd.DataFrame:
        """
        Mark empty cells and transform German comma to English decimal point for analysis purposes
        """
        s = self._df.astype(str).apply(lambda col: col.str.strip())
        s = s.replace(",", ".")
        # Build a boolean mask for empty / NA-like cells column-wise
        mask_empty = s.apply(lambda col: col.eq("") | col.isin(["NA", "NaN", "nan"]))

        # Replace those with real NaNs
        s_clean = s.mask(mask_empty, np.nan)

        return s_clean


    def define_column_types(self) -> typing.Dict[str,str]:
        """
        Figure out if columns have numerical or categorical data after removing empties/NaN
        """
        col_types = dict()

        for col in self._df_clean.columns:
            non_na_values = self._df_clean[col].dropna()
            #print(non_na_values)
            if non_na_values.empty:
                col_types[col] = "empty"
            elif pd.to_numeric(non_na_values.str.replace(",", "."), errors="coerce").notna().all():
                col_types[col] = "numeric"
            else:
                col_types[col] = "categorical"

        return col_types

    def perform_t_tests(self) -> typing.List[typing.Dict]:
        t_test_d_list = list();
        group_col = "Group"  #
        for col_name, col_series in self._df_clean.items():  # or df.iteritems() in older Pandas
            column_name = str(col_name)
            if column_name == "Alias" or column_name == "Group":
                continue
            col_t = self._col_type[str(col_name)]
            #print(f"Coloumn: {column_name}: {col_t}")
            if col_t != "numeric":
                continue
            data = self._df_clean[[group_col, str(col_name)]].dropna()
            result  = self.ttest_on_two_columns(two_col_df=data, value_col=column_name)
            t_test_d_list.append(result)
        sorted_results = sorted(t_test_d_list, key=lambda r: r['p_val'])
        return sorted_results

    
    def ttest_on_two_columns(
        self,
        two_col_df, 
        value_col: str,
        equal_var=False):
        """
        two_col_df: DataFrame with exactly (or at least) two columns,
                    first is group, second is value OR supply group_col/value_col names.
        equal_var: pass to ttest_ind (default False -> Welch's t-test).
        Returns dict with t_stat, p_val and diagnostics.
        """
        group_col = "Group"  #
        df = two_col_df[[group_col, value_col]].copy() ## TODO - wir brauchen das hier nicht mehr

        # 1) normalize group column to string
        df[group_col] = df[group_col].astype(str).str.strip()

        # 2) normalize value column to string, strip, replace comma by dot
        s = df[value_col].astype(str).str.strip().str.replace(',', '.', regex=False)

        # 3) build mask of non-numeric / empty tokens
        # Accept only strings that fully match a numeric pattern (integers, floats, .5, 5., exp notation)
        mask_non_numeric = ~s.str.match(NUMERIC_RE, na=False)

        # also treat common NA tokens as non-numeric
        mask_empty = s.isin(['', 'NA', 'NaN', 'nan'])
        mask = mask_non_numeric | mask_empty

        # 4) set those to NaN and convert to numeric
        s_clean = s.mask(mask, np.nan)
        df[value_col] = pd.to_numeric(s_clean, errors='coerce')

        # 5) drop rows where either group or value is missing
        df = df.dropna(subset=[group_col, value_col])

        # 7) prepare groups
        group_names = self._group_names
        
        missing_groups = [g for g in group_names if g not in df[group_col].unique()]
        if missing_groups:
            raise ValueError(f"Missing expected groups: {missing_groups}")

        g1 = df.loc[df[group_col] == group_names[0], value_col].dropna().tolist()
        g2 = df.loc[df[group_col] == group_names[1], value_col].dropna().tolist()

        if len(g1) < 2 or len(g2) <2:
            raise ValueError("One or both groups have no numeric data after cleaning.")

        # 8) t-test
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var, nan_policy='omit')

        diag = {
            'group_names': group_names,
            'n_obs': {group_names[0]: len(g1), group_names[1]: len(g2)},
            'means': {group_names[0]: np.mean(g1), group_names[1]: np.mean(g2)},
            'stds': {group_names[0]: np.std(g1, ddof=1), group_names[1]: np.std(g2, ddof=1)},
        }

        return {'column': value_col, 't_stat': t_stat, 'p_val': p_val, 'diag': diag}

           