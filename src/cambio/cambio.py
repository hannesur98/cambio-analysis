from pathlib import Path
import os
import re
import typing

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
    """

    def __init__(
        self,
        cambio_path: Path,
        group_names: typing.List[str] = ["AMD", "Control"],
    ) -> None:
        if not os.path.exists(cambio_path):
            raise FileNotFoundError(f"Could not find file at {cambio_path}")

        # Raw data
        self._df = pd.read_csv(cambio_path, sep=";")

        # Cleaned data (string-trim + comma->dot + NA handling)
        self._df_clean = self.get_clean_dataframe()

        # Column type map
        self._col_type = self.define_column_types()

        # Group labels (order matters)
        self._group_names = group_names

        # ------------------------------
        # Variablen-Gruppen definieren
        # ------------------------------
        self.eye_vars = [
            "EV concentration (NanoFCM (part./mL))",
            "EV concentration (NanoFCM (part./mL)) 2nd measurement 04.03.2025",
            "EV median size ((nm), NanoFCM)",
            "EV median size ((nm), NanoFCM) 2nd measurement 04.03.2025",
            "Fluorescent Particle Count total CD41a (ExoView)",
            "Fluorescent Particle Count total CD63 (ExoView)",
            "Fluorescent Particle Count total CD81 (ExoView)",
            "Fluorescent Particle Count total CD9 (ExoView)",
            "Fluorescent Particle Count FI CD41a (ExoView)",
            "Fluorescent Particle Count FI CD63 (ExoView)",
            "Fluorescent Particle Count FI CD81 (ExoView)",
            "Fluorescent Particle Count FI CD9 (ExoView)",
            "Fluorescent Particle Count FH CD41a (ExoView)",
            "Fluorescent Particle Count FH CD63 (ExoView)",
            "Fluorescent Particle Count FH CD81 (ExoView)",
            "Fluorescent Particle Count FH CD9 (ExoView)",
            "Fluorescent Particle Count C3 CD41a (ExoView)",
            "Fluorescent Particle Count C3 CD63 (ExoView)",
            "Fluorescent Particle Count C3 CD81 (ExoView)",
            "Fluorescent Particle Count C3 CD9 (ExoView)",
            "Fluorescence Intensity FI CD41a (ExoView)",
            "Fluorescence Intensity FI CD63 (ExoView)",
            "Fluorescence Intensity FI CD81 (ExoView)",
            "Fluorescence Intensity FI CD9 (ExoView)",
            "Fluorescence Intensity FH CD41a (ExoView)",
            "Fluorescence Intensity FH CD63 (ExoView)",
            "Fluorescence Intensity FH CD81 (ExoView)",
            "Fluorescence Intensity FH CD9 (ExoView)",
            "Fluorescence Intensity C3 CD41a (ExoView)",
            "Fluorescence Intensity C3 CD63 (ExoView)",
            "Fluorescence Intensity C3 CD81 (ExoView)",
            "Fluorescence Intensity C3 CD9 (ExoView)",
            "Fluorescence per particle FI CD41a (ExoView)",
            "Fluorescence per particle FI CD63 (ExoView)",
            "Fluorescence per particle FI CD81 (ExoView)",
            "Fluorescence per particle FI CD9 (ExoView)",
            "Fluorescence per particle FH CD41a (ExoView)",
            "Fluorescence per particle FH CD63 (ExoView)",
            "Fluorescence per particle FH CD81 (ExoView)",
            "Fluorescence per particle FH CD9 (ExoView)",
            "Fluorescence per particle C3 CD41a (ExoView)",
            "Fluorescence per particle C3 CD63 (ExoView)",
            "Fluorescence per particle C3 CD81 (ExoView)",
            "Fluorescence per particle C3 CD9 (ExoView)",
            "FI Particle Counts CD41a (fraction of total)",
            "FI Particle Counts CD63 (fraction of total)",
            "FI Particle Counts CD81 (fraction of total)",
            "FI Particle Counts CD9 (fraction of total)",
            "FH Particle Counts CD41a (fraction of total)",
            "FH Particle Counts CD63 (fraction of total)",
            "FH Particle Counts CD81 (fraction of total)",
            "FH Particle Counts CD9 (fraction of total)",
            "C3 Particle Counts CD41a (fraction of total)",
            "C3 Particle Counts CD63 (fraction of total)",
            "C3 Particle Counts CD81 (fraction of total)",
            "C3 Particle Counts CD9 (fraction of total)",
            "Co-Lokalisation Counts FI-FH CD41a (fraction of total)",
            "Co-Lokalisation Counts FI-FH CD63 (fraction of total)",
            "Co-Lokalisation Counts FI-FH CD81 (fraction of total)",
            "Co-Lokalisation Counts FI-FH CD9 (fraction of total)",
            "Co-Lokalisation Counts FI-C3 CD41a (fraction of total)",
            "Co-Lokalisation Counts FI-C3 CD63 (fraction of total)",
            "Co-Lokalisation Counts FI-C3 CD81 (fraction of total)",
            "Co-Lokalisation Counts FI-C3 CD9 (fraction of total)",
            "Co-Lokalisation Counts FH-C3 CD41a (fraction of total)",
            "Co-Lokalisation Counts FH-C3 CD63 (fraction of total)",
            "Co-Lokalisation Counts FH-C3 CD81 (fraction of total)",
            "Co-Lokalisation Counts FH-C3 CD9 (fraction of total)",
            "Co-Lokalisation Counts FI-FH-C3 CD41a (fraction of total)",
            "Co-Lokalisation Counts FI-FH-C3 CD63 (fraction of total)",
            "Co-Lokalisation Counts FI-FH-C3 CD81 (fraction of total)",
            "Co-Lokalisation Counts FI-FH-C3 CD9 (fraction of total)",
            "RPE lift (Drusen) area 3 mm circle (mm2)",
            "RPE lift (Drusen) area 5 mm circle (mm2)",
            "RPE lift (Drusen) volume 3 mm circle (mm3)",
            "RPE lift (Drusen) volume 5 mm circle (mm3)",
            "original Aqueous Humor Volume (µL)",
        ]

        self.person_vars = [
            "Status", "Healthy Retina", "No AMD", "Early AMD", "Dry AMD", "Geographic Atrophy",
            "Age at Admission", "Birthday", "Gender", "Weight (kg)", "Height (cm)",
            "Alcohol Consumption (0-7 days/week)", "Smoking Status", "Job (most recent)", "Sun Exposure   (1-5)",
            "Food Vegetables", "Food Fruit", "Food Carbs", "Food OliveOil", "Food Nuts", "Food Dairy",
            "Food Sweets", "Food Fish", "Disease High Blood Pressure", "Disease Diabetes", "Disease RA",
            "Disease PNH", "Disease Crohns", "Disease Bullous Pemphigoid", "Disease Dermatomyositis",
            "Disease Psoriasis", "Disease Quinckes Edema", "Disease NMOSD", "Disease MS", "Disease Alzheimers",
            "Disease Guillain Barre Syndrome", "Disease Chronic Renal", "Disease AA Hepatitis", "Disease AAV",
            "Disease Sjoegrens", "Disease SLE", "Disease Tumors", "Disease Infection", "Other Disease",
            "Medication", "Exclusion CNV", "Exclusion Anti VEGF", "Exclusion Vein Occlusion",
            "Exclusion Macula Edema", "Exclusion Myopia", "Exclusion Macula Foramen",
            "Exclusion Genetical Retina Disease", "Exclusion Uveitis", "Exclusion Keratoconjunctivitis",
            "Exclusion Surgery", "Genotyped", "C___8355565_10", "C___2530373_10", "C___2530278_10",
            "C__26330755_10", "C__34681305_20", "C__27473788_10", "C__29934973_20", "C__29910029_10",
            "C__2530382_10",
        ]

        # Eye DataFrame: keine Duplikate entfernen (pro Auge)
        self.eye_df = self._df[["Alias", "Eye", "Group"] + self.eye_vars].copy()

        # Person DataFrame: nur eine Zeile pro Alias
        self.person_df = (
            self._df[["Alias", "Group"] + self.person_vars]
            .drop_duplicates(subset="Alias")
            .copy()
        )

    def get_clean_dataframe(self) -> pd.DataFrame:
        """
        Trim strings, replace German commas with decimal points in numeric-like values,
        and set empty/NA markers to np.nan.
        """
        df = self._df.copy()

        # 1) Alles als String strippen (nur auf Objektspalten)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()

        # 2) Kommas durch Punkte ersetzen (nur für numerisch erkennbare Werte)
        for col in df.columns:
            df[col] = df[col].replace(",", ".", regex=True)

        # 3) Leere und NA-Zeichen zu NaN machen
        df = df.replace(["", "NA", "NaN", "nan"], np.nan)

        # 4) Versuchen, numerisch aussehende Spalten wieder in Float zu konvertieren
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass

        return df

    def define_column_types(self) -> typing.Dict[str, str]:
        """
        Figure out if columns have numerical or categorical data after removing empties/NaN.
        Robust against already-numeric columns (no .str on numeric series).
        """
        col_types: typing.Dict[str, str] = {}
        for col in self._df_clean.columns:
            non_na = self._df_clean[col].dropna()
            if non_na.empty:
                col_types[col] = "empty"
                continue

            # Convert to string (safe), strip, then try numeric
            non_na_str = non_na.astype(str).str.strip()
            as_num = pd.to_numeric(non_na_str.str.replace(",", "."), errors="coerce")
            col_types[col] = "numeric" if as_num.notna().all() else "categorical"
        return col_types

    def perform_t_tests(self) -> typing.List[typing.Dict]:
        """
        T-Tests:
         - Für numerische Eye-Vars: alle Augen einzeln.
         - Für numerische Person-Vars: nur eine Zeile pro Alias (self.person_df).
        """
        t_test_d_list: typing.List[typing.Dict] = []
        group_col = "Group"

        # numerische Variablen aus den definierten Gruppen
        numeric_eye_vars = [v for v in self.eye_vars if self._col_type.get(v) == "numeric"]
        numeric_person_vars = [v for v in self.person_vars if self._col_type.get(v) == "numeric"]

        # Eye-Vars (pro Auge)
        for col_name in numeric_eye_vars:
            if col_name not in self.eye_df.columns:
                continue
            data = self.eye_df[[group_col, col_name]].dropna()
            if data.empty:
                continue
            result = self.ttest_on_two_columns(two_col_df=data, value_col=col_name)
            result["data_count"] = data.groupby(group_col).size().to_dict()
            t_test_d_list.append(result)

        # Person-Vars (1x pro Alias)
        for col_name in numeric_person_vars:
            if col_name not in self.person_df.columns:
                continue
            data = self.person_df[[group_col, col_name]].dropna()
            if data.empty:
                continue
            result = self.ttest_on_two_columns(two_col_df=data, value_col=col_name)
            result["data_count"] = data.groupby(group_col).size().to_dict()
            t_test_d_list.append(result)

        return sorted(t_test_d_list, key=lambda r: r["p_val"])

    def ttest_on_two_columns(
    self,
    two_col_df: pd.DataFrame,
    value_col: str,
    equal_var: bool = False,
    ) -> typing.Dict:
        group_col = "Group"
        # 1) Guards
        if not isinstance(group_col, str):
            raise TypeError(f"group_col must be a string, got {type(group_col)}")
        if group_col not in two_col_df.columns or value_col not in two_col_df.columns:
            raise ValueError(f"Expected columns '{group_col}' and '{value_col}' not found in DataFrame")

        # 2) Relevant Spalten kopieren
        df = two_col_df[[group_col, value_col]].copy()

        # 3) Group-Spalte robust normalisieren (kein .str)
        df[group_col] = df[group_col].map(lambda x: str(x).strip())

        # 4) Wertespalte: falls nicht numerisch, als String reinigen; sonst direkt lassen
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            s = df[value_col].map(lambda x: str(x).strip().replace(",", "."))
            # nur voll-numerische Tokens behalten
            mask_non_numeric = ~s.str.match(NUMERIC_RE, na=False)
            mask_empty = s.isin(["", "NA", "NaN", "nan"])
            s = s.mask(mask_non_numeric | mask_empty, np.nan)
            df[value_col] = pd.to_numeric(s, errors="coerce")

        # 5) NaNs droppen
        df = df.dropna(subset=[group_col, value_col])

        # 6) Gruppen prüfen
        group_names = self._group_names
        missing = [g for g in group_names if g not in df[group_col].unique()]
        if missing:
            raise ValueError(f"Missing expected groups: {missing}")

        # 7) Werte pro Gruppe
        g1 = df.loc[df[group_col] == group_names[0], value_col].to_numpy()
        g2 = df.loc[df[group_col] == group_names[1], value_col].to_numpy()
        if len(g1) < 2 or len(g2) < 2:
            raise ValueError("One or both groups have too few numeric observations after cleaning.")

        # 8) t-Test (Welch standard)
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=equal_var, nan_policy="omit")

        diag = {
            "group_names": group_names,
            "n_obs": {group_names[0]: int(len(g1)), group_names[1]: int(len(g2))},
            "means": {group_names[0]: float(np.mean(g1)), group_names[1]: float(np.mean(g2))},
            "stds":  {group_names[0]: float(np.std(g1, ddof=1)), group_names[1]: float(np.std(g2, ddof=1))},
        }
        return {"column": value_col, "t_stat": float(t_stat), "p_val": float(p_val), "diag": diag}
        
    def perform_chi2_tests(self) -> typing.List[typing.Dict]:
        """
        Chi²-/Fisher-Tests nur auf kategorialen Personen-Variablen,
        ausgewertet auf self.person_df (eine Zeile pro Alias).
        Es werden nur die zwei Hauptgruppen aus self._group_names verwendet.
        """
        results: typing.List[typing.Dict] = []
        group_col = "Group"

        # Kandidaten: kategoriale Variablen aus person_vars, die es in person_df gibt
        cat_person_vars = [
            c for c in self.person_vars
            if c in self.person_df.columns and self._col_type.get(c) == "categorical"
        ]

        for col in cat_person_vars:
            # Daten holen (eine Zeile pro Person)
            df = self.person_df[[group_col, col]].dropna()
            if df.empty:
                continue

            # Nur die zwei definierten Hauptgruppen vergleichen
            df = df[df[group_col].isin(self._group_names)]
            if df.empty or df[group_col].nunique() < 2:
                continue

            # Strings vereinheitlichen
            g = df[group_col].astype(str).str.strip()
            v = df[col].astype(str).str.strip()

            # Kontingenztafel
            ct = pd.crosstab(g, v)
            # Mindestens 2 Kategorien in der Spalte benötigt
            if ct.shape[1] < 2:
                continue

            # Standard: Chi²
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            test_used = "chi2"
            chi2_stat = float(chi2)
            odds_ratio = np.nan

            # Bei 2x2 und kleinen erwarteten Häufigkeiten: Fisher
            if ct.shape == (2, 2) and (expected < 5).any():
                odds_ratio, p = stats.fisher_exact(ct.to_numpy())
                test_used = "fisher_exact"
                chi2_stat = np.nan

            results.append({
                "column": col,
                "test": test_used,
                "p_val": float(p),
                "chi2_stat": chi2_stat,
                "odds_ratio": float(odds_ratio) if not np.isnan(odds_ratio) else np.nan,
                "diag": {
                    "groups_used": self._group_names,
                    "levels": list(ct.columns.astype(str)),
                    "counts": ct
                }
            })
        return sorted(results, key=lambda r: r["p_val"])
    
    def get_df(self) -> pd.DataFrame:
        return self._df_clean

