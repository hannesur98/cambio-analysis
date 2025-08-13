
from pathlib import Path
import pandas as pd
import os
import typing
import numpy as np
from scipy import stats
import re
import numpy as np
import pandas as pd
import phenopackets as PPKt
from collections import defaultdict
from scipy import stats

# Regular expression to identify numbers
NUMERIC_RE = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$')

class Cambio2Phenopacket:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._ppkt_d = defaultdict(PPKt.Phenopacket)
        for idx, row in self._df.iterrows():
            ppkt_id = row[0]
            if ppkt_id not in self._ppkt_d:
                ppkt = PPKt.Phenopacket()
                ppkt.id = ppkt_id
                self._ppkt_d[ppkt_id] = ppkt
        
    
    def get_ppkt_d(self) -> typing.Dict[str, PPKt.Phenopacket]:
        return self._ppkt_d
    

    def add_demographic_data(self):
        """
        improve doc and function name
        The entries for the patient appear in both the L (left eye) and R (right eye) rows
        To avoid duplication, we take on the entry for L
        """
        for idx, row in self._df.iterrows():
            ppkt_id = row[0]
            ppkt = self.get_ppkt_d().get(ppkt_id)
            if ppkt is None:
                raise ValueError(f"Could not retrieve phenopacket for id={ppkt_id}")
            eye = row["Eye"]
            if eye != "L":
                continue
            sex = row["Gender"]
            if sex == "Female":
                ppkt.subject.sex = PPKt.Sex.FEMALE
            elif sex == "Male":
                ppkt.subject.sex = PPKt.Sex.MALE
            else:
                raise ValueError(f"Did not recognize gender entry {sex}")
            admission_age = row["Age at Admission"]
            try:
                y = int(admission_age)
                iso_age = f"P{y}Y"
                ppkt.subject.time_at_last_encounter.age.iso8601duration  = iso_age
            except Exception as exc:
                print(f"Could not parse iso age in years: {exc}")
            
    def add_hpo_columns(self):
        """
        Add data from columns that map 1:1 to HPO terms
        Note: We may need to revise this to map diseases vs phenotypic features explicitly
        """
        column_name_to_hpo_d = dict()
        column_name_to_hpo_d["Disease Psoriasis"] = ["Psoriasiform dermatitis", "HP:0003765"]
        for idx, row in self._df.iterrows():
            ppkt_id = row[0]
            ppkt = self.get_ppkt_d().get(ppkt_id)
            if ppkt is None:
                raise ValueError(f"Could not retrieve phenopacket for id={ppkt_id}")
            eye = row["Eye"]
            if eye != "L":
                continue
            for column_name, hpo_term in column_name_to_hpo_d.items():
                cell_contents = row[column_name]
                cell_contents = str(cell_contents)
                if cell_contents == "Yes":
                    pft = PPKt.PhenotypicFeature()
                    pft.type.id = hpo_term[1]
                    pft.type.label = hpo_term[0]
                    ppkt.phenotypic_features.append(pft)
                elif cell_contents == "No":
                    pft = PPKt.PhenotypicFeature()
                    pft.type.id = hpo_term[1]
                    pft.type.label = hpo_term[0]
                    pft.excluded = True
                    ppkt.phenotypic_features.append(pft)
                elif cell_contents == "?":
                    # No data available, OK, we can skip
                    pass
                else:
                    raise ValueError(f"Unrecognized error in HPO-type column '{cell_contents}'")