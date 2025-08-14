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
from google.protobuf.json_format import ParseDict

GENO_ZYGOSITY_MAP = {
    "hom_ref": {"id": "GENO:0000134", "label": "homozygous reference"},
    "het":     {"id": "GENO:0000135", "label": "heterozygous"},
    "hom_alt": {"id": "GENO:0000136", "label": "homozygous"},
}

def _clean_genotype_call(val):
    if pd.isna(val):
        return np.nan
    s = str(val).upper().strip().replace("|","/").replace(" ", "")
    s = re.sub(r"\(.*?\)", "", s)
    repl = {
        "UNDETERMINED": "",
        "C/UNDETERMINED": "C",
        "T/UNDETERMINED": "T",
        "TC/UNDETERMINED": "TC",
        "T?": "T",
    }
    s = repl.get(s, s)
    return s or np.nan

def _zygosity_from_call(call):
    if pd.isna(call):
        return None
    if call in ("CC","C","C/C"):
        return GENO_ZYGOSITY_MAP["hom_alt"]
    if call in ("TT","T","T/T"):
        return GENO_ZYGOSITY_MAP["hom_ref"]
    if call in ("CT","TC","C/T","T/C"):
        return GENO_ZYGOSITY_MAP["het"]
    return None

# Regular expression to identify numbers
NUMERIC_RE = re.compile(r'^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?$')

class Cambio2Phenopacket:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._ppkt_d = defaultdict(PPKt.Phenopacket)
        for idx, row in self._df.iterrows():
            ppkt_id = str(row.iloc[0])
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
            ppkt_id = str(row.iloc[0])  # robust gegen FutureWarning
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

            # --- Add anthropometrics as measurements (weight, height) ---
            # Sammle Werte über beide Augen
            pid_mask = self._df.iloc[:, 0].astype(str) == ppkt_id
            grp = self._df.loc[pid_mask]

            def _to_float(val):
                if val is None:
                    return None
                s = str(val).strip().replace(",", ".")
                m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
                return float(m.group(0)) if m else None

            def _first_numeric(df, col):
                if col not in df.columns:
                    return None
                for v in df[col].tolist():
                    f = _to_float(v)
                    if f is not None:
                        return f
                return None

            # Stelle sicher, dass es die Root-measurements-Liste gibt
            if getattr(ppkt, "measurements", None) is None:
                ppkt.measurements = []

            def _add_qty_measurement_root(target_ppkt, assay_id, assay_label, value_float, unit_key):
                try:
                    m = PPKt.Measurement()

                    # assay (OntologyClass)
                    oc = PPKt.OntologyClass()
                    oc.id = assay_id
                    oc.label = assay_label
                    if hasattr(m.assay, "CopyFrom"):
                        m.assay.CopyFrom(oc)
                    else:
                        m.assay = oc

                    # value.quantity mit OntologyClass-Unit (UO)
                    val_msg = PPKt.Value()
                    qty = PPKt.Quantity()
                    qty.value = float(value_float)

                    unit_id_map = {
                        "kg": ("UO:0000009", "kilogram"),
                        "cm": ("UO:0000015", "centimeter"),
                    }
                    uid, ulabel = unit_id_map[unit_key]
                    uoc = PPKt.OntologyClass()
                    uoc.id = uid
                    uoc.label = ulabel
                    if hasattr(qty.unit, "CopyFrom"):
                        qty.unit.CopyFrom(uoc)
                    else:
                        qty.unit = uoc

                    if hasattr(val_msg, "quantity") and hasattr(val_msg.quantity, "CopyFrom"):
                        val_msg.quantity.CopyFrom(qty)
                    else:
                        val_msg.quantity = qty

                    if hasattr(m, "value") and hasattr(m.value, "CopyFrom"):
                        m.value.CopyFrom(val_msg)
                    else:
                        m.value = val_msg

                    target_ppkt.measurements.append(m)
                except Exception as e:
                    print(f"Could not add root-level measurement: {e}")

            w_float = _first_numeric(grp, "Weight (kg)")
            if w_float is not None:
                _add_qty_measurement_root(ppkt, "NCIT:C48155", "Body Weight", w_float, "kg")

            h_float = _first_numeric(grp, "Height (cm)")
            if h_float is not None:
                _add_qty_measurement_root(ppkt, "NCIT:C48154", "Body Height", h_float, "cm")
            # --- End anthropometrics ---
            
    def add_hpo_columns(self):
        """
        Add data from columns that map 1:1 to HPO terms
        Note: We may need to revise this to map diseases vs phenotypic features explicitly
        """
        column_name_to_hpo_d = dict()
        column_name_to_hpo_d["Disease Psoriasis"] = ["Psoriasiform dermatitis", "HP:0003765"]
        for idx, row in self._df.iterrows():
            ppkt_id = str(row.iloc[0])
            ppkt = self.get_ppkt_d().get(ppkt_id)
            if ppkt is None:
                raise ValueError(f"Could not retrieve phenopacket for id={ppkt_id}")
            eye = row["Eye"]
            if eye != "L":
                continue
            # --- Special handling for "No AMD" column ---
            # Logic: If "No AMD" == "No" -> add HPO:0000608 (Macular degeneration)
            # If "No AMD" == "Yes" or unknown -> add HPO: 0000608 + excluded
            no_amd_val = str(row.get("No AMD", "")).strip()
            pft = PPKt.PhenotypicFeature()
            pft.type.id = "HP:0000608"
            pft.type.label = "Macular degeneration"

            if no_amd_val == "Yes":
                pft.excluded = True  # hat KEINE AMD
            elif no_amd_val == "No":
                pft.excluded = False  # hat AMD

            ppkt.phenotypic_features.append(pft)
            # --- End special handling ---
            # --- Special handling for "Geographic Atrophy" column ---
            # Logic: If value known, always encode HP:0031609 and mark excluded accordingly.
            ga_val = str(row.get("Geographic Atrophy", "")).strip()
            if ga_val in {"Yes", "No"}:
                pft = PPKt.PhenotypicFeature()
                pft.type.id = "HP:0031609"
                pft.type.label = "Geographic atrophy"
                if ga_val == "No":
                    pft.excluded = True   # ausdrücklich ausgeschlossen
                # bei "Yes" bleibt included (excluded=False default)
                ppkt.phenotypic_features.append(pft)
            # --- End special handling for Geographic Atrophy ---

            # Paroxysmal nocturnal hemoglobinuria (PNH)
            pnh_val = row.get("Disease PNH")
            if pd.notna(pnh_val) and str(pnh_val).strip() != "":
                sval = str(pnh_val).strip().lower()
                # If explicitly unknown, skip encoding
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0004818"
                    pf.type.label = "Paroxysmal nocturnal hemoglobinuria"
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False
                    ppkt.phenotypic_features.append(pf)

            # High Blood Pressure / Hypertension
            hbp_val = row.get("Disease High Blood Pressure")
            if pd.notna(hbp_val) and str(hbp_val).strip() != "":
                try:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0000822"
                    pf.type.label = "Hypertension"
                    # Optional: excluded setzen, wenn Wert 'No' o.ä. ist
                    if str(hbp_val).strip().lower() in ["no", "false", "0"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False
                    ppkt.phenotypic_features.append(pf)
                except Exception as e:
                    print(f"Could not add High Blood Pressure: {e}")

            # Rheumatoid Arthritis
            ra_val = row.get("Disease RA")
            if pd.notna(ra_val) and str(ra_val).strip() != "":
                pf = PPKt.PhenotypicFeature()
                pf.type.id = "HP:0001370"
                pf.type.label = "Rheumatoid arthritis"
                if str(ra_val).strip().lower() in ["no", "false", "0", "nein"]:
                    pf.excluded = True
                else:
                    pf.excluded = False
                ppkt.phenotypic_features.append(pf)

            # Quincke's edema → Angioedema (HPO: HP:0100665)
            qe_val = row.get("Disease Quinckes Edema")
            if pd.notna(qe_val) and str(qe_val).strip() != "":
                sval = str(qe_val).strip().lower()

                # Explizit unbekannt → nichts schreiben
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0100665"
                    pf.type.label = "Angioedema"

                    # Abwesenheit → excluded=True; alles andere → present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)


            # Alzheimer disease (HPO: HP:0002511)
            alz_val = row.get("Disease Alzheimers")
            if pd.notna(alz_val) and str(alz_val).strip() != "":
                sval = str(alz_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0002511"
                    pf.type.label = "Alzheimer disease"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)

            # Acute demyelinating polyneuropathy (HPO: HP:0007131)
            gbs_val = row.get("Disease Guillain Barre Syndrome")
            if pd.notna(gbs_val) and str(gbs_val).strip() != "":
                sval = str(gbs_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0007131"
                    pf.type.label = "Acute demyelinating polyneuropathy"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)
            
            # Chronic kidney disease (HPO: HP:0003774)
            ckd_val = row.get("Disease Chronic Renal")
            if pd.notna(ckd_val) and str(ckd_val).strip() != "":
                sval = str(ckd_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0003774"
                    pf.type.label = "Chronic kidney disease"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)

            # Vasculitis (HPO: HP:0002633)
            aav_val = row.get("Disease AAV")
            if pd.notna(aav_val) and str(aav_val).strip() != "":
                sval = str(aav_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0002633"
                    pf.type.label = "Vasculitis"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)

            # Systemic lupus erythematosus (HPO: HP:0002725)
            sle_val = row.get("Disease SLE")
            if pd.notna(sle_val) and str(sle_val).strip() != "":
                sval = str(sle_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0002725"
                    pf.type.label = "Systemic lupus erythematosus"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)

            # Neoplasm (HPO: HP:0002664)
            tumor_val = row.get("Disease Tumors")
            if pd.notna(tumor_val) and str(tumor_val).strip() != "":
                sval = str(tumor_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    pf = PPKt.PhenotypicFeature()
                    pf.type.id = "HP:0002664"
                    pf.type.label = "Neoplasm"

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        pf.excluded = True
                    else:
                        pf.excluded = False

                    ppkt.phenotypic_features.append(pf)


            # Dermatomyositis — encode as MONDO Disease (no HPO term exists)
            dm_val = row.get("Disease Dermatomyositis")
            if pd.notna(dm_val) and str(dm_val).strip() != "":
                sval = str(dm_val).strip().lower()

                # Explicit unknown → skip encoding
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    dis = PPKt.Disease()
                    dis.term.id = "MONDO:0016367"
                    dis.term.label = "Dermatomyositis"

                    # Absent patterns → excluded=True, otherwise present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        dis.excluded = True
                    else:
                        dis.excluded = False

                    ppkt.diseases.append(dis)

            # Neuromyelitis optica spectrum disorder (NMOSD)
            nmosd_val = row.get("Disease NMOSD")
            if pd.notna(nmosd_val) and str(nmosd_val).strip() != "":
                sval = str(nmosd_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    dis = PPKt.Disease()
                    dis.term.id = "MONDO:0019100"
                    dis.term.label = "Neuromyelitis optica spectrum disorder"  # ORPHA:71211

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        dis.excluded = True
                    else:
                        dis.excluded = False

                    ppkt.diseases.append(dis)

            # Multiple sclerosis 
            ms_val = row.get("Disease MS")
            if pd.notna(ms_val) and str(ms_val).strip() != "":
                sval = str(ms_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    dis = PPKt.Disease()
                    dis.term.id = "MONDO:0021571"
                    dis.term.label = "Multiple sclerosis"  # OMIM:126200

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        dis.excluded = True
                    else:
                        dis.excluded = False

                    ppkt.diseases.append(dis)

            # Autoimmune hepatitis — MONDO Disease
            aih_val = row.get("Disease AA Hepatitis")
            if pd.notna(aih_val) and str(aih_val).strip() != "":
                sval = str(aih_val).strip().lower()

                # explizit unbekannt → überspringen
                if sval in ["?", "unknown", "unk", "na", "n/a"]:
                    pass
                else:
                    dis = PPKt.Disease()
                    dis.term.id = "MONDO:0016264"
                    dis.term.label = "Autoimmune hepatitis"  # ORPHA:2137

                    # Abwesenheit → excluded=True, sonst present
                    if sval in ["no", "false", "0", "nein", "absent"]:
                        dis.excluded = True
                    else:
                        dis.excluded = False

                    ppkt.diseases.append(dis)
                

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
                


    def add_cfh_rs1061170_interpretation(self, assay_col: str = "C___8355565_10") -> None:
        """
        Für jede Zeile in self._df:
        - liest den Genotyp aus assay_col (C___8355565_10)
        - fügt im Phenopacket (get_ppkt_d()) einen Interpretation-Block hinzu **nur wenn** ein valider Genotyp vorliegt:
            progressStatus = SOLVED
            interpretationStatus = CONTRIB​UTORY
            variationDescriptor:
                expressions: dbSNP rs1061170
                geneContext: NCBIGene:3075 / CFH
                allelicState gemäß Genotyp
        Hinweis: Für fehlende/unklare Genotypen wird **kein** Interpretationseintrag erzeugt (Parser-sicher).
        """
        # kleine Normalisierungshilfe (wie zuvor genutzt)
        def _clean_call(val):
            import re, numpy as np, pandas as pd
            if pd.isna(val):
                return np.nan
            s = str(val).upper().strip().replace("|", "/").replace(" ", "")
            s = re.sub(r"\(.*?\)", "", s)
            # typische Sonderfälle mappen
            if s in ("UNDETERMINED", ""):
                return np.nan
            if s in ("C/UNDETERMINED",):
                return "C"
            if s in ("T/UNDETERMINED",):
                return "T"
            if s in ("TC/UNDETERMINED",):
                return "TC"
            if s in ("C/T",):
                return "CT"
            if s in ("T/C",):
                return "CT"
            return s

        for _, row in self._df.iterrows():
            # eine Zeile pro Patient auswählen (wie in deinen anderen Methoden)
            eye = str(row.get("Eye", "")).strip()
            if eye not in ("", "L", "Both", "B"):
                continue

            # Patienten-ID bestimmen (wie bisher bei dir üblich)
            pid = str(row.get("Alias", row.iloc[0]))
            ppkt = self.get_ppkt_d().get(pid)
            if ppkt is None:
                raise ValueError(f"Could not retrieve phenopacket for id={pid}")

            raw = row.get(assay_col)
            call = _clean_call(raw)

            # Nur bei eindeutigem Genotyp weitermachen
            if isinstance(call, float) and np.isnan(call):
                continue

            allelic_state = None
            if call in ("C", "CC"):
                # homozygous risk
                allelic_state = {"id": "GENO:0000136", "label": "homozygous"}
            elif call in ("CT", "TC"):
                # heterozygous risk
                allelic_state = {"id": "GENO:0000135", "label": "heterozygous"}
            elif call in ("T", "TT"):
                # homozygous normal/reference
                allelic_state = {"id": "GENO:0000134", "label": "homozygous reference"}
            else:
                # unklarer Wert → keinen Interpretationseintrag anlegen
                continue

            # Interpretation-Block nach Zielstruktur (Enum-konform)
            interp_dict = {
                "id": f"cfh-rs1061170-{pid}",
                "progressStatus": "SOLVED",
                "diagnosis": {
                    "disease": {"id": "NCBIGene:3075", "label": "CFH"},
                    "genomicInterpretations": [
                        {
                            "subjectOrBiosampleId": pid,
                            "interpretationStatus": "CONTRIBUTORY",
                            "variantInterpretation": {
                                "variationDescriptor": {
                                    "expressions": [{"syntax": "dbSNP", "value": "rs1061170"}],
                                    "geneContext": {"id": "NCBIGene:3075", "symbol": "CFH"},
                                    "allelicState": allelic_state,
                                    "extensions": [
                                        {"name": "assay_id", "value": assay_col},
                                        {"name": "observed_call_raw", "value": str(raw)},
                                    ]
                                }
                            }
                        }
                    ]
                }
            }

            # Protobuf erzeugen & anhängen
            try:
                new_interp = PPKt.Interpretation()
                ParseDict(interp_dict, new_interp, ignore_unknown_fields=True)
                if getattr(ppkt, "interpretations", None) is None:
                    ppkt.interpretations = []
                ppkt.interpretations.append(new_interp)
            except Exception as e:
                print(f"Could not append rs1061170 interpretation for {pid}: {e}")