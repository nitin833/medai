import psycopg2
from collections import defaultdict

conn = psycopg2.connect(
    host="unmtid-dbs.net",
    port=5433,
    dbname="drugcentral",
    user="drugman",
    password="dosage"
)

cur = conn.cursor()

cur.execute("""
SELECT
    s.name,
    s.cd_formula,
    s.cd_molweight,
    s.smiles,
    s.inchi,
    i.indi_pt
FROM structures s
JOIN "SOL_PHARMACOVIGILANCE_qzJrh0Bu_Drugs_PS_indications" i
    ON LOWER(TRIM(s.name)) = LOWER(TRIM(i.drugname))
ORDER BY s.name;
""")

rows = cur.fetchall()

print("Rows fetched:", len(rows))

drugs = defaultdict(lambda: {
    "formula": "",
    "weight": "",
    "smiles": "",
    "inchi": "",
    "diseases": set()
})

for name, formula, weight, smiles, inchi, disease in rows:
    drugs[name]["formula"] = formula
    drugs[name]["weight"] = weight
    drugs[name]["smiles"] = smiles
    drugs[name]["inchi"] = inchi

    if disease:
        drugs[name]["diseases"].add(disease)

print("Unique drugs:", len(drugs))

outfile = "DrugCentral_KnowledgeBase.txt"

with open(outfile, "w", encoding="utf-8") as f:

    for drug in sorted(drugs):

        info = drugs[drug]

        f.write(f"Drug: {drug}\n")
        f.write(f"Formula: {info['formula']}\n")
        f.write(f"Molecular Weight: {info['weight']}\n")
        f.write(f"SMILES: {info['smiles']}\n")
        f.write(f"InChI: {info['inchi']}\n")

        f.write("Diseases:\n")

        for disease in sorted(info["diseases"]):
            f.write(f"  - {disease}\n")

        f.write("\n")
        f.write("=" * 80)
        f.write("\n\n")

print(f"Saved {len(drugs)} drugs to {outfile}")

cur.close()
conn.close()