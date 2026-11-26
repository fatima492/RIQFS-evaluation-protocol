"""plain_tables.py -- copies outputs/tables/*.tex to outputs/tables_plain/ with programme acronyms (PRS, TRN, ...) in
place of the index symbols omega_1..omega_5, for readers of the manuscript."""
import os, re, glob
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "outputs", "tables"); DST = os.path.join(ROOT, "outputs", "tables_plain")
os.makedirs(DST, exist_ok=True)
ACR = {"1": "TRN", "2": "JSA", "3": "PRS", "4": "PSE", "5": "OTH"}
for f in glob.glob(os.path.join(SRC, "*.tex")):
    s = open(f).read()
    s = re.sub(r"\$\\omega_\{?(\d)\}?\$ & (\w+) & ", lambda m_: f"{m_.group(2)} & ", s)          # data table: drop the index column
    s = re.sub(r"\$\\omega_\{?(\d)\}?\$ (\w+)", lambda m_: m_.group(2), s)                       # row labels "omega_1 TRN" -> "TRN"
    s = re.sub(r"\\omega_\{?(\d)\}?", lambda m_: "\\mathrm{" + ACR[m_.group(1)] + "}", s)        # rankings
    s = s.replace("\\mathrm{PRS} over \\mathrm{JSA}", "PRS over JSA")
    open(os.path.join(DST, os.path.basename(f)), "w").write(s)
print("plain tables written:", len(glob.glob(os.path.join(DST, "*.tex"))))
