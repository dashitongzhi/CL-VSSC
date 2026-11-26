"""Compile main IEEE PDF, Supplementary Material PDF, venue-specific PDF Cover Letters, and ScholarOne LaTeX source zip."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MS_DIR = ROOT / "manuscript"
SUB_DIR = ROOT / "submission"

EXTRA_DYLD = ":".join(
    [
        "/opt/homebrew/Cellar/harfbuzz/15.2.0/lib",
        "/opt/homebrew/opt/freetype/lib",
        "/opt/homebrew/opt/graphite2/lib",
        "/opt/homebrew/opt/glib/lib",
        "/opt/homebrew/opt/icu4c@79/lib",
    ]
)


def _run_tectonic(tex_name: str, cwd: Path) -> None:
    env = os.environ.copy()
    prev_dyld = env.get("DYLD_LIBRARY_PATH", "")
    env["DYLD_LIBRARY_PATH"] = f"{EXTRA_DYLD}:{prev_dyld}" if prev_dyld else EXTRA_DYLD
    tectonic_bin = shutil.which("tectonic") or "/opt/homebrew/bin/tectonic"
    subprocess.run([tectonic_bin, "--keep-logs", tex_name], cwd=str(cwd), env=env, check=True)


def main() -> None:
    # 1. Main IEEE manuscript & Supplementary Material
    _run_tectonic("main.tex", MS_DIR)
    shutil.copy2(MS_DIR / "main.pdf", ROOT / "CL-VSSC_ClosedLoop_Control_IEEE.pdf")

    _run_tectonic("supplementary.tex", MS_DIR)
    shutil.copy2(MS_DIR / "supplementary.pdf", ROOT / "CL-VSSC_Supplementary_IEEE.pdf")

    # 2. Venue-specific Cover Letters, Highlights, and Declaration PDFs
    sub_targets = [
        ("cover_letter_ieee_tcyb.tex", "Cover_Letter_IEEE_TCYB.pdf"),
        ("cover_letter_ieee_tnnls.tex", "Cover_Letter_IEEE_TNNLS.pdf"),
        ("cover_letter_information_fusion.tex", "Cover_Letter_Information_Fusion.pdf"),
        ("highlights_elsevier.tex", "Highlights_Elsevier.pdf"),
        ("declaration_of_interest.tex", "Declaration_of_Interest_Elsevier.pdf"),
    ]
    for tex_fn, out_pdf in sub_targets:
        _run_tectonic(tex_fn, SUB_DIR)
        raw_pdf = SUB_DIR / tex_fn.replace(".tex", ".pdf")
        dst_pdf = SUB_DIR / out_pdf
        if raw_pdf.resolve() != dst_pdf.resolve():
            shutil.move(str(raw_pdf), str(dst_pdf))

    # 3. Clean ScholarOne / Editorial Manager LaTeX source bundle zip
    zip_path = SUB_DIR / "IEEE_LaTeX_Source_Submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in ("main.tex", "supplementary.tex", "references.bib"):
            zf.write(MS_DIR / fn, arcname=fn)
        for fig_p in sorted((MS_DIR / "figures").glob("*.pdf")):
            zf.write(fig_p, arcname=f"figures/{fig_p.name}")
        for tab_p in sorted((MS_DIR / "tables").glob("*.tex")):
            zf.write(tab_p, arcname=f"tables/{tab_p.name}")

    print(
        f"Built turnkey submission bundle: CL-VSSC_ClosedLoop_Control_IEEE.pdf, "
        f"CL-VSSC_Supplementary_IEEE.pdf, {len(sub_targets)} submission PDFs, and {zip_path.name}"
    )


if __name__ == "__main__":
    main()
