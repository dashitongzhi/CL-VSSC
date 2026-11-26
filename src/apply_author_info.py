"""Stamp personal author/affiliation/ORCID/funding fields from a JSON profile into all manuscript & submission files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_KEYS = [
    "author_full_name",
    "author_surname",
    "author_given_name",
    "author_title",
    "department",
    "university",
    "city",
    "postal_code",
    "country",
    "email",
    "orcid",
]


def _apply_replacements(text: str, p: dict, is_tex: bool) -> str:
    funding = p.get("funding_note", "").strip()
    if not funding or "TODO" in funding or "[Grant" in funding:
        funding = "This research received no specific external grant from funding agencies."
    date_str = p.get("submission_date", "November 27, 2026")
    title_dept = f"{p['author_title']}, {p['department']}, {p['university']}"
    city_line = f"{p['city']}, {p['postal_code']}, {p['country']}"

    replacements = [
        ("[TODO(PERSONAL_INFO): Date, e.g., November 27, 2026]", date_str),
        (r"[TODO(PERSONAL\_INFO): Date, e.g., November 27, 2026]", date_str),
        ("[TODO(PERSONAL_INFO): Date, e.g., November 26, 2026]", date_str),
        ("[TODO(PERSONAL_INFO): Corresponding Author Full Name]", p["author_full_name"]),
        ("[TODO(PERSONAL_INFO): Author Full Name]", p["author_full_name"]),
        (r"[TODO(PERSONAL\_INFO): Author Full Name]", p["author_full_name"]),
        ("[TODO(PERSONAL_INFO): Author 1 Name]", p["author_full_name"]),
        ("[TODO(PERSONAL_INFO): Title, Department / School, University / Institution]", title_dept),
        (r"[TODO(PERSONAL\_INFO): Title, Department / School, University / Institution]", title_dept),
        ("[TODO(PERSONAL_INFO): City, Postal Code, Country]", city_line),
        (r"[TODO(PERSONAL\_INFO): City, Postal Code, Country]", city_line),
        ("[TODO(PERSONAL_INFO): author@institution.edu]", p["email"]),
        (r"[TODO(PERSONAL\_INFO): author@institution.edu]", p["email"]),
        ("[TODO(PERSONAL_INFO): 0000-0000-0000-0000]", p["orcid"]),
        (r"[TODO(PERSONAL\_INFO): 0000-0000-0000-0000]", p["orcid"]),
        ("[TODO(PERSONAL_INFO): November 27, 2026]", date_str),
        (r"[TODO(PERSONAL\_INFO): November 27, 2026]", date_str),
        (
            "[TODO(PERSONAL_INFO): Insert grant agency and grant number, OR state \"This research received no specific external grant from funding agencies in the public, commercial, or not-for-profit sectors.\"]",
            funding,
        ),
        (
            r"[TODO(PERSONAL\_INFO): Insert grant agency and grant number, OR state ``This research received no specific external grant from funding agencies in the public, commercial, or not-for-profit sectors.'']",
            funding,
        ),
        ("This work was supported in part by [Grant / Funding Agency, if applicable, or delete this sentence].", funding),
        ("[Author Full Name]", p["author_full_name"]),
        ("[Author Surname]", p["author_surname"]),
        ("[Author Given Name]", p["author_given_name"]),
        ("[Department / School]", p["department"]),
        ("[University / Institution]", p["university"]),
        ("[City] [Postal Code], [Country]", f"{p['city']} {p['postal_code']}, {p['country']}"),
        ("[author@institution.edu]", p["email"]),
        ("[0000-0000-0000-0000]", p["orcid"]),
        ("0000-0000-0000-0000", p["orcid"]),
        ("author = {Author, Corresponding}", f"author = {{{p.get('paper1_authors', p['author_surname'] + ', ' + p['author_given_name'])}}}"),
        (
            "note   = {Companion manuscript (MIVS-CL; provided as supplementary material for review)}",
            f"note   = {{{p.get('paper1_note', 'Companion manuscript (MIVS-CL; provided as supplementary material for review)')}}}",
        ),
    ]
    out = text
    for old, new in replacements:
        out = out.replace(old, new)
    if is_tex:
        out = out.replace("% TODO(PERSONAL_INFO):", "% Author metadata stamped via src/apply_author_info.py:")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply author profile JSON across Paper 2 TeX and submission files.")
    parser.add_argument("--profile", required=True, type=Path, help="Path to filled author_profile.json")
    parser.add_argument("--dry-run", action="store_true", help="Report target files without modifying them")
    parser.add_argument("--build", action="store_true", help="Rebuild all submission PDFs and run verify_paper.py after stamping")
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text())
    for k in REQUIRED_KEYS:
        val = str(profile.get(k, "")).strip()
        if not val or "[" in val or "TODO" in val:
            raise ValueError(f"Profile field {k!r} still contains placeholder text: {val!r}")

    targets = [
        ROOT / "manuscript" / "main.tex",
        ROOT / "manuscript" / "supplementary.tex",
        ROOT / "manuscript" / "references.bib",
        ROOT / "CITATION.cff",
        ROOT / "submission" / "COVER_LETTER_IEEE.md",
        ROOT / "submission" / "DECLARATIONS_AND_METADATA.md",
        ROOT / "submission" / "cover_letter_ieee_tcyb.tex",
        ROOT / "submission" / "cover_letter_ieee_tnnls.tex",
        ROOT / "submission" / "cover_letter_information_fusion.tex",
        ROOT / "submission" / "declaration_of_interest.tex",
    ]

    for fp in targets:
        orig = fp.read_text()
        updated = _apply_replacements(orig, profile, is_tex=fp.suffix in (".tex", ".bib"))
        changed = updated != orig
        print(f"{'[DRY-RUN] ' if args.dry_run else ''}{'UPDATED' if changed else 'UNCHANGED'}: {fp.relative_to(ROOT)}")
        if changed and not args.dry_run:
            fp.write_text(updated)

    if args.build and not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "src" / "build_submission_bundle.py")], check=True)
        subprocess.run([sys.executable, str(ROOT / "src" / "verify_paper.py")], check=True)


if __name__ == "__main__":
    main()
