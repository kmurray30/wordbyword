"""One-time setup: installs the Argos Translate en<->es language packages.
Run after `pip install -r requirements.txt` and
`python -m spacy download es_core_news_sm`.
"""

import argostranslate.package


def main() -> None:
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    installed = {(p.from_code, p.to_code) for p in argostranslate.package.get_installed_packages()}

    wanted = [("en", "es"), ("es", "en")]
    for from_code, to_code in wanted:
        if (from_code, to_code) in installed:
            print(f"already installed: {from_code} -> {to_code}")
            continue
        pkg = next((p for p in available if p.from_code == from_code and p.to_code == to_code), None)
        if pkg is None:
            print(f"WARNING: no package found for {from_code} -> {to_code}")
            continue
        print(f"installing: {from_code} -> {to_code}")
        argostranslate.package.install_from_path(pkg.download())


if __name__ == "__main__":
    main()
