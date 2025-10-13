from pathlib import Path

import mkdocs_gen_files


src_root = Path("./agentflow")
for path in src_root.glob("**/*.py"):
    if path.stem == "__init__":
        rel_parent = path.parent.relative_to(src_root)
        if rel_parent.name == "":
            doc_path = Path("reference", "index.md")
            ident = "agentflow"
        else:
            doc_path = Path("reference", rel_parent, "index.md")
            ident = "agentflow." + ".".join(rel_parent.parts)
    else:
        doc_path = Path("reference", path.relative_to(src_root)).with_suffix(".md")
        ident = "agentflow." + ".".join(path.with_suffix("").relative_to(src_root).parts)

    with mkdocs_gen_files.open(doc_path, "w") as f:
        print("::: " + ident, file=f)
