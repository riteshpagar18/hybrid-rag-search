from pathlib import Path


def read_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {path.resolve()}"
        )

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent

    file_path = (
        project_root
        / "data"
        / "documents"
        / "employee_policy.txt"
    )

    text = read_document(file_path)

    print(text)