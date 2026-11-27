"""Script para executar o notebook Jupyter e persistir todos os outputs nas células."""

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import pandas as pd


def executar_notebook(caminho_notebook: Path | str) -> None:
    caminho = Path(caminho_notebook)
    with open(caminho, mode="r", encoding="utf-8") as f:
        nb = json.load(f)

    # Contexto global persistente entre células
    globs: dict[str, object] = {
        "__name__": "__main__",
    }

    # Armazenará displays manuais chamados via display()
    displays_atuais: list[dict] = []

    def mock_display(*args: object) -> None:
        for arg in args:
            data: dict[str, list[str]] = {}
            if hasattr(arg, "to_html"):
                html_str = arg.to_html()
                data["text/html"] = [line + "\n" for line in html_str.split("\n")]
                data["text/plain"] = [line + "\n" for line in str(arg).split("\n")]
            elif isinstance(arg, pd.io.formats.style.Styler):
                html_str = arg.to_html()
                data["text/html"] = [line + "\n" for line in html_str.split("\n")]
                data["text/plain"] = [line + "\n" for line in str(arg.data).split("\n")]
            else:
                data["text/plain"] = [line + "\n" for line in repr(arg).split("\n")]

            displays_atuais.append({
                "output_type": "display_data",
                "data": data,
                "metadata": {},
            })

    globs["display"] = mock_display

    exec_counter = 1
    total_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"Executando {total_code_cells} células de código do notebook: {caminho.name}...")

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue

        codigo = "".join(cell["source"])
        print(f"  -> Executando célula {exec_counter}/{total_code_cells}...")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        displays_atuais.clear()

        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(codigo, globs)

        cell["execution_count"] = exec_counter
        cell_outputs: list[dict] = []

        # 1. Stdout
        stdout_text = stdout_buf.getvalue()
        if stdout_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stdout",
                "text": [line + "\n" for line in stdout_text.splitlines()],
            })

        # 2. Stderr
        stderr_text = stderr_buf.getvalue()
        if stderr_text:
            cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": [line + "\n" for line in stderr_text.splitlines()],
            })

        # 3. Displays chamados via display(...)
        for disp in displays_atuais:
            cell_outputs.append(disp)

        cell["outputs"] = cell_outputs
        exec_counter += 1

    # Salva o notebook atualizado com todos os outputs
    with open(caminho, mode="w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print(f"Notebook executado e salvo com sucesso: {caminho.resolve()}\n")


if __name__ == "__main__":
    caminhos = [
        Path("pc_parts_cleaning/pc_part_dataset_cleaning.ipynb"),
        Path("pc_part_dataset_cleaning.ipynb"),
    ]
    for p in caminhos:
        if p.exists():
            executar_notebook(p)
