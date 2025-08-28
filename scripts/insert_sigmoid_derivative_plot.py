#!/usr/bin/env python3
"""
Insert sigmoid and its derivative plot cells into a Jupyter notebook
right after the existing Sigmoid section and plot.

- Creates a backup: <notebook>.pre_sigmoid_deriv_plot.bak
- Inserts two cells:
  1) markdown: short explanation
  2) code: plots sigma and sigma'

Usage:
  python scripts/insert_sigmoid_derivative_plot.py notebooks/math_intro.ipynb
"""
from __future__ import annotations
import json
import sys
import os
import shutil
from typing import Tuple, List, Any


def load_notebook(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(nb: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")


def find_sigmoid_section_and_plot_index(cells: List[dict]) -> Tuple[int, int]:
    """Return indices (sigmoid_markdown_idx, sigmoid_plot_code_idx).

    We look for a markdown cell starting with '## 2) Sigmoid' and then the first
    subsequent code cell, which is assumed to be the existing sigmoid plot.
    Raises ValueError if not found.
    """
    sig_md_idx = -1
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            if src.strip().startswith('## 2) Sigmoid'):
                sig_md_idx = i
                break
    if sig_md_idx == -1:
        raise ValueError("Sigmoid markdown section '## 2) Sigmoid' not found.")

    # find the first code cell after the markdown
    for j in range(sig_md_idx + 1, len(cells)):
        if cells[j].get('cell_type') == 'code':
            return sig_md_idx, j

    raise ValueError("No code cell found after the Sigmoid markdown section.")


def build_markdown_cell() -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Sigmoid and its derivative\n",
            "\n",
            "We now plot the sigmoid $\\sigma(z)$ together with its derivative $\\sigma'(z)=\\sigma(z)(1-\\sigma(z))$. ",
            "This shows the steepest slope at $z=0$ and saturation for large $|z|$.\n",
        ],
    }


def build_code_cell() -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": [
            "# Plot sigmoid and its derivative\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "z = np.linspace(-6, 6, 400)\n",
            "sig = 1/(1+np.exp(-z))\n",
            "sig_prime = sig * (1 - sig)\n",
            "\n",
            "plt.figure(figsize=(5,3))\n",
            "plt.plot(z, sig, label='σ(z)')\n",
            "plt.plot(z, sig_prime, label=\"σ'(z)\")\n",
            "plt.axvline(0, color='gray', linestyle='--', linewidth=1)\n",
            "plt.xlabel('z')\n",
            "plt.ylabel('value')\n",
            "plt.title('Sigmoid and its derivative')\n",
            "plt.ylim(-0.05, 1.05)\n",
            "plt.grid(alpha=0.3)\n",
            "plt.legend()\n",
            "plt.show()\n",
        ],
    }


def already_inserted(cells: List[dict]) -> bool:
    for cell in cells:
        if cell.get('cell_type') == 'markdown':
            src = ''.join(cell.get('source', []))
            if 'Sigmoid and its derivative' in src:
                return True
    return False


def insert_cells(nb_path: str) -> None:
    nb = load_notebook(nb_path)
    cells: List[dict] = nb.get('cells', [])

    if already_inserted(cells):
        print("Cells already present; no changes made.")
        return

    sig_md_idx, sig_plot_idx = find_sigmoid_section_and_plot_index(cells)

    md_cell = build_markdown_cell()
    code_cell = build_code_cell()

    insert_pos = sig_plot_idx + 1
    cells[insert_pos:insert_pos] = [md_cell, code_cell]

    backup = nb_path + ".pre_sigmoid_deriv_plot.bak"
    shutil.copy2(nb_path, backup)
    save_notebook(nb, nb_path)

    print(f"Inserted 2 cells after index {sig_plot_idx}. Backup saved to: {backup}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python scripts/insert_sigmoid_derivative_plot.py <path-to-notebook>")
        sys.exit(1)
    target = sys.argv[1]
    if not os.path.exists(target):
        print(f"Notebook not found: {target}")
        sys.exit(1)
    insert_cells(target)
