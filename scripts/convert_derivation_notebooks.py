#!/usr/bin/env python3
"""Build readable editions of the 21 original Wolfram coefficient notebooks.

This deliberately bounded converter preserves original content cells verbatim
and understands only the box types occurring in this source collection. It
does not evaluate Wolfram Language or treat cached results as fresh execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DERIVATIONS = ROOT / "notebooks" / "derivation"
STRING = re.compile(r'"(?:\\[\s\S]|[^"\\])*"')
TEX = re.compile(r'"input"\s*->\s*("(?:\\[\s\S]|[^"\\])*")')
CORRECTIONS = {
    "NN/Coefficient_B02": "The original mixed NN/phi-phi label and observer-transpose interval subscript are display errors. The corrected reading uses phi-phi and interval 0 for the observer; the direct integral and exported coefficient are unchanged.",
    "NN/Coefficient_B03": "One original displayed chi1 denominator has an extra closing TeX brace. The reading edition repairs that brace; the exact source remains below.",
    "NS/Coefficient_B04": "The displayed normalized redshift was called y, although the integrals, export and implementations call it z. The reading edition consistently uses z.",
    "NS/Coefficient_B07": "The source defines normalized power p, but its I7 input/output use y and its assumptions name p. The reading edition maps power y to p throughout. The clean SymPy calculation below checks the corrected integral against the unchanged equation export; no Wolfram execution is claimed.",
    "NS/Coefficient_B08": "The source consistently calls normalized power y. This reading edition renames it p, matching the export and implementations; the integration variable x is retained. This is a symbol rename, not an algebraic correction.",
    "SS/Coefficient_B06": "The dimensional observer display refers to J7, while its integral, input, export and implementations use J6. The reading edition corrects that cross-reference to J6.",
    "SS/Coefficient_B09": "The I9 input's assumption list contains y although the defined power parameter and integrand use p. The corrected input uses p. The clean SymPy calculation below verifies the unchanged equation export independently of cached Wolfram symbol state.",
}


def split_top(text: str) -> list[str]:
    """Split comma-delimited Wolfram expressions, respecting strings/brackets."""
    chunks, start, stack, index = [], 0, [], 0
    while index < len(text):
        char = text[index]
        if char == '"':
            match = STRING.match(text, index)
            if match is None:
                raise ValueError("Unterminated Wolfram string")
            index = match.end()
            continue
        if text.startswith("<|", index):
            stack.append("|>")
            index += 2
            continue
        if text.startswith("|>", index) and stack[-1:] == ["|>"]:
            stack.pop()
            index += 2
            continue
        if char in "[{(":
            stack.append({"[": "]", "{": "}", "(": ")"}[char])
        elif char in "]})":
            if not stack or stack.pop() != char:
                raise ValueError("Unbalanced Wolfram brackets")
        elif char == "," and not stack:
            chunks.append(text[start:index].strip())
            start = index + 1
        index += 1
    if stack:
        raise ValueError("Unclosed Wolfram brackets")
    chunks.append(text[start:].strip())
    return chunks


def call(text: str) -> tuple[str, list[str]]:
    text = text.strip()
    match = re.match(r"([A-Za-z][A-Za-z0-9]*)\[", text)
    if match and text.endswith("]"):
        return match[1], split_top(text[match.end():-1])
    if text.startswith("{") and text.endswith("}"):
        return "List", split_top(text[1:-1])
    return "", [text]


def unquote(text: str) -> str:
    text = re.sub(r"\\\r?\n", "", text[1:-1])
    return re.sub(
        r'\\([\\"nrt])',
        lambda match: {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}[match[1]],
        text,
    )


def box(text: str, latex: bool) -> str:
    name, args = call(text)
    if not name:
        value = unquote(args[0]) if args[0].startswith('"') else args[0]
        if latex:
            replacements = {
                r"\[Chi]": r"\chi ", r"\[Phi]": r"\phi ",
                r"\[Kappa]": r"\kappa ", r"\[ScriptL]": r"\ell ",
                r"\[CapitalOmega]": r"\Omega ", r"\[Integral]": r"\int ",
                r"\[LongEqual]": "=", r"\[Element]": r"\in ",
                r"\[DoubleStruckCapitalR]": r"\mathbb{R}",
            }
            for old, new in replacements.items():
                value = value.replace(old, new)
            if value == "Log":
                return r"\log "
            if value == " ":
                return r"\,"
        return value
    if name in {"BoxData", "FormBox", "StyleBox", "Cell", "TextData"}:
        return box(args[0], latex)
    if name in {"RowBox", "List"}:
        return "".join(box(item, latex) for item in (call(args[0])[1] if name == "RowBox" else args))
    if name == "FractionBox":
        upper, lower = (box(item, latex) for item in args[:2])
        return rf"\frac{{{upper}}}{{{lower}}}" if latex else f"({upper})/({lower})"
    if name in {"SuperscriptBox", "SubscriptBox", "SubsuperscriptBox"}:
        values = [box(item, latex) for item in args]
        if latex:
            base = "{" + values[0] + "}"
            if name != "SuperscriptBox":
                base += "_{" + values[1] + "}"
            if name != "SubscriptBox":
                base += "^{" + values[-1] + "}"
            return base
        if name == "SuperscriptBox":
            return f"({values[0]})^({values[1]})"
        raise ValueError(f"Unexpected executable box {name}")
    raise ValueError(f"Unsupported box {name}: {text[:100]}")


def source_cells(source: str) -> list[tuple[str, list[int]]]:
    content = source.split("(* Beginning of Notebook Content *)", 1)[1]
    content = content.split("(* End of Notebook Content *)", 1)[0].strip()
    name, notebook = call(content)
    if name != "Notebook":
        raise ValueError("Expected one notebook expression")
    result = []

    def visit(expression: str, group: list[int]) -> None:
        name, args = call(expression)
        if name != "Cell":
            raise ValueError("Expected a content cell")
        child, children = call(args[0])
        if child == "CellGroupData":
            for index, cell in enumerate(call(children[0])[1], 1):
                visit(cell, [*group, index])
        else:
            result.append((expression, group))

    for index, expression in enumerate(call(notebook[0])[1], 1):
        visit(expression, [index])
    return result


def corrected(text: str, key: str) -> str:
    if key in {"NS/Coefficient_B07", "NS/Coefficient_B08", "SS/Coefficient_B09"}:
        text = re.sub(r"\by\b", "p", text)
    if key == "NS/Coefficient_B04":
        text = re.sub(r"\by\b", "z", text)
    if key == "SS/Coefficient_B06":
        text = text.replace("J_7", "J_6")
    if key == "NN/Coefficient_B02":
        text = text.replace(r"B_{\mathrm{NN}, n}", r"B_{\phi\phi,n}")
        text = text.replace("_{NN,n}", r"_{\phi \phi,n}")
        if "i = 1, j = 0" in text:
            text = text.replace(r"\phi \phi,1", r"\phi \phi,0")
            text = text.replace(r"\phi \phi, 1", r"\phi \phi, 0")
    if key == "NN/Coefficient_B03":
        text = text.replace(r"\chi_{1}}}", r"\chi_{1}}")
        text = text.replace(r"\chi_1}}", r"\chi_1}")
    return text


def markdown(source: str, **metadata: object) -> dict:
    return {"cell_type": "markdown", "metadata": metadata, "source": source.splitlines(keepends=True)}


def symbolic_check(key: str) -> str:
    if key == "NS/Coefficient_B07":
        formula = "(b + c*t)*(1-t)/(a*t)*Q*Z"
        symbols = "b, c, p, z = sympy.symbols('b c p z', real=True)"
        observer = "(b+c*t)*(1-t)/t*t**3*(1-z*(1-t))"
        family, number = "NS", 7
    elif key == "SS/Coefficient_B09":
        formula = "(b + c*t)*(d + e*t)*Q*Z**2"
        symbols = "b, c, d, e, p, z = sympy.symbols('b c d e p z', real=True)"
        observer = "(b+c*t)*(d+e*t)*t**3*(1-z*(1-t))**2"
        family, number = "SS", 9
    else:
        return ""
    exports = (DERIVATIONS / family / f"Coefficient_B{number:02d}.txt").read_text().strip()
    return f'''# Corrected Python recomputation: explicit symbols, no Wolfram state.
import sympy
from IPython.display import display

t = sympy.Symbol('t', positive=True)
a = sympy.Symbol('a', positive=True)
{symbols}
# Ordinary interval: 0 < a < 1; all other parameters are real.
Q = 1-p*(1-t)/a
Z = 1-z*(1-t)/a
INTEGRAND = {formula}
# Integrate an explicit Laurent polynomial; t and 1-a are positive.
ANTIDERIVATIVE = sympy.integrate(sympy.expand(INTEGRAND), t)
INTEGRAL = ANTIDERIVATIVE.subs(t, 1)-ANTIDERIVATIVE.subs(t, 1-a)
OBSERVER = sympy.integrate(sympy.expand({observer}), (t, 0, 1))
# Exact unchanged .txt export copied for a data-independent notebook.
EXPORT_TEXT = {exports!r}
EXPORTS = {{}}
for LINE in EXPORT_TEXT.split('\\n\\n'):
    if '=' in LINE:
        NAME, EXPRESSION = LINE.split('=', 1)
        EXPORTS[NAME.strip()] = sympy.sympify(
            EXPRESSION.strip().replace('\\n', ' ').replace('numpy.log', 'log').replace('^', '**'),
            locals={{'a': a, **{{symbol.name: symbol for symbol in INTEGRAND.free_symbols}}, 'log': sympy.log}},
        )
ORDINARY_RESIDUAL = sympy.simplify(INTEGRAL-EXPORTS['I{number}'])
OBSERVER_RESIDUAL = sympy.simplify(OBSERVER-EXPORTS['J{number}'])
assert ORDINARY_RESIDUAL == 0 and OBSERVER_RESIDUAL == 0
display(INTEGRAL, OBSERVER, ORDINARY_RESIDUAL, OBSERVER_RESIDUAL)
'''


def main() -> None:
    inventory, totals = [], Counter()
    for family, count in (("NN", 3), ("NS", 8), ("SS", 10)):
        for number in range(1, count+1):
            path = DERIVATIONS / family / f"Coefficient_B{number:02d}.nb"
            source = path.read_bytes().decode("utf-8")
            key = f"{family}/{path.stem}"
            cells = [markdown(
                f"# {family}: Coefficient B{number:02d}\n\n"
                "Readable edition of the original Mathematica derivation. Scientific cells remain in source order. "
                "Normalized power uses $p=1-P_1/P_2$ in the reading edition; original spellings and complete source cells "
                "are preserved in the appendix. Cached Wolfram outputs are transcribed results, not a fresh execution.\n\n"
                f"[Original Mathematica notebook]({path.name}) · [Conversion inventory and notation ledger](../README.md)\n\n"
                "The displayed dimensional integrals, definitions, normalized I/J integrals, inputs and outputs "
                "follow the author's original sequence. No runtime data or Mathematica installation is needed to read this notebook."
            )]
            if key in CORRECTIONS:
                cells.append(markdown("**Verified notation note.** " + CORRECTIONS[key]))
            records, originals, counts = [], [], Counter()
            for ordinal, (raw, group) in enumerate(source_cells(source), 1):
                _, args = call(raw)
                style = unquote(args[1].lstrip("\\\r\n "))
                counts[style] += 1
                source_id = f"source-{ordinal:02d}"
                line = source[:source.index(raw)].count("\n")+1
                metadata = {"source_cell": ordinal, "source_group": group, "source_style": style}
                prefix = f"[Source cell {ordinal}](#{source_id}), `{style}`.\n\n"
                if style == "InlineFormula":
                    equations = [unquote(match[1]) for match in TEX.finditer(args[0])]
                    if not equations:
                        raise ValueError(f"No TeX source in {key} cell {ordinal}")
                    rendered = "\n\n".join("$$\n" + corrected(equation, key) + "\n$$" for equation in equations)
                    label = ""
                elif style in {"Input", "Output"}:
                    plain = box(args[0], latex=False)
                    if style == "Input":
                        label = "### Symbolic input\n\n"
                        rendered = "Corrected notation; Wolfram Language shown as text.\n\n```wolfram\n" + corrected(plain, key) + "\n```"
                    else:
                        label = "### Saved symbolic output\n\n"
                        rendered = "Transcribed cached result, with the noted symbol renaming only.\n\n$$\n" + corrected(box(args[0], latex=True), key) + "\n$$"
                    equations = []
                else:
                    raise ValueError(f"Unexpected source cell style {style}")
                target_cell = len(cells)
                cells.append(markdown(label + prefix + rendered, **metadata))
                originals.append(markdown(
                    f'<a id="{source_id}"></a>\n\n<details>\n<summary>Original source cell {ordinal}; '
                    f'{style}; source line {line}; group {".".join(map(str, group))}</summary>\n\n'
                    f"```wolfram\n{raw}\n```\n\n</details>",
                    **metadata,
                ))
                records.append({
                    "ordinal": ordinal, "style": style, "group": group, "source_line": line,
                    "source_cell_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "reading_cell_index": target_cell, "tex_expression_count": len(equations),
                })
            code = symbolic_check(key)
            if code:
                cells.append(markdown("## Corrected symbolic equivalence check\n\nOptional clean Python/SymPy recomputation. This checks the ordinary and observer integrals against the existing exports; it does not validate their matrix-placement domain."))
                cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code.splitlines(keepends=True)})
            cells.append(markdown("## Exact original content cells\n\nAll source content cells follow in their original order, including original spellings, assumptions, saved outputs and cell metadata. Frontend-level metadata remains in the unchanged `.nb` file. Expand an entry to inspect its Wolfram box source."))
            for record, original in zip(records, originals, strict=True):
                record["original_cell_index"] = len(cells)
                cells.append(original)
            target = path.with_suffix(".ipynb")
            notebook = {
                "cells": cells,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                    "limbercloud": {"kind": "symbolic_derivation", "source": path.name,
                                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
                },
                "nbformat": 4, "nbformat_minor": 4,
            }
            target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n")
            totals.update(counts)
            inventory.append({"source": str(path.relative_to(ROOT)), "target": str(target.relative_to(ROOT)),
                              "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                              "counts": dict(counts), "cells": records})
    expected = {"InlineFormula": 146, "Input": 42, "Output": 42}
    if totals != expected:
        raise ValueError(f"Content inventory changed: {dict(totals)} != {expected}")
    manifest = {"format": "limbercloud-wolfram-edition-v1", "wolfram_executed": False,
                "source_notebook_count": len(inventory), "content_cell_totals": dict(totals),
                "notebooks": inventory}
    (DERIVATIONS / "conversion_inventory.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Converted {len(inventory)} notebooks; preserved {dict(totals)}.")


if __name__ == "__main__":
    main()
