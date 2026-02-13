from pathlib import Path


def test_report_theory_sections_and_equations_exist():
    report_tex = Path(__file__).resolve().parents[2] / "report" / "assignment_template.tex"
    text = report_tex.read_text(encoding="utf-8")

    required_sections = [
        r"\section{Complete Theoretical Foundations}",
        r"\subsection{Security Theory: Backdoor Model and Neural Cleanse}",
        r"\subsection{Privacy Theory: Differential Privacy and Laplace Mechanism}",
        r"\subsection{Fairness Theory: Metrics and Mitigation Principles}",
        r"\subsection{Theory Robustness Guardrails}",
    ]
    for item in required_sections:
        assert item in text

    required_equation_tokens = [
        r"\mathcal{T}(x; m, p)",
        r"\mathrm{MAD}(s)",
        r"\mathrm{ASR}",
        r"\Pr[\mathcal{M}(D)\in S]",
        r"\tilde{q}(D)=q(D)+\eta",
        r"\Delta f_{\mathrm{unbounded}}",
        r"\mathrm{DI}",
        r"w(s,y)=",
    ]
    for token in required_equation_tokens:
        assert token in text


def test_report_contains_extended_understanding_figures():
    report_tex = Path(__file__).resolve().parents[2] / "report" / "assignment_template.tex"
    text = report_tex.read_text(encoding="utf-8")

    required_figures = [
        "security_unlearning_sweep.png",
        "privacy_epsilon_sweep.png",
        "fairness_swapk_sweep.png",
        "security_confusion_before_after.png",
        "privacy_tail_curves.png",
        "fairness_tradeoff.png",
    ]
    for fig in required_figures:
        assert fig in text
