"""
RecovAI — Engine 5: NLP Shift Report Generator
===============================================
Calls the Anthropic Claude API to generate a concise, plain-English
shift performance report from structured engine outputs.

The report is written in the voice of a metallurgical process expert:
factual, actionable, flagging anomalies, and recommending one clear
improvement action per shift.

Usage:
    from engines.engine5_nlp import ShiftReportGenerator
    gen = ShiftReportGenerator(api_key="sk-ant-...")   # or set ANTHROPIC_API_KEY env var

    report = gen.generate(
        shift_data     = shift_dict,
        predicted_rec  = 87.4,
        shap_result    = engine3_result,
        anomaly_result = engine2_result,
        reagent_result = engine1_result,
        drift_result   = engine4_result,
    )
    print(report["report"])
"""

import os
import anthropic
from typing import Dict, Optional
import json
from datetime import datetime


SYSTEM_PROMPT = """You are a senior metallurgical process expert and AI assistant at a copper concentrator plant.
Your role is to write concise, accurate shift performance reports for plant operators and metallurgists.

Your reports must be:
- 4–6 sentences, clearly structured
- Written in plain English that any operator can understand
- Factual — only state what the data shows
- Actionable — always end with ONE specific recommended action
- Alert-focused — flag anomalies and reagent mismatches clearly

Tone: professional, direct, not alarming unless truly necessary.
Never use jargon without a brief explanation.
Never make up data not provided to you."""


class ShiftReportGenerator:
    """
    Generates natural-language shift reports using the Claude API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 400,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key provided. Set ANTHROPIC_API_KEY env var "
                "or pass api_key= to ShiftReportGenerator()."
            )
        self.client     = anthropic.Anthropic(api_key=key)
        self.model      = model
        self.max_tokens = max_tokens

    # ── Main interface ────────────────────────────────────────────────────

    def generate(
        self,
        shift_data:     Dict[str, float],
        predicted_rec:  float,
        shap_result:    Dict,
        anomaly_result: Dict,
        reagent_result: Dict,
        drift_result:   Dict,
        shift_id:       Optional[str] = None,
    ) -> Dict:
        """
        Generate a complete shift report.

        Parameters
        ----------
        shift_data     : raw shift input dict (feature→value)
        predicted_rec  : XGBoost predicted Cu recovery (%)
        shap_result    : output from Engine 3 (ShapExplainer.explain_shift)
        anomaly_result : output from Engine 2 (AnomalyDetector.score_shift)
        reagent_result : output from Engine 1 (ReagentOptimizer.optimize)
        drift_result   : output from Engine 4 (PSIMonitor.check or check_single)
        shift_id       : optional shift identifier string

        Returns
        -------
        dict with:
            report        — str, the natural-language report
            shift_id      — str
            timestamp     — ISO UTC string
            model         — which Claude model was used
            prompt_tokens — int
            output_tokens — int
        """
        prompt = self._build_prompt(
            shift_data, predicted_rec, shap_result,
            anomaly_result, reagent_result, drift_result, shift_id
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        report_text = message.content[0].text.strip()

        return {
            "report":        report_text,
            "shift_id":      shift_id or datetime.utcnow().strftime("%Y%m%d_%H%M"),
            "timestamp":     datetime.utcnow().isoformat(),
            "model":         self.model,
            "prompt_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

    def generate_brief(
        self,
        shift_data:    Dict[str, float],
        predicted_rec: float,
        top_drivers:   list,
        anomaly_label: str,
    ) -> str:
        """
        Lightweight 2-sentence summary — for dashboard tooltips / push alerts.
        """
        drivers_str = ", ".join(
            f"{f} ({'+' if v > 0 else ''}{v:.2f} pp)" for f, v in top_drivers[:3]
        )
        prompt = (
            f"Write a 2-sentence shift alert for plant operators.\n"
            f"Recovery: {predicted_rec:.1f}%. "
            f"Top drivers: {drivers_str}. "
            f"Anomaly status: {anomaly_label}.\n"
            f"Be direct. Mention one action."
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    # ── Prompt construction ───────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        shift_data, predicted_rec, shap_result,
        anomaly_result, reagent_result, drift_result, shift_id
    ) -> str:

        # ── Shift inputs section ──────────────────────────────────────────
        key_inputs = {
            "Head Grade (%Cu)":       shift_data.get("Head Grade (%Cu)"),
            "Feed Rate (MT/h)":       shift_data.get("Feed Rate (MT/h)"),
            "Flotation pH":           shift_data.get("Flotation pH"),
            "Pulp Density (%)":       shift_data.get("Pulp Density (%)"),
            "Air Flow Rate (m3/min)": shift_data.get("Air Flow Rate (m3/min)"),
        }
        inputs_str = "\n".join(
            f"  - {k}: {v}" for k, v in key_inputs.items() if v is not None
        )

        # ── SHAP top drivers ──────────────────────────────────────────────
        top3 = shap_result.get("top_3_drivers", [])
        drivers_str = "\n".join(
            f"  {'▲' if v>0 else '▼'} {f}: {v:+.3f} pp impact on recovery"
            for f, v in top3
        )
        base_val = shap_result.get("base_value", "N/A")

        # ── Anomaly section ───────────────────────────────────────────────
        ano_score = anomaly_result.get("score", "N/A")
        ano_label = anomaly_result.get("label", "UNKNOWN")
        ano_top   = anomaly_result.get("top_contributors", [])
        ano_str   = "\n".join(
            f"  - {f}: z = {z:.2f}" for f, z in ano_top
        )

        # ── Reagent gaps section ──────────────────────────────────────────
        gaps      = reagent_result.get("gaps", {})
        gain      = reagent_result.get("recovery_gain", 0)
        reagent_str = "\n".join(
            f"  - {r}: actual={g['actual']} g/t, optimal={g['optimal']} g/t, "
            f"gap={g['gap_pct']:.1f}% [{g['label']}] → {g['action']}"
            for r, g in gaps.items()
        )

        # ── Drift section ─────────────────────────────────────────────────
        drift_status = drift_result.get("overall_status", "UNKNOWN")
        flagged      = drift_result.get("flagged", [])
        worst_psi    = drift_result.get("worst_psi", 0)
        drift_detail = (
            f"Flagged features: {', '.join(flagged)}" if flagged
            else "No features flagged."
        )

        prompt = f"""Please write a concise shift performance report for shift {shift_id or 'current'}.

=== SHIFT INPUTS ===
{inputs_str}

=== PREDICTION ===
  Predicted Cu Recovery: {predicted_rec:.2f}%
  Model base value (expected recovery): {base_val}%

=== TOP 3 SHAP DRIVERS ===
{drivers_str}

=== ANOMALY DETECTION ===
  Decision score: {ano_score} (threshold: -0.10 suspicious, -0.20 alert)
  Status: {ano_label}
  Most deviant features:
{ano_str}

=== REAGENT DOSE INTELLIGENCE ===
{reagent_str}
  Potential recovery gain if doses optimised: +{gain:.3f} pp

=== DATA DRIFT (PSI) ===
  Overall status: {drift_status} (worst PSI = {worst_psi:.4f})
  {drift_detail}

Write the report now. 4–6 sentences. End with exactly one recommended action."""

        return prompt


# ── Standalone test (requires ANTHROPIC_API_KEY) ─────────────────────────────
if __name__ == "__main__":
    import sys

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY to run this test.")
        sys.exit(0)

    print("Engine 5 — NLP Shift Report Generator\n" + "="*42)

    gen = ShiftReportGenerator(api_key=api_key)

    # Minimal mock inputs
    shift = {
        "Head Grade (%Cu)":       1.2,
        "Feed Rate (MT/h)":       120.0,
        "Flotation pH":           10.5,
        "Pulp Density (%)":       32.0,
        "Air Flow Rate (m3/min)": 14.0,
    }
    shap_result = {
        "base_value":    84.0,
        "prediction":    87.4,
        "top_3_drivers": [
            ("Head Grade (%Cu)",  2.3),
            ("Flotation pH",      1.1),
            ("SIPX Dose (g/t)",   0.8),
        ],
    }
    anomaly_result = {
        "score":             -0.04,
        "label":             "NORMAL",
        "top_contributors":  [("Flotation pH", 0.3), ("Feed Rate (MT/h)", -0.8)],
    }
    reagent_result = {
        "recovery_gain": 0.42,
        "gaps": {
            "SIPX Dose (g/t)":    {"actual": 40, "optimal": 38, "gap_pct": 5.0,  "label": "Optimal",        "action": "Maintain"},
            "Frother Dose (g/t)": {"actual": 20, "optimal": 17, "gap_pct": 17.6, "label": "Review needed",  "action": "Reduce by 3.0 units"},
            "Lime Dose (kg/t)":   {"actual": 2.5,"optimal": 2.5,"gap_pct": 0.0,  "label": "Optimal",        "action": "Maintain"},
        },
    }
    drift_result = {
        "overall_status": "MONITOR",
        "worst_psi":      0.18,
        "flagged":        ["Flotation pH", "SIPX Dose (g/t)"],
    }

    result = gen.generate(
        shift_data     = shift,
        predicted_rec  = 87.4,
        shap_result    = shap_result,
        anomaly_result = anomaly_result,
        reagent_result = reagent_result,
        drift_result   = drift_result,
        shift_id       = "SHIFT_2024_001",
    )

    print(f"\n{'─'*60}")
    print(result["report"])
    print(f"{'─'*60}")
    print(f"Tokens used: {result['prompt_tokens']} in / {result['output_tokens']} out")
    print("\nEngine 5 OK ✓")
