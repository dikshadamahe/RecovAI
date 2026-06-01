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


import groq
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
    Generates natural-language shift reports using the Groq API,
    with a robust rule-based metallurgical expert system fallback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 400,
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_key = key
        self.model = model
        self.max_tokens = max_tokens

        self.client = None
        if key:
            try:
                self.client = groq.Groq(api_key=key)
            except Exception as e:
                print(f"[Engine 5] Failed to initialize Groq client: {e}")

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
        Generate a complete shift report. Uses Groq Llama 3 if API key available,
        otherwise falls back to an expert rule-based local generator.
        """
        prompt = self._build_prompt(
            shift_data, predicted_rec, shap_result,
            anomaly_result, reagent_result, drift_result, shift_id
        )

        report_text = ""
        used_llm = False
        prompt_tokens = 0
        output_tokens = 0

        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model,
                    max_tokens=self.max_tokens,
                )
                report_text = chat_completion.choices[0].message.content.strip()
                used_llm = True
                prompt_tokens = chat_completion.usage.prompt_tokens if hasattr(chat_completion, "usage") else 0
                output_tokens = chat_completion.usage.completion_tokens if hasattr(chat_completion, "usage") else 0
            except Exception as e:
                print(f"[Engine 5] Groq API call failed: {e}. Falling back to rule-based generation.")

        # Fallback if no client or call failed
        if not report_text:
            report_text = self._generate_fallback(
                shift_data, predicted_rec, shap_result, anomaly_result, reagent_result, drift_result
            )

        return {
            "report":        report_text,
            "shift_id":      shift_id or datetime.utcnow().strftime("%Y%m%d_%H%M"),
            "timestamp":     datetime.utcnow().isoformat(),
            "model":         self.model if used_llm else "Local Expert Fallback Engine",
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "used_llm":      used_llm,
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

        if self.client:
            try:
                prompt = (
                    f"Write a 2-sentence shift alert for plant operators.\n"
                    f"Recovery: {predicted_rec:.1f}%. "
                    f"Top drivers: {drivers_str}. "
                    f"Anomaly status: {anomaly_label}.\n"
                    f"Be direct. Mention one action."
                )
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=120,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                pass

        # Rule-based brief fallback
        s1 = f"Shift copper recovery is projected at {predicted_rec:.1f}% with anomaly status flagged as {anomaly_label}."
        if top_drivers:
            s2 = f"Primary recovery driver is {top_drivers[0][0]} ({top_drivers[0][1]:+.2f} pp). Recommend close monitoring of key setpoints."
        else:
            s2 = "Process parameters are within expected ranges. Maintain current operations."
        return f"{s1} {s2}"

    def _generate_fallback(
        self,
        shift_data: Dict[str, float],
        predicted_rec: float,
        shap_result: Dict,
        anomaly_result: Dict,
        reagent_result: Dict,
        drift_result: Dict,
    ) -> str:
        """
        Rule-based metallurgical expert system that mimics a real metallurgical report.
        """
        top3 = shap_result.get("top_3_drivers", [])
        top_driver_str = f"{top3[0][0]} ({top3[0][1]:+.2f} pp)" if top3 else "process conditions"

        anomaly_label = anomaly_result.get("label", "NORMAL")
        anomaly_contributors = [f[0] for f in anomaly_result.get("top_contributors", [])]

        gaps = reagent_result.get("gaps", {})
        gain = reagent_result.get("recovery_gain", 0.0)
        reagent_actions = [g["action"] for g in gaps.values() if g.get("label") != "Optimal"]

        drift_status = drift_result.get("overall_status", "OK")
        worst_psi = drift_result.get("worst_psi", 0.0)

        # 1. Opening sentence
        s1 = f"The flotation circuit achieved a projected copper recovery of {predicted_rec:.2f}%, driven against the historical baseline ({shap_result.get('base_value', 84.0):.2f}%) primarily by {top_driver_str}."

        # 2. Anomaly assessment
        if anomaly_label == "ANOMALY":
            s2 = f"Severe process anomalies were detected during the shift (Isolation Forest Alert), showing significant deviations in {', '.join(anomaly_contributors[:2])}."
        elif anomaly_label == "SUSPICIOUS":
            s2 = f"Minor operational variance was flagged as suspicious, with deviations detected in {', '.join(anomaly_contributors[:2])}."
        else:
            s2 = "Process operations remained highly stable and well within standard operating envelopes, with no anomalies flagged."

        # 3. Reagent analysis
        if reagent_actions:
            s3 = f"Response-surface optimization indicates a potential recovery improvement of +{gain:.2f}% if reagent additions are adjusted. Specifically, operators should {'; '.join(reagent_actions[:2])}."
        else:
            s3 = "Reagent addition schemes were highly optimized, matching target setpoints and showing negligible deviation from response-surface maxima."

        # 4. Drift status
        if drift_status in ("RED", "RETRAIN"):
            s4 = f"Critical population stability index (PSI) drift has been detected (worst PSI = {worst_psi:.3f}), showing that current feed properties are statistically decoupled from the training dataset."
        elif drift_status in ("AMBER", "MONITOR"):
            s4 = f"Slight input feed property drift is observed (worst PSI = {worst_psi:.3f}), suggesting feed composition changes are starting to occur."
        else:
            s4 = "The input ore profile shows high statistical stability and is perfectly aligned with the training dataset baseline."

        # 5. Recommended Action
        if reagent_actions:
            action = f"Optimize reagent setpoints immediately: {reagent_actions[0]}."
        elif anomaly_label != "NORMAL":
            action = f"Investigate root cause of sensor deviation in {anomaly_contributors[0]} to stabilize circuit performance."
        elif drift_status in ("RED", "RETRAIN", "AMBER", "MONITOR"):
            action = "Collect composite shift samples for laboratory grade verification and consider model retraining."
        else:
            action = "Maintain current stable flotation cell settings and continue baseline monitoring."

        return f"{s1} {s2} {s3} {s4} Recommended Action: {action}"

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
