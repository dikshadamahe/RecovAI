"""
RecovAI — Engine 5: NLP Shift Report Generator
===============================================
Calls the Groq API (Llama 3) to generate a concise, plain-English
shift performance report from structured engine outputs.
Falls back to a rule-based expert system if the API is unavailable.

Usage:
    from engine5_nlp import ShiftReportGenerator
    gen = ShiftReportGenerator(api_key="gsk-...")   # or set GROQ_API_KEY env var

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
from typing import Dict, Optional
import json
from datetime import datetime

# ── Single SYSTEM_PROMPT (no duplicate) ─────────────────────────────────────
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


def chat(message: str, history: list, api_key: str = "") -> str:
    """
    Simple chat interface used by main.py /report endpoint (chat mode).
    Tries Groq → rule-based fallback.
    """
    gen = ShiftReportGenerator(api_key=api_key or os.environ.get("GROQ_API_KEY", ""))
    if gen.client:
        try:
            msgs = history[-6:] + [{"role": "user", "content": message}]
            resp = gen.client.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + msgs,
                model=gen.model,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            pass
    return (
        f"RecovAI (offline mode): I received your question — '{message}'. "
        "To enable live AI answers, set the GROQ_API_KEY environment variable "
        "and restart the backend. I can still provide shift reports and "
        "optimization results using the local expert engine."
    )


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
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.api_key   = key
        self.model     = model
        self.max_tokens = max_tokens
        self.client    = None

        if key:
            try:
                import groq
                self.client = groq.Groq(api_key=key)
            except ImportError:
                print("[Engine 5] groq package not installed. pip install groq")
            except Exception as e:
                print(f"[Engine 5] Failed to initialize Groq client: {e}")

    # ── Main interface ────────────────────────────────────────────────────

    def generate(
        self,
        shift_data:     Dict,
        predicted_rec:  float,
        shap_result:    Dict,
        anomaly_result: Dict,
        reagent_result: Dict,
        drift_result:   Dict,
        shift_id:       Optional[str] = None,
    ) -> Dict:
        """
        Generate a complete shift report.
        Uses Groq Llama 3 if API key available, otherwise falls back to
        an expert rule-based local generator.
        """
        prompt = self._build_prompt(
            shift_data, predicted_rec, shap_result,
            anomaly_result, reagent_result, drift_result, shift_id
        )

        report_text  = ""
        used_llm     = False
        prompt_tokens  = 0
        output_tokens  = 0

        if self.client:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    model=self.model,
                    max_tokens=self.max_tokens,
                )
                report_text   = chat_completion.choices[0].message.content.strip()
                used_llm      = True
                usage         = getattr(chat_completion, "usage", None)
                prompt_tokens  = getattr(usage, "prompt_tokens", 0)
                output_tokens  = getattr(usage, "completion_tokens", 0)
            except Exception as e:
                print(f"[Engine 5] Groq API call failed: {e}. Using rule-based fallback.")

        # Fallback if no client or call failed
        if not report_text:
            report_text = self._generate_fallback(
                shift_data, predicted_rec, shap_result,
                anomaly_result, reagent_result, drift_result
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
        shift_data:    Dict,
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
                resp = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=120,
                )
                return resp.choices[0].message.content.strip()
            except Exception:
                pass

        # Rule-based brief fallback
        s1 = (
            f"Shift copper recovery is projected at {predicted_rec:.1f}% "
            f"with anomaly status flagged as {anomaly_label}."
        )
        if top_drivers:
            s2 = (
                f"Primary recovery driver is {top_drivers[0][0]} "
                f"({top_drivers[0][1]:+.2f} pp). "
                "Recommend close monitoring of key setpoints."
            )
        else:
            s2 = "Process parameters are within expected ranges. Maintain current operations."
        return f"{s1} {s2}"

    def _generate_fallback(
        self,
        shift_data:     Dict,
        predicted_rec:  float,
        shap_result:    Dict,
        anomaly_result: Dict,
        reagent_result: Dict,
        drift_result:   Dict,
    ) -> str:
        """
        Rule-based metallurgical expert system report — used when LLM is unavailable.
        """
        top3       = shap_result.get("top_3_drivers", [])
        top_driver = f"{top3[0][0]} ({top3[0][1]:+.2f} pp)" if top3 else "process conditions"
        base_val   = shap_result.get("base_value", 84.0)

        ano_label   = anomaly_result.get("label", "NORMAL")
        ano_contrib = [f[0] for f in anomaly_result.get("top_contributors", [])]

        gaps            = reagent_result.get("gaps", {})
        gain            = reagent_result.get("recovery_gain", 0.0)
        reagent_actions = [g["action"] for g in gaps.values()
                           if g.get("label") not in ("Optimal", "")]

        drift_status = drift_result.get("overall_status", "OK")
        worst_psi    = drift_result.get("worst_psi", 0.0)

        # Sentence 1 — summary
        s1 = (
            f"The flotation circuit achieved a projected copper recovery of "
            f"{predicted_rec:.2f}%, driven against the historical baseline "
            f"({float(base_val):.2f}%) primarily by {top_driver}."
        )

        # Sentence 2 — anomaly
        if ano_label == "ANOMALY":
            s2 = (
                f"Severe process anomalies were detected (Isolation Forest Alert), "
                f"with significant deviations in {', '.join(ano_contrib[:2]) or 'key parameters'}."
            )
        elif ano_label == "SUSPICIOUS":
            s2 = (
                f"Minor operational variance was flagged as suspicious, with deviations "
                f"in {', '.join(ano_contrib[:2]) or 'key parameters'}."
            )
        else:
            s2 = (
                "Process operations remained stable and within standard "
                "operating envelopes; no anomalies detected."
            )

        # Sentence 3 — reagent
        if reagent_actions:
            s3 = (
                f"Response-surface optimization indicates a potential recovery "
                f"improvement of +{gain:.2f} pp if reagents are adjusted: "
                f"{'; '.join(reagent_actions[:2])}."
            )
        else:
            s3 = (
                "Reagent addition schemes are well-optimized, "
                "matching response-surface maxima with negligible deviation."
            )

        # Sentence 4 — drift
        if drift_status in ("RED", "RETRAIN"):
            s4 = (
                f"Critical PSI drift detected (worst PSI = {worst_psi:.3f}); "
                "current feed properties have decoupled from the training dataset."
            )
        elif drift_status in ("AMBER", "MONITOR"):
            s4 = (
                f"Slight feed property drift observed (worst PSI = {worst_psi:.3f}); "
                "monitor closely for further changes."
            )
        else:
            s4 = (
                "The ore feed profile is statistically stable "
                "and aligned with the training baseline."
            )

        # Recommended action
        if reagent_actions:
            action = f"Adjust reagent setpoints immediately: {reagent_actions[0]}."
        elif ano_label != "NORMAL" and ano_contrib:
            action = (
                f"Investigate sensor deviation in {ano_contrib[0]} "
                "to stabilise circuit performance."
            )
        elif drift_status in ("RED", "RETRAIN", "AMBER", "MONITOR"):
            action = (
                "Collect composite shift samples for laboratory grade verification "
                "and consider model retraining."
            )
        else:
            action = (
                "Maintain current stable flotation cell settings "
                "and continue baseline monitoring."
            )

        return f"{s1} {s2} {s3} {s4} Recommended Action: {action}"

    # ── Prompt construction ───────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        shift_data, predicted_rec, shap_result,
        anomaly_result, reagent_result, drift_result, shift_id
    ) -> str:
        key_inputs = {
            "Head Grade (%Cu)":       shift_data.get("Head Grade (%Cu)") or shift_data.get("head_grade"),
            "Feed Rate (MT/h)":       shift_data.get("Feed Rate (MT/h)") or shift_data.get("feed_rate"),
            "Flotation pH":           shift_data.get("Flotation pH") or shift_data.get("ph"),
            "Pulp Density (%)":       shift_data.get("Pulp Density (%)") or shift_data.get("pulp_density"),
            "Air Flow Rate (m3/min)": shift_data.get("Air Flow Rate (m3/min)") or shift_data.get("air_flow"),
        }
        inputs_str = "\n".join(
            f"  - {k}: {v}" for k, v in key_inputs.items() if v is not None
        )

        top3       = shap_result.get("top_3_drivers", [])
        base_val   = shap_result.get("base_value", "N/A")
        drivers_str = "\n".join(
            f"  {'▲' if v > 0 else '▼'} {f}: {v:+.3f} pp impact on recovery"
            for f, v in top3
        )

        ano_score = anomaly_result.get("score", "N/A")
        ano_label = anomaly_result.get("label", "UNKNOWN")
        ano_top   = anomaly_result.get("top_contributors", [])
        ano_str   = "\n".join(f"  - {f}: z = {z:.2f}" for f, z in ano_top)

        gaps          = reagent_result.get("gaps", {})
        gain          = reagent_result.get("recovery_gain", 0)
        reagent_str   = "\n".join(
            f"  - {r}: actual={g['actual']} g/t, optimal={g['optimal']} g/t, "
            f"gap={g['gap_pct']:.1f}% [{g['label']}] → {g['action']}"
            for r, g in gaps.items()
        )

        drift_status = drift_result.get("overall_status", "UNKNOWN")
        flagged      = drift_result.get("flagged", [])
        worst_psi    = drift_result.get("worst_psi", 0)
        drift_detail = (
            f"Flagged features: {', '.join(flagged)}" if flagged
            else "No features flagged."
        )

        return f"""Please write a concise shift performance report for shift {shift_id or 'current'}.

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


# ── Standalone test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    print("Engine 5 — NLP Shift Report Generator\n" + "=" * 42)

    gen = ShiftReportGenerator(api_key=os.environ.get("GROQ_API_KEY", ""))

    shap_result = {
        "base_value": 84.0,
        "prediction": 87.4,
        "top_3_drivers": [
            ("Head Grade (%Cu)", 2.3),
            ("Flotation pH",     1.1),
            ("SIPX Dose (g/t)",  0.8),
        ],
    }
    anomaly_result = {
        "score":            -0.04,
        "label":            "NORMAL",
        "top_contributors": [("Flotation pH", 0.3), ("Feed Rate (MT/h)", -0.8)],
    }
    reagent_result = {
        "recovery_gain": 0.42,
        "gaps": {
            "SIPX Dose (g/t)":    {"actual": 40, "optimal": 38, "gap_pct": 5.0,  "label": "Optimal",       "action": "Maintain"},
            "Frother Dose (g/t)": {"actual": 20, "optimal": 17, "gap_pct": 17.6, "label": "Review needed", "action": "Reduce by 3.0 units"},
        },
    }
    drift_result = {
        "overall_status": "MONITOR",
        "worst_psi":      0.18,
        "flagged":        ["Flotation pH", "SIPX Dose (g/t)"],
    }

    result = gen.generate(
        shift_data     = {"head_grade": 1.2, "feed_rate": 120.0, "ph": 10.5},
        predicted_rec  = 87.4,
        shap_result    = shap_result,
        anomaly_result = anomaly_result,
        reagent_result = reagent_result,
        drift_result   = drift_result,
        shift_id       = "SHIFT_TEST_001",
    )

    print(f"\n{'─' * 60}")
    print(result["report"])
    print(f"{'─' * 60}")
    print(f"Used LLM: {result['used_llm']} | Model: {result['model']}")
    print("\nEngine 5 OK ✓")
