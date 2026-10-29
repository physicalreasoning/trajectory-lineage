# Experiment 2 dry run: operating characteristics of the frozen analyzer

**Paired SD 1.88 points, 300 synthetic datasets per truth.** Analytic power at Δ_A = 5: 0.927.

| Truth (Δ_O, Δ_A, Δ_B, Δ_V in points) | A supported | A equivalent | A inconclusive | O | B | V | O−A | A−B | Interpretation rows (freq.) |
|---|---|---|---|---|---|---|---|---|---|
| global_null: (0, 0, 0, 0) | 0.00 | 1.00 | 0.00 | equivalent 0.91 | equivalent 0.93 | equivalent 0.93 | inconclusive 0.55 | equivalent 0.50 | insufficient_evidence 0.08 |
| A_at_margin_3: (0, 3, 0, 0) | 0.04 | 0.05 | 0.91 | equivalent 0.89 | equivalent 0.90 | equivalent 0.89 | inconclusive 0.99 | inconclusive 0.99 | survives_matching 0.04; length_balancing_attenuates 0.00; insufficient_evidence 0.93 |
| A_at_minus_margin_-3: (0, -3, 0, 0) | 0.00 | 0.03 | 0.97 | equivalent 0.90 | equivalent 0.92 | equivalent 0.92 | inconclusive 0.99 | inconclusive 1.00 | insufficient_evidence 0.97 |
| A_5_planning: (10, 5, 5, 5) | 0.89 | 0.00 | 0.11 | supported 1.00 | supported 0.64 | supported 0.65 | inconclusive 0.64 | inconclusive 0.59 | survives_matching 0.89; persists_after_balancing 0.64; insufficient_evidence 0.43; V_failures_lower 0.65 |
| historical_10: (10, 10, 10, 10) | 1.00 | 0.00 | 0.00 | supported 1.00 | supported 1.00 | supported 1.00 | inconclusive 0.51 | equivalent 0.52 | survives_matching 1.00; persists_after_balancing 1.00; V_failures_lower 1.00 |
| transition_confound: (10, 0, 0, 0) | 0.00 | 1.00 | 0.00 | supported 1.00 | equivalent 0.94 | equivalent 0.89 | supported 1.00 | inconclusive 0.50 | transition_budget_sensitive 1.00; insufficient_evidence 0.06 |
| length_shortcut: (10, 10, 0, 10) | 1.00 | 0.00 | 0.00 | supported 1.00 | equivalent 0.91 | supported 1.00 | inconclusive 0.53 | supported 1.00 | survives_matching 1.00; length_balancing_attenuates 0.91; insufficient_evidence 0.09; V_failures_lower 1.00 |
| persists_after_balancing: (8, 8, 8, 8) | 1.00 | 0.00 | 0.00 | supported 1.00 | supported 1.00 | supported 1.00 | inconclusive 0.51 | inconclusive 0.52 | survives_matching 1.00; persists_after_balancing 1.00; V_failures_lower 1.00 |
| reverse_-5: (0, -5, 0, 0) | 0.00 | 0.00 | 1.00 | equivalent 0.91 | equivalent 0.92 | equivalent 0.93 | inconclusive 0.68 | inconclusive 1.00 | insufficient_evidence 1.00 |
| missing_run_historical_10: (10, 10, 10, 10); missing A_phi0.65_s104 | 0.00 | 0.00 | 1.00 | supported 1.00 | supported 1.00 | supported 1.00 | inconclusive 1.00 | inconclusive 1.00 | persists_after_balancing 1.00; insufficient_evidence 1.00; V_failures_lower 1.00 |
| A_5_stress_sd5: (10, 5, 5, 5); SD 5.0 | 0.30 | 0.00 | 0.70 | supported 0.83 | inconclusive 0.92 | inconclusive 0.92 | inconclusive 0.97 | inconclusive 1.00 | survives_matching 0.30; persists_after_balancing 0.08; insufficient_evidence 0.97; V_failures_lower 0.08 |

| Validation | Value | Bound | Pass |
|---|---|---|---|
| false support of A at the margin (Δ_A = 3) ≤ 5% (+MC tolerance) | 0.037 | 0.081 | yes |
| false equivalence of A at the lower margin (Δ_A = −3) ≤ 5% (+MC tolerance) | 0.033 | 0.081 | yes |
| power for A at Δ_A = 5, SD 1.88 matches the analytic value 0.927 (±MC tolerance) | 0.890 | 0.927 | yes |
| missing run makes A, O−A and A−B inconclusive in every dataset | 1.000 | 1.000 | yes |
| global null: familywise false 'supported' among secondaries ≤ 5% (+MC tolerance) | 0.000 | 0.081 | yes |
| historical 10-point effect: all four drops supported with probability ≥ 0.99 | 1.000 | 0.990 | yes |

**Paired SD 3.0 points, 300 synthetic datasets per truth.** Analytic power at Δ_A = 5: 0.618.

| Truth (Δ_O, Δ_A, Δ_B, Δ_V in points) | A supported | A equivalent | A inconclusive | O | B | V | O−A | A−B | Interpretation rows (freq.) |
|---|---|---|---|---|---|---|---|---|---|
| global_null: (0, 0, 0, 0) | 0.00 | 0.79 | 0.21 | inconclusive 0.79 | inconclusive 0.76 | inconclusive 0.78 | inconclusive 0.97 | inconclusive 0.98 | insufficient_evidence 0.81 |
| A_at_margin_3: (0, 3, 0, 0) | 0.03 | 0.04 | 0.92 | inconclusive 0.76 | inconclusive 0.82 | inconclusive 0.77 | inconclusive 1.00 | inconclusive 0.99 | survives_matching 0.03; insufficient_evidence 0.99 |
| A_at_minus_margin_-3: (0, -3, 0, 0) | 0.00 | 0.04 | 0.96 | inconclusive 0.73 | inconclusive 0.76 | inconclusive 0.74 | inconclusive 0.99 | inconclusive 1.00 | insufficient_evidence 1.00 |
| A_5_planning: (10, 5, 5, 5) | 0.61 | 0.00 | 0.39 | supported 1.00 | inconclusive 0.76 | inconclusive 0.75 | inconclusive 0.90 | inconclusive 0.97 | survives_matching 0.61; persists_after_balancing 0.24; insufficient_evidence 0.86; V_failures_lower 0.25 |
| historical_10: (10, 10, 10, 10) | 1.00 | 0.00 | 0.00 | supported 1.00 | supported 1.00 | supported 1.00 | inconclusive 0.97 | inconclusive 0.95 | survives_matching 1.00; persists_after_balancing 1.00; V_failures_lower 1.00 |
| transition_confound: (10, 0, 0, 0) | 0.00 | 0.78 | 0.22 | supported 1.00 | inconclusive 0.73 | inconclusive 0.71 | supported 0.96 | inconclusive 0.97 | transition_budget_sensitive 0.76; insufficient_evidence 0.79 |
| length_shortcut: (10, 10, 0, 10) | 1.00 | 0.00 | 0.00 | supported 1.00 | inconclusive 0.71 | supported 1.00 | inconclusive 0.98 | supported 0.97 | survives_matching 1.00; length_balancing_attenuates 0.29; insufficient_evidence 0.71; V_failures_lower 1.00 |
| persists_after_balancing: (8, 8, 8, 8) | 1.00 | 0.00 | 0.00 | supported 0.97 | supported 0.97 | supported 0.96 | inconclusive 0.96 | inconclusive 0.95 | survives_matching 1.00; persists_after_balancing 0.97; insufficient_evidence 0.04; V_failures_lower 0.96 |
| reverse_-5: (0, -5, 0, 0) | 0.00 | 0.00 | 1.00 | inconclusive 0.77 | inconclusive 0.76 | inconclusive 0.79 | inconclusive 0.91 | inconclusive 1.00 | insufficient_evidence 1.00 |
| missing_run_historical_10: (10, 10, 10, 10); missing A_phi0.65_s104 | 0.00 | 0.00 | 1.00 | supported 1.00 | supported 1.00 | supported 1.00 | inconclusive 1.00 | inconclusive 1.00 | persists_after_balancing 1.00; insufficient_evidence 1.00; V_failures_lower 1.00 |
| A_5_stress_sd5: (10, 5, 5, 5); SD 5.0 | 0.30 | 0.00 | 0.70 | supported 0.83 | inconclusive 0.92 | inconclusive 0.92 | inconclusive 0.97 | inconclusive 1.00 | survives_matching 0.30; persists_after_balancing 0.08; insufficient_evidence 0.97; V_failures_lower 0.08 |

| Validation | Value | Bound | Pass |
|---|---|---|---|
| false support of A at the margin (Δ_A = 3) ≤ 5% (+MC tolerance) | 0.033 | 0.081 | yes |
| false equivalence of A at the lower margin (Δ_A = −3) ≤ 5% (+MC tolerance) | 0.043 | 0.081 | yes |
| power for A at Δ_A = 5, SD 3.0 matches the analytic value 0.618 (±MC tolerance) | 0.613 | 0.618 | yes |
| missing run makes A, O−A and A−B inconclusive in every dataset | 1.000 | 1.000 | yes |
| global null: familywise false 'supported' among secondaries ≤ 5% (+MC tolerance) | 0.000 | 0.081 | yes |
| historical 10-point effect: all four drops supported with probability ≥ 0.99 | 1.000 | 0.990 | yes |
