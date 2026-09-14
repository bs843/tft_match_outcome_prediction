# Evidence-classification rules

These labels organize interpretation; they are not additional hypothesis tests.

| Category | Deterministic rule |
|---|---|
| Convergent evidence | Raw and adjusted effects have the same sign, and mapped mean absolute TreeSHAP is in the upper quartile of reliably mapped features |
| Descriptive-only | Absolute raw uplift is at least 5 percentage points, but adjusted or meaningful mapped predictive evidence is absent |
| Predictive-only | Mapped TreeSHAP is in the upper quartile, while raw uplift is absent or below 5 percentage points |
| Attenuated after adjustment | Raw and adjusted odds ratios are valid, directions agree, and `1 - abs(log(adjusted OR)) / abs(log(raw OR)) ≥ 0.50` |
| Directionally unstable | Raw uplift and adjusted log-odds coefficient have opposite signs |
| Adjusted and descriptive | Raw and adjusted effects agree, but mapped TreeSHAP is absent or below the upper-quartile threshold |
| Insufficient evidence | Support is below the prespecified primary threshold, or the available evidence does not meet another rule |

The attenuation quantity is **association attenuation after covariate adjustment**.
It must not be described as the proportion of confounding explained. Null fields are
retained whenever two feature representations cannot be mapped without speculation.

