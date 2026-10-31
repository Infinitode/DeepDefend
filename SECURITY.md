# Security Policy

DeepDefend is a dual-use library: it implements both adversarial attacks and
defenses. Please read the "Responsible use" section below before filing a
report.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| < 0.2   | ❌ (several functions were non-functional or silently wrong; see `CHANGELOG.md`) |

## Reporting a vulnerability

**Please do not open a public issue for security problems.**

Report privately through GitHub's
[private vulnerability reporting](https://github.com/Infinitode/DeepDefend/security/advisories/new)
or by email to **infinitode.ltd@gmail.com**.

Please include:

- the affected version and Python / TensorFlow versions,
- a minimal reproduction (or the exact function and arguments),
- the impact you believe it has.

We aim to acknowledge reports within 5 working days and to publish a fixed
release plus an advisory once a fix is available. Please give us a reasonable
window to ship a fix before any public disclosure.

## Scope

In scope:

- code execution, denial of service, or data exfiltration caused by this library;
- functions that silently return incorrect results in a way that would lead a
  user to overstate the robustness of their model (this is treated as a security
  issue for a robustness library);
- unsafe deserialisation, path handling or network access in the library or in
  its build/publish pipeline.

Out of scope:

- the *effectiveness* of a defense against an adaptive attacker. Preprocessing
  defenses are not guarantees; see the limitations in `readme.md`;
- issues in TensorFlow, Keras or NumPy themselves (please report those upstream);
- the mere existence of attack implementations in this library.

## Responsible use

The attack implementations exist so that defenders can measure and improve the
robustness of models they are authorised to test. Do not use this library
against systems you do not own or have explicit permission to assess.
