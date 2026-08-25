# Security Policy

## Reporting a Vulnerability

ShortChain takes security seriously. If you find a security vulnerability,
please report it privately before opening a public issue.

**Do not** create a public GitHub issue for security problems.

**Instead**, report the issue by email to the maintainers at a private address,
or open a [GitHub security advisory](https://github.com/awab-ml/ShortChain/security/advisories/new).

Please include:

- A description of the vulnerability and its impact.
- The affected version(s).
- Steps to reproduce, or a minimal proof of concept.
- Any suggestions for a fix, if you have them.

You will receive a response within a few days. We will coordinate a fix, a
release, and — where appropriate — public disclosure with you.

## Scope

Things we hold to security standards:

- Trace / telemetry data handling. The receiver writes prompts and tool
  arguments to disk; it writes `data/runtime/` with restrictive permissions
  (`0600`) and should only be run on infrastructure you trust.
- Credential handling: API keys are read from the environment or passed
  explicitly, never logged or persisted.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Best Practices

- Set `SHORTCHAIN_API_KEY` via the environment, not in committed files.
- Treat collected traces (which may contain prompts and tool outputs) as
  sensitive material and store them on infrastructure you control.