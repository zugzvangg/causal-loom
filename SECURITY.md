# Security Policy

## Supported versions

CausalLoom is an experimental `0.x` project maintained on a
best-effort basis.

| Version | Security fixes |
| --- | --- |
| Latest release and `master` | Best-effort support |
| Older versions | Not supported |

There is no guaranteed response-time or remediation SLA.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue, pull request,
discussion, notebook, or social-media post.

Use GitHub's
[private vulnerability reporting](https://github.com/zugzvangg/causal-loom/security/advisories/new)
and include:

- the affected version or commit;
- a clear description of the impact;
- minimal reproduction steps or a proof of concept;
- affected operating systems or configurations;
- any known mitigations;
- whether the issue has been disclosed elsewhere.

Do not include real credentials, personal data, or proprietary datasets in the
report. Replace them with synthetic examples.

If private vulnerability reporting is unavailable, open a minimal public issue
asking the maintainer to establish a private contact channel. Do not include
security-sensitive details in that issue.

## What to expect

The maintainer will make a best-effort attempt to:

1. acknowledge a complete private report;
2. reproduce and assess its severity;
3. coordinate a fix and disclosure plan with the reporter;
4. credit the reporter unless anonymity is requested;
5. publish a security advisory when users need to take action.

Please allow time for investigation before public disclosure. This policy does
not authorize accessing third-party systems or data, degrading services,
persisting after a vulnerability is demonstrated, or violating applicable law.

## Scope

Security issues include, for example:

- dependency or packaging behavior that enables arbitrary code execution;
- unsafe deserialization or path handling;
- accidental disclosure of secrets or sensitive local data;
- a reproducible denial of service caused by malformed untrusted input.

Statistical disagreements, incorrect estimates, identification failures, and
documentation errors are important but normally are not security
vulnerabilities. Report those through a regular issue with a minimal,
non-sensitive reproducer.

Never use this library as the sole control for safety-critical, medical, legal,
financial, or policy decisions.
