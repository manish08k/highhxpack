# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please **do not open a public issue**. Report vulnerabilities privately through
GitHub's "Report a vulnerability" (Security → Advisories) on the repository. Include a
description, affected versions, and steps to reproduce.

We aim to acknowledge reports within 3 working days and to publish a fix or
mitigation as soon as practical, crediting reporters who wish to be named.

## Scope and design

HighHXPack stores potentially sensitive personal data locally. Relevant protections
are described in [docs/concepts/privacy.md](docs/concepts/privacy.md):
owner-only file permissions, no secrets in configuration files or logs, strict input
and import validation, bound SQL parameters, JSON-only import/export, and no network
access unless a remote provider is configured.

Out of scope: attacks that require control of the user's account or filesystem, and
the behavior of third-party providers you choose to configure.
