<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Online detection API

Tracking issue: [#289](https://github.com/NVIDIA-NeMo/Anonymizer/issues/289)

## Goal

Add a reusable asynchronous API for request-time applications that need
source-anchored entity spans for text already held in memory. The API calls the
existing OpenAI-compatible GLiNER endpoint directly and does not construct a
Data Designer workflow or create dataset artifacts.

## Scope

This change spans configuration and interface code:

- public label and threshold configuration plus validated connection parameters;
- ordered `TextRecord` input and record-correlated span results;
- a persistent asynchronous HTTP client with bounded concurrency;
- strict detector-response validation; and
- typed operational and response errors whose messages and chained exceptions
  do not include source text.

NeMo Relay event projection, provider codecs, replacement, LLM validation,
augmentation, and packaging a smaller Python distribution are out of scope.

## API boundary

`OnlineDetector` accepts `TextRecord` values and returns source-anchored spans.
It does not accept provider-shaped dictionaries or mutate input text. Returned
entity spans are always anchored to the submitted source by validated offsets;
detector-echoed text is never authoritative.

One `OnlineDetector` owns one reusable HTTP client unless a caller supplies a
client explicitly. Callers should use it as an async context manager and pass
an `OnlineDetectConfig` with the policy for each call. A semaphore bounds
requests emitted by one session, and each call creates only a bounded number
of worker tasks. The detector service remains responsible for model-side
batching and admission control.

## Alternatives considered

- **Reuse `Anonymizer.run()`:** rejected because it constructs the
  Data Designer-backed dataset pipeline and creates artifacts.
- **Add an online flag to `Anonymizer.run()`:** rejected because it would make
  one method carry two materially different lifecycle and result contracts.
- **Expose Relay-specific payloads:** rejected because provider and
  observability projection belongs to Relay integrations, not Anonymizer.
- **Add LLM validation now:** deferred until its latency and failure semantics
  can be qualified independently.

## Validation

Focused tests will cover configuration, authentication, record
ordering, concurrency limits, source-anchored offsets, overlap resolution,
empty records, cancellation, transport failures, HTTP failures, malformed
responses, unsafe offsets, external-client ownership, and lazy public imports.

The existing full test suite, formatting, lint, type checking, docs build, and
lock check must continue to pass.

## Rollout

The API is additive. Existing file-backed and in-memory full-pipeline behavior
does not change. The first release documents detector-only accuracy tradeoffs
and is not described as equivalent to the full validator-and-augmenter path.
Package splitting can follow after this public boundary is accepted.
