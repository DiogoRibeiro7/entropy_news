# Corpus contract

The paper estimator and the empirical Reuters replication are separate claims.

`methodological_reproduction` means the code reproduces the estimator on user-supplied monthly news text. It does not assert that the supplied corpus matches the Reuters sample used by Glasserman, Mamaysky, and Qin.

`empirical_reuters_replication` is a stronger declaration. Its manifest must state Reuters as the source, cover the requested months, declare first-rewrite selection, English-language filtering, the S&P 500 company universe, application of the paper's headline exclusions, and the same minimum article-length rule used by the paper protocol.

The validator checks internal consistency only. It cannot prove that proprietary source files really came from Reuters or that upstream filters were honestly applied. A manuscript-facing empirical replication should therefore preserve independent source receipts, extraction logs, or equivalent evidence alongside the corpus manifest and run provenance outputs.

Use `docs/corpus_contract.example.json` as the starting point for a Reuters-compatible declaration. Runs without an external corpus manifest should be labelled methodological reproductions, not empirical replications of the Reuters results.
