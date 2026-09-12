# Local modifications for the vendored copy (upstream: b-inary/postflop-solver)

- `src/lib.rs`: added `pub use hand::*;` to expose the `Hand` 7-card
  evaluator, which is public in `hand.rs` but not re-exported upstream.
  Needed by cuda-poker-solver's evaluator cross-validation
  (tools/verify_eval7.cpp).
- Upstream's optional `bincode` (save/load) feature is disabled; it does
  not compile with current toolchains.
