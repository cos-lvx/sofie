//! Identity benchmark — testuje, zda Core Memory drží identity content
//! z trénovaného korpusu (sofie identity pack).
//!
//! Liší se od retention benchmarku ([`super`] root-level moduly):
//! retention testuje krátkodobou paměť (inject fact, vzdálenost, ptej se);
//! identity testuje **co je trénované** — žádný injected fact, otázka
//! jde přímo na identity-load-bearing fakta z trénovaného korpusu.
//!
//! Tři varianty:
//! - [`IdentityVariant::Cold`] — žádná persona, žádná Core Memory
//! - [`IdentityVariant::Core`] — Core Memory, žádná persona
//! - [`IdentityVariant::Full`] — persona + Core Memory (default runtime)
//!
//! Klíčový diff: `(Core - Cold)` izoluje samostatný efekt trénovaného
//! SSM stavu; `(Full - Core)` měří doplňkový efekt persona prefilu.

pub mod harness;
pub mod probe;
pub mod report;
pub mod variant;

pub use harness::{DEFAULT_ANSWER_TOKENS, IdentityBench};
pub use probe::{
    IdentityOutcome, IdentityProbe, IdentityResult, built_in_identity_probes,
    built_in_identity_probes_en, built_in_reasoning_probes, built_in_reasoning_probes_en,
};
pub use report::{IdentityMeta, IdentityReport};
pub use variant::IdentityVariant;
