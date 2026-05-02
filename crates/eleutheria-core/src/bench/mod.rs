//! Benchmarky pro měření vlastností modelu a session API.
//!
//! `retention` — behaviorální test SSM state retence: fact recall na různých
//! vzdálenostech, porovnání variant (Full / SsmOnly / Cold). Prerekvizita pro
//! Core Memory design ve Fázi 5.
//!
//! Veřejné API:
//! - [`RetentionProbe`], [`ProbeResult`], [`built_in_probes`] — definice pokusů
//! - [`FillerCorpus`] — deterministický EN korpus pro dosažení cílové pozice
//! - [`BenchVariant`] — tři režimy zachovávání stavu
//! - [`RetentionBench`] — orchestrátor, zpouští pokusy přes `Sofie`
//! - [`BenchReport`] — agregovaný výstup s JSON + markdown serializací

pub mod filler;
pub mod harness;
pub mod identity;
pub mod probe;
pub mod report;
pub mod variant;

pub use filler::{FillerCorpus, FillerPlan};
pub use harness::RetentionBench;
pub use identity::{
    IdentityBench, IdentityMeta, IdentityOutcome, IdentityProbe, IdentityReport, IdentityResult,
    IdentityVariant, built_in_identity_probes,
};
pub use probe::{ProbeOutcome, ProbeResult, RetentionProbe, built_in_probes};
pub use report::BenchReport;
pub use variant::BenchVariant;
