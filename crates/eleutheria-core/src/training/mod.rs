//! Training infrastruktura pro Core Memory (Fáze 5).
//!
//! Core Memory = **trénovaný initial SSM state** — gradient-optimalizovaný
//! výchozí stav, který model načte při startu session místo nulového.
//! Nahrazuje textový persona system prompt kompresovaným "pluginem"
//! v prostoru SSM stavu.
//!
//! Pilotní benchmark v0.4.5 empiricky prokázal, že zachycený state
//! nenese diskrétní fakta (SsmOnly 0 %) — jediná cesta k funkční Core
//! Memory vede přes state tuning.
//!
//! ## Status: alpha bring-up
//!
//! - **v0.5.0-alpha.1 (tento modul):** autograd smoke test — ověření,
//!   že `loss.backward()` vytvoří non-zero gradient pro trainable `Var`
//!   reprezentující initial SSM state jedné vrstvy v Falcon-H1.
//! - **v0.5.0-alpha.2:** multi-layer init states, skutečný training loop
//!   s cross-entropy loss na LM next-token prediction, save/load.
//! - **v0.5.0:** tréning Core Memory na reálném datasetu (Sofie identity,
//!   Bootstrap, Ondra context), validace přes re-run retention benchmarku.

pub mod clip;
pub mod core_memory;
pub mod dataset;
pub mod loss;
pub mod repro;
pub mod smoke;
pub mod trace;
pub mod train;

pub use clip::clip_grad_norm;
pub use core_memory::{CoreMemory, CoreMemoryStack};
pub use dataset::TokenDataset;
pub use loss::cross_entropy_next_token;
pub use smoke::SmokeTrainResult;
pub use train::{TrainingConfig, TrainingResult};
