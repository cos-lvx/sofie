//! Živá konverzační session se Sofií.
//!
//! `SofieSession` drží `ModelState` mezi turny — SSM state akumuluje kontext,
//! KV cache roste s každým tokenem. Každý nový turn prefilluje jen delta
//! (ChatML wrapping nové zprávy), ne celou konverzaci.

use chrono::{DateTime, Utc};

use crate::falcon_h1::state::ModelState;
use crate::prompt::types::ChatMessage;

/// Živá konverzační session.
pub struct SofieSession {
    /// Stav modelu (SSM + conv + KV cache) — přežívá mezi turny.
    pub(crate) state: ModelState,
    /// Pozice v sekvenci (kolik tokenů model zpracoval).
    pub(crate) position: usize,
    /// Historie konverzace (user + assistant zprávy).
    pub(crate) history: Vec<ChatMessage>,
    /// Byl první turn zpracován? (full pipeline vs delta)
    pub(crate) initialized: bool,
    /// Počet dokončených turnů (user→assistant výměn).
    turn_count: usize,
    /// Timestamp startu session.
    started_at: DateTime<Utc>,
}

impl SofieSession {
    /// Vytvoří novou session s prázdným stavem.
    pub(crate) fn new(state: ModelState) -> Self {
        Self {
            state,
            position: 0,
            history: Vec::new(),
            initialized: false,
            turn_count: 0,
            started_at: Utc::now(),
        }
    }

    /// Vytvoří session z načteného checkpointu.
    pub(crate) fn from_checkpoint(state: ModelState, position: usize) -> Self {
        Self {
            state,
            position,
            history: Vec::new(),
            initialized: true, // checkpoint = stav po předchozím zpracování
            turn_count: 0,
            started_at: Utc::now(),
        }
    }

    /// Zaznamená dokončený turn.
    pub(crate) fn record_turn(&mut self, user_msg: &str, assistant_msg: &str, new_position: usize) {
        self.history.push(ChatMessage {
            role: crate::prompt::types::ChatRole::User,
            content: user_msg.to_string(),
        });
        self.history.push(ChatMessage {
            role: crate::prompt::types::ChatRole::Assistant,
            content: assistant_msg.to_string(),
        });
        self.position = new_position;
        self.turn_count += 1;
    }

    // -------------------------------------------------------------------
    // Veřejné accessory
    // -------------------------------------------------------------------

    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    pub fn started_at(&self) -> &DateTime<Utc> {
        &self.started_at
    }

    pub fn state(&self) -> &ModelState {
        &self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::types::ChatRole;
    use candle_core::{DType, Device};

    fn dummy_config() -> crate::falcon_h1::config::FalconH1Config {
        crate::falcon_h1::config::FalconH1Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 16,
            mamba_d_state: 8,
            mamba_n_heads: 2,
            mamba_d_head: 16,
            mamba_d_ssm: 32,
            mamba_d_conv: 4,
            mamba_expand: 2,
            mamba_n_groups: 1,
            mamba_chunk_size: 256,
            mamba_conv_bias: true,
            mamba_proj_bias: false,
            mamba_norm_before_gate: false,
            mamba_rms_norm: true,
            mamba_use_mlp: true,
            embedding_multiplier: 5.66,
            lm_head_multiplier: 0.0195,
            ssm_in_multiplier: 0.4167,
            ssm_out_multiplier: 0.1179,
            ssm_multipliers: vec![0.2946],
            attention_in_multiplier: 1.0,
            attention_out_multiplier: 0.1042,
            key_multiplier: 1.0,
            mlp_multipliers: vec![0.2946],
            rms_norm_eps: 1e-5,
            eos_token_id: Some(11),
            rope_theta: 1e11,
            tie_word_embeddings: false,
        }
    }

    #[test]
    fn test_new_session() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let session = SofieSession::new(state);

        assert_eq!(session.position(), 0);
        assert_eq!(session.turn_count(), 0);
        assert!(session.history().is_empty());
        assert!(!session.initialized);
    }

    #[test]
    fn test_record_turn() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let mut session = SofieSession::new(state);

        session.record_turn("Ahoj", "Ahoj, jak se máš?", 42);

        assert_eq!(session.turn_count(), 1);
        assert_eq!(session.position(), 42);
        assert_eq!(session.history().len(), 2);
        assert_eq!(session.history()[0].role, ChatRole::User);
        assert_eq!(session.history()[0].content, "Ahoj");
        assert_eq!(session.history()[1].role, ChatRole::Assistant);
        assert_eq!(session.history()[1].content, "Ahoj, jak se máš?");
    }

    #[test]
    fn test_from_checkpoint() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let session = SofieSession::from_checkpoint(state, 500);

        assert_eq!(session.position(), 500);
        assert!(session.initialized);
        assert_eq!(session.turn_count(), 0);
    }
}
