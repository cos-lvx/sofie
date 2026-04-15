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
    /// Maximální délka kontextu modelu (tokenů).
    context_limit: usize,
    /// Počet KV hlav × head_dim × num_layers × 2 (K+V) × bytes_per_element.
    /// Pro odhad KV cache VRAM z pozice.
    kv_bytes_per_token: usize,
}

impl SofieSession {
    /// Vytvoří novou session s prázdným stavem.
    pub(crate) fn new(
        state: ModelState,
        config: &crate::falcon_h1::config::FalconH1Config,
    ) -> Self {
        Self {
            state,
            position: 0,
            history: Vec::new(),
            initialized: false,
            turn_count: 0,
            started_at: Utc::now(),
            context_limit: config.max_position_embeddings,
            kv_bytes_per_token: Self::compute_kv_bytes_per_token(config),
        }
    }

    /// Vytvoří session z načteného checkpointu.
    pub(crate) fn from_checkpoint(
        state: ModelState,
        position: usize,
        config: &crate::falcon_h1::config::FalconH1Config,
    ) -> Self {
        Self {
            state,
            position,
            history: Vec::new(),
            initialized: true,
            turn_count: 0,
            started_at: Utc::now(),
            context_limit: config.max_position_embeddings,
            kv_bytes_per_token: Self::compute_kv_bytes_per_token(config),
        }
    }

    /// Spočítej KV cache bytů na token z konfigurace.
    /// Vzorec: num_layers × n_kv_heads × head_dim × 2 (K+V) × 2 (BF16 bytes)
    fn compute_kv_bytes_per_token(config: &crate::falcon_h1::config::FalconH1Config) -> usize {
        config.num_hidden_layers * config.num_key_value_heads * config.head_dim * 2 * 2
    }

    /// Nahradí stav novým a resetuje pozici. Pokud `mark_uninitialized=true`,
    /// příští `send_message`/`inject_turn` projde plnou pipeline jako turn 1
    /// (užitečné po vyčištění KV cache — RoPE indexy musí startovat od 0).
    ///
    /// `turn_count` a `history` zůstávají zachované pro audit a reporting —
    /// změna stavu není výmaz konverzace, jen restrukturalizace paměti.
    pub(crate) fn replace_state(
        &mut self,
        state: ModelState,
        position: usize,
        mark_uninitialized: bool,
    ) {
        self.state = state;
        self.position = position;
        if mark_uninitialized {
            self.initialized = false;
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

    // -------------------------------------------------------------------
    // Context budget monitoring
    // -------------------------------------------------------------------

    /// Maximální délka kontextu modelu (tokenů).
    pub fn context_limit(&self) -> usize {
        self.context_limit
    }

    /// Využití kontextu jako poměr (0.0–1.0).
    pub fn context_usage(&self) -> f64 {
        if self.context_limit == 0 {
            return 1.0;
        }
        self.position as f64 / self.context_limit as f64
    }

    /// Zbývající tokeny do limitu.
    pub fn remaining_tokens(&self) -> usize {
        self.context_limit.saturating_sub(self.position)
    }

    /// Odhad velikosti KV cache v bytech.
    pub fn kv_cache_bytes(&self) -> usize {
        self.position * self.kv_bytes_per_token
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
            max_position_embeddings: 1000, // malý limit pro testy
        }
    }

    #[test]
    fn test_new_session() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let session = SofieSession::new(state, &config);

        assert_eq!(session.position(), 0);
        assert_eq!(session.turn_count(), 0);
        assert!(session.history().is_empty());
        assert!(!session.initialized);
        assert_eq!(session.context_limit(), 1000);
    }

    #[test]
    fn test_record_turn() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let mut session = SofieSession::new(state, &config);

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
        let session = SofieSession::from_checkpoint(state, 500, &config);

        assert_eq!(session.position(), 500);
        assert!(session.initialized);
        assert_eq!(session.turn_count(), 0);
        assert_eq!(session.remaining_tokens(), 500);
    }

    #[test]
    fn test_context_usage() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let mut session = SofieSession::new(state, &config);

        assert_eq!(session.context_usage(), 0.0);
        assert_eq!(session.remaining_tokens(), 1000);

        session.record_turn("test", "response", 250);
        assert!((session.context_usage() - 0.25).abs() < 0.001);
        assert_eq!(session.remaining_tokens(), 750);

        session.record_turn("test2", "response2", 1000);
        assert!((session.context_usage() - 1.0).abs() < 0.001);
        assert_eq!(session.remaining_tokens(), 0);
    }

    #[test]
    fn test_kv_cache_bytes() {
        let config = dummy_config();
        let state = ModelState::new(&config, DType::F32, &Device::Cpu).unwrap();
        let mut session = SofieSession::new(state, &config);

        // 0 tokenů = 0 bytů
        assert_eq!(session.kv_cache_bytes(), 0);

        // config: 2 layerů × 1 kv_head × 16 head_dim × 2 (K+V) × 2 (BF16) = 128 B/token
        session.record_turn("test", "resp", 100);
        assert_eq!(session.kv_cache_bytes(), 100 * 128);
    }
}
