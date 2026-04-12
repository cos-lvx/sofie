//! Konfigurace Falcon-H1-7B modelu.
//! Načítá se přímo z config.json přes serde.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FalconH1Config {
    // === Základní rozměry modelu ===
    /// Velikost slovníku (kolik tokenů model zná). V JSONu: "vocab_size"
    pub vocab_size: usize,

    /// Šířka modelu - rozměr skrytého stavu. v JSONu: "hidden_size"
    /// U Falcon-H1-7B = 3072. Všechno protéká tímto rozměrem.
    pub hidden_size: usize,

    /// Počet vrstev (layerů). V JSONu: "num_hidden_layers"
    /// U Falcon-H1-7B = 44
    pub num_hidden_layers: usize,

    /// Rozměr MLP mezivrstvy. V JSONu: "intermediate_size"
    /// SwiGLU MLP: hidden_size -> intermediate_size -> hidden_size
    pub intermediate_size: usize,

    // === Attention ===
    /// Počet Q hlav v attention. V JSONu: "num_attention_heads"
    /// U Falcon-H1-7B = 12.
    pub num_attention_heads: usize,

    /// Počet KV hlav (Grouped Query Attention). V JSONu: "num_key_value_heads"
    /// U FAlcon-H1-7B = 2. Každá KV hlava obsluhuje 6 Q hlav (12/2).
    pub num_key_value_heads: usize,

    /// Rozměr jedné attention hlavy. V JSONu: "head_dim"
    /// 128. Q je [12x128]=1536, KV je [2x128]=256.
    pub head_dim: usize,

    // === SSM (Mamba-2) parametry - flat s prefixem mamba_ ===
    /// Rozměr SSM stavu. Každá hlava udržuje matici [headdim x d_state].
    /// U Falcon-H1-7B = 256
    pub mamba_d_state: usize,

    /// Počet SSM hlav.
    /// Falcon-H1-7B = [24x128] = 3072 = hidden_size
    pub mamba_n_heads: usize,

    /// Rozměr jedné SSM hlavy.
    /// Falcon-H1-7B = 128.
    pub mamba_d_head: usize,

    /// Celkový SSm rozměr.
    /// Falcon-H1-7B =  3072.
    pub mamba_d_ssm: usize,

    /// Šířka casual konvoluce.
    /// Falůcon-H1-7B = 4 (state drží d_conv-1 = 3 tokeny).
    pub mamba_d_conv: usize,

    /// Expand factor.
    ///Falcon-H1-7B = 2.
    pub mamba_expand: usize,

    /// Počet skupin.
    /// Falcon-H1-7B = 1-
    pub mamba_n_groups: usize,

    /// Chunk size pro SSD mód (my zatím nepoužíváme).
    /// Falcon-H1-7B = 256.
    pub mamba_chunk_size: usize,

    /// Conv bias ano/ne
    pub mamba_conv_bias: bool,

    /// Projection bias ano/ne
    pub mamba_proj_bias: bool,

    /// Norm before gate - false znamená gate se aplikuje PO normalizaci.
    pub mamba_norm_before_gate: bool,

    /// Používá RMSNorm (ne LayerNorm).
    pub mamba_rms_norm: bool,

    /// Má MLP v každém layeru.
    pub mamba_use_mlp: bool,

    // === mikroP multipliery ===
    pub embedding_multiplier: f64,
    pub lm_head_multiplier: f64,
    pub ssm_in_multiplier: f64,
    pub ssm_out_multiplier: f64,
    pub ssm_multipliers: Vec<f64>,
    pub attention_in_multiplier: f64,
    pub attention_out_multiplier: f64,
    pub key_multiplier: f64,
    pub mlp_multipliers: Vec<f64>,

    // === Ostatní ===
    /// RMSNorm  epsilon.
    pub rms_norm_eps: f64,

    /// EOS token ID z config.json. Falcon-H1 = 11.
    #[serde(default)]
    pub eos_token_id: Option<u32>,

    /// RoPE theta (frekvence pro rotační pozicové embeddingry). 10^11.
    pub rope_theta: f64,

    /// Sdílí embedding a lm_head váhy? false = separátní
    pub tie_word_embeddings: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config() {
        let json =
            std::fs::read_to_string("/home/lvx/Models/falcon-h1-7b-instruct/config.json").unwrap();
        let config: FalconH1Config = serde_json::from_str(&json).unwrap();

        assert_eq!(config.hidden_size, 3072);
        assert_eq!(config.num_hidden_layers, 44);
        assert_eq!(config.mamba_d_state, 256);
        println!("Config loaded: {:?}", config);
    }
}
