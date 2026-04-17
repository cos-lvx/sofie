//! Loss funkce pro Core Memory training.
//!
//! Cross-entropy na next-token prediction — standardní LM loss:
//! model dostane token sekvenci `[t_0, t_1, ..., t_{N-1}]`, predikuje
//! `[t_1, t_2, ..., t_N]`. Loss porovnává shifted logits proti shifted
//! targets.
//!
//! **Poznámka k shift-by-one konvenci:**
//! Standardní LM setup — model vidí `input[0..N]`, logits vracejí
//! predikci pro každou pozici. Loss bere logits[0..N-1] (predikce
//! next-token) proti targets[1..N] (skutečné next-token).
//!
//! **F32 upcast** pro log_softmax — BF16 má příliš malé rozlišení
//! v blízkosti ±max(logits), může numericky splynout.

use candle_core::{DType, Result, Tensor};
use candle_nn::ops::log_softmax;

/// Cross-entropy loss pro next-token prediction.
///
/// - `logits`: `[batch, seq_len, vocab_size]` — raw logits z lm_head
/// - `input_ids`: `[batch, seq_len]` — token IDs, co model dostal
///
/// Vrací scalar loss `[]` — průměr přes všech `batch * (seq_len - 1)`
/// predikovaných pozic (poslední pozice v seq nemá next-token target).
pub fn cross_entropy_next_token(logits: &Tensor, input_ids: &Tensor) -> Result<Tensor> {
    let (batch, seq_len, _vocab) = logits.dims3()?;
    if seq_len < 2 {
        return Err(candle_core::Error::Msg(
            "cross_entropy_next_token vyžaduje seq_len >= 2 (jinak není co predikovat)".into(),
        ));
    }

    // Shift-by-one: predikujeme token[t+1] z pozice t.
    // - logits[0..seq_len-1] = predikce pro pozice 1..seq_len
    // - targets = input_ids[1..seq_len]
    let shifted_logits = logits.narrow(1, 0, seq_len - 1)?; // [b, s-1, v]
    let shifted_targets = input_ids.narrow(1, 1, seq_len - 1)?; // [b, s-1]

    // log_softmax v F32 (BF16 nestabilní pro extreme logits)
    let logits_f32 = shifted_logits.to_dtype(DType::F32)?;
    let log_probs = log_softmax(&logits_f32, 2)?; // [b, s-1, v]

    // Gather — vytáhni log_prob pro target token na každé pozici.
    // Targets musí být [b, s-1, 1] pro gather na dim=2.
    let targets_idx = shifted_targets.unsqueeze(2)?; // [b, s-1, 1]
    let gathered = log_probs.gather(&targets_idx, 2)?; // [b, s-1, 1]
    let gathered = gathered.squeeze(2)?; // [b, s-1]

    // NLL = -log_prob, mean přes [batch, seq_len-1]
    let nll = gathered.neg()?;
    let loss = nll.mean_all()?; // skalár
    let _ = batch; // batch použitý implicitně přes mean_all
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn cross_entropy_on_confident_correct_prediction_is_near_zero() -> Result<()> {
        // Predikujeme sekvenci [5, 7, 3] → target shift [7, 3]
        // Logits nastavíme tak, že target token má masivní score, ostatní mizivý.
        let vocab = 10;
        let seq_len = 3;
        let batch = 1;
        let mut logits_data = vec![0.0f32; batch * seq_len * vocab];

        // Pozice 0: chceme predikovat token 7 (target shift[0])
        logits_data[7] = 100.0;
        // Pozice 1: chceme predikovat token 3 (target shift[1])
        logits_data[vocab + 3] = 100.0;
        // Pozice 2: nezajímá nás (není target pro shift)

        let logits = Tensor::from_vec(logits_data, (batch, seq_len, vocab), &Device::Cpu)?;
        let input_ids = Tensor::from_vec(vec![5u32, 7, 3], (batch, seq_len), &Device::Cpu)?;

        let loss = cross_entropy_next_token(&logits, &input_ids)?;
        let loss_val: f32 = loss.to_scalar()?;

        // Loss by měl být velmi blízko 0 (model si je jistý a má pravdu).
        assert!(
            loss_val < 1e-4,
            "confident correct prediction: loss={loss_val}, očekáváno ~0"
        );
        Ok(())
    }

    #[test]
    fn cross_entropy_on_uniform_logits_equals_log_vocab() -> Result<()> {
        // Uniform distribuce přes vocab → cross-entropy = ln(vocab)
        let vocab = 100;
        let seq_len = 5;
        let batch = 1;
        let logits = Tensor::zeros((batch, seq_len, vocab), DType::F32, &Device::Cpu)?;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4, 5], (batch, seq_len), &Device::Cpu)?;

        let loss = cross_entropy_next_token(&logits, &input_ids)?;
        let loss_val: f32 = loss.to_scalar()?;

        let expected = (vocab as f32).ln();
        let diff = (loss_val - expected).abs();
        assert!(
            diff < 1e-3,
            "uniform logits: loss={loss_val}, expected ln(vocab)={expected}, diff={diff}"
        );
        Ok(())
    }

    #[test]
    fn cross_entropy_requires_seq_len_at_least_2() {
        let logits = Tensor::zeros((1, 1, 10), DType::F32, &Device::Cpu).unwrap();
        let input_ids = Tensor::from_vec(vec![5u32], (1, 1), &Device::Cpu).unwrap();
        let result = cross_entropy_next_token(&logits, &input_ids);
        assert!(result.is_err(), "seq_len=1 musí odmítnout");
    }

    #[test]
    fn cross_entropy_gradient_flows_to_logits() -> Result<()> {
        use candle_core::Var;

        let vocab = 10;
        let seq_len = 4;
        let batch = 1;

        // Trainable logits Var.
        let logits_var =
            Var::randn_f64(0.0, 1.0, (batch, seq_len, vocab), DType::F32, &Device::Cpu)?;
        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (batch, seq_len), &Device::Cpu)?;

        let loss = cross_entropy_next_token(logits_var.as_tensor(), &input_ids)?;
        let grads = loss.backward()?;
        let logits_grad = grads
            .get(logits_var.as_tensor())
            .expect("gradient musí být přítomen");

        // Gradient non-zero — backward doteče k logits.
        let grad_norm: f32 = logits_grad.sqr()?.sum_all()?.to_scalar()?;
        assert!(
            grad_norm.is_finite() && grad_norm > 0.0,
            "logits gradient norm={grad_norm}, očekáváno finite a >0"
        );
        Ok(())
    }
}
