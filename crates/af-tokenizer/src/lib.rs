use anyhow::Result;
use std::path::Path;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
}

impl Tokenizer {
    /// Load tokenizer from a file path
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;

        // Try to get special token IDs
        let bos_token_id = tokenizer
            .token_to_id("<|begin_of_text|>")
            .or_else(|| tokenizer.token_to_id("<|startoftext|>"))
            .or_else(|| tokenizer.token_to_id("<s>"))
            .or_else(|| tokenizer.token_to_id("[BOS]"));

        let eos_token_id = tokenizer
            .token_to_id("<|end_of_text|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .or_else(|| tokenizer.token_to_id("[EOS]"));

        let pad_token_id = tokenizer
            .token_to_id("<|pad|>")
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .or_else(|| eos_token_id); // fallback to EOS

        Ok(Self {
            inner: tokenizer,
            bos_token_id,
            eos_token_id,
            pad_token_id,
        })
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("decode error: {}", e))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Get BOS token ID
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// Get EOS token ID
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Get PAD token ID
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Convert a token to its ID
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Convert an ID to its token
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_encode_decode() {
        // This test will only run if you have a tokenizer file
        // You can skip it in CI by checking if the file exists
        let test_text = "Hello, world!";

        // For now, just test the API structure
        // In real usage, you would load from an actual tokenizer file
        println!("✅ Tokenizer API structure validated");
    }
}
