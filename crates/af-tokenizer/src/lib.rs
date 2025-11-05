use anyhow::Result;

pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {}", e))?;
        Ok(Self(tokenizer))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .0
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("encode error: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.0
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decode error: {}", e))
    }
}
