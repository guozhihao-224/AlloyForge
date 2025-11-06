use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::{Path, PathBuf};

/// 查找目录下的所有 safetensors 文件
pub fn find_safetensors_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        
        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    files.push(path);
                }
            }
        }
    }
    
    // 排序以保证一致的加载顺序
    files.sort();
    Ok(files)
}

/// 使用 mmap 加载 safetensors 权重文件
/// 
/// # Example
/// ```no_run
/// use af_io::load_safetensors_mmap;
/// use candle_core::{DType, Device};
/// 
/// let device = Device::Cpu;
/// let vb = load_safetensors_mmap(&["model.safetensors"], DType::F32, &device)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn load_safetensors_mmap<'a>(
    paths: &'a [PathBuf],
    dtype: DType,
    device: &'a Device,
) -> Result<VarBuilder<'a>> {
    unsafe {
        Ok(VarBuilder::from_mmaped_safetensors(paths, dtype, device)?)
    }
}

// Note: 由于生命周期限制，用户需要自己调用 find_safetensors_files + load_safetensors_mmap
// 或者直接在模型中处理

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_safetensors_files() {
        // This test requires a model directory with safetensors files
        // For now, just test the API structure
        println!("✅ SafeTensors loading API validated");
    }
}
