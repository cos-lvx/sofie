//! Build script — detekuje host CUDA toolkit verzi a validuje
//! `CUDARC_CUDA_VERSION` env var.
//!
//! ## Proč existuje
//!
//! cudarc 0.18.2 čte `CUDARC_CUDA_VERSION` env var při svém vlastním
//! buildu pro mapování na supported CUDA toolkit. Cargo build script
//! nemůže přímo nastavit env var pro dependency build scripts
//! (`cargo:rustc-env` se vztahuje jen na aktuální crate, ne na
//! transitivní deps), takže env var **musí být nastaven před cargo build**.
//!
//! Workspace má v `.cargo/config.toml` default `CUDARC_CUDA_VERSION=13010`
//! s `force = false` — lze override shell exportem (např. ze
//! `scripts/detect-cuda.sh`).
//!
//! Tento build script slouží jako **safety net** — detekuje host CUDA,
//! porovná s nastavenou hodnotou a emituje warning pokud divergují.
//! Ne panic — divergence často OK (cudarc je zpětně kompatibilní), jen
//! upozorní uživatele na možný source build chyb.
//!
//! ## Detekční fallback
//!
//! 1. `nvcc --version` (autoritativní toolkit verze)
//! 2. `nvidia-smi` (driver report — méně přesný pro toolkit)
//! 3. Žádná detekce → tichý skip (CPU build nebo missing CUDA)

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDARC_CUDA_VERSION");

    // Build script běží vždy, ale CUDA validace má smysl jen když je
    // cuda feature aktivní.
    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let configured = match std::env::var("CUDARC_CUDA_VERSION") {
        Ok(v) => v,
        Err(_) => {
            println!(
                "cargo:warning=CUDARC_CUDA_VERSION není nastavená. \
                 cudarc-sys může selhat při buildu. Source \
                 scripts/detect-cuda.sh nebo nastav ručně \
                 (např. export CUDARC_CUDA_VERSION=13010)."
            );
            return;
        }
    };

    let host = match detect_host_cuda() {
        Some(v) => v,
        None => {
            // CUDA není detekovatelná — uživatel asi ví, co dělá
            // (cross-compile, manual override, ...). Nech bez warning.
            return;
        }
    };

    let recommended = recommended_cudarc_version(host.0, host.1);
    if configured != recommended {
        println!(
            "cargo:warning=Host CUDA {}.{} detekováno, ale CUDARC_CUDA_VERSION={configured}. \
             Pro tuto verzi doporučeno {recommended}. cudarc je zpětně kompatibilní, \
             ale pokud build selže s 'CUDA driver too old' nebo podobně: \
             export CUDARC_CUDA_VERSION={recommended}",
            host.0, host.1
        );
    }
}

/// Detekuje host CUDA toolkit verzi. Nejprve zkusí `nvcc --version`
/// (autoritativní), fallback na `nvidia-smi` (driver-reported).
fn detect_host_cuda() -> Option<(u32, u32)> {
    detect_via_nvcc().or_else(detect_via_nvidia_smi)
}

fn detect_via_nvcc() -> Option<(u32, u32)> {
    let output = Command::new("nvcc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_nvcc_release(&stdout)
}

/// Parse "Cuda compilation tools, release 13.0, V13.0.48" → (13, 0).
fn parse_nvcc_release(s: &str) -> Option<(u32, u32)> {
    for line in s.lines() {
        let Some(idx) = line.find("release ") else {
            continue;
        };
        let rest = &line[idx + "release ".len()..];
        let version_str = rest.split(',').next()?.trim();
        let mut parts = version_str.split('.');
        let major: u32 = parts.next()?.parse().ok()?;
        let minor: u32 = parts.next()?.parse().ok()?;
        return Some((major, minor));
    }
    None
}

fn detect_via_nvidia_smi() -> Option<(u32, u32)> {
    let output = Command::new("nvidia-smi").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_nvidia_smi_cuda(&stdout)
}

/// Parse output `nvidia-smi` — hledá řádek s "CUDA Version: X.Y".
fn parse_nvidia_smi_cuda(s: &str) -> Option<(u32, u32)> {
    for line in s.lines() {
        let Some(idx) = line.find("CUDA Version: ") else {
            continue;
        };
        let rest = &line[idx + "CUDA Version: ".len()..];
        let version_str = rest.split_whitespace().next()?;
        let mut parts = version_str.split('.');
        let major: u32 = parts.next()?.parse().ok()?;
        let minor: u32 = parts.next()?.parse().ok()?;
        return Some((major, minor));
    }
    None
}

/// Mapuje host CUDA verzi na `CUDARC_CUDA_VERSION` hodnotu pro cudarc 0.18.x.
///
/// cudarc 0.18.x oficiálně podporuje až CUDA 13.1 → vyšší verze se
/// clampují na 13010 (zpětně kompatibilní v cudarc API).
fn recommended_cudarc_version(major: u32, minor: u32) -> String {
    let value = match (major, minor) {
        (13, m) if m >= 1 => 13010,
        (13, _) => 13000,
        (12, m) if m >= 8 => 12080,
        (12, m) if m >= 6 => 12060,
        (12, m) if m >= 4 => 12040,
        (12, m) if m >= 2 => 12020,
        (12, 1) => 12010,
        (12, 0) => 12000,
        // CUDA 11 — cudarc 0.18.x už nemusí podporovat. Necháváme bez
        // doporučení, uživatel ať si nastaví manuálně pokud ví, co dělá.
        _ => return String::from("(unsupported)"),
    };
    format!("{value:05}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvcc_release_standard_output() {
        let s = "nvcc: NVIDIA (R) Cuda compiler driver\n\
                 Copyright (c) 2005-2025 NVIDIA Corporation\n\
                 Cuda compilation tools, release 13.0, V13.0.48\n\
                 Build cuda_13.0.r13.0/compiler.34112971_0";
        assert_eq!(parse_nvcc_release(s), Some((13, 0)));
    }

    #[test]
    fn parse_nvcc_release_cuda_12() {
        let s = "Cuda compilation tools, release 12.4, V12.4.131";
        assert_eq!(parse_nvcc_release(s), Some((12, 4)));
    }

    #[test]
    fn parse_nvcc_release_missing() {
        assert_eq!(parse_nvcc_release("garbage output"), None);
    }

    #[test]
    fn parse_nvidia_smi_cuda_standard() {
        let s = "Fri May  1 17:00:00 2026\n\
                 +-----------------------------------------------------------------------------+\n\
                 | NVIDIA-SMI 580.159.03   Driver Version: 580.159.03   CUDA Version: 13.0     |\n\
                 |-------------------------------+----------------------+----------------------+";
        assert_eq!(parse_nvidia_smi_cuda(s), Some((13, 0)));
    }

    #[test]
    fn recommended_cudarc_for_13_2() {
        // Arch CUDA 13.2 — clamp na cudarc max 13010
        assert_eq!(recommended_cudarc_version(13, 2), "13010");
    }

    #[test]
    fn recommended_cudarc_for_13_0() {
        // Starfield CUDA 13.0 → 13000
        assert_eq!(recommended_cudarc_version(13, 0), "13000");
    }

    #[test]
    fn recommended_cudarc_for_12_4() {
        assert_eq!(recommended_cudarc_version(12, 4), "12040");
    }

    #[test]
    fn recommended_cudarc_for_unsupported() {
        assert_eq!(recommended_cudarc_version(11, 8), "(unsupported)");
    }
}
