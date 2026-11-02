//! Only one localrag runs at a time (ARCHITECTURE.md §4): the server holds an exclusive
//! `flock` on `~/.config/localrag/instance.lock` for its lifetime. The OS releases it when the
//! process exits, even on a crash, so a stale lock can't block the next start.
//!
//! `instance.json` (mode 0600, since it carries the launch token) tells a second invocation
//! where the running one is.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::{ensure_private_dir, global_config_dir};

/// Held for as long as this process is the running instance.
pub struct InstanceGuard {
    _lock: File,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningInstance {
    pub pid: u32,
    /// The launch URL (with token), once the server is listening.
    pub url: Option<String>,
}

fn lock_path() -> PathBuf {
    global_config_dir().join("instance.lock")
}

fn info_path() -> PathBuf {
    global_config_dir().join("instance.json")
}

/// Becomes the running instance, or reports the one that already is.
pub fn acquire() -> anyhow::Result<Result<InstanceGuard, RunningInstance>> {
    ensure_private_dir(&global_config_dir())?;
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .mode(0o600)
        .open(lock_path())
        .with_context(|| format!("opening {}", lock_path().display()))?;
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let running = std::fs::read_to_string(info_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(RunningInstance { pid: 0, url: None });
        return Ok(Err(running));
    }
    write_info(&RunningInstance { pid: std::process::id(), url: None })?;
    Ok(Ok(InstanceGuard { _lock: file }))
}

impl InstanceGuard {
    /// Records the launch URL so a second `localrag serve` can open it instead of starting.
    pub fn publish_url(&self, url: &str) -> anyhow::Result<()> {
        write_info(&RunningInstance { pid: std::process::id(), url: Some(url.to_string()) })
    }
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(info_path());
    }
}

fn write_info(info: &RunningInstance) -> anyhow::Result<()> {
    let mut f = OpenOptions::new().create(true).truncate(true).write(true).mode(0o600).open(info_path())?;
    f.write_all(serde_json::to_string(info)?.as_bytes())?;
    Ok(())
}
