//! Only one localrag runs at a time (ARCHITECTURE.md §4): the server holds an exclusive lock
//! (`flock` on Unix, `LockFileEx` on Windows, via `File::try_lock`) on `instance.lock` in the
//! global config folder for its lifetime. The OS releases it when the
//! process exits, even on a crash, so a stale lock can't block the next start.
//!
//! `instance.json` (private to this user, since it carries the launch token) tells a second invocation
//! where the running one is.

use std::fs::{File, TryLockError};
use std::io::Write;
use std::path::PathBuf;

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::config::{ensure_private_dir, global_config_dir, private_file_options};

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
    let file = private_file_options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path())
        .with_context(|| format!("opening {}", lock_path().display()))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::Error(e)) => return Err(e).with_context(|| format!("locking {}", lock_path().display())),
        Err(TryLockError::WouldBlock) => {
        let running = std::fs::read_to_string(info_path())
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(RunningInstance { pid: 0, url: None });
            return Ok(Err(running));
        }
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
    let mut f = private_file_options().create(true).truncate(true).write(true).open(info_path())?;
    f.write_all(serde_json::to_string(info)?.as_bytes())?;
    Ok(())
}
