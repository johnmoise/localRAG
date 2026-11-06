//! PDFium runs in a separate helper process, `localrag-pdf` (ARCHITECTURE.md §14).
//!
//! PDFium is C++ and parses untrusted files. Running it out of process means a crash or
//! memory-corruption bug in a PDF can't take down or take over the app: the worst case is
//! that one file is marked `failed`. On macOS the helper also runs under `sandbox-exec` with
//! network access and file writes denied, so even a compromised parser can't exfiltrate data
//! or modify files. Windows has no equivalent of `sandbox-exec`: there the helper runs in a
//! Job Object that kills it if the app goes away and stops it from starting other processes,
//! so it keeps the crash isolation but not the network and write restrictions.
//!
//! One helper process per request, speaking one JSON line each way: simple, and no state
//! survives from one document to the next.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::parse::{FileError, ParsedDoc};

pub const HELPER_NAME: &str = "localrag-pdf";

/// The helper's file name on this platform (`localrag-pdf.exe` on Windows).
fn helper_file_name() -> String {
    format!("{HELPER_NAME}{}", std::env::consts::EXE_SUFFIX)
}

/// No network, no file writes (stdout/stderr are pipes, not files).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const SANDBOX_PROFILE: &str = r#"(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write-data (literal "/dev/null"))"#;

#[derive(Debug, Clone)]
pub struct PdfHelper {
    exe: PathBuf,
    pdfium_lib: PathBuf,
    sandbox: Option<PathBuf>,
}

/// Characters of one page, for highlighting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageChars {
    pub pdf_page: u32,
    pub width: f32,
    pub height: f32,
    /// Each char with its box [left, bottom, right, top] in PDF points, as PDFium reports
    /// them (including generated "\r\n").
    pub chars: Vec<(char, Option<[f32; 4]>)>,
}

impl PdfHelper {
    /// Finds the helper binary (next to the running executable, or `$LOCALRAG_PDF_HELPER`)
    /// and the PDFium library it should load.
    pub fn locate(pdfium_configured: Option<&Path>) -> anyhow::Result<Self> {
        let exe = match std::env::var_os("LOCALRAG_PDF_HELPER") {
            Some(p) => PathBuf::from(p),
            None => {
                let here = std::env::current_exe()?;
                let dir = here.parent().context("executable has no folder")?;
                let name = helper_file_name();
                [dir.join(&name), dir.join("..").join(&name)]
                    .into_iter()
                    .find(|p| p.is_file())
                    .with_context(|| format!("{HELPER_NAME} not found next to {}", here.display()))?
            }
        };
        let pdfium_lib = crate::parse::pdf::find_library(pdfium_configured)?;
        let sandbox = Some(PathBuf::from("/usr/bin/sandbox-exec"))
            .filter(|p| cfg!(target_os = "macos") && p.is_file() && std::env::var_os("LOCALRAG_NO_SANDBOX").is_none());
        Ok(PdfHelper { exe, pdfium_lib, sandbox })
    }

    pub fn is_sandboxed(&self) -> bool {
        self.sandbox.is_some()
    }

    /// How the helper is contained, for `localrag doctor`.
    pub fn isolation(&self) -> &'static str {
        if self.is_sandboxed() {
            "sandboxed (no network, no writes)"
        } else if cfg!(windows) {
            "separate process in a Job Object (no sandbox on Windows)"
        } else {
            "separate process, NOT sandboxed"
        }
    }

    /// Parses a PDF. File problems (damaged, password, crash, timeout) are `Ok(Err(_))`;
    /// helper problems (missing binary) are `Err(_)`.
    pub fn parse(&self, path: &Path) -> anyhow::Result<Result<ParsedDoc, FileError>> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        if let Err(e) = crate::parse::pdf::sniff(&bytes) {
            return Ok(Err(e));
        }
        Ok(self.call(json!({ "op": "parse", "path": path }), Duration::from_secs(300))?
            .and_then(|v| serde_json::from_value(v["doc"].clone()).map_err(|e| FileError(format!("helper sent a bad reply: {e}")))))
    }

    pub fn chars(&self, path: &Path, first: u32, last: u32) -> anyhow::Result<Result<Vec<PageChars>, FileError>> {
        Ok(self.call(json!({ "op": "chars", "path": path, "first": first, "last": last }), Duration::from_secs(60))?
            .and_then(|v| serde_json::from_value(v["pages"].clone()).map_err(|e| FileError(format!("helper sent a bad reply: {e}")))))
    }

    /// PNG of one page rendered `width` pixels wide (for OCR).
    pub fn render(&self, path: &Path, page: u32, width: u32) -> anyhow::Result<Result<Vec<u8>, FileError>> {
        Ok(self.call(json!({ "op": "render", "path": path, "page": page, "width": width }), Duration::from_secs(60))?.and_then(|v| {
            decode_hex(v["png_hex"].as_str().unwrap_or_default()).ok_or_else(|| FileError("helper sent a bad image".into()))
        }))
    }

    fn call(&self, request: Value, timeout: Duration) -> anyhow::Result<Result<Value, FileError>> {
        let mut cmd = match &self.sandbox {
            Some(sb) => {
                let mut c = Command::new(sb);
                c.arg("-p").arg(SANDBOX_PROFILE).arg(&self.exe);
                c
            }
            None => Command::new(&self.exe),
        };
        let mut child = cmd
            .env("LOCALRAG_PDFIUM", &self.pdfium_lib)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", self.exe.display()))?;
        #[cfg(windows)]
        let _job = job::contain(&child).with_context(|| format!("containing {}", self.exe.display()))?;
        child.stdin.take().expect("piped").write_all(format!("{request}\n").as_bytes())?;

        // Read stdout on a thread so a large reply can't fill the pipe and stall the helper.
        let mut stdout = child.stdout.take().expect("piped");
        let reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            stdout.read_to_end(&mut buf).map(|_| buf)
        });
        let started = Instant::now();
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break status;
            }
            if started.elapsed() > timeout {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(Err(FileError(format!("the PDF took longer than {}s to process and was stopped", timeout.as_secs()))));
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let out = reader.join().map_err(|_| anyhow::anyhow!("helper reader panicked"))??;
        let mut stderr = String::new();
        if let Some(mut e) = child.stderr.take() {
            let _ = e.read_to_string(&mut stderr);
        }
        if !status.success() {
            if let Some(how) = crash(&status) {
                return Ok(Err(FileError(format!("the PDF parser crashed on this file ({how}); the file may be malformed"))));
            }
            if out.is_empty() {
                bail!("{HELPER_NAME} failed ({status}): {}", stderr.trim());
            }
        }
        let v: Value = serde_json::from_slice(&out).with_context(|| format!("{HELPER_NAME} reply: {}", stderr.trim()))?;
        if let Some(e) = v["file_error"].as_str() {
            return Ok(Err(FileError(e.to_string())));
        }
        if let Some(e) = v["error"].as_str() {
            bail!("{HELPER_NAME}: {e}");
        }
        Ok(Ok(v))
    }
}

/// Describes an exit that was a crash rather than an orderly failure.
#[cfg(unix)]
fn crash(status: &std::process::ExitStatus) -> Option<String> {
    use std::os::unix::process::ExitStatusExt;
    status.signal().map(|sig| format!("signal {sig}"))
}

/// Windows reports crashes as NTSTATUS error codes (0xC…, e.g. 0xC0000005 access violation).
#[cfg(windows)]
fn crash(status: &std::process::ExitStatus) -> Option<String> {
    let code = status.code()? as u32;
    (code >= 0xC000_0000).then(|| format!("exception 0x{code:08X}"))
}

#[cfg(not(any(unix, windows)))]
fn crash(_status: &std::process::ExitStatus) -> Option<String> {
    None
}

/// A Job Object for the helper: closing it (when `call` returns, or when the app dies) kills
/// the helper, and the helper can't start processes of its own.
#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    pub struct Job(HANDLE);

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    pub fn contain(child: &std::process::Child) -> windows::core::Result<Job> {
        unsafe {
            let job = Job(CreateJobObjectW(None, windows::core::PCWSTR::null())?);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags =
                JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION;
            info.BasicLimitInformation.ActiveProcessLimit = 1;
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )?;
            AssignProcessToJobObject(job.0, HANDLE(child.as_raw_handle()))?;
            Ok(job)
        }
    }
}

pub fn encode_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn decode_hex(s: &str) -> Option<Vec<u8>> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(s.get(i..i + 2)?, 16).ok()).collect()
}
