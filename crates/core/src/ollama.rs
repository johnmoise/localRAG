//! Making sure Ollama is there (ARCHITECTURE.md §19.1): probe `ollama_host`, and when it
//! doesn't answer and points at this machine, find an installed Ollama and start it in the
//! background. Also lists and pulls the models a workspace needs.
//!
//! Nothing is ever started for a host on another machine, and the only process localRAG may
//! stop later is an `ollama serve` it spawned itself ([`ServeChild`]).

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader};
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::{Value, json};

pub const DOWNLOAD_URL: &str = "https://ollama.com/download";

/// How long a probe of `/api/version` may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
/// How long to wait for a freshly started Ollama to answer.
pub const START_TIMEOUT: Duration = Duration::from_secs(20);

/// Where Ollama stands, as the UI and `doctor` show it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum Status {
    /// Answering at `ollama_host`.
    Running { version: String, started: Started },
    /// Not answering, local, and no installation found.
    NotInstalled,
    /// Installed but not answering, even after trying to start it.
    NotRunning { error: String },
    /// `ollama_host` is another machine and didn't answer; never started from here.
    Remote { error: String },
}

/// What localRAG did to get Ollama running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Started {
    /// It was already running.
    No,
    /// Launched the desktop app (it keeps running after localRAG exits and updates itself).
    App,
    /// Spawned `ollama serve`; the only case localRAG may stop on exit.
    Serve,
}

/// An installed Ollama: the desktop app and/or the command-line binary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Installation {
    /// macOS: `Ollama.app`; Windows: the tray app `ollama app.exe`.
    pub app: Option<PathBuf>,
    /// `ollama` / `ollama.exe`.
    pub cli: Option<PathBuf>,
}

impl Installation {
    pub fn found(&self) -> bool {
        self.app.is_some() || self.cli.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Os {
    MacOs,
    Windows,
    Other,
}

impl Os {
    pub const CURRENT: Os = if cfg!(target_os = "macos") {
        Os::MacOs
    } else if cfg!(windows) {
        Os::Windows
    } else {
        Os::Other
    };

    pub fn name(self) -> &'static str {
        match self {
            Os::MacOs => "macos",
            Os::Windows => "windows",
            Os::Other => "other",
        }
    }
}

/// Where to look for an installation. [`SearchEnv::current`] reads the real environment;
/// tests build their own, and [`SearchEnv::none`] finds nothing (so nothing is started).
#[derive(Debug, Clone)]
pub struct SearchEnv {
    pub os: Os,
    pub home: Option<PathBuf>,
    /// Windows `%LOCALAPPDATA%`.
    pub local_app_data: Option<PathBuf>,
    /// `PATH`, already split.
    pub path: Vec<PathBuf>,
    /// Folders holding `Ollama.app` on macOS (besides `~/Applications`).
    pub app_dirs: Vec<PathBuf>,
    /// Extra folders searched for the binary after `PATH` (Homebrew's, which a GUI launch
    /// often lacks on `PATH`).
    pub extra_bin_dirs: Vec<PathBuf>,
}

impl SearchEnv {
    /// The real environment; `LOCALRAG_OLLAMA_SEARCH=none` finds nothing (to try the
    /// first-run dialog on a machine that has Ollama).
    pub fn current() -> Self {
        if std::env::var_os("LOCALRAG_OLLAMA_SEARCH").is_some_and(|v| v == "none") {
            return SearchEnv::none();
        }
        let os = Os::CURRENT;
        let path = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).collect()).unwrap_or_default();
        let (app_dirs, extra_bin_dirs) = match os {
            Os::MacOs => (vec!["/Applications".into()], vec!["/opt/homebrew/bin".into(), "/usr/local/bin".into()]),
            _ => (Vec::new(), Vec::new()),
        };
        SearchEnv {
            os,
            home: std::env::var_os(if os == Os::Windows { "USERPROFILE" } else { "HOME" }).map(PathBuf::from),
            local_app_data: std::env::var_os("LOCALAPPDATA").map(PathBuf::from),
            path,
            app_dirs,
            extra_bin_dirs,
        }
    }

    /// Finds nothing.
    pub fn none() -> Self {
        SearchEnv { os: Os::CURRENT, home: None, local_app_data: None, path: Vec::new(), app_dirs: Vec::new(), extra_bin_dirs: Vec::new() }
    }

    /// Looks for an installed Ollama.
    pub fn discover(&self) -> Installation {
        match self.os {
            Os::MacOs => {
                let app = self
                    .app_dirs
                    .iter()
                    .cloned()
                    .chain(self.home.iter().map(|h| h.join("Applications")))
                    .map(|d| d.join("Ollama.app"))
                    .find(|p| p.is_dir());
                let cli = self.find_bin("ollama").or_else(|| {
                    // The app carries the CLI inside its bundle.
                    app.as_ref().map(|a| a.join("Contents/Resources/ollama")).filter(|p| p.is_file())
                });
                Installation { app, cli }
            }
            Os::Windows => {
                let install_dir = self.local_app_data.as_ref().map(|d| d.join("Programs").join("Ollama"));
                let cli = install_dir
                    .as_ref()
                    .map(|d| d.join("ollama.exe"))
                    .filter(|p| p.is_file())
                    .or_else(|| self.find_bin("ollama.exe"));
                // The tray app sits next to the CLI in a normal install.
                let app = install_dir
                    .iter()
                    .chain(cli.as_ref().and_then(|c| c.parent()).map(Path::to_path_buf).iter())
                    .map(|d| d.join("ollama app.exe"))
                    .find(|p| p.is_file());
                Installation { app, cli }
            }
            Os::Other => Installation { app: None, cli: self.find_bin("ollama") },
        }
    }

    fn find_bin(&self, name: &str) -> Option<PathBuf> {
        self.path.iter().chain(&self.extra_bin_dirs).map(|d| d.join(name)).find(|p| p.is_file())
    }
}

/// True when `host` (an `http://host:port` URL) is this machine: `localhost`, a loopback
/// address, or the unspecified address. Anything unparsable counts as remote.
pub fn is_local_host(host: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(host) else { return false };
    let Some(h) = url.host_str() else { return false };
    let h = h.trim_start_matches('[').trim_end_matches(']');
    if h.eq_ignore_ascii_case("localhost") {
        return true;
    }
    h.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback() || ip.is_unspecified())
}

/// `host:port` for `OLLAMA_HOST`, so a spawned `ollama serve` listens where we'll look.
fn bind_address(host: &str) -> Option<String> {
    let url = reqwest::Url::parse(host).ok()?;
    let port = url.port_or_known_default()?;
    Some(format!("{}:{port}", url.host_str()?))
}

fn client(timeout: Option<Duration>) -> anyhow::Result<reqwest::blocking::Client> {
    Ok(reqwest::blocking::Client::builder().connect_timeout(PROBE_TIMEOUT).timeout(timeout).build()?)
}

/// Ollama's version at `host`, or why it didn't answer within a couple of seconds.
pub fn probe(host: &str) -> anyhow::Result<String> {
    let host = host.trim_end_matches('/');
    let v: Value = client(Some(PROBE_TIMEOUT))?
        .get(format!("{host}/api/version"))
        .send()
        .with_context(|| format!("no response from Ollama at {host}"))?
        .json()
        .context("not an Ollama server (unexpected /api/version reply)")?;
    Ok(v["version"].as_str().unwrap_or("?").to_string())
}

/// An `ollama serve` that localRAG spawned. Dropping it leaves Ollama running; only
/// [`ServeChild::stop`] ends it.
#[derive(Debug)]
pub struct ServeChild {
    child: Child,
}

impl ServeChild {
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Asks it to exit (SIGTERM on Unix), then reaps it.
    pub fn stop(mut self) {
        #[cfg(unix)]
        {
            // SAFETY: plain kill(2) on our own child's pid.
            unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) };
            for _ in 0..50 {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The result of [`ensure_running`].
#[derive(Debug)]
pub struct Ensured {
    pub status: Status,
    pub installation: Installation,
    /// Set when localRAG spawned `ollama serve` itself.
    pub serve: Option<ServeChild>,
}

/// Probes `host`; if Ollama isn't answering and `host` is this machine, finds an installation
/// with `env` and starts it: the desktop app if there is one, else `ollama serve` (output
/// appended to `log_dir/ollama.log`). Then waits up to `wait` for it to answer.
/// `on_starting` is called just before anything is launched.
pub fn ensure_running(host: &str, env: &SearchEnv, log_dir: &Path, wait: Duration, on_starting: &mut dyn FnMut()) -> Ensured {
    let first = match probe(host) {
        Ok(version) => {
            return Ensured { status: Status::Running { version, started: Started::No }, installation: Installation::default(), serve: None };
        }
        Err(e) => e,
    };
    if !is_local_host(host) {
        return Ensured { status: Status::Remote { error: format!("{first:#}") }, installation: Installation::default(), serve: None };
    }
    let installation = env.discover();
    if !installation.found() {
        return Ensured { status: Status::NotInstalled, installation, serve: None };
    }
    on_starting();
    let mut serve = None;
    let mut started = Started::App;
    let mut launch_err = None;
    let app_ok = match &installation.app {
        Some(app) => match launch_app(env.os, app) {
            Ok(()) => true,
            Err(e) => {
                launch_err = Some(e);
                false
            }
        },
        None => false,
    };
    if !app_ok {
        match installation.cli.as_deref().map(|cli| spawn_serve(env.os, cli, host, log_dir)) {
            Some(Ok(child)) => {
                serve = Some(child);
                started = Started::Serve;
            }
            Some(Err(e)) => launch_err = Some(e),
            None => {}
        }
        if serve.is_none() {
            let error = launch_err.map(|e| format!("couldn't start Ollama: {e:#}")).unwrap_or_else(|| "couldn't start Ollama".into());
            return Ensured { status: Status::NotRunning { error }, installation, serve: None };
        }
    }
    let status = match wait_until_up(host, wait, serve.as_mut()) {
        Ok(version) => Status::Running { version, started },
        Err(e) => Status::NotRunning { error: format!("{e:#}") },
    };
    Ensured { status, installation, serve }
}

fn launch_app(os: Os, app: &Path) -> anyhow::Result<()> {
    let mut cmd = match os {
        // -g: don't bring it to the front.
        Os::MacOs => {
            let mut c = Command::new("open");
            c.arg("-g").arg("-a").arg(app);
            c
        }
        _ => Command::new(app),
    };
    cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    detach(&mut cmd);
    if os == Os::MacOs {
        let st = cmd.status().context("running `open`")?;
        if !st.success() {
            bail!("`open -a {}` failed ({st})", app.display());
        }
    } else {
        cmd.spawn().with_context(|| format!("starting {}", app.display()))?;
    }
    Ok(())
}

fn spawn_serve(_os: Os, cli: &Path, host: &str, log_dir: &Path) -> anyhow::Result<ServeChild> {
    std::fs::create_dir_all(log_dir).with_context(|| format!("creating {}", log_dir.display()))?;
    let log_path = log_dir.join("ollama.log");
    let log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path).with_context(|| format!("opening {}", log_path.display()))?;
    let mut cmd = Command::new(cli);
    cmd.arg("serve").stdin(Stdio::null()).stdout(log.try_clone()?).stderr(log);
    if let Some(addr) = bind_address(host) {
        cmd.env("OLLAMA_HOST", addr);
    }
    detach(&mut cmd);
    let child = cmd.spawn().with_context(|| format!("starting `{} serve`", cli.display()))?;
    Ok(ServeChild { child })
}

/// Keeps a started process out of our terminal's process group (so Ctrl-C on localRAG doesn't
/// take Ollama down) and, on Windows, without a console window.
fn detach(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(not(any(unix, windows)))]
    let _ = cmd;
}

/// Polls `/api/version` until it answers or `wait` passes. Gives up early if the spawned
/// `serve` exits (e.g. the port is taken by something else).
fn wait_until_up(host: &str, wait: Duration, mut serve: Option<&mut ServeChild>) -> anyhow::Result<String> {
    let deadline = Instant::now() + wait;
    loop {
        match probe(host) {
            Ok(v) => return Ok(v),
            Err(e) if Instant::now() >= deadline => {
                return Err(e.context(format!("Ollama was started but didn't answer within {} s", wait.as_secs())));
            }
            Err(_) => {}
        }
        if let Some(s) = serve.as_deref_mut()
            && let Ok(Some(code)) = s.child.try_wait()
        {
            bail!("`ollama serve` exited ({code}); see ollama.log");
        }
        std::thread::sleep(Duration::from_millis(300));
    }
}

// ---------------------------------------------------------------- models

/// Approximate download sizes of the default models, shown before a pull reports its own.
pub fn known_size(model: &str) -> Option<u64> {
    Some(match model {
        "qwen3-embedding:0.6b" => 639_000_000,
        "qwen3.5:4b" => 3_400_000_000,
        "qwen3.5:9b" => 6_600_000_000,
        _ => return None,
    })
}

/// `name` with Ollama's implicit `:latest` tag.
fn canonical(name: &str) -> String {
    if name.contains(':') { name.to_string() } else { format!("{name}:latest") }
}

/// Names of the models installed at `host`.
pub fn installed_models(host: &str) -> anyhow::Result<Vec<String>> {
    let host = host.trim_end_matches('/');
    let v: Value = client(Some(Duration::from_secs(10)))?.get(format!("{host}/api/tags")).send()?.error_for_status()?.json()?;
    Ok(v["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| m["name"].as_str().or(m["model"].as_str()).map(str::to_string))
        .collect())
}

/// Which of `wanted` aren't in `installed` (comparing with the implicit `:latest` tag).
pub fn missing<'a>(wanted: &[&'a str], installed: &[String]) -> Vec<&'a str> {
    let have: Vec<String> = installed.iter().map(|n| canonical(n)).collect();
    wanted.iter().copied().filter(|w| !have.contains(&canonical(w))).collect()
}

/// Progress of one pull, summed over its layers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PullProgress {
    pub status: String,
    pub completed: u64,
    pub total: u64,
}

/// Downloads `model` through Ollama's `POST /api/pull`, reporting streamed progress.
pub fn pull(host: &str, model: &str, on_progress: &mut dyn FnMut(&PullProgress)) -> anyhow::Result<()> {
    let host = host.trim_end_matches('/');
    let resp = client(None)?
        .post(format!("{host}/api/pull"))
        .json(&json!({ "model": model, "stream": true }))
        .send()
        .with_context(|| format!("no response from Ollama at {host}"))?;
    if !resp.status().is_success() {
        let status = resp.status();
        let v: Value = resp.json().unwrap_or_default();
        bail!("pulling {model} failed ({status}): {}", v["error"].as_str().unwrap_or("unknown error"));
    }
    let mut layers: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut progress = PullProgress::default();
    for line in BufReader::new(resp).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(&line).context("bad pull stream line")?;
        if let Some(e) = v["error"].as_str() {
            bail!("pulling {model}: {e}");
        }
        if let (Some(d), Some(total)) = (v["digest"].as_str(), v["total"].as_u64()) {
            layers.insert(d.to_string(), (v["completed"].as_u64().unwrap_or(0), total));
        }
        progress.status = v["status"].as_str().unwrap_or_default().to_string();
        progress.completed = layers.values().map(|l| l.0).sum();
        progress.total = layers.values().map(|l| l.1).sum();
        on_progress(&progress);
        if progress.status == "success" {
            return Ok(());
        }
    }
    bail!("pulling {model}: the stream ended before Ollama reported success")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(p: &Path) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, b"").unwrap();
    }

    fn env(os: Os, root: &Path) -> SearchEnv {
        SearchEnv { os, home: Some(root.join("home")), local_app_data: None, path: Vec::new(), app_dirs: vec![root.join("Applications")], extra_bin_dirs: Vec::new() }
    }

    #[test]
    fn remote_hosts_are_never_local() {
        for h in ["http://127.0.0.1:11434", "http://localhost:11434", "http://LOCALHOST:1", "http://[::1]:11434", "http://0.0.0.0:11434", "http://127.0.0.2:11434"] {
            assert!(is_local_host(h), "{h}");
        }
        for h in ["http://10.0.0.5:11434", "http://gpu-box.local:11434", "https://ollama.example.com", "http://[2001:db8::1]:11434", "http://localhost.evil.com:11434", "not a url", ""] {
            assert!(!is_local_host(h), "{h}");
        }
    }

    #[test]
    fn remote_host_is_reported_without_starting_anything() {
        let dir = tempfile::tempdir().unwrap();
        let mut e = env(Os::MacOs, dir.path());
        // An installation that *would* be started if the host were local.
        std::fs::create_dir_all(dir.path().join("Applications/Ollama.app")).unwrap();
        e.path = vec![dir.path().join("bin")];
        touch(&dir.path().join("bin/ollama"));
        let mut starting = false;
        // TEST-NET-1: never routable, so the probe fails (fast, thanks to the connect timeout).
        let r = ensure_running("http://192.0.2.1:11434", &e, dir.path(), Duration::from_millis(10), &mut || starting = true);
        assert!(matches!(r.status, Status::Remote { .. }), "{:?}", r.status);
        assert!(!starting && r.serve.is_none());
    }

    #[test]
    fn nothing_installed_is_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        // A local port nothing listens on.
        let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let r = ensure_running(&format!("http://127.0.0.1:{port}"), &SearchEnv::none(), dir.path(), Duration::from_millis(10), &mut || panic!("started"));
        assert_eq!(r.status, Status::NotInstalled);
    }

    #[test]
    fn macos_prefers_the_app_and_finds_the_cli_on_path_or_homebrew() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut e = env(Os::MacOs, root);
        assert_eq!(e.discover(), Installation::default());

        // ~/Applications
        std::fs::create_dir_all(root.join("home/Applications/Ollama.app/Contents/Resources")).unwrap();
        assert_eq!(e.discover().app, Some(root.join("home/Applications/Ollama.app")));
        assert_eq!(e.discover().cli, None);
        touch(&root.join("home/Applications/Ollama.app/Contents/Resources/ollama"));
        assert_eq!(e.discover().cli, Some(root.join("home/Applications/Ollama.app/Contents/Resources/ollama")));

        // /Applications wins over ~/Applications.
        std::fs::create_dir_all(root.join("Applications/Ollama.app")).unwrap();
        assert_eq!(e.discover().app, Some(root.join("Applications/Ollama.app")));

        // Homebrew dir after PATH; PATH first.
        e.extra_bin_dirs = vec![root.join("opt/homebrew/bin")];
        touch(&root.join("opt/homebrew/bin/ollama"));
        assert_eq!(e.discover().cli, Some(root.join("opt/homebrew/bin/ollama")));
        e.path = vec![root.join("missing"), root.join("usr/bin")];
        touch(&root.join("usr/bin/ollama"));
        assert_eq!(e.discover().cli, Some(root.join("usr/bin/ollama")));

        // A directory named `ollama` isn't a binary.
        let mut e2 = env(Os::MacOs, &root.join("nowhere"));
        std::fs::create_dir_all(root.join("dirbin/ollama")).unwrap();
        e2.path = vec![root.join("dirbin")];
        assert_eq!(e2.discover(), Installation::default());
    }

    #[test]
    fn windows_finds_the_per_user_install_and_its_tray_app() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut e = env(Os::Windows, root);
        e.local_app_data = Some(root.join("AppData/Local"));
        assert_eq!(e.discover(), Installation::default());

        // ollama.exe on PATH only (e.g. a scoop install): no tray app.
        e.path = vec![root.join("scoop/shims")];
        touch(&root.join("scoop/shims/ollama.exe"));
        assert_eq!(e.discover(), Installation { app: None, cli: Some(root.join("scoop/shims/ollama.exe")) });

        // The installer's location wins, and its tray app is found.
        let install = root.join("AppData/Local/Programs/Ollama");
        touch(&install.join("ollama.exe"));
        touch(&install.join("ollama app.exe"));
        assert_eq!(e.discover(), Installation { app: Some(install.join("ollama app.exe")), cli: Some(install.join("ollama.exe")) });

        // The macOS name isn't looked for on Windows.
        let mut e3 = env(Os::Windows, &root.join("x"));
        touch(&root.join("unix/ollama"));
        e3.path = vec![root.join("unix")];
        assert_eq!(e3.discover(), Installation::default());
    }

    #[test]
    fn missing_models_respect_the_latest_tag() {
        let installed = vec!["llama3.2:latest".to_string(), "qwen3.5:4b".to_string()];
        assert_eq!(missing(&["llama3.2", "qwen3.5:4b", "qwen3.5:9b"], &installed), vec!["qwen3.5:9b"]);
    }

    #[test]
    fn bind_address_follows_the_configured_host() {
        assert_eq!(bind_address("http://127.0.0.1:11500").as_deref(), Some("127.0.0.1:11500"));
        assert_eq!(bind_address("http://localhost:11434/").as_deref(), Some("localhost:11434"));
    }
}
