//! A blocking rate + concurrency limiter for Crossref (ARCHITECTURE.md §7).
//!
//! Requests are spaced at least `1/rate` apart (no bursts), so no one-second window ever
//! holds more than `rate` starts, and at most `concurrency` run at once. The clock is
//! injectable so tests can check the guarantees exactly.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub trait Clock: Send + Sync {
    fn now(&self) -> Duration;
    fn sleep(&self, d: Duration);
}

pub struct RealClock(Instant);

impl Default for RealClock {
    fn default() -> Self {
        RealClock(Instant::now())
    }
}

impl Clock for RealClock {
    fn now(&self) -> Duration {
        self.0.elapsed()
    }
    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub per_second: f64,
    pub concurrency: usize,
}

struct State {
    limits: Limits,
    in_flight: usize,
    next_start: Duration,
}

pub struct Limiter {
    state: Mutex<State>,
    freed: Condvar,
    clock: Arc<dyn Clock>,
}

/// Held while a request runs; dropping it frees a concurrency slot.
pub struct Permit<'a> {
    limiter: &'a Limiter,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut s = self.limiter.state.lock().expect("limiter");
        s.in_flight -= 1;
        self.limiter.freed.notify_one();
    }
}

impl Limiter {
    pub fn new(limits: Limits, clock: Arc<dyn Clock>) -> Self {
        Limiter { state: Mutex::new(State { limits, in_flight: 0, next_start: Duration::ZERO }), freed: Condvar::new(), clock }
    }

    pub fn limits(&self) -> Limits {
        self.state.lock().expect("limiter").limits
    }

    /// Replaces the limits (pool switch, or stricter limits announced by the server).
    pub fn set_limits(&self, limits: Limits) {
        self.state.lock().expect("limiter").limits = limits;
        self.freed.notify_all();
    }

    /// Only ever tightens: used for `X-Rate-Limit-*` / `X-Concurrency-Limit` headers.
    pub fn tighten(&self, per_second: Option<f64>, concurrency: Option<usize>) {
        let mut s = self.state.lock().expect("limiter");
        if let Some(r) = per_second.filter(|r| *r > 0.0 && *r < s.limits.per_second) {
            s.limits.per_second = r;
        }
        if let Some(c) = concurrency.filter(|c| *c > 0 && *c < s.limits.concurrency) {
            s.limits.concurrency = c;
        }
    }

    pub fn acquire(&self) -> Permit<'_> {
        let mut s = self.state.lock().expect("limiter");
        loop {
            if s.in_flight >= s.limits.concurrency {
                s = self.freed.wait(s).expect("limiter");
                continue;
            }
            let now = self.clock.now();
            if now >= s.next_start {
                // Rounded *up* to the nanosecond: rounding down would let rate+1 requests
                // into some one-second window (e.g. 3 × 333_333_333 ns < 1 s).
                let spacing = Duration::from_nanos((1e9 / s.limits.per_second.max(0.001)).ceil() as u64);
                s.next_start = now.max(s.next_start) + spacing;
                s.in_flight += 1;
                return Permit { limiter: self };
            }
            let wait = s.next_start - now;
            drop(s);
            self.clock.sleep(wait);
            s = self.state.lock().expect("limiter");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    /// Time only moves when someone sleeps.
    #[derive(Default)]
    struct SimClock(AtomicU64);

    impl Clock for SimClock {
        fn now(&self) -> Duration {
            Duration::from_nanos(self.0.load(Ordering::SeqCst))
        }
        fn sleep(&self, d: Duration) {
            self.0.fetch_add(d.as_nanos() as u64, Ordering::SeqCst);
        }
    }

    fn max_in_any_second(starts: &[Duration]) -> usize {
        starts.iter().map(|&s| starts.iter().filter(|&&t| t >= s && t < s + Duration::from_secs(1)).count()).max().unwrap_or(0)
    }

    #[test]
    fn never_exceeds_the_rate_in_any_one_second_window() {
        for (rate, n) in [(10.0, 60), (5.0, 40), (3.0, 25), (1.0, 8)] {
            let clock = Arc::new(SimClock::default());
            let l = Limiter::new(Limits { per_second: rate, concurrency: 3 }, clock.clone());
            let starts: Vec<Duration> = (0..n).map(|_| {
                let _p = l.acquire();
                clock.now()
            }).collect();
            assert!(max_in_any_second(&starts) <= rate as usize, "rate {rate}: {}", max_in_any_second(&starts));
            let span = starts.last().unwrap().as_secs_f64();
            assert!(span >= (n as f64 - 1.0) / rate - 1e-6, "rate {rate}: too fast ({span}s for {n})");
        }
    }

    #[test]
    fn never_exceeds_the_concurrency_limit() {
        let l = Arc::new(Limiter::new(Limits { per_second: 1000.0, concurrency: 3 }, Arc::new(RealClock::default())));
        let running = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let handles: Vec<_> = (0..12).map(|_| {
            let (l, running, peak) = (l.clone(), running.clone(), peak.clone());
            std::thread::spawn(move || {
                let _p = l.acquire();
                let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                running.fetch_sub(1, Ordering::SeqCst);
            })
        }).collect();
        handles.into_iter().for_each(|h| h.join().unwrap());
        assert_eq!(peak.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn server_headers_only_tighten() {
        let l = Limiter::new(Limits { per_second: 10.0, concurrency: 3 }, Arc::new(SimClock::default()));
        l.tighten(Some(50.0), Some(5));
        assert_eq!(l.limits(), Limits { per_second: 10.0, concurrency: 3 });
        l.tighten(Some(5.0), Some(1));
        assert_eq!(l.limits(), Limits { per_second: 5.0, concurrency: 1 });
    }
}
