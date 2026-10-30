//! Live-resizable concurrency pools for delegated agents.
//!
//! One global pool covers every delegated run.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::Notify;

use {crate::config::MAX_AGENT_CONCURRENCY, rho_tools::cancellation::RunCancellation};

use super::agent_binding::CapacityClass;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ConcurrencyLimits {
    pub(crate) total: usize,
}

/// Shared, live-resizable capacity for delegated runs.
#[derive(Clone)]
pub(crate) struct AgentConcurrency {
    total: Arc<AdjustablePool>,
}

pub(crate) struct RuntimePermits {
    _total: PoolPermit,
}

/// Packed as `(limit << 32) | active` so resize and acquire CAS the same word.
/// A lower cap that lands after a snapshot is loaded fails that CAS; the retry
/// observes the new limit before it increments active work.
const ACTIVE_BITS: u64 = 32;
const ACTIVE_MASK: u64 = (1 << ACTIVE_BITS) - 1;

fn pack(limit: usize, active: usize) -> u64 {
    debug_assert!(u32::try_from(limit).is_ok(), "limit fits in 32 bits");
    debug_assert!(u32::try_from(active).is_ok(), "active fits in 32 bits");
    ((limit as u64) << ACTIVE_BITS) | (active as u64)
}

fn unpack(state: u64) -> (usize, usize) {
    (
        (state >> ACTIVE_BITS) as usize,
        (state & ACTIVE_MASK) as usize,
    )
}

struct AdjustablePool {
    state: AtomicU64,
    notify: Notify,
}

struct PoolPermit {
    pool: Arc<AdjustablePool>,
}

impl Drop for PoolPermit {
    fn drop(&mut self) {
        // Active occupies the low bits, so a saturating decrement cannot
        // disturb `limit` while in-flight work still holds a permit.
        self.pool.state.fetch_sub(1, Ordering::Release);
        self.pool.notify.notify_waiters();
    }
}

impl AdjustablePool {
    fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(pack(limit, 0)),
            notify: Notify::new(),
        })
    }

    fn limit(&self) -> usize {
        unpack(self.state.load(Ordering::Acquire)).0
    }

    #[cfg(test)]
    fn active(&self) -> usize {
        unpack(self.state.load(Ordering::Acquire)).1
    }

    #[cfg(test)]
    fn available(&self) -> usize {
        self.limit().saturating_sub(self.active())
    }

    fn set_limit(&self, limit: usize) {
        loop {
            let state = self.state.load(Ordering::Acquire);
            let active = unpack(state).1;
            if self
                .state
                .compare_exchange(
                    state,
                    pack(limit, active),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                self.notify.notify_waiters();
                return;
            }
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<PoolPermit> {
        self.try_acquire_after_observe(|_| {})
    }

    fn try_acquire_after_observe(
        self: &Arc<Self>,
        mut after_observe: impl FnMut(&Arc<Self>),
    ) -> Option<PoolPermit> {
        loop {
            let state = self.state.load(Ordering::Acquire);
            after_observe(self);
            let (limit, active) = unpack(state);
            if active >= limit {
                return None;
            }
            if self
                .state
                .compare_exchange(
                    state,
                    pack(limit, active + 1),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Some(PoolPermit {
                    pool: Arc::clone(self),
                });
            }
        }
    }

    async fn acquire(self: &Arc<Self>, cancellation: &RunCancellation) -> Option<PoolPermit> {
        loop {
            if cancellation.is_cancelled() {
                return None;
            }
            if let Some(permit) = self.take_unless_cancelled(cancellation) {
                return Some(permit);
            }
            let notified = self.notify.notified();
            tokio::pin!(notified);
            if let Some(permit) = self.take_unless_cancelled(cancellation) {
                return Some(permit);
            }
            tokio::select! {
                biased;
                () = cancellation.cancelled() => return None,
                () = notified => {}
            }
        }
    }

    fn take_unless_cancelled(
        self: &Arc<Self>,
        cancellation: &RunCancellation,
    ) -> Option<PoolPermit> {
        let permit = self.try_acquire()?;
        if cancellation.is_cancelled() {
            None
        } else {
            Some(permit)
        }
    }
}

impl AgentConcurrency {
    pub(crate) fn new(limits: ConcurrencyLimits) -> Self {
        Self {
            total: AdjustablePool::new(limits.total),
        }
    }

    pub(crate) fn from_config(configured_total: usize) -> Self {
        Self::new(concurrency_limits(configured_total))
    }

    /// Replace the global cap. In-flight runs keep their slots; a lower cap
    /// only delays new acquires until `active` falls under the new limit.
    pub(crate) fn set_total(&self, total: usize) {
        self.total.set_limit(total.clamp(1, MAX_AGENT_CONCURRENCY));
    }

    pub(crate) fn total_limit(&self) -> usize {
        self.total.limit()
    }

    #[cfg(test)]
    pub(crate) fn available_total(&self) -> usize {
        self.total.available()
    }

    /// Acquire concurrency for a delegated run: one global permit.
    /// Cancellation returns `Ok(None)` so capacity cannot leak.
    pub(crate) async fn acquire(
        &self,
        _capacity_class: CapacityClass,
        cancellation: &RunCancellation,
    ) -> Option<RuntimePermits> {
        let total = self.total.acquire(cancellation).await?;
        Some(RuntimePermits { _total: total })
    }
}

/// Resolve the startup cap from `behavior.agent_concurrency`.
pub(crate) fn concurrency_limits(configured_total: usize) -> ConcurrencyLimits {
    ConcurrencyLimits {
        total: clamp_concurrency(configured_total),
    }
}

fn clamp_concurrency(value: usize) -> usize {
    value.clamp(1, MAX_AGENT_CONCURRENCY)
}
