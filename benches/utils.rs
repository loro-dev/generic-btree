use std::fs::File;

use pprof::flamegraph::{Direction, Options};

pub struct PProfGuard {
    path: String,
    guard: pprof::ProfilerGuard<'static>,
}

impl PProfGuard {
    #[must_use]
    pub fn new(name: &str) -> Self {
        let guard = pprof::ProfilerGuard::new(10_000).unwrap();
        Self {
            path: name.to_string(),
            guard,
        }
    }
}

impl Drop for PProfGuard {
    fn drop(&mut self) {
        if let Ok(report) = self.guard.report().build() {
            let file = File::create(self.path.as_str()).unwrap();
            let mut options = Options::default();
            options.direction = Direction::Inverted;
            options.flame_chart = false;
            report.flamegraph_with_options(file, &mut options).unwrap();
        };
    }
}
