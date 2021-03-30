/*!
A lightweight micro-benchmarking library which:

* uses linear regression to screen off constant error;
* handles benchmarks which mutate state;
* can measure simple polynomial scaling behavior
* is very easy to use!

Easybench is designed for benchmarks with a running time in the range `1 ns <
x < 1 ms` - results may be unreliable if benchmarks are very quick or very
slow. It's inspired by [criterion], but doesn't do as much sophisticated
analysis (no outlier detection, no HTML output).

[criterion]: https://hackage.haskell.org/package/criterion

```
use easybench::{bench,bench_env,bench_scaling};

# fn fib(_: usize) -> usize { 0 }
#
// Simple benchmarks are performed with `bench` or `bench_scaling`.
println!("fib 200: {}", bench(|| fib(200) ));
println!("fib 500: {}", bench(|| fib(500) ));
println!("fib scaling: {}", bench_scaling(|n| fib(n), 0));

// If a function needs to mutate some state, use `bench_env`.
println!("reverse: {}", bench_env(vec![0;100], |xs| xs.reverse() ));
println!("sort:    {}", bench_env(vec![0;100], |xs| xs.sort()    ));
```

Running the above yields the following results:

```none
fib 200:        50ns (R²=0.995, 20435 iterations in 68 samples)
fib 500:       144ns (R²=0.999, 7235 iterations in 57 samples)
fib scaling:   0.30ns/N    (R²=0.999, 8645 iterations in 59 samples)
reverse:        46ns (R²=0.990, 30550 iterations in 72 samples)
sort:          137ns (R²=0.991, 187129 iterations in 91 samples)
```

Easy! However, please read the [caveats](#caveats) below before using.

# Benchmarking algorithm

An *iteration* is a single execution of your code. A *sample* is a measurement,
during which your code may be run many times. In other words: taking a sample
means performing some number of iterations and measuring the total time.

The first sample we take performs only 1 iteration, but as we continue
taking samples we increase the number of iterations with increasing
rapidity. We
stop either when a global time limit is reached (currently 10 seconds),
or when we have collected sufficient statistics (but have run for at
least a millisecond).

If a benchmark requires some state to run, `n` copies of the initial state are
prepared before the sample is taken.

Once we have the data, we perform OLS linear regression to find out how
the sample time varies with the number of iterations in the sample. The
gradient of the regression line tells us how long it takes to perform a
single iteration of the benchmark. The R² value is a measure of how much
noise there is in the data.

If the function is too slow (5 or 10 seconds), the linear regression is skipped,
and a simple average of timings is used.  For slow functions, any overhead will
be negligible.

# Caveats

## Caveat 1: Harness overhead

**TL;DR: Compile with `--release`; the overhead is likely to be within the
**noise of your
benchmark.**

Any work which easybench does once-per-sample is ignored (this is the purpose of the linear
regression technique described above). However, work which is done once-per-iteration *will* be
counted in the final times.

* In the case of [`bench()`] this amounts to incrementing the loop counter and
  [copying the return value](#bonus-caveat-black-box).
* In the case of [`bench_env`] and [`bench_gen_env`], we also do a lookup into a big vector in
  order to get the environment for that iteration.
* If you compile your program unoptimised, there may be additional overhead.

The cost of the above operations depend on the details of your benchmark;
namely: (1) how large is the return value? and (2) does the benchmark evict
the environment vector from the CPU cache? In practice, these criteria are only
satisfied by longer-running benchmarks, making these effects hard to measure.

## Caveat 2: Pure functions

**TL;DR: Return enough information to prevent the optimiser from eliminating
code from your benchmark.**

Benchmarking pure functions involves a nasty gotcha which users should be
aware of. Consider the following benchmarks:

```
# use easybench::{bench,bench_env};
#
# fn fib(_: usize) -> usize { 0 }
#
let fib_1 = bench(|| fib(500) );                     // fine
let fib_2 = bench(|| { fib(500); } );                // spoiler: NOT fine
let fib_3 = bench_env(0, |x| { *x = fib(500); } );   // also fine, but ugly
# let _ = (fib_1, fib_2, fib_3);
```

The results are a little surprising:

```none
fib_1:        110 ns   (R²=1.000, 9131585 iterations in 144 samples)
fib_2:          0 ns   (R²=1.000, 413289203 iterations in 184 samples)
fib_3:        109 ns   (R²=1.000, 9131585 iterations in 144 samples)
```

Oh, `fib_2`, why do you lie? The answer is: `fib(500)` is pure, and its
return value is immediately thrown away, so the optimiser replaces the call
with a no-op (which clocks in at 0 ns).

What about the other two? `fib_1` looks very similar, with one exception:
the closure which we're benchmarking returns the result of the `fib(500)`
call. When it runs your code, easybench takes the return value and tricks the
optimiser into thinking it's going to use it for something, before throwing
it away. This is why `fib_1` is safe from having code accidentally eliminated.

In the case of `fib_3`, we actually *do* use the return value: each
iteration we take the result of `fib(500)` and store it in the iteration's
environment. This has the desired effect, but looks a bit weird.

## Bonus caveat: Black box

The function which easybench uses to trick the optimiser (`black_box`)
is stolen from [bencher], which [states]:

[bencher]: https://docs.rs/bencher/
[states]: https://docs.rs/bencher/0.1.2/bencher/fn.black_box.html

> NOTE: We don't have a proper black box in stable Rust. This is a workaround
> implementation, that may have a too big performance overhead, depending on
> operation, or it may fail to properly avoid having code optimized out. It
> is good enough that it is used by default.
*/

use std::f64;
use std::fmt::{self, Display, Formatter};
use std::time::*;

// We try to spend at most this many seconds (roughly) in total on
// each benchmark.
const BENCH_TIME_MAX: Duration = Duration::from_secs(10);
// We try to spend at least this many seconds in total on each
// benchmark.
const BENCH_TIME_MIN: Duration = Duration::from_millis(1);

/// Statistics for a benchmark run.
#[derive(Debug, PartialEq, Clone)]
pub struct Stats {
    /// The time, in nanoseconds, per iteration. If the benchmark generated
    /// fewer than 2 samples in the allotted time then this will be NaN.
    pub ns_per_iter: f64,
    /// The coefficient of determination, R².
    ///
    /// This is an indication of how noisy the benchmark was, where 1 is
    /// good and 0 is bad. Be suspicious of values below 0.9.
    pub goodness_of_fit: f64,
    /// How many times the benchmarked code was actually run.
    pub iterations: usize,
    /// How many samples were taken (ie. how many times we allocated the
    /// environment and measured the time).
    pub samples: usize,
}

impl Display for Stats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.ns_per_iter.is_nan() {
            write!(
                f,
                "Only generated {} sample(s) - we can't fit a regression line to that! \
                 Try making your benchmark faster.",
                self.samples
            )
        } else {
            let per_iter = Duration::from_nanos(self.ns_per_iter as u64);
            let per_iter = format!("{:?}", per_iter);
            write!(
                f,
                "{:>11} (R²={:.3}, {} iterations in {} samples)",
                per_iter, self.goodness_of_fit, self.iterations, self.samples
            )
        }
    }
}

/// Run a benchmark.
///
/// The return value of `f` is not used, but we trick the optimiser into
/// thinking we're going to use it. Make sure to return enough information
/// to prevent the optimiser from eliminating code from your benchmark! (See
/// the module docs for more.)
pub fn bench<F, O>(mut f: F) -> Stats
where
    F: FnMut() -> O,
{
    bench_env((), |_| f())
}

/// Run a benchmark with an environment.
///
/// The value `env` is a clonable prototype for the "benchmark
/// environment". Each iteration receives a freshly-cloned mutable copy of
/// this environment. The time taken to clone the environment is not included
/// in the results.
///
/// Nb: it's very possible that we will end up allocating many (>10,000)
/// copies of `env` at the same time. Probably best to keep it small.
///
/// See `bench` and the module docs for more.
///
/// ## Overhead
///
/// Every iteration, `bench_env` performs a lookup into a big vector in
/// order to get the environment for that iteration. If your benchmark
/// is memory-intensive then this could, in the worst case, amount to a
/// systematic cache-miss (ie. this vector would have to be fetched from
/// DRAM at the start of every iteration). In this case the results could be
/// affected by a hundred nanoseconds. This is a worst-case scenario however,
/// and I haven't actually been able to trigger it in practice... but it's
/// good to be aware of the possibility.
pub fn bench_env<F, I, O>(env: I, f: F) -> Stats
where
    F: FnMut(&mut I) -> O,
    I: Clone,
{
    bench_gen_env(move || env.clone(), f)
}

/// Run a benchmark with a generated environment.
///
/// The function `gen_env` creates the "benchmark environment" for the
/// computation. Each iteration receives a freshly-created environment. The
/// time taken to create the environment is not included in the results.
///
/// Nb: it's very possible that we will end up generating many (>10,000)
/// copies of `env` at the same time. Probably best to keep it small.
///
/// See `bench` and the module docs for more.
///
/// ## Overhead
///
/// Every iteration, `bench_gen_env` performs a lookup into a big vector
/// in order to get the environment for that iteration. If your benchmark
/// is memory-intensive then this could, in the worst case, amount to a
/// systematic cache-miss (ie. this vector would have to be fetched from
/// DRAM at the start of every iteration). In this case the results could be
/// affected by a hundred nanoseconds. This is a worst-case scenario however,
/// and I haven't actually been able to trigger it in practice... but it's
/// good to be aware of the possibility.
pub fn bench_gen_env<G, F, I, O>(mut gen_env: G, mut f: F) -> Stats
where
    G: FnMut() -> I,
    F: FnMut(&mut I) -> O,
{
    let mut data = Vec::new();
    // The time we started the benchmark (not used in results)
    let bench_start = Instant::now();

    // Collect data until BENCH_TIME_MAX is reached.
    for iters in slow_fib(BENCH_SCALE_TIME) {
        // Prepare the environments - one per iteration
        let mut xs = std::iter::repeat_with(&mut gen_env)
            .take(iters)
            .collect::<Vec<I>>();
        // Start the clock
        let iter_start = Instant::now();
        // We iterate over `&mut xs` rather than draining it, because we
        // don't want to drop the env values until after the clock has stopped.
        for x in &mut xs {
            // Run the code and pretend to use the output
            pretend_to_use(f(x));
        }
        let time = iter_start.elapsed();
        data.push((iters, time));

        let elapsed = bench_start.elapsed();
        if elapsed > BENCH_TIME_MIN && data.len() > 3 {
            // If the first iter in a sample is consistently slow, that's fine -
            // that's why we do the linear regression. If the first sample is slower
            // than the rest, however, that's not fine.  Therefore, we discard the
            // first sample as a cache-warming exercise.

            // Compute some stats
            let (grad, r2) = regression(&data[1..]);
            let stats = Stats {
                ns_per_iter: grad,
                goodness_of_fit: r2,
                iterations: data[1..].iter().map(|&(x, _)| x).sum(),
                samples: data[1..].len(),
            };
            if elapsed > BENCH_TIME_MAX || r2 > 0.99 {
                return stats;
            }
        } else if elapsed > BENCH_TIME_MAX {
            let total_time: f64 = data.iter().map(|(_, t)| t.as_nanos() as f64).sum();
            let iterations = data.iter().map(|&(x, _)| x).sum();
            return Stats {
                ns_per_iter: total_time / iterations as f64,
                iterations,
                goodness_of_fit: 0.0,
                samples: data.len(),
            };
        }
    }
    unreachable!()
}

/// Statistics for a benchmark run determining the scaling of a function.
#[derive(Debug, PartialEq, Clone)]
pub struct ScalingStats {
    pub scaling: Scaling,
    pub goodness_of_fit: f64,
    /// How many times the benchmarked code was actually run.
    pub iterations: usize,
    /// How many samples were taken (ie. how many times we allocated the
    /// environment and measured the time).
    pub samples: usize,
}
/// The timing and scaling results (without statistics) for a benchmark.
#[derive(Debug, PartialEq, Clone)]
pub struct Scaling {
    /// The scaling power
    /// If this is 2, for instance, you have an O(N²) algorithm.
    pub power: usize,
    /// An exponetial behavior, i.e. 2ᴺ
    pub exponential: usize,
    /// The time, in nanoseconds, per scaled size of the problem. If
    /// the problem scales as O(N²) for instance, this is the number
    /// of nanoseconds per N².
    pub ns_per_scale: f64,
}

impl Display for ScalingStats {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "{} (R²={:.3}, {} iterations in {} samples)",
            self.scaling, self.goodness_of_fit, self.iterations, self.samples
        )
    }
}
impl Display for Scaling {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let per_iter = Duration::from_nanos(self.ns_per_scale as u64);
        let per_iter = if self.ns_per_scale < 1.0 {
            format!("{:.2}ns", self.ns_per_scale)
        } else if self.ns_per_scale < 10.0 {
            format!("{:.1}ns", self.ns_per_scale)
        } else {
            format!("{:?}", per_iter)
        };
        if self.exponential == 1 {
            match self.power {
                0 => write!(f, "{:>8}/iter", per_iter),
                1 => write!(f, "{:>8}/N   ", per_iter),
                2 => write!(f, "{:>8}/N²  ", per_iter),
                3 => write!(f, "{:>8}/N³  ", per_iter),
                4 => write!(f, "{:>8}/N⁴  ", per_iter),
                5 => write!(f, "{:>8}/N⁵  ", per_iter),
                6 => write!(f, "{:>8}/N⁶  ", per_iter),
                7 => write!(f, "{:>8}/N⁷  ", per_iter),
                8 => write!(f, "{:>8}/N⁸  ", per_iter),
                9 => write!(f, "{:>8}/N⁹  ", per_iter),
                _ => write!(f, "{:>8}/N^{}", per_iter, self.power),
            }
        } else {
            match self.power {
                0 => write!(f, "{:>8}/{}ᴺ", per_iter, self.exponential),
                1 => write!(f, "{:>8}/(N{}ᴺ)   ", per_iter, self.exponential),
                2 => write!(f, "{:>8}/(N²{}ᴺ)  ", per_iter, self.exponential),
                3 => write!(f, "{:>8}/(N³{}ᴺ)  ", per_iter, self.exponential),
                4 => write!(f, "{:>8}/(N⁴{}ᴺ)  ", per_iter, self.exponential),
                5 => write!(f, "{:>8}/(N⁵{}ᴺ)  ", per_iter, self.exponential),
                6 => write!(f, "{:>8}/(N⁶{}ᴺ)  ", per_iter, self.exponential),
                7 => write!(f, "{:>8}/(N⁷{}ᴺ)  ", per_iter, self.exponential),
                8 => write!(f, "{:>8}/(N⁸{}ᴺ)  ", per_iter, self.exponential),
                9 => write!(f, "{:>8}/(N⁹{}ᴺ)  ", per_iter, self.exponential),
                _ => write!(f, "{:>8}/(N^{}{}ᴺ)", per_iter, self.power, self.exponential),
            }
        }
    }
}

/// Benchmark the power-law scaling of the function
///
/// This function assumes that the function scales as 𝑶(𝑁ᴾ𝐸ᴺ).
/// It conisders higher powers for faster functions, and tries to
/// keep the measuring time around 10s.  It measures the power ᴾ and exponential base 𝐸
/// based on n R² goodness of fit parameter.
pub fn bench_scaling<F, O>(f: F, nmin: usize) -> ScalingStats
where
    F: Fn(usize) -> O,
{
    let mut data = Vec::new();
    // The time we started the benchmark (not used in results)
    let bench_start = Instant::now();

    // Collect data until BENCH_TIME_MAX is reached.
    for iters in slow_fib(BENCH_SCALE_TIME) {
        // Prepare the environments - nmin per iteration
        let n = if nmin > 0 { iters * nmin } else { iters };
        // Generate a Vec holding n's to hopefully keep the optimizer
        // from lifting the function out of the loop, as it could if
        // we had `f(n)` in there, and `f` were inlined or `const`.
        let xs = vec![n; iters];
        // Start the clock
        let iter_start = Instant::now();
        for x in xs.into_iter() {
            // Run the code and pretend to use the output
            pretend_to_use(f(x));
        }
        let time = iter_start.elapsed();
        data.push((n, iters, time));

        let elapsed = bench_start.elapsed();
        if elapsed > BENCH_TIME_MIN {
            let stats = compute_scaling_gen(&data);
            if elapsed > BENCH_TIME_MAX || stats.goodness_of_fit > 0.99 {
                return stats;
            }
        }
    }
    unreachable!()
}

/// Benchmark the power-law scaling of the function with generated input
///
/// This function is like [`bench_scaling`], but uses a generating function
/// to construct the input to your benchmarked function.
///
/// This function assumes that the function scales as 𝑶(𝑁ᴾ𝐸ᴺ).
/// It conisders higher powers for faster functions, and tries to
/// keep the measuring time around 10s.  It measures the power ᴾ and exponential base 𝐸
/// based on n R² goodness of fit parameter.
///
/// # Example
/// ```
/// use easybench::bench_scaling_gen;
///
/// let summation = bench_scaling_gen(|n| vec![3.0; n], |v| v.iter().cloned().sum::<f64>(),0);
/// println!("summation: {}", summation);
/// assert_eq!(1, summation.scaling.power); // summation must run in linear time.
/// ```
/// which gives output
/// ```none
/// summation:     43ns/N    (R²=0.996, 445 iterations in 29 samples)
/// ```
pub fn bench_scaling_gen<G, F, I, O>(mut gen_env: G, f: F, nmin: usize) -> ScalingStats
where
    G: FnMut(usize) -> I,
    F: Fn(&mut I) -> O,
{
    let mut data = Vec::new();
    // The time we started the benchmark (not used in results)
    let bench_start = Instant::now();

    // Collect data until BENCH_TIME_MAX is reached.
    for iters in slow_fib(BENCH_SCALE_TIME) {
        // Prepare the environments - nmin per iteration
        let n = if nmin > 0 { iters * nmin } else { iters };
        let mut xs = std::iter::repeat_with(|| gen_env(n))
            .take(iters)
            .collect::<Vec<I>>();
        // Start the clock
        let iter_start = Instant::now();
        // We iterate over `&mut xs` rather than draining it, because we
        // don't want to drop the env values until after the clock has stopped.
        for x in &mut xs {
            // Run the code and pretend to use the output
            pretend_to_use(f(x));
        }
        let time = iter_start.elapsed();
        data.push((n, iters, time));

        let elapsed = bench_start.elapsed();
        if elapsed > BENCH_TIME_MIN {
            let stats = compute_scaling_gen(&data);
            if elapsed > BENCH_TIME_MAX || stats.goodness_of_fit > 0.99 {
                return stats;
            }
        }
    }
    println!("how did I get here?!");
    unreachable!()
}

/// This function assumes that the function scales as 𝑶(𝑁ᴾ𝐸ᴺ).  It measures the scaling
/// based on n R² goodness of fit parameter, and returns the best fit.
/// If it believes itself clueless, the goodness_of_fit is set to zero.
fn compute_scaling_gen(data: &[(usize, usize, Duration)]) -> ScalingStats {
    let num_n = {
        let mut ns = data.iter().map(|(n, _, _)| *n).collect::<Vec<_>>();
        ns.dedup();
        ns.len()
    };

    // If the first iter in a sample is consistently slow, that's fine -
    // that's why we do the linear regression. If the first sample is slower
    // than the rest, however, that's not fine.  Therefore, we discard the
    // first sample as a cache-warming exercise.

    // Compute some stats for each of several different
    // powers, to see which seems most accurate.
    let mut stats = Vec::new();
    let mut best = 0;
    let mut second_best = 0;
    for i in 1..num_n / 2 + 2 {
        for power in 0..i {
            let exponential = i - power;
            let pdata: Vec<_> = data[1..]
                .iter()
                .map(|&(n, i, t)| {
                    (
                        (exponential as f64).powi(n as i32)
                            * (n as f64).powi(power as i32)
                            * (i as f64),
                        t,
                    )
                })
                .collect();
            let (grad, r2) = fregression(&pdata);
            stats.push(ScalingStats {
                scaling: Scaling {
                    power,
                    exponential,
                    ns_per_scale: grad,
                },
                goodness_of_fit: r2,
                iterations: data[1..].iter().map(|&(x, _, _)| x).sum(),
                samples: data[1..].len(),
            });
            if r2 > stats[best].goodness_of_fit {
                second_best = best;
                best = stats.len() - 1;
            }
        }
    }

    if num_n < 10 || stats[second_best].goodness_of_fit == stats[best].goodness_of_fit {
        stats[best].goodness_of_fit = 0.0;
    } else {
        println!("finished...");
        for s in stats.iter() {
            println!("  {}", s);
        }
        // for d in data[data.len()-4..].iter() {
        //     println!("    {}, {} -> {} ns", d.0, d.1, d.2.as_nanos());
        // }
        println!("best is {}", stats[best]);
    }
    stats[best].clone()
}

/// Compute the OLS linear regression line for the given data set, returning
/// the line's gradient and R². Requires at least 2 samples.
//
// Overflows:
//
// * sum(x * x): num_samples <= 0.5 * log_k (1 + 2 ^ 64 (FACTOR - 1))
fn regression(data: &[(usize, Duration)]) -> (f64, f64) {
    if data.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    // Do all the arithmetic using f64, because it can happen that the
    // squared numbers to overflow using integer arithmetic if the
    // tests are too fast (so we run too many iterations).
    let data: Vec<_> = data
        .iter()
        .map(|&(x, y)| (x as f64, y.as_nanos() as f64))
        .collect();
    let n = data.len() as f64;
    let nxbar = data.iter().map(|&(x, _)| x).sum::<f64>(); // iter_time > 5e-11 ns
    let nybar = data.iter().map(|&(_, y)| y).sum::<f64>(); // TIME_LIMIT < 2 ^ 64 ns
    let nxxbar = data.iter().map(|&(x, _)| x * x).sum::<f64>(); // num_iters < 13_000_000_000
    let nyybar = data.iter().map(|&(_, y)| y * y).sum::<f64>(); // TIME_LIMIT < 4.3 e9 ns
    let nxybar = data.iter().map(|&(x, y)| x * y).sum::<f64>();
    let ncovar = nxybar - ((nxbar * nybar) / n);
    let nxvar = nxxbar - ((nxbar * nxbar) / n);
    let nyvar = nyybar - ((nybar * nybar) / n);
    let gradient = ncovar / nxvar;
    let r2 = (ncovar * ncovar) / (nxvar * nyvar);
    assert!(r2.is_nan() || r2 <= 1.0);
    (gradient, r2)
}

/// Compute the OLS linear regression line for the given data set, returning
/// the line's gradient and R². Requires at least 2 samples.
//
// Overflows:
//
// * sum(x * x): num_samples <= 0.5 * log_k (1 + 2 ^ 64 (FACTOR - 1))
fn fregression(data: &[(f64, Duration)]) -> (f64, f64) {
    if data.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    // Do all the arithmetic using f64, because it can happen that the
    // squared numbers to overflow using integer arithmetic if the
    // tests are too fast (so we run too many iterations).
    let data: Vec<_> = data
        .iter()
        .map(|&(x, y)| (x as f64, y.as_nanos() as f64))
        .collect();
    let n = data.len() as f64;
    let nxbar = data.iter().map(|&(x, _)| x).sum::<f64>(); // iter_time > 5e-11 ns
    let nybar = data.iter().map(|&(_, y)| y).sum::<f64>(); // TIME_LIMIT < 2 ^ 64 ns
    let nxxbar = data.iter().map(|&(x, _)| x * x).sum::<f64>(); // num_iters < 13_000_000_000
    let nyybar = data.iter().map(|&(_, y)| y * y).sum::<f64>(); // TIME_LIMIT < 4.3 e9 ns
    let nxybar = data.iter().map(|&(x, y)| x * y).sum::<f64>();
    let ncovar = nxybar - ((nxbar * nybar) / n);
    let nxvar = nxxbar - ((nxbar * nxbar) / n);
    let nyvar = nyybar - ((nybar * nybar) / n);
    let gradient = ncovar / nxvar;
    let r2 = (ncovar * ncovar) / (nxvar * nyvar);
    assert!(r2.is_nan() || r2 <= 1.0);
    (gradient, r2)
}

// Stolen from `bencher`, where it's known as `black_box`.
//
// NOTE: We don't have a proper black box in stable Rust. This is a workaround
// implementation, that may have a too big performance overhead, depending
// on operation, or it may fail to properly avoid having code optimized
// out. It is good enough that it is used by default.
//
// A function that is opaque to the optimizer, to allow benchmarks to pretend
// to use outputs to assist in avoiding dead-code elimination.
fn pretend_to_use<T>(dummy: T) -> T {
    unsafe {
        let ret = ::std::ptr::read_volatile(&dummy);
        ::std::mem::forget(dummy);
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    fn fib(n: usize) -> usize {
        let mut i = 0;
        let mut sum = 0;
        let mut last = 0;
        let mut curr = 1usize;
        while i < n - 1 {
            sum = curr.wrapping_add(last);
            last = curr;
            curr = sum;
            i += 1;
        }
        sum
    }

    // This is only here because doctests don't work with `--nocapture`.
    #[test]
    #[ignore]
    fn doctests_again() {
        println!();
        println!("fib 200: {}", bench(|| fib(200)));
        println!("fib 500: {}", bench(|| fib(500)));
        println!("fib scaling: {}", bench_scaling(|n| fib(n), 0));
        println!("reverse: {}", bench_env(vec![0; 100], |xs| xs.reverse()));
        println!("sort:    {}", bench_env(vec![0; 100], |xs| xs.sort()));

        // This is fine:
        println!("fib 1:   {}", bench(|| fib(500)));
        // This is NOT fine:
        println!(
            "fib 2:   {}",
            bench(|| {
                fib(500);
            })
        );
        // This is also fine, but a bit weird:
        println!(
            "fib 3:   {}",
            bench_env(0, |x| {
                *x = fib(500);
            })
        );
    }

    #[test]
    fn scales_o_one() {
        println!();
        let stats = bench_scaling(|_| thread::sleep(Duration::from_millis(10)), 1);
        println!("O(N): {}", stats);
        assert_eq!(stats.scaling.power, 0);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1e7);
        assert!((stats.scaling.ns_per_scale - 1e7).abs() < 1e6);
        assert!(format!("{}", stats).contains("samples"));
    }

    #[test]
    fn scales_o_n() {
        println!();
        let stats = bench_scaling(|n| thread::sleep(Duration::from_millis(10 * n as u64)), 1);
        println!("O(N): {}", stats);
        assert_eq!(stats.scaling.power, 1);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1e7);
        assert!((stats.scaling.ns_per_scale - 1e7).abs() < 1e5);

        println!("Summing integers");
        let stats = bench_scaling_gen(
            |n| (0..n as u64).collect::<Vec<_>>(),
            |v| v.iter().cloned().sum::<u64>(),
            1,
        );
        println!("O(N): {}", stats);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1e7);
        assert_eq!(stats.scaling.power, 1);
    }

    #[test]
    fn scales_o_n_log_n_looks_like_n() {
        println!("Sorting integers");
        let stats = bench_scaling_gen(
            |n| (0..n as u64).map(|i| (i*13 + 5) % 137).collect::<Vec<_>>(),
            |v| v.sort(),
            1,
        );
        println!("O(N log N): {}", stats);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1e7);
        assert_eq!(stats.scaling.power, 1);
    }

    #[test]
    fn scales_o_2_to_the_n() {
        println!();
        let stats = bench_scaling(|n| thread::sleep(Duration::from_nanos((1 << n) as u64)), 1);
        println!("O(2ᴺ): {}", stats);
        assert_eq!(stats.scaling.power, 0);
        assert_eq!(stats.scaling.exponential, 2);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1.0);
        assert!((stats.scaling.ns_per_scale - 1.0).abs() < 0.2);
    }

    #[test]
    fn scales_o_n_square() {
        println!();
        let stats = bench_scaling(
            |n| thread::sleep(Duration::from_millis(10 * (n * n) as u64)),
            1,
        );
        println!("O(N): {}", stats);
        assert_eq!(stats.scaling.power, 2);
        println!("   error: {:e}", stats.scaling.ns_per_scale - 1e7);
        assert!((stats.scaling.ns_per_scale - 1e7).abs() < 1e5);
    }

    #[test]
    fn very_quick() {
        println!();
        println!("very quick: {}", bench(|| {}));
    }

    #[test]
    fn very_slow() {
        println!();
        let stats = bench(|| thread::sleep(Duration::from_millis(400)));
        println!("very slow: {}", stats);
        assert!(stats.ns_per_iter > 399.0e6);
        assert_eq!(3, stats.samples);
    }

    #[test]
    fn painfully_slow() {
        println!();
        let stats = bench(|| thread::sleep(Duration::from_secs(11)));
        println!("painfully slow: {}", stats);
        println!("ns {}", stats.ns_per_iter);
        assert!(stats.ns_per_iter > 11.0e9);
        assert_eq!(1, stats.iterations);
    }

    #[test]
    fn sadly_slow() {
        println!();
        let stats = bench(|| thread::sleep(Duration::from_secs(6)));
        println!("sadly slow: {}", stats);
        println!("ns {}", stats.ns_per_iter);
        assert!(stats.ns_per_iter > 6.0e9);
        assert_eq!(2, stats.iterations);
    }

    #[test]
    fn test_sleep() {
        println!();
        println!(
            "sleep 1 ms: {}",
            bench(|| thread::sleep(Duration::from_millis(1)))
        );
    }

    #[test]
    fn noop() {
        println!();
        println!("noop base: {}", bench(|| {}));
        println!("noop 0:    {}", bench_env(vec![0u64; 0], |_| {}));
        println!("noop 16:   {}", bench_env(vec![0u64; 16], |_| {}));
        println!("noop 64:   {}", bench_env(vec![0u64; 64], |_| {}));
        println!("noop 256:  {}", bench_env(vec![0u64; 256], |_| {}));
        println!("noop 512:  {}", bench_env(vec![0u64; 512], |_| {}));
    }

    #[test]
    fn ret_value() {
        println!();
        println!(
            "no ret 32:    {}",
            bench_env(vec![0u64; 32], |x| { x.clone() })
        );
        println!("return 32:    {}", bench_env(vec![0u64; 32], |x| x.clone()));
        println!(
            "no ret 256:   {}",
            bench_env(vec![0u64; 256], |x| { x.clone() })
        );
        println!(
            "return 256:   {}",
            bench_env(vec![0u64; 256], |x| x.clone())
        );
        println!(
            "no ret 1024:  {}",
            bench_env(vec![0u64; 1024], |x| { x.clone() })
        );
        println!(
            "return 1024:  {}",
            bench_env(vec![0u64; 1024], |x| x.clone())
        );
        println!(
            "no ret 4096:  {}",
            bench_env(vec![0u64; 4096], |x| { x.clone() })
        );
        println!(
            "return 4096:  {}",
            bench_env(vec![0u64; 4096], |x| x.clone())
        );
        println!(
            "no ret 50000: {}",
            bench_env(vec![0u64; 50000], |x| { x.clone() })
        );
        println!(
            "return 50000: {}",
            bench_env(vec![0u64; 50000], |x| x.clone())
        );
    }
}

// Each time we take a sample we increase the number of iterations
// using a slow version of the Fibonacci sequence, which
// asymptotically grows exponentially, but also gives us a different
// value each time (except for repeating 1 twice, once for warmup).

// For our standard `bench_*` we use slow_fib(25), which was chosen to
// asymptotically match the prior behavior of the library, which grew
// by an exponential of 1.1.
const BENCH_SCALE_TIME: usize = 25;

fn slow_fib(scale_time: usize) -> impl Iterator<Item = usize> {
    #[derive(Debug)]
    struct SlowFib {
        which: usize,
        buffer: Vec<usize>,
    }
    impl Iterator for SlowFib {
        type Item = usize;
        fn next(&mut self) -> Option<usize> {
            // println!("!!! {:?}", self);
            let oldwhich = self.which;
            self.which = (self.which + 1) % self.buffer.len();
            self.buffer[self.which] = self.buffer[oldwhich] + self.buffer[self.which];
            Some(self.buffer[self.which])
        }
    }
    assert!(scale_time > 3);
    let mut buffer = vec![1; scale_time];
    // buffer needs just the two zeros to make it start with two 1
    // values.  The rest should be 1s.
    buffer[1] = 0;
    buffer[2] = 0;
    SlowFib { which: 0, buffer }
}

#[test]
fn test_fib() {
    // The following code was used to demonstrate that asymptotically
    // the SlowFib grows as the 1.1 power, just as the old code.  It
    // differs in that it increases linearly at the beginning, which
    // leads to larger numbers earlier in the sequence.  It also
    // differs in that it does not repeat any numbers in the sequence,
    // which hopefully leads to better linear regression, particularly
    // if we can only run a few iterations.
    let mut prev = 1;
    for x in slow_fib(25).take(200) {
        let rat = x as f64 / prev as f64;
        println!("ratio: {}/{} = {}", prev, x, rat);
        prev = x;
    }
    let five: Vec<_> = slow_fib(25).take(5).collect();
    assert_eq!(&five, &[1, 1, 2, 3, 4]);
    let more: Vec<_> = slow_fib(25).take(32).collect();
    assert_eq!(
        &more,
        &[
            1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 28, 31, 35, 40, 46,
        ]
    );
    let previous_sequence: Vec<_> = (0..32).map(|n| (1.1f64).powi(n).round() as usize).collect();
    assert_eq!(
        &previous_sequence,
        &[
            1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13,
            14, 16, 17, 19,
        ]
    );
    let previous_sequence: Vec<_> = (20..40)
        .map(|n| (1.1f64).powi(n).round() as usize)
        .collect();
    assert_eq!(
        &previous_sequence,
        &[7, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 26, 28, 31, 34, 37, 41,]
    );
}
