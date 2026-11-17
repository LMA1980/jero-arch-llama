# jero-arch-llama

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/jero-arch-llama/card.svg" alt="jero-arch-llama Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-llama/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-llama/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-llama/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/jero-arch-llama/benchmark.svg" alt="Benchmark"/></a>
</p>

______________________________________________________________________

Meta Llama and Alibaba Qwen2 architecture recipe (supporting `llama`, `llama4` interleaved RoPE, and `qwen2` NeoX RoPE) for the SpacemiT K3 platform.

## Ecosystem & Documentation

- **Public Documentation Portal**: [`lma1980.github.io/model-runner`](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner) ([Reference Manual](https://github.com/LMA1980/lma1980.github.io/blob/main/model-runner/manual.md), [Crate Docs](https://github.com/LMA1980/lma1980.github.io/tree/main/model-runner/docs/crates))
- **Upstream Dependencies**: [`jero-core`](https://github.com/LMA1980/jero-core) and [`jero-kernels-rvv`](https://github.com/LMA1980/jero-kernels-rvv) (via the optional `with-kernels` feature)
- **Downstream Runtime**: [`jero-engine`](https://github.com/LMA1980/jero-engine)

## License

Licensed under the GNU Lesser General Public License v3.0 with the LGPL-3.0 Linking Exception (`LGPL-3.0-only WITH LGPL-3.0-linking-exception`). See [LICENSE](LICENSE) for details.
