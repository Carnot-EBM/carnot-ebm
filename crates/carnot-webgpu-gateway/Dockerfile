FROM rust:1.94-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
RUN cargo build -p carnot-webgpu-gateway --bin gateway --release

FROM debian:bookworm-slim
COPY --from=builder /build/target/release/gateway /usr/local/bin/gateway
EXPOSE 3000
ENV HOST=0.0.0.0 PORT=3000
CMD ["gateway"]
