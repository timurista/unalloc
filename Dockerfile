# Small runtime image. unalloc is a short-lived CLI: it queries systems you
# already run, prints a number, and exits. Nothing is persisted, so no volumes
# and no state directory.
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir .

# Runs as non-root; there is nothing to write.
RUN useradd --create-home --uid 10001 unalloc
USER unalloc

ENTRYPOINT ["unalloc"]
CMD ["report", "--fixtures", "--dimension", "team"]
