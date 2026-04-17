# Load-test samples

Drop representative documents here. They're volume-mounted into the CPU
container at `/app/samples` and referenced as `file:///app/samples/<name>`.

Recommended small starter set:

- `receipt.png` — ~1 megapixel retail receipt
- `table.png` — financial table with borders + numbers
- `invoice.pdf` — 2-3 page multi-region invoice

Keep these small (≤ ~2 MB each) so the load tests are network-bound on the
model, not on file I/O. For stress tests, add larger PDFs and adjust the
`LOCUST_IMAGES` / `--image-url` overrides.

The samples are **not** committed — add your own before running load tests.
