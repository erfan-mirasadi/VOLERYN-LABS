const MAX_PASSES = 4;
const WINDOW = 30;

// Measures each labelled pass on the GPU itself with timestamp queries; averages the last WINDOW frames.
export class GpuTimer {
  private querySet: GPUQuerySet | null = null;
  private resolveBuffer: GPUBuffer | null = null;
  private readBuffers: GPUBuffer[] = [];
  private labels: string[] = [];
  private samples = new Map<string, number[]>();

  constructor(private device: GPUDevice, enabled: boolean) {
    if (!enabled) return;
    this.querySet = device.createQuerySet({ type: 'timestamp', count: MAX_PASSES * 2 });
    this.resolveBuffer = device.createBuffer({
      size: MAX_PASSES * 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
  }

  get enabled() {
    return this.querySet !== null;
  }

  timestamps(label: string): GPUComputePassTimestampWrites | undefined {
    if (!this.querySet || this.labels.length === MAX_PASSES) return undefined;
    const index = this.labels.push(label) - 1;
    return { querySet: this.querySet, beginningOfPassWriteIndex: index * 2, endOfPassWriteIndex: index * 2 + 1 };
  }

  // Encodes the readback for this frame's passes; call the returned function after the queue submit.
  resolve(encoder: GPUCommandEncoder): () => void {
    const labels = this.labels;
    this.labels = [];
    if (!this.querySet || !this.resolveBuffer || labels.length === 0) return () => {};
    const bytes = labels.length * 16;
    const readBuffer =
      this.readBuffers.pop() ??
      this.device.createBuffer({ size: MAX_PASSES * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.resolveQuerySet(this.querySet, 0, labels.length * 2, this.resolveBuffer, 0);
    encoder.copyBufferToBuffer(this.resolveBuffer, 0, readBuffer, 0, bytes);
    return () => {
      readBuffer.mapAsync(GPUMapMode.READ, 0, bytes).then(() => {
        const ticks = new BigInt64Array(readBuffer.getMappedRange(0, bytes));
        labels.forEach((label, i) => this.record(label, Number(ticks[i * 2 + 1] - ticks[i * 2]) / 1e6));
        readBuffer.unmap();
        this.readBuffers.push(readBuffer);
      });
    };
  }

  averages(): Record<string, number> {
    const result: Record<string, number> = {};
    for (const [label, values] of this.samples) {
      result[label] = values.reduce((sum, value) => sum + value, 0) / values.length;
    }
    return result;
  }

  dispose() {
    this.querySet?.destroy();
    this.resolveBuffer?.destroy();
  }

  private record(label: string, ms: number) {
    const values = this.samples.get(label) ?? [];
    values.push(ms);
    if (values.length > WINDOW) values.shift();
    this.samples.set(label, values);
  }
}
